"""
Customer Profile & ML Initial Route Priority Scoring Engine.

Implements a supervised Machine Learning model (Gradient Boosting Regressor)
to score customer delivery priority upfront based on:
  - is_new_customer: Higher weight/priority to delight new customers for retention.
  - order_value_inr: High value orders get accelerated VIP priority.
  - past_miss_rate: Chronic missers/order rejecters get reduced priority weights.
  - customer_lifetime_value (CLV): Long-term relationship value.
  - package_demand: Freight load.
  - time_window_urgency: Delivery deadline constraint.

The generated ML SLA scores directly feed into pyvroom (vroom.Job priority)
to govern the initial route construction and van stop sequence.
"""

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "is_new_customer",
    "order_value_inr",
    "past_miss_rate",
    "clv_score",
    "package_demand",
    "time_urgency",
]


@dataclass
class CustomerProfile:
    """Raw input parameters and ML generated SLA score for a single customer."""
    cust_no: int
    customer_name: str
    segment: str                    # "New Customer", "High-Value VIP", "Loyal Regular", "Standard", "Chronic Misser"
    is_new_customer: int            # 1 = New, 0 = Existing
    order_value_inr: float          # In Rupees (₹)
    past_miss_rate: float           # 0.0 to 0.60 (fraction of past failed/missed deliveries)
    clv_score: float                # Customer Lifetime Value index (1.0 to 10.0)
    package_demand: int             # Weight in kg
    ready_time: float               # Start of delivery window
    due_date: float                 # Deadline of delivery window
    service_time: float             # Dwell time
    raw_feature_vector: List[float] = field(default_factory=list)
    ml_priority_score: float = 50.0 # 0.0 to 100.0 (higher = prioritized earlier in route)
    priority_tier: str = "Standard" # "VIP Expedited", "Priority High", "Standard", "Low / Saver"
    vroom_priority: int = 50        # Integer 1-100 fed into vroom.Job.priority
    priority_badge_html: str = ""   # Rendered badge for UI
    rationale: str = ""             # Business/algorithmic explanation


class CustomerScoringEngine:
    """
    Trained Gradient Boosting model that scores customer delivery urgency and priority
    before the initial dispatch route is calculated.
    """

    _instance: Optional["CustomerScoringEngine"] = None

    def __init__(self) -> None:
        self.model: Optional[GradientBoostingRegressor] = None
        self.feature_importances: Dict[str, float] = {}
        self.customer_profiles: Dict[int, CustomerProfile] = {}
        self._train_priority_model()
        self.customer_profiles = self._generate_canonical_profiles()

    @classmethod
    def get_instance(cls) -> "CustomerScoringEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _train_priority_model(self) -> None:
        """
        Train a Gradient Boosting Regressor on logistics prioritization economics.
        New customers and high-value orders receive positive utility;
        chronic order missers receive severe priority penalties.
        """
        t0 = time.perf_counter()
        np.random.seed(101)
        n_samples = 2500

        # Feature distributions
        is_new = np.random.choice([0, 1], p=[0.75, 0.25], size=n_samples)
        order_val = np.random.uniform(400.0, 9500.0, size=n_samples)
        
        # New customers have 0 past miss rate; existing customers range 0.0 to 0.55
        miss_rate = np.where(is_new == 1, 0.0, np.random.beta(a=1.5, b=5.0, size=n_samples) * 0.60)
        
        clv = np.where(is_new == 1, np.random.uniform(2.0, 4.5, size=n_samples), np.random.uniform(1.0, 10.0, size=n_samples))
        demand = np.random.choice([10, 15, 20, 25, 30, 35, 40], size=n_samples)
        urgency = np.random.uniform(0.1, 1.0, size=n_samples) # 1.0 = tightest due date

        X = np.column_stack([is_new, order_val, miss_rate, clv, demand, urgency])

        # Mathematical priority scoring function (ground truth)
        # 1. Base score 50
        # 2. New Customer Boost: +30 points (crucial for 1st order retention)
        # 3. High Order Value Boost: up to +28 points for orders approaching ₹10,000
        # 4. Chronic Misser Penalty: up to -45 points for high miss rate
        # 5. CLV & Urgency: up to +15 points
        score_latent = (
            42.0
            + (is_new * 32.0)
            + ((order_val / 9500.0) * 26.0)
            - (miss_rate * 55.0)
            + ((clv / 10.0) * 10.0)
            + (urgency * 12.0)
            - ((demand / 40.0) * 5.0)
        )
        noise = np.random.normal(0.0, 2.0, size=n_samples)
        y = np.clip(score_latent + noise, 5.0, 99.0)

        reg = GradientBoostingRegressor(
            n_estimators=80,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42,
        )
        reg.fit(X, y)
        self.model = reg

        # Feature importances
        raw_imp = reg.feature_importances_
        tot = float(sum(raw_imp)) if sum(raw_imp) > 0 else 1.0
        self.feature_importances = {
            name: round(float(val / tot) * 100.0, 1)
            for name, val in zip(FEATURE_NAMES, raw_imp)
        }
        train_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Customer Priority ML Model trained in {train_ms:.2f}ms. Importances: {self.feature_importances}")

    def score_customer(
        self,
        cust_no: int,
        customer_name: str,
        segment: str,
        is_new_customer: int,
        order_value_inr: float,
        past_miss_rate: float,
        clv_score: float,
        package_demand: int,
        ready_time: float,
        due_date: float,
        service_time: float,
    ) -> CustomerProfile:
        """Score an individual customer through the ML model."""
        urgency = max(0.1, min(1.0, 600.0 / max(1.0, due_date)))
        x_vec = np.array(
            [[
                float(is_new_customer),
                float(order_value_inr),
                float(past_miss_rate),
                float(clv_score),
                float(package_demand),
                float(urgency),
            ]],
            dtype=np.float32,
        )

        if self.model is not None:
            pred_score = float(self.model.predict(x_vec)[0])
        else:
            pred_score = 50.0 + (is_new_customer * 30.0) + (order_value_inr / 400.0) - (past_miss_rate * 50.0)

        pred_score = max(5.0, min(99.0, pred_score))
        vroom_priority = int(round(pred_score))

        # Assign Tier & Rationale
        if pred_score >= 80.0:
            tier = "VIP Expedited"
            badge = f'<span class="nexai-tag tag-violet" style="font-weight: 600; font-size: 0.78rem;">VIP Expedited ({pred_score:.1f})</span>'
            if is_new_customer:
                rationale = "New Customer: Maximum weight assigned to guarantee fastest initial delivery & retention."
            else:
                rationale = f"High Order Value (₹{order_value_inr:,.0f}): Priority route dispatch."
        elif pred_score >= 65.0:
            tier = "Priority High"
            badge = f'<span class="nexai-tag tag-blue" style="font-weight: 600; font-size: 0.78rem;">Priority High ({pred_score:.1f})</span>'
            rationale = "Elevated SLA weight. Consistent on-time order history."
        elif pred_score >= 42.0:
            tier = "Standard"
            badge = f'<span class="nexai-tag tag-slate" style="font-weight: 500; font-size: 0.78rem;">Standard ({pred_score:.1f})</span>'
            rationale = "Nominal dispatch weight. Standard routing allocation."
        else:
            tier = "Low / Saver"
            badge = f'<span class="nexai-tag tag-amber" style="font-weight: 600; font-size: 0.78rem;">Low / Saver ({pred_score:.1f})</span>'
            rationale = f"Chronic Misser ({past_miss_rate*100:.0f}% failed attempts): Reduced priority weight to prevent fleet delay."

        raw_features = [
            round(float(is_new_customer), 1),
            round(float(order_value_inr), 0),
            round(float(past_miss_rate), 2),
            round(float(clv_score), 1),
            round(float(package_demand), 0),
            round(float(urgency), 2),
        ]

        prof = CustomerProfile(
            cust_no=cust_no,
            customer_name=customer_name,
            segment=segment,
            is_new_customer=is_new_customer,
            order_value_inr=order_value_inr,
            past_miss_rate=past_miss_rate,
            clv_score=clv_score,
            package_demand=package_demand,
            ready_time=ready_time,
            due_date=due_date,
            service_time=service_time,
            raw_feature_vector=raw_features,
            ml_priority_score=round(pred_score, 1),
            priority_tier=tier,
            vroom_priority=vroom_priority,
            priority_badge_html=badge,
            rationale=rationale,
        )
        self.customer_profiles[cust_no] = prof
        return prof

    def _generate_canonical_profiles(self) -> Dict[int, CustomerProfile]:
        """
        Generate realistic customer profiles across the 25 Solomon benchmark customers
        covering the user's scenarios:
          - New Customers: Given top priority weight for acquisition & retention
          - High-Value VIP Orders: Given elevated priority weight
          - Chronic Order Missers: Given lower priority weight
          - Standard / Loyal Regulars
        """
        # Specific realistic profiles mapped to Solomon C101 customer IDs (1 to 25)
        spec = {
            # New Customers (High Weight)
            5:  ("Arjun Mehta (1st Order)", "New Customer", 1, 4850.0, 0.0, 3.5, 10, 0.0, 600.0, 5.0),
            14: ("Kavita Rao (1st Order)", "New Customer", 1, 5200.0, 0.0, 3.8, 20, 0.0, 720.0, 5.0),
            21: ("Siddharth Verma", "New Customer", 1, 6100.0, 0.0, 4.0, 15, 0.0, 650.0, 5.0),
            
            # High Value VIP Orders (High Weight)
            3:  ("Apex Diagnostic Labs", "High-Value VIP", 0, 8900.0, 0.0, 9.2, 25, 0.0, 500.0, 8.0),
            11: ("Tata Croma Electronics", "High-Value VIP", 0, 9400.0, 0.02, 9.5, 30, 0.0, 550.0, 8.0),
            18: ("Dr. Reddy Health Hub", "High-Value VIP", 0, 7800.0, 0.0, 8.8, 15, 0.0, 600.0, 5.0),
            
            # Chronic Missers (Low Weight / High Penalty)
            8:  ("Vikram Electronics", "Chronic Misser", 0, 650.0, 0.48, 1.8, 20, 0.0, 800.0, 5.0),
            17: ("Rahul Auto Spares", "Chronic Misser", 0, 720.0, 0.52, 1.5, 15, 0.0, 850.0, 5.0),
            24: ("Sunil Hardware", "Chronic Misser", 0, 580.0, 0.45, 1.2, 25, 0.0, 900.0, 5.0),

            # Loyal Regulars
            1:  ("Vellore Pharmacy Hub", "Loyal Regular", 0, 3200.0, 0.04, 7.5, 10, 0.0, 800.0, 5.0),
            2:  ("Lakshmi Supermarket", "Loyal Regular", 0, 2800.0, 0.05, 6.8, 20, 0.0, 750.0, 5.0),
            7:  ("Bala Provision Store", "Loyal Regular", 0, 3400.0, 0.03, 7.1, 20, 0.0, 600.0, 5.0),
            10: ("Sri Krishna Sweets", "Loyal Regular", 0, 2900.0, 0.02, 6.5, 10, 0.0, 650.0, 5.0),
            13: ("Ganga Departmental", "Loyal Regular", 0, 3100.0, 0.04, 7.0, 15, 0.0, 700.0, 5.0),
            20: ("Priya Medicals", "Loyal Regular", 0, 3600.0, 0.01, 7.8, 25, 0.0, 750.0, 5.0),
            23: ("Anand Stationery", "Loyal Regular", 0, 2200.0, 0.05, 5.5, 15, 0.0, 800.0, 5.0),
        }

        profiles: Dict[int, CustomerProfile] = {}
        for c_id in range(1, 26):
            if c_id in spec:
                name, seg, is_new, val, miss, clv, dem, rdy, due, srv = spec[c_id]
            else:
                name = f"Customer #{c_id} Retail"
                seg = "Standard"
                is_new = 0
                val = 1200.0 + (c_id * 85.0)
                miss = 0.08
                clv = 4.0
                dem = 15
                rdy = 0.0
                due = 700.0
                srv = 5.0

            prof = self.score_customer(
                cust_no=c_id,
                customer_name=name,
                segment=seg,
                is_new_customer=is_new,
                order_value_inr=val,
                past_miss_rate=miss,
                clv_score=clv,
                package_demand=dem,
                ready_time=rdy,
                due_date=due,
                service_time=srv,
            )
            profiles[c_id] = prof

        return profiles

    def get_profile(self, cust_no: int) -> CustomerProfile:
        """Retrieve profile or generate default for dynamic rush orders."""
        if cust_no in self.customer_profiles:
            return self.customer_profiles[cust_no]

        # Dynamic rush orders (cust_no >= 100) are high priority expedited
        prof = self.score_customer(
            cust_no=cust_no,
            customer_name=f"Rush Client #{cust_no}",
            segment="New Customer",
            is_new_customer=1,
            order_value_inr=4500.0,
            past_miss_rate=0.0,
            clv_score=5.0,
            package_demand=15,
            ready_time=0.0,
            due_date=500.0,
            service_time=3.0,
        )
        self.customer_profiles[cust_no] = prof
        return prof


def render_customer_priority_scoring_table_html(sim_state: Any) -> str:
    """
    Generate centered, Shadcn-styled table replacing the old Fleet Package Manifest table.
    Renders raw customer input parameters and ML-generated SLA priority scores.
    """
    engine = CustomerScoringEngine.get_instance()
    profiles = engine.customer_profiles

    # Map current routes to vehicle assignments
    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and hasattr(sim_state.current_solution.routes, "iterrows")
        else None
    )

    cust_to_van: Dict[int, int] = {}
    cust_to_seq: Dict[int, int] = {}
    if routes_df is not None and not routes_df.empty:
        job_rows = routes_df[routes_df["type"] == "job"]
        veh_counters: Dict[int, int] = {}
        for _, r in job_rows.iterrows():
            if pd.notna(r.get("id")) and pd.notna(r.get("vehicle_id")):
                cid = int(r["id"])
                vid = int(r["vehicle_id"])
                cust_to_van[cid] = vid
                veh_counters[vid] = veh_counters.get(vid, 0) + 1
                cust_to_seq[cid] = veh_counters[vid]

    # Model telemetry summary header
    feat_labels = {
        "is_new_customer": "New Customer Boost",
        "order_value_inr": "Order Value",
        "past_miss_rate": "Chronic Miss Penalty",
        "clv_score": "CLV Relationship",
        "time_urgency": "Window Urgency",
        "package_demand": "Cargo Load",
    }
    sorted_feats = sorted(engine.feature_importances.items(), key=lambda kv: kv[1], reverse=True)
    pills = [
        f'<span style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 9999px; padding: 2px 8px; font-size: 0.7rem; color: #334155;">'
        f'<strong>{feat_labels.get(k, k)}:</strong> {v}%</span>'
        for k, v in sorted_feats[:4]
    ]
    pills_str = " ".join(pills)

    # Sort customers according to Customer ID
    all_cids = sorted(profiles.keys(), key=lambda cid: profiles[cid].cust_no)

    rows = []
    for cid in all_cids:
        p = profiles[cid]
        vid = cust_to_van.get(cid)
        seq_idx = cust_to_seq.get(cid, 0)
        is_completed = cid in getattr(sim_state, "completed_jobs", {})

        # Status badge (Route Allocation)
        if is_completed:
            status_badge = '<span class="nexai-tag tag-emerald" style="font-size: 0.75rem;">Delivered</span>'
        elif vid is not None:
            v_state = sim_state.vehicles.get(vid)
            is_broken = v_state.is_broken if v_state else False
            if is_broken:
                status_badge = '<span class="nexai-tag tag-rose" style="font-size: 0.75rem;">Re-dispatching</span>'
            else:
                seq_str = f" · Stop #{seq_idx}" if seq_idx > 0 else ""
                status_badge = f'<span class="nexai-tag tag-blue" style="font-size: 0.75rem; font-weight: 600;">Van {vid}{seq_str}</span>'
        else:
            status_badge = '<span class="nexai-tag tag-slate" style="font-size: 0.75rem;">Pending Queue</span>'

        # Segment tag styling with concise words
        if p.segment in ("New Customer", "New"):
            seg_tag = '<span class="nexai-tag tag-emerald" style="font-weight: 600; font-size: 0.75rem;">New</span>'
        elif p.segment in ("High-Value VIP", "VIP"):
            seg_tag = '<span class="nexai-tag tag-violet" style="font-weight: 600; font-size: 0.75rem;">VIP</span>'
        elif p.segment in ("Chronic Misser", "Misser"):
            seg_tag = '<span class="nexai-tag tag-rose" style="font-weight: 600; font-size: 0.75rem;">Misser</span>'
        elif p.segment in ("Loyal Regular", "Loyal"):
            seg_tag = '<span class="nexai-tag tag-blue" style="font-size: 0.75rem;">Loyal</span>'
        else:
            seg_tag = '<span class="nexai-tag tag-slate" style="font-size: 0.75rem;">Standard</span>'

        # Miss rate tag (concise numeric format)
        if p.past_miss_rate > 0.30:
            miss_tag = f'<span style="color: #dc2626; font-weight: 600; font-size: 0.82rem;">{p.past_miss_rate*100:.0f}%</span>'
        elif p.past_miss_rate > 0:
            miss_tag = f'<span style="color: #64748b; font-size: 0.82rem;">{p.past_miss_rate*100:.0f}%</span>'
        else:
            miss_tag = '<span style="color: #047857; font-weight: 500; font-size: 0.82rem;">0%</span>'

        # ML Priority Score - just the number
        score_cell = f'<span style="font-weight: 700; font-size: 0.88rem; color: #09090b;">{p.ml_priority_score:.1f}</span>'

        rows.append(
            f'<tr>'
            f'<td style="text-align: center;">{status_badge}</td>'
            f'<td style="text-align: center; font-weight: 600; color: #09090b;">Customer #{p.cust_no}</td>'
            f'<td style="text-align: center;">{seg_tag}</td>'
            f'<td style="text-align: center; font-weight: 600; color: #09090b;">₹{p.order_value_inr:,.0f}</td>'
            f'<td style="text-align: center;">{miss_tag}</td>'
            f'<td style="text-align: center;">{score_cell}</td>'
            f'</tr>'
        )

    rows_str = "".join(rows)

    table_html = (
        '<div style="background: #ffffff; border: 1px solid #e4e4e7; border-radius: 8px; padding: 0.9rem 1.15rem; margin-bottom: 1.15rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.65rem;">'
        '<div>'
        '<div style="font-size: 0.92rem; font-weight: 700; color: #09090b;">'
        'Customer Profiles & ML SLA Priority Scores'
        '</div>'
        '</div>'
        '</div>'
        '<div class="manifest-table-container" style="max-height: 380px; overflow-y: auto;">'
        '<table class="manifest-table">'
        '<thead><tr>'
        '<th style="width: 20%; text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Route Allocation</th>'
        '<th style="width: 16%; text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Customer</th>'
        '<th style="width: 16%; text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Segment Profile</th>'
        '<th style="width: 16%; text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Order Value (₹)</th>'
        '<th style="width: 16%; text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Past Miss Rate</th>'
        '<th style="width: 16%; text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">ML Priority Score</th>'
        '</tr></thead>'
        f'<tbody>{rows_str}</tbody>'
        '</table>'
        '</div>'
        '</div>'
    )

    return table_html
