"""
NEXAI 2026 Hackathon - Track 4: Supply Chain & Logistics
Problem Statement 4.3: Last-Mile Route Planner with Live Re-planning (VRPTW)
Optimization Engine: pyvroom (High-performance VROOM C++ Bindings)

Interactive Dispatcher Dashboard with Real-Time Disruption Simulation.
"""

from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.parser import parse_solomon_instance
from src.simulator import SimulationState
from src.utils import resolve_location_index, unscale_value
from src.vrptw_solver import create_vroom_problem

# Set broad responsive layout and dark-friendly title
st.set_page_config(
    page_title="NEXAI 2026 | Last-Mile VRPTW Dispatcher",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom enterprise CSS for sleek modern dispatcher interface
st.markdown(
    """
    <style>
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    .dispatch-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-track { background-color: #0369a1; color: #e0f2fe; }
    .badge-problem { background-color: #065f46; color: #d1fae5; }
    .badge-engine { background-color: #6b21a8; color: #f3e8ff; }
    .status-panel {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 1. State Management (st.session_state)
# ==============================================================================
def init_simulation_state() -> None:
    """Initialize the problem instance, distance matrix, and baseline VROOM solution."""
    with st.spinner("Initializing Solomon C101 Benchmark & pyvroom engine..."):
        instance = parse_solomon_instance("data/C101_25.txt", max_customers=25)
        distance_matrix = instance.compute_distance_matrix()

        # Build and solve initial static routing problem at T = 0
        problem = create_vroom_problem(instance, distance_matrix)
        t0 = time.perf_counter()
        initial_solution = problem.solve(exploration_level=1, nb_threads=4)
        init_runtime_ms = (time.perf_counter() - t0) * 1000.0

        # Store SimulationState instance in Streamlit session state
        st.session_state.sim = SimulationState(
            instance=instance,
            distance_matrix=distance_matrix,
            initial_solution=initial_solution,
        )
        # Record initial solve in history
        st.session_state.sim.replanning_history.append(
            {
                "timestamp_scaled": 0,
                "timestamp_unscaled": 0.0,
                "runtime_ms": round(init_runtime_ms, 2),
                "cost": initial_solution.summary.cost,
                "routes_count": len(initial_solution.routes["vehicle_id"].unique()),
                "total_steps": len(initial_solution.routes),
                "pending_jobs_count": 25,
                "active_vehicles_count": instance.vehicle_count,
            }
        )
        st.session_state.current_time = 0.0
        st.session_state.rush_order_count = 0


if "sim" not in st.session_state:
    init_simulation_state()

sim: SimulationState = st.session_state.sim


# ==============================================================================
# 2. Coordinate Translation Helper
# ==============================================================================
def get_node_coords(sim_state: SimulationState, loc_idx: int) -> Tuple[float, float]:
    """Translate a location index to continuous unscaled (X, Y) coordinates."""
    if loc_idx == 0:
        return (sim_state.instance.depot.x, sim_state.instance.depot.y)
    if loc_idx < len(sim_state.scaled_coordinates):
        sx, sy = sim_state.scaled_coordinates[loc_idx]
        return (
            unscale_value(sx, sim_state.scale_factor),
            unscale_value(sy, sim_state.scale_factor),
        )
    return (0.0, 0.0)


# ==============================================================================
# 3. Plotly Route & Map Visualization
# ==============================================================================
# Curated high-contrast palette for route distinction
ROUTE_COLORS = [
    "#38bdf8",  # Sky Blue
    "#f43f5e",  # Vibrant Rose
    "#10b981",  # Emerald Green
    "#fbbf24",  # Amber Gold
    "#a855f7",  # Purple
    "#06b6d4",  # Cyan
    "#ec4899",  # Pink
    "#f97316",  # Bright Orange
]


def plot_routes(sim_state: SimulationState) -> go.Figure:
    """
    Generate an interactive 2D spatial dispatch map with active routes and dynamic nodes.
    
    Renders:
      - Central Depot (gold square)
      - Completed Customer stops (green circles)
      - Pending Customer stops (blue circles)
      - Dynamically Injected Rush Orders (orange diamonds)
      - Vehicle Live Start Positions (purple markers)
      - Broken / Disabled Vans (red X markers)
      - Active Route paths (colored lines with step details)
    """
    fig = go.Figure()

    # --------------------------------------------------------------------------
    # A. Draw Vehicle Routes (Edges)
    # --------------------------------------------------------------------------
    solution = sim_state.current_solution
    routes_df = (
        solution.routes
        if hasattr(solution, "routes") and isinstance(solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )

    if not routes_df.empty:
        for idx, (v_id, group) in enumerate(routes_df.groupby("vehicle_id")):
            color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
            xs: List[float] = []
            ys: List[float] = []
            hover_lines: List[str] = []

            for step_num, (_, step) in enumerate(group.iterrows()):
                loc_idx = int(step["location_index"])
                x, y = get_node_coords(sim_state, loc_idx)
                xs.append(x)
                ys.append(y)

                step_type = str(step["type"]).upper()
                job_id = step.get("id")
                arr_unscaled = unscale_value(step["arrival"], sim_state.scale_factor)
                serv_unscaled = unscale_value(step["service"], sim_state.scale_factor)

                hover_text = (
                    f"<b>Van #{v_id}</b> | Step {step_num + 1} [{step_type}]<br>"
                    f"Stop Location: ({x:.1f}, {y:.1f})<br>"
                    f"Arrival Time: T={arr_unscaled:.1f}<br>"
                    f"Service Duration: {serv_unscaled:.1f}"
                )
                if pd.notna(job_id):
                    hover_text += f"<br>Job ID: #{int(job_id)}"
                hover_lines.append(hover_text)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=2.8),
                    opacity=0.85,
                    name=f"Van #{v_id} Route",
                    hoverinfo="text",
                    hovertext=hover_lines,
                )
            )

    # --------------------------------------------------------------------------
    # B. Plot Completed Nodes
    # --------------------------------------------------------------------------
    completed_ids = set(sim_state.completed_jobs.keys())
    comp_x, comp_y, comp_hover = [], [], []

    for c_id, info in sim_state.completed_jobs.items():
        loc = info.get("location_index", 0)
        x, y = get_node_coords(sim_state, loc)
        comp_x.append(x)
        comp_y.append(y)
        comp_hover.append(
            f"<b>Customer #{c_id} [COMPLETED]</b><br>"
            f"Location: ({x:.1f}, {y:.1f})<br>"
            f"Delivered by Van #{info.get('vehicle_id')}<br>"
            f"Demand: {info.get('demand', '-')} units<br>"
            f"Finished at T={unscale_value(info.get('completion_time', 0), sim_state.scale_factor):.1f}"
        )

    if comp_x:
        fig.add_trace(
            go.Scatter(
                x=comp_x,
                y=comp_y,
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=11,
                    color="#10b981",  # Emerald Green
                    line=dict(color="#064e3b", width=1.5),
                ),
                text=[f"#{cid}" for cid in sim_state.completed_jobs.keys()],
                textposition="bottom center",
                textfont=dict(size=9, color="#6ee7b7"),
                name=f"Completed ({len(comp_x)})",
                hoverinfo="text",
                hovertext=comp_hover,
            )
        )

    # --------------------------------------------------------------------------
    # C. Plot Pending Regular Customers
    # --------------------------------------------------------------------------
    pend_x, pend_y, pend_hover, pend_labels = [], [], [], []
    rush_x, rush_y, rush_hover, rush_labels = [], [], [], []

    for job_id, job in sim_state.pending_jobs.items():
        loc_idx = resolve_location_index(job.location)
        x, y = get_node_coords(sim_state, loc_idx)
        is_rush = job_id >= 100

        # Retrieve customer metadata if available
        meta = sim_state.customer_lookup.get(job_id)
        demand = meta.demand if meta else (job.delivery[0] if job.delivery else "-")
        ready_t = meta.ready_time if meta else "-"
        due_t = meta.due_date if meta else "-"

        hover_info = (
            f"<b>{'[RUSH ORDER]' if is_rush else 'Customer'} #{job_id} [PENDING]</b><br>"
            f"Location: ({x:.1f}, {y:.1f})<br>"
            f"Demand: {demand} units<br>"
            f"Time Window: [{ready_t}, {due_t}]"
        )

        if is_rush:
            rush_x.append(x)
            rush_y.append(y)
            rush_hover.append(hover_info)
            rush_labels.append(f"⚡#{job_id}")
        else:
            pend_x.append(x)
            pend_y.append(y)
            pend_hover.append(hover_info)
            pend_labels.append(f"#{job_id}")

    if pend_x:
        fig.add_trace(
            go.Scatter(
                x=pend_x,
                y=pend_y,
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=11,
                    color="#38bdf8",  # Sky Blue
                    line=dict(color="#0369a1", width=1.5),
                ),
                text=pend_labels,
                textposition="top center",
                textfont=dict(size=9, color="#bae6fd"),
                name=f"Pending Orders ({len(pend_x)})",
                hoverinfo="text",
                hovertext=pend_hover,
            )
        )

    if rush_x:
        fig.add_trace(
            go.Scatter(
                x=rush_x,
                y=rush_y,
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=15,
                    color="#f97316",  # Bright Orange
                    line=dict(color="#ffffff", width=2),
                ),
                text=rush_labels,
                textposition="top right",
                textfont=dict(size=10, color="#fed7aa", family="Arial Black"),
                name=f"Rush Orders ({len(rush_x)})",
                hoverinfo="text",
                hovertext=rush_hover,
            )
        )

    # --------------------------------------------------------------------------
    # D. Plot Vehicle Start Positions & Breakdowns
    # --------------------------------------------------------------------------
    active_van_x, active_van_y, active_van_hover, active_van_text = [], [], [], []
    for v_id, v_state in sim_state.vehicles.items():
        if v_id in sim_state.broken_vehicles:
            continue
        # Only show vans currently active in the route plan
        if not routes_df.empty and v_id in routes_df["vehicle_id"].values:
            vx, vy = get_node_coords(sim_state, v_state.start_location)
            active_van_x.append(vx)
            active_van_y.append(vy)
            active_van_text.append(f"Van {v_id}")
            active_van_hover.append(
                f"<b>Van #{v_id} [ACTIVE]</b><br>"
                f"Current Location: ({vx:.1f}, {vy:.1f})<br>"
                f"Remaining Capacity: {v_state.remaining_capacity} units<br>"
                f"Completed Stops: {len(v_state.completed_job_ids)}"
            )

    if active_van_x:
        fig.add_trace(
            go.Scatter(
                x=active_van_x,
                y=active_van_y,
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=15,
                    color="#c084fc",  # Purple
                    line=dict(color="#ffffff", width=1.8),
                ),
                text=active_van_text,
                textposition="bottom right",
                textfont=dict(size=10, color="#e9d5ff", family="sans-serif"),
                name="Van Live Positions",
                hoverinfo="text",
                hovertext=active_van_hover,
            )
        )

    # Plot Broken Vans
    if sim_state.broken_vehicles:
        broken_x, broken_y, broken_hover, broken_labels = [], [], [], []
        for b_id in sim_state.broken_vehicles:
            bv = sim_state.vehicles[b_id]
            bx, by = get_node_coords(sim_state, bv.start_location)
            broken_x.append(bx)
            broken_y.append(by)
            broken_labels.append(f"💥 Van {b_id} Disabled")
            broken_hover.append(
                f"<b>Van #{b_id} [DISABLED / BREAKDOWN]</b><br>"
                f"Breakdown Location: ({bx:.1f}, {by:.1f})<br>"
                f"Completed {len(bv.completed_job_ids)} stops before failure.<br>"
                f"<i>Pending deliveries reassigned to fleet!</i>"
            )

        fig.add_trace(
            go.Scatter(
                x=broken_x,
                y=broken_y,
                mode="markers+text",
                marker=dict(
                    symbol="x",
                    size=16,
                    color="#ef4444",  # Crimson Red
                    line=dict(color="#ffffff", width=2.5),
                ),
                text=broken_labels,
                textposition="top center",
                textfont=dict(size=11, color="#fca5a5"),
                name="Disabled Vehicle",
                hoverinfo="text",
                hovertext=broken_hover,
            )
        )

    # --------------------------------------------------------------------------
    # E. Plot Central Depot
    # --------------------------------------------------------------------------
    depot = sim_state.instance.depot
    fig.add_trace(
        go.Scatter(
            x=[depot.x],
            y=[depot.y],
            mode="markers+text",
            marker=dict(
                symbol="square",
                size=18,
                color="#eab308",  # Gold
                line=dict(color="#ffffff", width=2.2),
            ),
            text=["DEPOT (0)"],
            textposition="bottom center",
            textfont=dict(size=11, color="#fef08a", family="sans-serif"),
            name="Central Depot",
            hoverinfo="text",
            hovertext=[
                f"<b>CENTRAL DEPOT (Node 0)</b><br>"
                f"Coordinates: ({depot.x:.1f}, {depot.y:.1f})<br>"
                f"Operational Horizon: [{depot.ready_time}, {depot.due_date}]<br>"
                f"Fleet Base: {sim_state.instance.vehicle_count} Vehicles"
            ],
        )
    )

    # Layout styling: sleek dark aesthetic with gridlines
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#111827",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(
            title="X Coordinate (Spatial Grid)",
            gridcolor="#1f2937",
            zerolinecolor="#374151",
            showgrid=True,
        ),
        yaxis=dict(
            title="Y Coordinate (Spatial Grid)",
            gridcolor="#1f2937",
            zerolinecolor="#374151",
            showgrid=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(17, 24, 39, 0.7)",
            bordercolor="#374151",
            borderwidth=1,
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_size=12,
            font_family="sans-serif",
        ),
        height=540,
    )

    return fig


# ==============================================================================
# 4. Main Dashboard Header
# ==============================================================================
st.markdown(
    """
    <div class="dispatch-header">
        <h2 style="margin: 0; color: #f8fafc; font-weight: 700;">
            Autonomous Dynamic Route Engine & Live Re-planner
        </h2>
        <div style="margin-top: 0.5rem;">
            <span class="badge badge-track">Track 4: AI for Supply Chain & Logistics</span>
            <span class="badge badge-problem">Problem 4.3: VRPTW Dynamic Re-planning</span>
            <span class="badge badge-engine">Engine: pyvroom (Sub-5ms Re-optimization)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 5. Sidebar: Setup & Live KPIs
# ==============================================================================
with st.sidebar:
    st.title("Dispatch Controls")
    st.markdown("---")

    # Current Virtual Clock Status
    st.markdown("### ⏱️ Virtual Dispatch Clock")
    col_clock1, col_clock2 = st.columns(2)
    with col_clock1:
        st.caption("Current Time")
        st.subheader(f"T = {st.session_state.current_time:.1f}")
    with col_clock2:
        st.caption("Depot Horizon")
        st.subheader(f"T = {sim.instance.depot.due_date:.0f}")

    st.markdown("---")
    st.markdown("### 📊 Operational KPIs")

    # 1. Total Distance (unscaled)
    unscaled_distance = sim.current_solution.summary.cost / float(sim.scale_factor)
    st.metric(
        label="Total Travel Distance",
        value=f"{unscaled_distance:.1f} units",
        delta=f"Cost: {sim.current_solution.summary.cost}",
    )

    # 2. Active Vehicles
    routes_df = (
        sim.current_solution.routes
        if hasattr(sim.current_solution, "routes") and isinstance(sim.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    active_vehicles_count = (
        len(routes_df["vehicle_id"].unique()) if not routes_df.empty else 0
    )
    total_active_fleet = len(sim.vehicles) - len(sim.broken_vehicles)
    st.metric(
        label="Active Fleet in Transit",
        value=f"{active_vehicles_count} Vans",
        delta=f"{total_active_fleet} Total Available",
    )

    # 3. Pending/Unserved Jobs
    pending_count = len(sim.pending_jobs)
    completed_count = len(sim.completed_jobs)
    st.metric(
        label="Pending Deliveries",
        value=f"{pending_count} Stops",
        delta=f"-{completed_count} Completed",
        delta_color="inverse",
    )

    # 4. Last Re-plan Compute Time
    latest_runtime = (
        sim.replanning_history[-1]["runtime_ms"]
        if sim.replanning_history
        else 0.0
    )
    st.metric(
        label="Last Re-plan Compute Time",
        value=f"{latest_runtime:.2f} ms",
        delta="Sub-5ms Guaranteed",
    )

    st.markdown("---")
    # Reset Simulation Button
    if st.button("🔄 Reset Simulation to Baseline", use_container_width=True):
        st.session_state.clear()
        init_simulation_state()
        st.rerun()


# ==============================================================================
# 6. Main View: Map Visualization (Plotly)
# ==============================================================================
map_figure = plot_routes(sim)
st.plotly_chart(map_figure, use_container_width=True)


# ==============================================================================
# 7. Simulation Controls (Disruption Triggers)
# ==============================================================================
st.markdown("### ⚡ Live Disruption & Simulation Controls")
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.2, 1.3], gap="medium")

# --- Column 1: Time Slider & Clock Advance ---
with col_ctrl1:
    st.markdown("**1. Advance Simulation Clock**")
    slider_val = st.slider(
        "Move clock forward (T_event):",
        min_value=0.0,
        max_value=1000.0,
        value=float(st.session_state.current_time),
        step=25.0,
        key="time_slider",
    )
    if st.button("⏩ Advance Time", use_container_width=True):
        if slider_val > st.session_state.current_time:
            adv_summary = sim.advance_clock(slider_val)
            st.session_state.current_time = slider_val
            st.toast(
                f"Clock advanced to T={slider_val:.1f}! {adv_summary['newly_completed_count']} stops completed.",
                icon="⏱️",
            )
            st.rerun()
        elif slider_val < st.session_state.current_time:
            st.warning("Clock cannot move backwards in time.")
        else:
            st.info("Time slider is already at current simulation time.")

# --- Column 2: Vehicle Breakdown Disruption ---
with col_ctrl2:
    st.markdown("**2. Mechanical Disruption**")
    st.caption("Abrupt failure triggers immediate mid-route re-routing.")
    van_1_broken = 1 in sim.broken_vehicles
    btn_label = "🚨 Van 1 Already Disabled" if van_1_broken else "🚨 Trigger Van 1 Breakdown"

    if st.button(
        btn_label,
        use_container_width=True,
        type="primary" if not van_1_broken else "secondary",
        disabled=van_1_broken,
    ):
        with st.spinner("Processing breakdown & hot re-planning..."):
            sim.trigger_breakdown(1)
            sim.replan(exploration_level=1)
            runtime = sim.replanning_history[-1]["runtime_ms"]
            st.toast(
                f"🚨 Van 1 breakdown handled! Reassigned in {runtime:.2f} ms.",
                icon="⚠️",
            )
            st.rerun()

# --- Column 3: Rush Order Injection ---
with col_ctrl3:
    st.markdown("**3. Dynamic Customer Demand**")
    st.caption("Inject urgent mid-day delivery requests into pending pool.")

    if st.button("📦 Inject Rush Orders (x3)", use_container_width=True):
        with st.spinner("Expanding coordinate matrix & re-planning..."):
            st.session_state.rush_order_count += 1
            offset = st.session_state.rush_order_count * 10
            cur_t = float(st.session_state.current_time)

            # Three realistic customer orders with distinct coordinates
            rush_orders = [
                {
                    "cust_no": 100 + offset + 1,
                    "x": 38.0,
                    "y": 55.0,
                    "demand": 15,
                    "ready_time": max(cur_t, 250.0),
                    "due_date": min(cur_t + 500.0, 1100.0),
                    "service_time": 90.0,
                },
                {
                    "cust_no": 100 + offset + 2,
                    "x": 28.0,
                    "y": 70.0,
                    "demand": 20,
                    "ready_time": max(cur_t, 280.0),
                    "due_date": min(cur_t + 550.0, 1150.0),
                    "service_time": 90.0,
                },
                {
                    "cust_no": 100 + offset + 3,
                    "x": 48.0,
                    "y": 45.0,
                    "demand": 10,
                    "ready_time": max(cur_t, 200.0),
                    "due_date": min(cur_t + 600.0, 1200.0),
                    "service_time": 90.0,
                },
            ]

            injected_ids = sim.inject_new_orders(rush_orders)
            sim.replan(exploration_level=1)
            runtime = sim.replanning_history[-1]["runtime_ms"]
            st.toast(
                f"⚡ Injected {len(injected_ids)} rush stops! Fleet re-routed in {runtime:.2f} ms.",
                icon="🚀",
            )
            st.rerun()


# ==============================================================================
# 8. Detailed Telemetry & Fleet Manifest Tabs
# ==============================================================================
st.markdown("---")
tab_routes, tab_telemetry, tab_fleet = st.tabs(
    ["📋 Live Route Manifests", "📈 Re-planning Telemetry Audit", "🚐 Fleet State Table"]
)

with tab_routes:
    if not routes_df.empty:
        # Format human-friendly route inspection table
        display_df = routes_df.copy()
        display_df["arrival_time"] = display_df["arrival"].apply(
            lambda v: f"{unscale_value(v, sim.scale_factor):.1f}"
        )
        display_df["service_time"] = display_df["service"].apply(
            lambda v: f"{unscale_value(v, sim.scale_factor):.1f}"
        )
        display_df["location_x"] = display_df["location_index"].apply(
            lambda loc: f"{get_node_coords(sim, loc)[0]:.1f}"
        )
        display_df["location_y"] = display_df["location_index"].apply(
            lambda loc: f"{get_node_coords(sim, loc)[1]:.1f}"
        )

        cols_to_show = [
            "vehicle_id",
            "type",
            "id",
            "location_index",
            "location_x",
            "location_y",
            "arrival_time",
            "service_time",
        ]
        available_cols = [c for c in cols_to_show if c in display_df.columns]
        st.dataframe(
            display_df[available_cols].rename(
                columns={
                    "vehicle_id": "Van ID",
                    "type": "Step Type",
                    "id": "Customer ID",
                    "location_index": "Matrix Index",
                    "location_x": "X Coord",
                    "location_y": "Y Coord",
                    "arrival_time": "Arrival (T)",
                    "service_time": "Service Duration",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No active vehicle routes to display.")

with tab_telemetry:
    if sim.replanning_history:
        history_df = pd.DataFrame(sim.replanning_history)
        st.dataframe(
            history_df.rename(
                columns={
                    "timestamp_unscaled": "Simulation Clock (T)",
                    "runtime_ms": "Solver Compute Time (ms)",
                    "cost": "VROOM Scaled Cost",
                    "routes_count": "Active Routes",
                    "pending_jobs_count": "Pending Jobs",
                    "active_vehicles_count": "Available Fleet",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No re-planning events recorded yet.")

with tab_fleet:
    fleet_rows = []
    for vid, vstate in sim.vehicles.items():
        # Only show vehicles with activity or first 10 reserve vehicles
        if vid <= 5 or vstate.completed_job_ids or vstate.is_broken:
            loc_x, loc_y = get_node_coords(sim, vstate.start_location)
            fleet_rows.append(
                {
                    "Vehicle ID": vid,
                    "Status": "DISABLED" if vstate.is_broken else "OPERATIONAL",
                    "Current Location": f"({loc_x:.1f}, {loc_y:.1f}) [Node {vstate.start_location}]",
                    "Remaining Capacity": f"{vstate.remaining_capacity} / {sim.instance.vehicle_capacity}",
                    "Available From (T)": f"{unscale_value(vstate.available_from, sim.scale_factor):.1f}",
                    "Completed Stops": len(vstate.completed_job_ids),
                }
            )
    st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True, hide_index=True)
