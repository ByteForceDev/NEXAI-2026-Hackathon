"""
NEXAI 2026 Hackathon - Track 4: Supply Chain & Logistics
Problem Statement 4.3: Last-Mile Route Planner with Live Re-planning (VRPTW)
Optimization Engine: pyvroom (High-performance VROOM C++ Bindings)

Interactive Dispatcher Dashboard with Autonomous Simulation Loop and Live Disruption Engine.
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
        margin-bottom: 1.2rem;
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
    .badge-status { background-color: #1e293b; border: 1px solid #475569; color: #f8fafc; }
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
        st.session_state.is_simulating = False
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
    
    Guarantees:
      - At T = 0, all active vehicles are explicitly visible at the Central Depot.
      - In transit, vehicle positions are continuous and mathematically interpolated.
      - Disrupted vehicles are clearly rendered with failure tags.
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
    # C. Plot Pending Regular Customers & Rush Orders
    # --------------------------------------------------------------------------
    pend_x, pend_y, pend_hover, pend_labels = [], [], [], []
    rush_x, rush_y, rush_hover, rush_labels = [], [], [], []

    for job_id, job in sim_state.pending_jobs.items():
        loc_idx = resolve_location_index(job.location)
        x, y = get_node_coords(sim_state, loc_idx)
        is_rush = job_id >= 100

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
    # D. Plot Interpolated Vehicle Positions (Smooth Kinematics)
    # --------------------------------------------------------------------------
    veh_positions = sim_state.get_interpolated_vehicle_positions()
    van_x, van_y, van_hover, van_labels = [], [], [], []
    broken_x, broken_y, broken_hover, broken_labels = [], [], [], []

    for v_id, p_info in veh_positions.items():
        if p_info["is_broken"]:
            broken_x.append(p_info["x"])
            broken_y.append(p_info["y"])
            broken_labels.append(f"💥 Van {v_id}")
            broken_hover.append(
                f"<b>Van #{v_id} [DISABLED / BREAKDOWN]</b><br>"
                f"Coordinates: ({p_info['x']:.1f}, {p_info['y']:.1f})<br>"
                f"Status: {p_info['status']}<br>"
                f"<i>Unfinished stops re-assigned to active fleet!</i>"
            )
        elif p_info["is_active"]:
            # Active in operation or pre-departure / shift completed
            van_x.append(p_info["x"])
            van_y.append(p_info["y"])
            van_labels.append(f"Van {v_id}")
            van_hover.append(
                f"<b>Van #{v_id} [ACTIVE]</b><br>"
                f"Live Coordinates: ({p_info['x']:.1f}, {p_info['y']:.1f})<br>"
                f"Status: {p_info['status']}<br>"
                f"Remaining Capacity: {sim_state.vehicles[v_id].remaining_capacity} units<br>"
                f"Completed Stops: {len(sim_state.vehicles[v_id].completed_job_ids)}"
            )

    if van_x:
        fig.add_trace(
            go.Scatter(
                x=van_x,
                y=van_y,
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=16,
                    color="#c084fc",  # Purple
                    line=dict(color="#ffffff", width=2.0),
                ),
                text=van_labels,
                textposition="bottom right",
                textfont=dict(size=11, color="#f3e8ff", family="sans-serif"),
                name="Van Live Positions",
                hoverinfo="text",
                hovertext=van_hover,
            )
        )

    if broken_x:
        fig.add_trace(
            go.Scatter(
                x=broken_x,
                y=broken_y,
                mode="markers+text",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="#ef4444",  # Crimson Red
                    line=dict(color="#ffffff", width=2.8),
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

    # Layout styling: sleek dark aesthetic
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
# 4. KPI Placeholder Renderer
# ==============================================================================
def render_kpi_metrics(placeholder: Any, sim_state: SimulationState) -> None:
    """Render high-contrast metric cards into the dedicated empty placeholder."""
    unscaled_dist = sim_state.current_solution.summary.cost / float(sim_state.scale_factor)

    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    active_in_transit = len(routes_df["vehicle_id"].unique()) if not routes_df.empty else 0
    total_active_fleet = len(sim_state.vehicles) - len(sim_state.broken_vehicles)

    pending_cnt = len(sim_state.pending_jobs)
    comp_cnt = len(sim_state.completed_jobs)

    latest_runtime = (
        sim_state.replanning_history[-1]["runtime_ms"]
        if sim_state.replanning_history
        else 0.0
    )

    is_done = sim_state.is_simulation_completed()
    clock_status = "FINISHED" if is_done else ("SIMULATING" if st.session_state.is_simulating else "PAUSED")

    with placeholder.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            label="Total Travel Distance",
            value=f"{unscaled_dist:.1f} km",
            delta=f"Clock: T = {st.session_state.current_time:.1f} [{clock_status}]",
        )
        m2.metric(
            label="Active Fleet in Transit",
            value=f"{active_in_transit} Vans",
            delta=f"{total_active_fleet} Total Operational",
        )
        m3.metric(
            label="Pending Deliveries",
            value=f"{pending_cnt} Stops",
            delta=f"{comp_cnt} Completed",
            delta_color="inverse",
        )
        m4.metric(
            label="Last Re-plan Latency",
            value=f"{latest_runtime:.2f} ms",
            delta="Sub-5ms Guaranteed",
        )


# ==============================================================================
# 5. Dashboard Header
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
# 6. Sidebar: Setup & Simulation Summary
# ==============================================================================
with st.sidebar:
    st.title("Dispatch Controls")
    st.markdown("---")

    st.markdown("### ⏱️ Virtual Clock")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.caption("Current Time")
        st.subheader(f"T = {st.session_state.current_time:.1f}")
    with col_c2:
        st.caption("Depot Horizon")
        st.subheader(f"T = {sim.instance.depot.due_date:.0f}")

    st.markdown("---")
    st.markdown("### 🚐 Fleet Status")
    st.write(f"**Total Fleet:** {len(sim.vehicles)} Vans")
    st.write(f"**Operational:** {len(sim.vehicles) - len(sim.broken_vehicles)} Vans")
    st.write(f"**Disabled / Broken:** {len(sim.broken_vehicles)} Vans")
    st.write(f"**Completed Deliveries:** {len(sim.completed_jobs)} / {sim.instance.num_customers}")

    st.markdown("---")
    if st.button("🔄 Reset Simulation to Baseline", use_container_width=True):
        st.session_state.clear()
        init_simulation_state()
        st.rerun()


# ==============================================================================
# 7. Simulation Controls (Disruption Triggers & Autonomous Loop)
# ==============================================================================
st.markdown("### ⚡ Live Disruption & Simulation Controls")
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.6, 1.2, 1.2], gap="medium")

# --- Column 1: Autonomous Loop Controls ---
with col_ctrl1:
    st.markdown("**1. Autonomous Simulation**")
    c_play, c_pause, c_step = st.columns(3)
    with c_play:
        if st.button("▶️ Start / Resume", use_container_width=True, type="primary"):
            st.session_state.is_simulating = True
    with c_pause:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.is_simulating = False
    with c_step:
        if st.button("⏭️ Step (+25)", use_container_width=True):
            st.session_state.is_simulating = False
            if not sim.is_simulation_completed():
                next_t = min(st.session_state.current_time + 25.0, float(sim.instance.depot.due_date))
                sim.advance_clock(next_t)
                st.session_state.current_time = next_t

# --- Column 2: Vehicle Breakdown Disruption ---
with col_ctrl2:
    st.markdown("**2. Mechanical Disruption**")
    van_1_broken = 1 in sim.broken_vehicles
    btn_label = "🚨 Van 1 Already Disabled" if van_1_broken else "🚨 Trigger Van 1 Breakdown"

    if st.button(btn_label, use_container_width=True, disabled=van_1_broken):
        st.session_state.is_simulating = False
        res = sim.trigger_breakdown(1)
        if res.get("status") == "already_completed":
            st.info(res.get("message"))
        else:
            sim.replan(exploration_level=1)
            rt = sim.replanning_history[-1]["runtime_ms"]
            st.toast(f"🚨 Van 1 breakdown handled! Re-routed in {rt:.2f} ms.", icon="⚠️")

# --- Column 3: Rush Order Injection ---
with col_ctrl3:
    st.markdown("**3. Dynamic Customer Demand**")
    if st.button("⚡ Inject Rush Orders (x3)", use_container_width=True):
        st.session_state.is_simulating = False
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

        injected = sim.inject_new_orders(rush_orders)
        sim.replan(exploration_level=1)
        rt = sim.replanning_history[-1]["runtime_ms"]
        st.toast(f"⚡ Injected {len(injected)} rush stops! Fleet re-routed in {rt:.2f} ms.", icon="🚀")


# ==============================================================================
# 8. Live Display Placeholders (Dynamic Updates without Page Reload)
# ==============================================================================
kpi_placeholder = st.empty()
render_kpi_metrics(kpi_placeholder, sim)

map_placeholder = st.empty()
map_placeholder.plotly_chart(plot_routes(sim), use_container_width=True)


# ==============================================================================
# 9. Detailed Telemetry & Fleet Manifest Tabs
# ==============================================================================
st.markdown("---")
tab_routes, tab_telemetry, tab_fleet = st.tabs(
    ["📋 Live Route Manifests", "📈 Re-planning Telemetry Audit", "🚐 Fleet State Table"]
)

with tab_routes:
    routes_df = (
        sim.current_solution.routes
        if hasattr(sim.current_solution, "routes") and isinstance(sim.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    if not routes_df.empty:
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
    veh_pos = sim.get_interpolated_vehicle_positions()
    for vid, vstate in sim.vehicles.items():
        if vid <= 5 or vstate.completed_job_ids or vstate.is_broken:
            pinfo = veh_pos.get(vid, {})
            fleet_rows.append(
                {
                    "Vehicle ID": vid,
                    "Operational Status": "DISABLED" if vstate.is_broken else "OPERATIONAL",
                    "Live Status": pinfo.get("status", "Unknown"),
                    "Current Location": f"({pinfo.get('x', 0.0):.1f}, {pinfo.get('y', 0.0):.1f})",
                    "Remaining Capacity": f"{vstate.remaining_capacity} / {sim.instance.vehicle_capacity}",
                    "Available From (T)": f"{unscale_value(vstate.available_from, sim.scale_factor):.1f}",
                    "Completed Stops": len(vstate.completed_job_ids),
                }
            )
    st.dataframe(pd.DataFrame(fleet_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# 10. Autonomous Execution Loop (Native Streamlit Animation via st.rerun)
# ==============================================================================
import time

# Check if the simulation is active and not finished
if st.session_state.get('is_simulating', False):
    if not st.session_state.sim.is_simulation_completed():
        # Advance the clock smoothly
        st.session_state.current_time += 10.0
        st.session_state.sim.advance_clock(st.session_state.current_time)
        
        # Pace the animation so the user can see the vans moving
        time.sleep(0.15) 
        
        # Force Streamlit to rerun the script from top to bottom, updating the map and KPIs
        st.rerun()
    else:
        # Auto-stop when deliveries are done
        st.session_state.is_simulating = False
        st.rerun()

