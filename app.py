"""
NEXAI 2026 Hackathon - Track 4: Supply Chain & Logistics
Problem Statement 4.3: Last-Mile Route Planner with Live Re-planning (VRPTW)
Optimization Engine: pyvroom (High-performance VROOM C++ Bindings)

Interactive Dispatcher Dashboard with Autonomous Simulation Loop and Live Disruption Engine.
Light Mode Shadcn UI design with high-contrast monochrome palette, subtle accents, and Poppins typography.
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

# Page Configuration: Clean, minimal, light mode
st.set_page_config(
    page_title="NEXAI | Autonomous Route Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shadcn UI Light Mode Design System & Global Styles
st.markdown(
    """
    <style>
    /* Google Fonts: Poppins & Material Symbols */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Global Typography (Excluding Material Icons) */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stHeader"], p, h1, h2, h3, h4, h5, h6, input, select, textarea,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
    .section-title, .control-label, [data-baseweb="tab"] {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    button p, button span:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Crucial: Explicitly protect and style Material Symbols font ligatures */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 1.2rem !important;
        line-height: 1 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        text-rendering: optimizeLegibility !important;
    }

    /* Base Canvas (Pure Light) */
    .stApp {
        background-color: #ffffff !important;
        color: #09090b !important;
    }

    /* Minimalist Light Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #ffffff;
    }
    ::-webkit-scrollbar-thumb {
        background: #e4e4e7;
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #d4d4d8;
    }

    /* Top Navigation Header */
    [data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #e4e4e7 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e4e4e7 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #e4e4e7 !important;
        margin: 1rem 0 !important;
    }

    /* Metric KPI Cards (Shadcn Light Dashboard style) */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e4e4e7 !important;
        border-radius: 8px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #d4d4d8 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-weight: 500 !important;
        color: #71717a !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 600 !important;
        color: #09090b !important;
        letter-spacing: -0.02em !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
        font-weight: 400 !important;
        color: #71717a !important;
    }

    /* Buttons: Shadcn Light Mode styling */
    button {
        font-size: 0.85rem !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }
    /* Primary Action Button (Solid Crisp Black with White Text) */
    button[kind="primary"] {
        background-color: #09090b !important;
        color: #ffffff !important;
        border: 1px solid #09090b !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    button[kind="primary"]:hover {
        background-color: #27272a !important;
        border-color: #27272a !important;
        color: #ffffff !important;
    }
    button[kind="primary"] [data-testid="stIconMaterial"],
    button[kind="primary"] [class*="material-symbols"] {
        color: #ffffff !important;
    }

    /* Secondary Action Button (Crisp White with Subtle Border) */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #09090b !important;
        border: 1px solid #e4e4e7 !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }
    button[kind="secondary"]:hover {
        background-color: #f4f4f5 !important;
        border-color: #d4d4d8 !important;
        color: #09090b !important;
    }
    button[kind="secondary"] [data-testid="stIconMaterial"],
    button[kind="secondary"] [class*="material-symbols"] {
        color: #09090b !important;
    }
    button:disabled {
        opacity: 0.45 !important;
        cursor: not-allowed !important;
    }

    /* Tabs (Shadcn Light Segmented Control style) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f4f4f5 !important;
        border: 1px solid #e4e4e7 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 34px !important;
        border-radius: 6px !important;
        padding: 0 16px !important;
        color: #71717a !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #09090b !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] {
        border: 1px solid #e4e4e7 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: #ffffff !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
    }

    /* Section Subheadings */
    h2, h3, h4 {
        color: #09090b !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #09090b;
        margin-bottom: 0.5rem;
    }
    .control-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #71717a;
        margin-bottom: 0.4rem;
    }

    /* Toast Notification Styling */
    [data-testid="stToast"] {
        background-color: #ffffff !important;
        border: 1px solid #e4e4e7 !important;
        color: #09090b !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
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
    with st.spinner("Initializing Solomon C101 Benchmark and pyvroom engine..."):
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
# 3. Plotly Route & Map Visualization (Light Mode Minimalist Palette)
# ==============================================================================
# High-contrast modern palette optimized for light mode canvas
ROUTE_COLORS = [
    "#09090b",  # Lead Van 1 (Crisp Charcoal / Black)
    "#2563eb",  # Van 2 (Cobalt Blue)
    "#059669",  # Van 3 (Emerald Green)
    "#7c3aed",  # Van 4 (Violet)
    "#d97706",  # Van 5 (Amber)
    "#475569",  # Van 6 (Slate)
    "#db2777",  # Van 7 (Rose)
    "#0891b2",  # Van 8 (Cyan)
]


def plot_routes(sim_state: SimulationState) -> go.Figure:
    """
    Generate an interactive 2D spatial dispatch map with active routes and dynamic nodes.
    Minimalist light mode aesthetic with high-contrast routes and subtle grid.
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
                    f"<b>Van #{v_id}</b> | Stop {step_num + 1} [{step_type}]<br>"
                    f"Location: ({x:.1f}, {y:.1f})<br>"
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
                    line=dict(color=color, width=2.4),
                    opacity=0.88,
                    name=f"Van {v_id}",
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
            f"<b>Customer #{c_id} [Delivered]</b><br>"
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
                    size=9,
                    color="#e2e8f0",
                    line=dict(color="#94a3b8", width=1.2),
                ),
                text=[f"#{cid}" for cid in sim_state.completed_jobs.keys()],
                textposition="bottom center",
                textfont=dict(size=8, color="#64748b", family="Poppins, sans-serif"),
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
            f"<b>{'[Rush Order]' if is_rush else 'Customer'} #{job_id} [Pending]</b><br>"
            f"Location: ({x:.1f}, {y:.1f})<br>"
            f"Demand: {demand} units<br>"
            f"Time Window: [{ready_t}, {due_t}]"
        )

        if is_rush:
            rush_x.append(x)
            rush_y.append(y)
            rush_hover.append(hover_info)
            rush_labels.append(f"R-{job_id}")
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
                    size=9,
                    color="#ffffff",
                    line=dict(color="#09090b", width=1.5),
                ),
                text=pend_labels,
                textposition="top center",
                textfont=dict(size=8, color="#475569", family="Poppins, sans-serif"),
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
                    size=13,
                    color="#f59e0b",
                    line=dict(color="#09090b", width=1.5),
                ),
                text=rush_labels,
                textposition="top right",
                textfont=dict(size=9, color="#b45309", family="Poppins, sans-serif"),
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
            broken_labels.append(f"Van {v_id} [Offline]")
            broken_hover.append(
                f"<b>Van #{v_id} [Disabled / Breakdown]</b><br>"
                f"Coordinates: ({p_info['x']:.1f}, {p_info['y']:.1f})<br>"
                f"Status: {p_info['status']}<br>"
                f"<i>Pending deliveries reassigned to active fleet.</i>"
            )
        elif p_info["is_active"]:
            van_x.append(p_info["x"])
            van_y.append(p_info["y"])
            van_labels.append(f"Van {v_id}")
            van_hover.append(
                f"<b>Van #{v_id} [Active]</b><br>"
                f"Coordinates: ({p_info['x']:.1f}, {p_info['y']:.1f})<br>"
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
                    size=14,
                    color="#09090b",
                    line=dict(color="#ffffff", width=2.0),
                ),
                text=van_labels,
                textposition="bottom right",
                textfont=dict(size=9, color="#09090b", family="Poppins, sans-serif"),
                name="Active Fleet",
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
                    size=16,
                    color="#ef4444",
                    line=dict(color="#ffffff", width=2.0),
                ),
                text=broken_labels,
                textposition="top center",
                textfont=dict(size=9, color="#dc2626", family="Poppins, sans-serif"),
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
                size=16,
                color="#09090b",
                line=dict(color="#ffffff", width=2.0),
            ),
            text=["DEPOT (0)"],
            textposition="bottom center",
            textfont=dict(size=10, color="#09090b", family="Poppins, sans-serif"),
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

    # Layout styling: clean light minimalist theme with Poppins font
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fcfcfd",
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Poppins, sans-serif", color="#64748b", size=10),
        xaxis=dict(
            title=dict(text="X Coordinate", font=dict(family="Poppins, sans-serif", size=11, color="#64748b")),
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(family="Poppins, sans-serif", color="#64748b", size=9),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Y Coordinate", font=dict(family="Poppins, sans-serif", size=11, color="#64748b")),
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(family="Poppins, sans-serif", color="#64748b", size=9),
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
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#e2e8f0",
            borderwidth=1,
            font=dict(family="Poppins, sans-serif", size=9, color="#334155"),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            font_size=11,
            font_family="Poppins, sans-serif",
            font_color="#09090b",
        ),
        height=530,
    )

    return fig


# ==============================================================================
# 4. KPI Placeholder Renderer (Shadcn Light Metric Style)
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
    clock_status = "Complete" if is_done else ("Simulating" if st.session_state.is_simulating else "Paused")

    with placeholder.container():
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            label="Total Travel Distance",
            value=f"{unscaled_dist:.1f} km",
            delta=f"Clock T = {st.session_state.current_time:.1f} ({clock_status})",
            delta_color="off",
        )
        m2.metric(
            label="Active Fleet in Transit",
            value=f"{active_in_transit} Vans",
            delta=f"{total_active_fleet} operational",
            delta_color="off",
        )
        m3.metric(
            label="Pending Deliveries",
            value=f"{pending_cnt} Stops",
            delta=f"{comp_cnt} completed",
            delta_color="off",
        )
        m4.metric(
            label="Last Re-plan Latency",
            value=f"{latest_runtime:.2f} ms",
            delta="Sub-5ms optimal",
            delta_color="off",
        )


# ==============================================================================
# 5. Sidebar: Setup & Simulation Summary
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="section-title">Dispatch Controls</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 1px; background-color: #e4e4e7; margin: 0.75rem 0 1.25rem 0;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="control-label">Virtual Clock</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.caption("Current Time")
        st.markdown(f'<div style="font-size: 1.25rem; font-weight: 600; color: #09090b;">T = {st.session_state.current_time:.1f}</div>', unsafe_allow_html=True)
    with col_c2:
        st.caption("Depot Horizon")
        st.markdown(f'<div style="font-size: 1.25rem; font-weight: 600; color: #71717a;">T = {sim.instance.depot.due_date:.0f}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 1px; background-color: #e4e4e7; margin: 1.25rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="control-label">Fleet Status</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; gap: 0.55rem; font-size: 0.85rem; margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; color: #71717a;">
                <span>Total Fleet</span>
                <span style="color: #09090b; font-weight: 500;">{len(sim.vehicles)} Vans</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #71717a;">
                <span>Operational</span>
                <span style="color: #10b981; font-weight: 500;">{len(sim.vehicles) - len(sim.broken_vehicles)} Vans</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #71717a;">
                <span>Disabled</span>
                <span style="color: {'#ef4444' if len(sim.broken_vehicles) > 0 else '#a1a1aa'}; font-weight: 500;">{len(sim.broken_vehicles)} Vans</span>
            </div>
            <div style="display: flex; justify-content: space-between; color: #71717a;">
                <span>Completed Deliveries</span>
                <span style="color: #09090b; font-weight: 500;">{len(sim.completed_jobs)} / {sim.instance.num_customers}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height: 1px; background-color: #e4e4e7; margin: 1.25rem 0;"></div>', unsafe_allow_html=True)
    if st.button("Reset Simulation", icon=":material/restart_alt:", width="stretch"):
        st.session_state.clear()
        init_simulation_state()
        st.rerun()


# ==============================================================================
# 6. Simulation Controls (Disruption Triggers & Autonomous Loop)
# ==============================================================================
st.markdown('<div class="section-title">Simulation & Disruption Controls</div>', unsafe_allow_html=True)
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.6, 1.2, 1.2], gap="medium")

# --- Column 1: Autonomous Loop Controls ---
with col_ctrl1:
    st.markdown('<div class="control-label">Autonomous Simulation</div>', unsafe_allow_html=True)
    is_running = st.session_state.get("is_simulating", False)
    is_completed = sim.is_simulation_completed()
    play_label = "Resume" if st.session_state.current_time > 0 else "Start"

    c_play, c_pause, c_step = st.columns(3)
    with c_play:
        if st.button(
            play_label,
            icon=":material/play_arrow:",
            width="stretch",
            type="primary" if not is_running else "secondary",
            disabled=is_running or is_completed,
        ):
            st.session_state.is_simulating = True
            st.rerun()
    with c_pause:
        if st.button(
            "Pause",
            icon=":material/pause:",
            width="stretch",
            type="primary" if is_running else "secondary",
            disabled=not is_running,
        ):
            st.session_state.is_simulating = False
            st.rerun()
    with c_step:
        if st.button(
            "Step (+25s)",
            icon=":material/skip_next:",
            width="stretch",
            disabled=is_running or is_completed,
        ):
            st.session_state.is_simulating = False
            if not sim.is_simulation_completed():
                next_t = min(st.session_state.current_time + 25.0, float(sim.instance.depot.due_date))
                sim.advance_clock(next_t)
                st.session_state.current_time = next_t
                st.rerun()

# --- Column 2: Vehicle Breakdown Disruption ---
with col_ctrl2:
    st.markdown('<div class="control-label">Mechanical Disruption</div>', unsafe_allow_html=True)
    van_1_broken = 1 in sim.broken_vehicles
    btn_label = "Van 1 Inactive" if van_1_broken else "Trigger Breakdown"

    if st.button(btn_label, icon=":material/warning:", width="stretch", disabled=van_1_broken):
        st.session_state.is_simulating = False
        res = sim.trigger_breakdown(1)
        if res.get("status") == "already_completed":
            st.info(res.get("message"))
        else:
            sim.replan(exploration_level=1)
            rt = sim.replanning_history[-1]["runtime_ms"]
            st.toast(f"Van 1 breakdown logged. Fleet re-routed in {rt:.2f} ms.")

# --- Column 3: Rush Order Injection ---
with col_ctrl3:
    st.markdown('<div class="control-label">Dynamic Customer Demand</div>', unsafe_allow_html=True)
    if st.button("Inject Rush Orders (+3)", icon=":material/bolt:", width="stretch"):
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
        st.toast(f"Injected {len(injected)} rush orders. Fleet re-routed in {rt:.2f} ms.")


# ==============================================================================
# 7. Live Display Placeholders (Dynamic Updates without Page Reload)
# ==============================================================================
kpi_placeholder = st.empty()
render_kpi_metrics(kpi_placeholder, sim)

map_placeholder = st.empty()
map_placeholder.plotly_chart(plot_routes(sim), width="stretch")


# ==============================================================================
# 8. Detailed Telemetry & Fleet Manifest Tabs
# ==============================================================================
st.markdown('<div style="height: 1px; background-color: #e4e4e7; margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
tab_routes, tab_telemetry, tab_fleet = st.tabs(
    ["Route Manifests", "Telemetry Audit", "Fleet State"]
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
            width="stretch",
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
            width="stretch",
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
    st.dataframe(pd.DataFrame(fleet_rows), width="stretch", hide_index=True)


# ==============================================================================
# 9. Autonomous Execution Loop (Native Streamlit Animation via st.rerun)
# ==============================================================================
if st.session_state.get('is_simulating', False):
    if not st.session_state.sim.is_simulation_completed():
        st.session_state.current_time += 10.0
        st.session_state.sim.advance_clock(st.session_state.current_time)
        time.sleep(0.15)
        st.rerun()
    else:
        st.session_state.is_simulating = False
        st.rerun()
