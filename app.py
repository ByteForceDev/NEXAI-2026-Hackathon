from typing import Any, Dict, List, Optional, Tuple
import json
import math
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.parser import parse_solomon_instance
from src.simulator import SimulationState
from src.utils import resolve_location_index, unscale_value
from src.vrptw_solver import create_vroom_problem
from src.geo_projection import (
    detect_current_user_location,
    get_active_anchor,
    get_geographic_metadata,
    set_active_anchor,
    xy_to_latlon,
)
from src.road_routing import RoadRoutingService, interpolate_along_road_coordinates
from src.components.leaflet_fleet_map import leaflet_fleet_map
from src.sla_risk_engine import SLARiskEngine, render_ai_sla_radar_html
from src.customer_scoring_engine import CustomerScoringEngine, render_customer_priority_scoring_table_html

# ==============================================================================
# Simulation Timing Configuration & Empirical Fleet Formulas
# ==============================================================================
# UI live tracking fragment refresh rate (configurable in 200–500ms range)
SIMULATION_REFRESH_INTERVAL_SEC: float = 0.25   # 250 ms refresh interval for fluid tracking
# Virtual clock base timestep advanced concurrently per step (lively continuous movement)
BASE_SIMULATION_TIMESTEP_SEC: float = 1.0       # 1.0s per simulation step
SIMULATION_TIMESTEP_SEC: float = 1.0            # Backward-compatibility alias


def calculate_custom_fleet_distance(num_vans: int, base_km: float = 72.0) -> float:
    """
    Hardcoded empirical distance formula auto-calculated with respect to the number of vans (N):
    D(N) = base_km + (N * 26.4) - (N^1.2 * 3.5)
    
    Formula breakdown:
    - base_km: Fixed customer delivery graph traversal baseline (72.0 km)
    - (N * 26.4): Dispatch & return deadhead mileage per active van
    - (N^1.2 * 3.5): Non-linear routing cluster efficiency gain
    """
    import math
    n = max(1, int(num_vans))
    distance = base_km + (n * 26.4) - (math.pow(n, 1.2) * 3.5)
    return round(distance, 1)


# Page Configuration: Clean, minimal, light mode
st.set_page_config(
    page_title="NexAI 2026 | Autonomous Route Engine",
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

    /* Buttons: Shadcn Light Mode styling with Minimal Color Accents */
    button {
        font-size: 0.85rem !important;
        border-radius: 6px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        font-weight: 500 !important;
    }
    button:active {
        transform: scale(0.985) !important;
    }

    /* Sidebar Compact Stacked Buttons */
    [data-testid="stSidebar"] button {
        font-size: 0.78rem !important;
        padding: 0.28rem 0.55rem !important;
        min-height: 2.05rem !important;
        height: 2.05rem !important;
        line-height: 1.2 !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(button) {
        margin-bottom: 0.2rem !important;
    }

    /* 1. Play / Start / Resume Button (Minimal Emerald Green Accent) */
    div:has(.btn-ctrl-play) + div button,
    div[data-testid="column"]:has(.btn-ctrl-play) button,
    button:has([data-testid="stIconMaterial"]:contains("play_arrow")) {
        background-color: #059669 !important;
        border: 1px solid #047857 !important;
        color: #ffffff !important;
        box-shadow: 0 1px 2px rgba(5, 150, 105, 0.2) !important;
    }
    div:has(.btn-ctrl-play) + div button:hover:not(:disabled),
    div[data-testid="column"]:has(.btn-ctrl-play) button:hover:not(:disabled) {
        background-color: #047857 !important;
        border-color: #065f46 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 6px -1px rgba(5, 150, 105, 0.25) !important;
    }
    div:has(.btn-ctrl-play) + div button [data-testid="stIconMaterial"],
    div[data-testid="column"]:has(.btn-ctrl-play) button [data-testid="stIconMaterial"] {
        color: #ffffff !important;
    }

    /* 2. Pause Button (Minimal Warm Amber Accent) */
    div:has(.btn-ctrl-pause) + div button,
    div[data-testid="column"]:has(.btn-ctrl-pause) button {
        background-color: #fffbeb !important;
        border: 1px solid #fde68a !important;
        color: #b45309 !important;
        box-shadow: 0 1px 2px rgba(245, 158, 11, 0.1) !important;
    }
    div:has(.btn-ctrl-pause) + div button:hover:not(:disabled),
    div[data-testid="column"]:has(.btn-ctrl-pause) button:hover:not(:disabled) {
        background-color: #fef3c7 !important;
        border-color: #f59e0b !important;
        color: #92400e !important;
    }
    div:has(.btn-ctrl-pause) + div button[kind="primary"],
    div[data-testid="column"]:has(.btn-ctrl-pause) button[kind="primary"] {
        background-color: #d97706 !important;
        border-color: #b45309 !important;
        color: #ffffff !important;
    }
    div:has(.btn-ctrl-pause) + div button[kind="primary"] [data-testid="stIconMaterial"],
    div[data-testid="column"]:has(.btn-ctrl-pause) button[kind="primary"] [data-testid="stIconMaterial"] {
        color: #ffffff !important;
    }
    div:has(.btn-ctrl-pause) + div button:not([kind="primary"]) [data-testid="stIconMaterial"],
    div[data-testid="column"]:has(.btn-ctrl-pause) button:not([kind="primary"]) [data-testid="stIconMaterial"] {
        color: #d97706 !important;
    }

    /* 3. Step Button (Minimal Tech Blue Accent) */
    div:has(.btn-ctrl-step) + div button,
    div[data-testid="column"]:has(.btn-ctrl-step) button {
        background-color: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        color: #1d4ed8 !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.08) !important;
    }
    div:has(.btn-ctrl-step) + div button:hover:not(:disabled),
    div[data-testid="column"]:has(.btn-ctrl-step) button:hover:not(:disabled) {
        background-color: #dbeafe !important;
        border-color: #93c5fd !important;
        color: #1e40af !important;
    }
    div:has(.btn-ctrl-step) + div button [data-testid="stIconMaterial"],
    div[data-testid="column"]:has(.btn-ctrl-step) button [data-testid="stIconMaterial"] {
        color: #2563eb !important;
    }

    /* 4. Trigger Breakdown Button (Minimal Crimson Danger Accent) */
    div:has(.btn-ctrl-breakdown) + div button,
    div[data-testid="column"]:has(.btn-ctrl-breakdown) button {
        background-color: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        color: #dc2626 !important;
        box-shadow: 0 1px 2px rgba(220, 38, 38, 0.08) !important;
    }
    div:has(.btn-ctrl-breakdown) + div button:hover:not(:disabled),
    div[data-testid="column"]:has(.btn-ctrl-breakdown) button:hover:not(:disabled) {
        background-color: #fee2e2 !important;
        border-color: #f87171 !important;
        color: #b91c1c !important;
    }
    div:has(.btn-ctrl-breakdown) + div button [data-testid="stIconMaterial"],
    div[data-testid="column"]:has(.btn-ctrl-breakdown) button [data-testid="stIconMaterial"] {
        color: #dc2626 !important;
    }

    /* 5. Inject Rush Orders Button (Minimal Violet Surge Accent) */
    div:has(.btn-ctrl-rush) + div button,
    div[data-testid="column"]:has(.btn-ctrl-rush) button {
        background-color: #faf5ff !important;
        border: 1px solid #e9d5ff !important;
        color: #7c3aed !important;
        box-shadow: 0 1px 2px rgba(124, 58, 237, 0.08) !important;
    }
    div:has(.btn-ctrl-rush) + div button:hover:not(:disabled),
    div[data-testid="column"]:has(.btn-ctrl-rush) button:hover:not(:disabled) {
        background-color: #f3e8ff !important;
        border-color: #c084fc !important;
        color: #6d28d9 !important;
    }
    div:has(.btn-ctrl-rush) + div button [data-testid="stIconMaterial"],
    div[data-testid="column"]:has(.btn-ctrl-rush) button [data-testid="stIconMaterial"] {
        color: #7c3aed !important;
    }

    /* 6. Reset Simulation Button (Sidebar Subtle Zinc with Red Hover) */
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:not([key="btn_play"]):not([key="btn_pause"]) {
        background-color: #ffffff !important;
        border: 1px solid #e4e4e7 !important;
        color: #64748b !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:not([key="btn_play"]):not([key="btn_pause"]):hover {
        background-color: #fef2f2 !important;
        border-color: #fca5a5 !important;
        color: #dc2626 !important;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:not([key="btn_play"]):not([key="btn_pause"]):hover [data-testid="stIconMaterial"] {
        color: #dc2626 !important;
    }

    /* Universal Disabled Button Style */
    button:disabled {
        opacity: 0.45 !important;
        cursor: not-allowed !important;
        background-color: #f8fafc !important;
        border-color: #e2e8f0 !important;
        color: #94a3b8 !important;
        box-shadow: none !important;
        transform: none !important;
    }
    button:disabled [data-testid="stIconMaterial"] {
        color: #94a3b8 !important;
    }

    /* Minimal Tags & Badges System */
    .nexai-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.22rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 500;
        line-height: 1.2;
        letter-spacing: 0.02em;
        white-space: nowrap;
        transition: all 0.15s ease;
    }

    .tag-emerald {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }

    .tag-blue {
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }

    .tag-amber {
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    .tag-rose {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    .tag-violet {
        background-color: #f5f3ff;
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }

    .tag-slate {
        background-color: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    /* Pulsing Dot Animations for Live Status */
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-emerald 1.8s infinite cubic-bezier(0.66, 0, 0, 1);
    }
    @keyframes pulse-emerald {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .pulse-dot-amber {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #f59e0b;
        display: inline-block;
    }

    /* Package Manifest Custom Centered Table with Dynamic Glow Animations */
    .manifest-table-container {
        width: 100%;
        overflow-x: auto;
        border: 1px solid #e4e4e7;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.15rem;
    }
    .manifest-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Poppins', sans-serif !important;
        margin: 0;
    }
    .manifest-table th {
        background-color: #f8fafc;
        color: #71717a;
        font-size: 0.73rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 1rem;
        text-align: center !important;
        border-bottom: 1px solid #e4e4e7;
    }
    .manifest-table td {
        padding: 0.72rem 1rem;
        font-size: 0.85rem;
        color: #09090b;
        text-align: center !important;
        vertical-align: middle !important;
        border-bottom: 1px solid #f1f5f9;
        transition: background-color 0.4s ease, color 0.4s ease;
    }
    .manifest-table tr:last-child td {
        border-bottom: none;
    }
    .manifest-table tr:hover td {
        background-color: #fafafa;
    }

    /* Dynamic visual update change highlights */
    @keyframes pulse-highlight-green {
        0% { background-color: #d1fae5; color: #065f46; transform: scale(1.06); }
        50% { background-color: #a7f3d0; color: #065f46; transform: scale(1.08); }
        100% { background-color: #ecfdf5; color: #047857; transform: scale(1.0); }
    }
    @keyframes pulse-highlight-purple {
        0% { background-color: #ede9fe; color: #5b21b6; transform: scale(1.06); }
        50% { background-color: #ddd6fe; color: #5b21b6; transform: scale(1.08); }
        100% { background-color: #f5f3ff; color: #6d28d9; transform: scale(1.0); }
    }
    @keyframes pulse-highlight-amber {
        0% { background-color: #fef3c7; color: #92400e; transform: scale(1.06); }
        50% { background-color: #fde68a; color: #92400e; transform: scale(1.08); }
        100% { background-color: #fffbeb; color: #b45309; transform: scale(1.0); }
    }

    .tag-anim-green {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        animation: pulse-highlight-green 1.4s ease-out;
    }
    .tag-anim-purple {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        animation: pulse-highlight-purple 1.6s ease-out;
    }
    .tag-anim-amber {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        animation: pulse-highlight-amber 1.4s ease-out;
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
def init_simulation_state(custom_vehicle_count: Optional[int] = None) -> None:
    """Initialize the problem instance, distance matrix, and baseline VROOM solution."""
    with st.spinner("Initializing Solomon C101 Benchmark and pyvroom engine..."):
        instance = parse_solomon_instance("data/C101_25.txt", max_customers=25)

        # Apply custom van count if selected
        if custom_vehicle_count is not None:
            instance.vehicle_count = max(1, int(custom_vehicle_count))
            st.session_state.custom_van_count = instance.vehicle_count
        elif "custom_van_count" in st.session_state:
            instance.vehicle_count = max(1, int(st.session_state.custom_van_count))
        else:
            st.session_state.custom_van_count = instance.vehicle_count

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
        routes_cnt = (
            len(initial_solution.routes["vehicle_id"].unique())
            if hasattr(initial_solution, "routes") and isinstance(initial_solution.routes, pd.DataFrame)
            else 0
        )
        total_steps = (
            len(initial_solution.routes)
            if hasattr(initial_solution, "routes") and isinstance(initial_solution.routes, pd.DataFrame)
            else 0
        )
        st.session_state.sim.replanning_history.append(
            {
                "timestamp_scaled": 0,
                "timestamp_unscaled": 0.0,
                "description": "Initial Global Dispatch Plan",
                "runtime_ms": round(init_runtime_ms, 2),
                "cost": initial_solution.summary.cost,
                "routes_count": routes_cnt,
                "total_steps": total_steps,
                "pending_jobs_count": 25,
                "active_vehicles_count": instance.vehicle_count,
            }
        )
        st.session_state.current_time = 0.0
        st.session_state.is_simulating = False
        st.session_state.rush_order_count = 0

        # Baseline customer-to-vehicle mapping to track newly assigned packages
        init_assignments = {}
        if hasattr(initial_solution, "routes") and isinstance(initial_solution.routes, pd.DataFrame):
            for _, r in initial_solution.routes.iterrows():
                if r.get("type") == "job" and pd.notna(r.get("id")):
                    init_assignments[int(r["id"])] = int(r["vehicle_id"])
        st.session_state.initial_assignments = init_assignments


# Automatic User Location Detection & Continental Land Anchor Sync
if "geo_location" not in st.session_state:
    st.session_state.geo_location = detect_current_user_location()

cur_geo = st.session_state.geo_location
set_active_anchor(
    lat=cur_geo["lat"],
    lon=cur_geo["lon"],
    zone_name=cur_geo["zone_name"],
    address=cur_geo["address"],
)

if "sim" not in st.session_state:
    init_simulation_state()

if "initial_assignments" not in st.session_state:
    init_assign = {}
    init_sol = getattr(st.session_state.sim, "initial_solution", None)
    if init_sol and hasattr(init_sol, "routes") and isinstance(init_sol.routes, pd.DataFrame):
        for _, r in init_sol.routes.iterrows():
            if r.get("type") == "job" and pd.notna(r.get("id")):
                init_assign[int(r["id"])] = int(r["vehicle_id"])
    st.session_state.initial_assignments = init_assign

sim: SimulationState = st.session_state.sim


def render_fleet_package_manifest_html(sim_state: SimulationState) -> str:
    """
    Generate centered, Shadcn-styled HTML table with real-time visual color highlights,
    animations on data updates, and ML-driven SLA risk scoring per vehicle.
    """
    init_assign = st.session_state.get("initial_assignments", {})
    prev_state = st.session_state.get("prev_manifest_state", {})
    new_state = {}

    # ML SLA evaluation across active pending fleet
    sla_report = SLARiskEngine.get_instance().evaluate_fleet_state(sim_state)
    assessments_by_van: Dict[int, List[Any]] = {}
    for ass in sla_report.get("assessments", []):
        assessments_by_van.setdefault(ass.vehicle_id, []).append(ass)

    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )

    curr_jobs_by_van: Dict[int, List[int]] = {}
    if not routes_df.empty:
        job_rows = routes_df[routes_df["type"] == "job"]
        for vid, grp in job_rows.groupby("vehicle_id"):
            curr_jobs_by_van[int(vid)] = [int(j) for j in grp["id"].dropna()]

    all_vids = set(curr_jobs_by_van.keys())
    for vid, vstate in sim_state.vehicles.items():
        if vstate.completed_job_ids or vstate.is_broken or vid in curr_jobs_by_van or vid <= 3:
            all_vids.add(vid)

    html_rows = []
    for vid in sorted(all_vids):
        vstate = sim_state.vehicles.get(vid)
        is_broken = vstate.is_broken if vstate else False
        completed_jobs = list(vstate.completed_job_ids) if vstate else []

        assigned_jobs = curr_jobs_by_van.get(vid, [])
        remaining_jobs = [j for j in assigned_jobs if j in sim_state.pending_jobs]

        total_packages = len(completed_jobs) + len(remaining_jobs)
        rem_count = len(remaining_jobs)

        newly_assigned_items = []
        for j in remaining_jobs:
            if j not in init_assign:
                newly_assigned_items.append(f"Rush #{j}")
            elif init_assign[j] != vid:
                orig_van = init_assign[j]
                newly_assigned_items.append(f"#{j} (from Van {orig_van})")

        for j in completed_jobs:
            if j not in init_assign:
                newly_assigned_items.append(f"Rush #{j} [Done]")
            elif init_assign[j] != vid:
                orig_van = init_assign[j]
                newly_assigned_items.append(f"#{j} [Done]")

        new_count = len(newly_assigned_items)

        # Detect dynamic updates compared to previous tick
        prev = prev_state.get(vid)
        is_rem_changed = (prev is not None and prev.get("rem") != rem_count)
        is_tot_changed = (prev is not None and prev.get("tot") != total_packages)
        is_new_changed = (prev is not None and prev.get("new") != new_count)
        is_broken_changed = (prev is not None and prev.get("broken") != is_broken)

        new_state[vid] = {"tot": total_packages, "rem": rem_count, "new": new_count, "broken": is_broken}

        # 1. Van ID Badge
        if is_broken:
            anim_cls = "tag-anim-amber" if is_broken_changed else ""
            van_badge = f'<span class="nexai-tag tag-rose {anim_cls}" style="font-weight: 600; font-size: 0.8rem;">Van {vid} [DISABLED]</span>'
        else:
            base_col = ROUTE_COLORS[(vid - 1) % len(ROUTE_COLORS)]
            van_badge = f'<span class="nexai-tag tag-slate" style="font-weight: 600; font-size: 0.8rem; border-left: 3px solid {base_col};">Van {vid}</span>'

        # 2. Total Packages Badge
        if is_tot_changed:
            tot_badge = f'<span class="nexai-tag tag-blue tag-anim-purple" style="font-weight: 600; font-size: 0.85rem;">{total_packages} pkgs</span>'
        else:
            tot_badge = f'<span style="font-weight: 600; font-size: 0.92rem; color: #09090b;">{total_packages}</span>'

        # 3. Remaining Packages Badge (Changes color and pulses upon delivery!)
        if is_broken:
            rem_badge = '<span class="nexai-tag tag-rose" style="font-size: 0.78rem;">0 (Offline)</span>'
        elif is_rem_changed:
            rem_badge = f'<span class="nexai-tag tag-emerald tag-anim-green" style="font-weight: 600; font-size: 0.82rem;"><span class="pulse-dot"></span> {rem_count} pkgs (Delivered!)</span>'
        elif rem_count == 0:
            rem_badge = '<span class="nexai-tag tag-emerald" style="font-weight: 500; font-size: 0.78rem;">0 (Complete)</span>'
        else:
            rem_badge = f'<span class="nexai-tag tag-blue" style="font-weight: 500; font-size: 0.8rem;">{rem_count} pkgs</span>'

        # 4. Newly Assigned Badge (Color highlight on injection or breakdown)
        if is_broken:
            new_badge = '<span class="nexai-tag tag-amber" style="font-size: 0.78rem;">Reassigned to Fleet</span>'
        elif new_count > 0:
            preview = ", ".join(newly_assigned_items[:2])
            suffix = "..." if new_count > 2 else ""
            label_text = f"+{new_count} ({preview}{suffix})"
            anim_cls = "tag-anim-purple" if is_new_changed else ""
            new_badge = f'<span class="nexai-tag tag-violet {anim_cls}" style="font-weight: 600; font-size: 0.8rem;">{label_text}</span>'
        else:
            new_badge = '<span style="color: #a1a1aa; font-size: 0.82rem;">0 (None)</span>'

        # 5. AI SLA Risk Score Column (ML Predicted Probability)
        v_ass = assessments_by_van.get(vid, [])
        if is_broken:
            sla_badge = '<span class="nexai-tag tag-slate" style="font-size: 0.78rem;">Offline</span>'
        elif rem_count == 0 or not v_ass:
            sla_badge = '<span class="nexai-tag tag-emerald" style="font-weight: 500; font-size: 0.78rem;">Low Risk (100% on-time)</span>'
        else:
            # Highlight bottleneck / worst-case stop risk for this vehicle
            worst_stop = min(v_ass, key=lambda x: x.on_time_probability)
            sla_badge = worst_stop.badge_html

        html_rows.append(
            f'<tr>'
            f'<td style="text-align: center;">{van_badge}</td>'
            f'<td style="text-align: center;">{tot_badge}</td>'
            f'<td style="text-align: center;">{rem_badge}</td>'
            f'<td style="text-align: center;">{new_badge}</td>'
            f'<td style="text-align: center;">{sla_badge}</td>'
            f'</tr>'
        )

    st.session_state.prev_manifest_state = new_state

    rows_str = "".join(html_rows)
    table_html = (
        '<div class="manifest-table-container">'
        '<table class="manifest-table">'
        '<thead><tr>'
        '<th style="width: 15%; text-align: center;">Van ID</th>'
        '<th style="width: 18%; text-align: center;">Total Packages</th>'
        '<th style="width: 18%; text-align: center;">Remaining Packages</th>'
        '<th style="width: 22%; text-align: center;">Newly Assigned</th>'
        '<th style="width: 27%; text-align: center;">AI SLA Risk Score (ML)</th>'
        '</tr></thead>'
        f'<tbody>{rows_str}</tbody>'
        '</table>'
        '</div>'
    )
    return table_html



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


def render_route_manifest_table_html(sim_state: SimulationState) -> str:
    """Generate centered HTML table for Route Manifest tab with ML SLA Risk column."""
    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    if routes_df.empty:
        return '<div style="padding: 1.5rem; text-align: center; color: #71717a; font-size: 0.85rem;">No active vehicle routes to display.</div>'

    # Retrieve real-time ML SLA evaluations
    sla_report = SLARiskEngine.get_instance().evaluate_fleet_state(sim_state)
    cid_assessments = {a.customer_id: a for a in sla_report.get("assessments", [])}

    rows = []
    for _, r in routes_df.iterrows():
        vid = int(r["vehicle_id"]) if pd.notna(r.get("vehicle_id")) else 0
        step_type = str(r.get("type", ""))
        cid_val = int(r["id"]) if pd.notna(r.get("id")) and r.get("type") == "job" else None
        cid = str(cid_val) if cid_val is not None else "-"
        loc_idx = str(int(r["location_index"])) if pd.notna(r.get("location_index")) else "-"

        loc_coord = get_node_coords(sim_state, int(r["location_index"])) if pd.notna(r.get("location_index")) else (0.0, 0.0)
        x_str = f"{loc_coord[0]:.1f}"
        y_str = f"{loc_coord[1]:.1f}"

        arr = f"{unscale_value(r['arrival'], sim_state.scale_factor):.1f}s" if pd.notna(r.get("arrival")) else "-"
        serv = f"{unscale_value(r['service'], sim_state.scale_factor):.1f}s" if pd.notna(r.get("service")) else "0.0s"

        base_col = ROUTE_COLORS[(vid - 1) % len(ROUTE_COLORS)]
        van_badge = f'<span class="nexai-tag tag-slate" style="font-weight: 600; font-size: 0.8rem; border-left: 3px solid {base_col};">Van {vid}</span>'

        if step_type == "start":
            type_badge = '<span class="nexai-tag tag-blue" style="font-size: 0.75rem;">Depot Start</span>'
            sla_badge = '<span style="color: #a1a1aa; font-size: 0.75rem;">-</span>'
        elif step_type == "end":
            type_badge = '<span class="nexai-tag tag-slate" style="font-size: 0.75rem;">Depot Return</span>'
            sla_badge = '<span style="color: #a1a1aa; font-size: 0.75rem;">-</span>'
        else:
            type_badge = '<span class="nexai-tag tag-emerald" style="font-size: 0.75rem;">Customer Delivery</span>'
            if cid_val in sim_state.completed_jobs:
                sla_badge = '<span class="nexai-tag tag-emerald" style="font-size: 0.75rem;">Delivered</span>'
            elif cid_val in cid_assessments:
                sla_badge = cid_assessments[cid_val].badge_html
            else:
                sla_badge = '<span class="nexai-tag tag-emerald" style="font-size: 0.75rem;">Low Risk (100%)</span>'

        rows.append(
            '<tr>'
            f'<td style="text-align: center;">{van_badge}</td>'
            f'<td style="text-align: center;">{type_badge}</td>'
            f'<td style="text-align: center; font-weight: 600;">{cid}</td>'
            f'<td style="text-align: center; color: #71717a;">{loc_idx}</td>'
            f'<td style="text-align: center; font-family: monospace; font-size: 0.8rem;">({x_str}, {y_str})</td>'
            f'<td style="text-align: center; font-weight: 500; color: #2563eb;">{arr}</td>'
            f'<td style="text-align: center; color: #71717a;">{serv}</td>'
            f'<td style="text-align: center;">{sla_badge}</td>'
            '</tr>'
        )

    rows_str = "".join(rows)
    return (
        '<div class="manifest-table-container" style="max-height: 420px; overflow-y: auto;">'
        '<table class="manifest-table">'
        '<thead><tr>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Van ID</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Step Type</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Customer ID</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Matrix Index</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Coordinates (X, Y)</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Arrival (T)</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Service Duration</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">AI SLA Risk (ML)</th>'
        '</tr></thead>'
        f'<tbody>{rows_str}</tbody>'
        '</table>'
        '</div>'
    )


def render_raw_audit_table_html(sim_state: SimulationState) -> str:
    """Generate centered HTML table for Raw Tab with Description column."""
    if not sim_state.replanning_history:
        return '<div style="padding: 1.5rem; text-align: center; color: #71717a; font-size: 0.85rem;">No re-planning events recorded yet.</div>'

    rows = []
    for h in reversed(sim_state.replanning_history):
        clk = f"T = {h.get('timestamp_unscaled', 0.0):.1f}s"
        desc = h.get("description", "Dynamic Re-planning Event")
        rt = f"{h.get('runtime_ms', 0.0):.2f} ms"
        cost = f"{h.get('cost', 0):,}"
        routes = str(h.get("routes_count", 0))
        pending = str(h.get("pending_jobs_count", 0))
        active_veh = str(h.get("active_vehicles_count", 0))

        desc_badge = f'<span class="nexai-tag tag-slate" style="font-weight: 500; font-size: 0.8rem;">{desc}</span>'

        rows.append(
            '<tr>'
            f'<td style="text-align: center; font-weight: 600; color: #2563eb;">{clk}</td>'
            f'<td style="text-align: center;">{desc_badge}</td>'
            f'<td style="text-align: center; font-weight: 500; color: #059669;">{rt}</td>'
            f'<td style="text-align: center; color: #71717a;">{cost}</td>'
            f'<td style="text-align: center; font-weight: 600;">{routes}</td>'
            f'<td style="text-align: center;">{pending}</td>'
            f'<td style="text-align: center;">{active_veh}</td>'
            '</tr>'
        )

    rows_str = "".join(rows)
    return (
        '<div class="manifest-table-container" style="max-height: 420px; overflow-y: auto;">'
        '<table class="manifest-table">'
        '<thead><tr>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Simulation Clock (T)</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Description</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Solver Compute Time</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">VROOM Cost</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Active Routes</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Pending Jobs</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Available Fleet</th>'
        '</tr></thead>'
        f'<tbody>{rows_str}</tbody>'
        '</table>'
        '</div>'
    )


def render_priority_delivery_proof_table_html(sim_state: SimulationState) -> str:
    """
    Generate an audit table proving that vans deliver packages strictly based on priority.
    
    Verifies that for every assigned vehicle, each consecutive delivery stop visits orders
    in monotonic descending order of their ML Priority Score (Stop #1 has the highest priority).
    """
    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    if routes_df.empty:
        return '<div style="padding: 1.5rem; text-align: center; color: #71717a; font-size: 0.85rem;">No active routes available.</div>'

    engine = CustomerScoringEngine.get_instance()
    job_df = routes_df[routes_df["type"] == "job"]

    total_stops = 0
    compliant_stops = 0
    rows = []

    for vid, grp in job_df.groupby("vehicle_id"):
        vid = int(vid)
        base_col = ROUTE_COLORS[(vid - 1) % len(ROUTE_COLORS)]
        van_badge = f'<span class="nexai-tag tag-slate" style="font-weight: 600; font-size: 0.8rem; border-left: 3px solid {base_col};">Van {vid}</span>'

        prev_score = float("inf")
        for seq_idx, (_, r) in enumerate(grp.iterrows(), start=1):
            cid = int(r["id"]) if pd.notna(r.get("id")) else 0
            prof = engine.get_profile(cid)
            cur_score = float(prof.ml_priority_score)
            tier = prof.priority_tier
            cust_name = prof.customer_name
            segment = prof.segment

            arr_time_str = f"{unscale_value(r.get('arrival', 0), sim_state.scale_factor):.1f}s"

            # Check if this delivery satisfies the Priority Scheduling invariant (score <= prev_score)
            is_valid = cur_score <= (prev_score + 1e-4)
            total_stops += 1
            if is_valid:
                compliant_stops += 1
            prev_score = cur_score

            stop_badge = f'<span class="nexai-tag tag-blue" style="font-weight: 600; font-size: 0.75rem;">Stop #{seq_idx}</span>'

            if is_valid:
                proof_badge = '<span class="nexai-tag tag-emerald" style="font-weight: 600; font-size: 0.75rem;">Priority Compliant</span>'
            else:
                proof_badge = '<span class="nexai-tag tag-rose" style="font-weight: 600; font-size: 0.75rem;">Out of Order</span>'

            # Tier badge styling
            if tier == "VIP Expedited":
                tier_badge = '<span class="nexai-tag tag-amber" style="font-weight: 600; font-size: 0.75rem;">VIP Expedited</span>'
            elif tier == "Priority High":
                tier_badge = '<span class="nexai-tag tag-blue" style="font-weight: 600; font-size: 0.75rem;">Priority High</span>'
            elif tier == "Low / Saver":
                tier_badge = '<span class="nexai-tag tag-rose" style="font-weight: 600; font-size: 0.75rem;">Low / Saver</span>'
            else:
                tier_badge = '<span class="nexai-tag tag-slate" style="font-size: 0.75rem;">Standard</span>'

            status_str = "Delivered" if cid in sim_state.completed_jobs else "Scheduled"
            status_badge = (
                '<span class="nexai-tag tag-emerald" style="font-size: 0.75rem;">Delivered</span>'
                if cid in sim_state.completed_jobs
                else '<span class="nexai-tag tag-slate" style="font-size: 0.75rem;">Scheduled</span>'
            )

            rows.append(
                '<tr>'
                f'<td style="text-align: center;">{van_badge}</td>'
                f'<td style="text-align: center;">{stop_badge}</td>'
                f'<td style="text-align: center; font-weight: 600;">#{cid}</td>'
                f'<td style="text-align: left; font-weight: 500;">{cust_name}</td>'
                f'<td style="text-align: center; color: #475569; font-size: 0.8rem;">{segment}</td>'
                f'<td style="text-align: center; font-weight: 700; color: #0284c7; font-size: 0.9rem;">{cur_score:.1f}</td>'
                f'<td style="text-align: center;">{tier_badge}</td>'
                f'<td style="text-align: center; font-family: monospace; color: #2563eb; font-weight: 500;">{arr_time_str}</td>'
                f'<td style="text-align: center;">{status_badge}</td>'
                f'<td style="text-align: center;">{proof_badge}</td>'
                '</tr>'
            )

    compliance_pct = (compliant_stops / total_stops * 100.0) if total_stops > 0 else 100.0
    rows_str = "".join(rows)

    summary_banner = (
        f'<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">'
        f'  <div style="font-size: 0.85rem; color: #166534; font-weight: 500;">'
        f'    <b>Priority Scheduling Audit:</b> {compliant_stops} of {total_stops} customer stops delivered in exact priority order sequence.'
        f'  </div>'
        f'  <span class="nexai-tag tag-emerald" style="font-size: 0.85rem; font-weight: 700; padding: 0.25rem 0.75rem;">'
        f'    {compliance_pct:.1f}% Compliant'
        f'  </span>'
        f'</div>'
    )

    return (
        f'{summary_banner}'
        '<div class="manifest-table-container" style="max-height: 480px; overflow-y: auto;">'
        '<table class="manifest-table">'
        '<thead><tr>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Assigned Van</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Delivery Sequence</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Customer ID</th>'
        '<th style="text-align: left; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Customer Name</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Segment</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">ML Priority Score</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Priority Tier</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Scheduled Arrival</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Status</th>'
        '<th style="text-align: center; position: sticky; top: 0; background: #f8fafc; z-index: 2;">Algorithm Proof</th>'
        '</tr></thead>'
        f'<tbody>{rows_str}</tbody>'
        '</table>'
        '</div>'
    )


def _compute_map_bounds(sim_state: SimulationState) -> Tuple[List[float], List[float]]:
    """
    Compute static, stable dynamic axis ranges [min, max] so the map fully fits the viewport
    without any jitter, edge cutoffs, or excessive whitespace.
    """
    all_x: List[float] = [float(sim_state.instance.depot.x)]
    all_y: List[float] = [float(sim_state.instance.depot.y)]

    # Customer nodes
    for c in sim_state.instance.customers:
        all_x.append(float(c.x))
        all_y.append(float(c.y))

    # Dynamic rush order nodes
    for cid, cdata in sim_state.customer_lookup.items():
        if cid >= 100:
            all_x.append(float(cdata.x))
            all_y.append(float(cdata.y))

    if not all_x:
        return [10.0, 50.0], [45.0, 92.0]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_span = max(x_max - x_min, 10.0)
    y_span = max(y_max - y_min, 10.0)

    # 7% horizontal margin gives breathing room on edges
    pad_x = x_span * 0.07

    # 7% bottom margin (depot label room) and 14% top margin (room for high stops like #5, #19 and route arcs)
    pad_y_bottom = y_span * 0.07
    pad_y_top = y_span * 0.14

    return [round(x_min - pad_x, 1), round(x_max + pad_x, 1)], [round(y_min - pad_y_bottom, 1), round(y_max + pad_y_top, 1)]


def plot_routes(sim_state: SimulationState) -> go.Figure:
    """
    Generate an interactive 2D spatial dispatch map with continuous animated kinematics.
    Fixed trace order, stable UIDs, and point IDs guarantee butter-smooth client-side transitions.
    """
    fig = go.Figure()
    depot = sim_state.instance.depot

    solution = sim_state.current_solution
    routes_df = (
        solution.routes
        if hasattr(solution, "routes") and isinstance(solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )

    grouped_routes = {}
    if not routes_df.empty:
        for vid, grp in routes_df.groupby("vehicle_id"):
            grouped_routes[int(vid)] = grp

    active_vids = set(grouped_routes.keys())
    active_vids.update(sim_state.broken_vehicles)
    max_active_id = max([5] + list(active_vids))
    total_fleet_count = min(len(sim_state.vehicles), max_active_id + 1)

    # --------------------------------------------------------------------------
    # 1. Central Depot (Fixed Trace 0, uid="depot_fixed")
    # --------------------------------------------------------------------------
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
            text=["DEPOT"],
            textposition="bottom center",
            textfont=dict(size=10, color="#09090b", family="Poppins, sans-serif"),
            name="Central Depot",
            uid="depot_fixed",
            hoverinfo="text",
            hovertext=[
                f"<b>CENTRAL DEPOT (Node 0)</b><br>"
                f"Coordinates: ({depot.x:.1f}, {depot.y:.1f})<br>"
                f"Operational Horizon: [{depot.ready_time}, {depot.due_date}]<br>"
                f"Fleet Base: {sim_state.instance.vehicle_count} Vehicles"
            ],
        )
    )

    # --------------------------------------------------------------------------
    # 2. Vehicle Route Edges (Fixed Traces 1..N, uid="route_line_v_id")
    # --------------------------------------------------------------------------
    for v_id in range(1, total_fleet_count + 1):
        color = ROUTE_COLORS[(v_id - 1) % len(ROUTE_COLORS)]
        group = grouped_routes.get(v_id)
        xs: List[float] = []
        ys: List[float] = []
        hover_lines: List[str] = []

        if group is not None and not group.empty:
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
                opacity=0.82,
                name=f"Route Van {v_id}",
                uid=f"route_line_{v_id}",
                hoverinfo="text",
                hovertext=hover_lines,
                showlegend=True if xs else False,
            )
        )

    # --------------------------------------------------------------------------
    # 3. Completed Customer Nodes (Fixed Trace, uid="comp_nodes_fixed")
    # --------------------------------------------------------------------------
    comp_x, comp_y, comp_hover, comp_labels = [], [], [], []
    for c_id, info in sim_state.completed_jobs.items():
        loc = info.get("location_index", 0)
        x, y = get_node_coords(sim_state, loc)
        comp_x.append(x)
        comp_y.append(y)
        comp_labels.append(f"#{c_id}")
        comp_hover.append(
            f"<b>Customer #{c_id} [Delivered]</b><br>"
            f"Location: ({x:.1f}, {y:.1f})<br>"
            f"Delivered by Van #{info.get('vehicle_id')}<br>"
            f"Demand: {info.get('demand', '-')} units<br>"
            f"Finished at T={unscale_value(info.get('completion_time', 0), sim_state.scale_factor):.1f}"
        )

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
            text=comp_labels,
            textposition="bottom center",
            textfont=dict(size=8, color="#64748b", family="Poppins, sans-serif"),
            name="Completed Orders",
            uid="comp_nodes_fixed",
            hoverinfo="text",
            hovertext=comp_hover,
        )
    )

    # --------------------------------------------------------------------------
    # 4. Pending Regular Orders (Fixed Trace, uid="pend_nodes_fixed")
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
            name="Pending Orders",
            uid="pend_nodes_fixed",
            hoverinfo="text",
            hovertext=pend_hover,
        )
    )

    # --------------------------------------------------------------------------
    # 5. Rush Orders (Fixed Trace, uid="rush_nodes_fixed")
    # --------------------------------------------------------------------------
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
            name="Rush Orders",
            uid="rush_nodes_fixed",
            hoverinfo="text",
            hovertext=rush_hover,
        )
    )

    # --------------------------------------------------------------------------
    # 6. Smooth Animated Vehicle Positions (INDIVIDUAL Fixed Traces per Vehicle)
    # --------------------------------------------------------------------------
    veh_positions = sim_state.get_interpolated_vehicle_positions()

    for v_id in range(1, total_fleet_count + 1):
        p_info = veh_positions.get(v_id, {
            "x": depot.x,
            "y": depot.y,
            "is_broken": False,
            "is_active": False,
            "status": "At Depot",
        })
        is_broken = p_info.get("is_broken", False)
        is_active = p_info.get("is_active", False)
        base_color = ROUTE_COLORS[(v_id - 1) % len(ROUTE_COLORS)]
        van_color = "#ef4444" if is_broken else base_color
        symbol = "x" if is_broken else "triangle-up"
        size = 17 if is_broken else 14
        status_str = "OFFLINE" if is_broken else p_info.get("status", "Active")

        # Hide idle reserve fleet markers from piling on top of the Depot icon
        is_idle_at_depot = (
            abs(p_info["x"] - depot.x) < 0.1
            and abs(p_info["y"] - depot.y) < 0.1
            and not is_active
            and not is_broken
        )
        van_text = None if is_idle_at_depot else ([f"Van {v_id} [OFF]" if is_broken else f"Van {v_id}"])

        fig.add_trace(
            go.Scatter(
                x=[p_info["x"]],
                y=[p_info["y"]],
                ids=[f"van_{v_id}"],
                mode="markers+text" if van_text else "markers",
                marker=dict(
                    symbol=symbol,
                    size=0 if is_idle_at_depot else size,
                    color=van_color,
                    line=dict(color="#ffffff", width=2.0),
                    opacity=0.0 if is_idle_at_depot else 1.0,
                ),
                text=van_text,
                textposition="top center" if is_broken else "bottom right",
                textfont=dict(size=9, color=van_color, family="Poppins, sans-serif"),
                name=f"Van {v_id}",
                uid=f"van_marker_{v_id}",
                hoverinfo="text" if not is_idle_at_depot else "none",
                hovertext=[
                    f"<b>Van #{v_id} [{'Disabled' if is_broken else 'Active'}]</b><br>"
                    f"Coordinates: ({p_info['x']:.1f}, {p_info['y']:.1f})<br>"
                    f"Status: {status_str}<br>"
                    f"Waypoint: Stop #{p_info.get('waypoint_index', 0)}<br>"
                    f"Distance to Next Stop: {p_info.get('distance_to_next_waypoint', 0.0):.1f} units<br>"
                    f"Speed: {p_info.get('speed', 1.0):.1f} units/s<br>"
                    f"Route Progress: {p_info.get('route_progress', 0.0):.1f}%<br>"
                    f"Remaining Capacity: {sim_state.vehicles[v_id].remaining_capacity} units"
                ] if not is_idle_at_depot else None,
                showlegend=False,
            )
        )

    # --------------------------------------------------------------------------
    # 7. Fixed Viewport Layout with Continuous Interpolated Transitions
    # --------------------------------------------------------------------------
    # Compute dynamic axis bounds from all visible coordinates
    x_range, y_range = _compute_map_bounds(sim_state)

    fig.update_layout(
        uirevision="continuous_dispatch_view",
        transition=dict(
            duration=230,
            easing="cubic-in-out",
            ordering="traces first",
        ),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fcfcfd",
        margin=dict(l=25, r=25, t=35, b=25),
        font=dict(family="Poppins, sans-serif", color="#64748b", size=10),
        xaxis=dict(
            title=dict(text="X Coordinate", font=dict(family="Poppins, sans-serif", size=11, color="#64748b")),
            range=x_range,
            autorange=False,
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(family="Poppins, sans-serif", color="#64748b", size=9),
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Y Coordinate", font=dict(family="Poppins, sans-serif", size=11, color="#64748b")),
            range=y_range,
            autorange=False,
            gridcolor="#f1f5f9",
            zerolinecolor="#e2e8f0",
            tickfont=dict(family="Poppins, sans-serif", color="#64748b", size=9),
            showgrid=True,
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
        height=540,
    )

    return fig


def build_leaflet_payload(sim_state: SimulationState) -> Dict[str, Any]:
    """
    Construct serializable geographic payload for Leaflet OpenStreetMap component.
    Maps Solomon Cartesian nodes to real-world WGS-84 coordinates and fetches road geometry.
    """
    cur_geo = st.session_state.get("geo_location", {})
    if cur_geo:
        set_active_anchor(
            lat=cur_geo["lat"],
            lon=cur_geo["lon"],
            zone_name=cur_geo["zone_name"],
            address=cur_geo["address"],
        )
    anchor = get_active_anchor()

    depot_cart = sim_state.instance.depot
    depot_lat, depot_lon = xy_to_latlon(depot_cart.x, depot_cart.y)

    depot_data = {
        "lat": depot_lat,
        "lon": depot_lon,
        "address": anchor.get("address", "Central Logistics Hub"),
        "ready_time": int(depot_cart.ready_time),
        "due_date": int(depot_cart.due_date),
        "vehicle_count": int(sim_state.instance.vehicle_count),
    }

    # Group current solution routes
    solution = sim_state.current_solution
    routes_df = (
        solution.routes
        if hasattr(solution, "routes") and isinstance(solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )

    grouped_routes = {}
    if not routes_df.empty:
        for vid, grp in routes_df.groupby("vehicle_id"):
            grouped_routes[int(vid)] = grp

    active_vids = set(grouped_routes.keys())
    active_vids.update(sim_state.broken_vehicles)
    total_fleet_count = len(sim_state.vehicles)

    routes_payload = []
    legs_by_vid = {}

    for v_id in range(1, total_fleet_count + 1):
        group = grouped_routes.get(v_id)
        waypoints_latlon = []
        assigned_jobs = 0
        if group is not None and not group.empty:
            for _, step in group.iterrows():
                loc_idx = int(step["location_index"])
                x, y = get_node_coords(sim_state, loc_idx)
                lat, lon = xy_to_latlon(x, y)
                waypoints_latlon.append((lat, lon))
                if step.get("type") == "job":
                    assigned_jobs += 1

        if waypoints_latlon:
            full_coords, legs = RoadRoutingService.get_route_legs_and_polyline(waypoints_latlon)
            legs_by_vid[v_id] = legs
            routes_payload.append({
                "vehicle_id": v_id,
                "road_coordinates": full_coords,
                "provider": "OSRM OpenStreetMap Road Network",
                "assigned_packages": assigned_jobs,
                "is_broken": (v_id in sim_state.broken_vehicles),
            })

    # Vehicles authoritative positions strictly computed along road polylines
    veh_positions = sim_state.get_interpolated_vehicle_positions()
    vehicles_payload = []
    for v_id in range(1, total_fleet_count + 1):
        p_info = veh_positions.get(v_id, {
            "x": float(depot_cart.x),
            "y": float(depot_cart.y),
            "is_broken": False,
            "is_active": False,
            "status": "At Depot",
            "speed": 1.0,
            "route_progress": 0.0,
            "waypoint_index": 0,
            "leg_alpha": 0.0,
        })
        is_broken = bool(p_info.get("is_broken", False))
        is_active = bool(p_info.get("is_active", False))
        status_str = str(p_info.get("status", "Active"))

        # Exact road path position calculation
        heading = 0.0
        if is_broken:
            v_lat, v_lon = xy_to_latlon(p_info["x"], p_info["y"], snap_to_road=False)
        elif not is_active or "At Depot" in status_str:
            v_lat, v_lon = (depot_lat, depot_lon)
        else:
            legs = legs_by_vid.get(v_id, [])
            leg_idx = int(p_info.get("waypoint_index", 0))
            alpha = float(p_info.get("leg_alpha", 0.0))
            if legs and 0 <= leg_idx < len(legs):
                curr_leg = legs[leg_idx]
                v_lat, v_lon, heading = interpolate_along_road_coordinates(curr_leg, alpha)
            else:
                v_lat, v_lon = xy_to_latlon(p_info["x"], p_info["y"], snap_to_road=False)

        rem_cap = (
            int(sim_state.vehicles[v_id].remaining_capacity)
            if v_id in sim_state.vehicles
            else 0
        )
        vehicles_payload.append({
            "vehicle_id": v_id,
            "lat": v_lat,
            "lon": v_lon,
            "heading": heading,
            "is_broken": is_broken,
            "is_active": is_active,
            "status": str("OFFLINE" if is_broken else status_str),
            "waypoint_index": int(p_info.get("waypoint_index", 0)),
            "distance_to_next_waypoint": float(p_info.get("distance_to_next_waypoint", 0.0)),
            "speed": float(p_info.get("speed", 1.0)),
            "route_progress": float(p_info.get("route_progress", 0.0)),
            "remaining_capacity": rem_cap,
        })

    # Completed orders
    completed_payload = []
    scoring_engine = CustomerScoringEngine.get_instance()
    for c_id, info in sim_state.completed_jobs.items():
        loc = info.get("location_index", 0)
        x, y = get_node_coords(sim_state, loc)
        c_lat, c_lon = xy_to_latlon(x, y)
        prof = scoring_engine.get_profile(int(c_id))
        completed_payload.append({
            "id": int(c_id),
            "lat": c_lat,
            "lon": c_lon,
            "vehicle_id": int(info.get("vehicle_id", 0)),
            "demand": info.get("demand", "-"),
            "priority_score": float(prof.ml_priority_score),
            "priority_tier": prof.priority_tier,
            "completion_time": float(unscale_value(info.get("completion_time", 0), sim_state.scale_factor)),
        })

    # Pending & Rush orders
    pending_payload = []
    rush_payload = []
    for job_id, job in sim_state.pending_jobs.items():
        loc_idx = resolve_location_index(job.location)
        x, y = get_node_coords(sim_state, loc_idx)
        p_lat, p_lon = xy_to_latlon(x, y)
        meta = sim_state.customer_lookup.get(job_id)
        demand = meta.demand if meta else (job.delivery[0] if job.delivery else "-")
        ready_t = meta.ready_time if meta else 0
        due_t = meta.due_date if meta else 0
        prof = scoring_engine.get_profile(int(job_id))

        ord_obj = {
            "id": int(job_id),
            "lat": p_lat,
            "lon": p_lon,
            "demand": demand,
            "ready_time": ready_t,
            "due_date": due_t,
            "priority_score": float(prof.ml_priority_score),
            "priority_tier": prof.priority_tier,
        }
        if job_id >= 100:
            rush_payload.append(ord_obj)
        else:
            pending_payload.append(ord_obj)

    route_version = f"sol_{len(sim_state.replanning_history)}_{len(sim_state.broken_vehicles)}_{len(sim_state.completed_jobs)}"

    return {
        "depot": depot_data,
        "routes": routes_payload,
        "vehicles": vehicles_payload,
        "completed_orders": completed_payload,
        "pending_orders": pending_payload,
        "rush_orders": rush_payload,
        "center": [depot_lat, depot_lon],
        "zoom": 13,
        "sim_time": float(st.session_state.current_time),
        "route_version": route_version,
        "geo_metadata": get_geographic_metadata(),
    }


# ==============================================================================
# 4. KPI Placeholder Renderer (Shadcn Light Metric Style)
# ==============================================================================
def render_kpi_metrics(placeholder: Any, sim_state: SimulationState) -> None:
    """Render high-contrast metric cards into the dedicated empty placeholder."""
    van_count = st.session_state.get("custom_van_count", len(sim_state.vehicles))
    calc_dist = calculate_custom_fleet_distance(van_count)

    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    active_in_transit = len(routes_df["vehicle_id"].unique()) if not routes_df.empty else 0
    total_active_fleet = len(sim_state.vehicles) - len(sim_state.broken_vehicles)

    pending_cnt = len(sim_state.pending_jobs)
    comp_cnt = len(sim_state.completed_jobs)

    latest_event = (
        sim_state.replanning_history[-1]
        if sim_state.replanning_history
        else {}
    )
    latest_runtime = latest_event.get("runtime_ms", 0.0)
    latest_desc = latest_event.get("description", "Dispatch Plan")

    is_done = sim_state.is_simulation_completed()
    clock_status = "Complete" if is_done else ("Simulating" if st.session_state.is_simulating else "Paused")

    with placeholder.container():
        sla_report = SLARiskEngine.get_instance().evaluate_fleet_state(sim_state)
        avg_sla_on_time = sla_report.get("avg_on_time_pct", 100.0)
        crit_sla_cnt = sla_report.get("critical_count", 0)
        low_sla_cnt = sla_report.get("low_count", 0)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            label="Total Travel Distance",
            value=f"{calc_dist:.1f} km",
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
            label="Time to Calculate Change",
            value=f"{latest_runtime:.2f} ms",
            delta=f"{latest_desc[:24]}",
            delta_color="off",
        )
        m5.metric(
            label="AI SLA On-Time Health",
            value=f"{avg_sla_on_time:.1f}%",
            delta=f"{low_sla_cnt} Safe · {crit_sla_cnt} Critical" if crit_sla_cnt > 0 else f"{low_sla_cnt} Safe (GBM ML)",
            delta_color="normal" if crit_sla_cnt == 0 else "inverse",
        )


# ==============================================================================
# ==============================================================================
# 5. Sidebar: Setup, Simulation & Disruption Controls
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 0.85rem; padding-bottom: 0.65rem; border-bottom: 1px solid #e4e4e7;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #09090b; letter-spacing: -0.02em; display: flex; align-items: center; gap: 8px;">
                <span style="background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; width: 28px; height: 28px; border-radius: 6px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 800;">NX</span>
                NexAI 2026
            </div>
            <div style="font-size: 0.72rem; color: #71717a; margin-top: 3px; font-weight: 500;">
                Autonomous Fleet & Route Engine
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title" style="font-size: 0.95rem; font-weight: 600; margin-bottom: 0.35rem;">Simulation & Disruption Controls</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 1px; background-color: #e4e4e7; margin: 0.25rem 0 0.5rem 0;"></div>', unsafe_allow_html=True)

    # Location Status
    geo_info = st.session_state.get("geo_location", {})
    city_display = geo_info.get("zone_name", "Vellore, Tamil Nadu, India")
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e4e4e7; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 0.75rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="font-size: 0.65rem; color: #71717a; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">Operational Location</div>
            <div style="font-size: 0.82rem; font-weight: 600; color: #09090b; display: flex; align-items: center; gap: 6px; margin-top: 3px;">
                <span style="width: 8px; height: 8px; display: inline-block; background-color: #10b981; border-radius: 50%;"></span> {city_display}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Fleet Size ---
    st.markdown('<div class="control-label" style="font-size: 0.68rem; color: #71717a; text-transform: uppercase; margin-bottom: 0.2rem;">Fleet Size</div>', unsafe_allow_html=True)
    current_van_cnt = st.session_state.get("custom_van_count", len(sim.vehicles))
    selected_vans = st.slider(
        "Active Vans in Fleet",
        min_value=1,
        max_value=8,
        value=min(8, max(1, current_van_cnt)),
        step=1,
        key="slider_custom_vans",
        help="Select custom number of vans in fleet."
    )
    if selected_vans != current_van_cnt:
        st.session_state.custom_van_count = selected_vans
        init_simulation_state(custom_vehicle_count=selected_vans)
        st.rerun()

    # --- 1. Autonomous Simulation Controls ---
    st.markdown('<div class="control-label" style="font-size: 0.68rem; color: #71717a; text-transform: uppercase; margin-bottom: 0.2rem;">Autonomous Simulation</div>', unsafe_allow_html=True)
    is_running = st.session_state.get("is_simulating", False)
    play_label = "Resume" if st.session_state.current_time > 0 else "Start"

    st.markdown('<div class="btn-ctrl-play"></div>', unsafe_allow_html=True)
    if st.button(
        play_label,
        width="stretch",
        type="primary" if not is_running else "secondary",
        disabled=is_running,
        key="btn_play",
    ):
        st.session_state.is_simulating = True
        st.rerun()

    st.markdown('<div class="btn-ctrl-pause"></div>', unsafe_allow_html=True)
    if st.button(
        "Pause",
        width="stretch",
        type="primary" if is_running else "secondary",
        disabled=not is_running,
        key="btn_pause",
    ):
        st.session_state.is_simulating = False
        st.rerun()



    st.session_state.map_expanded = False

    # --- 2. Mechanical Disruption ---
    st.markdown('<div class="control-label" style="font-size: 0.68rem; color: #71717a; text-transform: uppercase; margin-top: 0.45rem; margin-bottom: 0.2rem;">Mechanical Disruption</div>', unsafe_allow_html=True)
    active_broken = sim.broken_vehicles
    candidate_vans = [vid for vid in sim.vehicles if vid not in active_broken]
    target_van = min(candidate_vans) if candidate_vans else None
    van_disabled = target_van is None
    btn_label = f"Breakdown (Van {target_van})" if target_van else "All Vans Inactive"
    st.markdown('<div class="btn-ctrl-breakdown"></div>', unsafe_allow_html=True)

    if st.button(btn_label, width="stretch", disabled=van_disabled, key="btn_breakdown"):
        res = sim.trigger_breakdown(target_van)
        if res.get("status") == "already_completed":
            st.toast(res.get("message"))
        else:
            sim.replan(exploration_level=1, reason=f"Mechanical Breakdown: Van {target_van} Disabled")
            rt = sim.replanning_history[-1]["runtime_ms"]
            st.toast(f"Van {target_van} breakdown logged. Fleet re-routed in {rt:.2f} ms.")
        st.rerun()

    # --- 3. Dynamic Customer Demand ---
    st.markdown('<div class="control-label" style="font-size: 0.68rem; color: #71717a; text-transform: uppercase; margin-top: 0.45rem; margin-bottom: 0.2rem;">Dynamic Customer Demand</div>', unsafe_allow_html=True)
    if st.button("Custom Customer Order Intake", width="stretch", key="btn_open_dispatcher"):
        st.session_state.show_order_dispatcher = True
        st.session_state.is_simulating = False
        st.rerun()

    st.markdown('<div class="btn-ctrl-rush"></div>', unsafe_allow_html=True)
    if st.button("Inject Rush Orders (+3)", width="stretch", key="btn_rush"):
        st.session_state.rush_order_count += 1
        offset = st.session_state.rush_order_count * 10
        cur_t = float(st.session_state.current_time)
        serv = 3.0

        rush_orders = [
            {
                "cust_no": 100 + offset + 1,
                "x": 38.0 + ((offset * 7) % 35) - 15,
                "y": 55.0 + ((offset * 11) % 30) - 15,
                "demand": 15,
                "ready_time": cur_t,
                "due_date": cur_t + serv + 450.0,
                "service_time": serv,
            },
            {
                "cust_no": 100 + offset + 2,
                "x": 28.0 + ((offset * 5) % 40) - 15,
                "y": 70.0 - ((offset * 9) % 35) + 10,
                "demand": 20,
                "ready_time": cur_t,
                "due_date": cur_t + serv + 500.0,
                "service_time": serv,
            },
            {
                "cust_no": 100 + offset + 3,
                "x": 48.0 - ((offset * 13) % 30) + 15,
                "y": 45.0 + ((offset * 7) % 25) - 10,
                "demand": 10,
                "ready_time": cur_t,
                "due_date": cur_t + serv + 550.0,
                "service_time": serv,
            },
        ]

        try:
            injected = sim.inject_new_orders(rush_orders)
            sim.replan(exploration_level=1, reason=f"Demand Surge: +{len(injected)} Rush Orders Scheduled")
            rt = sim.replanning_history[-1]["runtime_ms"]
            st.toast(f"Injected {len(injected)} rush orders. Fleet re-routed in {rt:.2f} ms.")
        except Exception as e:
            st.error(f"Failed to schedule rush orders: {e}")
        st.rerun()

    # --- 4. System Reset ---
    st.markdown('<div class="control-label" style="font-size: 0.68rem; color: #71717a; text-transform: uppercase; margin-top: 0.45rem; margin-bottom: 0.2rem;">System Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="btn-ctrl-reset"></div>', unsafe_allow_html=True)
    if st.button("Reset Simulation", width="stretch", key="btn_reset"):
        st.session_state.clear()
        CustomerScoringEngine._instance = None
        init_simulation_state()
        st.rerun()


def render_add_customer_order_section(sim_state: SimulationState) -> None:
    """
    Interactive customer order intake and live ML route re-optimization panel.
    Allows configuring custom customer profiles (order value, past miss rate, segment, coordinates),
    scores priority in real-time with GradientBoostingRegressor, and dynamically remaps fleet routes.
    """
    engine = CustomerScoringEngine.get_instance()

    # Check for recent replan notification banner
    notice = st.session_state.get("recent_replan_notice")
    if notice:
        st.markdown(
            '<div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 8px; padding: 0.75rem 1.1rem; margin-bottom: 0.85rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">'
            '<div style="display: flex; align-items: center; gap: 0.65rem;">'
            '<span style="background: #16a34a; color: #ffffff; font-weight: 700; font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; text-transform: uppercase;">Route Remapped</span>'
            f'<span style="font-size: 0.84rem; color: #166534; font-weight: 600;">Customer #{notice["id"]} ({notice["name"]}) · Segment: {notice["segment"]} · ML Priority: {notice["score"]:.1f} → <strong>{notice["van_info"]}</strong> (Re-routed in {notice["runtime_ms"]:.2f} ms)</span>'
            '</div>'
            '<span style="font-size: 0.75rem; color: #15803d; font-weight: 500;">Live Road Polyline Updated</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    is_open = st.session_state.get("show_order_dispatcher", False)

    with st.expander("Dispatch New Customer Order", expanded=is_open):
        preset = st.radio(
            "Select Scenario Preset (1-Click Fill) or Custom:",
            [
                "Custom Order",
                "VIP Expedited (₹15,000 High-Value)",
                "New Customer (₹5,200 Retention Target)",
                "Chronic Misser (₹1,400 with 45% Miss Rate Penalty)",
                "Loyal Regular (₹6,500 Safe)",
            ],
            horizontal=True,
            key="preset_choice",
        )

        if preset == "VIP Expedited (₹15,000 High-Value)":
            def_name = "Zenith Luxury Mart"
            def_is_new = False
            def_order_val = 15000.0
            def_miss_rate = 0.0
            def_demand = 15
        elif preset == "New Customer (₹5,200 Retention Target)":
            def_name = "Aditi Verma (1st Purchase)"
            def_is_new = True
            def_order_val = 5200.0
            def_miss_rate = 0.0
            def_demand = 10
        elif preset == "Chronic Misser (₹1,400 with 45% Miss Rate Penalty)":
            def_name = "Metro Wholesale Bargain"
            def_is_new = False
            def_order_val = 1400.0
            def_miss_rate = 45.0
            def_demand = 30
        elif preset == "Loyal Regular (₹6,500 Safe)":
            def_name = "Bluefin Tech Campus"
            def_is_new = False
            def_order_val = 6500.0
            def_miss_rate = 3.0
            def_demand = 20
        else:
            def_name = "Express Client Order"
            def_is_new = True
            def_order_val = 4500.0
            def_miss_rate = 0.0
            def_demand = 15

        c1, c2, c3 = st.columns([1.3, 1.1, 1.1])

        with c1:
            cust_name = st.text_input("Customer / Business Name", value=def_name, key="input_cust_name")
            is_new = st.checkbox("Is First-Time Customer? (Retention Target)", value=def_is_new, key="input_is_new")

        with c2:
            order_val = st.number_input("Order Value (₹)", min_value=200.0, max_value=50000.0, value=float(def_order_val), step=250.0, key="input_order_val")
            miss_rate = st.slider("Past Delivery Miss Rate (%)", min_value=0.0, max_value=70.0, value=float(def_miss_rate), step=1.0, key="input_miss_rate")

        with c3:
            demand = st.slider("Package Demand (kg)", min_value=5, max_value=50, value=int(def_demand), step=5, key="input_demand")

        # Automatically determine Segment Classification
        if is_new:
            segment = "New Customer"
        elif order_val >= 9000.0:
            segment = "High-Value VIP"
        elif miss_rate >= 30.0:
            segment = "Chronic Misser"
        elif order_val >= 4500.0:
            segment = "Loyal Regular"
        else:
            segment = "Standard"

        # Automatically determine Customer Lifetime Value (CLV)
        if segment == "High-Value VIP":
            clv = 9.5
        elif segment == "New Customer":
            clv = 5.0
        elif segment == "Chronic Misser":
            clv = 2.0
        elif segment == "Loyal Regular":
            clv = 8.0
        else:
            clv = min(9.0, max(2.0, (order_val / 1200.0)))

        # Automatically determine Delivery Location (distributed across road grid)
        cust_idx = len(sim_state.customer_lookup)
        angle = (cust_idx * 137.5) * (math.pi / 180.0)
        radius = 12.0 + ((cust_idx * 7) % 22)
        loc_x = round(max(15.0, min(65.0, 40.0 + radius * math.cos(angle))), 1)
        loc_y = round(max(20.0, min(75.0, 50.0 + radius * math.sin(angle))), 1)

        # Automatically determine Delivery SLA Window
        if segment in ("High-Value VIP", "New Customer"):
            due_horizon = 350.0
            urgency_feature = 0.90
        elif segment == "Chronic Misser":
            due_horizon = 900.0
            urgency_feature = 0.25
        else:
            due_horizon = 550.0
            urgency_feature = 0.60

        # Real-time ML Evaluation
        x_vec = np.array([[
            float(1 if is_new else 0),
            float(order_val),
            float(miss_rate / 100.0),
            float(clv),
            float(demand),
            float(urgency_feature),
        ]], dtype=np.float32)

        if engine.model is not None:
            preview_score = float(engine.model.predict(x_vec)[0])
        else:
            preview_score = 50.0 + (30.0 if is_new else 0.0) + (order_val / 400.0) - (miss_rate * 0.5)
        preview_score = max(5.0, min(99.0, preview_score))

        if preview_score >= 80.0:
            tier_badge = '<span class="nexai-tag tag-violet" style="font-weight: 600;">VIP Expedited</span>'
            score_color = "#7c3aed"
            tier_desc = "Top dispatch priority. Assigned to lead active van on earliest stop sequence."
        elif preview_score >= 65.0:
            tier_badge = '<span class="nexai-tag tag-blue" style="font-weight: 600;">Priority High</span>'
            score_color = "#2563eb"
            tier_desc = "High priority SLA. Preferred routing with guaranteed tight delivery window."
        elif preview_score >= 42.0:
            tier_badge = '<span class="nexai-tag tag-slate" style="font-weight: 500;">Standard</span>'
            score_color = "#475569"
            tier_desc = "Standard dispatch route allocation balanced with cargo capacity."
        else:
            tier_badge = '<span class="nexai-tag tag-amber" style="font-weight: 600;">Low / Saver</span>'
            score_color = "#d97706"
            tier_desc = "Reduced priority penalty due to elevated order rejection history."

        st.markdown(
            '<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.85rem 1.1rem; margin: 0.75rem 0 0.9rem 0;">'
            '<div style="font-size: 0.72rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Live ML Model Evaluation</div>'
            f'<div style="font-size: 1.15rem; font-weight: 800; color: {score_color}; display: flex; align-items: center; gap: 0.5rem; margin-top: 2px;">'
            f'<span>Predicted ML Priority: {preview_score:.1f} / 100</span>'
            f'{tier_badge}'
            '</div>'
            f'<div style="font-size: 0.76rem; color: #64748b; margin-top: 2px;">{tier_desc}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("Add Customer", width="stretch", type="primary", key="btn_dispatch_order"):
            existing_ids = set(sim_state.customer_lookup.keys()) | set(sim_state.pending_jobs.keys()) | set(sim_state.completed_jobs.keys()) | set(engine.customer_profiles.keys())
            next_cid = max(existing_ids) + 1 if existing_ids else 1

            prof = engine.score_customer(
                cust_no=next_cid,
                customer_name=cust_name.strip() or f"Customer #{next_cid}",
                segment=segment,
                is_new_customer=1 if is_new else 0,
                order_value_inr=float(order_val),
                past_miss_rate=float(miss_rate) / 100.0,
                clv_score=float(clv),
                package_demand=int(demand),
                ready_time=float(sim_state.current_time),
                due_date=float(sim_state.current_time + due_horizon),
                service_time=3.0,
            )

            order_dict = {
                "cust_no": next_cid,
                "customer_name": prof.customer_name,
                "segment": prof.segment,
                "is_new_customer": prof.is_new_customer,
                "order_value_inr": prof.order_value_inr,
                "past_miss_rate": prof.past_miss_rate,
                "clv_score": prof.clv_score,
                "demand": prof.package_demand,
                "x": float(loc_x),
                "y": float(loc_y),
                "ready_time": float(sim_state.current_time),
                "due_date": float(sim_state.current_time + due_horizon),
                "service_time": 3.0,
            }

            try:
                injected = sim_state.inject_new_orders([order_dict])
                sim_state.replan(exploration_level=2, reason=f"New Order #{next_cid} ({prof.customer_name}) Scheduled: ML Priority {prof.ml_priority_score:.1f}")
                rt_ms = sim_state.replanning_history[-1]["runtime_ms"]

                assigned_van = None
                assigned_stop = None
                if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame):
                    r_df = sim_state.current_solution.routes
                    j_rows = r_df[r_df["id"] == next_cid]
                    if not j_rows.empty:
                        assigned_van = int(j_rows.iloc[0]["vehicle_id"])
                        v_group = r_df[(r_df["vehicle_id"] == assigned_van) & (r_df["type"] == "job")]
                        for seq_i, (_, step) in enumerate(v_group.iterrows(), start=1):
                            if step.get("id") == next_cid:
                                assigned_stop = seq_i
                                break

                van_info = f"Van {assigned_van} · Stop #{assigned_stop}" if (assigned_van and assigned_stop) else (f"Van {assigned_van}" if assigned_van else "Pending Dispatch")

                st.session_state["recent_replan_notice"] = {
                    "id": next_cid,
                    "name": prof.customer_name,
                    "segment": prof.segment,
                    "score": prof.ml_priority_score,
                    "van_info": van_info,
                    "runtime_ms": rt_ms,
                }
                st.session_state["show_order_dispatcher"] = False
                st.toast(f"Customer #{next_cid} added! ML Priority: {prof.ml_priority_score:.1f} → Assigned to {van_info} (re-planned in {rt_ms:.1f}ms)")
            except Exception as e:
                st.error(f"Failed to inject customer order: {e}")
            st.rerun()


# ==============================================================================
# 7. Live Display Fragment (Smooth In-Place Updates Without Page Reload)
# ==============================================================================
# Dynamic fragment interval: configurable via SIMULATION_REFRESH_INTERVAL_SEC (300ms)
sim_interval = SIMULATION_REFRESH_INTERVAL_SEC if st.session_state.get("is_simulating", False) else None

@st.fragment(run_every=sim_interval)
def render_live_simulation_board():
    sim_state: SimulationState = st.session_state.sim

    # Central simulation update with continuous smooth motion
    is_sim = st.session_state.get("is_simulating", False)
    effective_timestep = BASE_SIMULATION_TIMESTEP_SEC

    if is_sim:
        st.session_state.current_time += effective_timestep
        sim_state.advance_clock(st.session_state.current_time)

    # --- A. Live Status & Progress Bar ---
    depot_due = float(sim_state.instance.depot.due_date)
    pct = min(100.0, (st.session_state.current_time / depot_due) * 100.0) if depot_due > 0 else 0.0
    all_current_done = sim_state.is_simulation_completed()

    if is_sim and all_current_done:
        status_tag = '<span class="nexai-tag tag-emerald"><span class="pulse-dot"></span> LIVE (FLEET AT DEPOT / READY)</span>'
    elif is_sim:
        status_tag = '<span class="nexai-tag tag-emerald"><span class="pulse-dot"></span> LIVE SIMULATION</span>'
    elif all_current_done:
        status_tag = '<span class="nexai-tag tag-blue">READY FOR DISPATCH</span>'
    else:
        status_tag = '<span class="nexai-tag tag-amber"><span class="pulse-dot-amber"></span> PAUSED</span>'

    time_horizon_str = f"(Continuous / Shift 2)" if st.session_state.current_time > depot_due else f"/ {depot_due:.0f}s ({pct:.1f}%)"

    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e4e4e7; border-radius: 8px; padding: 0.75rem 1.1rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.65rem;">
                    <span style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #71717a;">Simulation Clock</span>
                    {status_tag}
                </div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #09090b;">
                    T = <span style="font-size: 1.1rem; color: #2563eb; font-weight: 700;">{st.session_state.current_time:.1f}s</span> {time_horizon_str}
                </div>
            </div>
            <div style="height: 6px; width: 100%; background: #f1f5f9; border-radius: 9999px; overflow: hidden;">
                <div style="height: 100%; width: {min(100.0, pct)}%; background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%); transition: width 0.25s linear; border-radius: 9999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- B. KPI Metrics Cards ---
    van_count = st.session_state.get("custom_van_count", len(sim_state.vehicles))
    calc_dist = calculate_custom_fleet_distance(van_count)

    routes_df = (
        sim_state.current_solution.routes
        if hasattr(sim_state.current_solution, "routes") and isinstance(sim_state.current_solution.routes, pd.DataFrame)
        else pd.DataFrame()
    )
    active_in_transit = len(routes_df["vehicle_id"].unique()) if not routes_df.empty else 0
    total_active_fleet = len(sim_state.vehicles) - len(sim_state.broken_vehicles)
    pending_cnt = len(sim_state.pending_jobs)
    comp_cnt = len(sim_state.completed_jobs)
    latest_event = (
        sim_state.replanning_history[-1]
        if sim_state.replanning_history
        else {}
    )
    latest_runtime = latest_event.get("runtime_ms", 0.0)
    latest_desc = latest_event.get("description", "Dispatch Plan")

    # Real-time ML SLA inference
    sla_report = SLARiskEngine.get_instance().evaluate_fleet_state(sim_state)
    avg_sla_on_time = sla_report.get("avg_on_time_pct", 100.0)
    crit_sla_cnt = sla_report.get("critical_count", 0)
    low_sla_cnt = sla_report.get("low_count", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(
        label="Total Travel Distance",
        value=f"{calc_dist:.1f} km",
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
        label="Time to Calculate Change",
        value=f"{latest_runtime:.2f} ms",
        delta=f"{latest_desc[:24]}",
        delta_color="off",
    )
    m5.metric(
        label="AI SLA On-Time Health",
        value=f"{avg_sla_on_time:.1f}%",
        delta=f"{low_sla_cnt} Safe · {crit_sla_cnt} Critical" if crit_sla_cnt > 0 else f"{low_sla_cnt} Safe (GBM ML)",
        delta_color="normal" if crit_sla_cnt == 0 else "inverse",
    )

    # --- C. Customer Order Intake & Dynamic Route Remapping ---
    render_add_customer_order_section(sim_state)

    # Customer Profiles & ML SLA Priority Scores
    st.markdown(render_customer_priority_scoring_table_html(sim_state), unsafe_allow_html=True)

    # --- D. Real-World Leaflet + OpenStreetMap Fleet Road Tracker ---
    leaflet_data = build_leaflet_payload(sim_state)
    map_height = 880 if st.session_state.get("map_expanded", False) else 560
    map_action = leaflet_fleet_map(
        depot=leaflet_data["depot"],
        routes=leaflet_data["routes"],
        vehicles=leaflet_data["vehicles"],
        completed_orders=leaflet_data["completed_orders"],
        pending_orders=leaflet_data["pending_orders"],
        rush_orders=leaflet_data["rush_orders"],
        center=leaflet_data["center"],
        zoom=leaflet_data["zoom"],
        sim_time=leaflet_data["sim_time"],
        route_version=leaflet_data["route_version"],
        height=map_height,
        theme="light",
        geo_metadata=leaflet_data["geo_metadata"],
        is_simulating=bool(st.session_state.get("is_simulating", False)),
        is_completed=bool(sim_state.is_simulation_completed()),
        is_expanded=bool(st.session_state.get("map_expanded", False)),
        broken_vehicles=list(sim_state.broken_vehicles),
        key="leaflet_fleet_map_instance",
    )

    # Dispatch interactive actions triggered from Fullscreen overlays
    if map_action and isinstance(map_action, dict):
        act = map_action.get("action")
        act_ts = map_action.get("timestamp")
        last_ts = st.session_state.get("_last_map_action_ts")
        if act and act_ts and act_ts != last_ts:
            st.session_state["_last_map_action_ts"] = act_ts
            if act == "resume":
                st.session_state.is_simulating = True
                st.rerun()
            elif act == "pause":
                st.session_state.is_simulating = False
                st.rerun()
            elif act == "breakdown":
                v_target = map_action.get("vehicle_id")
                if v_target is None:
                    active_broken = sim_state.broken_vehicles
                    candidate_vans = [vid for vid in sim_state.vehicles if vid not in active_broken]
                    v_target = min(candidate_vans) if candidate_vans else None
                if v_target is not None:
                    v_target = int(v_target)
                    res = sim_state.trigger_breakdown(v_target)
                    if res.get("status") == "already_completed":
                        st.toast(res.get("message"))
                    else:
                        sim_state.replan(exploration_level=1, reason=f"Mechanical Breakdown: Van {v_target} Disabled")
                        rt = sim_state.replanning_history[-1]["runtime_ms"]
                        st.toast(f"Van {v_target} breakdown logged. Fleet re-routed in {rt:.2f} ms.")
                st.rerun()
            elif act == "add_rush_orders":
                st.session_state.rush_order_count += 1
                offset = st.session_state.rush_order_count * 10
                cur_t = float(st.session_state.current_time)
                serv = 3.0
                rush_orders = [
                    {
                        "cust_no": 100 + offset + 1,
                        "x": 38.0 + ((offset * 7) % 35) - 15,
                        "y": 55.0 + ((offset * 11) % 30) - 15,
                        "demand": 15,
                        "ready_time": cur_t,
                        "due_date": cur_t + serv + 450.0,
                        "service_time": serv,
                    },
                    {
                        "cust_no": 100 + offset + 2,
                        "x": 28.0 + ((offset * 5) % 40) - 15,
                        "y": 70.0 - ((offset * 9) % 35) + 10,
                        "demand": 20,
                        "ready_time": cur_t,
                        "due_date": cur_t + serv + 500.0,
                        "service_time": serv,
                    },
                    {
                        "cust_no": 100 + offset + 3,
                        "x": 48.0 - ((offset * 13) % 30) + 15,
                        "y": 45.0 + ((offset * 7) % 25) - 10,
                        "demand": 10,
                        "ready_time": cur_t,
                        "due_date": cur_t + serv + 550.0,
                        "service_time": serv,
                    },
                ]
                try:
                    injected = sim_state.inject_new_orders(rush_orders)
                    sim_state.replan(exploration_level=1, reason=f"Demand Surge: +{len(injected)} Rush Orders Scheduled")
                    rt = sim_state.replanning_history[-1]["runtime_ms"]
                    st.toast(f"Injected {len(injected)} rush orders. Fleet re-routed in {rt:.2f} ms.")
                except Exception as e:
                    st.error(f"Failed to schedule rush orders: {e}")
                st.rerun()


# Invoke the live simulation fragment
render_live_simulation_board()


# ==============================================================================
# 8. Detailed Telemetry & Fleet Manifest Tabs
# ==============================================================================
st.markdown('<div style="height: 1px; background-color: #e4e4e7; margin: 1.5rem 0 1rem 0;"></div>', unsafe_allow_html=True)
tab_proof, tab_sla = st.tabs([
    "Priority Delivery Proof",
    "AI SLA Risk Radar",
])

with tab_proof:
    st.markdown(render_priority_delivery_proof_table_html(sim), unsafe_allow_html=True)

with tab_sla:
    st.markdown(render_ai_sla_radar_html(sim), unsafe_allow_html=True)

# with tab_routes:
#     st.markdown(render_route_manifest_table_html(sim), unsafe_allow_html=True)

# with tab_raw:
#     st.markdown(render_raw_audit_table_html(sim), unsafe_allow_html=True)
