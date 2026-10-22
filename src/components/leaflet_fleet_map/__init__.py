"""
Leaflet + OpenStreetMap Custom Streamlit Component for Autonomous Fleet Visualization.

Provides real-world interactive road mapping, OpenStreetMap tiles, true road polylines,
client-side 60 FPS vehicle kinematics interpolation, and ShadCN theme styling.
"""

from typing import Any, Dict, List, Optional
import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

_leaflet_component = components.declare_component(
    "leaflet_fleet_map",
    path=_COMPONENT_DIR,
)


def leaflet_fleet_map(
    depot: Dict[str, Any],
    routes: List[Dict[str, Any]],
    vehicles: List[Dict[str, Any]],
    completed_orders: List[Dict[str, Any]],
    pending_orders: List[Dict[str, Any]],
    rush_orders: List[Dict[str, Any]],
    center: List[float],
    zoom: int = 13,
    sim_time: float = 0.0,
    route_version: str = "v1",
    height: int = 560,
    theme: str = "light",
    geo_metadata: Optional[Dict[str, Any]] = None,
    is_simulating: bool = False,
    is_completed: bool = False,
    is_expanded: bool = False,
    broken_vehicles: Optional[List[int]] = None,
    default: Any = None,
    key: Optional[str] = "leaflet_fleet_map",
    **kwargs: Any,
) -> Any:
    """
    Render Leaflet OpenStreetMap fleet map with client-side 60 FPS kinematics interpolation.

    Args:
        depot: Depot geographic coordinates (lat, lon) and metadata.
        routes: List of vehicle road route polylines and stop metadata.
        vehicles: Current authoritative vehicle positions (lat, lon) and status.
        completed_orders: List of completed order nodes with (lat, lon).
        pending_orders: List of pending order nodes with (lat, lon).
        rush_orders: List of dynamic rush orders with (lat, lon).
        center: Initial map center [latitude, longitude].
        zoom: Initial map zoom level.
        sim_time: Current simulation timestamp.
        route_version: Version identifier for route geometry cache busting.
        height: Map container height in pixels.
        theme: "light" or "dark" theme mode.
        geo_metadata: Metadata regarding the geographic testbed projection.
        is_simulating: Current simulation running state.
        is_completed: Whether all deliveries are completed.
        is_expanded: Whether expanded fullscreen view is active.
        broken_vehicles: List of broken down vehicle IDs.
        default: Default return value.
        key: Unique Streamlit component key.
        **kwargs: Additional optional parameters.
    """
    return _leaflet_component(
        depot=depot,
        routes=routes,
        vehicles=vehicles,
        completed_orders=completed_orders,
        pending_orders=pending_orders,
        rush_orders=rush_orders,
        center=center,
        zoom=zoom,
        sim_time=sim_time,
        route_version=route_version,
        height=height,
        theme=theme,
        geo_metadata=geo_metadata or {},
        is_simulating=is_simulating,
        is_completed=is_completed,
        is_expanded=is_expanded,
        broken_vehicles=broken_vehicles or [],
        default=default,
        key=key,
        **kwargs,
    )
