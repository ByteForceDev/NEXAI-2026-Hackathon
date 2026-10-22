"""
Dynamic Simulation State and Live Re-planning Engine for VRPTW.

Manages mid-route state freezing at event timestamps T_event, vehicle capacity
tracking, disruption triggers (vehicle breakdowns, new rush orders), and lightning-fast
re-optimization via pyvroom.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Bind vroom module (distributed via pyvroom wheel)
try:
    import vroom
    import vroom as pyvroom
    try:
        from vroom.solution.solution import Solution
    except ImportError:
        from vroom._vroom import Solution
except ImportError as err:
    raise ImportError(
        "pyvroom / vroom is not installed. Please run: pip install pyvroom"
    ) from err


from src.parser import NodeData, SolomonInstance
from src.utils import (
    DEFAULT_SCALE_FACTOR,
    compute_distance_matrix,
    scale_value,
    to_vroom_matrix,
    unscale_value,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class VehicleState:
    """Tracks dynamic mid-day operational status and parallel kinematics of a single fleet vehicle."""
    vehicle_id: int
    start_location: int = 0
    remaining_capacity: int = 200
    available_from: int = 0
    completed_job_ids: List[int] = field(default_factory=list)
    completed_demands: int = 0
    last_completed_job_id: Optional[int] = None
    last_completed_location: int = 0
    last_completion_time: int = 0
    route_end_time: int = 0
    is_returned_to_depot: bool = False
    is_completed_route: bool = False
    is_broken: bool = False

    # Independent parallel movement state
    current_x: float = 0.0
    current_y: float = 0.0
    current_route: List[Dict[str, Any]] = field(default_factory=list)
    current_waypoint_index: int = 0
    speed: float = 1.0  # Units of coordinate distance per unscaled time unit
    distance_to_next_waypoint: float = 0.0
    status: str = "At Depot (Idle)"
    is_active: bool = False
    assigned_orders: List[int] = field(default_factory=list)
    route_progress: float = 0.0  # 0.0 to 100.0%

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "start_location": self.start_location,
            "remaining_capacity": self.remaining_capacity,
            "available_from": self.available_from,
            "completed_job_ids": list(self.completed_job_ids),
            "completed_demands": self.completed_demands,
            "last_completed_job_id": self.last_completed_job_id,
            "last_completed_location": self.last_completed_location,
            "last_completion_time": self.last_completion_time,
            "route_end_time": self.route_end_time,
            "is_returned_to_depot": self.is_returned_to_depot,
            "is_completed_route": self.is_completed_route,
            "is_broken": self.is_broken,
            "current_x": self.current_x,
            "current_y": self.current_y,
            "current_waypoint_index": self.current_waypoint_index,
            "speed": self.speed,
            "distance_to_next_waypoint": self.distance_to_next_waypoint,
            "status": self.status,
            "is_active": self.is_active,
            "assigned_orders": list(self.assigned_orders),
            "route_progress": self.route_progress,
        }


class SimulationState:
    """
    Simulation State Manager for Dynamic Vehicle Routing with Time Windows (VRPTW).
    
    Coordinates:
      - Temporal simulation clock advancing to T_event (with 100x scaling).
      - State slicing: freezing completed stops, tracking van locations and remaining loads.
      - Disruption handling: dynamic customer order injection and vehicle breakdown events.
      - Live sub-second re-planning via pyvroom.
    """

    def __init__(
        self,
        instance: SolomonInstance,
        distance_matrix: np.ndarray,
        initial_solution: Any,
        scale_factor: int = DEFAULT_SCALE_FACTOR,
    ) -> None:
        """
        Initialize simulation state from parsed instance and initial routing solution.
        
        Args:
            instance: Parsed Solomon benchmark problem instance.
            distance_matrix: Initial 2D square distance/duration matrix.
            initial_solution: Baseline vroom.Solution computed at T = 0.
            scale_factor: Integer scaling factor for temporal and spatial metrics (default: 100).
        """
        self.instance = instance
        self.distance_matrix = distance_matrix.copy()
        self.current_solution = initial_solution
        self.scale_factor = scale_factor

        # Temporal clock (scaled integer)
        self.current_time: int = 0

        # Coordinate registry: index matches row/col in distance_matrix
        # Index 0 is Depot; Indices 1..N are original customers; >N are newly injected orders
        self.scaled_coordinates: List[Tuple[int, int]] = list(self.instance.scaled_coordinates)

        # Fast customer metadata lookup by customer ID
        self.customer_lookup: Dict[int, Any] = {
            node.cust_no: node for node in self.instance.customers
        }

        # Jobs state tracking
        self.completed_jobs: Dict[int, Dict[str, Any]] = {}
        self.pending_jobs: Dict[int, vroom.Job] = {}

        # Broken vehicles registry
        self.broken_vehicles: Set[int] = set()

        # Initialize fleet vehicle states
        self.vehicles: Dict[int, VehicleState] = {}
        depot_start_time = int(self.instance.depot.scaled_ready_time)
        depot_capacity = int(self.instance.vehicle_capacity)

        for v_id in range(1, self.instance.vehicle_count + 1):
            self.vehicles[v_id] = VehicleState(
                vehicle_id=v_id,
                start_location=0,
                remaining_capacity=depot_capacity,
                available_from=depot_start_time,
            )

        # Seed initial pending jobs pool
        self._seed_initial_pending_jobs()

        # Historical audit log for re-planning events
        self.replanning_history: List[Dict[str, Any]] = []

        # Initialize parallel movement state across fleet at T = 0
        self._update_parallel_vehicle_movement(0.0)

    def _seed_initial_pending_jobs(self) -> None:
        """Populate initial pending jobs pool with all customer nodes from the problem instance."""
        self.pending_jobs.clear()
        for idx, cust in enumerate(self.instance.customers, start=1):
            job = vroom.Job(
                id=int(cust.cust_no),
                location=idx,
                default_service=int(cust.scaled_service_time),
                delivery=[int(cust.demand)],
                time_windows=[
                    vroom.TimeWindow(
                        start=int(cust.scaled_ready_time),
                        end=int(cust.scaled_due_date),
                    )
                ],
            )
            self.pending_jobs[int(cust.cust_no)] = job

    def advance_clock(self, T_event_unscaled: float) -> Dict[str, Any]:
        """
        Advance the simulation clock to T_event and freeze historical delivery states.
        
        Slices the active VROOM solution:
          - Marks jobs with (arrival + service) <= T_event as Completed.
          - Re-pools jobs with (arrival + service) > T_event into pending_jobs.
          - Updates vehicle start location to its last completed stop (or depot).
          - Updates remaining vehicle capacity to original_capacity - completed_demands.
          - Sets vehicle available_from time to max(T_event, completion_time_of_last_job).
          
        Args:
            T_event_unscaled: Target simulation timestamp in original unscaled time units.
            
        Returns:
            Dictionary summary of the clock advance event.
        """
        T_event = scale_value(T_event_unscaled, self.scale_factor)
        if T_event < self.current_time:
            raise ValueError(
                f"Clock cannot move backwards: requested {T_event} ({T_event_unscaled}), "
                f"current {self.current_time} ({unscale_value(self.current_time, self.scale_factor)})"
            )

        self.current_time = T_event
        routes_data = self._extract_routes_data()

        # Track completed job IDs observed in this clock advance
        newly_completed_job_ids: Set[int] = set()

        for r in routes_data:
            v_id = int(r["vehicle_id"])
            if v_id not in self.vehicles:
                continue

            v_state = self.vehicles[v_id]
            steps = r.get("steps", [])

            for step in steps:
                if step.get("type") != "job":
                    continue

                job_id = int(step["id"])
                arr = int(step["arrival"])
                serv = int(step["service"])
                wait = int(step.get("waiting_time", 0))
                loc_idx = int(step.get("location_index", 0))

                # Step completion time
                comp_time = arr + wait + serv

                # CRITICAL RULE: If arrival + service <= T_event, mark completed
                if (arr + serv) <= T_event:
                    if job_id not in self.completed_jobs:
                        cust_demand = (
                            self.customer_lookup[job_id].demand
                            if job_id in self.customer_lookup
                            else 0
                        )
                        self.completed_jobs[job_id] = {
                            "job_id": job_id,
                            "vehicle_id": v_id,
                            "location_index": loc_idx,
                            "arrival": arr,
                            "service": serv,
                            "waiting_time": wait,
                            "completion_time": comp_time,
                            "demand": cust_demand,
                        }
                        newly_completed_job_ids.add(job_id)
                        if job_id not in v_state.completed_job_ids:
                            v_state.completed_job_ids.append(job_id)
                            v_state.completed_demands += cust_demand

                        # Track last completed stop details for vehicle positioning
                        v_state.last_completed_job_id = job_id
                        v_state.last_completed_location = loc_idx
                        v_state.last_completion_time = comp_time

                    # Remove from pending pool
                    if job_id in self.pending_jobs:
                        del self.pending_jobs[job_id]
                else:
                    # Occurs > T_event: job remains in pending pool
                    if job_id in self.completed_jobs:
                        # Already completed in an earlier step
                        continue
                    if job_id not in self.pending_jobs:
                        if job_id in self.customer_lookup:
                            cust = self.customer_lookup[job_id]
                            self.pending_jobs[job_id] = vroom.Job(
                                id=cust.cust_no,
                                location=loc_idx,
                                default_service=int(cust.scaled_service_time),
                                delivery=[int(cust.demand)],
                                time_windows=[
                                    vroom.TimeWindow(
                                        start=int(cust.scaled_ready_time),
                                        end=int(cust.scaled_due_date),
                                    )
                                ],
                            )

        # Map route end times and steps per vehicle from active routes
        route_end_times: Dict[int, int] = {}
        for r in routes_data:
            vid = int(r["vehicle_id"])
            steps = r.get("steps", [])
            if steps:
                end_step = steps[-1]
                route_end_times[vid] = int(end_step.get("arrival", 0))

        # Update mid-route vehicle states across entire fleet
        for v_id, v_state in self.vehicles.items():
            route_end = route_end_times.get(v_id)
            if route_end is not None:
                v_state.route_end_time = route_end
                if T_event >= route_end:
                    # Edge Case B & C: Van has returned to Depot: shift complete, ready to reload for new dispatches!
                    v_state.start_location = 0
                    v_state.is_returned_to_depot = True
                    v_state.is_completed_route = True
                    v_state.available_from = max(T_event, route_end)
                    v_state.remaining_capacity = int(self.instance.vehicle_capacity)
                    continue

            # Vehicle is en route or idle at depot
            v_state.is_returned_to_depot = False
            v_state.is_completed_route = False
            if v_state.last_completed_job_id is not None:
                # Van is located at its last completed customer location
                v_state.start_location = v_state.last_completed_location
                v_state.available_from = max(T_event, v_state.last_completion_time)
            else:
                # Van has not completed any jobs: stays at Depot
                v_state.start_location = 0
                v_state.available_from = max(
                    T_event, int(self.instance.depot.scaled_ready_time)
                )

            # Updated remaining capacity
            v_state.remaining_capacity = max(
                0, int(self.instance.vehicle_capacity) - v_state.completed_demands
            )

        # Central parallel kinematic update: advances all active vehicles concurrently
        self._update_parallel_vehicle_movement(T_event_unscaled)

        summary = {
            "t_event_unscaled": T_event_unscaled,
            "t_event_scaled": T_event,
            "completed_total": len(self.completed_jobs),
            "newly_completed_count": len(newly_completed_job_ids),
            "pending_count": len(self.pending_jobs),
            "active_vehicles_count": len(self.vehicles) - len(self.broken_vehicles),
            "broken_vehicles_count": len(self.broken_vehicles),
        }
        logger.info(
            "Clock advanced to T=%s (scaled %s): %s completed, %s pending",
            T_event_unscaled,
            T_event,
            summary["completed_total"],
            summary["pending_count"],
        )
        return summary

    def _extract_routes_data(self) -> List[Dict[str, Any]]:
        """Extract routes and chronological steps from current vroom.Solution."""
        if hasattr(self.current_solution, "to_dict"):
            try:
                dict_sol = self.current_solution.to_dict()
                if "routes" in dict_sol and isinstance(dict_sol["routes"], list):
                    return [
                        {
                            "vehicle_id": r.get("vehicle", r.get("vehicle_id")),
                            "steps": r.get("steps", []),
                        }
                        for r in dict_sol["routes"]
                    ]
            except Exception as e:
                logger.debug("Failed extracting routes from to_dict: %s", e)

        # Fallback to pandas DataFrame representation
        if hasattr(self.current_solution, "routes") and isinstance(
            self.current_solution.routes, pd.DataFrame
        ):
            df = self.current_solution.routes
            extracted = []
            for v_id, group in df.groupby("vehicle_id"):
                steps_list = []
                for _, row in group.iterrows():
                    steps_list.append(row.to_dict())
                extracted.append({"vehicle_id": int(v_id), "steps": steps_list})
            return extracted

        return []

    def inject_new_orders(self, new_customers: List[Dict[str, Any]]) -> List[int]:
        """
        Inject dynamic mid-day customer orders into the pending jobs pool.
        
        Scales coordinates and temporal bounds by factor 100, computes exact Euclidean
        distances to expand the distance matrix, and registers new vroom.Job objects.
        
        Args:
            new_customers: List of customer dictionaries containing:
                           - x, y: Continuous Cartesian coordinates
                           - demand: Delivery item quantity
                           - ready_time (or ready): Earliest service start
                           - due_date (or due): Latest service start
                           - service_time (or service): Duration required
                           - Optional cust_no or id
                           
        Returns:
            List of assigned integer customer IDs for the newly injected orders.
        """
        injected_ids: List[int] = []
        if not new_customers:
            return injected_ids

        # Determine safe starting ID for new customers
        existing_ids = set(self.customer_lookup.keys()) | set(self.pending_jobs.keys()) | set(self.completed_jobs.keys())
        max_existing_id = max(existing_ids) if existing_ids else 0

        for cust_dict in new_customers:
            # 1. Resolve ID
            raw_id = (
                cust_dict.get("cust_no")
                or cust_dict.get("id")
                or cust_dict.get("CUST NO.")
            )
            cust_id = int(raw_id) if raw_id is not None else (max_existing_id + 1)
            max_existing_id = max(max_existing_id, cust_id)

            # 2. Extract spatial coordinates
            x_val = float(cust_dict.get("x", cust_dict.get("XCOORD.", cust_dict.get("XCOORD", 0.0))))
            y_val = float(cust_dict.get("y", cust_dict.get("YCOORD.", cust_dict.get("YCOORD", 0.0))))
            demand_val = int(cust_dict.get("demand", cust_dict.get("DEMAND", 10)))

            ready_val = float(cust_dict.get("ready_time", cust_dict.get("ready", cust_dict.get("READY TIME", 0.0))))
            due_val = float(cust_dict.get("due_date", cust_dict.get("due", cust_dict.get("DUE DATE", 1236.0))))
            serv_val = float(cust_dict.get("service_time", cust_dict.get("service", cust_dict.get("SERVICE TIME", 90.0))))

            # 3. Mathematical integer scaling (factor 100)
            scaled_x = scale_value(x_val, self.scale_factor)
            scaled_y = scale_value(y_val, self.scale_factor)
            scaled_ready = scale_value(ready_val, self.scale_factor)
            scaled_due = scale_value(due_val, self.scale_factor)
            scaled_serv = scale_value(serv_val, self.scale_factor)

            # Defensive validation: Ensure time window start <= end to prevent VroomInputException
            if scaled_due < scaled_ready:
                scaled_due = scaled_ready + max(scaled_serv, scale_value(60.0, self.scale_factor))
                due_val = unscale_value(scaled_due, self.scale_factor)

            # 4. Check if existing location index is explicitly passed (shared node)
            explicit_loc = cust_dict.get("location_index")
            if explicit_loc is not None and int(explicit_loc) < len(self.scaled_coordinates):
                loc_idx = int(explicit_loc)
            else:
                # Append new coordinates and dynamically expand the distance matrix
                self.scaled_coordinates.append((scaled_x, scaled_y))
                loc_idx = len(self.scaled_coordinates) - 1

            # 5. Build and register NodeData
            node = NodeData(
                cust_no=cust_id,
                x=x_val,
                y=y_val,
                scaled_x=scaled_x,
                scaled_y=scaled_y,
                demand=demand_val,
                ready_time=ready_val,
                due_date=due_val,
                scaled_ready_time=scaled_ready,
                scaled_due_date=scaled_due,
                service_time=serv_val,
                scaled_service_time=scaled_serv,
            )
            self.customer_lookup[cust_id] = node

            # 6. Create vroom.Job and append to pending pool
            job = vroom.Job(
                id=cust_id,
                location=loc_idx,
                default_service=scaled_serv,
                delivery=[demand_val],
                time_windows=[vroom.TimeWindow(start=scaled_ready, end=scaled_due)],
            )
            self.pending_jobs[cust_id] = job
            injected_ids.append(cust_id)

        # 7. Recompute expanded integer Euclidean distance matrix
        self.distance_matrix = compute_distance_matrix(
            self.scaled_coordinates,
            scale_factor=self.scale_factor,
            already_scaled=True,
        )

        logger.info(
            "Injected %d dynamic orders: IDs=%s. Distance matrix expanded to %s",
            len(injected_ids),
            injected_ids,
            self.distance_matrix.shape,
        )
        return injected_ids

    def trigger_breakdown(self, vehicle_id: int) -> Dict[str, Any]:
        """
        Simulate an abrupt mid-route vehicle breakdown.
        
        Removes the vehicle from active fleet dispatch. Its completed deliveries remain
        frozen, while its pending stops remain in pending_jobs for automated reassignment.
        
        Args:
            vehicle_id: Integer identifier of the disabled vehicle.
            
        Returns:
            Dictionary confirmation with affected vehicle status.
        """
        if vehicle_id not in self.vehicles:
            logger.warning("Vehicle ID %d not found in fleet.", vehicle_id)
            return {
                "vehicle_id": vehicle_id,
                "status": "not_found",
                "message": f"Vehicle {vehicle_id} is not in the fleet.",
            }

        v_state = self.vehicles[vehicle_id]

        # Edge Case A: Check if vehicle has already completed its route and returned to depot
        if v_state.is_completed_route or v_state.is_returned_to_depot:
            logger.info(
                "Vehicle %d breakdown ignored: vehicle has already completed its route and returned to depot.",
                vehicle_id,
            )
            return {
                "vehicle_id": vehicle_id,
                "status": "already_completed",
                "message": f"Van #{vehicle_id} has already completed its shift and safely returned to depot. Breakdown ignored.",
                "completed_jobs": list(v_state.completed_job_ids),
                "pending_jobs_to_reassign": 0,
            }

        if v_state.is_broken or vehicle_id in self.broken_vehicles:
            return {
                "vehicle_id": vehicle_id,
                "status": "already_broken",
                "message": f"Van #{vehicle_id} is already disabled.",
                "completed_jobs": list(v_state.completed_job_ids),
                "pending_jobs_to_reassign": len(self.pending_jobs),
            }

        self.broken_vehicles.add(vehicle_id)
        v_state.is_broken = True

        logger.warning(
            "Vehicle %d breakdown triggered! Removed from dispatch. Completed jobs: %s",
            vehicle_id,
            v_state.completed_job_ids,
        )

        return {
            "vehicle_id": vehicle_id,
            "status": "disabled",
            "message": f"Van #{vehicle_id} broken down at node {v_state.start_location}. Unfinished stops re-pooled.",
            "completed_jobs": list(v_state.completed_job_ids),
            "pending_jobs_to_reassign": len(self.pending_jobs),
        }

    def replan(
        self,
        exploration_level: int = 1,
        nb_threads: int = 4,
        profile: str = "car",
        reason: str = "Dynamic Re-planning Event",
    ) -> Any:
        """
        Execute live sub-second re-planning with the updated operational state.
        
        Instantiates a fresh vroom.Input problem:
          - Applies the current (possibly expanded) integer duration matrix.
          - Ingests active vehicles with current mid-route start coordinates, remaining
            capacities, and adjusted operational horizons.
          - Ingests all pending jobs (uncompleted original stops + injected rush orders).
          - Calls pyvroom.solve() and records performance telemetry.
          
        Args:
            exploration_level: VROOM search thoroughness (0 to 5; default 1 for sub-10ms solving).
            nb_threads: Concurrency thread count (default: 4).
            profile: Routing profile key (default: "car").
            
        Returns:
            Newly computed vroom.Solution.
        """
        start_time = time.perf_counter()

        problem = vroom.Input()

        # 1. Attach duration matrix
        vroom_mat = to_vroom_matrix(self.distance_matrix)
        problem.set_durations_matrix(profile=profile, matrix_input=vroom_mat)

        # 2. Add active operational vehicles with dynamically extended horizon for indefinite simulation
        horizon_buffer = scale_value(3000.0, self.scale_factor)
        max_horizon = max(int(self.instance.depot.scaled_due_date), self.current_time + horizon_buffer)
        active_vehicles: List[vroom.Vehicle] = []

        for v_id, v_state in self.vehicles.items():
            # Exclude disabled/broken vehicles
            if v_id in self.broken_vehicles or v_state.is_broken:
                continue

            # If vehicle has returned to depot, reload to full capacity for new dispatches
            if v_state.is_returned_to_depot or v_state.start_location == 0:
                v_state.remaining_capacity = int(self.instance.vehicle_capacity)

            rem_capacity = v_state.remaining_capacity
            if rem_capacity <= 0:
                continue

            start_loc = v_state.start_location
            time_window_start = v_state.available_from

            veh_end = max(max_horizon, time_window_start + scale_value(1200.0, self.scale_factor))

            veh = vroom.Vehicle(
                id=v_id,
                start=start_loc,
                end=0,  # Vehicles canonically return to the central depot
                profile=profile,
                capacity=[rem_capacity],
                time_window=vroom.TimeWindow(start=time_window_start, end=veh_end),
            )
            active_vehicles.append(veh)

        if not active_vehicles:
            raise RuntimeError("No active vehicles available for re-planning!")

        problem.add_vehicle(active_vehicles)

        # 3. Add all pending jobs (Edge Case C: handle 0 pending jobs safely)
        if not self.pending_jobs:
            logger.info("All deliveries completed: 0 pending jobs. No re-plan required.")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            telemetry = {
                "timestamp_scaled": self.current_time,
                "timestamp_unscaled": unscale_value(self.current_time, self.scale_factor),
                "description": reason,
                "runtime_ms": round(elapsed_ms, 2),
                "cost": 0,
                "routes_count": 0,
                "total_steps": 0,
                "pending_jobs_count": 0,
                "active_vehicles_count": len(active_vehicles),
            }
            self.replanning_history.append(telemetry)
            return self.current_solution

        problem.add_job(list(self.pending_jobs.values()))

        # 4. Invoke pyvroom optimization engine
        solution = problem.solve(exploration_level=exploration_level, nb_threads=nb_threads)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Record telemetry and update current solution
        self.current_solution = solution
        self._update_parallel_vehicle_movement(unscale_value(self.current_time, self.scale_factor))

        num_routes = (
            len(solution.routes["vehicle_id"].unique())
            if hasattr(solution, "routes") and isinstance(solution.routes, pd.DataFrame) and not solution.routes.empty
            else len(solution.to_dict().get("routes", [])) if hasattr(solution, "to_dict") else 0
        )

        telemetry = {
            "timestamp_scaled": self.current_time,
            "timestamp_unscaled": unscale_value(self.current_time, self.scale_factor),
            "description": reason,
            "runtime_ms": round(elapsed_ms, 2),
            "cost": solution.summary.cost,
            "routes_count": num_routes,
            "total_steps": len(solution.routes) if hasattr(solution, "routes") else 0,
            "pending_jobs_count": len(self.pending_jobs),
            "active_vehicles_count": len(active_vehicles),
        }
        self.replanning_history.append(telemetry)

        logger.info(
            "Live re-planning finished in %.2f ms (Cost: %s, Routes: %d)",
            elapsed_ms,
            solution.summary.cost,
            num_routes,
        )

        return solution

    def get_summary(self) -> Dict[str, Any]:
        """Return an operational snapshot of the simulation state."""
        return {
            "current_time_unscaled": unscale_value(self.current_time, self.scale_factor),
            "current_time_scaled": self.current_time,
            "total_nodes": len(self.scaled_coordinates),
            "completed_jobs_count": len(self.completed_jobs),
            "pending_jobs_count": len(self.pending_jobs),
            "total_fleet_size": len(self.vehicles),
            "active_vehicles_count": len(self.vehicles) - len(self.broken_vehicles),
            "broken_vehicles_count": len(self.broken_vehicles),
            "replan_count": len(self.replanning_history),
            "latest_replan_runtime_ms": (
                self.replanning_history[-1]["runtime_ms"]
                if self.replanning_history
                else None
            ),
        }

    def is_simulation_completed(self) -> bool:
        """
        Check if all deliveries are completed and all active vehicles have returned to the depot.
        
        Returns:
            True if 0 pending jobs remain and all active fleet routes have finished.
        """
        if len(self.pending_jobs) > 0:
            return False

        routes_data = self._extract_routes_data()
        if not routes_data:
            return True

        for r in routes_data:
            v_id = int(r["vehicle_id"])
            if v_id in self.broken_vehicles:
                continue
            steps = r.get("steps", [])
            if steps:
                final_step = steps[-1]
                end_arr = int(final_step.get("arrival", 0))
                if self.current_time < end_arr:
                    return False

        return True

    def _get_unscaled_node_coords(self, loc_idx: int) -> Tuple[float, float]:
        """Translate a location index to continuous unscaled (X, Y) coordinates."""
        if loc_idx == 0:
            return (float(self.instance.depot.x), float(self.instance.depot.y))
        if loc_idx < len(self.scaled_coordinates):
            sx, sy = self.scaled_coordinates[loc_idx]
            return (
                unscale_value(sx, self.scale_factor),
                unscale_value(sy, self.scale_factor),
            )
        return (0.0, 0.0)

    def _update_parallel_vehicle_movement(self, t_curr_unscaled: float) -> None:
        """
        Parallel Vehicle Movement Central Update.
        
        Advances all active vehicles concurrently during each simulation step.
        Calculates independent kinematic positions, waypoint indices, distances,
        speeds, order statuses, and route progress for every vehicle in parallel.
        """
        depot_x, depot_y = float(self.instance.depot.x), float(self.instance.depot.y)
        t_curr_scaled = scale_value(t_curr_unscaled, self.scale_factor)

        routes_data = self._extract_routes_data()
        routes_by_veh = {int(r["vehicle_id"]): r.get("steps", []) for r in routes_data}

        # Central parallel loop: compute movement state independently for each vehicle
        for v_id, v_state in self.vehicles.items():
            # 1. Broken down vehicle (pinned at breakdown location)
            if v_id in self.broken_vehicles or v_state.is_broken:
                v_state.is_broken = True
                v_state.is_active = False
                v_state.speed = 0.0
                v_state.distance_to_next_waypoint = 0.0
                v_state.status = "DISABLED / BREAKDOWN"
                if v_state.current_x == 0.0 and v_state.current_y == 0.0:
                    loc = (
                        v_state.last_completed_location
                        if v_state.last_completed_location is not None
                        else v_state.start_location
                    )
                    c = self._get_unscaled_node_coords(loc)
                    v_state.current_x, v_state.current_y = c[0], c[1]
                continue

            v_steps = routes_by_veh.get(v_id, [])
            v_state.current_route = v_steps

            # Extract assigned order IDs
            assigned_orders = [
                int(s["id"]) for s in v_steps
                if s.get("type") == "job" and s.get("id") is not None
            ]
            v_state.assigned_orders = assigned_orders

            # 2. Idle / Reserve fleet vehicle with no assigned route
            if not v_steps:
                v_state.current_x = depot_x
                v_state.current_y = depot_y
                v_state.current_waypoint_index = 0
                v_state.speed = 0.0
                v_state.distance_to_next_waypoint = 0.0
                v_state.status = "At Depot (Reserve Fleet)"
                v_state.is_active = False
                v_state.route_progress = 0.0
                continue

            first_step = v_steps[0]
            last_step = v_steps[-1]
            first_arr_scaled = int(first_step.get("arrival", 0))
            last_arr_scaled = int(last_step.get("arrival", 0))
            num_steps = len(v_steps)

            # 3. Pre-departure idle at Depot
            if t_curr_scaled <= first_arr_scaled:
                v_state.current_x = depot_x
                v_state.current_y = depot_y
                v_state.current_waypoint_index = 0
                v_state.speed = 0.0
                v_state.status = "At Depot (Pre-departure)"
                v_state.is_active = True
                v_state.route_progress = 0.0
                if num_steps > 1:
                    next_loc = int(v_steps[1].get("location_index", 0))
                    nx, ny = self._get_unscaled_node_coords(next_loc)
                    v_state.distance_to_next_waypoint = float(np.hypot(nx - depot_x, ny - depot_y))
                else:
                    v_state.distance_to_next_waypoint = 0.0
                continue

            # 4. Route completed: parked back at Depot
            if t_curr_scaled >= last_arr_scaled:
                v_state.current_x = depot_x
                v_state.current_y = depot_y
                v_state.current_waypoint_index = num_steps - 1
                v_state.speed = 0.0
                v_state.distance_to_next_waypoint = 0.0
                v_state.status = "At Depot (Route Completed)"
                v_state.is_active = False
                v_state.is_returned_to_depot = True
                v_state.is_completed_route = True
                v_state.route_progress = 100.0
                continue

            # 5. Active route execution: traverse waypoints with distance-based interpolation
            v_state.is_active = True
            v_state.is_returned_to_depot = False
            v_state.is_completed_route = False

            matched = False
            for i in range(num_steps - 1):
                s_curr = v_steps[i]
                s_next = v_steps[i + 1]

                arr_curr = int(s_curr.get("arrival", 0))
                serv_curr = int(s_curr.get("service", 0))
                wait_curr = int(s_curr.get("waiting_time", 0))
                dep_curr = arr_curr + wait_curr + serv_curr
                arr_next = int(s_next.get("arrival", 0))

                loc_curr = int(s_curr.get("location_index", 0))
                loc_next = int(s_next.get("location_index", 0))
                c_curr = self._get_unscaled_node_coords(loc_curr)
                c_next = self._get_unscaled_node_coords(loc_next)
                seg_dist = float(np.hypot(c_next[0] - c_curr[0], c_next[1] - c_curr[1]))

                # Case A: Vehicle stopped at waypoint i (waiting or servicing order)
                if arr_curr <= t_curr_scaled <= dep_curr:
                    v_state.current_x = c_curr[0]
                    v_state.current_y = c_curr[1]
                    v_state.current_waypoint_index = i
                    v_state.distance_to_next_waypoint = seg_dist
                    v_state.speed = 0.0

                    job_lbl = f"Stop #{s_curr.get('id')}" if s_curr.get("id") is not None else "Depot"
                    action_lbl = "Waiting" if t_curr_scaled < (arr_curr + wait_curr) else "Servicing"
                    v_state.status = f"{action_lbl} at {job_lbl}"
                    v_state.route_progress = round((i / max(1, num_steps - 1)) * 100.0, 1)
                    matched = True
                    break

                # Case B: In transit between waypoint i and waypoint i+1
                elif dep_curr < t_curr_scaled < arr_next:
                    travel_time_scaled = arr_next - dep_curr
                    alpha = (t_curr_scaled - dep_curr) / travel_time_scaled if travel_time_scaled > 0 else 1.0
                    alpha = max(0.0, min(1.0, alpha))

                    # Continuous distance-based coordinate interpolation
                    v_state.current_x = c_curr[0] + alpha * (c_next[0] - c_curr[0])
                    v_state.current_y = c_curr[1] + alpha * (c_next[1] - c_curr[1])
                    v_state.current_waypoint_index = i
                    v_state.distance_to_next_waypoint = (1.0 - alpha) * seg_dist

                    travel_time_unscaled = unscale_value(travel_time_scaled, self.scale_factor)
                    v_state.speed = seg_dist / travel_time_unscaled if travel_time_unscaled > 0 else 1.0

                    from_lbl = f"#{s_curr.get('id')}" if s_curr.get("id") is not None else "Depot"
                    to_lbl = f"#{s_next.get('id')}" if s_next.get("id") is not None else "Depot"
                    v_state.status = f"In Transit: {from_lbl} ➔ {to_lbl} ({alpha*100:.0f}%)"
                    v_state.route_progress = round(((i + alpha) / max(1, num_steps - 1)) * 100.0, 1)
                    matched = True
                    break

            if not matched:
                loc = v_state.start_location
                c = self._get_unscaled_node_coords(loc)
                v_state.current_x, v_state.current_y = c[0], c[1]
                v_state.status = "In Operation"
                v_state.speed = 1.0
                v_state.distance_to_next_waypoint = 0.0

    def get_interpolated_vehicle_positions(self) -> Dict[int, Dict[str, Any]]:
        """
        Return the current parallel kinematic positions and movement states for all fleet vehicles.
        
        Guarantees that all active vehicles advance concurrently during each simulation interval.
        Each vehicle maintains its own movement state (ID, coordinates, route, waypoint index,
        speed, distance to next waypoint, status, assigned orders, route progress).
        """
        positions: Dict[int, Dict[str, Any]] = {}
        for vid, v in self.vehicles.items():
            positions[vid] = {
                "x": v.current_x,
                "y": v.current_y,
                "status": v.status,
                "is_broken": v.is_broken,
                "is_active": v.is_active,
                "waypoint_index": v.current_waypoint_index,
                "speed": v.speed,
                "distance_to_next_waypoint": v.distance_to_next_waypoint,
                "assigned_orders": list(v.assigned_orders),
                "route_progress": v.route_progress,
                "location_index": v.start_location,
            }
        return positions
