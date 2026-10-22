"""
VRPTW Optimization Engine Interface using pyvroom (VROOM C++ bindings).

Prepares high-performance VROOM Input problems with vehicle fleet constraints,
scaled time windows, customer delivery demands, and custom duration matrices.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

# Bind vroom module (distributed via pyvroom wheel)
try:
    import vroom
    import vroom as pyvroom
except ImportError as err:
    raise ImportError(
        "pyvroom / vroom is not installed. Please run: pip install pyvroom"
    ) from err

from src.utils import to_vroom_matrix


def create_vroom_problem(
    parsed_data: Any,
    distance_matrix: Union[np.ndarray, Sequence[Sequence[int]]],
    profile: str = "car",
) -> vroom.Input:
    """
    Initialize and configure a pyvroom.Input problem instance for the VRPTW.
    
    Populates:
      1. Duration matrix computed from Euclidean integer coordinates.
      2. Vehicle fleet with depot start/end steps, capacity constraints, and operational horizons.
      3. Customer delivery jobs with location indices, demands, service durations, and time windows.
      
    Args:
        parsed_data: SolomonInstance object or dictionary containing:
                     - vehicle_count: int
                     - vehicle_capacity: int
                     - depot: dict/NodeData (with scaled_ready_time, scaled_due_date)
                     - customers: list of dicts/NodeData (with scaled_ready_time, scaled_due_date,
                                  scaled_service_time, demand, cust_no)
        distance_matrix: 2D integer distance/duration matrix (shape: (N+1, N+1)).
        profile: Routing profile identifier (default: "car").
        
    Returns:
        Configured pyvroom.Input problem object (unsolved).
    """
    problem = vroom.Input()

    # 1. Attach duration matrix (scaled integer travel times)
    vroom_mat = to_vroom_matrix(distance_matrix)
    problem.set_durations_matrix(profile=profile, matrix_input=vroom_mat)

    # 2. Extract fleet parameters
    if hasattr(parsed_data, "vehicle_count"):
        vehicle_count = parsed_data.vehicle_count
        vehicle_capacity = parsed_data.vehicle_capacity
        depot = parsed_data.depot
        customers = parsed_data.customers
    else:
        vehicle_count = parsed_data["vehicle_count"]
        vehicle_capacity = parsed_data["vehicle_capacity"]
        depot = parsed_data["depot"]
        customers = parsed_data["customers"]

    # Depot time window extraction (scaled integers)
    depot_start = (
        depot.scaled_ready_time
        if hasattr(depot, "scaled_ready_time")
        else depot["scaled_ready_time"]
    )
    depot_end = (
        depot.scaled_due_date
        if hasattr(depot, "scaled_due_date")
        else depot["scaled_due_date"]
    )

    depot_tw = vroom.TimeWindow(start=int(depot_start), end=int(depot_end))

    # 3. Add Vehicles
    # Depot location index in distance_matrix is 0
    vehicles: List[vroom.Vehicle] = []
    for v_id in range(1, vehicle_count + 1):
        veh = vroom.Vehicle(
            id=v_id,
            start=0,
            end=0,
            profile=profile,
            capacity=[int(vehicle_capacity)],
            time_window=depot_tw,
        )
        vehicles.append(veh)

    problem.add_vehicle(vehicles)

    # 4. Add Jobs (Customer Stops)
    # Location index corresponds to matrix row/col (1..N)
    jobs: List[vroom.Job] = []
    for loc_idx, cust in enumerate(customers, start=1):
        cust_id = cust.cust_no if hasattr(cust, "cust_no") else cust["cust_no"]
        demand = cust.demand if hasattr(cust, "demand") else cust["demand"]
        ready_time = (
            cust.scaled_ready_time
            if hasattr(cust, "scaled_ready_time")
            else cust["scaled_ready_time"]
        )
        due_date = (
            cust.scaled_due_date
            if hasattr(cust, "scaled_due_date")
            else cust["scaled_due_date"]
        )
        service_time = (
            cust.scaled_service_time
            if hasattr(cust, "scaled_service_time")
            else cust["scaled_service_time"]
        )

        job_tw = vroom.TimeWindow(start=int(ready_time), end=int(due_date))

        job = vroom.Job(
            id=int(cust_id),
            location=loc_idx,
            default_service=int(service_time),
            delivery=[int(demand)],
            time_windows=[job_tw],
        )
        jobs.append(job)

    problem.add_job(jobs)

    return problem
