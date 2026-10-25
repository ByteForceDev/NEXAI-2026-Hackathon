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
from src.customer_scoring_engine import CustomerScoringEngine


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
         Integrates ML-derived customer priority scores (new customer retention boost vs chronic misser penalty).
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

    # 4. Add Jobs (Customer Stops) with ML-derived priority scoring
    scoring_engine = CustomerScoringEngine.get_instance()
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

        profile = scoring_engine.get_profile(int(cust_id))
        vroom_priority = int(profile.vroom_priority)

        # Expedite due date for VIP / New Customers to ensure early schedule priority
        effective_due = int(due_date)
        if profile.priority_tier == "VIP Expedited":
            effective_due = max(int(ready_time) + int(service_time) * 4, int(due_date * 0.75))
        elif profile.priority_tier == "Priority High":
            effective_due = max(int(ready_time) + int(service_time) * 4, int(due_date * 0.88))

        job_tw = vroom.TimeWindow(start=int(ready_time), end=int(effective_due))

        job = vroom.Job(
            id=int(cust_id),
            location=loc_idx,
            default_service=int(service_time),
            delivery=[int(demand)],
            priority=vroom_priority,
            time_windows=[job_tw],
        )
        jobs.append(job)

    problem.add_job(jobs)

    return problem
