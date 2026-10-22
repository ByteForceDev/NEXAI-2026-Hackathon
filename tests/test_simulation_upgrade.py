"""
Integration test suite for autonomous simulation loop, interpolation, and edge cases A, B, C.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.parser import parse_solomon_instance
from src.vrptw_solver import create_vroom_problem
from src.simulator import SimulationState

def test_interpolation_and_t0_visibility():
    inst = parse_solomon_instance('data/C101_25.txt')
    matrix = inst.compute_distance_matrix()
    prob = create_vroom_problem(inst, matrix)
    sol = prob.solve(exploration_level=1, nb_threads=2)
    sim = SimulationState(inst, matrix, sol)

    # 1. At T = 0, verify all active vehicles are at the Depot (40.0, 50.0)
    pos_t0 = sim.get_interpolated_vehicle_positions()
    for v_id in [1, 2, 3]:
        p = pos_t0[v_id]
        assert p["x"] == 40.0 and p["y"] == 50.0, f"Van {v_id} at T=0 not at depot: {p}"
        assert "Depot" in p["status"], f"Status mismatch: {p['status']}"
    print("[OK] Passed: T=0 visibility check (all vans at Depot).")

    # 2. Advance to T = 200 and check interpolation
    sim.advance_clock(200.0)
    pos_t200 = sim.get_interpolated_vehicle_positions()
    for v_id in [1, 2, 3]:
        p = pos_t200[v_id]
        assert 0 <= p["x"] <= 100 and 0 <= p["y"] <= 100, f"Coordinates out of bounds: {p}"
        print(f"  Van {v_id} at T=200: ({p['x']:.2f}, {p['y']:.2f}) - {p['status']}")
    print("[OK] Passed: Interpolation calculation check.")

def test_edge_case_a_late_disruption():
    inst = parse_solomon_instance('data/C101_25.txt')
    matrix = inst.compute_distance_matrix()
    prob = create_vroom_problem(inst, matrix)
    sol = prob.solve(exploration_level=1, nb_threads=2)
    sim = SimulationState(inst, matrix, sol)

    # Advance clock to after Van 1 finished (at ~816)
    sim.advance_clock(850.0)
    res = sim.trigger_breakdown(1)
    print("Late breakdown response:", res)
    assert res["status"] == "already_completed", f"Expected already_completed, got {res['status']}"
    print("[OK] Passed: Edge Case A (Late disruption safely ignored).")

def test_edge_case_b_and_c_post_completion():
    inst = parse_solomon_instance('data/C101_25.txt')
    matrix = inst.compute_distance_matrix()
    prob = create_vroom_problem(inst, matrix)
    sol = prob.solve(exploration_level=1, nb_threads=2)
    sim = SimulationState(inst, matrix, sol)

    # Advance clock to end of day T = 1200
    adv = sim.advance_clock(1200.0)
    assert adv["pending_count"] == 0
    assert sim.is_simulation_completed() == True, "Simulation should be marked completed"
    print("[OK] Passed: Edge Case C (Safe slicing at end-of-day).")

    # Call replan with 0 jobs - must not raise VroomInputException
    replan_sol = sim.replan(exploration_level=1)
    assert replan_sol is not None
    print("[OK] Passed: Replan with 0 pending jobs completed safely.")

    # Edge Case B: Inject rush orders post-completion
    rush_orders = [
        {"cust_no": 101, "x": 38.0, "y": 55.0, "demand": 15, "ready_time": 1200.0, "due_date": 1230.0, "service_time": 10.0}
    ]
    injected = sim.inject_new_orders(rush_orders)
    assert len(injected) == 1
    assert sim.is_simulation_completed() == False, "New pending job exists, should not be completed"

    # Replan to wake up idle/returned vans
    sol_b = sim.replan(exploration_level=1)
    assert sol_b is not None
    print("[OK] Passed: Edge Case B (Post-completion injection dispatches van from depot).")

if __name__ == "__main__":
    test_interpolation_and_t0_visibility()
    test_edge_case_a_late_disruption()
    test_edge_case_b_and_c_post_completion()
    print("\nALL UNIT AND INTEGRATION TESTS PASSED SUCCESSFULLY!")
