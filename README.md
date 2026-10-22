# NEXAI 2026 Hackathon: Autonomous Dynamic Route Engine

**Track 4:** AI for Supply Chain & Logistics  
**Problem Statement 4.3:** Last-Mile Route Planner with Live Re-planning (VRPTW)  
**Engine:** `pyvroom` (VROOM C++ Core) for Sub-Second Live Re-planning  

---

## Overview
This project delivers a real-time, resilient last-mile vehicle routing system with live re-planning capabilities under Vehicle Routing Problem with Time Windows (VRPTW) constraints. 

To meet the sub-5-second re-planning challenge when operational disruptions occur (traffic bottlenecks, customer cancellations, new priority stops), this engine leverages **`pyvroom`** to solve 25–100 node VRPTW problems in tens of milliseconds.

## Key Features (Step 1 Baseline)
- **Standard Solomon VRPTW Ingestion:** Built-in parser for Solomon VRPTW benchmark instances (e.g., `C101`, `R101`, `RC101`).
- **Mathematical Integer Discretization:** Scales continuous coordinates, time windows, and service times by a factor of 100 to ensure strict integer arithmetic compliance for VROOM while preserving spatial fidelity.
- **Euclidean Distance & Duration Matrix:** High-performance pairwise matrix calculation using NumPy.
- **Cross-Platform C-Buffer Alignment:** Native memory alignment (`np.uintc`) resolving NumPy 2.x / Windows MSVC buffer compatibility with pybind11.
- **VROOM Problem Formulation:** Decoupled problem builder preparing vehicles, depot horizons, and customer jobs.

## Directory Structure
```
NEXAI-2026-Hackathon/
├── data/
│   ├── C101.txt            # Full 100-customer Solomon C101 benchmark
│   └── C101_25.txt         # Standard 25-customer Solomon benchmark
├── src/
│   ├── __init__.py         # Core package exports
│   ├── parser.py           # Solomon VRPTW data parser with factor-100 scaling
│   ├── utils.py            # Integer Euclidean distance matrix & memory adapters
│   ├── vrptw_solver.py     # pyvroom problem configuration
│   ├── simulator.py        # Event simulator (Step 2)
│   └── ...
├── .env.example            # Environment variables template
├── .gitignore              # Repository exclusion rules
├── DEV_LOG.md              # Architecture decisions (ADRs) & experiment log
├── requirements.txt        # Core dependencies
└── README.md               # Project documentation
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Data Ingestion & Solver
```python
from src.parser import parse_solomon_instance
from src.vrptw_solver import create_vroom_problem

# Parse standard 25-customer Solomon instance
instance = parse_solomon_instance("data/C101_25.txt")
distance_matrix = instance.compute_distance_matrix()

# Build VROOM problem instance
problem = create_vroom_problem(instance, distance_matrix)
print(f"Loaded {len(problem.vehicles)} vehicles and {len(problem.jobs)} jobs.")
```
