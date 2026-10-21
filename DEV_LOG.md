# NEXAI 2026 Hackathon - Development Log & Engineering Record

**Track 4:** AI for Supply Chain & Logistics  
**Problem Statement 4.3:** Last-Mile Route Planner with Live Re-planning (VRPTW)  
**Team / Project:** ByteForceDev / Autonomous Dynamic Route Engine  

---

## 1. Architecture Decisions (ADRs)

### ADR 001: Optimization Engine Selection — `pyvroom` vs. Google OR-Tools
* **Status:** Accepted
* **Context:**
  Problem Statement 4.3 requires dynamic, live re-planning for last-mile fleet operations when real-time disruptions occur (e.g., dynamic order cancellations, traffic bottlenecks, customer delivery window updates, new rush pickups). Under operational constraints, re-optimization must be executed in near real-time (sub-5-second target; ideally sub-second for interactive dispatch dashboards).
  
  Traditional constraint-programming / MILP solvers such as Google OR-Tools Routing Library rely on CP-SAT or local search metaheuristics. While OR-Tools is flexible and feature-rich, initializing search models and converging on tightly constrained VRPTW instances can require 10 to 60+ seconds per re-plan call, causing perceptible UI freezing and lagging behind rapidly moving fleet state.
* **Decision:**
  Adopt **`pyvroom`** (the high-performance C++ VROOM engine Python binding) as the core routing and optimization engine.
* **Rationale & Trade-offs:**
  1. **Algorithmic Efficiency:** VROOM is purpose-built for vehicle routing heuristics (utilizing sophisticated regret-based insertion, tabu search, and fast local search operators), achieving solutions within 50ms–500ms for 25–100 node instances.
  2. **Sub-second Re-planning:** Enables seamless hot-recomputations as delivery events arrive without blocking the interactive Streamlit / event simulator loops.
  3. **Strict Academic Compliance:** Preserves capacity, multi-dimensional amounts, depot constraints, and strict customer time windows while maximizing fleet utilization.
  4. **Trade-off:** VROOM requires integer costs, durations, and coordinates, demanding a formal mathematical scaling factor across spatial and temporal dimensions (see ADR 002).

---

### ADR 002: Spatial and Temporal Scaling (Discretization Factor = 100)
* **Status:** Accepted
* **Context:**
  The underlying C++ VROOM engine operates strictly on integer arithmetic (`uint32_t`) for duration matrices, coordinates, time windows, and service durations to eliminate floating-point non-determinism, rounding divergence, and computational overhead. Standard benchmark instances (such as Solomon VRPTW) use decimal coordinates and fractional time intervals.
* **Decision:**
  Implement a deterministic integer scaling factor of `SCALE_FACTOR = 100` across all spatial coordinates ($X, Y$), customer time windows ($[e_i, l_i]$), and service durations ($s_i$).
* **Mathematical Invariant:**
  - Scaled Coordinate: $X_{\text{scaled}} = \lfloor X \times 100 + 0.5 \rfloor$
  - Scaled Distance / Duration: $D_{ij,\text{scaled}} = \lfloor \sqrt{(X_{i,\text{scaled}} - X_{j,\text{scaled}})^2 + (Y_{i,\text{scaled}} - Y_{j,\text{scaled}})^2} + 0.5 \rfloor$
  - Since Solomon assumes a vehicle speed of $v = 1.0 \text{ distance unit / time unit}$, $100 \times \text{distance}$ maps 1:1 to $100 \times \text{travel time}$, maintaining exact physical consistency across routing and scheduling.

---

### ADR 003: NumPy 2.x and Pybind11 Windows Memory Buffer Alignment
* **Status:** Accepted
* **Context:**
  On 64-bit Windows environments running NumPy 2.x, default `numpy.uint32` arrays expose a type character `'L'` (`unsigned long`), whereas pybind11 C++ bindings compiled with MSVC expect unsigned integer buffer type `'I'` (`unsigned int`). Direct instantiation of `vroom.Input.set_durations_matrix(numpy.asarray(..., dtype="uint32"))` triggers a `RuntimeError: Incompatible buffer format!`.
* **Decision:**
  Encapsulate matrix generation in `src/utils.py` and `src/vrptw_solver.py` using `numpy.uintc` / `'I'` and construct explicit `vroom._vroom.Matrix` wrapper objects, ensuring zero-copy memory compatibility across Windows, Linux, and macOS runtimes.

---

### ADR 004: Clean Separation of Domain Models, Parsers, and Solvers
* **Status:** Accepted
* **Context:**
  Downstream components (event simulator, live visualizer, interactive dashboard) require rich access to both raw human-readable telemetry (unscaled floats) and solver-ready tensors (scaled integers).
* **Decision:**
  Structure data parsing into clean typed dictionaries / domain records containing both raw metrics and scaled integers, enabling visualizers to render physical units directly while the solver executes on scaled integers.

---

### ADR 005: State-Freezing & Temporal Route Slicing for Dynamic VRPTW Re-planning
* **Status:** Accepted
* **Context:**
  When real-time operational disruptions occur during daily dispatch (e.g., vehicle mechanical breakdowns, traffic snarls, rush customer orders injected at $T_{\text{event}}$), the global routing plan cannot be recomputed from scratch ($T = 0$). Past physical actions are irreversible: packages already handed over to customers cannot be retrieved, and drivers currently en route cannot teleport back to the central depot.
* **Decision:**
  Implement a temporal slicing and state-freezing strategy within `SimulationState`:
  1. **Temporal Horizon Partitioning:** For each route in the active solution, evaluate each step's physical completion boundary: $T_{\text{complete}} = \text{arrival} + \text{waiting\_time} + \text{service}$.
     - **Completed Deliveries:** Any stop where $\text{arrival} + \text{service} \le T_{\text{event}}$ is permanently frozen as completed.
     - **Pending Stops:** Stops scheduled at $> T_{\text{event}}$ are released back into the shared `pending_jobs` pool.
  2. **Vehicle Mid-Route Kinematics:** Each active vehicle's state is dynamically translated:
     - Its new departure location index is set to the spatial node of its *last completed delivery* (or the central depot if no stops were finished).
     - Its remaining capacity is updated: $C_{\text{remaining}} = C_{\text{initial}} - \sum_{j \in \text{completed}} d_j$.
     - Its operational start horizon is bounded by $\max(T_{\text{event}}, T_{\text{complete, last\_job}})$, ensuring causal feasibility without backwards time travel.
     - Its destination step remains pinned to the central depot to guarantee safe vehicle return.
  3. **Fleet Disruption Handling:**
     - **Breakdown:** The disabled vehicle is barred from future dispatches (`is_broken = True`). Its unfinished stops in the pending pool are dynamically absorbed and re-routed across surviving vehicles or available depot reserves.
     - **Order Injection:** Dynamically expands the integer Euclidean distance matrix to shape $(N+K, N+K)$ and registers fresh `vroom.Job` instances in the pending pool.
  4. **Sub-second Re-optimization:** Re-invoking `pyvroom.Input.solve()` on the residual state completes in under 5 milliseconds, enabling real-time interactive dispatch without blocking user interactions.

---

## 2. Experiments & Benchmarks

| Experiment ID | Date | Instance | Nodes | Engine | Exploration Level | Threads | Runtime | Solution Status |
|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-10-21 | Synthetic 2-node | 2 | pyvroom 1.15.2 | 1 | 1 | < 5 ms | Verified (Cost: 1000) |
| EXP-002 | 2026-10-21 | Solomon C101-25 | 26 | pyvroom 1.15.2 | 1 | 2 | ~80 ms | Verified (Cost: 19183, 3 Routes) |
| EXP-003 | 2026-10-21 | C101-25 Disruption (T=300, breakdown + new order) | 27 | pyvroom 1.15.2 | 1 | 2 | ~3.5 ms | Verified (Cost: 15430, 3 Active Routes) |

---

## 3. AI Assistance Log

| Step / Prompt | AI Role & Scope | Human Review & Verification |
|---|---|---|
| Step 1 Scaffold & Setup | Architectural setup: designed repository hierarchy, configured dependencies (`pyvroom`, `streamlit`, `plotly`), established `DEV_LOG.md` and `.gitignore`. | Inspected dependency compatibility and Windows environment prerequisites. |
| Memory Buffer Diagnostics | Identified NumPy 2.x / Windows MSVC buffer signature mismatch (`'L'` vs `'I'`) in `pyvroom` and implemented `np.uintc` buffer translation. | Verified resolution via direct C++ matrix binding tests. |
| Solomon Parser & Solver | Authored parsing algorithm for Solomon VRPTW format, scaling pipeline, Euclidean integer distance matrix calculator, and VROOM input problem generator. | Validated coordinate scaling and data integrity against standard Solomon C101 specifications. |
| Step 2 Simulation Engine & Live Re-planning | Designed `SimulationState` manager, mid-route route slicing at $T_{\text{event}}$, vehicle kinematic translation, dynamic matrix expansion for new orders, vehicle breakdown reassignment, and sub-5ms re-planning. | Executed full automated integration verification suite testing simultaneous vehicle breakdown and order injection. |

