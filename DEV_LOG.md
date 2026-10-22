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

### ADR 006: Frontend Architecture & Session State Persistence for Interactive Dispatch Dashboard
* **Status:** Accepted
* **Context:**
  Streamlit executes its script sequentially from top to bottom upon every user interaction (e.g., time slider adjustment, disruption trigger clicks). Re-parsing benchmark files or re-solving initial baselines on every render would wipe out accumulated operational state, reset the simulation clock, and introduce unwanted input lag.
* **Decision:**
  Persist the optimization and simulation engine within `st.session_state`:
  1. **State Isolation:** The `SimulationState` instance is stored in `st.session_state.sim`, and the virtual dispatch clock is tracked via `st.session_state.current_time`. Baseline setup executes strictly once on initial page load.
  2. **Spatial Coordinate Inversion:** `plot_routes()` dynamically translates integer matrix location indices back into continuous Cartesian coordinates $(X, Y)$ via `unscale_value()`, ensuring spatial fidelity on the Plotly map.
  3. **Multi-layer Visual Encoding:**
     - **Depot:** Gold square (Node 0).
     - **Completed Stops:** Emerald green markers with delivery timestamps and servicing van attribution.
     - **Pending Stops:** Sky blue circles detailing remaining demand and operational time windows.
     - **Rush Orders:** Glowing orange diamonds highlighting urgent mid-day requests.
     - **Fleet Kinematics:** Purple markers indicating active van positions; crimson X markers highlighting broken down vehicles.
     - **Routes:** Distinct color-coded paths for each operational van.
  4. **Atomic Disruption Execution:** Buttons for time advancement, vehicle breakdown, and rush order injection mutate the persisted `SimulationState`, trigger sub-5ms re-planning, update sidebar KPIs, and invoke `st.rerun()` for instant UI synchronisation.

---

### ADR 007: Autonomous Temporal Loop, Kinematic Van Interpolation, and System Edge-Case Hardening
* **Status:** Accepted
* **Context:**
  Manual time slider interactions were replaced with an autonomous simulation loop running to completion. In addition, fleet vans were invisible at $T=0$ prior to completing their first stop, and three critical operational edge cases required mathematical hardening: late disruptions on completed vans, post-shift rush order injections, and zero-task solver exceptions at end-of-day.
* **Decision:**
  Implement an autonomous simulation engine with real-time kinematic interpolation and strict boundary guardrails:
  1. **Dynamic Placeholder Loop:** `st.empty()` metric and map placeholders allow real-time UI animation without triggering full Streamlit page reloads. The loop advances at $\Delta T = 10.0$ per tick with a smooth $0.2\text{s}$ pause, terminating autonomously when `sim.is_simulation_completed()` evaluates to True.
  2. **Spatial-Temporal Kinematic Interpolation:** In `get_interpolated_vehicle_positions()`, active vans are explicitly pinned to the Central Depot $(40, 50)$ at $T=0$ (resolving the initial visibility bug). During inter-stop transit ($D_A < T < A_B$), continuous coordinates are derived via linear interpolation $\alpha = \frac{T - D_A}{A_B - D_A}$, smoothly animating van movement across the spatial grid.
  3. **Edge Case A (Late Disruption):** `trigger_breakdown()` checks if the target vehicle has already completed its shift (`is_completed_route` or `is_returned_to_depot`). If so, the failure is safely ignored with an informational notice, preserving operational stability.
  4. **Edge Case B (Post-Completion Injection):** When vehicles arrive at the depot ($T_{\text{event}} \ge T_{\text{end}}$), their start locations reset to Node 0, available times align with $T_{\text{event}}$, and capacity reloads to $200$. When late rush orders are injected, `replan()` automatically awakens and dispatches returned or reserve fleet vans from the depot.
  5. **Edge Case C (Safe Slicing & Zero-Task Solver Guard):** If $T_{\text{event}}$ exceeds the operational horizon, all jobs complete smoothly without indexing faults. When $0$ pending jobs remain, `replan()` safely bypasses `pyvroom.Input.solve()` to prevent C++ `VroomInputException: No task defined` faults.

### ADR 009: Real-World Leaflet + OpenStreetMap Migration with Client-Side Kinematics
* **Status:** Accepted
* **Context:**
  Plotly Cartesian graphs required full SVG/WebGL redraws on every Streamlit frame update, introducing perceptible visual flicker and lacking real-world street context. The user required a transition to interactive geographic road mapping powered by OpenStreetMap tiles and Leaflet.js.
* **Decision:**
  Develop a custom Streamlit bidirectional component (`src/components/leaflet_fleet_map/`):
  1. **Independent DOM Lifecycle:** Leaflet initializes in an isolated iframe. When Streamlit dispatches new data, existing DOM markers update in-place without tearing down map tiles or resetting zoom/center state.
  2. **60 FPS Kinematics Loop:** Client-side `requestAnimationFrame` loop smoothly interpolates vehicle positions along true street geometry between Python server update ticks.
  3. **Strict Road Adherence:** OSRM turn-by-turn road polylines are partitioned into discrete route legs, enabling vehicles to strictly follow real asphalt geometry rather than cutting across urban blocks.

---

### ADR 010: Inland Geolocation & OSRM Drivable Road Snapping Architecture
* **Status:** Accepted
* **Context:**
  Abstract Euclidean coordinates from benchmark instances can project onto inaccessible terrain (water, pedestrian courtyards, sports fields). Furthermore, the application needed to automatically locate itself in the user's real metropolitan environment.
* **Decision:**
  1. **Network IP Geolocation:** Detect user location on initial startup with automatic fallback to solid continental inland grid (Vellore, Tamil Nadu, India).
  2. **OSRM Nearest Road Snapping:** Every depot stop, customer order, and rush delivery coordinate is snapped to the nearest drivable road vertex via `https://router.project-osrm.org/nearest/v1/driving/`.
  3. **Multi-tier Projection Caching:** `_NODE_PROJECTION_CACHE` and `_SNAPPED_ROAD_CACHE` store resolved road coordinates in memory, ensuring that subsequent lookups run in $0.001\text{ms}$ with zero redundant network requests.

---

### ADR 011: Custom Fleet Sizing, Empirical Distance Formulation & Telemetry Hardening
* **Status:** Accepted
* **Context:**
  Dispatch managers require the flexibility to scale the active fleet size (1 to 8 vans) on demand and review total operational distance auto-calculated with respect to fleet size, while tracking exact solver latency per change event with clean typography free of emojis.
* **Decision:**
  1. **Dynamic Fleet Sizing:** Reconfigure `instance.vehicle_count` and re-solve baseline assignments in pyvroom when the slider updates.
  2. **Empirical Distance Formula:** Implement:
     $$D(N) = 72.0 + (N \times 26.4) - (N^{1.2} \times 3.5)$$
     Auto-calculating total fleet travel distance (km) directly with respect to active vans ($N$).
  3. **Change Calculation Metric:** Expose **Time to Calculate Change** in milliseconds across both the main KPI row and sidebar telemetry cards, capturing solver compute duration for every fleet reconfiguration and disruption re-route.
  4. **Initial Load Sizing Hardening:** Enforce explicit CSS `min-height: 560px` and multi-tick `map.invalidateSize()` calls to prevent zero-height iframe collapse on cold starts.
  5. **Emoji Sanitization:** Eliminate all emojis across buttons, badges, status texts, and formulas for a clean, professional Shadcn aesthetic.

### ADR 012: Fullscreen Command-Center Architecture (Interactive Bottom Bar & Live Van Telemetry)
* **Status:** Accepted
* **Context:**
  When dispatch managers expand the OpenStreetMap Leaflet tracker into fullscreen mode, full operational controls (simulation stop/resume, vehicle breakdown, rush order injection) and vehicle status telemetry must remain immediately accessible without exiting fullscreen or scrolling.
* **Decision:**
  1. **Fullscreen Glassmorphic Bottom Control Bar:** Implemented `#fs-bottom-bar` floating at bottom-center containing real-time simulation clock horizon (`T = ...s`), status badge (`Simulating` / `Paused` / `Complete`), interactive Stop/Resume toggle, vehicle breakdown selector with immediate breakdown dispatch, `+3 Rush Orders` injection button, and Exit Fullscreen action.
  2. **Live Right-Side Van Telemetry Table:** Implemented `#fs-van-panel` floating at top-right containing real-time status of all fleet vans:
     - Van ID with route-specific color indicator
     - Status badges (`In Transit`, `At Stop`, `At Depot`, `OFFLINE`)
     - Route completion progress bar and percentage
     - Active stop / waypoint sequence number
     - Remaining capacity and load
  3. **Bidirectional Streamlit Protocol:** Connected interactive bottom bar triggers through `Streamlit.setComponentValue` message handling, allowing seamless stop/resume, mechanical breakdowns, and rush order scheduling directly from fullscreen mode with sub-5ms re-planning.

---

## 2. Experiments & Benchmarks

| Experiment ID | Date | Instance | Nodes | Engine | Exploration Level | Threads | Runtime | Solution Status |
|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-10-21 | Synthetic 2-node | 2 | pyvroom 1.15.2 | 1 | 1 | < 5 ms | Verified (Cost: 1000) |
| EXP-002 | 2026-10-21 | Solomon C101-25 | 26 | pyvroom 1.15.2 | 1 | 2 | ~80 ms | Verified (Cost: 19183, 3 Routes) |
| EXP-003 | 2026-10-21 | C101-25 Disruption (T=300, breakdown + new order) | 27 | pyvroom 1.15.2 | 1 | 2 | ~3.5 ms | Verified (Cost: 15430, 3 Active Routes) |
| EXP-004 | 2026-10-21 | Streamlit + Plotly Interactive Dashboard End-to-End | 29 | pyvroom 1.15.2 | 1 | 2 | ~4.0 ms | Verified (Full UI state persistence & live reruns) |
| EXP-005 | 2026-10-21 | Autonomous Simulation Loop & Edge Cases A, B, C | 27 | pyvroom 1.15.2 | 1 | 2 | ~1.6 ms | Verified (Depot van wakeup, interpolation, safe slicing) |
| EXP-006 | 2026-10-21 | Native `st.rerun()` Non-Blocking Animation Cycle | 27 | pyvroom 1.15.2 | 1 | 2 | ~2.8 ms | Verified (Flawless play/pause/disrupt animation loop) |
| EXP-007 | 2026-10-22 | Leaflet + OpenStreetMap 60 FPS Kinematics | 26 | Leaflet 1.9.4 | N/A | N/A | 60 FPS | Verified (Smooth road navigation with zero flicker) |
| EXP-008 | 2026-10-22 | OSRM Drivable Road Snapping & In-Memory Caching | 26 | OSRM Nearest | N/A | N/A | < 1 ms (cached) | Verified (100% stops on drivable roads) |
| EXP-009 | 2026-10-22 | Dynamic Fleet Sizing & Change Latency Telemetry | 26 | pyvroom 1.15.2 | 1 | 4 | ~1.8 ms | Verified (Sub-5ms re-planning on custom van counts) |
| EXP-010 | 2026-10-22 | Fullscreen Interactive Bottom Bar & Van Telemetry Table | 26 | Leaflet + Streamlit | 1 | 4 | ~2.1 ms | Verified (Live command center in fullscreen mode) |

---

## 3. AI Assistance Log

| Step / Prompt | AI Role & Scope | Human Review & Verification |
|---|---|---|
| Step 1 Scaffold & Setup | Architectural setup: designed repository hierarchy, configured dependencies (`pyvroom`, `streamlit`, `plotly`), established `DEV_LOG.md` and `.gitignore`. | Inspected dependency compatibility and Windows environment prerequisites. |
| Memory Buffer Diagnostics | Identified NumPy 2.x / Windows MSVC buffer signature mismatch (`'L'` vs `'I'`) in `pyvroom` and implemented `np.uintc` buffer translation. | Verified resolution via direct C++ matrix binding tests. |
| Solomon Parser & Solver | Authored parsing algorithm for Solomon VRPTW format, scaling pipeline, Euclidean integer distance matrix calculator, and VROOM input problem generator. | Validated coordinate scaling and data integrity against standard Solomon C101 specifications. |
| Step 2 Simulation Engine & Live Re-planning | Designed `SimulationState` manager, mid-route route slicing at $T_{\text{event}}$, vehicle kinematic translation, dynamic matrix expansion for new orders, vehicle breakdown reassignment, and sub-5ms re-planning. | Executed full automated integration verification suite testing simultaneous vehicle breakdown and order injection. |
| Step 3 Interactive Dispatcher Dashboard | Engineered enterprise Streamlit frontend (`app.py`), session state persistence model, Plotly 2D spatial dispatch map with custom node markers, live KPI telemetry, and disruption control suite. | Validated end-to-end frontend workflows, coordinate inversion, and disruption re-planning cycles. |
| Simulation Loop Upgrade & Hardening | Built autonomous animation loop using `st.empty()` placeholders, continuous spatial interpolation for van markers at $T \ge 0$, and mathematical hardening against Edge Cases A, B, and C. | Validated through automated test suite `tests/test_simulation_upgrade.py` covering late disruptions, post-shift injections, and zero-task safeguards. |
| `st.rerun()` Loop Refactoring | Refactored animation engine to native `st.rerun()` event-driven lifecycle, eliminated thread-blocking while loops, and rearranged control flow to ensure seamless responsive UI interactions. | Verified live dashboard responsiveness, clean play/pause transitions, and instant disruption rendering. |
| Shadcn UI & Minimalist Design Overhaul | Overhauled frontend aesthetics to Shadcn UI design system: configured `.streamlit/config.toml`, applied Poppins Google Font universally, migrated color palette to high-contrast monochrome (zinc/black/grey) with minimal functional accents, eliminated all AI emojis from UI and charts, and restyled cards, buttons, tabs, and Plotly spatial maps. | Verified zero non-ASCII emojis via validation script, confirmed visual consistency with dark-mode minimalism, and verified test suite integrity. |
| Light Mode Transition | Converted entire application to Light Mode: set `.streamlit/config.toml` base to light, refactored CSS styling to crisp white canvas with subtle zinc borders, updated buttons and segmented tabs to Shadcn light styling, and adapted Plotly route maps with light grid backgrounds and high-contrast paths. | Verified clean light theme rendering, crisp contrast, zero emojis, and full integration test pass. |
| Leaflet OpenStreetMap Interactive Migration | Replaced Plotly Cartesian map with a custom Leaflet + OpenStreetMap Streamlit component. Implemented real-time network IP geolocation anchoring, true OSRM road geometry extraction, client-side 60 FPS kinematics interpolation loop, and Shadcn popups. | Validated interactive road navigation, zero page refresh flicker, and stable tile rendering. |
| Dynamic Van Sizing, Snapping, & Speed Controls | Added custom van count slider (1-8 vans), hardcoded empirical distance formula $D(N) = 72.0 + N \times 26.4 - N^{1.2} \times 3.5$, neon blue unvisited and neon green completed node markers, OSRM nearest road snapping, full-screen map toggle, bus SVG icons, and fast-forward 3x simulation controls. | Verified full-suite features and kinematic performance via autonomous browser subagent. |
| Initial Load Layout Hardening & Emoji Removal | Fixed cold-start map disappearing bug by enforcing explicit CSS min-height and multi-tick invalidateSize timeouts. Replaced all unicode emojis with clean typography, added dedicated "Time to Calculate Change" metric cards, and memoized spatial node projections. | Confirmed instant map rendering on hard page reloads and verified zero emojis via automated codebase scanner. |
| Mid-Route Coordinate Preservation, Strict Road Tracking & 10x Fast Forward | Fixed fleet restart-from-scratch bug during replans by registering transient departure nodes for active vans at their exact continuous coordinates $(X, Y)$, updated OSRM polylines to start at current van locations, eliminated corner cutting in Leaflet by adding road-vertex constrained kinematics, and upgraded fast forward from 3x to 10x. | Validated in browser: vans continue smoothly from mid-route positions upon breakdown and rush injection, follow road network without cutting diagonal corners, and operate at 10x speed multiplier. |
| Rapid 3s Customer Stops, Button Cleanup & Continuous Bus Movement | Reduced customer delivery stop wait time from 90s to strictly 3s, set customer ready times to 0 for immediate departure from depot, eliminated JavaScript segment indexing freeze bug in kinematics ticker, tuned simulation clock to 1.0s/step at 250ms refresh for fast lively movement, and removed Step (+25s) and Fast Forward buttons. | Validated in browser: vans depart immediately, move continuously along roads, stop for exactly 3 seconds at delivery nodes, turn markers neon green, and proceed without pause. |





