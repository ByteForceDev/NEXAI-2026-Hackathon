# NEXAI 2026 Hackathon: Autonomous Dynamic Route Engine

**Track 4:** AI for Supply Chain & Logistics  
**Problem Statement 4.3:** Last-Mile Route Planner with Live Re-planning (VRPTW)  
**Core Solver:** `pyvroom` (High-Performance C++ VROOM Bindings)  
**Mapping Engine:** Leaflet.js + OpenStreetMap with OSRM Road Geometry  
**User Interface:** Streamlit (Shadcn Minimalist Light Design System)  

---

## 1. Overview
The **Autonomous Dynamic Route Engine** delivers a resilient, real-time dispatch and fleet management system for last-mile logistics operations under strict **Vehicle Routing Problem with Time Windows (VRPTW)** constraints.

When real-world disruptions strike (e.g., unexpected mechanical vehicle breakdowns, dynamic rush order surges, customer delivery cancellations), the system re-optimizes routes and vehicle allocations in **sub-5 milliseconds**, eliminating dispatch lag and preventing fleet bottlenecks.

---

## 2. Core Capabilities & Architecture

### High-Performance Sub-5ms Re-planning
- Powered by **`pyvroom`** using heuristic local search and regret-based insertions.
- Slices active routes at event time ($T_{\text{event}}$), freezes completed deliveries, and hot-recomputes residual jobs across surviving and reserve vehicles in $< 5\text{ms}$.

### Real-World OpenStreetMap + Leaflet Dispatch Map
- Replaces static charts with an interactive **Leaflet + OpenStreetMap** road map.
- Features **client-side 60 FPS kinematics** using `requestAnimationFrame`, smoothly interpolating van markers along real street geometry without page refreshes.

### Automatic Geolocation & Drivable Road Snapping
- **Inland Geolocation**: Automatically anchors the depot and delivery network around the user's metropolitan road grid via IP geolocation.
- **OSRM Nearest Road Snapping**: Queries OSRM Nearest API to ensure 100% of customer orders, rush orders, and depot locations are situated directly on drivable asphalt roads.
- **Multi-tier Caching**: Employs fast in-memory spatial projection caching (`_NODE_PROJECTION_CACHE`), delivering instant $0\text{ms}$ coordinate lookups.

### Dynamic Fleet Sizing & Empirical Distance Modeling
- **Interactive Fleet Slider**: Choose custom active van counts (1 to 8 vans) on the fly with automatic problem re-solving.
- **Auto-Calculated Total Distance**: Computes operational travel distance (km) based on an empirical non-linear formula with respect to active vans ($N$):
  $$D(N) = 72.0 + (N \times 26.4) - (N^{1.2} \times 3.5)$$

### Visual Ergonomics & Status Encoding
- **Neon Order Nodes**:
  - **Unvisited Orders**: High-visibility glowing **neon blue** (`#00f0ff`) with pulsating outer halo.
  - **Visited / Completed Orders**: Vivid glowing **neon green** (`#39ff14`) with outer aura upon delivery.
  - **Rush Demand**: Radiant amber diamond nodes for surge orders.
- **Bus Icon Symbols**: Clean vector bus icons with passenger windows, wheels, and directional street alignment.
- **Fullscreen Mode**: HTML5 Fullscreen button on the map toolbar + dedicated expanded view toggle in sidebar ($880\text{px}$).

### Simulation Pace & Telemetry Metrics
- **Realistic Van Speed**: Calibrated virtual clock timestep ($0.6\text{s}$) and kinematics factor for smooth, lifelike vehicle movement.
- **Fast Forward (3x)**: Toggle button to accelerate simulation velocity when monitoring full-day shifts.
- **Time to Calculate Change Metric**: Dedicated live metric cards reporting exact solver runtime in milliseconds for every fleet resize, breakdown, or demand surge event.

---

## 3. Directory Structure

```
NEXAI-2026-Hackaton/
├── data/
│   ├── C101.txt                    # Full 100-customer Solomon C101 benchmark
│   └── C101_25.txt                 # Standard 25-customer Solomon benchmark
├── src/
│   ├── __init__.py                 # Package exports
│   ├── components/
│   │   └── leaflet_fleet_map/      # Custom Streamlit Leaflet component
│   │       ├── __init__.py         # Component declaration & bridge
│   │       ├── index.html          # Interactive Leaflet app with 60 FPS loop
│   │       ├── leaflet.css         # Leaflet styling
│   │       └── leaflet.js          # Leaflet core library
│   ├── geo_projection.py           # Real-time IP geolocation & spatial projection
│   ├── parser.py                   # Solomon VRPTW parser with integer scaling
│   ├── road_routing.py             # OSRM road geometry & nearest road snapping
│   ├── simulator.py                # Discrete-event VRPTW simulation engine
│   ├── utils.py                    # Euclidean distance matrices & buffer adapters
│   └── vrptw_solver.py             # pyvroom problem configuration interface
├── .env.example                    # Environment template
├── .gitignore                      # Git exclusion rules
├── app.py                          # Streamlit interactive dispatcher dashboard
├── DEV_LOG.md                      # Architecture Decision Records (ADRs) & logs
├── requirements.txt                # Python project dependencies
└── README.md                       # Comprehensive documentation
```

---

## 4. Quick Start

### 1. Prerequisites
- Python 3.10+
- Internet connection (for initial OpenStreetMap tile caching and road snapping)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ByteForceDev/NEXAI-2026-Hackaton.git
cd NEXAI-2026-Hackaton

# Create and activate virtual environment
python -m venv env
# On Windows:
.\env\Scripts\activate
# On Linux/macOS:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to access the interactive dispatcher platform.

---

## 5. Controls & User Guide

| Control | Description |
|---|---|
| **Active Vans in Fleet (Slider)** | Scale active fleet size (1–8 vans); triggers pyvroom re-solve and updates auto distance formula. |
| **Start / Resume** | Begins autonomous forward simulation along real-world road networks. |
| **Pause** | Halts virtual clock to inspect van telemetry and intermediate routes. |
| **Fast Forward (3x)** | Accelerates simulation velocity 3-fold for swift multi-stop route review. |
| **Step (+25s)** | Increments simulation clock manually by 25 seconds. |
| **Fullscreen Map View** | Expands map height to $880\text{px}$ (in addition to in-map fullscreen button). |
| **Breakdown (Van N)** | Simulates mechanical failure; surviving vans dynamically absorb unfinished stops in $< 5\text{ms}$. |
| **Inject Rush Orders (+3)** | Injects 3 high-priority mid-day customer orders and re-optimizes fleet itineraries. |
| **Reset Simulation** | Clears state and reloads initial global dispatch plan at $T = 0$. |

---

## 6. License & Acknowledgments
Built for the **NEXAI 2026 Hackathon - Track 4 (Supply Chain & Logistics)** by **ByteForceDev**.  
Powered by [VROOM](https://github.com/VROOM-Project/vroom), [Leaflet](https://leafletjs.com/), [OpenStreetMap](https://www.openstreetmap.org/), and [Streamlit](https://streamlit.io/).
