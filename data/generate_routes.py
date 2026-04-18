"""
PHASE 1: Route Generation
- Uses OSMnx to pull real Bangalore road network
- Generates 3 route types: city, highway, tunnel/low-signal
- Falls back to synthetic coords if OSMnx is unavailable
- Saves routes as JSON for downstream use
"""

import json
import numpy as np
import os
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).parent
ROUTES_FILE = DATA_DIR / "routes.json"
GRAPH_FILE  = DATA_DIR / "graph.pkl"

# ── Hardcoded fallback coords (Bangalore, real lat/lon) ─────────────────────
FALLBACK_ROUTES = {
    "city_route": {
        "description": "City center to Koramangala (dense urban, frequent stops)",
        "waypoints": [
            (12.9716, 77.5946), (12.9700, 77.5970), (12.9685, 77.5998),
            (12.9660, 77.6030), (12.9630, 77.6065), (12.9600, 77.6100),
            (12.9572, 77.6145), (12.9540, 77.6180), (12.9510, 77.6210),
            (12.9480, 77.6235), (12.9450, 77.6248), (12.9420, 77.6255),
            (12.9390, 77.6252), (12.9360, 77.6248), (12.9352, 77.6245),
        ],
        "route_type": "city",
    },
    "highway_route": {
        "description": "City center to Hebbal (NH-44, fast, good signal)",
        "waypoints": [
            (12.9716, 77.5946), (12.9760, 77.5930), (12.9800, 77.5915),
            (12.9850, 77.5900), (12.9900, 77.5888), (12.9950, 77.5880),
            (13.0000, 77.5878), (13.0100, 77.5877), (13.0200, 77.5877),
            (13.0400, 77.5877), (13.0600, 77.5877), (13.0700, 77.5877),
            (13.0827, 77.5877),
        ],
        "route_type": "highway",
    },
    "tunnel_route": {
        "description": "City to Whitefield (mixed signal, underpass + dense areas)",
        "waypoints": [
            (12.9716, 77.5946), (12.9710, 77.6100), (12.9705, 77.6250),
            (12.9700, 77.6400), (12.9695, 77.6500), (12.9690, 77.6600),
            # tunnel zone – signal drops here
            (12.9692, 77.6700), (12.9694, 77.6800), (12.9696, 77.6900),
            (12.9697, 77.7000), (12.9698, 77.7100), (12.9698, 77.7200),
            (12.9698, 77.7300), (12.9698, 77.7400), (12.9698, 77.7499),
        ],
        "route_type": "tunnel",
    },
}

# Signal quality profile per route segment (index → quality 0-1)
ROUTE_SIGNAL_PROFILES = {
    "city_route":    [0.7, 0.75, 0.6, 0.65, 0.55, 0.7, 0.8, 0.75, 0.6, 0.65, 0.7, 0.72, 0.68, 0.73, 0.75],
    "highway_route": [0.9, 0.92, 0.95, 0.96, 0.97, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90],
    "tunnel_route":  [0.75, 0.70, 0.65, 0.55, 0.45, 0.30,
                      0.15, 0.10, 0.08, 0.12, 0.20, 0.35,   # tunnel zone
                      0.55, 0.70, 0.80],
}


def try_osmnx_routes(city: str = "Bangalore, India"):
    """
    Attempt to pull real road graph from OSMnx.
    Returns (graph, routes_dict) or raises on failure.
    """
    import osmnx as ox
    import networkx as nx

    print(f"[OSMnx] Downloading road network for {city}…")
    G = ox.graph_from_place(city, network_type="drive", simplify=True)
    print(f"[OSMnx] Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    endpoints = {
        "city_route":    ((12.9716, 77.5946), (12.9352, 77.6245)),
        "highway_route": ((12.9716, 77.5946), (13.0827, 77.5877)),
        "tunnel_route":  ((12.9716, 77.5946), (12.9698, 77.7499)),
    }

    routes = {}
    for name, (orig, dest) in endpoints.items():
        orig_node = ox.nearest_nodes(G, orig[1], orig[0])
        dest_node = ox.nearest_nodes(G, dest[1], dest[0])
        path      = nx.shortest_path(G, orig_node, dest_node, weight="length")
        coords    = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
        info      = FALLBACK_ROUTES[name].copy()
        info["waypoints"] = coords
        routes[name] = info
        print(f"[OSMnx] {name}: {len(coords)} nodes")

    return G, routes


def load_or_generate_routes(use_osmnx: bool = True, force: bool = False):
    """
    Main entry point.  Returns routes dict.
    Caches to disk so you don't re-download during hackathon.
    """
    if not force and ROUTES_FILE.exists():
        print(f"[Routes] Loaded from cache: {ROUTES_FILE}")
        return json.loads(ROUTES_FILE.read_text())

    routes = None

    if use_osmnx:
        try:
            G, routes = try_osmnx_routes()
            # cache graph
            with open(GRAPH_FILE, "wb") as f:
                pickle.dump(G, f)
            print("[Routes] OSMnx routes ready.")
        except Exception as e:
            print(f"[Routes] OSMnx failed ({e}), using fallback coords.")

    if routes is None:
        routes = FALLBACK_ROUTES
        print("[Routes] Using synthetic fallback routes.")

    # Attach signal profiles
    for name, profile in ROUTE_SIGNAL_PROFILES.items():
        if name in routes:
            routes[name]["signal_profile"] = profile

    ROUTES_FILE.write_text(json.dumps(routes, indent=2))
    print(f"[Routes] Saved to {ROUTES_FILE}")
    return routes


if __name__ == "__main__":
    routes = load_or_generate_routes(use_osmnx=True)
    for name, r in routes.items():
        wps = r["waypoints"]
        print(f"  {name}: {len(wps)} waypoints | type={r['route_type']}")
