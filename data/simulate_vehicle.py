"""
PHASE 2: Vehicle Movement Simulation
- Interpolates routes → smooth per-second positions
- Adds realistic speed variation (city braking, highway cruise)
- Inserts traffic stops at probabilistic intervals
- Output: list of timestep dicts ready for signal overlay
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR   = Path(__file__).parent
ROUTES_FILE = DATA_DIR / "routes.json"
VEHICLE_FILE = DATA_DIR / "vehicle_data.json"

# ── Speed profiles (m/s) ────────────────────────────────────────────────────
SPEED_PROFILES = {
    "city":    {"mean": 8.0,  "std": 3.0, "stop_prob": 0.08, "stop_dur": (5, 30)},
    "highway": {"mean": 22.0, "std": 4.0, "stop_prob": 0.01, "stop_dur": (2, 8)},
    "tunnel":  {"mean": 12.0, "std": 5.0, "stop_prob": 0.05, "stop_dur": (3, 20)},
    "mixed":   {"mean": 14.0, "std": 7.0, "stop_prob": 0.04, "stop_dur": (3, 18)},
}

EARTH_RADIUS_M = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two GPS points."""
    r   = EARTH_RADIUS_M
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a     = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))


def interpolate_waypoints(waypoints: List, total_points: int = 500) -> np.ndarray:
    """
    Cubic-spline-style interpolation of waypoints → dense coordinate array.
    Returns shape (N, 2) array of (lat, lon).
    """
    from scipy.interpolate import interp1d

    lats = [w[0] for w in waypoints]
    lons = [w[1] for w in waypoints]
    t    = np.linspace(0, 1, len(waypoints))
    t_new = np.linspace(0, 1, total_points)

    f_lat = interp1d(t, lats, kind="cubic")
    f_lon = interp1d(t, lons, kind="cubic")

    return np.column_stack([f_lat(t_new), f_lon(t_new)])


def simulate_vehicle(route_name: str, route_data: Dict, dt: float = 1.0,
                     seed: int = 42) -> List[Dict[str, Any]]:
    """
    Simulate vehicle movement along a route.

    Args:
        route_name: "city_route" | "highway_route" | "tunnel_route"
        route_data: dict with 'waypoints', 'route_type'
        dt: time step in seconds
        seed: RNG seed for reproducibility

    Returns:
        List of timestep dicts with: t, lat, lon, speed, heading, stopped
    """
    rng          = np.random.default_rng(seed)
    route_type   = route_data["route_type"]
    profile      = SPEED_PROFILES[route_type]
    waypoints    = route_data["waypoints"]

    # Dense interpolated path
    coords       = interpolate_waypoints(waypoints, total_points=300)
    n            = len(coords)

    # Pre-compute segment distances
    seg_dist     = np.array([
        haversine(coords[i,0], coords[i,1], coords[i+1,0], coords[i+1,1])
        for i in range(n-1)
    ])
    total_dist   = seg_dist.sum()

    # ── Simulate traversal ─────────────────────────────────────────────────
    timesteps    = []
    t            = 0.0
    pos          = 0.0          # distance covered (m)
    seg_idx      = 0
    seg_covered  = 0.0
    stop_timer   = 0            # seconds remaining in current stop

    # Safety bound in case route data is malformed or advancement stalls.
    max_steps = max(10_000, int(total_dist / max(dt, 1e-6)) * 10)

    while seg_idx < n - 1:
        lat = coords[seg_idx, 0]
        lon = coords[seg_idx, 1]

        # Heading (degrees)
        dlat = coords[seg_idx+1, 0] - lat
        dlon = coords[seg_idx+1, 1] - lon
        heading = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360

        # Stopped?
        if stop_timer > 0:
            speed      = 0.0
            stop_timer -= dt
            stopped    = True
        else:
            stopped = False
            # Random stop event
            if rng.random() < profile["stop_prob"] * dt:
                stop_dur   = rng.integers(*profile["stop_dur"])
                stop_timer = int(stop_dur)
                speed      = 0.0
                stopped    = True
            else:
                # Speed with Gaussian noise, clipped to [1, max]
                max_spd = profile["mean"] + 2 * profile["std"]
                speed   = float(np.clip(
                    rng.normal(profile["mean"], profile["std"]),
                    1.0, max_spd
                ))

        timesteps.append({
            "t":        round(t, 1),
            "lat":      round(float(lat),  6),
            "lon":      round(float(lon),  6),
            "speed":    round(float(speed), 2),   # m/s
            "speed_kmh": round(float(speed) * 3.6, 1),
            "heading":  round(float(heading), 1),
            "stopped":  stopped,
            "seg_idx":  int(seg_idx),
            "route":    route_name,
        })

        # Advance position
        if not stopped:
            dist_step   = speed * dt
            seg_covered += dist_step
            # Move across as many segments as covered, including the final one.
            while seg_idx < n - 1 and seg_covered >= seg_dist[seg_idx]:
                seg_covered -= seg_dist[seg_idx]
                seg_idx     += 1

        t += dt

        if len(timesteps) >= max_steps:
            print(f"[Vehicle] Warning: step cap reached on {route_name}; "
                  "stopping early to avoid runaway memory usage.")
            break

        if seg_idx >= n - 1:
            break

    print(f"[Vehicle] {route_name}: {len(timesteps)} timesteps, "
          f"{total_dist/1000:.1f} km, "
          f"{t/60:.1f} min simulated")
    return timesteps


def simulate_all_routes(routes: Dict) -> Dict[str, List[Dict]]:
    """Simulate all routes. Returns {route_name: [timesteps]}"""
    all_data = {}
    for i, (name, data) in enumerate(routes.items()):
        all_data[name] = simulate_vehicle(name, data, seed=42 + i)
    return all_data


if __name__ == "__main__":
    routes = json.loads(ROUTES_FILE.read_text())
    all_data = simulate_all_routes(routes)

    # Save
    VEHICLE_FILE.write_text(json.dumps(all_data, indent=2))
    print(f"\n[Vehicle] Saved to {VEHICLE_FILE}")

    for name, steps in all_data.items():
        speeds = [s["speed_kmh"] for s in steps if not s["stopped"]]
        stops  = sum(1 for s in steps if s["stopped"])
        print(f"  {name}: avg speed {np.mean(speeds):.1f} km/h | "
              f"stopped {stops}/{len(steps)} steps")
