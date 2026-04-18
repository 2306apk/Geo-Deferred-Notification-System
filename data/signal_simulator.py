"""
PHASE 3: Signal Simulation
- Synthetic cell towers in OpenCelliD format (pre-generated for Bangalore)
- Signal = f(distance to nearest tower, frequency band)
- Adds realistic noise: Gaussian + sudden drops (tunnels, underpasses)
- Output: raw RSSI per timestep (dBm scale)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR    = Path(__file__).parent
SIGNAL_FILE = DATA_DIR / "signal_data.json"
TOWER_FILE  = DATA_DIR / "towers.json"

# ── Synthetic tower generation ───────────────────────────────────────────────

BANGALORE_BOUNDS = {
    "lat_min": 12.85, "lat_max": 13.15,
    "lon_min": 77.45, "lon_max": 77.80,
}

BANDS = {
    "4G": {"freq_mhz": 1800, "range_m": 3000, "power_dbm": 43},
    "3G": {"freq_mhz":  900, "range_m": 5000, "power_dbm": 40},
    "2G": {"freq_mhz":  900, "range_m": 8000, "power_dbm": 38},
}

# Dead zones: (lat_center, lon_center, radius_m)
# Simulates tunnels, dense buildings, industrial zones
DEAD_ZONES = [
    (12.9692, 77.6800, 400),   # tunnel zone on tunnel_route
    (12.9695, 77.6900, 350),
    (12.9694, 77.7000, 300),
    (12.9660, 77.6030, 200),   # underpass on city_route
]


def generate_towers(n: int = 300, seed: int = 0) -> List[Dict]:
    """
    Generate synthetic cell towers covering Bangalore.
    Denser near city center, sparser on outskirts — realistic.
    """
    rng    = np.random.default_rng(seed)
    towers = []

    # City center cluster
    for _ in range(n // 2):
        lat  = rng.normal(12.97, 0.05)
        lon  = rng.normal(77.59, 0.05)
        band = rng.choice(["4G", "3G"], p=[0.7, 0.3])
        towers.append(_make_tower(lat, lon, band, len(towers)))

    # Spread across bounds
    for _ in range(n // 2):
        lat  = rng.uniform(BANGALORE_BOUNDS["lat_min"], BANGALORE_BOUNDS["lat_max"])
        lon  = rng.uniform(BANGALORE_BOUNDS["lon_min"], BANGALORE_BOUNDS["lon_max"])
        band = rng.choice(["4G", "3G", "2G"], p=[0.5, 0.3, 0.2])
        towers.append(_make_tower(lat, lon, band, len(towers)))

    return towers


def _make_tower(lat: float, lon: float, band: str, tid: int) -> Dict:
    b = BANDS[band]
    return {
        "id":       tid,
        "lat":      round(lat, 6),
        "lon":      round(lon, 6),
        "band":     band,
        "range_m":  b["range_m"],
        "power_dbm": b["power_dbm"],
    }


# ── Signal model ─────────────────────────────────────────────────────────────

EARTH_R = 6_371_000.0


def haversine_np(lat1, lon1, lat2_arr, lon2_arr) -> np.ndarray:
    phi1  = np.radians(lat1)
    phi2  = np.radians(lat2_arr)
    dphi  = phi2 - phi1
    dlam  = np.radians(lon2_arr - lon1)
    a     = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def free_space_path_loss(distance_m: float, freq_mhz: float) -> float:
    """FSPL in dB."""
    if distance_m < 1:
        distance_m = 1
    return 20*np.log10(distance_m) + 20*np.log10(freq_mhz) - 27.55


def compute_rssi(lat: float, lon: float,
                 tower_lats: np.ndarray, tower_lons: np.ndarray,
                 tower_powers: np.ndarray, tower_freqs: np.ndarray,
                 tower_ranges: np.ndarray) -> float:
    """
    Compute received signal strength (dBm) at (lat, lon).
    Considers top-3 nearest in-range towers, picks best.
    """
    dists   = haversine_np(lat, lon, tower_lats, tower_lons)
    in_range = dists <= tower_ranges

    if not in_range.any():
        return -120.0  # no service

    d   = dists[in_range]
    pw  = tower_powers[in_range]
    fr  = tower_freqs[in_range]

    # Sort by distance, take top 3
    idx    = np.argsort(d)[:3]
    rssi_candidates = pw[idx] - np.array([
        free_space_path_loss(d[i], fr[i]) for i in idx
    ])
    return float(rssi_candidates.max())


def in_dead_zone(lat: float, lon: float) -> bool:
    for (dlat, dlon, radius) in DEAD_ZONES:
        dist = haversine_np(lat, lon, np.array([dlat]), np.array([dlon]))[0]
        if dist < radius:
            return True
    return False


def add_signal_to_timesteps(timesteps: List[Dict],
                             towers: List[Dict],
                             seed: int = 1) -> List[Dict]:
    """
    Overlay raw RSSI signal onto vehicle timestep data.
    Adds: rssi_raw (dBm), in_dead_zone, signal_quality (0-1)
    """
    rng = np.random.default_rng(seed)

    # Pre-convert towers to arrays
    t_lats   = np.array([t["lat"]      for t in towers])
    t_lons   = np.array([t["lon"]      for t in towers])
    t_powers = np.array([t["power_dbm"] for t in towers])
    t_freqs  = np.array([BANDS[t["band"]]["freq_mhz"] for t in towers])
    t_ranges = np.array([t["range_m"]  for t in towers])

    enriched = []
    for step in timesteps:
        lat, lon = step["lat"], step["lon"]

        # Base RSSI from tower model
        rssi = compute_rssi(lat, lon, t_lats, t_lons, t_powers, t_freqs, t_ranges)

        # Dead zone override
        dead = in_dead_zone(lat, lon)
        if dead:
            rssi = float(rng.uniform(-115, -105))

        # Gaussian noise  (±3 dB realistic urban)
        rssi += rng.normal(0, 3.0)

        # Sudden spike/drop (multipath fading, rare)
        if rng.random() < 0.03:
            rssi += rng.choice([-15, -20, -10])

        rssi = float(np.clip(rssi, -120, -50))

        # Normalise → 0-1 quality score
        quality = float(np.clip((rssi + 120) / 70, 0, 1))

        step = dict(step)
        step["rssi_raw"]       = round(rssi, 2)
        step["signal_quality"] = round(quality, 4)
        step["in_dead_zone"]   = dead
        enriched.append(step)

    return enriched


def simulate_signals_all(vehicle_data: Dict[str, List[Dict]],
                         towers: List[Dict]) -> Dict[str, List[Dict]]:
    result = {}
    for i, (name, steps) in enumerate(vehicle_data.items()):
        print(f"[Signal] Processing {name} ({len(steps)} steps)…")
        result[name] = add_signal_to_timesteps(steps, towers, seed=10 + i)
        rssi_vals = [s["rssi_raw"] for s in result[name]]
        print(f"  RSSI: min={min(rssi_vals):.1f} max={max(rssi_vals):.1f} "
              f"mean={np.mean(rssi_vals):.1f} dBm")
    return result


if __name__ == "__main__":
    from pathlib import Path
    import json

    vehicle_data = json.loads((DATA_DIR / "vehicle_data.json").read_text())

    print("[Signal] Generating towers…")
    towers = generate_towers(n=300)
    TOWER_FILE.write_text(json.dumps(towers, indent=2))
    print(f"[Signal] {len(towers)} towers saved.")

    signal_data = simulate_signals_all(vehicle_data, towers)
    SIGNAL_FILE.write_text(json.dumps(signal_data, indent=2))
    print(f"[Signal] Saved to {SIGNAL_FILE}")
