"""
PHASE 3: Signal Simulation
- OpenCelliD tower ingestion (if API key is available)
- Synthetic fallback towers (if OpenCelliD is unavailable)
- Signal = f(distance to nearest tower, frequency band)
- Adds realistic noise: Gaussian + sudden drops (tunnels, underpasses)
- Output: raw RSSI per timestep (dBm scale)
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import requests

DATA_DIR    = Path(__file__).parent
SIGNAL_FILE = DATA_DIR / "signal_data.json"
TOWER_FILE  = DATA_DIR / "towers.json"
OPENCELLID_CACHE_FILE = DATA_DIR / "opencellid_towers.json"

# ── Synthetic tower generation ───────────────────────────────────────────────

BANGALORE_BOUNDS = {
    "lat_min": 12.85, "lat_max": 13.15,
    "lon_min": 77.45, "lon_max": 77.80,
}

OPENCELLID_MAX_BBOX_M2 = 4_000_000
OPENCELLID_TARGET_TILE_M2 = 3_200_000

BANDS = {
    "5G": {"freq_mhz": 3500, "range_m": 2000, "power_dbm": 46},
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


def _radio_to_band(radio: str) -> str:
    r = (radio or "").strip().upper()
    if r in {"NR", "5G"}:
        return "5G"
    if r in {"LTE", "4G"}:
        return "4G"
    if r in {"UMTS", "WCDMA", "HSPA", "HSPA+", "3G"}:
        return "3G"
    return "2G"


def _fetch_opencellid_towers(bounds: Dict[str, float], api_key: str,
                             timeout: int = 30,
                             max_towers: int = 300,
                             per_tile_limit: int = 20) -> List[Dict]:
    """Fetch towers from OpenCelliD getInArea API using tiled requests."""
    url = "https://opencellid.org/cell/getInArea"

    def _meters_per_deg_lat() -> float:
        return 111_320.0

    def _meters_per_deg_lon(lat_deg: float) -> float:
        return 111_320.0 * np.cos(np.radians(lat_deg))

    def _estimate_area_m2(b: Dict[str, float]) -> float:
        mid_lat = 0.5 * (b["lat_min"] + b["lat_max"])
        h = abs(b["lat_max"] - b["lat_min"]) * _meters_per_deg_lat()
        w = abs(b["lon_max"] - b["lon_min"]) * _meters_per_deg_lon(mid_lat)
        return float(max(h * w, 0.0))

    def _tile_bounds(b: Dict[str, float]) -> List[Dict[str, float]]:
        area = _estimate_area_m2(b)
        if area <= OPENCELLID_MAX_BBOX_M2:
            return [b]

        mid_lat = 0.5 * (b["lat_min"] + b["lat_max"])
        lat_span_deg = b["lat_max"] - b["lat_min"]
        lon_span_deg = b["lon_max"] - b["lon_min"]
        lat_m = abs(lat_span_deg) * _meters_per_deg_lat()
        lon_m = abs(lon_span_deg) * _meters_per_deg_lon(mid_lat)

        # Build approximately square tiles under API limit.
        tile_side_m = float(np.sqrt(OPENCELLID_TARGET_TILE_M2))
        n_lat = max(1, int(np.ceil(lat_m / tile_side_m)))
        n_lon = max(1, int(np.ceil(lon_m / tile_side_m)))

        lat_step = lat_span_deg / n_lat
        lon_step = lon_span_deg / n_lon

        tiles = []
        for i in range(n_lat):
            for j in range(n_lon):
                lat0 = b["lat_min"] + i * lat_step
                lat1 = b["lat_min"] + (i + 1) * lat_step
                lon0 = b["lon_min"] + j * lon_step
                lon1 = b["lon_min"] + (j + 1) * lon_step
                tiles.append({
                    "lat_min": min(lat0, lat1),
                    "lat_max": max(lat0, lat1),
                    "lon_min": min(lon0, lon1),
                    "lon_max": max(lon0, lon1),
                })
        return tiles

    tiles = _tile_bounds(bounds)
    if len(tiles) > 1:
        print(f"[Signal] OpenCelliD area too large; querying {len(tiles)} tiles.")

    cells = []
    seen_cells = set()
    empty_tiles = 0
    for idx, tile in enumerate(tiles, start=1):
        # OpenCelliD expects: BBOX=latmin,lonmin,latmax,lonmax
        bbox = (
            f"{tile['lat_min']},{tile['lon_min']},"
            f"{tile['lat_max']},{tile['lon_max']}"
        )
        params = {
            "key": api_key,
            "BBOX": bbox,
            "limit": int(per_tile_limit),
            "format": "json",
        }

        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        tile_cells = payload.get("cells", [])
        if not tile_cells:
            empty_tiles += 1
            status = payload.get("status", "unknown")
            message = payload.get("message") or payload.get("error") or "no message"
            print(
                f"[Signal] OpenCelliD tile {idx}/{len(tiles)} returned 0 cells "
                f"(status={status}, message={message})."
            )
            if "daily limit" in str(message).lower() or "limit" in str(message).lower() and "exceed" in str(message).lower():
                print("[Signal] OpenCelliD daily quota reached. Stopping tile queries.")
                break
            # If many early tiles are empty and no data has been collected,
            # this usually indicates key/plan access limits rather than sparse coverage.
            if idx >= 12 and not cells and empty_tiles == idx:
                print(
                    "[Signal] Stopping further OpenCelliD tile queries early: "
                    "no cells found in initial tiles."
                )
                break
            continue

        for c in tile_cells:
            cid = c.get("cellid") or c.get("cid")
            lac = c.get("lac") or c.get("tac")
            mcc = c.get("mcc")
            mnc = c.get("mnc")
            radio = str(c.get("radio", "")).upper()
            sig = (radio, mcc, mnc, lac, cid)
            if sig in seen_cells:
                continue
            seen_cells.add(sig)
            cells.append(c)

        if len(cells) >= max_towers:
            print(f"[Signal] Reached target of {max_towers} OpenCelliD cells; stopping early.")
            break

    if not cells:
        print("[Signal] OpenCelliD returned no cells across all tiles.")

    towers = []
    seen = set()
    for idx, c in enumerate(cells):
        lat = c.get("lat")
        lon = c.get("lon")
        if lat is None or lon is None:
            continue

        # Deduplicate by cell identity where possible, else by coordinates.
        cid = c.get("cellid") or c.get("cid")
        lac = c.get("lac")
        mcc = c.get("mcc")
        mnc = c.get("mnc")
        radio = str(c.get("radio", "")).upper()
        sig = (radio, mcc, mnc, lac, cid)
        if cid is None:
            sig = (round(float(lat), 5), round(float(lon), 5), radio)
        if sig in seen:
            continue
        seen.add(sig)

        band = _radio_to_band(str(c.get("radio", "")))
        b = BANDS.get(band, BANDS["3G"])
        towers.append({
            "id": idx,
            "lat": round(float(lat), 6),
            "lon": round(float(lon), 6),
            "band": band,
            "range_m": b["range_m"],
            "power_dbm": b["power_dbm"],
            "source": "opencellid",
            "samples": int(c.get("samples", 0) or 0),
            "changeable": int(c.get("changeable", 1) or 1),
        })

    towers.sort(key=lambda t: (t.get("samples", 0), -t.get("changeable", 1)), reverse=True)
    return towers


def generate_towers(n: int = 300, seed: int = 0,
                    use_opencellid: bool = False,
                    api_key: str = None,
                    force_refresh: bool = False,
                    bounds: Dict[str, float] = None) -> List[Dict]:
    """
    Generate tower set for simulation.

    If use_opencellid=True and key exists, real towers are fetched/cached.
    Otherwise falls back to synthetic tower generation.
    """
    if use_opencellid:
        key = api_key or os.getenv("OPENCELLID_API_KEY")
        search_bounds = bounds or BANGALORE_BOUNDS

        if not key:
            print("[Signal] OpenCelliD key not found. Falling back to synthetic towers.")
        else:
            try:
                if OPENCELLID_CACHE_FILE.exists() and not force_refresh:
                    cached = json.loads(OPENCELLID_CACHE_FILE.read_text())
                    if cached:
                        print(f"[Signal] Loaded {len(cached)} cached OpenCelliD towers.")
                        return cached[:n]

                towers = _fetch_opencellid_towers(
                    search_bounds,
                    key,
                    max_towers=n,
                    per_tile_limit=20,
                )
                if towers:
                    selected = towers[:n]
                    OPENCELLID_CACHE_FILE.write_text(json.dumps(selected, indent=2))
                    print(f"[Signal] OpenCelliD fetched {len(towers)} towers, using {len(selected)}.")
                    return selected

                print("[Signal] OpenCelliD returned no cells. Falling back to synthetic towers.")
            except Exception as e:
                print(f"[Signal] OpenCelliD fetch failed ({e}). Falling back to synthetic towers.")

            # If refresh fails or returns no cells, still reuse stale cache when available.
            if OPENCELLID_CACHE_FILE.exists():
                try:
                    cached = json.loads(OPENCELLID_CACHE_FILE.read_text())
                    if cached:
                        print(f"[Signal] Reusing stale OpenCelliD cache ({len(cached)} towers).")
                        return cached[:n]
                except Exception:
                    pass

    # Synthetic fallback path
    rng    = np.random.default_rng(seed)
    towers = []

    # City center cluster
    for _ in range(n // 2):
        lat  = rng.normal(12.97, 0.05)
        lon  = rng.normal(77.59, 0.05)
        band = rng.choice(["4G", "3G"], p=[0.7, 0.3])
        t = _make_tower(lat, lon, band, len(towers))
        t["source"] = "synthetic"
        towers.append(t)

    # Spread across bounds
    for _ in range(n // 2):
        lat  = rng.uniform(BANGALORE_BOUNDS["lat_min"], BANGALORE_BOUNDS["lat_max"])
        lon  = rng.uniform(BANGALORE_BOUNDS["lon_min"], BANGALORE_BOUNDS["lon_max"])
        band = rng.choice(["4G", "3G", "2G"], p=[0.5, 0.3, 0.2])
        t = _make_tower(lat, lon, band, len(towers))
        t["source"] = "synthetic"
        towers.append(t)

    print(f"[Signal] Using {len(towers)} synthetic towers.")
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

    route_name = str(timesteps[0].get("route", "unknown")) if timesteps else "unknown"

    # Route-aware attenuation for offline synthetic realism.
    base_env_loss = {
        "city_route": 11.0,
        "highway_route": 18.0,
        "tunnel_route": 22.0,
    }.get(route_name, 12.0)

    # Deterministic outage windows create stronger transition supervision.
    n_steps = len(timesteps)
    rng_outage = np.random.default_rng(seed + 10_000)
    outage_windows = []
    outage_count = {
        "city_route": 3,
        "highway_route": 2,
        "tunnel_route": 4,
    }.get(route_name, 2)
    for _ in range(outage_count):
        if n_steps < 60:
            break
        start = int(rng_outage.integers(low=20, high=max(21, n_steps - 40)))
        duration = int(rng_outage.integers(low=18, high=70))
        end = min(n_steps - 1, start + duration)
        depth = float(rng_outage.uniform(10.0, 22.0))
        outage_windows.append((start, end, depth))

    # Time-correlated signal state to create realistic good<->bad transitions.
    shadow_db = 0.0
    fade_remaining = 0
    fade_total = 0
    fade_depth = 0.0

    for idx, step in enumerate(timesteps):
        lat, lon = step["lat"], step["lon"]

        # Base RSSI from tower model
        rssi = compute_rssi(lat, lon, t_lats, t_lons, t_powers, t_freqs, t_ranges)

        # Dead zone override
        dead = in_dead_zone(lat, lon)
        if dead:
            rssi = float(rng.uniform(-115, -105))

        # Correlated shadowing (slow drift) improves temporal continuity.
        shadow_db = 0.90 * shadow_db + float(rng.normal(0.0, 1.6))

        # Start a temporary deep-fade event; more likely in dead zones and at speed.
        speed = float(step.get("speed", 0.0))
        if fade_remaining <= 0:
            fade_p = 0.012 + (0.018 if dead else 0.0) + (0.008 if speed > 8.0 else 0.0)
            if rng.random() < fade_p:
                fade_total = int(rng.integers(4, 14))
                fade_remaining = fade_total
                fade_depth = float(rng.uniform(8.0, 20.0))

        fade_penalty = 0.0
        if fade_remaining > 0 and fade_total > 0:
            progress = 1.0 - (fade_remaining / max(fade_total, 1))
            fade_penalty = -fade_depth * (1.0 - 0.5 * progress)
            fade_remaining -= 1

        # Occasional short recovery bump creates sharper transition boundaries.
        recovery_bump = 0.0
        if fade_remaining <= 0 and rng.random() < 0.01:
            recovery_bump = float(rng.uniform(3.0, 7.0))

        rssi += shadow_db + fade_penalty + recovery_bump

        # Route attenuation makes synthetic coverage less overly optimistic.
        rssi -= base_env_loss

        # Inject deterministic temporary outages for better transition learning.
        outage_penalty = 0.0
        for start, end, depth in outage_windows:
            if start <= idx <= end:
                width = max(1.0, (end - start) / 2.0)
                mid = (start + end) / 2.0
                shape = 1.0 - min(abs(idx - mid) / width, 1.0)
                outage_penalty += depth * (0.5 + 0.5 * shape)
        if outage_penalty > 0.0:
            rssi -= outage_penalty

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
        step["synthetic_outage"] = bool(outage_penalty > 0.0)
        enriched.append(step)

    return enriched


def simulate_signals_all(vehicle_data: Dict[str, List[Dict]],
                         towers: List[Dict]) -> Dict[str, List[Dict]]:
    result = {}
    for i, (name, steps) in enumerate(vehicle_data.items()):
        print(f"[Signal] Processing {name} ({len(steps)} steps)…")
        result[name] = add_signal_to_timesteps(steps, towers, seed=10 + i)
        rssi_vals = [s["rssi_raw"] for s in result[name]]
        good_frac = float(np.mean(np.array(rssi_vals) >= -85.0)) if rssi_vals else float("nan")
        print(f"  RSSI: min={min(rssi_vals):.1f} max={max(rssi_vals):.1f} "
                            f"mean={np.mean(rssi_vals):.1f} dBm | good_frac={good_frac:.2f}")
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
