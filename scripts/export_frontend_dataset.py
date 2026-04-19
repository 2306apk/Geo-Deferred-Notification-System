#!/usr/bin/env python3
"""Export frontend dataset.csv from data/kalman_data.json.

This bridges backend simulation output to the pulled frontend dashboard, which expects
frontend/data/dataset.csv with mobility + radio columns.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "kalman_data.json"
DEFAULT_OUTPUT = ROOT / "frontend" / "data" / "dataset.csv"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bars_from_rssi(rssi: float) -> int:
    if rssi >= -80:
        return 5
    if rssi >= -90:
        return 4
    if rssi >= -100:
        return 3
    if rssi >= -108:
        return 2
    if rssi >= -115:
        return 1
    return 0


def _flatten_rows(data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for route_name, steps in data.items():
        last_good_t = None
        dist_since_good_m = 0.0

        prev_signal = 1
        prev_signal_1 = 1
        prev_rsrp = -90.0
        prev_sinr = 5.0
        prev_speed = None
        prev_t = None

        for step in steps:
            t = float(step.get("t", 0.0))
            speed_mps = float(step.get("speed", 0.0))
            speed_kmh = float(step.get("speed_kmh", speed_mps * 3.6))
            rssi = float(step.get("rssi_filtered", step.get("rssi_raw", -120.0)))
            quality = float(step.get("signal_quality_filtered", step.get("signal_quality", 0.0)))
            dead_zone = bool(step.get("in_dead_zone", False))

            signal = 1 if rssi >= -90.0 else 0
            signal_1 = prev_signal
            signal_2 = prev_signal_1

            if signal == 1:
                last_good_t = t
                dist_since_good_m = 0.0
            else:
                if prev_t is not None:
                    dt = max(0.0, t - prev_t)
                    dist_since_good_m += speed_mps * dt

            time_since_last_good = 0.0 if last_good_t is None else max(0.0, t - last_good_t)

            if prev_speed is None or prev_t is None:
                accel = 0.0
            else:
                dt = max(1e-6, t - prev_t)
                accel = (speed_mps - prev_speed) / dt

            rsrp_dbm = rssi
            rsrq_db = -20.0 + quality * 15.0
            sinr_db = -10.0 + quality * 30.0

            tower_distance_m = 50.0 + (1.0 - quality) * 1200.0
            if dead_zone:
                tower_distance_m += 500.0

            tower_count_nearby = 1 + int(round(quality * 4.0))
            if dead_zone:
                tower_count_nearby = max(0, tower_count_nearby - 1)

            real_tower_visible = 1 if rssi > -112.0 else 0
            signal_delta = signal - prev_signal
            rsrp_delta = rsrp_dbm - prev_rsrp
            handover = 1 if abs(rsrp_delta) >= 8.0 or signal_delta != 0 else 0

            row = {
                "route": route_name,
                "t": round(t, 3),
                "lat": float(step.get("lat", 0.0)),
                "lon": float(step.get("lon", 0.0)),
                "speed": round(speed_mps, 4),
                "speed_kmh": round(speed_kmh, 3),
                "accel": round(_clamp(accel, -6.0, 6.0), 4),
                "signal": signal,
                "signal_1": signal_1,
                "signal_2": signal_2,
                "time_since_last_good": round(time_since_last_good, 3),
                "signal_bars": _bars_from_rssi(rssi),
                "rsrp_dbm": round(rsrp_dbm, 3),
                "rsrq_db": round(_clamp(rsrq_db, -20.0, -3.0), 3),
                "sinr_db": round(_clamp(sinr_db, -10.0, 20.0), 3),
                "tower_distance_m": round(tower_distance_m, 3),
                "tower_count_nearby": int(_clamp(float(tower_count_nearby), 0.0, 5.0)),
                "real_tower_visible": int(real_tower_visible),
                "rsrp_1": round(prev_rsrp, 3),
                "sinr_1": round(prev_sinr, 3),
                "signal_delta": int(signal_delta),
                "rsrp_delta": round(rsrp_delta, 3),
                "handover": int(handover),
                "distance_since_last_good_m": round(dist_since_good_m, 3),
            }
            rows.append(row)

            prev_signal_1 = prev_signal
            prev_signal = signal
            prev_rsrp = rsrp_dbm
            prev_sinr = sinr_db
            prev_speed = speed_mps
            prev_t = t

    return rows


def export_dataset(input_file: Path, output_file: Path) -> int:
    if not input_file.exists():
        raise FileNotFoundError(f"Input not found: {input_file}")

    data = json.loads(input_file.read_text(encoding="utf-8"))
    rows = _flatten_rows(data)
    if not rows:
        raise RuntimeError("No rows generated from input data")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frontend dataset.csv from kalman_data.json")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to input kalman_data.json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output dataset.csv")
    args = parser.parse_args()

    n = export_dataset(Path(args.input), Path(args.output))
    print(f"[FrontendDataset] Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
