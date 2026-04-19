#!/usr/bin/env python3
"""Rebuild signal/kalman/training artifacts using cached OpenCelliD towers only."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.signal_simulator import simulate_signals_all
from signal_processing.kalman_filter import apply_kalman_to_dataset
from ml.train_models import run_training


def main():
    data_dir = ROOT / "data"
    cache_file = data_dir / "opencellid_towers.json"
    vehicle_file = data_dir / "vehicle_data.json"

    if not cache_file.exists():
        raise FileNotFoundError(f"OpenCelliD cache not found: {cache_file}")
    if not vehicle_file.exists():
        raise FileNotFoundError(f"Vehicle data not found: {vehicle_file}. Run pipeline phase 1/2 first.")

    towers = json.loads(cache_file.read_text(encoding="utf-8"))
    for t in towers:
        t["source"] = "opencellid"

    vehicle_data = json.loads(vehicle_file.read_text(encoding="utf-8"))

    print(f"[CacheRebuild] Using {len(towers)} cached OpenCelliD towers")

    (data_dir / "towers.json").write_text(json.dumps(towers), encoding="utf-8")

    signal_data = simulate_signals_all(vehicle_data, towers)
    (data_dir / "signal_data.json").write_text(json.dumps(signal_data), encoding="utf-8")

    kalman_data = apply_kalman_to_dataset(signal_data)
    (data_dir / "kalman_data.json").write_text(json.dumps(kalman_data), encoding="utf-8")

    run_training()
    print("[CacheRebuild] Rebuild complete.")


if __name__ == "__main__":
    main()
