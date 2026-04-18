#!/usr/bin/env python3
"""
run_pipeline.py — Master data pipeline for Smart Notify
Run this ONCE before starting the API server.

Executes in order:
  Phase 1: generate_routes      → data/routes.json
  Phase 2: simulate_vehicle     → data/vehicle_data.json
  Phase 3: signal_simulator     → data/signal_data.json + towers.json
  Phase 4: kalman_filter        → data/kalman_data.json
  Phase 5&6: train_models       → models/*.pkl

Usage:
  python run_pipeline.py                  # full pipeline
  python run_pipeline.py --skip-osmnx    # use synthetic routes
  python run_pipeline.py --skip-training  # only data, no ML
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def banner(msg: str):
    print("\n" + "═" * 60)
    print(f"  {msg}")
    print("═" * 60)


def run_phase(name: str, fn, *args, **kwargs):
    print(f"\n[Pipeline] ▶ {name}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"[Pipeline] ✓ {name} done in {time.time()-t0:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-osmnx",    action="store_true",
                        help="Use synthetic route coords instead of OSMnx download")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip ML model training (use existing models)")
    parser.add_argument("--force",         action="store_true",
                        help="Force re-run even if cached files exist")
    args = parser.parse_args()

    banner("SMART NOTIFY — DATA & TRAINING PIPELINE")

    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "db").mkdir(exist_ok=True)

    # ── Phase 1: Routes ──────────────────────────────────────────────────
    routes_file = data_dir / "routes.json"
    if not routes_file.exists() or args.force:
        from data.generate_routes import load_or_generate_routes
        routes = run_phase(
            "Phase 1: Route Generation",
            load_or_generate_routes,
            use_osmnx=not args.skip_osmnx,
            force=args.force,
        )
    else:
        routes = json.loads(routes_file.read_text())
        print(f"[Pipeline] Phase 1 cached — {list(routes.keys())}")

    # ── Phase 2: Vehicle simulation ──────────────────────────────────────
    vehicle_file = data_dir / "vehicle_data.json"
    if not vehicle_file.exists() or args.force:
        from data.simulate_vehicle import simulate_all_routes
        vehicle_data = run_phase(
            "Phase 2: Vehicle Movement Simulation",
            simulate_all_routes,
            routes,
        )
        vehicle_file.write_text(json.dumps(vehicle_data))
    else:
        vehicle_data = json.loads(vehicle_file.read_text())
        print(f"[Pipeline] Phase 2 cached — {sum(len(v) for v in vehicle_data.values())} timesteps")

    # ── Phase 3: Signal simulation ────────────────────────────────────────
    signal_file = data_dir / "signal_data.json"
    tower_file  = data_dir / "towers.json"
    if not signal_file.exists() or args.force:
        from data.signal_simulator import generate_towers, simulate_signals_all
        towers = run_phase("Phase 3a: Tower Generation", generate_towers, n=300)
        tower_file.write_text(json.dumps(towers))
        signal_data = run_phase(
            "Phase 3b: Signal Simulation",
            simulate_signals_all,
            vehicle_data, towers,
        )
        signal_file.write_text(json.dumps(signal_data))
    else:
        signal_data = json.loads(signal_file.read_text())
        print(f"[Pipeline] Phase 3 cached")

    # ── Phase 4: Kalman filtering ─────────────────────────────────────────
    kalman_file = data_dir / "kalman_data.json"
    if not kalman_file.exists() or args.force:
        from signal_processing.kalman_filter import apply_kalman_to_dataset
        kalman_data = run_phase(
            "Phase 4: Kalman Filter",
            apply_kalman_to_dataset,
            signal_data,
        )
        kalman_file.write_text(json.dumps(kalman_data))
    else:
        kalman_data = json.loads(kalman_file.read_text())
        print(f"[Pipeline] Phase 4 cached")

    # ── Phase 5&6: ML Training ────────────────────────────────────────────
    models_dir = ROOT / "models"
    gb_exists  = (models_dir / "gb_model.pkl").exists()

    if not gb_exists or args.force:
        if args.skip_training:
            print("[Pipeline] ⚠ Skipping training (--skip-training). "
                  "Decision engine will use heuristics only.")
        else:
            from ml.train_models import run_training
            run_phase("Phase 5&6: ML Model Training", run_training)
    else:
        print("[Pipeline] Phase 5&6 cached — models loaded from disk")

    banner("PIPELINE COMPLETE ✓")
    print("\n  Next steps:")
    print("  1. Start API:       cd api && uvicorn main:app --reload --port 8000")
    print("  2. Open docs:       http://localhost:8000/docs")
    print("  3. Start sim:       POST /simulate/start")
    print("  4. Connect WS:      ws://localhost:8000/ws/simulation")
    print()


if __name__ == "__main__":
    main()
