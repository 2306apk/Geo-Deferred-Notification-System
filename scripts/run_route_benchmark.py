#!/usr/bin/env python3
"""Run offline simulation passes for benchmark profiles to populate DB metrics."""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation.realtime_loop import run_simulation


PROFILES = {
    "full": {
        "routes": ["city_route", "highway_route", "tunnel_route", "mixed_route"],
        "notif_rate": 6.0,
    },
    "legacy": {
        "routes": ["city_route", "tunnel_route"],
        "notif_rate": 4.0,
    },
}


async def main(profile: str):
    config = PROFILES[profile]
    routes = config["routes"]
    notif_rate = float(config["notif_rate"])
    print(f"[Benchmark] Profile={profile} routes={routes} notif_rate={notif_rate}")
    for route in routes:
        print(f"[Benchmark] Running {route}...")
        async for _ in run_simulation(route_name=route, speed_factor=500.0, notif_rate=notif_rate):
            pass
    print("[Benchmark] Completed all routes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="full",
        help="Benchmark profile: full (stress) or legacy (higher-RSSI baseline)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.profile))
