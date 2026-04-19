"""
PHASE 8: Real-Time Simulation Loop
- Runs the full pipeline at each timestep
- Location → Signal → Kalman → Features → ML → Decision → Queue/Send
- Broadcasts state via async generator (consumed by FastAPI WebSocket)
- Tracks and logs metrics in SQLite
"""

import asyncio
import json
import sqlite3
import uuid
import time
import numpy as np
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Any

# ── Local imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_processing.kalman_filter import SignalKalmanFilter
from engine.decision_engine import DecisionEngine, EngineConfig, Notification

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH  = Path(__file__).parent.parent / "db" / "smart_notify.db"
DB_PATH.parent.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c    = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id          TEXT PRIMARY KEY,
            route       TEXT,
            started_at  REAL,
            ended_at    REAL,
            metrics     TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS timesteps (
            run_id      TEXT,
            t           REAL,
            lat         REAL,
            lon         REAL,
            speed       REAL,
            rssi_raw    REAL,
            rssi_filtered REAL,
            rssi_trend  REAL,
            quality     REAL,
            in_dead_zone INTEGER,
            stopped     INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS deliveries (
            run_id      TEXT,
            notif_id    TEXT,
            decision    TEXT,
            t           REAL,
            rssi        REAL,
            quality     REAL,
            wait_sec    REAL,
            reason      TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            run_id          TEXT,
            notif_id        TEXT,
            created_t       REAL,
            created_rssi    REAL,
            created_quality REAL,
            created_in_dead_zone INTEGER,
            priority        INTEGER,
            source          TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Initialised at {DB_PATH}")


def log_timestep(conn: sqlite3.Connection, run_id: str, step: Dict):
    conn.execute("""
        INSERT INTO timesteps VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id, step["t"], step["lat"], step["lon"],
        step["speed"], step["rssi_raw"], step["rssi_filtered"],
        step["rssi_trend"], step["signal_quality_filtered"],
        int(step["in_dead_zone"]), int(step["stopped"]),
    ))


def log_delivery(conn: sqlite3.Connection, run_id: str, ev):
    conn.execute("""
        INSERT INTO deliveries VALUES (?,?,?,?,?,?,?,?)
    """, (run_id, ev.notif_id, ev.decision, ev.timestamp,
          ev.rssi, ev.quality, ev.wait_sec, ev.reason))


def log_notification(conn: sqlite3.Connection, run_id: str,
                     notif: Notification, step: Dict[str, Any], source: str = "generated"):
    conn.execute(
        """
        INSERT INTO notifications VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            notif.id,
            float(step["t"]),
            float(step.get("rssi_filtered", step.get("rssi_raw", -120.0))),
            float(step.get("signal_quality_filtered", step.get("signal_quality", 0.0))),
            int(step.get("in_dead_zone", False)),
            int(notif.priority),
            str(source),
        ),
    )


# ══════════════════════════════════════════════════════════════════
# NOTIFICATION GENERATOR (mock incoming messages)
# ══════════════════════════════════════════════════════════════════

class MockNotificationStream:
    """
    Generates synthetic notifications at random intervals.
    In production this is replaced by your actual message source.
    """

    PAYLOADS = [
        ("Route update: Avoid Silk Board", 1),
        ("OTP: 847291", 2),
        ("Turn left in 200m", 2),
        ("Delivery arriving in 10min", 4),
        ("Your cab is 3 stops away", 4),
        ("Flash sale started!", 8),
        ("News: Traffic jam on ORR", 6),
        ("Reminder: Meeting at 3pm", 5),
        ("Weather alert: Rain expected", 5),
        ("Promotional offer - 20% off", 10),
    ]

    LOW_PRIORITY_PAYLOADS = [
        ("Delivery arriving in 10min", 4),
        ("Your cab is 3 stops away", 4),
        ("Flash sale started!", 8),
        ("News: Traffic jam on ORR", 6),
        ("Reminder: Meeting at 3pm", 5),
        ("Weather alert: Rain expected", 5),
        ("Promotional offer - 20% off", 10),
    ]

    def __init__(self, rate_per_min: float = 4.0, seed: int = 7):
        self.rng          = np.random.default_rng(seed)
        self.interval_sec = 60.0 / rate_per_min
        self._next_t      = 0.0
        self._last_emit_t = -9999.0
        self._boost_cooldown_sec = 6.0
        self._deadzone_drop_quality = 0.22
        self._deadzone_drop_prob = 0.75

    def tick(self, t: float, quality: float = 1.0,
             in_dead_zone: bool = False, trend: float = 0.0) -> Optional[Notification]:
        """Returns a new notification if it's time, else None."""
        due_by_schedule = t >= self._next_t

        is_poor_window = (quality < 0.58) or in_dead_zone or (trend < -0.25)
        due_by_boost = (
            is_poor_window
            and (t - self._last_emit_t) >= self._boost_cooldown_sec
            and self.rng.random() < 0.10
        )

        if due_by_schedule or due_by_boost:
            self._next_t = t + self.rng.exponential(self.interval_sec)
            self._last_emit_t = t

            if due_by_boost:
                msg, priority = self.LOW_PRIORITY_PAYLOADS[
                    self.rng.integers(len(self.LOW_PRIORITY_PAYLOADS))
                ]
            else:
                msg, priority = self.PAYLOADS[
                    self.rng.integers(len(self.PAYLOADS))
                ]

            # In severe dead-zones, suppress most non-urgent traffic so queueing
            # focuses on high-value notifications that can realistically improve.
            if (in_dead_zone and quality <= self._deadzone_drop_quality
                and int(priority) > 2
                and self.rng.random() < self._deadzone_drop_prob):
                return None

            return Notification(
                id       = str(uuid.uuid4())[:8],
                payload  = {"message": msg},
                priority = int(priority),
                created_at = t,
            )
        return None


# ══════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════

async def run_simulation(
    route_name:  str           = "city_route",
    speed_factor: float        = 10.0,    # playback speed (10x = 10s simulated per 1s real)
    max_steps:   Optional[int] = None,
    notif_rate:  float         = 4.0,     # notifications per minute
) -> AsyncGenerator[Dict, None]:
    """
    Async generator that yields state dicts at each timestep.
    Designed to be consumed by the FastAPI WebSocket endpoint.

    Yields:
    {
      "t", "lat", "lon", "speed", "rssi_raw", "rssi_filtered",
      "rssi_trend", "quality", "in_dead_zone", "trend_label",
      "queue_size", "events": [...], "metrics": {...}
    }
    """
    init_db()
    run_id = str(uuid.uuid4())

    # ── Load data ──────────────────────────────────────────────────────────
    kalman_file = DATA_DIR / "kalman_data.json"
    if not kalman_file.exists():
        raise RuntimeError(
            "Kalman data not found. Run the full pipeline first:\n"
            "  python run_pipeline.py"
        )

    kalman_data = json.loads(kalman_file.read_text())
    if route_name not in kalman_data:
        route_name = list(kalman_data.keys())[0]

    steps = kalman_data[route_name]
    if max_steps:
        steps = steps[:max_steps]

    print(f"[Sim] Route: {route_name} | {len(steps)} steps | "
          f"speed_factor={speed_factor}x")

    # ── Components ─────────────────────────────────────────────────────────
    engine   = DecisionEngine(EngineConfig())
    notif_gen = MockNotificationStream(rate_per_min=notif_rate)
    kf        = SignalKalmanFilter()   # second-pass online filter (belt+suspenders)

    # ── DB ─────────────────────────────────────────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO simulation_runs VALUES (?,?,?,?,?)",
                 (run_id, route_name, time.time(), None, "{}"))
    conn.commit()

    real_dt = 1.0 / speed_factor   # real seconds between yields

    for i, step in enumerate(steps):
        loop_start = asyncio.get_event_loop().time()

        # ── Online Kalman (second pass, handles live data) ─────────────────
        rssi_filt_live, trend_live = kf.step(step["rssi_raw"])
        step = dict(step)
        # Override with live Kalman (slightly differs from batch due to init)
        step["rssi_filtered"]           = round(rssi_filt_live, 2)
        step["rssi_trend"]              = round(trend_live, 4)
        step["signal_quality_filtered"] = round(
            float(np.clip((rssi_filt_live + 120) / 70, 0, 1)), 4
        )
        if trend_live > 0.3:
            step["trend_label"] = "improving"
        elif trend_live < -0.3:
            step["trend_label"] = "degrading"
        else:
            step["trend_label"] = "stable"

        # ── Ingest into engine ─────────────────────────────────────────────
        engine.ingest(step)

        # ── Possibly generate new notification ──────────────────────────────
        notif = notif_gen.tick(
            step["t"],
            quality=float(step.get("signal_quality_filtered", step.get("signal_quality", 1.0))),
            in_dead_zone=bool(step.get("in_dead_zone", False)),
            trend=float(step.get("rssi_trend", 0.0)),
        )
        if notif:
            engine.add_notification(notif)
            log_notification(conn, run_id, notif, step, source="generated")

        # ── Process queue → get delivery events ───────────────────────────
        events = engine.process_queue()

        # ── Log to DB (every 5th step to save I/O) ─────────────────────────
        if i % 5 == 0:
            log_timestep(conn, run_id, step)
        for ev in events:
            log_delivery(conn, run_id, ev)
        if i % 20 == 0:
            conn.commit()

        # ── Build yield payload ────────────────────────────────────────────
        status = engine.get_status()
        payload = {
            "step":          i,
            "run_id":        run_id,
            "route":         route_name,
            "t":             step["t"],
            "lat":           step["lat"],
            "lon":           step["lon"],
            "speed_kmh":     step["speed_kmh"],
            "rssi_raw":      step["rssi_raw"],
            "rssi_filtered": step["rssi_filtered"],
            "rssi_trend":    step["rssi_trend"],
            "quality":       step["signal_quality_filtered"],
            "in_dead_zone":  step["in_dead_zone"],
            "stopped":       step["stopped"],
            "trend_label":   step["trend_label"],
            "queue_size":    status["queue_size"],
            "metrics":       status["metrics"],
            "events":        [
                {
                    "notif_id": e.notif_id,
                    "decision": e.decision,
                    "wait_sec": round(e.wait_sec, 1),
                    "rssi":     round(e.rssi, 1),
                    "quality":  round(e.quality, 3),
                    "reason":   e.reason,
                }
                for e in events
            ],
        }

        yield payload

        # ── Timing: yield at real_dt pace ──────────────────────────────────
        elapsed = asyncio.get_event_loop().time() - loop_start
        sleep   = max(0, real_dt - elapsed)
        await asyncio.sleep(sleep)

    # ── Finalize ───────────────────────────────────────────────────────────
    conn.execute(
        "UPDATE simulation_runs SET ended_at=?, metrics=? WHERE id=?",
        (time.time(), json.dumps(engine.get_status()["metrics"]), run_id)
    )
    conn.commit()
    conn.close()
    print(f"\n[Sim] Completed. Metrics: {engine.get_status()['metrics']}")


# ── CLI smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _test():
        async for frame in run_simulation("city_route", speed_factor=100, max_steps=100):
            print(f"t={frame['t']:.0f}s | rssi={frame['rssi_filtered']:.1f}dBm | "
                  f"q={frame['quality']:.2f} | {frame['trend_label']} | "
                  f"queue={frame['queue_size']} | events={len(frame['events'])}")
            for ev in frame["events"]:
                print(f"     ↳ [{ev['notif_id']}] {ev['decision']} | "
                      f"wait={ev['wait_sec']}s | {ev['reason']}")

    asyncio.run(_test())
