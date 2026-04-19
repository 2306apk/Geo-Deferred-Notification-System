"""
FastAPI Backend — Smart Notify System
Endpoints:
  GET  /                     → health
  GET  /routes               → available routes
  GET  /metrics              → delivery metrics from DB
  POST /notify               → inject a notification
  POST /simulate/start       → start/restart simulation
  WS   /ws/simulation        → real-time stream (JSON frames)
  GET  /history              → recent delivery history
"""

import asyncio
import csv
import json
import sqlite3
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_runtime import SignalPredictor

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_PATH = ROOT_DIR / "data" / "dataset.csv"
ROUTES_PATH = ROOT_DIR / "data" / "routes.json"
TOWERS_PATH = ROOT_DIR / "data" / "towers.json"
DB_PATH = BASE_DIR / "smart_notify.db"

app = FastAPI(
    title="Smart Notify API",
    description="AI-powered notification delivery for moving vehicles",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared simulation state ────────────────────────────────────────────────

class SimState:
    active_route:   str   = "city_route"
    speed_factor:   float = 15.0
    running:        bool  = False
    latest_frame:   Optional[Dict] = None
    injected_notifs: list = []
    connected_ws:   list  = []   # list of WebSocket objects
    task: Optional[asyncio.Task] = None

sim = SimState()


class PredictRequest(BaseModel):
    data: dict[str, Any]


try:
    predictor = SignalPredictor(str(BASE_DIR / "signal_model_bundle.pkl"))
except Exception as exc:
    predictor = None
    print(f"[API] Warning: model not loaded yet ({exc})")


def init_db() -> None:
    ensure_db_tables()


# ---------- SQLite helpers (lightweight, idempotent) -----------------------
def ensure_db_tables() -> None:
    """Create required tables if they don't exist. Safe to call repeatedly."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                notif_id TEXT,
                decision TEXT,
                timestamp TEXT,
                speed REAL,
                signal INTEGER,
                distraction_risk REAL,
                prob_good_signal REAL,
                decision_threshold REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                delivery_rate REAL,
                avg_delay REAL,
                total_sent INTEGER,
                total_queued INTEGER
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception:
        # be non-fatal — simulation should continue even if DB unavailable
        pass

    # Ensure additional columns exist for older DBs
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(events)")
        cols = {row[1] for row in cur.fetchall()}
        if "prob_good_signal" not in cols:
            cur.execute("ALTER TABLE events ADD COLUMN prob_good_signal REAL")
        if "decision_threshold" not in cols:
            cur.execute("ALTER TABLE events ADD COLUMN decision_threshold REAL")
        conn.commit()
        conn.close()
    except Exception:
        pass


def log_event_db(notif_id: str, decision: str, speed: float, signal: int, distraction_risk: float, prob_good_signal: float | None = None, decision_threshold: float | None = None) -> None:
    """Insert a single event row into the `events` table."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            "INSERT INTO events(notif_id, decision, timestamp, speed, signal, distraction_risk, prob_good_signal, decision_threshold) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                notif_id,
                decision,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                float(speed),
                int(signal),
                float(distraction_risk),
                (float(prob_good_signal) if prob_good_signal is not None else None),
                (float(decision_threshold) if decision_threshold is not None else None),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass



# -------------------- Notification manager + safety helpers ------------------

_prev_speed = 0.0
_prev_time = None
_acc_history: list[float] = []


def get_acceleration(speed: float) -> float:
    """Compute smoothed acceleration (m/s^2) from speed (km/h).

    Keeps a tiny history (last 3 values), clamps to [-3, 3].
    """
    global _prev_speed, _prev_time, _acc_history
    current_time = time.time()
    if _prev_time is None:
        _prev_time = current_time
        _prev_speed = speed
        return 0.0
    dt = current_time - _prev_time
    if dt <= 0:
        return _acc_history[-1] if _acc_history else 0.0
    # convert km/h to m/s
    sp = speed * (1000.0 / 3600.0)
    prev_sp = _prev_speed * (1000.0 / 3600.0)
    acc = (sp - prev_sp) / max(dt, 1e-6)
    acc = max(min(acc, 3.0), -3.0)
    _acc_history.append(acc)
    if len(_acc_history) > 3:
        _acc_history.pop(0)
    smoothed = sum(_acc_history) / len(_acc_history)
    _prev_speed = speed
    _prev_time = current_time
    return smoothed


def compute_distraction_risk(speed: float, accel: float, handover: int, signal_fluctuation: bool) -> float:
    """Compute distraction risk per PART 2 rules.

    - Start at 0
    - speed > 60 -> +2
    - accel < -2 -> +2 (sudden braking)
    - handover == 1 -> +1
    - signal_fluctuation True -> +1
    - speed < 5 -> -3
    """
    risk = 0.0
    if speed > 60:
        risk += 2.0
    if accel < -2.0:
        risk += 2.0
    if int(handover) == 1:
        risk += 1.0
    if signal_fluctuation:
        risk += 1.0
    if speed < 5:
        risk -= 3.0
    return risk


class NotificationManager:
    """In-memory notification queue manager with safety-aware evaluation.

    - keeps pending and delivered lists
    - soft queue cap and collapse of deferred messages
    - evaluate_pending(frame, predictor) applies pre-ML safety checks and
      optionally consults a predictor for 'predicted_good_signal_ahead'
    """

    def __init__(self, soft_cap: int = 8):
        self.pending: list[dict] = []
        self.delivered: list[dict] = []
        self.soft_cap = soft_cap

    def add(
        self,
        message: str,
        priority: int = 5,
        notif_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> dict:
        # priority: 1=urgent ... 10=low
        is_urgent = int(priority) <= 3
        # collapse exact duplicate deferred messages
        if not is_urgent:
            for p in self.pending:
                if p.get("message") == message and not p.get("urgent"):
                    p["repeat_count"] = p.get("repeat_count", 1) + 1
                    return p

        notif = {
            "id": notif_id or str(uuid.uuid4())[:8],
            "message": message,
            "priority": priority,
            "urgent": is_urgent,
            "created_at": created_at or datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "repeat_count": 1,
        }

        # enforce soft cap: collapse deferred into a summary
        if len(self.pending) >= self.soft_cap and not is_urgent:
            deferred = [n for n in self.pending if not n.get("urgent")]
            if deferred:
                collapsed_count = sum(n.get("repeat_count", 1) for n in deferred)
                summary = {
                    "id": f"summary-{int(time.time())}",
                    "message": f"{collapsed_count} deferred notifications summarized",
                    "priority": 10,
                    "urgent": False,
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "repeat_count": collapsed_count,
                    "is_summary": True,
                }
                self.pending = [n for n in self.pending if n.get("urgent")]
                self.pending.append(summary)
                return summary

        self.pending.append(notif)
        return notif

    def evaluate_pending(self, frame: dict, predictor: Optional[object] = None) -> list[dict]:
        """Evaluate queued notifications using the trained ML runtime when available."""
        delivered_now: list[dict] = []
        remaining: list[dict] = []
        nonurgent_sent = 0
        max_send_per_tick = 3

        for notif in self.pending:
            enriched = {**frame}
            enriched["urgent"] = bool(notif.get("urgent", False))
            enriched["queue_size"] = len(self.pending)
            enriched["repeated_count"] = notif.get("repeat_count", 1)

            try:
                created = datetime.fromisoformat(notif["created_at"].replace("Z", "+00:00"))
                age = (datetime.utcnow() - created).total_seconds()
                enriched["queue_age_seconds"] = age
            except Exception:
                enriched["queue_age_seconds"] = 0.0
            # 🔥 TIMEOUT SAFETY (prevents infinite queue)
            if enriched["queue_age_seconds"] > 15:
                decision = "SEND_TIMEOUT"
                reason = "timeout_fallback"

            speed = float(frame.get("speed", 0.0) or 0.0)
            try:
                accel = float(frame.get("accel", 0.0) or 0.0)
            except Exception:
                accel = 0.0
            try:
                accel = get_acceleration(speed)
            except Exception:
                pass

            signal = int(frame.get("signal", 0) or 0)
            signal_1 = int(frame.get("signal_1", signal) or signal)
            signal_2 = int(frame.get("signal_2", signal_1) or signal_1)
            signal_fluct = (abs(signal - signal_1) + abs(signal_1 - signal_2)) >= 2
            # 🔥 STABILITY CHECK
            if signal_fluct:
                enriched["unstable_signal"] = True
            else:
                enriched["unstable_signal"] = False
            handover = int(frame.get("handover", 0) or 0)
            distraction_risk = compute_distraction_risk(speed, accel, handover, signal_fluct)

            predictor_out: dict[str, Any] | None = None
            if predictor is not None and hasattr(predictor, "predict"):
                try:
                    predictor_out = predictor.predict(enriched)
                except Exception:
                    predictor_out = None

            if predictor_out is not None:
                decision = str(predictor_out.get("decision", "QUEUE"))
                reason = str(predictor_out.get("reason", "model_decision"))
                confidence = float(predictor_out.get("confidence", 0.0) or 0.0)
                probability = float(predictor_out.get("prob_good_signal", 0.0) or 0.0)
                decision_threshold = float(
                    predictor_out.get("decision_threshold_used", predictor_out.get("decision_threshold", 0.0)) or 0.0
                )
                minimal_ui = bool(predictor_out.get("minimal_ui", False))
                distraction_risk = float(predictor_out.get("distraction_risk", distraction_risk) or distraction_risk)
                accel = float(predictor_out.get("acceleration", accel) or accel)
                main_model_prob = float(predictor_out.get("main_model_prob", 0.0) or 0.0)
                backup_model_prob = float(predictor_out.get("backup_model_prob", 0.0) or 0.0)
                legacy_model_prob = float(predictor_out.get("legacy_model_prob", 0.0) or 0.0)
            else:
                probability = 1.0 if int(frame.get("signal", 0) or 0) == 1 else 0.0
                decision_threshold = 0.6
                main_model_prob = backup_model_prob = legacy_model_prob = 0.0
                confidence = abs(probability - 0.5) * 2.0
                minimal_ui = bool(enriched["urgent"])
                if enriched["urgent"]:
                    decision = "SEND"
                    reason = "urgent_override"
                elif distraction_risk >= 2:
                    decision = "WAIT_DISTRACTION"
                    reason = "driver_distraction"
                elif accel < -2.0:
                    decision = "WAIT_BRAKE"
                    reason = "sudden_braking_block"
                elif speed > 60:
                    decision = "WAIT_HIGH_SPEED"
                    reason = "fast_highway_safety"
                elif probability > decision_threshold:
                    decision = "SEND"
                    reason = "heuristic_good_signal"
                else:
                    decision = "QUEUE"
                    reason = "default_safe_queue"

            should_send = decision.startswith("SEND")
            if should_send and (enriched["urgent"] or nonurgent_sent < max_send_per_tick):
                delivered = {
                    **notif,
                    "delivered_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "decision": decision,
                    "reason": reason,
                    "confidence": confidence,
                    "probability": probability,
                    "decision_threshold_used": decision_threshold,
                    "prob_good_signal": probability,
                    "main_model_prob": main_model_prob,
                    "backup_model_prob": backup_model_prob,
                    "legacy_model_prob": legacy_model_prob,
                    "minimal_ui": minimal_ui,
                    "distraction_risk": float(distraction_risk),
                    "acceleration": float(accel),
                }
                delivered_now.append(delivered)
                self.delivered.append(delivered)
                if not enriched["urgent"]:
                    nonurgent_sent += 1
                try:
                    log_event_db(
                        delivered.get("id"),
                        delivered.get("decision"),
                        speed,
                        signal,
                        delivered.get("distraction_risk", 0.0),
                        delivered.get("probability", None),
                        delivered.get("decision_threshold_used", None),
                    )
                except Exception:
                    pass
            else:
                remaining.append(notif)

        self.pending = remaining
        return delivered_now


# manager instance used by the simulation loop
manager = NotificationManager()


def _coerce_row_types(row: dict[str, str]) -> dict[str, Any]:
    numeric_fields = {
        "t", "lat", "lon", "speed", "speed_kmh", "accel", "signal", "signal_1", "signal_2",
        "signal_delta", "rsrp", "sinr", "rsrp_dbm", "sinr_db", "rsrq_db", "rssi_raw",
        "signal_quality", "signal_slope", "signal_variance", "handover", "handover_rate",
        "time_since_last_good", "distance_since_last_good_m", "time_in_bad_signal",
        "distance_in_bad_signal_m", "rsrp_kalman", "sinr_kalman", "rsrp_mean_5", "rsrp_std_5",
        "sinr_mean_5", "rsrp_slope", "future_good_ratio", "future_mean_rsrp", "label",
        "label_stable", "tower_distance_m",
    }
    int_like_fields = {
        "signal", "signal_1", "signal_2", "handover", "handover_rate", "is_fluctuating",
        "is_fast", "is_braking", "time_since_last_good", "time_in_bad_signal", "label", "label_stable",
    }
    converted: dict[str, Any] = {}
    for key, value in row.items():
        if value in (None, ""):
            converted[key] = None
            continue
        if key in numeric_fields:
            try:
                num = float(value)
                converted[key] = int(num) if key in int_like_fields else num
                continue
            except Exception:
                pass
        if value in {"True", "False"}:
            converted[key] = value == "True"
        else:
            converted[key] = value
    return converted


def _load_dataset_rows(route: str) -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise HTTPException(404, f"Dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        rows = [_coerce_row_types(r) for r in csv.DictReader(fh)]

    if route and route not in {"all", "*"}:
        filtered = [row for row in rows if str(row.get("route", "")) == route]
        if filtered:
            return filtered
    return rows



# ══════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status":   "ok",
        "service":  "Smart Notify",
        "sim_running": sim.running,
        "route":    sim.active_route,
        "model_loaded": predictor is not None,
    }


@app.post("/predict")
async def predict_signal(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Model bundle not loaded. Train the model first.")
    return predictor.predict(req.data)


@app.get("/routes")
async def get_routes():
    if not ROUTES_PATH.exists():
        raise HTTPException(404, "Routes not generated yet. Run pipeline first.")
    routes = json.loads(ROUTES_PATH.read_text())
    return {
        name: {
            "description": r.get("description", ""),
            "route_type":  r.get("route_type", ""),
            "num_waypoints": len(r.get("waypoints", [])),
        }
        for name, r in routes.items()
    }


class SimStartRequest(BaseModel):
    route:        str   = "city_route"
    speed_factor: float = 15.0
    notif_rate:   float = 4.0


@app.post("/simulate/start")
async def start_simulation(req: SimStartRequest, bg: BackgroundTasks):
    """Start or restart the simulation loop."""
    if sim.task and not sim.task.done():
        sim.running = False
        sim.task.cancel()
        try:
            await sim.task
        except Exception:
            pass

    sim.active_route = req.route
    sim.speed_factor = req.speed_factor
    sim.running      = True
    sim.injected_notifs = []
    sim.task = asyncio.create_task(_run_sim_task(req.route, req.speed_factor, req.notif_rate))
    # Log run start in DB (non-blocking)
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("INSERT INTO runs(start_time) VALUES(?)", (datetime.utcnow().isoformat(timespec="seconds") + "Z",))
        conn.commit()
        conn.close()
    except Exception:
        pass
    return {"status": "started", "route": req.route, "speed_factor": req.speed_factor}


@app.post("/simulate/stop")
async def stop_simulation():
    sim.running = False
    if sim.task and not sim.task.done():
        sim.task.cancel()
        try:
            await sim.task
        except Exception:
            pass
    sim.task = None
    return {"status": "stopped"}


class NotifyRequest(BaseModel):
    message:  str
    priority: int = 5   # 1=urgent ... 10=low


@app.post("/notify")
async def inject_notification(req: NotifyRequest):
    """Inject a notification into the engine during live simulation."""
    notif = manager.add(req.message, priority=req.priority)
    return {"status": "queued", "notif": notif}


@app.get("/metrics")
async def get_metrics():
    """Pull aggregated delivery metrics from SQLite."""
    if not DB_PATH.exists():
        return {"error": "No data yet. Run a simulation first."}
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # recent runs (from our `runs` table)
    runs = conn.execute("SELECT * FROM runs ORDER BY start_time DESC LIMIT 5").fetchall()

    # aggregated counts by decision from events
    deliveries = conn.execute(
        "SELECT decision, COUNT(*) as cnt FROM events GROUP BY decision"
    ).fetchall()

    # simple metrics summary
    total_sent = conn.execute("SELECT COUNT(*) as cnt FROM events WHERE decision LIKE 'SEND%'").fetchone()[0]
    total_queued = conn.execute("SELECT COUNT(*) as cnt FROM events WHERE decision = 'QUEUE'").fetchone()[0]
    delivery_rate = float(total_sent) / float(max(1, total_sent + total_queued))

    # persist a snapshot into metrics table (lightweight)
    try:
        conn.execute(
            "INSERT INTO metrics(created_at, delivery_rate, avg_delay, total_sent, total_queued) VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(timespec="seconds") + "Z", delivery_rate, None, int(total_sent), int(total_queued)),
        )
        conn.commit()
    except Exception:
        pass

    conn.close()

    return {
        "runs": [dict(r) for r in runs],
        "deliveries_by_decision": [dict(d) for d in deliveries],
        "summary": {
            "delivery_rate": delivery_rate,
            "total_sent": int(total_sent),
            "total_queued": int(total_queued),
        },
        "latest_frame": sim.latest_frame,
    }


@app.get("/history")
async def get_history(limit: int = 50):
    """Recent delivery events."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/towers")
async def get_towers():
    """Return tower positions for map overlay."""
    if not TOWERS_PATH.exists():
        return []
    return json.loads(TOWERS_PATH.read_text())


# ══════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    """
    Real-time simulation stream.
    Client connects → receives JSON frames at simulation tick rate.
    Client can also send: {"type": "notify", "message": "...", "priority": 3}
    """
    await ws.accept()
    sim.connected_ws.append(ws)
    print(f"[WS] Client connected. Total: {len(sim.connected_ws)}")

    try:
        while True:
            # Receive any client messages (non-blocking)
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=0.05)
                msg  = json.loads(data)
                if msg.get("type") == "notify":
                    manager.add(
                        msg.get("message", "Client notification"),
                        priority=int(msg.get("priority", 5)),
                    )
            except asyncio.TimeoutError:
                pass

            # Push latest frame if available
            if sim.latest_frame:
                await ws.send_text(json.dumps(sim.latest_frame))

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        sim.connected_ws.remove(ws)
        print(f"[WS] Client disconnected. Remaining: {len(sim.connected_ws)}")


# ══════════════════════════════════════════════════════════════════
# BACKGROUND SIMULATION TASK
# ══════════════════════════════════════════════════════════════════

async def _run_sim_task(route: str, speed_factor: float, notif_rate: float):
    """
    Run a local CSV-backed simulation and score notifications with the ML predictor.
    """
    print(f"[Task] Simulation starting: {route}")
    try:
        init_db()
        rows = _load_dataset_rows(route)
        tick_seconds = max(0.05, 1.0 / max(float(speed_factor), 1.0))

        for idx, base_row in enumerate(rows):
            if not sim.running:
                print("[Task] Simulation stopped by user.")
                break

            frame = dict(base_row)
            frame["t"] = int(frame.get("t", idx) or idx)
            frame.setdefault("events", [])
            if predictor is not None:
                try:
                    frame["prediction"] = predictor.predict(frame)
                except Exception:
                    frame["prediction"] = None
            else:
                frame["prediction"] = None

            delivered = manager.evaluate_pending(frame, predictor=predictor)
            for d in delivered:
                frame.setdefault("events", []).append({
                    "notif_id": d.get("id"),
                    "message": d.get("message"),
                    "decision": d.get("decision"),
                    "wait_sec": 0,
                    "rssi": frame.get("rssi_raw", frame.get("rsrp_dbm")),
                    "quality": frame.get("signal_quality"),
                    "reason": d.get("reason"),
                    "probability": d.get("probability"),
                    "decision_threshold": d.get("decision_threshold_used"),
                    "main_model_prob": d.get("main_model_prob"),
                    "backup_model_prob": d.get("backup_model_prob"),
                    "legacy_model_prob": d.get("legacy_model_prob"),
                    "distraction_risk": d.get("distraction_risk"),
                    "acceleration": d.get("acceleration"),
                    "minimal_ui": d.get("minimal_ui"),
                })

            sim.latest_frame = frame

            # Broadcast to all connected clients
            dead = []
            for ws in sim.connected_ws:
                try:
                    await ws.send_text(json.dumps(frame))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                sim.connected_ws.remove(ws)
            await asyncio.sleep(tick_seconds)

    except Exception as e:
        print(f"[Task] Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.running = False
        print("[Task] Simulation ended.")


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    init_db()
    ensure_db_tables()
    print("[API] Smart Notify API ready.")
    print("[API] Docs: http://localhost:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
