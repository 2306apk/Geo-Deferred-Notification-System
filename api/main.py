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
import json
import sqlite3
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.realtime_loop import run_simulation, DB_PATH, init_db
from engine.decision_engine import DecisionEngine, EngineConfig

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

# ─────────────────────────────────────────────────────────────────────────────
# Shared simulation state (thread‑safe with asyncio.Lock)
# ─────────────────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.active_route: str = "city_route"
        self.speed_factor: float = 15.0
        self.running: bool = False
        self.latest_frame: Optional[Dict] = None
        self.injected_notifs: List[Dict[str, Any]] = []
        self.connected_ws: List[WebSocket] = []
        self.session_id: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

sim = SimState()


# ─────────────────────────────────────────────────────────────────────────────
# SQLite helpers (lightweight, idempotent)
# ─────────────────────────────────────────────────────────────────────────────
def ensure_db_tables() -> None:
    """Create required tables if they don't exist. Safe to call repeatedly."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT,
                session_id TEXT
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
                decision_threshold REAL,
                reason TEXT
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
        # non‑fatal – simulation should continue even if DB unavailable
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
        if "reason" not in cols:
            cur.execute("ALTER TABLE events ADD COLUMN reason TEXT")
        conn.commit()
        conn.close()
    except Exception:
        pass


def log_event_db(notif_id: str, decision: str, speed: float, signal: int,
                 distraction_risk: float, prob_good_signal: float | None = None,
                 decision_threshold: float | None = None, reason: str = "") -> None:
    """Insert a single event row into the `events` table."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute(
            """INSERT INTO events(notif_id, decision, timestamp, speed, signal,
                                  distraction_risk, prob_good_signal,
                                  decision_threshold, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                notif_id,
                decision,
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                float(speed),
                int(signal),
                float(distraction_risk),
                (float(prob_good_signal) if prob_good_signal is not None else None),
                (float(decision_threshold) if decision_threshold is not None else None),
                reason,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Safety helpers (acceleration & distraction risk)
# ─────────────────────────────────────────────────────────────────────────────
_prev_speed = 0.0
_prev_time = None
_acc_history: List[float] = []


def get_acceleration(speed: float) -> float:
    """Compute smoothed acceleration (m/s^2) from speed (km/h)."""
    global _prev_speed, _prev_time, _acc_history
    current_time = time.time()
    if _prev_time is None:
        _prev_time = current_time
        _prev_speed = speed
        return 0.0
    dt = current_time - _prev_time
    if dt <= 0:
        return _acc_history[-1] if _acc_history else 0.0
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


def compute_distraction_risk(speed: float, accel: float, handover: int,
                             signal_fluctuation: bool) -> float:
    """Compute distraction risk per PART 2 rules."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Notification Manager (core intelligence from Version 1)
# ─────────────────────────────────────────────────────────────────────────────
class NotificationManager:
    """In‑memory notification queue manager with safety‑aware evaluation."""

    def __init__(self, soft_cap: int = 8):
        self.pending: List[dict] = []
        self.delivered: List[dict] = []
        self.soft_cap = soft_cap

    def add(self, message: str, priority: int = 5) -> dict:
        is_urgent = int(priority) <= 3
        # collapse exact duplicate deferred messages
        if not is_urgent:
            for p in self.pending:
                if p.get("message") == message and not p.get("urgent"):
                    p["repeat_count"] = p.get("repeat_count", 1) + 1
                    return p

        notif = {
            "id": str(uuid.uuid4())[:8],
            "message": message,
            "priority": priority,
            "urgent": is_urgent,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
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

    def evaluate_pending(self, frame: dict,
                         predictor: Optional[object] = None) -> List[dict]:
        """Evaluate pending notifications against the current frame.

        Returns list of delivered notifications (moved from pending to delivered).
        """
        delivered_now: List[dict] = []
        remaining: List[dict] = []
        nonurgent_sent = 0
        MAX_SEND_PER_TICK = 3

        def predicted_good(enriched: dict) -> bool:
            try:
                if predictor is None:
                    q = float(enriched.get("quality", 0) or 0)
                    r = float(enriched.get("rssi", -120) or -120)
                    return (q >= 0.7) or (r >= -95)
                if hasattr(predictor, "predict"):
                    out = predictor.predict(enriched)
                    prob = (out.get("prob_good_signal") or
                            out.get("probability") or
                            out.get("main_model_prob"))
                    if prob is None:
                        return False
                    return float(prob) >= 0.68
            except Exception:
                return False
            return False

        for notif in list(self.pending):
            enriched = {**frame}
            enriched["urgent"] = notif.get("urgent", False)
            enriched["queue_size"] = len(self.pending)
            enriched["repeated_count"] = notif.get("repeat_count", 1)
            try:
                created = datetime.fromisoformat(notif["created_at"].replace("Z", "+00:00"))
                age = (datetime.utcnow() - created).total_seconds()
                enriched["queue_age_seconds"] = age
            except Exception:
                enriched["queue_age_seconds"] = 0.0

            speed = float(frame.get("speed", 0) or 0)
            accel = float(frame.get("accel", 0) or 0)
            try:
                accel = get_acceleration(speed)
            except Exception:
                pass

            s = int(frame.get("signal", 0) or 0)
            s1 = int(frame.get("signal_1", s) or s)
            s2 = int(frame.get("signal_2", s1) or s1)
            signal_fluct = (abs(s - s1) + abs(s1 - s2)) >= 2

            handover = int(frame.get("handover", 0) or 0)
            distraction_risk = compute_distraction_risk(speed, accel, handover, signal_fluct)

            minimal_ui = False
            predictor_out = None
            if predictor is not None and hasattr(predictor, "predict"):
                try:
                    predictor_out = predictor.predict(enriched)
                    for k in ("prob_good_signal", "main_model_prob", "backup_model_prob",
                              "legacy_model_prob", "confidence", "anomaly"):
                        if k in predictor_out:
                            enriched[k] = predictor_out[k]
                except Exception:
                    predictor_out = None

            # Decision rules
            if enriched.get("urgent", False):
                minimal_ui = True
                decision = "SEND"
                reason = "urgent_override"
            elif predictor_out and bool(predictor_out.get("anomaly", False)):
                decision = "WAIT_ANOMALY"
                reason = "anomaly_detected"
            elif distraction_risk >= 2:
                decision = "WAIT_DISTRACTION"
                reason = "driver_distraction"
            elif accel < -2.0:
                decision = "WAIT_BRAKE"
                reason = "sudden_braking_block"
            elif speed > 60:
                decision = "WAIT_HIGH_SPEED"
                reason = "fast_highway_safety"
            else:
                prob = None
                if predictor_out is not None and "prob_good_signal" in predictor_out:
                    try:
                        prob = float(predictor_out.get("prob_good_signal", 0.0) or 0.0)
                    except Exception:
                        prob = None
                if prob is None:
                    try:
                        q = float(enriched.get("quality", 0) or 0)
                        r = float(enriched.get("rssi", -120) or -120)
                        prob = 1.0 if (q >= 0.7 or r >= -95) else 0.0
                    except Exception:
                        prob = 0.0

                decision_threshold = 0.6
                if prob > decision_threshold:
                    decision = "SEND"
                    reason = "model_confident_good"
                elif 0.4 <= prob <= decision_threshold:
                    decision = "WAIT_PREDICTED"
                    reason = "model_uncertain_wait"
                else:
                    decision = "QUEUE"
                    reason = "default_safe_queue"

            # Respect batch caps
            if decision.startswith("SEND") and (enriched.get("urgent", False) or nonurgent_sent < MAX_SEND_PER_TICK):
                delivered = {
                    **notif,
                    "delivered_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "decision": decision,
                    "reason": reason,
                    "confidence": float(enriched.get("confidence", frame.get("confidence", 0.0) or 0.0)),
                    "probability": float(enriched.get("prob_good_signal", frame.get("prob_good_signal", 0.0) or 0.0)),
                    "decision_threshold_used": float(enriched.get("decision_threshold", 0.6)),
                    "prob_good_signal": float(enriched.get("prob_good_signal", 0.0) or 0.0),
                    "main_model_prob": float(enriched.get("main_model_prob", 0.0) or 0.0),
                    "backup_model_prob": float(enriched.get("backup_model_prob", 0.0) or 0.0),
                    "legacy_model_prob": float(enriched.get("legacy_model_prob", 0.0) or 0.0),
                    "minimal_ui": bool(minimal_ui),
                    "distraction_risk": float(distraction_risk),
                    "acceleration": float(accel),
                }
                delivered_now.append(delivered)
                self.delivered.append(delivered)
                try:
                    log_event_db(
                        delivered.get("id"),
                        delivered.get("decision"),
                        speed,
                        int(frame.get("signal", 0) or 0),
                        delivered.get("distraction_risk", 0.0),
                        delivered.get("probability"),
                        delivered.get("decision_threshold_used"),
                        reason
                    )
                except Exception:
                    pass
                if not enriched.get("urgent", False):
                    nonurgent_sent += 1
            else:
                remaining.append(notif)

        self.pending = remaining
        return delivered_now


manager = NotificationManager()


# ═════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Smart Notify",
        "sim_running": sim.running,
        "route": sim.active_route,
        "session_id": sim.session_id,
    }


@app.get("/routes")
async def get_routes():
    routes_file = Path(__file__).parent.parent / "data" / "routes.json"
    if not routes_file.exists():
        raise HTTPException(404, "Routes not generated yet. Run pipeline first.")
    routes = json.loads(routes_file.read_text())
    return {
        name: {
            "description": r.get("description", ""),
            "route_type": r.get("route_type", ""),
            "num_waypoints": len(r.get("waypoints", [])),
        }
        for name, r in routes.items()
    }


class SimStartRequest(BaseModel):
    route: str = "city_route"
    speed_factor: float = 15.0
    notif_rate: float = 4.0


@app.post("/simulate/start")
async def start_simulation(req: SimStartRequest):
    """Start or restart the simulation loop."""
    restarted = False
    async with sim.lock:
        if sim.task and not sim.task.done():
            restarted = True
            sim.running = False
            sim.task.cancel()
            try:
                await sim.task
            except asyncio.CancelledError:
                pass

        sim.active_route = req.route
        sim.speed_factor = req.speed_factor
        sim.running = True
        sim.injected_notifs = []
        sim.latest_frame = None
        sim.session_id = str(uuid.uuid4())[:8]
        sim.task = asyncio.create_task(
            _run_sim_task(sim.session_id, req.route, req.speed_factor, req.notif_rate)
        )

    return {
        "status": "restarted" if restarted else "started",
        "route": req.route,
        "speed_factor": req.speed_factor,
        "session_id": sim.session_id,
    }


@app.post("/simulate/stop")
async def stop_simulation():
    async with sim.lock:
        sim.running = False
        if sim.task and not sim.task.done():
            sim.task.cancel()
            try:
                await sim.task
            except asyncio.CancelledError:
                pass
        sim.task = None

    return {"status": "stopped", "session_id": sim.session_id}


class NotifyRequest(BaseModel):
    message: str
    priority: int = 5  # 1=urgent ... 10=low


@app.post("/notify")
async def inject_notification(req: NotifyRequest):
    """Inject a notification into the engine during live simulation."""
    notif = {
        "id": str(uuid.uuid4())[:8],
        "message": req.message,
        "priority": req.priority,
    }
    sim.injected_notifs.append(notif)
    return {"status": "queued", "notif_id": notif["id"]}


@app.get("/metrics")
async def get_metrics():
    """Pull aggregated delivery metrics from SQLite."""
    if not DB_PATH.exists():
        return {"error": "No data yet. Run a simulation first."}
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    runs = conn.execute(
        "SELECT * FROM runs ORDER BY start_time DESC LIMIT 5"
    ).fetchall()

    deliveries = conn.execute(
        "SELECT decision, COUNT(*) as cnt FROM events GROUP BY decision"
    ).fetchall()

    total_sent = conn.execute(
        "SELECT COUNT(*) FROM events WHERE decision LIKE 'SEND%'"
    ).fetchone()[0]
    total_queued = conn.execute(
        "SELECT COUNT(*) FROM events WHERE decision = 'QUEUE'"
    ).fetchone()[0]
    delivery_rate = float(total_sent) / float(max(1, total_sent + total_queued))

    try:
        conn.execute(
            "INSERT INTO metrics(created_at, delivery_rate, avg_delay, total_sent, total_queued) VALUES (?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(timespec="seconds") + "Z",
             delivery_rate, None, int(total_sent), int(total_queued)),
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
    f = Path(__file__).parent.parent / "data" / "towers.json"
    if not f.exists():
        return []
    return json.loads(f.read_text())


# ═════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    """
    Real-time simulation stream.
    Frames are broadcast by the simulation task; this handler only receives
    client‑side injected notifications.
    """
    await ws.accept()
    sim.connected_ws.append(ws)
    print(f"[WS] Client connected. Total: {len(sim.connected_ws)}")

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "notify":
                sim.injected_notifs.append({
                    "id": str(uuid.uuid4())[:8],
                    "message": msg.get("message", "Client notification"),
                    "priority": msg.get("priority", 5),
                })
    except WebSocketDisconnect:
        if ws in sim.connected_ws:
            sim.connected_ws.remove(ws)
        print(f"[WS] Client disconnected. Remaining: {len(sim.connected_ws)}")


# ═════════════════════════════════════════════════════════════════════════════
# BACKGROUND SIMULATION TASK
# ═════════════════════════════════════════════════════════════════════════════

async def _run_sim_task(session_id: str, route: str, speed_factor: float, notif_rate: float):
    """Runs the simulation and pushes frames to all connected WS clients."""
    print(f"[Task] Simulation starting: {route} (session={session_id})")
    try:
        init_db()
        ensure_db_tables()

        # Log run start
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute(
                "INSERT INTO runs(start_time, session_id) VALUES (?, ?)",
                (datetime.utcnow().isoformat(timespec="seconds") + "Z", session_id)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        # Instantiate predictor (DecisionEngine)
        engine = None
        try:
            engine = DecisionEngine(EngineConfig())
        except Exception:
            engine = None

        async for frame in run_simulation(route, speed_factor, notif_rate=notif_rate):
            if not sim.running:
                print("[Task] Simulation stopped by user.")
                break

            frame["session_id"] = session_id

            # Move injected notifications into the manager queue
            while sim.injected_notifs:
                n = sim.injected_notifs.pop(0)
                manager.add(n.get("message", "Injected"), priority=n.get("priority", 5))

            # Evaluate pending notifications with safety & ML
            delivered = manager.evaluate_pending(frame, predictor=engine)
            for d in delivered:
                frame.setdefault("events", []).append({
                    "notif_id": d.get("id"),
                    "decision": d.get("decision"),
                    "wait_sec": 0,
                    "rssi": frame.get("rssi_filtered"),
                    "quality": frame.get("quality"),
                    "reason": d.get("reason"),
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
                if ws in sim.connected_ws:
                    sim.connected_ws.remove(ws)

    except asyncio.CancelledError:
        print("[Task] Simulation task cancelled.")
    except Exception as e:
        print(f"[Task] Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.running = False
        if sim.task and sim.task is asyncio.current_task():
            sim.task = None
        print("[Task] Simulation ended.")


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    init_db()
    ensure_db_tables()
    print("[API] Smart Notify API ready.")
    print("[API] Docs: http://localhost:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)