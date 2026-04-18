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
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.realtime_loop import run_simulation, DB_PATH, init_db
from engine.decision_engine import Notification

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
    message:  str
    priority: int = 5   # 1=urgent ... 10=low


@app.post("/notify")
async def inject_notification(req: NotifyRequest):
    """Inject a notification into the engine during live simulation."""
    notif = {
        "id":       str(uuid.uuid4())[:8],
        "message":  req.message,
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
        "SELECT * FROM simulation_runs ORDER BY started_at DESC LIMIT 5"
    ).fetchall()

    deliveries = conn.execute("""
        SELECT decision, COUNT(*) as cnt,
               AVG(wait_sec) as avg_wait,
               AVG(rssi)     as avg_rssi,
               AVG(quality)  as avg_quality
        FROM deliveries
        GROUP BY decision
    """).fetchall()

    conn.close()

    return {
        "runs": [dict(r) for r in runs],
        "deliveries_by_decision": [dict(d) for d in deliveries],
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
        "SELECT * FROM deliveries ORDER BY t DESC LIMIT ?", (limit,)
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
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "notify":
                sim.injected_notifs.append({
                    "id":       str(uuid.uuid4())[:8],
                    "message":  msg.get("message", "Client notification"),
                    "priority": msg.get("priority", 5),
                })

    except WebSocketDisconnect:
        if ws in sim.connected_ws:
            sim.connected_ws.remove(ws)
        print(f"[WS] Client disconnected. Remaining: {len(sim.connected_ws)}")


# ══════════════════════════════════════════════════════════════════
# BACKGROUND SIMULATION TASK
# ══════════════════════════════════════════════════════════════════

async def _run_sim_task(session_id: str, route: str, speed_factor: float, notif_rate: float):
    """
    Runs the simulation and pushes frames to all connected WS clients.
    Handles injected notifications from /notify endpoint.
    """
    print(f"[Task] Simulation starting: {route} (session={session_id})")
    try:
        init_db()

        async for frame in run_simulation(route, speed_factor, notif_rate=notif_rate):
            if not sim.running:
                print("[Task] Simulation stopped by user.")
                break

            frame["session_id"] = session_id

            # Handle injected notifications
            while sim.injected_notifs:
                n = sim.injected_notifs.pop(0)
                # Patch into current frame's events
                frame["events"].append({
                    "notif_id": n["id"],
                    "decision": "INJECTED_QUEUED",
                    "wait_sec": 0,
                    "rssi":     frame["rssi_filtered"],
                    "quality":  frame["quality"],
                    "reason":   f"Injected: {n['message']}",
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

    except Exception as e:
        print(f"[Task] Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sim.running = False
        if sim.task and sim.task is asyncio.current_task():
            sim.task = None
        print("[Task] Simulation ended.")


# ══════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    init_db()
    print("[API] Smart Notify API ready.")
    print("[API] Docs: http://localhost:8000/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
