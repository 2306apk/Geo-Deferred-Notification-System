"""
PHASE 7: Decision Engine — The Brain of the System

Decision logic (in priority order):
  1. URGENT      → send immediately regardless of signal
  2. ANOMALY     → wait (Isolation Forest flagged instability)
  3. ML_BETTER   → wait (GB predicts signal improves soon)
  4. STABLE_GOOD → send (signal is good & stable right now)
  5. TIMEOUT     → force-send (waited too long, must deliver)
  6. QUEUE       → hold for next evaluation

Queue management:
  - Max queue age: configurable (default 60s)
  - Batch send: group low-priority notifications
  - Tracks delivery success / delay metrics
"""

import time
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [Engine] %(message)s")
log = logging.getLogger("decision_engine")

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_ORDER = [
    "mean_rssi_w", "std_rssi_w", "trend_mean_w",
    "range_rssi_w",
    "speed", "stopped",
    "time_since_good", "good_frac_w", "rssi_slope_w",
    "delta_rssi", "rolling_slope_w", "signal_var_w", "time_since_last_drop",
]

# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    # Signal thresholds
    good_rssi_threshold:   float = -85.0    # dBm — anything above = "good"
    stable_quality_min:    float = 0.45     # 0-1 quality score

    # ML thresholds
    gb_send_prob_min:      float = 0.65     # min P(good signal ahead) to send
    gb_wait_prob:          float = 0.70     # wait if future prob is THIS good
    anomaly_score_cutoff:  float = -0.1     # IF decision_function < this → anomaly

    # Timing
    max_queue_age_sec:     float = 60.0     # force-send after this wait
    urgent_timeout_sec:    float = 5.0      # urgent msgs get a tiny grace period
    batch_window_sec:      float = 3.0      # batch similar-priority msgs

    # Window for real-time features
    feature_window:        int   = 5

    # Heuristic fallback thresholds (used when ML models are unavailable)
    fallback_send_quality: float = 0.55
    fallback_send_quality_trend: float = 0.35
    fallback_wait_deadzone_age_frac: float = 0.7


class Decision(str, Enum):
    SEND_URGENT   = "SEND_URGENT"
    SEND_NOW      = "SEND_NOW"
    SEND_BATCH    = "SEND_BATCH"
    SEND_TIMEOUT  = "SEND_TIMEOUT"
    WAIT_ANOMALY  = "WAIT_ANOMALY"
    WAIT_BETTER   = "WAIT_BETTER"
    QUEUE         = "QUEUE"


@dataclass
class Notification:
    id:         str
    payload:    Dict[str, Any]
    priority:   int         = 5    # 1=urgent … 10=low
    created_at: float       = field(default_factory=time.time)
    attempts:   int         = 0
    delivered:  bool        = False
    decision_log: List[str] = field(default_factory=list)


@dataclass
class DeliveryEvent:
    notif_id:   str
    decision:   str
    timestamp:  float
    rssi:       float
    quality:    float
    wait_sec:   float
    reason:     str


# ══════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════

class ModelBundle:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.ready = False
        self.mode  = "heuristic"
        self._load(models_dir)

    def _load(self, d: Path):
        try:
            def _pkl(name):
                with open(d / f"{name}.pkl", "rb") as f:
                    return pickle.load(f)

            self.gb_model      = _pkl("gb_model")
            self.gb_scaler     = _pkl("gb_scaler")
            self.if_model      = _pkl("if_model")
            self.if_scaler     = _pkl("if_scaler")
            self.feature_cols  = _pkl("feature_cols")
            self.ready         = True
            self.mode          = "ml"
            log.info("Models loaded successfully.")
        except FileNotFoundError as e:
            log.warning(f"Models not found ({e}). Running without ML.")
        except Exception as e:
            log.warning(f"Model load failed ({e}). Running without ML.")
        if not hasattr(self, "feature_cols"):
            self.feature_cols = list(FEATURE_ORDER)

    def predict_gb(self, feat: np.ndarray) -> float:
        """Returns P(good signal ahead)."""
        if not self.ready:
            return 0.5
        Xs = self.gb_scaler.transform(feat.reshape(1, -1))
        return float(self.gb_model.predict_proba(Xs)[0, 1])

    def predict_if(self, feat: np.ndarray) -> Tuple[bool, float]:
        """Returns (is_anomaly, score). Score: lower = more anomalous."""
        if not self.ready:
            return False, 0.0
        Xs    = self.if_scaler.transform(feat.reshape(1, -1))
        score = float(self.if_model.decision_function(Xs)[0])
        pred  = self.if_model.predict(Xs)[0]   # -1=anomaly, 1=normal
        return pred == -1, score


# ══════════════════════════════════════════════════════════════════
# REAL-TIME FEATURE STATE
# ══════════════════════════════════════════════════════════════════

class SignalBuffer:
    """Rolling window of recent signal states for real-time features."""

    def __init__(self, window: int = 5, good_rssi: float = -85.0):
        self.window    = window
        self.good_rssi = good_rssi
        self._rssi     : List[float] = []
        self._trend    : List[float] = []
        self._t        : List[float] = []
        self.last_good_t = -9999.0
        self.last_drop_t = -9999.0
        self._prev_good: Optional[bool] = None

    def push(self, rssi_filtered: float, trend: float, t: float):
        is_good = rssi_filtered >= self.good_rssi
        if self._prev_good is True and not is_good:
            self.last_drop_t = t
        self._prev_good = is_good

        self._rssi.append(rssi_filtered)
        self._trend.append(trend)
        self._t.append(t)
        if is_good:
            self.last_good_t = t
        if len(self._rssi) > self.window:
            self._rssi.pop(0)
            self._trend.pop(0)
            self._t.pop(0)

    def build_feature_map(self, speed: float, in_dead_zone: bool,
                          stopped: bool, t: float) -> Dict[str, float]:
        r = np.array(self._rssi) if self._rssi else np.array([-100.0])
        tr = np.array(self._trend) if self._trend else np.array([0.0])

        time_since_good = t - self.last_good_t
        time_since_last_drop = t - self.last_drop_t
        if len(r) >= 2:
            slope = float(np.polyfit(np.arange(len(r)), r, 1)[0])
            delta_rssi = float(r[-1] - r[-2])
        else:
            slope = 0.0
            delta_rssi = 0.0

        return {
            "mean_rssi_w": float(r.mean()),
            "std_rssi_w": float(r.std()),
            "trend_mean_w": float(tr.mean()),
            "range_rssi_w": float(r.max() - r.min()),
            "speed": float(speed),
            "in_dead_zone": float(in_dead_zone),
            "stopped": float(stopped),
            "time_since_good": float(np.clip(time_since_good, 0, 300)),
            "good_frac_w": float(np.mean(r >= self.good_rssi)),
            "rssi_slope_w": float(slope),
            "rolling_slope_w": float(slope),
            "delta_rssi": float(delta_rssi),
            "signal_var_w": float(np.var(r)),
            "time_since_last_drop": float(np.clip(time_since_last_drop, 0, 300)),
        }


# ══════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ══════════════════════════════════════════════════════════════════

class DecisionEngine:
    def __init__(self, config: EngineConfig = None):
        self.cfg     = config or EngineConfig()
        self.models  = ModelBundle()
        self.buffer  = SignalBuffer(self.cfg.feature_window,
                                    self.cfg.good_rssi_threshold)
        self.queue   : List[Notification]  = []
        self.history : List[DeliveryEvent] = []
        self.metrics = {
            "total":        0,
            "sent":         0,
            "waited":       0,
            "force_sent":   0,
            "avg_wait_sec": 0.0,
            "avg_rssi_at_send": 0.0,
        }

    # ── Ingest sensor tick ────────────────────────────────────────────────
    def ingest(self, sensor: Dict[str, Any]):
        """
        Call every timestep with:
        {
          t, lat, lon, speed, stopped,
          rssi_raw, rssi_filtered, rssi_trend,
          signal_quality_filtered, in_dead_zone
        }
        """
        self.buffer.push(
            sensor["rssi_filtered"],
            sensor["rssi_trend"],
            sensor["t"],
        )
        self._current = sensor

    # ── Core decision logic ───────────────────────────────────────────────
    def decide(self, notif: Notification) -> Tuple[Decision, str]:
        sensor = self._current
        t      = sensor["t"]
        qual   = sensor["signal_quality_filtered"]
        age    = t - notif.created_at
        urgent = notif.priority <= 2

        # 1. URGENT — send immediately (with tiny grace)
        if urgent:
            if age >= self.cfg.urgent_timeout_sec or qual >= 0.3:
                return Decision.SEND_URGENT, f"urgent priority={notif.priority}"

        # If ML artifacts are unavailable, use resilient heuristics.
        if not self.models.ready:
            return self._decide_without_ml(notif)

        # 2. Build features for ML
        feat_map = self.buffer.build_feature_map(
            speed        = sensor["speed"],
            in_dead_zone = sensor["in_dead_zone"],
            stopped      = sensor["stopped"],
            t            = t,
        )
        feat_cols = self.models.feature_cols if self.models.feature_cols else FEATURE_ORDER
        feat = np.array([feat_map.get(c, 0.0) for c in feat_cols], dtype=float)

        is_anomaly, if_score = self.models.predict_if(feat)
        p_good_ahead         = self.models.predict_gb(feat)

        # 3. ANOMALY — signal unstable, don't waste bandwidth
        if is_anomaly and age < self.cfg.max_queue_age_sec * 0.8:
            return Decision.WAIT_ANOMALY, \
                f"anomaly detected (if_score={if_score:.3f})"

        # 4. ML says BETTER signal is coming — hold on
        if (p_good_ahead >= self.cfg.gb_wait_prob
                and age < self.cfg.max_queue_age_sec * 0.6):
            return Decision.WAIT_BETTER, \
                f"ML predicts better signal ahead (p={p_good_ahead:.2f})"

        # 5. STABLE + GOOD signal right now — send
        if (qual >= self.cfg.stable_quality_min
                and p_good_ahead >= self.cfg.gb_send_prob_min):
            return Decision.SEND_NOW, \
                f"signal good (q={qual:.2f}, p_ahead={p_good_ahead:.2f})"

        # 6. TIMEOUT — waited too long, force send
        if age >= self.cfg.max_queue_age_sec:
            return Decision.SEND_TIMEOUT, \
                f"max wait exceeded ({age:.0f}s)"

        # 7. DEFAULT — keep queuing
        return Decision.QUEUE, \
            f"holding (q={qual:.2f}, p_ahead={p_good_ahead:.2f}, age={age:.0f}s)"

    def _decide_without_ml(self, notif: Notification) -> Tuple[Decision, str]:
        """Heuristic fallback used when trained models are unavailable."""
        sensor = self._current
        t = sensor["t"]
        qual = sensor["signal_quality_filtered"]
        age = t - notif.created_at
        in_dead_zone = bool(sensor.get("in_dead_zone", False))
        trend = float(sensor.get("rssi_trend", 0.0))
        urgent = notif.priority <= 2

        if urgent and (age >= self.cfg.urgent_timeout_sec or qual >= 0.30):
            return Decision.SEND_URGENT, "heuristic urgent policy"

        if age >= self.cfg.max_queue_age_sec:
            return Decision.SEND_TIMEOUT, f"heuristic timeout ({age:.0f}s)"

        if qual >= self.cfg.fallback_send_quality:
            return Decision.SEND_NOW, (
                f"heuristic send: quality={qual:.2f} >= {self.cfg.fallback_send_quality:.2f}"
            )

        if (not in_dead_zone and trend > 0.25
                and qual >= self.cfg.fallback_send_quality_trend):
            return Decision.SEND_NOW, (
                f"heuristic send: improving trend={trend:.2f}, quality={qual:.2f}"
            )

        if in_dead_zone and age < self.cfg.max_queue_age_sec * self.cfg.fallback_wait_deadzone_age_frac:
            return Decision.WAIT_BETTER, "heuristic wait in dead-zone"

        return Decision.QUEUE, (
            f"heuristic hold (q={qual:.2f}, trend={trend:.2f}, age={age:.0f}s)"
        )

    # ── Queue management ──────────────────────────────────────────────────
    def add_notification(self, notif: Notification):
        self.queue.append(notif)
        self.metrics["total"] += 1
        log.debug(f"Queued [{notif.id}] priority={notif.priority}")

    def process_queue(self) -> List[DeliveryEvent]:
        """
        Process all queued notifications.
        Returns list of delivery events for this tick.
        """
        if not hasattr(self, "_current"):
            return []

        sensor   = self._current
        t        = sensor["t"]
        events   = []
        pending  = []

        for notif in self.queue:
            notif.attempts += 1
            decision, reason = self.decide(notif)
            notif.decision_log.append(f"t={t:.0f}: {decision.value} — {reason}")

            if decision in (Decision.SEND_URGENT, Decision.SEND_NOW,
                            Decision.SEND_BATCH, Decision.SEND_TIMEOUT):
                notif.delivered = True
                wait            = t - notif.created_at
                ev = DeliveryEvent(
                    notif_id  = notif.id,
                    decision  = decision.value,
                    timestamp = t,
                    rssi      = sensor["rssi_filtered"],
                    quality   = sensor["signal_quality_filtered"],
                    wait_sec  = wait,
                    reason    = reason,
                )
                events.append(ev)
                self.history.append(ev)
                self._update_metrics(ev, decision)

                log.info(f"SEND [{notif.id}] via {decision.value} | "
                         f"rssi={sensor['rssi_filtered']:.1f} dBm | "
                         f"wait={wait:.1f}s | {reason}")
            else:
                pending.append(notif)
                log.debug(f"HOLD [{notif.id}] {decision.value} | {reason}")

        self.queue = pending
        return events

    def _update_metrics(self, ev: DeliveryEvent, decision: Decision):
        self.metrics["sent"] += 1
        if ev.wait_sec > 0:
            self.metrics["waited"] += 1
        if decision == Decision.SEND_TIMEOUT:
            self.metrics["force_sent"] += 1
        n = self.metrics["sent"]
        self.metrics["avg_wait_sec"] = (
            (self.metrics["avg_wait_sec"] * (n-1) + ev.wait_sec) / n
        )
        self.metrics["avg_rssi_at_send"] = (
            (self.metrics["avg_rssi_at_send"] * (n-1) + ev.rssi) / n
        )

    def get_status(self) -> Dict:
        sensor = getattr(self, "_current", {})
        return {
            "metrics":      self.metrics,
            "queue_size":   len(self.queue),
            "model_mode":   self.models.mode,
            "current_rssi": sensor.get("rssi_filtered"),
            "current_quality": sensor.get("signal_quality_filtered"),
            "in_dead_zone": sensor.get("in_dead_zone"),
            "trend":        sensor.get("trend_label"),
        }


# ── Standalone smoke test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uuid

    engine = DecisionEngine()

    # Inject some fake sensor ticks
    test_ticks = [
        dict(t=i, lat=12.97, lon=77.59, speed=8.0, stopped=False,
             rssi_raw=-95.0 + np.random.normal(0, 3),
             rssi_filtered=-90.0, rssi_trend=0.5,
             signal_quality_filtered=0.43, in_dead_zone=False,
             trend_label="improving")
        for i in range(20)
    ]

    # Add some notifications
    for p in [1, 3, 7, 5, 2]:
        engine.add_notification(Notification(
            id=str(uuid.uuid4())[:8],
            payload={"msg": f"Test notification priority={p}"},
            priority=p,
            created_at=0.0,
        ))

    # Simulate
    for tick in test_ticks:
        engine.ingest(tick)
        events = engine.process_queue()
        for e in events:
            print(f"  → {e.decision} | rssi={e.rssi:.1f} | wait={e.wait_sec:.1f}s")

    print("\nFinal status:", engine.get_status())
