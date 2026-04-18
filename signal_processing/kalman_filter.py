"""
PHASE 4: Kalman Filter for Signal Processing
- 1D Kalman filter tuned for RSSI smoothing
- Outputs: filtered signal + velocity (trend/rate of change)
- Can run in real-time (step-by-step) or batch
- No external library needed — pure numpy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class SignalKalmanFilter:
    """
    1D constant-velocity Kalman filter for RSSI.

    State vector: [rssi, rssi_rate_of_change]
    Measurement:  [rssi_raw]

    Tuning:
      process_noise (Q): how much signal can change per step
      measurement_noise (R): how noisy the raw sensor is
    """

    def __init__(self,
                 process_noise: float  = 0.5,
                 measurement_noise: float = 5.0,
                 dt: float = 1.0):
        self.dt  = dt
        self.Q_s = process_noise
        self.R_s = measurement_noise

        # State transition (constant velocity model)
        self.F = np.array([[1, dt],
                           [0, 1 ]], dtype=float)

        # Measurement matrix (we only measure position/rssi, not rate)
        self.H = np.array([[1, 0]], dtype=float)

        # Process noise covariance
        q = process_noise
        self.Q = np.array([[q*dt**3/3, q*dt**2/2],
                           [q*dt**2/2, q*dt      ]], dtype=float)

        # Measurement noise covariance
        self.R = np.array([[measurement_noise ** 2]], dtype=float)

        # State & covariance (will be initialised on first measurement)
        self.x  : Optional[np.ndarray] = None
        self.P  : Optional[np.ndarray] = None

    def reset(self, initial_rssi: float):
        self.x = np.array([[initial_rssi], [0.0]])   # [rssi, d_rssi/dt]
        self.P = np.eye(2) * 50.0                    # initial uncertainty

    def step(self, z: float) -> Tuple[float, float]:
        """
        Process one measurement.
        Returns (filtered_rssi, rssi_trend_dBm_per_sec)
        """
        if self.x is None:
            self.reset(z)
            return z, 0.0

        # ── Predict ────────────────────────────────────────────────────────
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # ── Update ─────────────────────────────────────────────────────────
        z_vec  = np.array([[z]])
        y      = z_vec - self.H @ x_pred                    # innovation
        S      = self.H @ P_pred @ self.H.T + self.R        # innovation cov
        K      = P_pred @ self.H.T @ np.linalg.inv(S)       # Kalman gain

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        filtered_rssi = float(self.x[0, 0])
        rssi_trend    = float(self.x[1, 0])   # dBm/sec — positive = improving
        return filtered_rssi, rssi_trend

    def batch(self, measurements: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run filter over a list of raw RSSI values.
        Returns (filtered, trends) as numpy arrays.
        """
        self.x = None
        filtered = np.zeros(len(measurements))
        trends   = np.zeros(len(measurements))
        for i, m in enumerate(measurements):
            filtered[i], trends[i] = self.step(m)
        return filtered, trends


def apply_kalman_to_dataset(signal_data: Dict[str, List[Dict]],
                             process_noise: float = 0.5,
                             measurement_noise: float = 5.0) -> Dict[str, List[Dict]]:
    """
    Apply Kalman filter to all routes in signal_data.
    Adds fields: rssi_filtered, rssi_trend, signal_quality_filtered
    """
    result = {}

    for route_name, steps in signal_data.items():
        kf   = SignalKalmanFilter(process_noise, measurement_noise)
        raw  = [s["rssi_raw"] for s in steps]
        filt, trends = kf.batch(raw)

        enriched = []
        for i, step in enumerate(steps):
            s = dict(step)
            s["rssi_filtered"]          = round(float(filt[i]),   2)
            s["rssi_trend"]             = round(float(trends[i]),  4)  # dBm/s
            # Normalise filtered to quality 0-1
            s["signal_quality_filtered"] = round(
                float(np.clip((filt[i] + 120) / 70, 0, 1)), 4
            )
            # Trend category
            if trends[i] >  0.3:
                s["trend_label"] = "improving"
            elif trends[i] < -0.3:
                s["trend_label"] = "degrading"
            else:
                s["trend_label"] = "stable"
            enriched.append(s)

        result[route_name] = enriched
        print(f"[Kalman] {route_name}: "
              f"raw_std={np.std(raw):.2f} → filtered_std={np.std(filt):.2f} dBm")

    return result


# ── Standalone usage ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from pathlib import Path

    DATA_DIR    = Path(__file__).parent.parent / "data"
    SIGNAL_FILE = DATA_DIR / "signal_data.json"
    KALMAN_FILE = DATA_DIR / "kalman_data.json"

    signal_data = json.loads(SIGNAL_FILE.read_text())
    kalman_data = apply_kalman_to_dataset(signal_data)

    KALMAN_FILE.write_text(json.dumps(kalman_data, indent=2))
    print(f"[Kalman] Saved to {KALMAN_FILE}")

    # Quick stats per route
    import numpy as np
    for name, steps in kalman_data.items():
        trends    = [s["rssi_trend"] for s in steps]
        improving = sum(1 for t in trends if t > 0.3)
        degrading = sum(1 for t in trends if t < -0.3)
        print(f"  {name}: improving={improving} stable={len(trends)-improving-degrading} degrading={degrading}")
