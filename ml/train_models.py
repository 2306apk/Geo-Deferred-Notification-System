"""
PHASE 5 & 6: Feature Engineering + ML Model Training
- Feature engineering per timestep (sliding window)
- Gradient Boosting: P(good signal in next 5-10 sec)
- Isolation Forest: anomaly / unstable signal detection
- Saves trained models to disk
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

DATA_DIR   = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

WINDOW     = 5        # look-back window (seconds)
HORIZON    = 8        # predict signal quality in next N seconds
GOOD_RSSI  = -85.0   # dBm threshold for "good" signal


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def extract_features(steps: List[Dict], window: int = WINDOW) -> pd.DataFrame:
    """
    For each timestep, build a feature vector from a sliding window.

    Features:
      - mean_rssi_w        : mean filtered RSSI over last W steps
      - std_rssi_w         : variance proxy
      - trend_mean_w       : mean Kalman trend over window
      - min_rssi_w         : worst-case in window
      - max_rssi_w         : best-case in window
      - range_rssi_w       : max - min (stability measure)
      - speed              : vehicle speed m/s
      - in_dead_zone       : binary flag
      - stopped            : binary flag
      - time_since_good    : seconds since last good signal
      - good_frac_w        : fraction of window with good signal
      - rssi_slope_w       : linear slope over window (trend)
    """
    rows = []
    last_good_t = -9999.0

    for i in range(len(steps)):
        step   = steps[i]
        window_steps = steps[max(0, i - window + 1): i + 1]

        rssi_w  = np.array([s["rssi_filtered"] for s in window_steps])
        trend_w = np.array([s["rssi_trend"]    for s in window_steps])

        # Time since last good signal
        if step["rssi_filtered"] >= GOOD_RSSI:
            last_good_t = step["t"]
        time_since_good = step["t"] - last_good_t

        # Linear slope of RSSI in window
        if len(rssi_w) >= 2:
            x_    = np.arange(len(rssi_w))
            slope = float(np.polyfit(x_, rssi_w, 1)[0])
        else:
            slope = 0.0

        rows.append({
            "t":               step["t"],
            "lat":             step["lat"],
            "lon":             step["lon"],
            "mean_rssi_w":     float(rssi_w.mean()),
            "std_rssi_w":      float(rssi_w.std()),
            "trend_mean_w":    float(trend_w.mean()),
            "min_rssi_w":      float(rssi_w.min()),
            "max_rssi_w":      float(rssi_w.max()),
            "range_rssi_w":    float(rssi_w.max() - rssi_w.min()),
            "speed":           step["speed"],
            "in_dead_zone":    int(step["in_dead_zone"]),
            "stopped":         int(step["stopped"]),
            "time_since_good": float(np.clip(time_since_good, 0, 300)),
            "good_frac_w":     float(np.mean(rssi_w >= GOOD_RSSI)),
            "rssi_slope_w":    slope,
            # raw targets (will be used to build labels)
            "_rssi_raw":       step["rssi_raw"],
            "_rssi_filtered":  step["rssi_filtered"],
            "_route":          step["route"],
        })

    return pd.DataFrame(rows)


def build_labels(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    For GB: label = 1 if future signal (t+horizon) is good
    """
    df = df.copy()
    future_rssi = df["_rssi_filtered"].shift(-horizon)
    df["label_gb"] = (future_rssi >= GOOD_RSSI).astype(int)
    df = df.iloc[:-horizon]   # drop last rows without future
    return df


def get_feature_cols() -> List[str]:
    return [
        "mean_rssi_w", "std_rssi_w", "trend_mean_w",
        "min_rssi_w", "max_rssi_w", "range_rssi_w",
        "speed", "in_dead_zone", "stopped",
        "time_since_good", "good_frac_w", "rssi_slope_w",
    ]


# ══════════════════════════════════════════════════════════════════
# GRADIENT BOOSTING — P(good signal in next N sec)
# ══════════════════════════════════════════════════════════════════

def train_gradient_boosting(df: pd.DataFrame) -> Tuple:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feat_cols = get_feature_cols()
    df_clean  = df.dropna(subset=feat_cols + ["label_gb"])

    X = df_clean[feat_cols].values
    y = df_clean["label_gb"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"\n[GB] Training on {len(X_train)} samples "
          f"(pos={y_train.sum()}, neg={(1-y_train).sum()})")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train_s, y_train)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"[GB] AUC={auc:.4f} | F1={report['weighted avg']['f1-score']:.4f}")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = dict(zip(feat_cols, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:5]
    print("[GB] Top features:", top)

    return model, scaler, auc


# ══════════════════════════════════════════════════════════════════
# ISOLATION FOREST — anomaly / instability detection
# ══════════════════════════════════════════════════════════════════

def train_isolation_forest(df: pd.DataFrame) -> Tuple:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feat_cols = get_feature_cols()
    df_clean  = df.dropna(subset=feat_cols)

    X = df_clean[feat_cols].values

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    print(f"\n[IF] Training Isolation Forest on {len(X)} samples…")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,   # ~8% expected anomalies
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_s)

    scores  = model.decision_function(X_s)   # higher = more normal
    preds   = model.predict(X_s)              # -1 = anomaly, 1 = normal
    anomaly_frac = (preds == -1).mean()
    print(f"[IF] Anomaly fraction: {anomaly_frac:.3f}")

    return model, scaler


# ══════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_training():
    print("=" * 60)
    print("SMART NOTIFY — ML TRAINING PIPELINE")
    print("=" * 60)

    # Load Kalman-filtered data
    kalman_file = DATA_DIR / "kalman_data.json"
    if not kalman_file.exists():
        raise FileNotFoundError(
            "Run signal_processing/kalman_filter.py first!"
        )

    kalman_data = json.loads(kalman_file.read_text())

    # Feature engineering
    all_dfs = []
    for route_name, steps in kalman_data.items():
        df = extract_features(steps)
        df = build_labels(df)
        all_dfs.append(df)
        print(f"[Features] {route_name}: {len(df)} rows")

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"[Features] Total: {len(full_df)} rows | "
          f"label distribution: {full_df['label_gb'].value_counts().to_dict()}")

    # Train models
    gb_model,   gb_scaler,   auc = train_gradient_boosting(full_df)
    if_model,   if_scaler        = train_isolation_forest(full_df)

    # Save everything
    artifacts = {
        "gb_model":   gb_model,
        "gb_scaler":  gb_scaler,
        "if_model":   if_model,
        "if_scaler":  if_scaler,
        "feature_cols": get_feature_cols(),
        "gb_auc":     auc,
        "horizon":    HORIZON,
        "good_rssi_threshold": GOOD_RSSI,
    }

    for name, obj in artifacts.items():
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    print(f"\n[Training] All models saved to {MODELS_DIR}")
    return artifacts


if __name__ == "__main__":
    run_training()
