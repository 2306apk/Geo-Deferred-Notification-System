"""
PHASE 5 & 6: Feature Engineering + ML Model Training
- Feature engineering per timestep (sliding window)
- Gradient Boosting: P(good signal in next 5-10 sec)
- Isolation Forest: anomaly / unstable signal detection
- Strict validation:
  - Time-ordered split per route
  - Route-holdout evaluation
  - Minority-aware metrics + threshold sweeps
- Saves trained models + validation report to disk
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

WINDOW = 5       # look-back window (seconds)
HORIZON = 8      # evaluate majority signal quality over next N seconds
GOOD_RSSI = -85.0


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_features(steps: List[Dict[str, Any]], window: int = WINDOW) -> pd.DataFrame:
    """Build per-timestep features from a rolling window."""
    rows = []
    last_good_t = -9999.0
    last_drop_t = -9999.0
    prev_good = None

    for i in range(len(steps)):
        step = steps[i]
        window_steps = steps[max(0, i - window + 1): i + 1]

        rssi_w = np.array([s["rssi_filtered"] for s in window_steps], dtype=float)
        trend_w = np.array([s["rssi_trend"] for s in window_steps], dtype=float)

        is_good = step["rssi_filtered"] >= GOOD_RSSI
        if is_good:
            last_good_t = step["t"]
        if prev_good is True and not is_good:
            last_drop_t = step["t"]
        prev_good = is_good

        time_since_good = step["t"] - last_good_t
        time_since_last_drop = step["t"] - last_drop_t

        if len(rssi_w) >= 2:
            slope = float(np.polyfit(np.arange(len(rssi_w)), rssi_w, 1)[0])
            delta_rssi = float(rssi_w[-1] - rssi_w[-2])
        else:
            slope = 0.0
            delta_rssi = 0.0

        rows.append({
            "t": float(step["t"]),
            "lat": float(step["lat"]),
            "lon": float(step["lon"]),
            "mean_rssi_w": float(rssi_w.mean()),
            "std_rssi_w": float(rssi_w.std()),
            "trend_mean_w": float(trend_w.mean()),
            "min_rssi_w": float(rssi_w.min()),
            "max_rssi_w": float(rssi_w.max()),
            "range_rssi_w": float(rssi_w.max() - rssi_w.min()),
            "speed": float(step["speed"]),
            "in_dead_zone": int(step["in_dead_zone"]),
            "stopped": int(step["stopped"]),
            "time_since_good": float(np.clip(time_since_good, 0, 300)),
            "good_frac_w": float(np.mean(rssi_w >= GOOD_RSSI)),
            "rssi_slope_w": slope,
            "rolling_slope_w": slope,
            "delta_rssi": delta_rssi,
            "signal_var_w": float(np.var(rssi_w)),
            "time_since_last_drop": float(np.clip(time_since_last_drop, 0, 300)),
            "_rssi_raw": float(step["rssi_raw"]),
            "_rssi_filtered": float(step["rssi_filtered"]),
            "_route": step["route"],
        })

    return pd.DataFrame(rows)


def build_labels(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Binary label based on majority-good signal over the next horizon window."""
    df = df.copy()
    future_good = (df["_rssi_filtered"] >= GOOD_RSSI).astype(float)

    # Exclude current step: evaluate next K seconds [t+1, ..., t+K].
    future_frac = (
        future_good.shift(-1)
        .iloc[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .mean()
        .iloc[::-1]
    )
    df["future_good_frac"] = future_frac
    df["label_gb"] = (df["future_good_frac"] >= 0.5).astype(int)
    return df.dropna(subset=["future_good_frac"]).reset_index(drop=True)


def get_feature_cols() -> List[str]:
    return [
        "mean_rssi_w", "std_rssi_w", "trend_mean_w",
        "range_rssi_w",
        "speed", "stopped",
        "time_since_good", "good_frac_w", "rssi_slope_w",
        "delta_rssi", "rolling_slope_w", "signal_var_w", "time_since_last_drop",
    ]


# ============================================================================
# METRICS + VALIDATION HELPERS
# ============================================================================

def _class_weights(y: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y.astype(int), minlength=2)
    total = float(len(y))
    eps = 1e-9
    return {
        0: total / (2.0 * (counts[0] + eps)),
        1: total / (2.0 * (counts[1] + eps)),
    }


def _sample_weights(y: np.ndarray, class_weights: Dict[int, float]) -> np.ndarray:
    return np.array([class_weights[int(v)] for v in y], dtype=float)


def _upsample_minority(train_df: pd.DataFrame, label_col: str = "label_gb") -> pd.DataFrame:
    """Lightweight random upsampling to reduce temporal class imbalance."""
    counts = train_df[label_col].value_counts().to_dict()
    if len(counts) < 2:
        return train_df

    majority_label = max(counts, key=counts.get)
    minority_label = min(counts, key=counts.get)
    majority_count = int(counts[majority_label])
    minority_count = int(counts[minority_label])

    if minority_count == 0 or majority_count <= minority_count:
        return train_df

    # Keep upsampling practical for hackathon speed.
    target_minority = min(majority_count, minority_count * 3)
    minority_df = train_df[train_df[label_col] == minority_label]
    upsampled = minority_df.sample(n=target_minority - minority_count, replace=True, random_state=42)
    return pd.concat([train_df, upsampled], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def _safe_div(num: float, den: float) -> float:
        if den == 0:
            return 0.0
        return float(num / den)

    precision_0 = _safe_div(tn, tn + fn)
    recall_0 = _safe_div(tn, tn + fp)
    f1_0 = _safe_div(2 * precision_0 * recall_0, precision_0 + recall_0)

    precision_1 = _safe_div(tp, tp + fp)
    recall_1 = _safe_div(tp, tp + fn)
    f1_1 = _safe_div(2 * precision_1 * recall_1, precision_1 + recall_1)

    balanced_accuracy = (recall_0 + recall_1) / 2.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    except ValueError:
        roc_auc = float("nan")

    pr_auc_pos = float("nan")
    if (y_true == 1).any():
        pr_auc_pos = float(average_precision_score(y_true, y_prob))

    pr_auc_neg = float("nan")
    if (y_true == 0).any():
        pr_auc_neg = float(average_precision_score(1 - y_true, 1 - y_prob))

    metrics = {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc_pos": pr_auc_pos,
        "pr_auc_neg": pr_auc_neg,
        "balanced_accuracy": float(balanced_accuracy),
        "precision_class_0": float(precision_0),
        "recall_class_0": float(recall_0),
        "f1_class_0": float(f1_0),
        "support_class_0": int((y_true == 0).sum()),
        "precision_class_1": float(precision_1),
        "recall_class_1": float(recall_1),
        "f1_class_1": float(f1_1),
        "support_class_1": int((y_true == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    thresholds = np.linspace(0.10, 0.90, 17)
    rows = [_evaluate(y_true, y_prob, threshold=float(t)) for t in thresholds]

    best_balanced = max(rows, key=lambda x: x["balanced_accuracy"])
    best_recall_0 = max(rows, key=lambda x: (x["recall_class_0"], x["balanced_accuracy"]))

    return {
        "rows": rows,
        "best_balanced": best_balanced,
        "best_recall_class_0": best_recall_0,
    }


def _train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: List[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    train_df = _upsample_minority(train_df)

    X_train = train_df[feat_cols].values
    y_train = train_df["label_gb"].values.astype(int)
    X_test = test_df[feat_cols].values

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        only_class = int(unique_classes[0])
        y_prob = np.full(len(test_df), float(only_class), dtype=float)
        train_meta = {
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "class_weights": {"class_0": None, "class_1": None},
            "train_label_distribution": {
                "0": int((y_train == 0).sum()),
                "1": int((y_train == 1).sum()),
            },
            "test_label_distribution": {
                "0": int((test_df["label_gb"].values == 0).sum()),
                "1": int((test_df["label_gb"].values == 1).sum()),
            },
            "skipped_model_training": True,
            "skip_reason": "single_class_train_split",
            "constant_probability": float(only_class),
        }
        return y_prob, train_meta

    class_weights = _class_weights(y_train)
    weights = _sample_weights(y_train, class_weights)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train_s, y_train, sample_weight=weights)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    train_meta = {
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "class_weights": {
            "class_0": float(class_weights[0]),
            "class_1": float(class_weights[1]),
        },
        "train_label_distribution": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        },
        "test_label_distribution": {
            "0": int((test_df["label_gb"].values == 0).sum()),
            "1": int((test_df["label_gb"].values == 1).sum()),
        },
    }
    return y_prob, train_meta


def _find_time_split_with_both_classes(
    df: pd.DataFrame,
    min_train_frac: float = 0.6,
    max_train_frac: float = 0.9,
    min_train_rows: int = 50,
    min_test_rows: int = 20,
) -> int:
    """Find a chronological split where both train and test contain both classes."""
    n = len(df)
    start = max(int(n * min_train_frac), min_train_rows)
    end = min(int(n * max_train_frac), n - min_test_rows)
    if start >= end:
        return -1

    for split_idx in range(start, end + 1):
        train_y = df.iloc[:split_idx]["label_gb"].values
        test_y = df.iloc[split_idx:]["label_gb"].values
        if len(np.unique(train_y)) == 2 and len(np.unique(test_y)) == 2:
            return split_idx
    return -1


def _find_transition_window_split(
    df: pd.DataFrame,
    min_train_rows: int = 50,
    min_test_rows: int = 20,
) -> Tuple[int, int]:
    """Fallback split: choose a contiguous future window containing both classes."""
    n = len(df)
    if n < (min_train_rows + min_test_rows):
        return -1, -1

    window = max(min_test_rows, int(0.2 * n))
    latest_start = n - window
    for start in range(min_train_rows, latest_start + 1):
        end = start + window
        train_y = df.iloc[:start]["label_gb"].values
        test_y = df.iloc[start:end]["label_gb"].values
        if len(np.unique(train_y)) == 2 and len(np.unique(test_y)) == 2:
            return start, end

    return -1, -1


def run_time_ordered_validation(per_route_df: Dict[str, pd.DataFrame], feat_cols: List[str]) -> Dict[str, Any]:
    results = []
    skipped = []

    for route_name, df in per_route_df.items():
        df = df.sort_values("t").dropna(subset=feat_cols + ["label_gb"]).reset_index(drop=True)
        if len(df) < 50:
            skipped.append({"route": route_name, "reason": "too_few_rows"})
            continue

        if len(np.unique(df["label_gb"].values)) < 2:
            skipped.append({"route": route_name, "reason": "single_class_route"})
            continue

        split_idx = _find_time_split_with_both_classes(df)
        split_strategy = "tail_split"
        if split_idx >= 0:
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
        else:
            start, end = _find_transition_window_split(df)
            if start < 0:
                skipped.append({"route": route_name, "reason": "no_valid_time_split_with_both_classes"})
                continue
            train_df = df.iloc[:start].copy()
            test_df = df.iloc[start:end].copy()
            split_idx = start
            split_strategy = "transition_window_split"

        y_true = test_df["label_gb"].values.astype(int)
        y_prob, train_meta = _train_and_predict(train_df, test_df, feat_cols)

        base = _evaluate(y_true, y_prob, threshold=0.5)
        sweep = _threshold_sweep(y_true, y_prob)

        results.append({
            "route": route_name,
            "split_idx": int(split_idx),
            "split_strategy": split_strategy,
            "base_threshold_metrics": base,
            "threshold_sweep": sweep,
            "train_meta": train_meta,
        })

    return {
        "per_route": results,
        "skipped_routes": skipped,
        "recommended_next_step": (
            "If many routes are skipped, use rolling-origin validation with wider windows "
            "or increase simulation length to include more good/bad transitions."
        ),
    }


def run_route_holdout_validation(per_route_df: Dict[str, pd.DataFrame], feat_cols: List[str]) -> Dict[str, Any]:
    routes = sorted(per_route_df.keys())
    results = []
    skipped = []

    for heldout in routes:
        train_parts = []
        for r in routes:
            if r != heldout:
                train_parts.append(per_route_df[r])
        if not train_parts:
            continue

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = per_route_df[heldout].copy()

        train_df = train_df.dropna(subset=feat_cols + ["label_gb"]).reset_index(drop=True)
        test_df = test_df.dropna(subset=feat_cols + ["label_gb"]).reset_index(drop=True)
        if len(train_df) < 50 or len(test_df) < 20:
            skipped.append({"heldout_route": heldout, "reason": "insufficient_rows"})
            continue

        if len(np.unique(train_df["label_gb"].values)) < 2 or len(np.unique(test_df["label_gb"].values)) < 2:
            skipped.append({"heldout_route": heldout, "reason": "single_class_fold"})
            continue

        y_true = test_df["label_gb"].values.astype(int)
        y_prob, train_meta = _train_and_predict(train_df, test_df, feat_cols)

        base = _evaluate(y_true, y_prob, threshold=0.5)
        sweep = _threshold_sweep(y_true, y_prob)

        results.append({
            "heldout_route": heldout,
            "base_threshold_metrics": base,
            "threshold_sweep": sweep,
            "train_meta": train_meta,
        })

    return {"folds": results, "skipped_folds": skipped}


def choose_global_threshold(time_ordered: Dict[str, Any]) -> float:
    """Choose a practical global threshold by maximizing mean balanced accuracy."""
    per_route = time_ordered.get("per_route", [])
    if not per_route:
        return 0.5

    candidates = np.linspace(0.20, 0.80, 13)
    threshold_scores: Dict[float, List[float]] = {float(t): [] for t in candidates}

    for row in per_route:
        sweep_rows = row.get("threshold_sweep", {}).get("rows", [])
        by_t = {round(float(s["threshold"]), 2): s for s in sweep_rows}
        for t in candidates:
            key = round(float(t), 2)
            if key in by_t:
                threshold_scores[float(t)].append(float(by_t[key]["balanced_accuracy"]))

    agg = []
    for t, vals in threshold_scores.items():
        if vals:
            agg.append((t, float(np.mean(vals))))

    if not agg:
        return 0.5

    agg.sort(key=lambda x: (x[1], -abs(x[0] - 0.5)), reverse=True)
    return float(agg[0][0])


def _write_validation_report(summary: Dict[str, Any], out_md: Path):
    lines: List[str] = []
    lines.append("# Smart Notify Validation Summary")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Total rows: {summary['dataset']['total_rows']}")
    lines.append(f"- Label distribution: {summary['dataset']['label_distribution']}")
    lines.append(f"- Feature columns: {', '.join(summary['dataset']['feature_cols'])}")
    lines.append(f"- Label definition: {summary['dataset'].get('label_definition', 'n/a')}")
    lines.append(f"- Recommended threshold (balanced accuracy): {summary.get('recommended_decision_threshold', 0.5):.2f}")
    lines.append("")

    lines.append("## Time-Ordered Validation (per route)")
    for row in summary["time_ordered"]["per_route"]:
        m = row["base_threshold_metrics"]
        b = row["threshold_sweep"]["best_balanced"]
        c0 = row["threshold_sweep"]["best_recall_class_0"]
        lines.append(
            f"- {row['route']}: base@0.5 bal_acc={m['balanced_accuracy']:.4f}, "
            f"recall_0={m['recall_class_0']:.4f}, pr_auc_pos={m['pr_auc_pos']:.4f}, "
            f"pr_auc_neg={m['pr_auc_neg']:.4f}; "
            f"best_balanced@{b['threshold']:.2f}={b['balanced_accuracy']:.4f}; "
            f"best_recall_0@{c0['threshold']:.2f}={c0['recall_class_0']:.4f}"
        )
    if summary["time_ordered"].get("skipped_routes"):
        lines.append("- Skipped routes:")
        for s in summary["time_ordered"]["skipped_routes"]:
            lines.append(f"  - {s['route']}: {s['reason']}")
    lines.append("")

    lines.append("## Route-Holdout Validation")
    for row in summary["route_holdout"]["folds"]:
        m = row["base_threshold_metrics"]
        b = row["threshold_sweep"]["best_balanced"]
        c0 = row["threshold_sweep"]["best_recall_class_0"]
        lines.append(
            f"- holdout {row['heldout_route']}: base@0.5 bal_acc={m['balanced_accuracy']:.4f}, "
            f"recall_0={m['recall_class_0']:.4f}, pr_auc_pos={m['pr_auc_pos']:.4f}, "
            f"pr_auc_neg={m['pr_auc_neg']:.4f}; "
            f"best_balanced@{b['threshold']:.2f}={b['balanced_accuracy']:.4f}; "
            f"best_recall_0@{c0['threshold']:.2f}={c0['recall_class_0']:.4f}"
        )
    if summary["route_holdout"].get("skipped_folds"):
        lines.append("- Skipped folds:")
        for s in summary["route_holdout"]["skipped_folds"]:
            lines.append(f"  - {s['heldout_route']}: {s['reason']}")

    out_md.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_gradient_boosting(df: pd.DataFrame) -> Tuple[Any, Any, Dict[str, Any]]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    feat_cols = get_feature_cols()
    df_clean = df.dropna(subset=feat_cols + ["label_gb"]).reset_index(drop=True)

    df_balanced = _upsample_minority(df_clean)

    X = df_balanced[feat_cols].values
    y = df_balanced["label_gb"].values.astype(int)

    class_weights = _class_weights(y)
    weights = _sample_weights(y, class_weights)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
        verbose=0,
    )
    model.fit(X_s, y, sample_weight=weights)

    y_prob = model.predict_proba(X_s)[:, 1]
    train_metrics = _evaluate(y, y_prob, threshold=0.5)
    importances = dict(zip(feat_cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:8]

    meta = {
        "train_samples": int(len(df_balanced)),
        "class_weights": {
            "class_0": float(class_weights[0]),
            "class_1": float(class_weights[1]),
        },
        "label_distribution": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "train_metrics": train_metrics,
        "top_features": [{"feature": k, "importance": float(v)} for k, v in top_features],
        "upsampling_applied": int(len(df_balanced)) > int(len(df_clean)),
    }

    print(
        f"\n[GB] Final model trained on {len(df_balanced)} rows "
        f"(class_0={(y == 0).sum()}, class_1={(y == 1).sum()})"
    )
    print(f"[GB] Class weights: {meta['class_weights']}")
    print(f"[GB] Top features: {[(x['feature'], round(x['importance'], 4)) for x in meta['top_features'][:5]]}")

    return model, scaler, meta


def train_isolation_forest(df: pd.DataFrame) -> Tuple[Any, Any, Dict[str, Any]]:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feat_cols = get_feature_cols()
    df_clean = df.dropna(subset=feat_cols).reset_index(drop=True)

    X = df_clean[feat_cols].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_s)

    preds = model.predict(X_s)
    anomaly_frac = float((preds == -1).mean())

    meta = {
        "train_samples": int(len(df_clean)),
        "anomaly_fraction": anomaly_frac,
    }

    print(f"\n[IF] Trained on {len(df_clean)} rows")
    print(f"[IF] Anomaly fraction: {anomaly_frac:.3f}")

    return model, scaler, meta


def run_training() -> Dict[str, Any]:
    print("=" * 64)
    print("SMART NOTIFY — ML TRAINING + STRICT VALIDATION")
    print("=" * 64)

    kalman_file = DATA_DIR / "kalman_data.json"
    if not kalman_file.exists():
        raise FileNotFoundError("Run signal_processing/kalman_filter.py first")

    kalman_data = json.loads(kalman_file.read_text(encoding="utf-8"))

    per_route_df: Dict[str, pd.DataFrame] = {}
    route_horizons: Dict[str, int] = {}
    for route_name, steps in kalman_data.items():
        feat_df = extract_features(steps)

        # Try shorter future windows if needed so each route has both classes.
        chosen_df = None
        chosen_horizon = HORIZON
        for h in range(HORIZON, 2, -1):
            candidate = build_labels(feat_df, horizon=h)
            if len(candidate) >= 50 and len(np.unique(candidate["label_gb"].values)) == 2:
                chosen_df = candidate
                chosen_horizon = h
                break

        if chosen_df is None:
            chosen_df = build_labels(feat_df, horizon=HORIZON)

        df = chosen_df
        per_route_df[route_name] = df
        route_horizons[route_name] = int(chosen_horizon)
        print(f"[Features] {route_name}: {len(df)} rows (horizon={chosen_horizon})")

    full_df = pd.concat(list(per_route_df.values()), ignore_index=True)
    feat_cols = get_feature_cols()
    label_dist = full_df["label_gb"].value_counts().to_dict()

    print(f"[Features] Total: {len(full_df)} rows")
    print(f"[Features] Label distribution: {label_dist}")

    time_ordered = run_time_ordered_validation(per_route_df, feat_cols)
    route_holdout = run_route_holdout_validation(per_route_df, feat_cols)

    validation_summary = {
        "dataset": {
            "total_rows": int(len(full_df)),
            "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
            "feature_cols": feat_cols,
            "horizon": HORIZON,
            "route_horizons": route_horizons,
            "good_rssi_threshold": GOOD_RSSI,
            "label_definition": "majority of next horizon seconds are GOOD",
        },
        "time_ordered": time_ordered,
        "route_holdout": route_holdout,
    }

    optimal_threshold = choose_global_threshold(time_ordered)
    validation_summary["recommended_decision_threshold"] = optimal_threshold

    validation_json = ARTIFACTS_DIR / "validation_summary.json"
    validation_md = ARTIFACTS_DIR / "validation_summary.md"
    validation_json.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    _write_validation_report(validation_summary, validation_md)
    print(f"[Validation] Saved: {validation_json}")
    print(f"[Validation] Saved: {validation_md}")

    gb_model, gb_scaler, gb_meta = train_gradient_boosting(full_df)
    if_model, if_scaler, if_meta = train_isolation_forest(full_df)

    artifacts = {
        "gb_model": gb_model,
        "gb_scaler": gb_scaler,
        "if_model": if_model,
        "if_scaler": if_scaler,
        "feature_cols": feat_cols,
        "horizon": HORIZON,
        "good_rssi_threshold": GOOD_RSSI,
        "gb_optimal_threshold": optimal_threshold,
    }

    for name, obj in artifacts.items():
        with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    training_summary = {
        "gb": gb_meta,
        "if": if_meta,
        "recommended_decision_threshold": optimal_threshold,
        "validation_files": {
            "json": str(validation_json),
            "markdown": str(validation_md),
        },
    }
    (ARTIFACTS_DIR / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n[Training] Model artifacts saved to {MODELS_DIR}")
    return {
        "artifacts": artifacts,
        "validation_summary": validation_summary,
        "training_summary": training_summary,
    }


if __name__ == "__main__":
    run_training()
