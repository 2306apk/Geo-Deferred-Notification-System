#!/usr/bin/env python3
"""Sweep label/feature configurations and pick the most robust setup."""

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "kalman_data.json"
OUT_FILE = ROOT / "artifacts" / "model_config_sweep.json"

sys.path.insert(0, str(ROOT))
import ml.train_models as tm


def _safe_avg(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(mean(values))


def _build_per_route(kalman_data: Dict[str, Any], threshold: float, horizon: int) -> Dict[str, pd.DataFrame]:
    per_route: Dict[str, pd.DataFrame] = {}

    # Keep feature extraction behavior aligned with threshold under test.
    tm.GOOD_RSSI = threshold

    for route, steps in kalman_data.items():
        df = tm.extract_features(steps)
        future_rssi = df["_rssi_filtered"].shift(-horizon)
        df["label_gb"] = (future_rssi >= threshold).astype(int)
        df = df.iloc[:-horizon].copy()
        per_route[route] = df

    return per_route


def _collect_fold_metrics(section: Dict[str, Any], key: str) -> Dict[str, float]:
    rows = section.get(key, [])

    bal = []
    rec0 = []
    pr_neg = []
    support0 = []

    for row in rows:
        m = row["base_threshold_metrics"]
        bal.append(float(m["balanced_accuracy"]))
        rec0.append(float(m["recall_class_0"]))
        support0.append(int(m["support_class_0"]))
        pr = m["pr_auc_neg"]
        if pr == pr:  # not NaN
            pr_neg.append(float(pr))

    return {
        "avg_balanced_accuracy": _safe_avg(bal),
        "avg_recall_class_0": _safe_avg(rec0),
        "avg_pr_auc_neg": _safe_avg(pr_neg),
        "avg_support_class_0": _safe_avg([float(x) for x in support0]),
    }


def _score(result: Dict[str, Any]) -> float:
    # Prioritize minority recall and balanced accuracy, then PR-AUC-neg.
    return (
        0.50 * result["combined"]["avg_recall_class_0"]
        + 0.35 * result["combined"]["avg_balanced_accuracy"]
        + 0.15 * result["combined"]["avg_pr_auc_neg"]
    )


def run():
    kalman_data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    thresholds = [-85.0, -82.0, -80.0, -78.0]
    horizons = [8, 12, 16]
    feature_variants = ["full", "no_dead_zone"]

    results: List[Dict[str, Any]] = []

    for threshold in thresholds:
        for horizon in horizons:
            per_route = _build_per_route(kalman_data, threshold=threshold, horizon=horizon)
            base_features = tm.get_feature_cols()

            for variant in feature_variants:
                feat_cols = list(base_features)
                if variant == "no_dead_zone":
                    feat_cols = [c for c in feat_cols if c != "in_dead_zone"]

                time_ordered = tm.run_time_ordered_validation(per_route, feat_cols)
                route_holdout = tm.run_route_holdout_validation(per_route, feat_cols)

                time_stats = _collect_fold_metrics(time_ordered, "per_route")
                holdout_stats = _collect_fold_metrics(route_holdout, "folds")

                combined = {
                    "avg_balanced_accuracy": _safe_avg([
                        time_stats["avg_balanced_accuracy"],
                        holdout_stats["avg_balanced_accuracy"],
                    ]),
                    "avg_recall_class_0": _safe_avg([
                        time_stats["avg_recall_class_0"],
                        holdout_stats["avg_recall_class_0"],
                    ]),
                    "avg_pr_auc_neg": _safe_avg([
                        time_stats["avg_pr_auc_neg"],
                        holdout_stats["avg_pr_auc_neg"],
                    ]),
                    "avg_support_class_0": _safe_avg([
                        time_stats["avg_support_class_0"],
                        holdout_stats["avg_support_class_0"],
                    ]),
                }

                item = {
                    "threshold": threshold,
                    "horizon": horizon,
                    "feature_variant": variant,
                    "time_ordered": time_stats,
                    "route_holdout": holdout_stats,
                    "combined": combined,
                }
                item["score"] = _score(item)
                results.append(item)
                print(
                    f"[Sweep] thr={threshold:>5.1f} hor={horizon:>2} var={variant:<12} "
                    f"score={item['score']:.4f} rec0={combined['avg_recall_class_0']:.4f} "
                    f"bal={combined['avg_balanced_accuracy']:.4f}"
                )

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    best = ranked[0] if ranked else None

    output = {
        "best": best,
        "top5": ranked[:5],
        "all": ranked,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[Sweep] Saved {OUT_FILE}")


if __name__ == "__main__":
    run()
