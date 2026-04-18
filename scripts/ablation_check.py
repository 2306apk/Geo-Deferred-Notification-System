#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ml.train_models as tm

kalman = json.loads((ROOT / "data" / "kalman_data.json").read_text())

all_dfs = []
for _, steps in kalman.items():
    df = tm.build_labels(tm.extract_features(steps))
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

variants = {
    "full": tm.get_feature_cols(),
    "no_dead_zone": [c for c in tm.get_feature_cols() if c != "in_dead_zone"],
}

for name, cols in variants.items():
    work = df.dropna(subset=cols + ["label_gb"]).copy()
    X = work[cols].values
    y = work["label_gb"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    c0 = (y_train == 0).sum()
    c1 = (y_train == 1).sum()
    w0 = len(y_train) / (2 * max(c0, 1))
    w1 = len(y_train) / (2 * max(c1, 1))
    weights = np.where(y_train == 0, w0, w1)

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
    )
    model.fit(X_train_s, y_train, sample_weight=weights)

    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    bal = balanced_accuracy_score(y_test, y_pred)
    p, r, f1, s = precision_recall_fscore_support(y_test, y_pred, labels=[0,1], zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"[{name}] AUC={auc:.4f} BAL={bal:.4f} R0={r[0]:.4f} R1={r[1]:.4f} S0={s[0]} S1={s[1]}")
