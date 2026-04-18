"""
Train and evaluate income classifiers on the 1994/95 CPS dataset.

Produces:
  outputs/classifier_metrics.json   -- all eval metrics
  outputs/classifier_report.txt     -- human-readable summary
  outputs/feature_importance.csv    -- top features from LightGBM
  figures/06_roc_pr.png             -- ROC + PR curves
  figures/07_confusion.png          -- confusion matrix @ tuned threshold
  figures/08_feature_importance.png -- feature importance bar
  outputs/classifier.joblib         -- final pipeline (preprocessor + model)

Design choices (see report):
  * Stratified 70/15/15 train/val/test split, seeded.
  * CPS sample weights used as sample_weight during training AND evaluation.
  * Exact duplicates dropped before splitting.
  * Baseline: logistic regression (class_weight='balanced').
  * Main model: LightGBM with modest hyperparameter tuning; handles
    high-cardinality one-hot inputs and extreme class imbalance well.
  * Decision threshold tuned on validation set to maximise F1 (the default
    0.5 threshold is poor on a ~6% positive problem).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (  # noqa: E402
    LABEL_COL, WEIGHT_COL,
    NUMERIC_COLS, CATEGORICAL_COLS,
    build_preprocessor, drop_exact_duplicates, load_raw, split_features,
)

DATA_PATH = ROOT / "data" / "census-bureau.data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 42


def weighted_metrics(y, p, w, thresh=0.5):
    yhat = (p >= thresh).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
        "f1": float(f1_score(y, yhat, sample_weight=w)),
        "accuracy": float(((yhat == y) * w).sum() / w.sum()),
        "threshold": float(thresh),
    }


def tune_threshold(y_val, p_val, w_val) -> float:
    """Pick decision threshold that maximises weighted F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(
        y_val, p_val, sample_weight=w_val
    )
    # precision_recall_curve returns len(thresholds) = len(precision) - 1.
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # drop the last element (corresponds to recall=0 edge)
    if len(thresholds) == 0:
        return 0.5
    best_idx = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_idx])


def main():
    t0 = time.time()
    print("Loading data ...")
    df = drop_exact_duplicates(load_raw(DATA_PATH))
    X, y, w = split_features(df)
    print(f"Shape after dedup: X={X.shape}  positive rate={y.mean():.4f}")

    # ---- splits ---------------------------------------------------
    X_tv, X_test, y_tv, y_test, w_tv, w_test = train_test_split(
        X, y, w, test_size=0.15, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_tv, y_tv, w_tv, test_size=0.1765, stratify=y_tv, random_state=SEED
    )  # 0.1765 of 85% ~= 15% -> 70/15/15 overall
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ---- preprocessing -------------------------------------------
    pre = build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS, scale_numeric=True)
    pre.fit(X_train)

    Z_train = pre.transform(X_train)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    print(f"Transformed dim: {Z_train.shape[1]}")

    # ---- baseline: logistic regression ---------------------------
    print("Training baseline logistic regression ...")
    t = time.time()
    lr = LogisticRegression(
        solver="saga", penalty="l2", C=1.0, max_iter=400,
        class_weight="balanced", n_jobs=-1,
    )
    lr.fit(Z_train, y_train, sample_weight=w_train)
    lr_val_p = lr.predict_proba(Z_val)[:, 1]
    lr_test_p = lr.predict_proba(Z_test)[:, 1]
    lr_thresh = tune_threshold(y_val, lr_val_p, w_val)
    lr_val = weighted_metrics(y_val, lr_val_p, w_val, lr_thresh)
    lr_test = weighted_metrics(y_test, lr_test_p, w_test, lr_thresh)
    print(f"  LR val:  {lr_val}")
    print(f"  LR test: {lr_test}   ({time.time()-t:.1f}s)")

    # ---- main model: LightGBM ------------------------------------
    print("Training LightGBM ...")
    t = time.time()
    # Convert sparse to dense only if moderate size; LightGBM accepts CSR.
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=100,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        n_estimators=600,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        verbose=-1,
        random_state=SEED,
        n_jobs=-1,
    )
    gbm = lgb.LGBMClassifier(**params)
    gbm.fit(
        Z_train, y_train,
        sample_weight=w_train,
        eval_set=[(Z_val, y_val)],
        eval_sample_weight=[w_val],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )
    gbm_val_p = gbm.predict_proba(Z_val)[:, 1]
    gbm_test_p = gbm.predict_proba(Z_test)[:, 1]
    gbm_thresh = tune_threshold(y_val, gbm_val_p, w_val)
    gbm_val = weighted_metrics(y_val, gbm_val_p, w_val, gbm_thresh)
    gbm_test = weighted_metrics(y_test, gbm_test_p, w_test, gbm_thresh)
    print(f"  GBM val:  {gbm_val}")
    print(f"  GBM test: {gbm_test}   ({time.time()-t:.1f}s)")
    print(f"  best_iter={gbm.best_iteration_}")

    # ---- plots ----------------------------------------------------
    # ROC + PR curves for both models on test set.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, p, color in [
        ("LogReg", lr_test_p, "#4C78A8"),
        ("LightGBM", gbm_test_p, "#F58518"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, p, sample_weight=w_test)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p, sample_weight=w_test):.3f})", color=color)
        prec, rec, _ = precision_recall_curve(y_test, p, sample_weight=w_test)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p, sample_weight=w_test):.3f})", color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC curve (test)")
    axes[0].legend()
    axes[1].axhline(y_test.mean(), color="gray", ls="--", lw=1,
                    label=f"chance = {y_test.mean():.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("Precision-Recall (test)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "06_roc_pr.png", dpi=140)
    plt.close(fig)

    # Confusion matrix at tuned threshold (LightGBM).
    yhat = (gbm_test_p >= gbm_thresh).astype(int)
    cm = confusion_matrix(y_test, yhat, sample_weight=w_test)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["<=50K", ">50K"]); ax.set_yticklabels(["<=50K", ">50K"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (weighted) @ thr={gbm_thresh:.3f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIGURES / "07_confusion.png", dpi=140)
    plt.close(fig)

    # Feature importance.
    feat_names = pre.get_feature_names_out()
    gains = gbm.booster_.feature_importance(importance_type="gain")
    imp_df = (pd.DataFrame({"feature": feat_names, "gain": gains})
                .sort_values("gain", ascending=False)
                .reset_index(drop=True))
    imp_df.head(40).to_csv(OUTPUTS / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    imp_df.head(20).iloc[::-1].plot.barh(
        x="feature", y="gain", ax=ax, legend=False, color="#72B7B2")
    ax.set_xlabel("LightGBM gain")
    ax.set_title("Top 20 features (LightGBM gain)")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_feature_importance.png", dpi=140)
    plt.close(fig)

    # ---- persist ------------------------------------------------
    pipe = Pipeline([("preprocessor", pre), ("model", gbm)])
    joblib.dump(pipe, OUTPUTS / "classifier.joblib")

    metrics = {
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "transformed_dim": int(Z_train.shape[1]),
        "logreg": {"val": lr_val, "test": lr_test, "threshold": lr_thresh},
        "lightgbm": {"val": gbm_val, "test": gbm_test, "threshold": gbm_thresh,
                     "best_iteration": int(gbm.best_iteration_ or gbm.n_estimators)},
        "confusion_matrix_weighted": cm.astype(float).tolist(),
        "total_runtime_sec": float(time.time() - t0),
    }
    with (OUTPUTS / "classifier_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    # Human-readable report.
    lines = []
    lines.append("Classifier evaluation summary")
    lines.append("=" * 40)
    lines.append(f"Rows after dedup: {len(df):,}")
    lines.append(f"Train/Val/Test : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    lines.append(f"Positive rate (train / test): {y_train.mean():.4f} / {y_test.mean():.4f}")
    lines.append(f"Transformed feature dim     : {Z_train.shape[1]}")
    lines.append("")
    lines.append("Logistic regression (baseline)")
    lines.append(f"  test ROC-AUC = {lr_test['roc_auc']:.4f}   PR-AUC = {lr_test['pr_auc']:.4f}   F1 = {lr_test['f1']:.4f}  @ thr={lr_thresh:.3f}")
    lines.append("LightGBM (tuned)")
    lines.append(f"  test ROC-AUC = {gbm_test['roc_auc']:.4f}   PR-AUC = {gbm_test['pr_auc']:.4f}   F1 = {gbm_test['f1']:.4f}  @ thr={gbm_thresh:.3f}")
    lines.append(f"  best_iteration = {gbm.best_iteration_}")
    lines.append("")
    lines.append("Top 10 features by gain:")
    for _, row in imp_df.head(10).iterrows():
        lines.append(f"  {row['feature']:<60}  {row['gain']:>10.0f}")
    (OUTPUTS / "classifier_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
