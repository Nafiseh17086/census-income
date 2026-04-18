"""
Customer segmentation for the 1994/95 CPS dataset.

Approach (see report for justification):

  1. Restrict to a marketing-actionable, low-cardinality feature set covering
     *life stage*, *economic activity*, and *capital / investment behaviour*
     -- things a retailer can plausibly condition marketing on.
  2. One-hot encode categoricals, standardise numerics, and reduce to 10
     principal components so K-Means operates in a dense, decorrelated space.
  3. Use MiniBatchKMeans with k selected by silhouette score and elbow over
     k = 3..10 on a stratified sample of 30k rows (for tractability).
  4. Refit the chosen k on the FULL cleaned dataset and profile segments by
     label, demographics, and employment characteristics.

Outputs:
  outputs/segmentation_summary.json
  outputs/segment_profiles.csv
  figures/09_elbow_silhouette.png
  figures/10_segments_scatter.png
  figures/11_segment_profiles.png
  outputs/segmenter.joblib
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import (  # noqa: E402
    LABEL_COL, WEIGHT_COL,
    build_preprocessor, drop_exact_duplicates, load_raw,
)

DATA_PATH = ROOT / "data" / "census-bureau.data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 42

# Marketing-actionable feature subset. Deliberately excludes the many
# "Not in universe" migration / veteran columns (low info) and highly
# granular industry / occupation recodes (noisy without supervision).
SEG_NUMERIC = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]
SEG_CATEGORICAL = [
    "education",
    "marital_stat",
    "sex",
    "race",
    "class_of_worker",
    "major_occupation_code",
    "full_or_part_time_employment_stat",
    "tax_filer_stat",
    "detailed_household_summary_in_household",
    "citizenship",
]


def main():
    t0 = time.time()
    print("Loading data ...")
    df = drop_exact_duplicates(load_raw(DATA_PATH))
    print(f"Rows: {len(df):,}")

    X = df[SEG_NUMERIC + SEG_CATEGORICAL].copy()
    y = df[LABEL_COL].values
    w = df[WEIGHT_COL].values

    pre = build_preprocessor(SEG_NUMERIC, SEG_CATEGORICAL, scale_numeric=True)
    Z = pre.fit_transform(X)
    # Dense matrix for PCA; dim should be modest (<~80).
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    print(f"Transformed dim: {Z.shape[1]}")

    # PCA to 10 dims for k-means stability.
    pca = PCA(n_components=10, random_state=SEED)
    Zp = pca.fit_transform(Z)
    print(f"PCA explained variance (first 10): "
          f"{pca.explained_variance_ratio_.round(3).tolist()}  "
          f"cum={pca.explained_variance_ratio_.sum():.3f}")

    # --- choose k on a stratified sample ----------------------------------
    rng = np.random.default_rng(SEED)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    sample_idx = np.concatenate([
        rng.choice(idx_pos, size=min(5000, len(idx_pos)), replace=False),
        rng.choice(idx_neg, size=min(25000, len(idx_neg)), replace=False),
    ])
    Zp_sample = Zp[sample_idx]

    ks = list(range(3, 11))
    inertias, sils = [], []
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED,
                             n_init=5, batch_size=2048)
        labels = km.fit_predict(Zp_sample)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Zp_sample, labels, sample_size=8000, random_state=SEED))
        print(f"  k={k}  inertia={km.inertia_:.0f}  silhouette={sils[-1]:.4f}")

    best_k = ks[int(np.argmax(sils))]
    print(f"Selected k = {best_k} (by silhouette)")

    # Elbow + silhouette plot.
    fig, ax1 = plt.subplots(figsize=(6, 3.5))
    ax1.plot(ks, inertias, "o-", color="#4C78A8", label="inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia", color="#4C78A8")
    ax2 = ax1.twinx()
    ax2.plot(ks, sils, "s-", color="#F58518", label="silhouette")
    ax2.set_ylabel("silhouette", color="#F58518")
    ax1.axvline(best_k, color="gray", ls=":", lw=1)
    ax1.set_title(f"K selection (best k = {best_k})")
    fig.tight_layout()
    fig.savefig(FIGURES / "09_elbow_silhouette.png", dpi=140)
    plt.close(fig)

    # --- refit on full data ------------------------------------------------
    print(f"Refitting MiniBatchKMeans(k={best_k}) on full data ...")
    km = MiniBatchKMeans(n_clusters=best_k, random_state=SEED, n_init=10,
                         batch_size=4096, max_iter=300)
    labels = km.fit_predict(Zp)
    df["segment"] = labels

    # 2D scatter (PC1 vs PC2) of a 20k sample coloured by segment.
    sample = rng.choice(len(Zp), size=20000, replace=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for seg in range(best_k):
        m = labels[sample] == seg
        ax.scatter(Zp[sample][m, 0], Zp[sample][m, 1], s=2, alpha=0.4,
                   label=f"S{seg} (n={(labels == seg).sum():,})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Segments in PCA space (20k-row sample)")
    ax.legend(loc="upper right", fontsize=8, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "10_segments_scatter.png", dpi=140)
    plt.close(fig)

    # --- segment profiles --------------------------------------------------
    # Weighted positive rate, size, median age, median weeks worked, etc.
    # We compute both raw sizes and weighted population shares for marketing.
    rows = []
    for seg in sorted(df["segment"].unique()):
        sub = df[df["segment"] == seg]
        ws = sub[WEIGHT_COL]
        row = {
            "segment": int(seg),
            "n_rows": int(len(sub)),
            "share_of_sample": float(len(sub) / len(df)),
            "share_of_population_weighted":
                float(ws.sum() / df[WEIGHT_COL].sum()),
            "weighted_positive_rate":
                float((sub[LABEL_COL] * ws).sum() / ws.sum()),
            "median_age": float(sub["age"].median()),
            "median_weeks_worked": float(sub["weeks_worked_in_year"].median()),
            "mean_capital_gains": float(sub["capital_gains"].mean()),
            "mean_dividends": float(sub["dividends_from_stocks"].mean()),
            "pct_female":
                float((sub["sex"] == "Female").mean()),
            "top_education": sub["education"].mode().iat[0],
            "top_occupation": sub["major_occupation_code"].mode().iat[0],
            "top_marital": sub["marital_stat"].mode().iat[0],
            "top_class_of_worker": sub["class_of_worker"].mode().iat[0],
            "top_household_role": sub["detailed_household_summary_in_household"].mode().iat[0],
        }
        rows.append(row)
    profile = pd.DataFrame(rows).sort_values("weighted_positive_rate",
                                              ascending=False).reset_index(drop=True)
    profile.to_csv(OUTPUTS / "segment_profiles.csv", index=False)
    print(profile.to_string(index=False))

    # Profile bar chart.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    order = profile["segment"].tolist()
    axes[0].bar([f"S{s}" for s in order], profile["weighted_positive_rate"].values,
                color="#F58518")
    axes[0].set_ylabel("P(income > $50K) weighted")
    axes[0].set_title("Segment income propensity")
    axes[0].axhline(
        float((df[LABEL_COL] * df[WEIGHT_COL]).sum() / df[WEIGHT_COL].sum()),
        color="gray", ls="--", lw=1, label="overall mean")
    axes[0].legend()
    axes[1].bar([f"S{s}" for s in order],
                profile["share_of_population_weighted"].values, color="#4C78A8")
    axes[1].set_ylabel("Weighted share of population")
    axes[1].set_title("Segment size")
    fig.tight_layout()
    fig.savefig(FIGURES / "11_segment_profiles.png", dpi=140)
    plt.close(fig)

    # Persist pipeline.
    seg_pipeline = Pipeline([("preprocessor", pre), ("pca", pca), ("kmeans", km)])
    joblib.dump(seg_pipeline, OUTPUTS / "segmenter.joblib")

    summary = {
        "best_k": int(best_k),
        "pca_explained_variance": pca.explained_variance_ratio_.round(4).tolist(),
        "pca_cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "k_search": {
            "ks": ks, "inertia": inertias, "silhouette": sils,
        },
        "segment_profiles": profile.to_dict(orient="records"),
        "total_runtime_sec": float(time.time() - t0),
    }
    with (OUTPUTS / "segmentation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote outputs/segmentation_summary.json")


if __name__ == "__main__":
    main()
