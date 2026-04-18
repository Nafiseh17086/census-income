"""
Exploratory Data Analysis script.

Writes summary statistics to outputs/eda_summary.json and figures to figures/.
Run as:  python -m src.eda  (from the project root) or  python src/eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import (
    LABEL_COL,
    WEIGHT_COL,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    load_raw,
    drop_exact_duplicates,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "census-bureau.data"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
OUTPUTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    df = load_raw(DATA_PATH)
    print(f"Raw shape: {df.shape}")

    dedup = drop_exact_duplicates(df)
    print(f"After dropping exact duplicates: {dedup.shape}  "
          f"(dropped {len(df) - len(dedup):,})")

    summary: dict = {
        "n_rows_raw": int(len(df)),
        "n_rows_dedup": int(len(dedup)),
        "n_columns": int(df.shape[1]),
        "label_distribution_unweighted": {
            "<=50K": int((df[LABEL_COL] == 0).sum()),
            ">50K": int((df[LABEL_COL] == 1).sum()),
        },
        "label_distribution_weighted": {
            "<=50K": float(df.loc[df[LABEL_COL] == 0, WEIGHT_COL].sum()),
            ">50K": float(df.loc[df[LABEL_COL] == 1, WEIGHT_COL].sum()),
        },
    }

    # Weighted positive rate.
    w = df[WEIGHT_COL]
    summary["positive_rate_unweighted"] = float(df[LABEL_COL].mean())
    summary["positive_rate_weighted"] = float(
        (df[LABEL_COL] * w).sum() / w.sum()
    )

    # Missingness (NaN introduced by '?' in load_raw).
    miss = df.isna().mean().sort_values(ascending=False)
    summary["columns_with_missing_top10"] = {
        c: float(miss[c]) for c in miss[miss > 0].head(10).index
    }

    # "Not in universe"-style sentinels: count per column.
    sentinels = [
        "Not in universe", "Not in universe or children",
        "Not in universe under 1 year old",
    ]
    nu_rates = {}
    for c in CATEGORICAL_COLS:
        s = df[c].astype("string")
        rate = s.isin(sentinels).mean()
        if rate > 0:
            nu_rates[c] = float(rate)
    summary["not_in_universe_top10"] = dict(
        sorted(nu_rates.items(), key=lambda kv: -kv[1])[:10]
    )

    # Numeric summaries (weighted vs unweighted for age, key features).
    numeric_desc = df[NUMERIC_COLS].describe().round(2).to_dict()
    summary["numeric_describe"] = numeric_desc

    # ---- Plots ------------------------------------------------------------
    # 1. Label distribution.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    counts = df[LABEL_COL].map({0: "<=50K", 1: ">50K"}).value_counts()
    counts.plot.bar(ax=ax, color=["#4C78A8", "#F58518"])
    ax.set_title("Income label distribution (unweighted)")
    ax.set_ylabel("Rows")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_label_distribution.png", dpi=140)
    plt.close(fig)

    # 2. Age histogram by class.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(df.loc[df[LABEL_COL] == 0, "age"], bins=50, alpha=0.6,
            label="<=50K", color="#4C78A8", density=True)
    ax.hist(df.loc[df[LABEL_COL] == 1, "age"], bins=50, alpha=0.6,
            label=">50K", color="#F58518", density=True)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.set_title("Age distribution by income class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "02_age_by_class.png", dpi=140)
    plt.close(fig)

    # 3. Top education levels, positive rate per level.
    ed = (df.groupby("education")
            .agg(n=(LABEL_COL, "size"),
                 pos_rate=(LABEL_COL, "mean"))
            .sort_values("pos_rate", ascending=False))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ed["pos_rate"].plot.barh(ax=ax, color="#54A24B")
    ax.set_xlabel("P(income > $50K)")
    ax.set_title("Positive rate by education level")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_education_pos_rate.png", dpi=140)
    plt.close(fig)

    # 4. Capital gains vs label (log scale because of heavy skew).
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for cls, color in [(0, "#4C78A8"), (1, "#F58518")]:
        vals = df.loc[df[LABEL_COL] == cls, "capital_gains"]
        vals = vals[vals > 0]
        ax.hist(np.log10(vals + 1), bins=40, alpha=0.6,
                label=f"{'<=50K' if cls==0 else '>50K'} (n={len(vals):,})",
                color=color, density=True)
    ax.set_xlabel("log10(capital_gains + 1)  (only rows with gains>0)")
    ax.set_ylabel("Density")
    ax.set_title("Capital gains by income class (positive values only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "04_capital_gains.png", dpi=140)
    plt.close(fig)

    # 5. Missingness bar chart (top 12).
    miss12 = miss[miss > 0].head(12)
    if len(miss12):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        miss12.sort_values().plot.barh(ax=ax, color="#B279A2")
        ax.set_xlabel("Fraction missing")
        ax.set_title("Columns with the highest NaN rate")
        fig.tight_layout()
        fig.savefig(FIGURES / "05_missingness.png", dpi=140)
        plt.close(fig)

    # Save summary JSON.
    summary["n_categorical"] = len(CATEGORICAL_COLS)
    summary["n_numeric"] = len(NUMERIC_COLS)
    summary["figures"] = sorted(str(p.name) for p in FIGURES.glob("*.png"))

    out_path = OUTPUTS / "eda_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_path}")

    # Print a few highlights to the console.
    print("\n-- Highlights --")
    print(f"Positive rate (unweighted): {summary['positive_rate_unweighted']:.4f}")
    print(f"Positive rate (weighted):   {summary['positive_rate_weighted']:.4f}")
    print(f"Exact duplicates dropped:   {len(df) - len(dedup):,}")
    print("Top columns by missingness:",
          list(summary["columns_with_missing_top10"].items())[:5])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
