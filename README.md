# Census-Bureau Income Classification & Marketing Segmentation

Take-home project on the 1994/95 Current Population Survey (CPS) "census-bureau"
dataset. Two deliverables:

1. **Classifier** — predicts whether an individual earns more than \$50k/yr,
   for income-tier targeting.
2. **Segmentation** — a marketing-actionable clustering of the population into
   life-stage / economic groups.

The full write-up is in [`report.md`](report.md).

## Results at a glance

| Model | ROC-AUC | PR-AUC | F1 (tuned thr.) | Test size |
| --- | --- | --- | --- | --- |
| Logistic Regression (baseline) | 0.823 | 0.167 | 0.287 | 29,445 |
| **LightGBM (final)** | **0.935** | **0.538** | **0.541** | 29,445 |

Segmentation: **k = 7** selected by silhouette; segments range from high-income
professionals (31% >\$50k) to retirees and children — see
[`outputs/segment_profiles.csv`](outputs/segment_profiles.csv).

## Repo layout

```
census-income/
├── data/
│   ├── census-bureau.data        # raw CSV (comma-delimited, no header)
│   └── census-bureau.columns     # column names file
├── src/
│   ├── data_utils.py             # load_raw, build_preprocessor, split helpers
│   ├── eda.py                    # EDA -> outputs/eda_summary.json + figures/
│   ├── train_classifier.py       # classifier end-to-end
│   └── segment.py                # segmentation end-to-end
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_classifier.ipynb
│   └── 03_segmentation.ipynb
├── outputs/                      # generated: metrics, profiles, joblib models
├── figures/                      # generated: PNG plots used in the report
├── build_notebooks.py            # regenerates the three notebooks
├── report.md                     # project report for the client
├── README.md                     # this file
└── requirements.txt
```

## Running it

Requires **Python 3.10+**. Install dependencies:

```bash
pip install -r requirements.txt
```

Reproduce every artefact:

```bash
# from the project root
python src/eda.py                  # EDA summary + figures 01-05
python src/train_classifier.py     # trains LR + LightGBM, writes metrics + figures 06-08
python src/segment.py              # fits segmentation, writes profiles + figures 09-11
python build_notebooks.py          # regenerates the three notebooks
```

The scripts import each other only through `src/data_utils.py` and write every
artefact to `outputs/` and `figures/`. No notebook execution is required to
reproduce the numbers in the report; the notebooks are for reading / telling
the story, not for computation.

To open the notebooks:

```bash
jupyter lab notebooks/
```

## Key design choices

* **Sample weights**: the CPS is a stratified sample, so `weight` is a sampling
  weight — it is dropped from `X` and passed as `sample_weight` to training
  and to every metric. This matters because the weighted positive rate (6.4%)
  is noticeably different from the unweighted one (6.2%).
* **Exact-duplicate removal**: ~3.2k identical rows; dropped before splitting
  to prevent train/test leakage.
* **Stratified 70/15/15 split** seeded at 42.
* **Threshold tuning**: the 0.5 default threshold is wrong for a ~6% positive
  problem; we pick the threshold that maximises weighted F1 on the validation
  set (~0.29 for LightGBM).
* **"Not in universe" sentinels are kept as a level** rather than treated as
  missing — they carry real signal (e.g. "not in the workforce").

## Outputs

Running the three scripts produces:

* `outputs/classifier.joblib` — full sklearn pipeline (preprocessor + LightGBM)
* `outputs/classifier_metrics.json` — val / test metrics, threshold
* `outputs/classifier_report.txt` — human-readable summary
* `outputs/feature_importance.csv` — top-40 LightGBM gain
* `outputs/segmenter.joblib` — full pipeline (preprocessor + PCA + k-means)
* `outputs/segment_profiles.csv` — per-segment size, income propensity, and
  modal demographics
* `outputs/segmentation_summary.json` — k-search results, PCA variance, profiles
* `figures/*.png` — all 11 plots used in the report

## Using the models at serve time

```python
import joblib, pandas as pd
from src.data_utils import load_raw, split_features, drop_exact_duplicates

clf = joblib.load("outputs/classifier.joblib")
seg = joblib.load("outputs/segmenter.joblib")

# new_df has the same columns as the training file (minus label/weight is OK)
income_prob = clf.predict_proba(new_df)[:, 1]
segment_id  = seg.predict(new_df[seg.named_steps["preprocessor"]
                                   .feature_names_in_])
```
