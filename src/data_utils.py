"""
Data loading and preprocessing utilities for the 1994/95 Current Population Survey
(Census Bureau KDD) income classification / segmentation project.

The raw data file contains 40 features + a `weight` column (sample weight used
because the CPS uses stratified sampling) + a year column + a binary income
label. Missing values are encoded as the literal string '?'. Some categorical
features use the value "Not in universe" to mean "question not applicable".

This module provides:
  * load_raw(...)  -> pandas.DataFrame                                 (raw load)
  * build_preprocessor(numeric_cols, categorical_cols)                 (sklearn ColumnTransformer)
  * split_features(df)                                                  (X, y, weights)
  * column names constants (LABEL_COL, WEIGHT_COL, NUMERIC_COLS, CAT_COLS)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ------------------------------------------------------------------
# Column definitions
# ------------------------------------------------------------------

# The raw file has 42 comma-separated fields per line. The first 41 are listed
# in census-bureau.columns; the 42nd is the income label (" - 50000." or
# " 50000+.").
RAW_COLUMNS = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_a_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "weight",
    "migration_code_change_in_msa",
    "migration_code_change_in_reg",
    "migration_code_move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "label",
]

LABEL_COL = "label"
WEIGHT_COL = "weight"

# The dataset documentation is slightly misleading: some numeric-looking columns
# (detailed_industry_recode, detailed_occupation_recode,
#  own_business_or_self_employed, veterans_benefits, year) are really coded
# categoricals, not continuous quantities. We treat them as categorical.
NUMERIC_COLS = [
    "age",
    "wage_per_hour",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "num_persons_worked_for_employer",
    "weeks_worked_in_year",
]

# Everything else (except weight / label) is categorical.
CATEGORICAL_COLS = [
    c for c in RAW_COLUMNS if c not in NUMERIC_COLS + [WEIGHT_COL, LABEL_COL]
]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_raw(
    data_path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load the raw Census Bureau CSV.

    * replaces ' ?' with NaN
    * trims whitespace from string columns
    * binarizes the label (' 50000+.' -> 1, else 0)
    """
    data_path = Path(data_path)
    df = pd.read_csv(
        data_path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?", " ?"],
        skipinitialspace=True,
        nrows=nrows,
        low_memory=False,
    )
    # Trim whitespace on string columns.
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

    # Binarize label.
    df[LABEL_COL] = (df[LABEL_COL] == "50000+.").astype(np.int8)
    return df


def split_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split a loaded frame into (X, y, sample_weight)."""
    y = df[LABEL_COL].astype(int)
    w = df[WEIGHT_COL].astype(float)
    X = df.drop(columns=[LABEL_COL, WEIGHT_COL])
    return X, y, w


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def build_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Return a ColumnTransformer that:
       * median-imputes numeric features (optionally standard-scales them)
       * one-hot encodes categoricals, filling missings with 'missing'
         (min_frequency=50 collapses rare levels to limit dimensionality).
    """
    numeric_cols = numeric_cols or NUMERIC_COLS
    categorical_cols = categorical_cols or CATEGORICAL_COLS

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=50,
            sparse_output=True,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are exact duplicates across all columns including label.
    Some of the 200k rows are literal duplicates in this dataset."""
    return df.drop_duplicates().reset_index(drop=True)
