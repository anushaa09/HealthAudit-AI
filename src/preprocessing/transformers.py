"""
transformers.py

Handles two Step 6 concerns identified during EDA:
    1. Missing values in avg_*_payment_amt columns (NaN = "no claims filed",
       not "measured as zero" -- these are semantically different).
    2. Heavy right-skew (and a handful of negative values) in financial /
       utilization columns, corrected via a signed-log transform.
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns that are right-skewed and/or contain negative values (claim
# adjustments), identified during EDA (notebooks/01_eda.ipynb).
SKEWED_COLUMNS = [
    "total_inpatient_payment_amt",
    "total_outpatient_payment_amt",
    "total_inpatient_primary_payer_amt",
    "total_outpatient_primary_payer_amt",
    "total_inpatient_days",
    "avg_inpatient_payment_amt",
    "avg_outpatient_payment_amt",
]


def signed_log_transform(series: pd.Series) -> pd.Series:
    """
    Apply signed_log(x) = sign(x) * log1p(|x|).

    Handles zero (log1p(0) = 0, no error) and negative values (sign
    preserved) safely -- a plain np.log() would raise/return NaN for both.
    """
    return np.sign(series) * np.log1p(series.abs())


def handle_missing_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN in avg_inpatient_payment_amt / avg_outpatient_payment_amt with 0,
    but ONLY after confirming the NaNs correspond exactly to beneficiaries
    with zero claims (this should already be guaranteed by merger.py's
    design -- we re-verify here defensively rather than assuming).

    The corresponding total_*_claims == 0 columns already serve as the
    "no claims filed" indicator, so we don't need a separate flag column --
    filling with 0 here is safe precisely because that indicator already
    exists elsewhere in the dataset.
    """
    df = df.copy()

    checks = [
        ("avg_inpatient_payment_amt", "total_inpatient_claims"),
        ("avg_outpatient_payment_amt", "total_outpatient_claims"),
    ]
    for avg_col, count_col in checks:
        expected_missing = (df[count_col] == 0).sum()
        actual_missing = df[avg_col].isna().sum()
        if expected_missing != actual_missing:
            logger.warning(
                "Mismatch for %s: expected %d missing (zero claims), found %d. "
                "Investigate before proceeding -- missingness may not be purely structural.",
                avg_col,
                expected_missing,
                actual_missing,
            )
        df[avg_col] = df[avg_col].fillna(0)

    logger.info("Missing average-payment values filled (0 = no claims filed, verified).")
    return df


def apply_skew_correction(df: pd.DataFrame, columns: list[str] = SKEWED_COLUMNS) -> pd.DataFrame:
    """
    Apply the signed-log transform to each skewed column, adding a new
    '{col}_log' column rather than overwriting the original.

    We keep the original raw columns too -- they remain useful for EDA,
    reporting, and human-readable SHAP explanations later ("this claim
    cost $34,000" is far more interpretable than a log-transformed value).
    The *_log columns are what will actually feed the model.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Column '%s' not found, skipping skew correction.", col)
            continue
        df[f"{col}_log"] = signed_log_transform(df[col])

    logger.info("Applied signed-log transform to %d columns.", len(columns))
    return df


if __name__ == "__main__":
    # Run with: python -m src.preprocessing.transformers
    from src.config.paths import LABELED_DATASET_FILE

    df = pd.read_parquet(LABELED_DATASET_FILE)

    df = handle_missing_averages(df)
    df = apply_skew_correction(df)

    print("\n--- Missing values remaining in avg_* columns ---")
    print(df[["avg_inpatient_payment_amt", "avg_outpatient_payment_amt"]].isna().sum())

    print("\n--- Before vs after skew correction (total_inpatient_payment_amt) ---")
    print(df[["total_inpatient_payment_amt", "total_inpatient_payment_amt_log"]].describe())

    df.to_parquet(LABELED_DATASET_FILE, index=False)
    print(f"\nUpdated dataset saved to: {LABELED_DATASET_FILE}")