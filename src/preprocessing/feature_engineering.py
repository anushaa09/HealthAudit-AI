"""
feature_engineering.py

Derives model-ready features from the labeled dataset:
    - age (from BENE_BIRTH_DT, relative to the 2008 reference date)
    - is_deceased flag (from BENE_DEATH_DT)
    - total chronic condition count (sum of 11 SP_* flags)
    - combined inpatient+outpatient claim count and cost
    - cost per claim ratio
    - categorical dtype casting for sex/state codes (for native XGBoost support)

DESYNPUF_ID is deliberately never engineered into a feature -- it's a unique
identifier kept only for traceability/joins, excluded from training inputs.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Beneficiary Summary file used is specifically the 2008 sample, so age must
# be computed relative to the end of that data year -- NOT today's real-world
# date, which would produce nonsensical ages for 2008 Medicare data.
_AGE_REFERENCE_DATE = pd.Timestamp("2008-12-31")

_CHRONIC_CONDITION_COLUMNS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]

_CATEGORICAL_COLUMNS = ["BENE_SEX_IDENT_CD", "SP_STATE_CODE", "BENE_RACE_CD", "BENE_COUNTY_CD"]


def add_age(df: pd.DataFrame, reference_date: pd.Timestamp = _AGE_REFERENCE_DATE) -> pd.DataFrame:
    """
    Compute age in whole years as of `reference_date`, from BENE_BIRTH_DT.

    Age is far more directly useful/interpretable to both the model and
    SHAP explanations than a raw birth date.
    """
    df = df.copy()
    age_in_days = (reference_date - df["BENE_BIRTH_DT"]).dt.days
    df["age"] = (age_in_days / 365.25).astype(int)
    logger.info("Computed age. Range: %d - %d years.", df["age"].min(), df["age"].max())
    return df


def add_deceased_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add a clean 0/1 is_deceased flag derived from BENE_DEATH_DT presence."""
    df = df.copy()
    df["is_deceased"] = df["BENE_DEATH_DT"].notna().astype(int)
    logger.info("Deceased beneficiaries: %d / %d", df["is_deceased"].sum(), len(df))
    return df


def add_chronic_condition_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the 11 chronic condition flags into a single comorbidity-burden count.
    Original individual flags are kept too -- this is an additional summary
    feature, not a replacement.
    """
    df = df.copy()
    df["chronic_condition_count"] = df[_CHRONIC_CONDITION_COLUMNS].sum(axis=1)
    logger.info(
        "Chronic condition count computed. Mean: %.2f, Max: %d",
        df["chronic_condition_count"].mean(),
        df["chronic_condition_count"].max(),
    )
    return df


def add_combined_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined (inpatient + outpatient) claim count and cost features."""
    df = df.copy()
    df["total_claims_combined"] = df["total_inpatient_claims"] + df["total_outpatient_claims"]
    df["total_cost_combined"] = df["total_inpatient_payment_amt"] + df["total_outpatient_payment_amt"]
    return df


def add_cost_per_claim(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add average cost per claim (combined), guarded against division by zero.

    Beneficiaries with zero total claims get 0 here (not NaN) -- "no claims,
    no cost per claim" is a defensible, model-safe default, distinct from
    the avg_*_payment_amt NaN handling in transformers.py because this ratio
    is derived fresh here, not carried over from a groupby mean.
    """
    df = df.copy()
    df["cost_per_claim"] = 0.0
    has_claims = df["total_claims_combined"] > 0
    df.loc[has_claims, "cost_per_claim"] = (
        df.loc[has_claims, "total_cost_combined"] / df.loc[has_claims, "total_claims_combined"]
    )
    return df


def encode_categoricals(df: pd.DataFrame, columns: list[str] = _CATEGORICAL_COLUMNS) -> pd.DataFrame:
    """
    Cast categorical code columns to pandas 'category' dtype.

    XGBoost (>=1.5, enable_categorical=True) natively handles 'category'
    dtype columns with proper split logic -- casting here means we do NOT
    need manual one-hot encoding, which would otherwise blow up column count
    for high-cardinality columns like SP_STATE_CODE.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            logger.warning("Categorical column '%s' not found, skipping.", col)
            continue
        df[col] = df[col].astype("category")
    logger.info("Cast %d columns to category dtype.", len(columns))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering sequence in order."""
    logger.info("Starting feature engineering...")
    df = add_age(df)
    df = add_deceased_flag(df)
    df = add_chronic_condition_count(df)
    df = add_combined_utilization_features(df)
    df = add_cost_per_claim(df)
    df = encode_categoricals(df)
    logger.info("Feature engineering complete. Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    # Run with: python -m src.preprocessing.feature_engineering
    from src.config.paths import FEATURE_DATASET_FILE, LABELED_DATASET_FILE

    df = pd.read_parquet(LABELED_DATASET_FILE)
    feature_df = engineer_features(df)

    print("\n--- New feature sample ---")
    print(
        feature_df[
            [
                "DESYNPUF_ID", "age", "is_deceased", "chronic_condition_count",
                "total_claims_combined", "total_cost_combined", "cost_per_claim",
            ]
        ].head(10)
    )

    print("\n--- Age distribution ---")
    print(feature_df["age"].describe())

    feature_df.to_parquet(FEATURE_DATASET_FILE, index=False)
    print(f"\nSaved feature dataset to: {FEATURE_DATASET_FILE}")