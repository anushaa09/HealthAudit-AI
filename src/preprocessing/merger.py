"""
merger.py

Combines the three raw CMS DE-SynPUF datasets into a single beneficiary-level
DataFrame (one row per DESYNPUF_ID).

Core problem being solved:
    - beneficiary_df: 1 row per beneficiary
    - inpatient_df / outpatient_df: many rows per beneficiary (claim-level)

A naive 3-way merge on DESYNPUF_ID would duplicate beneficiary rows for every
combination of inpatient x outpatient claim (a cartesian-style blow-up).

Solution: aggregate each claims file down to ONE row per beneficiary first
(counts, sums, averages, distinct diagnosis codes), then LEFT JOIN those
aggregates onto the beneficiary summary. Beneficiaries with zero claims keep
all their beneficiary info and get 0 (not NaN) in the new aggregate columns,
since "zero claims" is meaningful data, not missing data.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Diagnosis code columns present in both claims files (up to 10 per claim).
_DIAGNOSIS_CODE_COLUMNS = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]


def _count_distinct_diagnosis_codes(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Count the number of DISTINCT diagnosis codes each beneficiary has across
    all their claims (not just within a single claim).

    Each claim can have up to 10 diagnosis code columns. To count unique
    codes per beneficiary, we reshape (melt) all 10 columns into one long
    column of codes, drop missing/empty entries, then count unique codes
    per DESYNPUF_ID.

    Parameters
    ----------
    df : pd.DataFrame
        Claims DataFrame (inpatient or outpatient) containing DESYNPUF_ID
        and the ICD9_DGNS_CD_1..10 columns.
    prefix : str
        'inpatient' or 'outpatient' -- used to name the output column.

    Returns
    -------
    pd.DataFrame
        Two columns: DESYNPUF_ID, distinct_{prefix}_diagnosis_codes
    """
    diagnosis_cols_present = [c for c in _DIAGNOSIS_CODE_COLUMNS if c in df.columns]

    long_df = df[["DESYNPUF_ID"] + diagnosis_cols_present].melt(
        id_vars="DESYNPUF_ID", value_name="diagnosis_code"
    )

    # Drop missing codes and CMS's blank-string placeholder for "no code".
    long_df = long_df.dropna(subset=["diagnosis_code"])
    long_df = long_df[long_df["diagnosis_code"].str.strip() != ""]

    result = (
        long_df.groupby("DESYNPUF_ID")["diagnosis_code"]
        .nunique()
        .reset_index()
        .rename(columns={"diagnosis_code": f"distinct_{prefix}_diagnosis_codes"})
    )
    return result


def aggregate_inpatient_claims(inpatient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate inpatient claims from claim-level to beneficiary-level.

    Returns
    -------
    pd.DataFrame
        One row per DESYNPUF_ID with inpatient utilization/cost summary
        columns. Beneficiaries with no inpatient claims simply will not
        appear in this DataFrame (handled later at merge time).
    """
    logger.info("Aggregating inpatient claims to beneficiary level...")

    df = inpatient_df.copy()
    # CLM_UTLZTN_DAY_CNT (length of stay) wasn't converted in the loader
    # since it's specific to inpatient aggregation, not generic loading.
    df["CLM_UTLZTN_DAY_CNT"] = pd.to_numeric(df["CLM_UTLZTN_DAY_CNT"], errors="coerce")

    agg = df.groupby("DESYNPUF_ID").agg(
        total_inpatient_claims=("CLM_ID", "count"),
        total_inpatient_payment_amt=("CLM_PMT_AMT", "sum"),
        avg_inpatient_payment_amt=("CLM_PMT_AMT", "mean"),
        total_inpatient_primary_payer_amt=("NCH_PRMRY_PYR_CLM_PD_AMT", "sum"),
        total_inpatient_days=("CLM_UTLZTN_DAY_CNT", "sum"),
    ).reset_index()

    diagnosis_counts = _count_distinct_diagnosis_codes(df, prefix="inpatient")
    agg = agg.merge(diagnosis_counts, on="DESYNPUF_ID", how="left")

    logger.info("Inpatient aggregation produced %d beneficiary rows.", len(agg))
    return agg


def aggregate_outpatient_claims(outpatient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate outpatient claims from claim-level to beneficiary-level.

    Returns
    -------
    pd.DataFrame
        One row per DESYNPUF_ID with outpatient utilization/cost summary
        columns.
    """
    logger.info("Aggregating outpatient claims to beneficiary level...")

    df = outpatient_df.copy()

    agg = df.groupby("DESYNPUF_ID").agg(
        total_outpatient_claims=("CLM_ID", "count"),
        total_outpatient_payment_amt=("CLM_PMT_AMT", "sum"),
        avg_outpatient_payment_amt=("CLM_PMT_AMT", "mean"),
        total_outpatient_primary_payer_amt=("NCH_PRMRY_PYR_CLM_PD_AMT", "sum"),
    ).reset_index()

    diagnosis_counts = _count_distinct_diagnosis_codes(df, prefix="outpatient")
    agg = agg.merge(diagnosis_counts, on="DESYNPUF_ID", how="left")

    logger.info("Outpatient aggregation produced %d beneficiary rows.", len(agg))
    return agg


def merge_datasets(
    beneficiary_df: pd.DataFrame,
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge beneficiary summary with aggregated inpatient and outpatient claims.

    Every beneficiary is retained (LEFT JOIN from beneficiary_df), even those
    with zero claims. Aggregate columns are NaN after the join for
    beneficiaries with no claims -- these are then explicitly filled with 0,
    since "zero claims" is the correct value, not a missing one.

    Returns
    -------
    pd.DataFrame
        One row per beneficiary, combining demographic/chronic-condition
        info with inpatient and outpatient utilization summaries.
    """
    logger.info("Merging beneficiary summary with claims aggregates...")

    inpatient_agg = aggregate_inpatient_claims(inpatient_df)
    outpatient_agg = aggregate_outpatient_claims(outpatient_df)

    merged = beneficiary_df.merge(inpatient_agg, on="DESYNPUF_ID", how="left")
    merged = merged.merge(outpatient_agg, on="DESYNPUF_ID", how="left")

    # Columns that should be 0 (not NaN) for beneficiaries with no claims.
    zero_fill_columns = [
        "total_inpatient_claims",
        "total_inpatient_payment_amt",
        "total_inpatient_primary_payer_amt",
        "total_inpatient_days",
        "distinct_inpatient_diagnosis_codes",
        "total_outpatient_claims",
        "total_outpatient_payment_amt",
        "total_outpatient_primary_payer_amt",
        "distinct_outpatient_diagnosis_codes",
    ]
    for col in zero_fill_columns:
        merged[col] = merged[col].fillna(0)

    # avg_*_payment_amt is intentionally NOT zero-filled: a beneficiary with
    # zero claims has no defined "average claim amount" -- leaving it as NaN
    # is more honest than implying an average of $0, which would misleadingly
    # suggest they had claims that cost nothing.

    n_no_inpatient = (merged["total_inpatient_claims"] == 0).sum()
    n_no_outpatient = (merged["total_outpatient_claims"] == 0).sum()
    logger.info(
        "Merged dataset: %d beneficiaries total. %d with zero inpatient claims, "
        "%d with zero outpatient claims.",
        len(merged),
        n_no_inpatient,
        n_no_outpatient,
    )

    return merged


if __name__ == "__main__":
    # Run with: python -m src.preprocessing.merger
    from src.config.paths import MERGED_DATASET_FILE
    from src.preprocessing.data_loader import (
        load_beneficiary_summary,
        load_inpatient_claims,
        load_outpatient_claims,
    )

    beneficiary_df = load_beneficiary_summary()
    inpatient_df = load_inpatient_claims()
    outpatient_df = load_outpatient_claims()

    final_df = merge_datasets(beneficiary_df, inpatient_df, outpatient_df)

    print("\n--- Merged dataset shape ---")
    print(final_df.shape)

    print("\n--- Sample of new aggregate columns ---")
    print(
        final_df[
            [
                "DESYNPUF_ID",
                "total_inpatient_claims",
                "total_inpatient_payment_amt",
                "total_outpatient_claims",
                "total_outpatient_payment_amt",
            ]
        ].head(10)
    )

    final_df.to_parquet(MERGED_DATASET_FILE, index=False)
    print(f"\nSaved merged dataset to: {MERGED_DATASET_FILE}")