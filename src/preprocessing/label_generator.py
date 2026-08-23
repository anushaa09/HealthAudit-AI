"""
label_generator.py

Generates proxy approve/deny labels for claims using weak supervision, since
CMS DE-SynPUF has no real ground-truth approval/denial labels.

Theory:
    We write several independent "labeling functions" (LFs), each expressing
    one heuristic about what an anomalous/high-risk claim pattern looks like.
    Each LF votes DENY (1), APPROVE (0), or ABSTAIN (no opinion) for every
    beneficiary. We combine the non-abstaining votes by majority; the
    proportion of agreement becomes a confidence score.

    This is deliberately imperfect -- it's a documented limitation, not a
    substitute for real historical claims-adjudication data. Every LF here
    is derived directly from patterns found during EDA (see notebooks/01_eda.ipynb),
    not arbitrary guesses.

Output columns added to the dataset:
    - lf_high_cost, lf_payer_mismatch, lf_high_utilization,
      lf_long_stay, lf_negative_adjustment   (each: -1 abstain, 0 approve, 1 deny)
    - proxy_label       (0 = Approve, 1 = Deny)
    - label_confidence  (0.0 - 1.0, agreement strength among voting LFs)
"""

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ABSTAIN = -1
APPROVE = 0
DENY = 1

# Percentile used to define "unusually high" for cost/utilization/stay LFs.
# 95th percentile = top 5% flagged as anomalous. Chosen because it's a
# common convention for outlier flagging and matches what we visually
# confirmed as the "long tail" in EDA histograms.
_HIGH_PERCENTILE = 0.95

# Ratio below which "primary payer paid almost nothing relative to billed
# amount" is considered suspicious. Chosen because EDA showed the 75th
# percentile of primary-payer amounts was already 0 -- so ANY claim with
# nonzero total cost but near-zero payer contribution is already unusual.
_PAYER_MISMATCH_RATIO_THRESHOLD = 0.05


def _lf_high_cost(df: pd.DataFrame) -> pd.Series:
    """DENY if combined inpatient+outpatient cost is in the top 5%."""
    total_cost = df["total_inpatient_payment_amt"] + df["total_outpatient_payment_amt"]
    threshold = total_cost.quantile(_HIGH_PERCENTILE)
    logger.info("LF[high_cost] threshold (95th pct total cost): $%.2f", threshold)

    votes = pd.Series(APPROVE, index=df.index)
    votes[total_cost > threshold] = DENY
    # Abstain for beneficiaries with zero total cost -- "cheap" isn't
    # informative either way, it's just the majority baseline.
    votes[total_cost == 0] = ABSTAIN
    return votes


def _lf_payer_mismatch(df: pd.DataFrame) -> pd.Series:
    """DENY if primary payer covered almost none of a nonzero billed amount."""
    total_cost = df["total_inpatient_payment_amt"] + df["total_outpatient_payment_amt"]
    total_payer = df["total_inpatient_primary_payer_amt"] + df["total_outpatient_primary_payer_amt"]

    votes = pd.Series(ABSTAIN, index=df.index)

    has_cost = total_cost > 0
    ratio = pd.Series(np.nan, index=df.index)
    ratio[has_cost] = total_payer[has_cost] / total_cost[has_cost]

    votes[has_cost & (ratio < _PAYER_MISMATCH_RATIO_THRESHOLD)] = DENY
    votes[has_cost & (ratio >= _PAYER_MISMATCH_RATIO_THRESHOLD)] = APPROVE
    # Beneficiaries with zero total cost: abstain, nothing to evaluate.
    return votes


def _lf_high_utilization(df: pd.DataFrame) -> pd.Series:
    """DENY if total claim count (inpatient + outpatient) is in the top 5%."""
    total_claims = df["total_inpatient_claims"] + df["total_outpatient_claims"]
    threshold = total_claims.quantile(_HIGH_PERCENTILE)
    logger.info("LF[high_utilization] threshold (95th pct claim count): %.1f", threshold)

    votes = pd.Series(APPROVE, index=df.index)
    votes[total_claims > threshold] = DENY
    votes[total_claims == 0] = ABSTAIN
    return votes


def _lf_long_stay(df: pd.DataFrame) -> pd.Series:
    """DENY if total inpatient days is in the top 5% (only applies to those with inpatient claims)."""
    has_inpatient = df["total_inpatient_claims"] > 0
    threshold = df.loc[has_inpatient, "total_inpatient_days"].quantile(_HIGH_PERCENTILE)
    logger.info("LF[long_stay] threshold (95th pct inpatient days): %.1f", threshold)

    votes = pd.Series(ABSTAIN, index=df.index)  # abstain by default (no inpatient claims)
    votes[has_inpatient & (df["total_inpatient_days"] > threshold)] = DENY
    votes[has_inpatient & (df["total_inpatient_days"] <= threshold)] = APPROVE
    return votes


def _lf_negative_adjustment(df: pd.DataFrame) -> pd.Series:
    """DENY if either inpatient or outpatient payment amount is negative (adjustment/reversal)."""
    votes = pd.Series(APPROVE, index=df.index)
    is_negative = (df["total_inpatient_payment_amt"] < 0) | (df["total_outpatient_payment_amt"] < 0)
    votes[is_negative] = DENY
    return votes


def generate_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all labeling functions, combine their votes, and attach proxy
    labels + confidence scores to the dataset.

    Returns
    -------
    pd.DataFrame
        Original df with new columns: each LF's raw vote, plus
        'proxy_label' and 'label_confidence'.
    """
    logger.info("Running labeling functions...")

    df = df.copy()
    df["lf_high_cost"] = _lf_high_cost(df)
    df["lf_payer_mismatch"] = _lf_payer_mismatch(df)
    df["lf_high_utilization"] = _lf_high_utilization(df)
    df["lf_long_stay"] = _lf_long_stay(df)
    df["lf_negative_adjustment"] = _lf_negative_adjustment(df)

    lf_columns = [
        "lf_high_cost",
        "lf_payer_mismatch",
        "lf_high_utilization",
        "lf_long_stay",
        "lf_negative_adjustment",
    ]
    lf_votes = df[lf_columns]

    # For each row: count DENY votes and APPROVE votes, ignoring ABSTAIN.
    deny_votes = (lf_votes == DENY).sum(axis=1)
    approve_votes = (lf_votes == APPROVE).sum(axis=1)
    total_voting = deny_votes + approve_votes  # excludes abstains

    # Majority vote. If NO labeling function voted at all (all abstained,
    # which can happen for beneficiaries with zero claims across the board),
    # default to Approve -- "no evidence of anything unusual" is the
    # conservative, defensible default.
    proxy_label = pd.Series(APPROVE, index=df.index)
    proxy_label[deny_votes > approve_votes] = DENY
    df["proxy_label"] = proxy_label

    # Confidence = fraction of voting (non-abstaining) LFs that agree with
    # the final majority label. Beneficiaries where every voting LF agrees
    # get confidence 1.0; a near 50/50 split gets confidence close to 0.5.
    winning_votes = np.where(proxy_label == DENY, deny_votes, approve_votes)
    with np.errstate(invalid="ignore", divide="ignore"):
        confidence = np.where(total_voting > 0, winning_votes / total_voting, 0.5)
    df["label_confidence"] = confidence

    n_deny = (df["proxy_label"] == DENY).sum()
    n_approve = (df["proxy_label"] == APPROVE).sum()
    logger.info(
        "Proxy labels generated: %d Approve (%.1f%%), %d Deny (%.1f%%)",
        n_approve,
        100 * n_approve / len(df),
        n_deny,
        100 * n_deny / len(df),
    )
    logger.info(
        "Mean label confidence: %.3f | Beneficiaries with confidence < 0.6: %d",
        df["label_confidence"].mean(),
        (df["label_confidence"] < 0.6).sum(),
    )

    return df


if __name__ == "__main__":
    # Run with: python -m src.preprocessing.label_generator
    from src.config.paths import LABELED_DATASET_FILE, MERGED_DATASET_FILE

    df = pd.read_parquet(MERGED_DATASET_FILE)
    labeled_df = generate_proxy_labels(df)

    print("\n--- Proxy label distribution ---")
    print(labeled_df["proxy_label"].value_counts())

    print("\n--- Label confidence distribution ---")
    print(labeled_df["label_confidence"].describe())

    print("\n--- LF agreement sample (first 10 rows) ---")
    print(
        labeled_df[
            [
                "DESYNPUF_ID",
                "lf_high_cost",
                "lf_payer_mismatch",
                "lf_high_utilization",
                "lf_long_stay",
                "lf_negative_adjustment",
                "proxy_label",
                "label_confidence",
            ]
        ].head(10)
    )

    labeled_df.to_parquet(LABELED_DATASET_FILE, index=False)
    print(f"\nSaved labeled dataset to: {LABELED_DATASET_FILE}")