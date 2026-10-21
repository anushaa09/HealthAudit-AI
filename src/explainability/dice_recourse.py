"""
dice_recourse.py

Generates counterfactual recourse suggestions using DiCE: for a claim
predicted Deny, what minimal changes to ACTIONABLE features would flip the
prediction to Approve?

Critical design constraint: only a small set of ACTIONABLE features may
change -- the length of the hospital stay and the inpatient / outpatient
cost of the claim, and only downwards. Everything else is held fixed:
  * immutable or inappropriate features (age, sex, race, existing chronic
    conditions, location, coverage type) -- suggesting someone change these
    to get a claim approved would be both nonsensical and inappropriate;
  * insurer-side amounts (beneficiary responsibility, primary-payer and
    carrier payments), which a claimant cannot change;
  * counts of claims and diagnosis codes, which are facts of the care given.

The model also sees values derived from the actionable ones (log-scaled
totals, averages, cost per claim). When DiCE tries a cheaper claim, those
derived values are recomputed from it (see make_claim_consistent), so every
suggestion is a coherent claim -- not a cheaper claim that still carries the
old, expensive totals.

Authors: Nandita Gaur, Anusha Sarkar
"""

import dice_ml
import numpy as np
import pandas as pd
from dice_ml import Dice

from src.models.claim_classifier import ClaimClassifier
from src.training.train_baseline_models import CATEGORICAL_COLUMNS, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Features that must NEVER be suggested as a "change" -- either immutable
# (age, sex, race, county) or inappropriate to recommend changing (existing
# diagnosed chronic conditions).
FIXED_FEATURES = [
    "age",
    "is_deceased",
    "BENE_ESRD_IND",
    "BENE_SEX_IDENT_CD",
    "BENE_RACE_CD",
    "SP_STATE_CODE",
    "BENE_COUNTY_CD",
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
    "chronic_condition_count",
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
    "SP_RA_OA", "SP_STRKETIA",
]


# The ONLY features DiCE is allowed to change. Each can only go down:
#   MEDREIMB_IP              -> total cost of the inpatient (hospital) part of the claim
#   MEDREIMB_OP              -> total cost of the outpatient part of the claim
#   total_inpatient_days_log -> length of the hospital stay (signed-log of days)
ACTIONABLE_FEATURES = [
    "MEDREIMB_IP",
    "MEDREIMB_OP",
    "total_inpatient_days_log",
]

# Dollar amounts in a suggestion are rounded DOWN to this step (and days to
# whole days), then re-checked against the real model.
DOLLAR_STEP = 100.0


def _signed_log(x):
    """sign(x) * log1p(|x|) -- the same transform the model was trained with."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_claim_consistent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute every feature that is derived from the claim's amounts and
    claim counts, using the same definitions as
    src/preprocessing/simple_to_features.py:

        total inpatient  = MEDREIMB_IP + BENRES_IP + PPPYMT_IP
        total outpatient = MEDREIMB_OP + BENRES_OP + PPPYMT_OP

    so that changing an actionable amount also moves its log total, its
    per-claim average and the overall cost per claim. Returns a copy.
    """
    out = df.copy()
    total_ip = out["MEDREIMB_IP"].astype(float) + out["BENRES_IP"].astype(float) + out["PPPYMT_IP"].astype(float)
    total_op = out["MEDREIMB_OP"].astype(float) + out["BENRES_OP"].astype(float) + out["PPPYMT_OP"].astype(float)
    ip_claims = out["total_inpatient_claims"].astype(float)
    op_claims = out["total_outpatient_claims"].astype(float)

    out["total_inpatient_payment_amt_log"] = _signed_log(total_ip)
    out["total_outpatient_payment_amt_log"] = _signed_log(total_op)
    out["avg_inpatient_payment_amt_log"] = _signed_log(total_ip / ip_claims.clip(lower=1))
    out["avg_outpatient_payment_amt_log"] = _signed_log(total_op / op_claims.clip(lower=1))
    out["cost_per_claim"] = (total_ip + total_op) / (ip_claims + op_claims).clip(lower=1)
    return out


class ConsistentClaimModel:
    """
    Thin sklearn-style wrapper handed to DiCE. Before every prediction it
    calls make_claim_consistent, so the derived columns always agree with the
    actionable ones DiCE is changing.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, X):
        return self.pipeline.predict_proba(make_claim_consistent(X))

    def predict(self, X):
        return self.pipeline.predict(make_claim_consistent(X))


def build_dice_explainer(classifier: ClaimClassifier, background_df: pd.DataFrame) -> tuple[Dice, list[str]]:
    """
    Build a DiCE explainer around the trained pipeline.

    Parameters
    ----------
    classifier : ClaimClassifier
        The trained model wrapper. Its pipeline (preprocessor + XGBoost)
        is used directly -- DiCE calls predict_proba() on raw, un-encoded
        feature rows, and our pipeline already handles preprocessing
        internally, so no extra wrapping is needed.
    background_df : pd.DataFrame
        A reference dataset (e.g. the training set) DiCE uses to learn
        realistic feature ranges/categories to search within.

    Returns
    -------
    (dice_explainer, variable_features)
    """
    feature_columns = classifier.feature_columns
    numeric_cols = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]

    dice_df = background_df[feature_columns + [TARGET_COLUMN]].copy()

    data = dice_ml.Data(
        dataframe=dice_df,
        continuous_features=numeric_cols,
        outcome_name=TARGET_COLUMN,
    )
    model = dice_ml.Model(model=ConsistentClaimModel(classifier.pipeline), backend="sklearn")

    # method="random" is used because our model (XGBoost) is non-differentiable
    # -- gradient-based DiCE methods require a differentiable model, which
    # tree ensembles are not. "random" works with any sklearn-compatible
    # predict_proba model, at the cost of being less sample-efficient than
    # gradient methods -- acceptable here given we only need a few
    # counterfactuals per query, not real-time bulk generation.
    dice_explainer = Dice(data, model, method="random")

    variable_features = [c for c in ACTIONABLE_FEATURES if c in feature_columns]
    logger.info(
        "DiCE explainer built. %d actionable features (%s); all other features are held fixed.",
        len(variable_features), ", ".join(variable_features),
    )

    return dice_explainer, variable_features


def generate_recourse(
    dice_explainer: Dice,
    variable_features: list[str],
    query_df: pd.DataFrame,
    total_cfs: int = 3,
    permitted_range: dict | None = None,
) -> pd.DataFrame:
    """
    Generate counterfactual recourse suggestions for ONE denied claim.

    Parameters
    ----------
    query_df : pd.DataFrame
        A single-row DataFrame (the denied claim) to find recourse for.
    total_cfs : int
        Number of diverse counterfactuals to generate.
    permitted_range : dict, optional
        Feature -> [min, max] bounds restricting where DiCE is allowed to
        search. Used here specifically to keep signed-log cost/day columns
        non-negative: their raw scale can include negative values (payment
        adjustments/reversals -- a different real phenomenon, not "a
        cheaper claim"), and without this restriction DiCE could suggest
        confusing, non-actionable "recourse" like turning a claim into a
        negative adjustment. Restricting to [0, current_max] keeps
        suggestions in the sensible "fewer/cheaper claims" direction only.

    Returns
    -------
    pd.DataFrame
        total_cfs rows, each a hypothetical version of the claim that the
        model would predict as Approve, showing only what changed.
    """
    counterfactuals = dice_explainer.generate_counterfactuals(
        query_df,
        total_CFs=total_cfs,
        desired_class=0,  # 0 = Approve; we're generating recourse FROM Deny TO Approve
        features_to_vary=variable_features,
        permitted_range=permitted_range,
    )
    return counterfactuals.cf_examples_list[0].final_cfs_df


def build_search_space(
    query_row: pd.Series,
    variable_features: list[str],
    permitted_range: dict | None = None,
) -> tuple[list[str], dict]:
    """
    Narrow the search for ONE claim: an actionable feature may only go DOWN
    (to anywhere between zero and its current value), and a feature that is
    already zero is left out because there is nothing to reduce.
    """
    active = [f for f in variable_features if float(query_row[f]) > 0]
    ranges = dict(permitted_range or {})
    for feature in active:
        ranges[feature] = [0.0, float(query_row[feature])]
    return active, ranges


def _round_down(feature: str, original: float, value: float) -> float:
    """Round a suggested value DOWN to a tidy figure (whole days / $100 steps)."""
    if abs(value - original) < 1e-9:
        return original  # untouched by this suggestion
    if feature == "total_inpatient_days_log":
        days = np.floor(np.expm1(value) + 1e-9)
        return float(_signed_log(max(days, 0.0)))
    return float(max(np.floor(value / DOLLAR_STEP) * DOLLAR_STEP, 0.0))


def finalise_counterfactuals(
    classifier: ClaimClassifier,
    query_row: pd.Series,
    cfs_df: pd.DataFrame,
    variable_features: list[str],
) -> list[dict]:
    """
    Turn DiCE's raw output into the suggestions we actually show.

    For each counterfactual: round it to tidy figures, rebuild the full claim
    (derived values included), and keep it ONLY if the real trained model still
    predicts Approve for that exact claim. Exact dollar figures are used as a
    fallback when rounding the dollars would undo the approval (the stay is
    always a whole number of days). Duplicates are dropped.

    Returns a list of {feature: {"from": x, "to": y}} dicts, one per suggestion,
    containing only the actionable features that actually changed.
    """
    suggestions: list[dict] = []
    seen: set[tuple] = set()

    for i in range(len(cfs_df)):
        raw = {f: float(cfs_df.iloc[i][f]) for f in variable_features}
        tidy = {f: _round_down(f, float(query_row[f]), v) for f, v in raw.items()}
        # Fallback if rounding the dollars undoes the approval: keep the exact
        # dollar figures, but the stay is always a whole number of days.
        exact_dollars = {
            f: (tidy[f] if f == "total_inpatient_days_log" else v) for f, v in raw.items()
        }

        for candidate in (tidy, exact_dollars):
            claim = query_row.to_dict()
            claim.update(candidate)
            # A stay with no inpatient cost (or inpatient cost with no stay)
            # is not a coherent claim, so such a suggestion is never shown.
            stay, cost = claim["total_inpatient_days_log"], claim["MEDREIMB_IP"]
            if float(query_row["total_inpatient_days_log"]) > 0 and float(query_row["MEDREIMB_IP"]) > 0 \
                    and (stay <= 0) != (cost <= 0):
                continue
            full = make_claim_consistent(pd.DataFrame([claim]))
            if int(classifier.predict(full).iloc[0]) != 0:
                continue
            changes = {
                f: {"from": float(query_row[f]), "to": candidate[f]}
                for f in variable_features
                if abs(candidate[f] - float(query_row[f])) > 1e-9
            }
            key = tuple(sorted((f, round(c["to"], 6)) for f, c in changes.items()))
            if changes and key not in seen:
                seen.add(key)
                suggestions.append(changes)
            break

    return suggestions


def build_permitted_range(background_df: pd.DataFrame) -> dict:
    """
    Build sensible search bounds for the signed-log cost/utilization
    columns: floor at 0 (i.e. "no claims"), cap at the observed max.
    Prevents DiCE from suggesting negative (payment-adjustment) values as
    a way to "lower cost," which would be a confusing, non-actionable
    suggestion for a claimant to receive.
    """
    log_cost_columns = [
        "total_inpatient_payment_amt_log",
        "total_outpatient_payment_amt_log",
        "total_inpatient_primary_payer_amt_log",
        "total_outpatient_primary_payer_amt_log",
        "total_inpatient_days_log",
        "avg_inpatient_payment_amt_log",
        "avg_outpatient_payment_amt_log",
    ]
    return {col: [0.0, float(background_df[col].max())] for col in log_cost_columns if col in background_df.columns}


def summarize_changes(original_row: pd.Series, counterfactual_row: pd.Series, feature_columns: list[str]) -> dict:
    """
    Compare one counterfactual against the original claim, returning only
    the features that actually changed -- this is what should be shown to
    a claimant/reviewer, not the full 44-column row.
    """
    changes = {}
    for col in feature_columns:
        original_val = original_row[col]
        cf_val = counterfactual_row[col]
        if original_val != cf_val:
            changes[col] = {"from": original_val, "to": cf_val}
    return changes


if __name__ == "__main__":
    # Run with: python -m src.explainability.dice_recourse
    from src.config.paths import PROCESSED_DATA_DIR, TRAINED_MODEL_FILE

    classifier = ClaimClassifier.load(TRAINED_MODEL_FILE)
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")

    dice_explainer, variable_features = build_dice_explainer(classifier, train_df)

    # Find one claim the model predicted Deny, to generate recourse for.
    predictions = classifier.predict(test_df)
    denied_claims = test_df[predictions == 1]
    query_row = denied_claims.iloc[[0]]

    print("\n--- Original (Denied) Claim: key features ---")
    print(query_row[["total_outpatient_claims", "total_inpatient_payment_amt_log", "cost_per_claim"]])

    cfs_df = generate_recourse(
        dice_explainer,
        variable_features,
        query_row[classifier.feature_columns],
        total_cfs=3,
        permitted_range=build_permitted_range(train_df),
    )

    print("\n--- Counterfactual suggestions (what would flip this to Approve) ---")
    for i in range(len(cfs_df)):
        changes = summarize_changes(query_row.iloc[0], cfs_df.iloc[i], classifier.feature_columns)
        print(f"\nCounterfactual {i + 1}:")
        for feature, change in changes.items():
            print(f"  {feature}: {change['from']} -> {change['to']}")