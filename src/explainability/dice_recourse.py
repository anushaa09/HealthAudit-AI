"""
dice_recourse.py

Generates counterfactual recourse suggestions using DiCE: for a claim
predicted Deny, what minimal changes to ACTIONABLE features would flip the
prediction to Approve?

Critical design constraint: immutable/inappropriate features (age, sex,
race, existing chronic conditions, location, coverage type) are explicitly
excluded from the search. Only claim/utilization/cost features -- things
that genuinely represent the claim itself, not the person -- are allowed
to vary. Suggesting someone change their age or medical conditions to get
a claim approved would be both nonsensical and inappropriate.
"""

import dice_ml
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
    model = dice_ml.Model(model=classifier.pipeline, backend="sklearn")

    # method="random" is used because our model (XGBoost) is non-differentiable
    # -- gradient-based DiCE methods require a differentiable model, which
    # tree ensembles are not. "random" works with any sklearn-compatible
    # predict_proba model, at the cost of being less sample-efficient than
    # gradient methods -- acceptable here given we only need a few
    # counterfactuals per query, not real-time bulk generation.
    dice_explainer = Dice(data, model, method="random")

    variable_features = [c for c in feature_columns if c not in FIXED_FEATURES]
    logger.info(
        "DiCE explainer built. %d variable features, %d fixed features.",
        len(variable_features), len(FIXED_FEATURES),
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