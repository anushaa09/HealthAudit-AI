"""
shap_explainer.py

Explainability layer for HealthAudit-AI, using SHAP's TreeExplainer --
chosen specifically because it computes EXACT Shapley values in polynomial
time for tree-based models like XGBoost (unlike generic SHAP, which is
exponential and infeasible at scale).

Two levels of explanation:
    - GLOBAL: which features matter most across all claims (summary plot).
    - LOCAL: why one specific claim got its prediction (waterfall plot +
      a reusable function returning top contributing features -- this is
      what FastAPI will call later to return an explanation alongside a
      prediction).

IMPORTANT interpretation note: SHAP values here are in LOG-ODDS (margin)
space, not probability -- this is the exact, standard choice for
TreeExplainer. A positive SHAP value pushes toward Deny, negative pushes
toward Approve. sum(SHAP values) + base_value = raw model margin output,
which passes through a sigmoid to produce the final predicted probability.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.claim_classifier import ClaimClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_transformed_features(classifier: ClaimClassifier, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Run the model's fitted preprocessor on raw feature columns, returning
    a dense numeric array plus the expanded feature names (e.g.
    'SP_STATE_CODE_34' after one-hot encoding) -- TreeExplainer operates on
    this transformed space, not the original raw columns.
    """
    preprocessor = classifier.pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(df[classifier.feature_columns])

    # OneHotEncoder output can be sparse; TreeExplainer needs a dense array.
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = list(preprocessor.get_feature_names_out())
    return X_transformed, feature_names


def build_explainer(classifier: ClaimClassifier) -> shap.TreeExplainer:
    """Build a TreeExplainer around the fitted XGBoost model inside the pipeline."""
    xgb_model = classifier.pipeline.named_steps["model"]
    return shap.TreeExplainer(xgb_model)


def plot_global_summary(
    explainer: shap.TreeExplainer, X_transformed: np.ndarray, feature_names: list[str], save_path
) -> None:
    """
    Global explanation: a beeswarm summary plot showing, across all claims
    in the sample, which features have the largest overall impact on the
    model's output, and in which direction.
    """
    shap_values = explainer.shap_values(X_transformed)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False, max_display=15)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved global SHAP summary plot to: %s", save_path)


def plot_local_waterfall(
    explainer: shap.TreeExplainer,
    X_transformed: np.ndarray,
    feature_names: list[str],
    row_index: int,
    save_path,
) -> None:
    """
    Local explanation for ONE claim: a waterfall plot showing exactly which
    features pushed this specific prediction up (toward Deny) or down
    (toward Approve), starting from the base value.
    """
    shap_values = explainer(X_transformed[row_index : row_index + 1])
    shap_values.feature_names = feature_names

    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[0], show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info("Saved local SHAP waterfall plot (row %d) to: %s", row_index, save_path)


def explain_single_claim(
    classifier: ClaimClassifier, explainer: shap.TreeExplainer, row_df: pd.DataFrame, top_n: int = 5
) -> dict:
    """
    Return a structured explanation for ONE claim -- this is the function
    FastAPI will call later to attach an explanation to a prediction.

    Returns
    -------
    dict with:
        - predicted_label: 0 (Approve) or 1 (Deny)
        - predicted_probability: model's Deny probability
        - base_value: the model's average prediction (log-odds) before
          any feature's contribution is applied
        - top_factors: list of {feature, value, shap_contribution},
          sorted by absolute impact, largest first
    """
    X_transformed, feature_names = get_transformed_features(classifier, row_df)
    shap_values = explainer(X_transformed)

    contributions = shap_values.values[0]
    feature_values = X_transformed[0]
    base_value = shap_values.base_values[0]

    order = np.argsort(-np.abs(contributions))[:top_n]
    top_factors = [
        {
            "feature": feature_names[i],
            "value": round(float(feature_values[i]), 3),
            "shap_contribution": round(float(contributions[i]), 4),
        }
        for i in order
    ]

    return {
        "predicted_label": int(classifier.predict(row_df).iloc[0]),
        "predicted_probability": round(float(classifier.predict_proba(row_df).iloc[0]), 4),
        "base_value_log_odds": round(float(base_value), 4),
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    # Run with: python -m src.explainability.shap_explainer
    from src.config.paths import PROCESSED_DATA_DIR, SHAP_PLOTS_DIR, TRAINED_MODEL_FILE

    classifier = ClaimClassifier.load(TRAINED_MODEL_FILE)
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")

    # Use a sample for the global plot -- fast, and a few hundred claims is
    # plenty to see stable overall feature importance patterns.
    sample_df = test_df.sample(n=min(500, len(test_df)), random_state=42).reset_index(drop=True)
    X_sample, feature_names = get_transformed_features(classifier, sample_df)

    explainer = build_explainer(classifier)

    plot_global_summary(explainer, X_sample, feature_names, SHAP_PLOTS_DIR / "global_summary.png")

    # Local explanations: one confirmed Deny prediction, one confirmed Approve.
    predictions = classifier.predict(sample_df)
    deny_idx = predictions[predictions == 1].index
    approve_idx = predictions[predictions == 0].index

    if len(deny_idx) > 0:
        i = deny_idx[0]
        plot_local_waterfall(explainer, X_sample, feature_names, i, SHAP_PLOTS_DIR / "local_deny_example.png")
        print("\n--- Explanation for a DENY prediction ---")
        print(explain_single_claim(classifier, explainer, sample_df.iloc[[i]]))

    if len(approve_idx) > 0:
        i = approve_idx[0]
        plot_local_waterfall(
            explainer, X_sample, feature_names, i, SHAP_PLOTS_DIR / "local_approve_example.png"
        )
        print("\n--- Explanation for an APPROVE prediction ---")
        print(explain_single_claim(classifier, explainer, sample_df.iloc[[i]]))

    print(f"\nAll SHAP plots saved to: {SHAP_PLOTS_DIR}")