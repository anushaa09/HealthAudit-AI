"""
evaluate_final_model.py

Final, one-time evaluation of the tuned XGBoost model on the held-out TEST
set (touched here for the first and only time -- untouched since the Step 8
split). Produces:
    - Precision / Recall / F1 / ROC-AUC / PR-AUC on test data
    - Confusion matrix
    - Calibration check: are predicted probabilities trustworthy enough to
      drive the pipeline's Confidence Check / Human Reviewer routing?
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.utils.logger import get_logger
from src.training.train_baseline_models import TARGET_COLUMN, evaluate_model, get_feature_columns

logger = get_logger(__name__)


def plot_confusion_matrix(y_true, y_pred, save_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Approve", "Deny"], ax=ax, cmap="Blues"
    )
    ax.set_title("Final Model - Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to: %s", save_path)


def plot_roc_pr_curves(y_true, y_proba, roc_path, pr_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Final Model - ROC Curve (Test Set)")
    fig.tight_layout()
    fig.savefig(roc_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Final Model - Precision-Recall Curve (Test Set)")
    fig.tight_layout()
    fig.savefig(pr_path)
    plt.close(fig)

    logger.info("Saved ROC curve to: %s", roc_path)
    logger.info("Saved PR curve to: %s", pr_path)


def plot_calibration_curve(y_true, y_proba, save_path) -> None:
    """
    Plot predicted probability vs. actual observed frequency, in bins.

    A well-calibrated model's curve sits close to the diagonal (y = x):
    among claims the model says are "70% likely Deny", roughly 70% of them
    should actually BE Deny. This matters directly for the pipeline's
    Confidence Check step, which relies on predicted probabilities being
    meaningful, not just well-RANKED.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of Deny")
    ax.set_title("Final Model - Calibration Curve (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved calibration curve to: %s", save_path)


if __name__ == "__main__":
    # Run with: python -m src.evaluation.evaluate_final_model
    from src.config.paths import MODEL_DIR, PROCESSED_DATA_DIR, METRICS_DIR

    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")
    numeric_cols, categorical_cols = get_feature_columns(test_df)
    X_test, y_test = test_df[numeric_cols + categorical_cols], test_df[TARGET_COLUMN]

    model_path = MODEL_DIR / "xgboost_tuned_baseline.pkl"
    pipeline = joblib.load(model_path)
    logger.info("Loaded tuned model from: %s", model_path)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- FINAL TEST SET EVALUATION (first and only look at test data) ---")
    result = evaluate_model("XGBoost (final, test set)", y_test, y_pred, y_proba)
    print(pd.DataFrame([result]).set_index("model").round(3))

    plot_confusion_matrix(y_test, y_pred, METRICS_DIR / "confusion_matrix_test.png")
    plot_roc_pr_curves(
        y_test, y_proba, METRICS_DIR / "roc_curve_test.png", METRICS_DIR / "pr_curve_test.png"
    )
    plot_calibration_curve(y_test, y_proba, METRICS_DIR / "calibration_curve_test.png")

    print(f"\nAll evaluation plots saved to: {METRICS_DIR}")