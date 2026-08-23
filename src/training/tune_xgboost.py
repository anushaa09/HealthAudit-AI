"""
tune_xgboost.py

Hyperparameter tuning for XGBoost -- the winning model from Step 9's
baseline comparison. Uses RandomizedSearchCV (not exhaustive GridSearchCV)
for efficient coverage of the hyperparameter space with far fewer model
fits than a full grid.

Reuses the exact feature selection and preprocessing logic from
train_baseline_models.py, so the tuned model is directly, fairly
comparable to the Step 9 baseline results.

Tuning uses the TRAIN set only (with internal cross-validation folds).
The validation set is touched only AFTER tuning, to confirm the selected
configuration generalizes. The test set remains completely untouched,
as planned back in Step 8.
"""

import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.training.train_baseline_models import (
    TARGET_COLUMN,
    build_preprocessor,
    evaluate_model,
    get_feature_columns,
)

logger = get_logger(__name__)

# Hyperparameter search space. Prefixed with "model__" because XGBClassifier
# sits inside a Pipeline step named "model" -- scikit-learn requires this
# "stepname__paramname" syntax to route parameters to the right step.
PARAM_DISTRIBUTIONS = {
    "model__max_depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.29),  # samples in [0.01, 0.30)
    "model__n_estimators": randint(100, 600),
    "model__min_child_weight": randint(1, 10),
    "model__subsample": uniform(0.6, 0.4),         # samples in [0.6, 1.0)
    "model__colsample_bytree": uniform(0.6, 0.4),  # samples in [0.6, 1.0)
}

N_ITER = 30       # number of random hyperparameter combinations to try
CV_FOLDS = 5


def tune_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float
) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV over XGBoost hyperparameters.

    scoring="average_precision" (i.e. PR-AUC) is used, NOT accuracy or plain
    ROC-AUC -- consistent with Step 9's reasoning: under severe class
    imbalance (97.7% / 2.3%), PR-AUC is the metric that actually reflects
    how well the model identifies the rare Deny class.

    scale_pos_weight is fixed (not searched) -- it's a value derived
    directly from the training data's class ratio, not a free hyperparameter
    to tune.
    """
    numeric_cols, categorical_cols = get_feature_columns(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb)])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="average_precision",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    logger.info(
        "Starting RandomizedSearchCV: %d candidates x %d folds = %d fits.",
        N_ITER, CV_FOLDS, N_ITER * CV_FOLDS,
    )
    search.fit(X_train[numeric_cols + categorical_cols], y_train)

    logger.info("Best CV PR-AUC: %.4f", search.best_score_)
    logger.info("Best hyperparameters:")
    for param, value in search.best_params_.items():
        logger.info("  %s: %s", param, value)

    return search


if __name__ == "__main__":
    # Run with: python -m src.training.tune_xgboost
    from src.config.paths import MODEL_DIR, PROCESSED_DATA_DIR

    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")

    numeric_cols, categorical_cols = get_feature_columns(train_df)
    X_train, y_train = train_df[numeric_cols + categorical_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[numeric_cols + categorical_cols], val_df[TARGET_COLUMN]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    search = tune_xgboost(X_train, y_train, scale_pos_weight)
    best_pipeline = search.best_estimator_

    print("\n--- Evaluating tuned XGBoost on validation set ---")
    y_pred = best_pipeline.predict(X_val)
    y_proba = best_pipeline.predict_proba(X_val)[:, 1]
    result = evaluate_model("XGBoost (tuned)", y_val, y_pred, y_proba)
    print(pd.DataFrame([result]).set_index("model").round(3))

    # Save the tuned pipeline for now. Step 13 will formalize final model
    # serialization properly (naming, versioning, load/predict wrapper) --
    # this save here is just so we don't lose today's tuning work.
    tuned_model_path = MODEL_DIR / "xgboost_tuned_baseline.pkl"
    joblib.dump(best_pipeline, tuned_model_path)
    print(f"\nSaved tuned pipeline to: {tuned_model_path}")