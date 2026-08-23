"""
train_baseline_models.py

Trains and compares three baseline models on the same train/val split:
    - Logistic Regression  (interpretable linear baseline)
    - Random Forest        (bagged trees, comparison point for boosting)
    - XGBoost               (primary candidate for this project)

CRITICAL: proxy_label was derived FROM the lf_* labeling function votes and
label_confidence (see label_generator.py). Using those columns as features
would let the model trivially reverse-engineer our own labeling rule instead
of learning real patterns -- this is the label leakage risk flagged back in
Step 5. They are explicitly excluded below, along with identifier/date
columns that carry no generalizable signal.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "proxy_label"

# Columns that must NEVER be used as model features.
#   - DESYNPUF_ID: unique identifier, would let the model memorize individuals.
#   - BENE_BIRTH_DT / BENE_DEATH_DT: raw dates, already captured via age / is_deceased.
#   - lf_* / label_confidence: these DEFINE proxy_label. Using them as inputs
#     is direct label leakage -- the model would learn our labeling rule, not
#     real patterns.
#   - Raw skewed columns replaced by their *_log counterpart (transformers.py).
EXCLUDED_COLUMNS = [
    "DESYNPUF_ID",
    "BENE_BIRTH_DT",
    "BENE_DEATH_DT",
    "lf_high_cost",
    "lf_payer_mismatch",
    "lf_high_utilization",
    "lf_long_stay",
    "lf_negative_adjustment",
    "label_confidence",
    TARGET_COLUMN,
    "total_inpatient_payment_amt",
    "total_outpatient_payment_amt",
    "total_inpatient_primary_payer_amt",
    "total_outpatient_primary_payer_amt",
    "total_inpatient_days",
    "avg_inpatient_payment_amt",
    "avg_outpatient_payment_amt",
    # Found via feature-importance inspection: total_cost_combined is the
    # EXACT sum lf_high_cost thresholds on, and total_claims_combined is the
    # exact sum lf_high_utilization thresholds on. Including these let the
    # model reconstruct our own labeling rule almost perfectly (PR-AUC 1.00
    # was the giveaway) rather than learning generalizable patterns.
    # cost_per_claim is a ratio of these, not a direct threshold match, so
    # it's kept -- it's a legitimate derived feature, not a leakage path.
    "total_cost_combined",
    "total_claims_combined",
]

CATEGORICAL_COLUMNS = ["BENE_SEX_IDENT_CD", "SP_STATE_CODE", "BENE_RACE_CD", "BENE_COUNTY_CD"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split usable columns into (numeric_columns, categorical_columns),
    after removing everything in EXCLUDED_COLUMNS.
    """
    usable_columns = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    categorical_columns = [c for c in usable_columns if c in CATEGORICAL_COLUMNS]
    numeric_columns = [c for c in usable_columns if c not in categorical_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """
    Build a shared preprocessing pipeline used identically by all three models:
        - Numeric columns: StandardScaler (needed for Logistic Regression;
          harmless no-op effect on tree-based split logic for RF/XGBoost,
          since scaling doesn't change relative ordering).
        - Categorical columns: OneHotEncoder, handle_unknown="ignore" so the
          pipeline doesn't crash if validation/test data contains a category
          value not seen during training.

    Using ONE shared preprocessor for all three models keeps the comparison
    fair -- every model sees the exact same input representation.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    """
    Construct the three candidate model pipelines. Each uses class-imbalance
    handling appropriate to its own algorithm:
        - LogisticRegression / RandomForest: class_weight="balanced"
        - XGBoost: scale_pos_weight (ratio of negative to positive samples)
    """
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=200, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, y_true: pd.Series, y_pred, y_proba) -> dict:
    """
    Compute and log evaluation metrics for one model's predictions.

    We report metrics for the DENY class (label=1) specifically, plus
    ROC-AUC and PR-AUC -- PR-AUC matters more than ROC-AUC here given the
    severe class imbalance (97.7% Approve / 2.3% Deny).
    """
    report = classification_report(y_true, y_pred, target_names=["Approve", "Deny"], output_dict=True)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(
        "[%s] Deny -> Precision: %.3f | Recall: %.3f | F1: %.3f | ROC-AUC: %.3f | PR-AUC: %.3f",
        name,
        report["Deny"]["precision"],
        report["Deny"]["recall"],
        report["Deny"]["f1-score"],
        roc_auc,
        pr_auc,
    )
    logger.info("[%s] Confusion matrix:\n%s", name, cm)

    return {
        "model": name,
        "precision_deny": report["Deny"]["precision"],
        "recall_deny": report["Deny"]["recall"],
        "f1_deny": report["Deny"]["f1-score"],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


if __name__ == "__main__":
    # Run with: python -m src.training.train_baseline_models
    from src.config.paths import PROCESSED_DATA_DIR

    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")

    numeric_cols, categorical_cols = get_feature_columns(train_df)
    logger.info(
        "Using %d numeric features and %d categorical features.",
        len(numeric_cols),
        len(categorical_cols),
    )

    X_train, y_train = train_df[numeric_cols + categorical_cols], train_df[TARGET_COLUMN]
    X_val, y_val = val_df[numeric_cols + categorical_cols], val_df[TARGET_COLUMN]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info("XGBoost scale_pos_weight: %.2f", scale_pos_weight)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = build_models(scale_pos_weight)

    results = []
    for name, model in models.items():
        logger.info("Training %s...", name)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)[:, 1]

        results.append(evaluate_model(name, y_val, y_pred, y_proba))

    print("\n--- Model comparison summary (validation set) ---")
    print(pd.DataFrame(results).set_index("model").round(3))