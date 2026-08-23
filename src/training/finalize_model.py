"""
finalize_model.py

Wraps the tuned XGBoost pipeline (from Step 10) in a ClaimClassifier and
saves it as the final, canonical model artifact -- the one FastAPI will
load later. Records validation and test metrics directly in the saved
metadata for full traceability.
"""

import joblib
import pandas as pd

from src.models.claim_classifier import ClaimClassifier
from src.utils.logger import get_logger
from src.training.train_baseline_models import TARGET_COLUMN, evaluate_model, get_feature_columns

logger = get_logger(__name__)

if __name__ == "__main__":
    # Run with: python -m src.training.finalize_model
    from src.config.paths import MODEL_DIR, PROCESSED_DATA_DIR, TRAINED_MODEL_FILE

    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")

    numeric_cols, categorical_cols = get_feature_columns(val_df)
    feature_columns = numeric_cols + categorical_cols

    tuned_pipeline_path = MODEL_DIR / "xgboost_tuned_baseline.pkl"
    pipeline = joblib.load(tuned_pipeline_path)
    logger.info("Loaded tuned pipeline from: %s", tuned_pipeline_path)

    # Re-derive val and test metrics for the metadata record (cheap to
    # recompute, and guarantees the saved numbers match this exact pipeline).
    X_val, y_val = val_df[feature_columns], val_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_columns], test_df[TARGET_COLUMN]

    val_metrics = evaluate_model("val", y_val, pipeline.predict(X_val), pipeline.predict_proba(X_val)[:, 1])
    test_metrics = evaluate_model(
        "test", y_test, pipeline.predict(X_test), pipeline.predict_proba(X_test)[:, 1]
    )

    classifier = ClaimClassifier(
        pipeline=pipeline,
        feature_columns=feature_columns,
        metadata={
            "model_type": "XGBoost (tuned via RandomizedSearchCV)",
            "trained_on": "train split only (train+val NOT combined, to preserve test-set integrity)",
            "validation_metrics": {k: v for k, v in val_metrics.items() if k != "model"},
            "test_metrics": {k: v for k, v in test_metrics.items() if k != "model"},
            "notes": (
                "proxy_label was generated via weak supervision (see label_generator.py); "
                "high performance partially reflects the model recovering the labeling "
                "function's structure, not a guarantee of real-world claim-decision accuracy."
            ),
        },
    )
    classifier.save(TRAINED_MODEL_FILE)

    print(f"\nFinal model saved to: {TRAINED_MODEL_FILE}")
    print(f"Metadata saved to: {TRAINED_MODEL_FILE.with_name(TRAINED_MODEL_FILE.stem + '_metadata.json')}")