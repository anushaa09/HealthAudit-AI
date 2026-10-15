"""
claim_classifier.py

A clean wrapper around the trained model pipeline, so every other part of
the project (FastAPI later, notebooks, evaluation scripts) interacts with
the model through one consistent interface instead of scattering
joblib.load() + manual predict_proba() calls everywhere.

Saves the model together with metadata (feature columns, training metrics,
timestamp) so anyone loading the model later can verify what they're
actually getting.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClaimClassifier:
    """
    Wraps a fitted sklearn Pipeline (preprocessor + XGBoost) with a clean
    save/load/predict interface and companion metadata.
    """

    def __init__(self, pipeline, feature_columns: list[str], metadata: dict | None = None):
        self.pipeline = pipeline
        self.feature_columns = feature_columns
        self.metadata = metadata or {}

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return predicted class labels (0 = Approve, 1 = Deny)."""
        return pd.Series(self.pipeline.predict(df[self.feature_columns]), index=df.index)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Return predicted probability of Deny (class 1)."""
        proba = self.pipeline.predict_proba(df[self.feature_columns])[:, 1]
        return pd.Series(proba, index=df.index)

    def save(self, model_path: Path) -> None:
        """
        Save the pipeline (.pkl) and a companion metadata (.json) file with
        the same base name -- e.g. xgboost_claim_classifier.pkl +
        xgboost_claim_classifier_metadata.json.
        """
        model_path = Path(model_path)
        joblib.dump(self.pipeline, model_path)

        metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
        full_metadata = {
            "feature_columns": self.feature_columns,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            **self.metadata,
        }
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        logger.info("Saved model to: %s", model_path)
        logger.info("Saved metadata to: %s", metadata_path)

    @classmethod
    def load(cls, model_path: Path) -> "ClaimClassifier":
        """Load a previously saved pipeline + its companion metadata file."""
        model_path = Path(model_path)
        pipeline = joblib.load(model_path)

        metadata_path = model_path.with_name(model_path.stem + "_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        feature_columns = metadata.pop("feature_columns")
        logger.info("Loaded model from: %s (trained %s)", model_path, metadata.get("saved_at_utc"))

        return cls(pipeline=pipeline, feature_columns=feature_columns, metadata=metadata)