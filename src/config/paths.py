"""
paths.py

Centralized filesystem path configuration for the HealthAudit-AI ML pipeline.
...
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

BENEFICIARY_FILE: Path = RAW_DATA_DIR / "beneficiary_summary_2008.csv"
INPATIENT_CLAIMS_FILE: Path = RAW_DATA_DIR / "inpatient_claims.csv"
OUTPATIENT_CLAIMS_FILE: Path = RAW_DATA_DIR / "outpatient_claims.csv"

MERGED_DATASET_FILE: Path = PROCESSED_DATA_DIR / "merged_dataset.parquet"
FEATURE_DATASET_FILE: Path = PROCESSED_DATA_DIR / "feature_dataset.parquet"
LABELED_DATASET_FILE: Path = PROCESSED_DATA_DIR / "labeled_dataset.parquet"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MODEL_DIR: Path = OUTPUTS_DIR / "models"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
SHAP_PLOTS_DIR: Path = OUTPUTS_DIR / "shap_plots"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

TRAINED_MODEL_FILE: Path = MODEL_DIR / "xgboost_claim_classifier.pkl"


def ensure_directories_exist() -> None:
    """
    Create all output directories if they don't already exist.

    Raw data directories are intentionally NOT created here — if raw data
    is missing, we want the data loader to fail loudly with a clear error,
    not silently create an empty folder that hides the real problem.
    """
    for directory in (
        PROCESSED_DATA_DIR,
        MODEL_DIR,
        METRICS_DIR,
        SHAP_PLOTS_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories_exist()