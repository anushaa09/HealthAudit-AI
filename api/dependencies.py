"""
dependencies.py

FastAPI dependency providers. Both the trained model and the SHAP explainer
are expensive to construct (disk load + tree traversal setup), so they are
loaded ONCE via functools.lru_cache -- the first request triggers loading,
every subsequent request reuses the same cached instance instead of
reloading from disk each time.
"""

from functools import lru_cache

from src.explainability.shap_explainer import build_explainer
from src.models.claim_classifier import ClaimClassifier
from src.config.paths import TRAINED_MODEL_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_classifier() -> ClaimClassifier:
    """Load the trained ClaimClassifier once and reuse it across all requests."""
    logger.info("Loading ClaimClassifier (first request only -- cached afterward)...")
    return ClaimClassifier.load(TRAINED_MODEL_FILE)


@lru_cache
def get_explainer():
    """Build the SHAP TreeExplainer once, from the same cached classifier."""
    logger.info("Building SHAP TreeExplainer (first request only -- cached afterward)...")
    classifier = get_classifier()
    return build_explainer(classifier)


def _load_dice_background(classifier: ClaimClassifier):
    """
    Return the background DataFrame DiCE uses to learn realistic feature
    ranges, or raise a clear HTTP 503 if none is usable.

    Preference order:
      1. data/processed/dice_background.parquet -- small, committed to Git,
         and therefore present on Render.
      2. data/processed/train.parquet -- the full training split. It is
         git-ignored, so it only exists on a machine that built it locally.

    A file that exists but cannot be read (e.g. a corrupted download) is
    skipped with a warning, not allowed to crash the request with a 500.
    """
    import pandas as pd
    from fastapi import HTTPException

    from src.config.paths import DICE_BACKGROUND_FILE, TRAIN_DATASET_FILE
    from src.training.train_baseline_models import TARGET_COLUMN

    required_columns = list(classifier.feature_columns) + [TARGET_COLUMN]

    for path in (DICE_BACKGROUND_FILE, TRAIN_DATASET_FILE):
        if not path.exists():
            logger.warning("DiCE background file not found: %s", path)
            continue
        try:
            background_df = pd.read_parquet(path)
        except Exception as exc:  # corrupted / truncated / unreadable parquet
            logger.warning("DiCE background file %s could not be read: %s", path, exc)
            continue

        missing = [c for c in required_columns if c not in background_df.columns]
        if missing:
            logger.warning("DiCE background file %s is missing columns: %s", path, missing)
            continue

        logger.info("Using DiCE background data from %s (%d rows).", path, len(background_df))
        try:
            import pyarrow.parquet as pq

            origin = (pq.read_schema(path).metadata or {}).get(b"healthaudit_background_source")
        except Exception:
            origin = None
        if origin:
            logger.warning(
                "DiCE background is a %s. Counterfactuals are still verified against the real "
                "model, but the search ranges are approximate.",
                origin.decode(),
            )
        return background_df

    logger.error("No usable DiCE background dataset found; /recourse is unavailable.")
    raise HTTPException(
        status_code=503,
        detail=(
            "Counterfactual suggestions are temporarily unavailable: "
            "the background dataset for this feature is missing or unreadable."
        ),
    )


@lru_cache
def get_dice_explainer():
    """
    Build the DiCE counterfactual explainer once, using a background dataset
    for realistic feature ranges. Cached the same way as the model/SHAP
    explainer -- DiCE setup is not free, and every request would otherwise
    rebuild it from scratch. (Failures are not cached: if the background
    data is fixed, the next request picks it up without a restart.)
    """
    from src.explainability.dice_recourse import build_dice_explainer, build_permitted_range

    logger.info("Building DiCE explainer (first request only -- cached afterward)...")
    classifier = get_classifier()
    background_df = _load_dice_background(classifier)
    dice_explainer, variable_features = build_dice_explainer(classifier, background_df)
    permitted_range = build_permitted_range(background_df)
    return dice_explainer, variable_features, permitted_range