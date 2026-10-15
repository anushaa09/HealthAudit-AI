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


@lru_cache
def get_dice_explainer():
    """
    Build the DiCE counterfactual explainer once, using the training set as
    background data for realistic feature ranges. Cached the same way as
    the model/SHAP explainer -- DiCE setup is not free, and every request
    would otherwise rebuild it from scratch.
    """
    import pandas as pd

    from src.config.paths import PROCESSED_DATA_DIR
    from src.explainability.dice_recourse import build_dice_explainer, build_permitted_range

    logger.info("Building DiCE explainer (first request only -- cached afterward)...")
    classifier = get_classifier()
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    dice_explainer, variable_features = build_dice_explainer(classifier, train_df)
    permitted_range = build_permitted_range(train_df)
    return dice_explainer, variable_features, permitted_range