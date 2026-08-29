"""
predict.py

The core /predict endpoint: takes a beneficiary's engineered feature
profile, returns the model's prediction, its SHAP-based explanation, and
a routing decision (Final Decision vs. Human Reviewer Queue) based on
prediction confidence.

Confidence formula: confidence = abs(probability - 0.5) * 2
    - probability = 0.5 (maximally uncertain) -> confidence = 0.0
    - probability = 0.0 or 1.0 (fully certain either way) -> confidence = 1.0
Claims with confidence below CONFIDENCE_THRESHOLD are routed to human
review rather than an automatic final decision -- this is the pipeline's
"Confidence Check" step made concrete.
"""

import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_classifier, get_explainer
from api.schemas import ClaimFeatures, PredictionResponse, TopFactor
from src.explainability.shap_explainer import explain_single_claim
from src.utils.logger import get_logger
from src.blockchain.hashing import hash_decision_record, build_hashable_record

logger = get_logger(__name__)

router = APIRouter()

# Claims with confidence below this threshold go to human review instead of
# an automatic final decision. 0.5 means predictions with probability
# roughly between 0.25 and 0.75 are considered too uncertain to auto-decide.
CONFIDENCE_THRESHOLD = 0.5


@router.post("/predict", response_model=PredictionResponse)
def predict_claim(
    claim: ClaimFeatures,
    classifier=Depends(get_classifier),
    explainer=Depends(get_explainer),
) -> PredictionResponse:
    """
    Predict Approve/Deny for one beneficiary's claim profile, with a SHAP
    explanation and a routing decision.
    """
    claim_df = pd.DataFrame([claim.model_dump()])

    explanation = explain_single_claim(classifier, explainer, claim_df, top_n=5)

    probability = explanation["predicted_probability"]
    label = explanation["predicted_label"]
    confidence = round(abs(probability - 0.5) * 2, 4)
    routing = "Human Reviewer Queue" if confidence < CONFIDENCE_THRESHOLD else "Final Decision"

    logger.info(
        "Prediction: label=%d prob=%.4f confidence=%.4f routing=%s",
        label, probability, confidence, routing,
    )

    return PredictionResponse(
        predicted_label=label,
        predicted_label_name="Deny" if label == 1 else "Approve",
        predicted_probability=probability,
        confidence=confidence,
        routing=routing,
        top_factors=[TopFactor(**f) for f in explanation["top_factors"]],
    )