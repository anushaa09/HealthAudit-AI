"""
recourse.py

The /recourse endpoint: given a claim, if the model predicts Deny, returns
DiCE counterfactual suggestions -- minimal changes to actionable features
that would flip the prediction to Approve. If the claim is already
predicted Approve, recourse is not applicable, and we say so rather than
running DiCE's search pointlessly.
"""

import pandas as pd
from fastapi import APIRouter, Depends

from api.dependencies import get_classifier, get_dice_explainer
from api.schemas import ClaimFeatures, Counterfactual, RecourseChange, RecourseResponse
from src.explainability.dice_recourse import generate_recourse, summarize_changes
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

TOTAL_COUNTERFACTUALS = 3


@router.post("/recourse", response_model=RecourseResponse)
def get_recourse(
    claim: ClaimFeatures,
    classifier=Depends(get_classifier),
    dice_bundle=Depends(get_dice_explainer),
) -> RecourseResponse:
    """
    Return counterfactual recourse suggestions for a claim, if it was
    predicted Deny. Applicable=False (no counterfactuals) if the claim was
    already predicted Approve.
    """
    dice_explainer, variable_features, permitted_range = dice_bundle

    claim_df = pd.DataFrame([claim.model_dump()])
    predicted_label = int(classifier.predict(claim_df).iloc[0])

    if predicted_label == 0:
        logger.info("Recourse requested for a claim already predicted Approve -- not applicable.")
        return RecourseResponse(
            applicable=False,
            message="This claim was already predicted Approve -- no recourse needed.",
            counterfactuals=[],
        )

    logger.info("Generating DiCE recourse for a Deny-predicted claim...")
    cfs_df = generate_recourse(
        dice_explainer,
        variable_features,
        claim_df[classifier.feature_columns],
        total_cfs=TOTAL_COUNTERFACTUALS,
        permitted_range=permitted_range,
    )

    original_row = claim_df.iloc[0]
    counterfactuals = []
    for i in range(len(cfs_df)):
        changes = summarize_changes(original_row, cfs_df.iloc[i], classifier.feature_columns)
        counterfactuals.append(
            Counterfactual(
                changes=[
                    RecourseChange(feature=feature, from_value=float(c["from"]), to_value=float(c["to"]))
                    for feature, c in changes.items()
                ]
            )
        )

    return RecourseResponse(
        applicable=True,
        message=f"This claim was predicted Deny. Below are {len(counterfactuals)} possible ways it could have been approved.",
        counterfactuals=counterfactuals,
    )