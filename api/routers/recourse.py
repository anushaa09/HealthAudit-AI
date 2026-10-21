"""
recourse.py

The /recourse endpoint: given a claim, if the model predicts Deny, returns
DiCE counterfactual suggestions -- minimal changes to actionable features
that would flip the prediction to Approve. If the claim is already
predicted Approve, recourse is not applicable, and we say so rather than
running DiCE's search pointlessly.
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_classifier, get_dice_explainer
from api.schemas import ClaimFeatures, Counterfactual, RecourseChange, RecourseResponse
from src.explainability.dice_recourse import (
    build_search_space,
    finalise_counterfactuals,
    generate_recourse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Ask DiCE for a few extra: some are dropped when re-checked against the real
# model or found to be duplicates. At most MAX_SUGGESTIONS are returned.
TOTAL_COUNTERFACTUALS = 5
MAX_SUGGESTIONS = 3


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
    original_row = claim_df[classifier.feature_columns].iloc[0]
    # Only the stay length and the inpatient / outpatient cost may change, and
    # only downwards (see src/explainability/dice_recourse.py).
    search_features, search_ranges = build_search_space(original_row, variable_features, permitted_range)

    cfs_df = None
    if search_features:
        try:
            cfs_df = generate_recourse(
                dice_explainer,
                search_features,
                claim_df[classifier.feature_columns],
                total_cfs=TOTAL_COUNTERFACTUALS,
                permitted_range=search_ranges,
            )
        except Exception as exc:
            # DiCE raises (rather than returning nothing) when its search finds
            # no counterfactual within the allowed features and ranges. That is
            # a legitimate outcome, not a server error.
            if "No counterfactuals found" in str(exc):
                logger.info("DiCE found no counterfactuals for this claim.")
            else:
                logger.exception("DiCE counterfactual generation failed")
                raise HTTPException(
                    status_code=500,
                    detail="Counterfactual generation failed.",
                ) from exc

    suggestions = (
        finalise_counterfactuals(classifier, original_row, cfs_df, search_features)
        if cfs_df is not None and len(cfs_df) > 0
        else []
    )

    if not suggestions:
        return RecourseResponse(
            applicable=True,
            message=(
                "This claim was predicted Deny, but lowering the length of stay or "
                "the cost of the claim was not enough to flip the decision."
            ),
            counterfactuals=[],
        )

    counterfactuals = [
        Counterfactual(
            changes=[
                RecourseChange(feature=feature, from_value=float(c["from"]), to_value=float(c["to"]))
                for feature, c in changes.items()
            ]
        )
        for changes in suggestions[:MAX_SUGGESTIONS]
    ]

    return RecourseResponse(
        applicable=True,
        message=f"This claim was predicted Deny. Below are {len(counterfactuals)} possible ways it could have been approved.",
        counterfactuals=counterfactuals,
    )
