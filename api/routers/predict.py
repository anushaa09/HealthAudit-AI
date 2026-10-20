
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_classifier, get_explainer
from api.schemas import ClaimFeatures, PredictionResponse
from api.supabase_client import supabase
from blockchain.chain_client import anchor_decision
from src.blockchain.hashing import (
    hash_decision_record,
    build_hashable_record,
)
from src.explainability.shap_explainer import explain_single_claim
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

CONFIDENCE_THRESHOLD = 0.5


@router.post("/predict", response_model=PredictionResponse)
def predict_claim(
    claim: ClaimFeatures,
    classifier=Depends(get_classifier),
    explainer=Depends(get_explainer),
) -> PredictionResponse:
    # 1. Generate prediction and SHAP explanation.
    claim_payload = claim.model_dump()
    patient_id = claim_payload.pop("patient_id", None)
    claim_payload = claim.model_dump()
    patient_id = claim_payload.pop("patient_id", None)
    claim_id = claim_payload.pop("claim_id", None)

    claim_features = claim_payload
    claim_df = pd.DataFrame([claim_features])
    try:
        explanation = explain_single_claim(
            classifier,
            explainer,
            claim_df,
            top_n=5,
        )

        probability = float(explanation["predicted_probability"])
        label = int(explanation["predicted_label"])

        if label not in (0, 1):
            raise ValueError(f"Unexpected prediction label: {label}")

        if not 0.0 <= probability <= 1.0:
            raise ValueError("Predicted probability must be between 0 and 1.")

    except Exception as exc:
        logger.exception("Prediction or SHAP explanation failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction or explanation generation failed.",
        ) from exc

    confidence = round(abs(probability - 0.5) * 2, 4)

    routing = (
        "Human Reviewer Queue"
        if confidence < CONFIDENCE_THRESHOLD
        else "Final Decision"
    )

    response_data = {
        "predicted_label": label,
        "predicted_label_name": "Deny" if label == 1 else "Approve",
        "predicted_probability": probability,
        "confidence": confidence,
        "routing": routing,
        "top_factors": explanation["top_factors"],
    }

    # 2. Generate a deterministic hash for the decision.
    hashable_record = build_hashable_record(
        response_data,
        claim_features,
    )
    decision_hash = hash_decision_record(hashable_record)
    decision_id = str(uuid.uuid4())

    # 3. Save the audit record before attempting blockchain anchoring.
    row = {
        "decision_id": decision_id,
        "claim_features": claim_features,
        "predicted_label": label,
        "predicted_label_name": response_data["predicted_label_name"],
        "predicted_probability": probability,
        "confidence": confidence,
        "routing": routing,
        "top_factors": response_data["top_factors"],
        "decision_hash": decision_hash,
        "patient_id": claim.patient_id,
        "patient_id": patient_id,
        "claim_id": claim_id,
        
    }

    try:
        supabase.table("audit_decisions").insert(row).execute()
        logger.info("Audit record inserted: %s", decision_id)

    except Exception as exc:
        logger.exception(
            "Supabase insert failed for decision %s",
            decision_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction generated, but audit storage failed. "
                "No blockchain anchor was attempted."
            ),
        ) from exc

    # 4. Anchor the decision hash on Sepolia.
    try:
        chain_result = anchor_decision(
            decision_id=decision_id,
            decision_hash=decision_hash,
        )

        transaction_hash = chain_result.get("transaction_hash")
        block_number = int(chain_result["block_number"])

        if not transaction_hash:
            raise RuntimeError(
                "Blockchain result did not contain a transaction hash."
            )

        # 5. Save transaction details against the existing audit record.
        update_result = (
            supabase.table("audit_decisions")
            .update({
                "transaction_hash": transaction_hash,
                "block_number": block_number,
            })
            .eq("decision_id", decision_id)
            .execute()
        )

        if not update_result.data:
            raise RuntimeError(
                f"Supabase updated no rows for decision {decision_id}."
            )

        logger.info(
            "Blockchain details saved: decision=%s block=%s tx=%s",
            decision_id,
            block_number,
            transaction_hash,
        )

    except Exception as exc:
        logger.exception(
            "Blockchain anchoring or Supabase update failed for %s",
            decision_id,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "The prediction and audit record were saved, but "
                    "blockchain anchoring or transaction-detail storage "
                    "failed. Check server logs before retrying because "
                    "the transaction may already have succeeded."
                ),
                "decision_id": decision_id,
                "decision_hash": decision_hash,
            },
        ) from exc

    # 6. Return the prediction and its audit proof.
    return PredictionResponse(
    **response_data,
    decision_id=decision_id,
    decision_hash=decision_hash,
    transaction_hash=transaction_hash,
    block_number=block_number,
)
