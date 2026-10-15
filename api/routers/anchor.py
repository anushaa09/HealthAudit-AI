from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from blockchain.chain_client import (
    anchor_decision,
    get_anchored_hash,
)


router = APIRouter(
    prefix="/anchor",
    tags=["Blockchain"],
)


# -------------------------------------------------------------------
# Request / Response models
# -------------------------------------------------------------------

class AnchorRequest(BaseModel):
    decision_id: str
    hash: str


class AnchorResponse(BaseModel):
    success: bool
    decision_id: str
    hash: str
    transaction_hash: str
    block_number: int


class VerifyResponse(BaseModel):
    success: bool
    decision_id: str
    hash: str
    timestamp: int


# -------------------------------------------------------------------
# Anchor decision
# -------------------------------------------------------------------

@router.post(
    "",
    response_model=AnchorResponse,
)
def anchor(request: AnchorRequest):
    """
    Store a decision hash on Ethereum Sepolia.
    """

    try:
        result = anchor_decision(
            decision_id=request.decision_id,
            decision_hash=request.hash,
        )

        return AnchorResponse(
            success=True,
            decision_id=result["decision_id"],
            hash=result["hash"],
            transaction_hash=result["transaction_hash"],
            block_number=result["block_number"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Blockchain anchoring failed: {exc}",
        )


# -------------------------------------------------------------------
# Verify anchored decision
# -------------------------------------------------------------------

@router.get(
    "/{decision_id}",
    response_model=VerifyResponse,
)
def verify_anchor(
    decision_id: str,
):
    """
    Retrieve an anchored decision from Ethereum.
    """

    try:
        result = get_anchored_hash(
            decision_id
        )

        return VerifyResponse(
            success=True,
            decision_id=result["decision_id"],
            hash=result["hash"],
            timestamp=result["timestamp"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Blockchain verification failed: {exc}",
        )