
"""
schemas.py

Pydantic models defining the API's request/response contracts.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClaimFeatures(BaseModel):
    patient_id: str | None = None

    """Beneficiary-level engineered features expected by the trained model."""

    # Coverage and demographics
    
    BENE_ESRD_IND: int = Field(..., ge=0, le=1)
    BENE_HI_CVRAGE_TOT_MONS: int = Field(..., ge=0, le=12)
    BENE_SMI_CVRAGE_TOT_MONS: int = Field(..., ge=0, le=12)
    BENE_HMO_CVRAGE_TOT_MONS: int = Field(..., ge=0, le=12)
    PLAN_CVRG_MOS_NUM: int = Field(..., ge=0, le=12)
    age: int = Field(..., ge=0, le=120)
    is_deceased: int = Field(..., ge=0, le=1)

    # Chronic condition flags
    SP_ALZHDMTA: int = Field(..., ge=0, le=1)
    SP_CHF: int = Field(..., ge=0, le=1)
    SP_CHRNKIDN: int = Field(..., ge=0, le=1)
    SP_CNCR: int = Field(..., ge=0, le=1)
    SP_COPD: int = Field(..., ge=0, le=1)
    SP_DEPRESSN: int = Field(..., ge=0, le=1)
    SP_DIABETES: int = Field(..., ge=0, le=1)
    SP_ISCHMCHT: int = Field(..., ge=0, le=1)
    SP_OSTEOPRS: int = Field(..., ge=0, le=1)
    SP_RA_OA: int = Field(..., ge=0, le=1)
    SP_STRKETIA: int = Field(..., ge=0, le=1)
    chronic_condition_count: int = Field(..., ge=0, le=11)

    # Annual reimbursement/liability amounts
    MEDREIMB_IP: float = 0.0
    BENRES_IP: float = 0.0
    PPPYMT_IP: float = 0.0
    MEDREIMB_OP: float = 0.0
    BENRES_OP: float = 0.0
    PPPYMT_OP: float = 0.0
    MEDREIMB_CAR: float = 0.0
    BENRES_CAR: float = 0.0
    PPPYMT_CAR: float = 0.0

    # Utilization counts
    total_inpatient_claims: float = Field(..., ge=0)
    distinct_inpatient_diagnosis_codes: float = Field(..., ge=0)
    total_outpatient_claims: float = Field(..., ge=0)
    distinct_outpatient_diagnosis_codes: float = Field(..., ge=0)

    # Signed-log-transformed features
    total_inpatient_payment_amt_log: float
    total_outpatient_payment_amt_log: float
    total_inpatient_primary_payer_amt_log: float
    total_outpatient_primary_payer_amt_log: float
    total_inpatient_days_log: float
    avg_inpatient_payment_amt_log: float
    avg_outpatient_payment_amt_log: float

    # Derived ratio
    cost_per_claim: float = Field(..., ge=0)

    # Categorical codes
    BENE_SEX_IDENT_CD: str
    BENE_RACE_CD: str
    SP_STATE_CODE: str
    BENE_COUNTY_CD: str


class TopFactor(BaseModel):
    feature: str
    value: float
    shap_contribution: float = Field(
        ...,
        description="Positive pushes toward Deny, negative toward Approve",
    )


class PredictionResponse(BaseModel):
    predicted_label: Literal[0, 1] = Field(
        ..., description="0 = Approve, 1 = Deny"
    )
    predicted_label_name: Literal["Approve", "Deny"]
    predicted_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model's probability of Deny (0-1)",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Distance from the 0.5 decision boundary, scaled to 0-1",
    )
    routing: Literal["Final Decision", "Human Reviewer Queue"]
    top_factors: list[TopFactor]

    # Blockchain audit proof — these MUST be inside the response model.
    decision_id: str | None = None
    decision_hash: str | None = None
    transaction_hash: str | None = None
    block_number: int | None = None


class RecourseChange(BaseModel):
    feature: str
    from_value: float
    to_value: float


class Counterfactual(BaseModel):
    changes: list[RecourseChange]


class RecourseResponse(BaseModel):
    applicable: bool = Field(
        ..., description="False if the claim was already predicted Approve"
    )
    message: str
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
