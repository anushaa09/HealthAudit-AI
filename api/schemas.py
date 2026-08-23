"""
schemas.py

Pydantic models defining the API's request/response contracts.

IMPORTANT SIMPLIFICATION (documented, not hidden): the trained model expects
beneficiary-level AGGREGATED features (e.g. total_inpatient_claims,
chronic_condition_count) computed from a full claims history -- the same
features produced by src/preprocessing/. A production system would compute
these live via a feature store (e.g. looking up Supabase by beneficiary ID).
For this capstone, the API accepts these engineered feature values directly
in the request body, mirroring the exact training schema. This keeps the
API's scope focused on serving the model + explainability + recourse,
rather than re-implementing the full feature pipeline as a live service.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ClaimFeatures(BaseModel):
    """
    One beneficiary's engineered feature profile -- matches EXACTLY the 44
    columns the trained model was fit on (see get_feature_columns() in
    src/training/train_baseline_models.py). Field order here doesn't matter;
    the model pipeline selects columns by name.
    """

    # --- Coverage / demographic fields ---
    BENE_ESRD_IND: int = Field(..., ge=0, le=1, description="1 = has End-Stage Renal Disease, 0 = no")
    BENE_HI_CVRAGE_TOT_MONS: int = Field(..., ge=0, le=12)
    BENE_SMI_CVRAGE_TOT_MONS: int = Field(..., ge=0, le=12)
    BENE_HMO_CVRAGE_TOT_MONS: int = Field(..., ge=0, le=12)
    PLAN_CVRG_MOS_NUM: int = Field(..., ge=0, le=12)
    age: int = Field(..., ge=0, le=120)
    is_deceased: int = Field(..., ge=0, le=1)

    # --- Chronic condition flags (1 = has condition, 0 = does not) ---
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

    # --- Annual reimbursement/liability amounts (raw dollars) ---
    MEDREIMB_IP: float = 0.0
    BENRES_IP: float = 0.0
    PPPYMT_IP: float = 0.0
    MEDREIMB_OP: float = 0.0
    BENRES_OP: float = 0.0
    PPPYMT_OP: float = 0.0
    MEDREIMB_CAR: float = 0.0
    BENRES_CAR: float = 0.0
    PPPYMT_CAR: float = 0.0

    # --- Utilization counts ---
    total_inpatient_claims: float = Field(..., ge=0)
    distinct_inpatient_diagnosis_codes: float = Field(..., ge=0)
    total_outpatient_claims: float = Field(..., ge=0)
    distinct_outpatient_diagnosis_codes: float = Field(..., ge=0)

    # --- Signed-log-transformed cost/utilization features ---
    # NOTE: these must ALREADY be signed-log transformed by the caller,
    # using the same transform as src/preprocessing/transformers.py:
    #   signed_log(x) = sign(x) * log1p(|x|)
    total_inpatient_payment_amt_log: float
    total_outpatient_payment_amt_log: float
    total_inpatient_primary_payer_amt_log: float
    total_outpatient_primary_payer_amt_log: float
    total_inpatient_days_log: float
    avg_inpatient_payment_amt_log: float
    avg_outpatient_payment_amt_log: float

    # --- Derived ratio ---
    cost_per_claim: float = Field(..., ge=0)

    # --- Categorical codes (as strings, matching CMS raw encodings) ---
    BENE_SEX_IDENT_CD: str
    BENE_RACE_CD: str
    SP_STATE_CODE: str
    BENE_COUNTY_CD: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "BENE_ESRD_IND": 0, "BENE_HI_CVRAGE_TOT_MONS": 12, "BENE_SMI_CVRAGE_TOT_MONS": 12,
                "BENE_HMO_CVRAGE_TOT_MONS": 0, "PLAN_CVRG_MOS_NUM": 0, "age": 72, "is_deceased": 0,
                "SP_ALZHDMTA": 0, "SP_CHF": 1, "SP_CHRNKIDN": 0, "SP_CNCR": 0, "SP_COPD": 0,
                "SP_DEPRESSN": 1, "SP_DIABETES": 1, "SP_ISCHMCHT": 1, "SP_OSTEOPRS": 0,
                "SP_RA_OA": 0, "SP_STRKETIA": 0, "chronic_condition_count": 4,
                "MEDREIMB_IP": 4000.0, "BENRES_IP": 1000.0, "PPPYMT_IP": 0.0,
                "MEDREIMB_OP": 800.0, "BENRES_OP": 200.0, "PPPYMT_OP": 0.0,
                "MEDREIMB_CAR": 500.0, "BENRES_CAR": 125.0, "PPPYMT_CAR": 0.0,
                "total_inpatient_claims": 1.0, "distinct_inpatient_diagnosis_codes": 3.0,
                "total_outpatient_claims": 2.0, "distinct_outpatient_diagnosis_codes": 2.0,
                "total_inpatient_payment_amt_log": 8.29, "total_outpatient_payment_amt_log": 4.11,
                "total_inpatient_primary_payer_amt_log": 0.0, "total_outpatient_primary_payer_amt_log": 0.0,
                "total_inpatient_days_log": 1.79, "avg_inpatient_payment_amt_log": 8.29,
                "avg_outpatient_payment_amt_log": 3.4, "cost_per_claim": 1350.0,
                "BENE_SEX_IDENT_CD": "1", "BENE_RACE_CD": "1", "SP_STATE_CODE": "34",
                "BENE_COUNTY_CD": "230",
            }
        }
    }


class TopFactor(BaseModel):
    feature: str
    value: float
    shap_contribution: float = Field(..., description="Positive pushes toward Deny, negative toward Approve")


class PredictionResponse(BaseModel):
    predicted_label: Literal[0, 1] = Field(..., description="0 = Approve, 1 = Deny")
    predicted_label_name: Literal["Approve", "Deny"]
    predicted_probability: float = Field(..., description="Model's probability of Deny (0-1)")
    confidence: float = Field(..., description="How far the probability sits from the 0.5 decision boundary")
    routing: Literal["Final Decision", "Human Reviewer Queue"]
    top_factors: list[TopFactor]


class RecourseChange(BaseModel):
    feature: str
    from_value: float
    to_value: float


class Counterfactual(BaseModel):
    changes: list[RecourseChange]


class RecourseResponse(BaseModel):
    applicable: bool = Field(..., description="False if the claim was already predicted Approve")
    message: str
    counterfactuals: list[Counterfactual] = []