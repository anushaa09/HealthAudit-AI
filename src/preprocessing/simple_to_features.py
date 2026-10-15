"""
simple_to_features.py

Converts a simplified, patient-facing questionnaire into the full
ClaimFeatures schema the trained model expects. This is the layer that
lets the frontend collect plain-language answers instead of raw
engineered/log-transformed values, per panel feedback on the input UX.
"""

import math


def signed_log(x: float) -> float:
    """Matches src/preprocessing/transformers.py: sign(x) * log1p(|x|)"""
    if x == 0:
        return 0.0
    return math.copysign(1, x) * math.log1p(abs(x))


def build_claim_features(simple_answers: dict) -> dict:
    """
    simple_answers: dict matching the SimpleClaimInput schema (see below).
    Returns: a dict matching ClaimFeatures exactly.
    """
    total_inpatient_payment = (
        simple_answers["medreimb_ip"] + simple_answers["benres_ip"] + simple_answers.get("pppymt_ip", 0)
    )
    total_outpatient_payment = (
        simple_answers["medreimb_op"] + simple_answers["benres_op"] + simple_answers.get("pppymt_op", 0)
    )
    total_claims = max(
        simple_answers["total_inpatient_claims"] + simple_answers["total_outpatient_claims"], 1
    )

    chronic_flags = [
        "sp_alzhdmta", "sp_chf", "sp_chrnkidn", "sp_cncr", "sp_copd",
        "sp_depressn", "sp_diabetes", "sp_ischmcht", "sp_osteoprs",
        "sp_ra_oa", "sp_stroketia",
    ]
    chronic_condition_count = sum(simple_answers[flag] for flag in chronic_flags)

    return {
        "BENE_ESRD_IND": simple_answers["bene_esrd_ind"],
        "BENE_HI_CVRAGE_TOT_MONS": simple_answers["part_a_months"],
        "BENE_SMI_CVRAGE_TOT_MONS": simple_answers["part_b_months"],
        "BENE_HMO_CVRAGE_TOT_MONS": simple_answers.get("hmo_months", 0),
        "PLAN_CVRG_MOS_NUM": simple_answers["part_a_months"],
        "age": simple_answers["age"],
        "is_deceased": 0,

        "SP_ALZHDMTA": simple_answers["sp_alzhdmta"],
        "SP_CHF": simple_answers["sp_chf"],
        "SP_CHRNKIDN": simple_answers["sp_chrnkidn"],
        "SP_CNCR": simple_answers["sp_cncr"],
        "SP_COPD": simple_answers["sp_copd"],
        "SP_DEPRESSN": simple_answers["sp_depressn"],
        "SP_DIABETES": simple_answers["sp_diabetes"],
        "SP_ISCHMCHT": simple_answers["sp_ischmcht"],
        "SP_OSTEOPRS": simple_answers["sp_osteoprs"],
        "SP_RA_OA": simple_answers["sp_ra_oa"],
        "SP_STRKETIA": simple_answers["sp_stroketia"],
        "chronic_condition_count": chronic_condition_count,

        "MEDREIMB_IP": simple_answers["medreimb_ip"],
        "BENRES_IP": simple_answers["benres_ip"],
        "PPPYMT_IP": simple_answers.get("pppymt_ip", 0),
        "MEDREIMB_OP": simple_answers["medreimb_op"],
        "BENRES_OP": simple_answers["benres_op"],
        "PPPYMT_OP": simple_answers.get("pppymt_op", 0),
        "MEDREIMB_CAR": 0.0,
        "BENRES_CAR": 0.0,
        "PPPYMT_CAR": 0.0,

        "total_inpatient_claims": simple_answers["total_inpatient_claims"],
        "distinct_inpatient_diagnosis_codes": simple_answers["inpatient_diagnoses"],
        "total_outpatient_claims": simple_answers["total_outpatient_claims"],
        "distinct_outpatient_diagnosis_codes": simple_answers["outpatient_diagnoses"],

        "total_inpatient_payment_amt_log": signed_log(total_inpatient_payment),
        "total_outpatient_payment_amt_log": signed_log(total_outpatient_payment),
        "total_inpatient_primary_payer_amt_log": signed_log(simple_answers.get("pppymt_ip", 0)),
        "total_outpatient_primary_payer_amt_log": signed_log(simple_answers.get("pppymt_op", 0)),
        "total_inpatient_days_log": signed_log(simple_answers["inpatient_days"]),
        "avg_inpatient_payment_amt_log": signed_log(
            total_inpatient_payment / max(simple_answers["total_inpatient_claims"], 1)
        ),
        "avg_outpatient_payment_amt_log": signed_log(
            total_outpatient_payment / max(simple_answers["total_outpatient_claims"], 1)
        ),

        "cost_per_claim": (total_inpatient_payment + total_outpatient_payment) / total_claims,

        "BENE_SEX_IDENT_CD": simple_answers["sex"],
        "BENE_RACE_CD": simple_answers["race"],
        "SP_STATE_CODE": simple_answers["state"],
        "BENE_COUNTY_CD": simple_answers["county"],
    }