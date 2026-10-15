"""
data_loader.py
Typed loading functions for the raw CMS DE-SynPUF files.
"""

from pathlib import Path

import pandas as pd

from src.config.paths import BENEFICIARY_FILE, INPATIENT_CLAIMS_FILE, OUTPATIENT_CLAIMS_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CLAIMS_FINANCIAL_COLUMNS = ["CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT"]
_CLAIMS_DATE_COLUMNS = ["CLM_FROM_DT", "CLM_THRU_DT"]
_CHRONIC_CONDITION_COLUMNS = [
    "SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
    "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS", "SP_RA_OA", "SP_STRKETIA",
]
_BENEFICIARY_FINANCIAL_COLUMNS = [
    "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
    "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
    "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR",
]

# Genuinely numeric columns (months of coverage, 0-12) that were left as
# strings in the original dtype pass -- caught later when training threw
# "could not convert string to float" on these. Fixed at the source here,
# per our own convention: dtype correction belongs in the loader, not
# patched downstream.
_BENEFICIARY_NUMERIC_COLUMNS = [
    "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS",
    "BENE_HMO_CVRAGE_TOT_MONS",
    "PLAN_CVRG_MOS_NUM",
]

# BENE_ESRD_IND (End-Stage Renal Disease indicator) uses a different flag
# encoding than the SP_* chronic conditions: '0' = No, 'Y' = Yes -- NOT the
# 1/2 convention used elsewhere. Handled separately to avoid conflating the
# two different encodings.
_ESRD_FLAG_MAP = {"0": 0, "Y": 1}


def _parse_yyyymmdd_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    return df


def _convert_financial_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        original = df[col]
        df[col] = pd.to_numeric(original, errors="coerce")
        n_coerced = df[col].isna().sum() - original.isna().sum()
        if n_coerced > 0:
            logger.warning("Column '%s': %d values coerced to NaN.", col, n_coerced)
    return df


def _convert_chronic_condition_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    flag_map = {"1": 1, "2": 0}
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].map(flag_map)
    return df


def load_beneficiary_summary(path: Path = BENEFICIARY_FILE) -> pd.DataFrame:
    logger.info("Loading beneficiary summary from: %s", path)
    df = pd.read_csv(path, dtype=str)
    logger.info("Loaded %d beneficiary rows, %d columns.", len(df), df.shape[1])
    df = _parse_yyyymmdd_columns(df, ["BENE_BIRTH_DT", "BENE_DEATH_DT"])
    df = _convert_chronic_condition_flags(df, _CHRONIC_CONDITION_COLUMNS)
    df = _convert_financial_columns(df, _BENEFICIARY_FINANCIAL_COLUMNS)
    df = _convert_financial_columns(df, _BENEFICIARY_NUMERIC_COLUMNS)  # reuse: same to_numeric logic

    df["BENE_ESRD_IND"] = df["BENE_ESRD_IND"].map(_ESRD_FLAG_MAP)
    n_unmapped = df["BENE_ESRD_IND"].isna().sum()
    if n_unmapped > 0:
        logger.warning("BENE_ESRD_IND: %d values did not match {'0','Y'} and became NaN.", n_unmapped)

    return df


def load_inpatient_claims(path: Path = INPATIENT_CLAIMS_FILE) -> pd.DataFrame:
    logger.info("Loading inpatient claims from: %s", path)
    df = pd.read_csv(path, dtype=str)
    logger.info("Loaded %d inpatient claim rows, %d columns.", len(df), df.shape[1])
    df = _parse_yyyymmdd_columns(df, _CLAIMS_DATE_COLUMNS)
    df = _convert_financial_columns(df, _CLAIMS_FINANCIAL_COLUMNS)
    return df


def load_outpatient_claims(path: Path = OUTPATIENT_CLAIMS_FILE) -> pd.DataFrame:
    logger.info("Loading outpatient claims from: %s", path)
    df = pd.read_csv(path, dtype=str)
    logger.info("Loaded %d outpatient claim rows, %d columns.", len(df), df.shape[1])
    df = _parse_yyyymmdd_columns(df, _CLAIMS_DATE_COLUMNS)
    df = _convert_financial_columns(df, _CLAIMS_FINANCIAL_COLUMNS)
    return df