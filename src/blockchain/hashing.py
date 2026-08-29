import hashlib
import json
from typing import Any, Dict


def canonicalize(record: Dict[str, Any]) -> str:
    """
    Convert a decision record dict into a deterministic JSON string.
    Same input -> same string, every time, regardless of key order
    or how the dict was built.
    """
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def hash_decision_record(record: Dict[str, Any]) -> str:
    """
    Produce a SHA-256 hex digest for a claim decision record.
    """
    canonical_str = canonicalize(record)
    digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    return digest


def verify_decision_record(record: Dict[str, Any], expected_hash: str) -> bool:
    """
    Recompute the hash from a record and check it against a stored hash.
    """
    return hash_decision_record(record) == expected_hash


def build_hashable_record(predict_response: dict, input_features: dict) -> dict:
    """
    predict_response: whatever /predict returned
    input_features: the ClaimFeatures the caller sent in
    """
    return {
        "input_features": input_features,
        "predicted_label": predict_response["predicted_label"],
        "predicted_label_name": predict_response["predicted_label_name"],
        "predicted_probability": predict_response["predicted_probability"],
        "confidence": predict_response["confidence"],
        "routing": predict_response["routing"],
        "top_factors": predict_response["top_factors"],
    }