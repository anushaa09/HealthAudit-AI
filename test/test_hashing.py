"""
Quick tests for src/blockchain/hashing.py
Run with: pytest test/test_hashing.py -v
(or just: python test/test_hashing.py  -- see __main__ block at bottom)
"""

from src.blockchain.hashing import (
    canonicalize,
    hash_decision_record,
    verify_decision_record,
    build_hashable_record,
)


# ---- Sample data, similar shape to what /predict returns ----

SAMPLE_PREDICT_RESPONSE = {
    "prediction": "Deny",
    "probability": 0.83,
    "confidence": 0.66,
    "routing": "Final Decision",
    "top_shap_factors": [
        {"feature": "total_cost", "shap_value": 0.42},
        {"feature": "chronic_condition_count", "shap_value": 0.31},
    ],
}

SAMPLE_INPUT_FEATURES = {
    "age": 67,
    "chronic_condition_count": 3,
    "total_cost": 15200.50,
    # ... imagine the other ~41 ClaimFeatures fields here
}


def test_canonicalize_is_deterministic():
    """Same dict, different key order -> identical JSON string."""
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert canonicalize(a) == canonicalize(b)


def test_hash_is_deterministic():
    """Hashing the same record twice gives the same hash."""
    record = build_hashable_record(SAMPLE_PREDICT_RESPONSE, SAMPLE_INPUT_FEATURES)
    hash1 = hash_decision_record(record)
    hash2 = hash_decision_record(record)
    assert hash1 == hash2


def test_hash_is_sha256_hex():
    """Hash should be a 64-char hex string (SHA-256 digest length)."""
    record = build_hashable_record(SAMPLE_PREDICT_RESPONSE, SAMPLE_INPUT_FEATURES)
    h = hash_decision_record(record)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_different_records_give_different_hashes():
    record1 = build_hashable_record(SAMPLE_PREDICT_RESPONSE, SAMPLE_INPUT_FEATURES)

    modified_response = dict(SAMPLE_PREDICT_RESPONSE)
    modified_response["prediction"] = "Approve"  # flip the decision
    record2 = build_hashable_record(modified_response, SAMPLE_INPUT_FEATURES)

    assert hash_decision_record(record1) != hash_decision_record(record2)


def test_verify_decision_record_true_case():
    record = build_hashable_record(SAMPLE_PREDICT_RESPONSE, SAMPLE_INPUT_FEATURES)
    expected_hash = hash_decision_record(record)
    assert verify_decision_record(record, expected_hash) is True


def test_verify_decision_record_false_case_tampered():
    """If even one field changes after hashing, verification must fail."""
    record = build_hashable_record(SAMPLE_PREDICT_RESPONSE, SAMPLE_INPUT_FEATURES)
    original_hash = hash_decision_record(record)

    tampered_record = dict(record)
    tampered_record["prediction"] = "Approve"  # simulate tampering

    assert verify_decision_record(tampered_record, original_hash) is False


# Lets you run this file directly without pytest, for a quick sanity check
if __name__ == "__main__":
    tests = [
        test_canonicalize_is_deterministic,
        test_hash_is_deterministic,
        test_hash_is_sha256_hex,
        test_different_records_give_different_hashes,
        test_verify_decision_record_true_case,
        test_verify_decision_record_false_case_tampered,
    ]
    for t in tests:
        t()
        print(f"✅ {t.__name__}")
    print("\nAll hashing tests passed.")