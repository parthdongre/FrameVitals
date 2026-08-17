import pandas as pd

import framevitals
from framevitals.column_roles import infer_column_roles, summarize_roles
from framevitals.dataset_signals import detect_dataset_signals
from framevitals.profiler import build_profile
from framevitals.semantic_types import infer_semantic_types


def test_semantic_type_detector_recognizes_common_value_patterns():
    cases = {
        "email": pd.Series(["a@example.com", "b@example.com", "c@example.org"]),
        "url": pd.Series(["https://example.com/a", "http://openai.com", "www.python.org"]),
        "uuid": pd.Series([
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
        ]),
        "ip_address": pd.Series(["192.168.1.1", "10.0.0.8", "2001:db8::1"]),
        "phone": pd.Series(["+91 98765 43210", "020-1234-5678", "+1 (415) 555-0100"]),
        "percentage": pd.Series(["12%", "3.5%", "-0.5%"]),
        "currency": pd.Series(["₹1,200", "$19.99", "INR 5000"]),
        "json": pd.Series(['{"a": 1}', '[1, 2]', '{"ok": true}']),
        "boolean_token": pd.Series(["yes", "NO", "true", "off"]),
    }

    for expected, series in cases.items():
        result = infer_semantic_types(series)
        assert result["primary"] == expected
        assert result["candidates"][0]["confidence"] >= 0.7


def test_semantic_type_detector_is_bounded_and_ignores_numeric_columns():
    values = [f"user{index}@example.com" for index in range(250)]
    text_result = infer_semantic_types(pd.Series(values), max_samples=40)
    numeric_result = infer_semantic_types(pd.Series([1, 2, 3, 4]))

    assert text_result["primary"] == "email"
    assert text_result["sample_size"] == 40
    assert numeric_result == {
        "primary": None,
        "candidates": [],
        "sample_size": 0,
    }


def test_value_semantics_augment_column_roles_without_name_hints():
    df = pd.DataFrame({
        "contact_value": [
            "a@example.com",
            "b@example.com",
            "c@example.com",
            "d@example.com",
        ],
        "record_key": [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
        ],
        "web": [
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
            "https://example.com/d",
        ],
    })

    roles = infer_column_roles(df)

    assert roles["contact_value"]["semantic_type"] == "email"
    assert "email_like" in roles["contact_value"]["roles"]
    assert "sensitive" in roles["contact_value"]["roles"]

    assert roles["record_key"]["semantic_type"] == "uuid"
    assert "uuid_like" in roles["record_key"]["roles"]
    assert "id_like" in roles["record_key"]["roles"]

    assert roles["web"]["semantic_type"] == "url"
    assert "url_like" in roles["web"]["roles"]

    summary = summarize_roles(roles)
    assert summary["email_like"] == ["contact_value"]
    assert summary["uuid_like"] == ["record_key"]
    assert summary["url_like"] == ["web"]


def test_dataset_signals_reuse_semantic_roles():
    df = pd.DataFrame({
        "contact": ["a@example.com", "b@example.com", "c@example.com"],
        "homepage": ["https://a.example", "https://b.example", "https://c.example"],
        "host": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
        "price_text": ["₹100", "₹200", "₹300"],
    })
    profile = build_profile(df)
    roles = infer_column_roles(df)

    signals = detect_dataset_signals(df, profile, column_roles=roles)

    assert signals["has_email_like_columns"] is True
    assert signals["email_like_columns"] == ["contact"]
    assert signals["has_url_like_columns"] is True
    assert signals["url_like_columns"] == ["homepage"]
    assert signals["has_ip_address_like_columns"] is True
    assert signals["ip_address_like_columns"] == ["host"]
    assert signals["has_currency_like_columns"] is True
    assert signals["currency_like_columns"] == ["price_text"]
    assert set(signals["sensitive_columns"]) >= {"contact", "host"}


def test_column_result_surfaces_semantic_information():
    df = pd.DataFrame({
        "contact": [
            "a@example.com",
            "b@example.com",
            "c@example.com",
            "d@example.com",
        ],
        "value": [1, 2, 3, 4],
    })

    result = framevitals.analyze(df, mode="quick")
    column = result.column("contact")

    assert column.semantic_type == "email"
    assert column.semantic_sample_size == 4
    assert column.semantic_candidates[0]["type"] == "email"
    assert "email_like" in column.roles
