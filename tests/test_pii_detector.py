import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_detects_ssn():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("Employee SSN is 123-45-6789 for records.")
    assert result.has_pii
    assert "SSN" in result.categories


def test_redacts_ssn():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("SSN: 123-45-6789")
    assert "[SSN]" in result.redacted_text
    assert "123-45-6789" not in result.redacted_text


def test_detects_email():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("Contact alice@example.com for details.")
    assert result.has_pii
    assert "EMAIL" in result.categories
    assert "[EMAIL]" in result.redacted_text


def test_detects_phone_us():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("Call us at +1-800-555-1234 anytime.")
    assert result.has_pii
    assert "PHONE" in result.categories


def test_detects_credit_card():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("Card number: 4111-1111-1111-1111")
    assert result.has_pii
    assert "CREDIT_CARD" in result.categories


def test_detects_iban():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("Transfer to GB29NWBK60161331926819")
    assert result.has_pii
    assert "IBAN" in result.categories


def test_clean_text_not_flagged():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("The quarterly revenue increased by 12 percent.")
    assert not result.has_pii
    assert result.redacted_text == "The quarterly revenue increased by 12 percent."


def test_allowed_category_not_flagged():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False, allowed_categories=["EMAIL"])
    result = d.scan("Contact alice@example.com for details.")
    assert not result.has_pii


def test_pii_scan_result_fields():
    from core.pii_detector import PIIDetector, PIIScanResult
    d = PIIDetector(use_ner=False)
    result = d.scan("SSN 123-45-6789")
    assert isinstance(result, PIIScanResult)
    assert hasattr(result, "has_pii")
    assert hasattr(result, "categories")
    assert hasattr(result, "redacted_text")
    assert hasattr(result, "span_count")
    assert result.span_count >= 1


def test_span_count_increments_per_match():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("SSN 123-45-6789 and email user@test.com")
    assert result.span_count >= 2


def test_multiple_allowed_categories():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False, allowed_categories=["SSN", "EMAIL"])
    result = d.scan("SSN 123-45-6789 and alice@example.com")
    assert not result.has_pii


# NER gate tests (skip gracefully if spaCy model not installed)

def test_ner_detects_person_name():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        import pytest; pytest.skip("spaCy en_core_web_sm not installed")
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=True)
    result = d.scan("The report was approved by John Smith yesterday.")
    assert result.has_pii
    assert "PERSON" in result.categories


def test_ner_allowed_person_not_flagged():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        import pytest; pytest.skip("spaCy en_core_web_sm not installed")
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=True, allowed_categories=["PERSON"])
    result = d.scan("Report approved by John Smith.")
    assert "PERSON" not in result.categories


def test_ner_graceful_degradation_without_model():
    from core.pii_detector import PIIDetector
    from unittest.mock import patch
    with patch.object(PIIDetector, "_load_ner", return_value=None):
        d = PIIDetector(use_ner=True)
        d._nlp = None
    result = d.scan("Contact alice@example.com.")
    assert result.has_pii
    assert "EMAIL" in result.categories


def test_span_count_two_ssns():
    from core.pii_detector import PIIDetector
    d = PIIDetector(use_ner=False)
    result = d.scan("First SSN 123-45-6789 and second SSN 987-65-4321")
    assert result.span_count == 2
    assert result.categories == ["SSN"]
