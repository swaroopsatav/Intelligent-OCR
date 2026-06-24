from backend.services import extraction_service
from backend.services.extraction_service import (
    EXTRACTION_FIELDS,
    ExtractionService
)


def test_extract_data_parses_json_from_llm_response(monkeypatch):
    def fake_generate(prompt, **kwargs):
        return """
Some preface that should be ignored.
{
  "document_title": "Invoice",
  "document_type": "Invoice",
  "name": "Swaroop Satav",
  "company_name": "Acme Services",
  "document_id": "INV-001",
  "date": "24/06/2026",
  "address": "Pune",
  "phone": "9876543210",
  "email": "billing@example.com",
  "invoice_number": "INV-001",
  "total_amount": "1200",
  "pan": "ABCDE1234F",
  "gst": "27ABCDE1234F1Z5",
  "aadhaar": "",
  "dynamic_fields": {
    "payment_status": "Paid"
  }
}
Trailing text should be ignored.
"""

    monkeypatch.setattr(
        extraction_service.LLMService,
        "generate",
        staticmethod(fake_generate)
    )

    result = ExtractionService.extract_data(
        "invoice text"
    )

    assert result["document_title"] == "Invoice"
    assert result["document_id"] == "INV-001"
    assert result["dynamic_fields"] == {
        "payment_status": "Paid"
    }


def test_extract_data_returns_all_fields_when_json_is_invalid(monkeypatch):
    monkeypatch.setattr(
        extraction_service.LLMService,
            "generate",
        staticmethod(lambda prompt, **kwargs: "not valid json")
    )

    result = ExtractionService.extract_data(
        "unstructured text"
    )

    assert set(EXTRACTION_FIELDS).issubset(result.keys())
    assert result["dynamic_fields"] == {}
    assert result["document_title"] == "unstructured text"
    assert "error" in result


def test_extract_data_uses_empty_dynamic_fields_for_non_dict(monkeypatch):
    monkeypatch.setattr(
        extraction_service.LLMService,
        "generate",
        staticmethod(
            lambda prompt, **kwargs: (
                '{"document_title": "Resume", '
                '"dynamic_fields": ["skills"]}'
            )
        )
    )

    result = ExtractionService.extract_data(
        "resume text"
    )

    assert result["document_title"] == "Resume"
    assert result["dynamic_fields"] == {}


def test_normalize_gst_fixes_common_ocr_digit_letter_confusion():
    assert ExtractionService.normalize_gst(
        "27ABCDE1234F1ZS"
    ) == "27ABCDE1234F1Z5"


def test_fallback_extracts_invoice_line_items_and_reconciliation():
    result = ExtractionService.fallback_extract_data(
        "\n".join(
            [
                "INVOICE",
                "Invoice No: INV-001",
                "GST: 27ABCDE1234F1ZS",
                "Total Amount: Rs. 400.00",
                "Notebook 2 Rs. 240.00",
                "Pen Set 1 Rs. 160.00"
            ]
        )
    )

    processed = ExtractionService.post_process_extraction(
        result,
        "\n".join(
            [
                "Notebook 2 Rs. 240.00",
                "Pen Set 1 Rs. 160.00"
            ]
        )
    )

    assert processed["gst"] == "27ABCDE1234F1Z5"
    assert processed["dynamic_fields"]["line_items"][0]["description"] == "Notebook"
    assert processed["dynamic_fields"]["subtotal"] == "400.00"
    assert processed["dynamic_fields"]["total_reconciliation"]["matches"] is True
