from pathlib import Path

from backend.api.routes import ocr as ocr_route
from backend.services import llm_service


def test_sample_document_full_pipeline(client, monkeypatch):
    sample_pdf = Path("samples/demo_invoice.pdf")

    assert sample_pdf.exists()

    monkeypatch.setattr(
        ocr_route.OCRService,
        "extract_text",
        staticmethod(
            lambda image_path: {
                "text": (
                    "INVOICE\n"
                    "Invoice No: INV-2026-014\n"
                    "Date: 24/06/2026\n"
                    "Bill To: Swaroop Satav\n"
                    "Email: billing@example.com\n"
                    "Phone: 9876543210\n"
                    "GST: 27ABCDE1234F1Z5\n"
                    "PAN: ABCDE1234F\n"
                    "Total Amount: Rs. 12,450.00"
                ),
                "confidence": 0.84,
                "line_count": 9,
                "bounding_boxes": [],
                "table_rows": [],
                "processing_time": 0.01,
                "status": "success"
            }
        )
    )

    def fake_generate(prompt, **kwargs):
        if "Correct OCR errors" in prompt:
            return (
                "INVOICE\n"
                "Invoice No: INV-2026-014\n"
                "Date: 24/06/2026\n"
                "Bill To: Swaroop Satav\n"
                "Email: billing@example.com\n"
                "Phone: 9876543210\n"
                "GST: 27ABCDE1234F1Z5\n"
                "PAN: ABCDE1234F\n"
                "Total Amount: Rs. 12,450.00"
            )

        if "Classify document" in prompt:
            return '{"document_type": "Invoice", "confidence": 0.91}'

        if "Extract information" in prompt:
            return """
{
  "document_title": "Invoice",
  "document_type": "Invoice",
  "name": "Swaroop Satav",
  "company_name": "",
  "document_id": "INV-2026-014",
  "date": "24/06/2026",
  "address": "",
  "phone": "9876543210",
  "email": "billing@example.com",
  "invoice_number": "INV-2026-014",
  "total_amount": "Rs. 12,450.00",
  "pan": "ABCDE1234F",
  "gst": "27ABCDE1234F1Z5",
  "aadhaar": "",
  "dynamic_fields": {
    "payment_status": "Pending",
    "subtotal": "12450"
  }
}
"""

        if "QUESTION" in prompt:
            return "The invoice total is Rs. 12,450.00."

        return ""

    monkeypatch.setattr(
        llm_service.LLMService,
        "generate",
        staticmethod(fake_generate)
    )

    with sample_pdf.open("rb") as file_handle:
        upload_response = client.post(
            "/upload",
            files={
                "file": (
                    sample_pdf.name,
                    file_handle,
                    "application/pdf"
                )
            }
        )

    assert upload_response.status_code == 200

    uploaded_file_path = upload_response.json()["file_path"]

    preprocess_response = client.post(
        "/preprocess",
        params={
            "file_path": uploaded_file_path
        }
    )

    assert preprocess_response.status_code == 200

    processed_image = preprocess_response.json()["processed_images"][0]

    ocr_response = client.post(
        "/ocr",
        params={
            "image_path": processed_image
        }
    )

    assert ocr_response.status_code == 200

    ocr_text = ocr_response.json()["text"]

    correction_response = client.post(
        "/correct",
        params={
            "ocr_text": ocr_text
        }
    )

    assert correction_response.status_code == 200

    corrected_text = correction_response.json()["corrected_text"]

    classification_response = client.post(
        "/classify",
        params={
            "text": corrected_text
        }
    )

    assert classification_response.status_code == 200
    assert classification_response.json()["document_type"] == "Invoice"

    extraction_response = client.post(
        "/extract",
        json={
            "text": corrected_text
        }
    )

    assert extraction_response.status_code == 200

    extracted_data = extraction_response.json()

    validation_response = client.post(
        "/validate",
        json=extracted_data
    )

    assert validation_response.status_code == 200
    assert validation_response.json()["total_amount"] is True

    qa_response = client.post(
        "/qa",
        params={
            "document_text": corrected_text,
            "question": "What is the invoice total?"
        }
    )

    assert qa_response.status_code == 200
    assert "Rs. 12,450.00" in qa_response.json()["answer"]
