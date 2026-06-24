from backend.api.routes import (
    classify as classify_route,
    correct as correct_route,
    extract as extract_route,
    ocr as ocr_route,
    preprocess as preprocess_route,
    qa as qa_route,
    upload as upload_route
)
from backend.models.qwen_loader import ModelLoadingError


def test_health_check_smoke(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_upload_route_smoke(client, monkeypatch):
    async def fake_save_file(file):
        return {
            "filename": file.filename,
            "file_path": "data/uploads/test.pdf",
            "file_size": 4,
            "message": "Upload successful"
        }

    monkeypatch.setattr(
        upload_route.UploadService,
        "save_file",
        fake_save_file
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.pdf",
                b"test",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "test.pdf"


def test_preprocess_route_smoke(client, monkeypatch):
    monkeypatch.setattr(
        preprocess_route.Path,
        "exists",
        lambda self: True
    )

    monkeypatch.setattr(
        preprocess_route.ImageService,
        "preprocess_image",
        staticmethod(
            lambda image_path: {
                "original_image": image_path,
                "processed_image": "data/processed/test.png",
                "applied_steps": ["threshold"]
            }
        )
    )

    response = client.post(
        "/preprocess",
        params={
            "file_path": "data/uploads/test.png"
        }
    )

    assert response.status_code == 200
    assert response.json()["processed_images"] == [
        "data/processed/test.png"
    ]


def test_ocr_route_smoke(client, monkeypatch):
    monkeypatch.setattr(
        ocr_route.OCRService,
        "extract_text",
        staticmethod(
            lambda image_path: {
                "text": "Invoice",
                "confidence": 0.95,
                "line_count": 1,
                "processing_time": 0.01,
                "status": "success"
            }
        )
    )

    response = client.post(
        "/ocr",
        params={
            "image_path": "data/processed/test.png"
        }
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Invoice"


def test_ocr_route_returns_clear_failure(client, monkeypatch):
    monkeypatch.setattr(
        ocr_route.OCRService,
        "extract_text",
        staticmethod(
            lambda image_path: {
                "text": "",
                "confidence": 0,
                "line_count": 0,
                "processing_time": 0,
                "status": "error",
                "message": "image could not be read"
            }
        )
    )

    response = client.post(
        "/ocr",
        params={
            "image_path": "data/processed/bad.png"
        }
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "OCR failed: image could not be read"
    )


def test_correct_route_smoke(client, monkeypatch):
    monkeypatch.setattr(
        correct_route.CorrectionService,
        "correct_text",
        staticmethod(
            lambda text: {
                "corrected_text": "Corrected invoice",
                "method": "test",
                "processing_time": 0.01,
                "status": "success"
            }
        )
    )

    response = client.post(
        "/correct",
        params={
            "ocr_text": "Correted invoice"
        }
    )

    assert response.status_code == 200
    assert response.json()["corrected_text"] == "Corrected invoice"


def test_classify_route_smoke(client, monkeypatch):
    monkeypatch.setattr(
        classify_route.ClassificationService,
        "classify",
        staticmethod(
            lambda text: {
                "document_type": "Invoice",
                "confidence": 0.9
            }
        )
    )

    response = client.post(
        "/classify",
        params={
            "text": "invoice"
        }
    )

    assert response.status_code == 200
    assert response.json()["document_type"] == "Invoice"


def test_classify_route_returns_model_loading_failure(client, monkeypatch):
    def fail_model_load(text):
        raise ModelLoadingError(
            "MODEL_NAME is not configured."
        )

    monkeypatch.setattr(
        classify_route.ClassificationService,
        "classify",
        staticmethod(
            fail_model_load
        )
    )

    response = client.post(
        "/classify",
        params={
            "text": "invoice"
        }
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Model loading failed: MODEL_NAME is not configured."
    )


def test_extract_route_smoke(client, monkeypatch):
    monkeypatch.setattr(
        extract_route.ExtractionService,
        "extract_data",
        staticmethod(
            lambda text: {
                "document_title": "Invoice",
                "dynamic_fields": {}
            }
        )
    )

    response = client.post(
        "/extract",
        json={
            "text": "invoice"
        }
    )

    assert response.status_code == 200
    assert response.json()["document_title"] == "Invoice"


def test_validate_route_smoke(client):
    response = client.post(
        "/validate",
        json={
            "email": "person@example.com",
            "phone": "9876543210",
            "date": "24/06/2026",
            "pan": "ABCDE1234F",
            "aadhaar": "123456789012",
            "gst": "27ABCDE1234F1Z5"
        }
    )

    assert response.status_code == 200
    assert response.json()["email"] is True


def test_qa_route_smoke(client, monkeypatch):
    monkeypatch.setattr(
        qa_route.QAService,
        "answer_question",
        staticmethod(
            lambda document_text, question: "The total is Rs. 1200."
        )
    )

    response = client.post(
        "/qa",
        params={
            "document_text": "Total Rs. 1200",
            "question": "What is the total?"
        }
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "The total is Rs. 1200."
