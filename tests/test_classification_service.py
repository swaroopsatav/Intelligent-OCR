from backend.services import classification_service
from backend.services.classification_service import ClassificationService


def test_classify_uses_heuristic_when_llm_returns_invalid_json(monkeypatch):
    monkeypatch.setattr(
        classification_service.LLMService,
        "generate",
        staticmethod(lambda prompt: "Invoice")
    )

    result = ClassificationService.classify(
        "INVOICE Invoice No INV-001 Bill To Customer Total Amount 1200"
    )

    assert result["document_type"] == "Invoice"
    assert result["confidence"] >= 0.75
    assert result["method"] == "heuristic"


def test_classify_rejects_unsupported_llm_document_type(monkeypatch):
    monkeypatch.setattr(
        classification_service.LLMService,
        "generate",
        staticmethod(
            lambda prompt: '{"document_type": "Contract", "confidence": 0.99}'
        )
    )

    result = ClassificationService.classify(
        "Resume Skills Experience Education"
    )

    assert result["document_type"] == "Resume"
    assert result["method"] == "heuristic"


def test_classify_preserves_valid_llm_result(monkeypatch):
    monkeypatch.setattr(
        classification_service.LLMService,
        "generate",
        staticmethod(
            lambda prompt: '{"document_type": "Bank Statement", "confidence": 0.84}'
        )
    )

    result = ClassificationService.classify(
        "Monthly statement"
    )

    assert result == {
        "document_type": "Bank Statement",
        "confidence": 0.84,
        "method": "llm"
    }
