import json
import re

from backend.services.llm_service import (
    LLMService
)


class ClassificationService:

    ALLOWED_TYPES = {
        "Invoice",
        "Receipt",
        "Resume",
        "Aadhaar Card",
        "PAN Card",
        "Driving License",
        "Passport",
        "Bank Statement",
        "Other"
    }

    KEYWORD_RULES = [
        ("Invoice", ["invoice", "invoice no", "bill to", "total amount"]),
        ("Receipt", ["receipt", "payment received", "paid on"]),
        ("Resume", ["resume", "curriculum vitae", "skills", "experience"]),
        ("Aadhaar Card", ["aadhaar", "uidai", "government of india"]),
        ("PAN Card", ["permanent account number", "income tax", "pan"]),
        ("Driving License", ["driving licence", "driving license", "dl no"]),
        ("Passport", ["passport", "republic of india", "surname"]),
        ("Bank Statement", ["bank statement", "account number", "opening balance"])
    ]

    @classmethod
    def heuristic_classify(cls, text):

        normalized_text = re.sub(
            r"\s+",
            " ",
            text.lower()
        )

        best_type = "Other"
        best_score = 0

        for document_type, keywords in cls.KEYWORD_RULES:

            score = sum(
                1
                for keyword in keywords
                if keyword in normalized_text
            )

            if score > best_score:
                best_type = document_type
                best_score = score

        if best_score == 0:
            return {
                "document_type": "Other",
                "confidence": 0.2,
                "method": "heuristic"
            }

        confidence = min(
            0.95,
            0.45 + (best_score * 0.15)
        )

        return {
            "document_type": best_type,
            "confidence": round(
                confidence,
                2
            ),
            "method": "heuristic"
        }

    @classmethod
    def normalize_result(
        cls,
        document_type,
        confidence,
        fallback
    ):

        if document_type not in cls.ALLOWED_TYPES:
            return fallback

        confidence = max(
            0,
            min(
                float(confidence or 0),
                1
            )
        )

        if (
            fallback["document_type"] != "Other"
            and document_type == fallback["document_type"]
        ):
            confidence = max(
                confidence,
                fallback["confidence"]
            )

        if confidence <= 0 and document_type != "Other":
            confidence = fallback["confidence"]

        return {
            "document_type": document_type,
            "confidence": float(round(
                confidence,
                2
            )),
            "method": "llm"
        }

    @staticmethod
    def classify(text):

        fallback = ClassificationService.heuristic_classify(
            text
        )

        prompt = f"""
Classify document.

Return ONLY valid JSON in this exact schema:

{{
    "document_type": "",
    "confidence": 0.0
}}

Allowed document_type values:

- Invoice
- Receipt
- Resume
- Aadhaar Card
- PAN Card
- Driving License
- Passport
- Bank Statement
- Other

Rules:
- confidence must be a number between 0 and 1.
- Use "Other" if the document category is unclear.
- Do not add markdown.
- Do not add explanations.

Document:

{text}

JSON:
"""

        result = LLMService.generate(
            prompt
        )

        try:
            start = result.find("{")
            end = result.rfind("}") + 1

            if start == -1 or end <= start:
                raise ValueError(
                    "No JSON object found in response"
                )

            parsed = json.loads(
                result[start:end]
            )

            document_type = (
                parsed.get(
                    "document_type",
                    "Other"
                )
                or "Other"
            )

            confidence = float(
                parsed.get(
                    "confidence",
                    0
                )
                or 0
            )

            return ClassificationService.normalize_result(
                document_type,
                confidence,
                fallback
            )

        except Exception:
            return fallback
