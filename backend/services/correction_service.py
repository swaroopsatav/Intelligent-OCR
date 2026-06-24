import re
import time
import logging

logger = logging.getLogger(__name__)


class CorrectionService:

    @staticmethod
    def normalize_text(
        text
    ):

        # Normalize line endings
        text = text.replace(
            "\r\n",
            "\n"
        )

        # Remove trailing spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Remove excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        # Remove extra spaces before punctuation
        text = re.sub(
            r"\s+([.,;:])",
            r"\1",
            text
        )

        # Fix common email formatting
        text = re.sub(
            r"\s*@\s*",
            "@",
            text
        )

        text = re.sub(
            r"\s*\.\s*",
            ".",
            text
        )

        return text.strip()

    @classmethod
    def correct_text(
        cls,
        ocr_text
    ):

        normalized_text = cls.normalize_text(
            ocr_text or ""
        )

        try:

            start_time = time.time()

            prompt = f"""
Correct OCR errors in the following document text.

Return ONLY the corrected document text.
Do not summarize.
Do not add explanations.
Do not add markdown.
Preserve the original meaning, names, numbers, dates, addresses, emails, phone numbers, and line order as much as possible.
Fix obvious OCR spelling, spacing, punctuation, and formatting mistakes.

OCR TEXT:

{normalized_text}

CORRECTED TEXT:
"""

            from backend.services.llm_service import LLMService

            llm_output = LLMService.generate(
                prompt
            )

            corrected_text = cls.normalize_text(
                llm_output
            )

            if not corrected_text:
                raise ValueError(
                    "LLM returned empty corrected text"
                )

            processing_time = round(
                time.time() - start_time,
                2
            )

            logger.info(
                f"OCR correction completed in "
                f"{processing_time} sec"
            )

            return {
                "corrected_text":
                    corrected_text,
                "method":
                    "llm",
                "processing_time":
                    processing_time,
                "status":
                    "success"
            }

        except Exception as e:

            logger.warning(
                "LLM OCR correction unavailable; using regex fallback: %s",
                e
            )

            fallback_text = normalized_text

            processing_time = round(
                time.time() - start_time,
                2
            )

            logger.info(
                "OCR correction completed with fallback"
            )

            return {
                "corrected_text":
                    fallback_text,
                "method":
                    "regex_fallback",
                "processing_time":
                    processing_time,
                "status": "success",
                "warning":
                    f"LLM correction unavailable: {e}"
            }
