import json
import logging
import re

from backend.services.llm_service import (
    LLMService
)

logger = logging.getLogger(__name__)


EXTRACTION_FIELDS = [
    "document_title",
    "document_type",
    "name",
    "company_name",
    "document_id",
    "date",
    "address",
    "phone",
    "email",
    "invoice_number",
    "total_amount",
    "pan",
    "gst",
    "aadhaar",
    "dynamic_fields"
]


class ExtractionService:

    @staticmethod
    def extract_data(text):

        prompt = f"""
Extract information from the document.

Return ONLY valid JSON.

JSON Schema:

{{
    "document_title": "",
    "document_type": "",
    "name": "",
    "company_name": "",
    "document_id": "",
    "date": "",
    "address": "",
    "phone": "",
    "email": "",
    "invoice_number": "",
    "total_amount": "",
    "pan": "",
    "gst": "",
    "aadhaar": "",
    "dynamic_fields": {{}}
}}

Rules:
- Return JSON only.
- Do not add explanations.
- Do not add markdown.
- Do not add code fences.
- Use empty string if a value cannot be found.
- Preserve values exactly as they appear.
- document_id should contain the most relevant ID number for identity documents, bank statements, receipts, invoices, resumes, or other document types.
- dynamic_fields must contain extra fields relevant to the detected document type.
- For invoices/receipts, dynamic_fields may include vendor_name, buyer_name, subtotal, tax_amount, due_date, payment_status, line_items.
- If table-like rows are present in the document, include them in dynamic_fields.line_items when they represent invoice, receipt, or statement rows.
- For resumes, dynamic_fields may include skills, education, experience, current_role, links.
- For identity documents, dynamic_fields may include date_of_birth, gender, nationality, expiry_date, issue_date.
- For bank statements, dynamic_fields may include account_number, bank_name, statement_period, opening_balance, closing_balance.

Document:

{text}

JSON:
"""

        result = LLMService.generate(
            prompt,
            max_new_tokens=768
        )

        print(
            "\n===== RAW EXTRACTION OUTPUT =====\n"
        )

        print(result)

        print(
            "\n=================================\n"
        )

        try:

            start = result.find("{")

            end = (
                result.rfind("}")
                + 1
            )

            if (
                start == -1
                or end <= start
            ):
                raise ValueError(
                    "No JSON object found in response"
                )

            json_text = result[
                start:end
            ]

            extracted_data = json.loads(
                json_text
            )

            response = {}

            for field in EXTRACTION_FIELDS:

                if field == "dynamic_fields":

                    dynamic_fields = extracted_data.get(
                        field,
                        {}
                    )

                    response[field] = (
                        dynamic_fields
                        if isinstance(
                            dynamic_fields,
                            dict
                        )
                        else {}
                    )

                    continue

                response[field] = extracted_data.get(
                    field,
                    ""
                )

            return ExtractionService.post_process_extraction(
                response,
                text
            )

        except Exception as e:

            logger.exception(
                "Information extraction failed"
            )

            response = ExtractionService.fallback_extract_data(
                text
            )

            response["error"] = str(e)

            return ExtractionService.post_process_extraction(
                response,
                text
            )

    @staticmethod
    def match_value(
        text,
        pattern
    ):

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        return (
            match.group(1).strip()
            if match
            else ""
        )

    @staticmethod
    def fallback_extract_data(
        text
    ):

        normalized_text = text or ""

        document_title = (
            normalized_text.strip().splitlines()[0].strip()
            if normalized_text.strip()
            else ""
        )

        invoice_number = ExtractionService.match_value(
            normalized_text,
            r"invoice\s*(?:no|number)?\s*[:#-]\s*([A-Z0-9/-]+)"
        )

        total_amount = ExtractionService.match_value(
            normalized_text,
            r"total\s*amount\s*[:#-]\s*([^\n\r]+)"
        )

        name = ExtractionService.match_value(
            normalized_text,
            r"(?:bill\s*to|name)\s*[:#-]\s*([^\n\r]+)"
        )

        company_name = ExtractionService.match_value(
            normalized_text,
            r"company\s*[:#-]\s*([^\n\r]+)"
        )

        date = ExtractionService.match_value(
            normalized_text,
            r"date\s*[:#-]\s*([0-9]{1,4}[/-][0-9]{1,2}[/-][0-9]{2,4})"
        )

        email = ExtractionService.match_value(
            normalized_text,
            r"([\w\.-]+@[\w\.-]+\.\w+)"
        )

        phone = ExtractionService.match_value(
            normalized_text,
            r"\b([6-9]\d{9})\b"
        )

        pan = ExtractionService.match_value(
            normalized_text,
            r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
        )

        gst = ExtractionService.match_value(
            normalized_text,
            r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]{3})\b"
        )

        aadhaar = ExtractionService.match_value(
            normalized_text,
            r"\b(\d{12})\b"
        )

        payment_status = ExtractionService.match_value(
            normalized_text,
            r"payment\s*status\s*[:#-]\s*([^\n\r]+)"
        )

        dynamic_fields = {}

        if payment_status:
            dynamic_fields["payment_status"] = payment_status

        line_items = ExtractionService.extract_invoice_line_items(
            normalized_text
        )

        if line_items:
            dynamic_fields["line_items"] = line_items

            subtotal = sum(
                item.get(
                    "line_total_value",
                    0
                )
                for item in line_items
            )

            if subtotal:
                dynamic_fields["subtotal"] = f"{subtotal:.2f}"

        return {
            "document_title": document_title,
            "document_type": (
                "Invoice"
                if invoice_number or total_amount
                else ""
            ),
            "name": name,
            "company_name": company_name,
            "document_id": invoice_number,
            "date": date,
            "address": "",
            "phone": phone,
            "email": email,
            "invoice_number": invoice_number,
            "total_amount": total_amount,
            "pan": pan,
            "gst": gst,
            "aadhaar": aadhaar,
            "dynamic_fields": dynamic_fields
        }

    @staticmethod
    def normalize_gst(
        gst
    ):

        if not gst:
            return ""

        value = re.sub(
            r"[^A-Z0-9]",
            "",
            str(gst).upper()
        )

        if len(value) != 15:
            return value

        chars = list(value)

        digit_positions = set(
            list(range(0, 2))
            + list(range(7, 11))
            + [14]
        )

        letter_positions = set(
            list(range(2, 7))
            + [11, 13]
        )

        digit_fixes = {
            "O": "0",
            "Q": "0",
            "I": "1",
            "L": "1",
            "S": "5",
            "Z": "2",
            "B": "8"
        }

        letter_fixes = {
            "0": "O",
            "1": "I",
            "5": "S",
            "2": "Z",
            "8": "B"
        }

        for index, char in enumerate(chars):

            if index in digit_positions:
                chars[index] = digit_fixes.get(
                    char,
                    char
                )

            if index in letter_positions:
                chars[index] = letter_fixes.get(
                    char,
                    char
                )

        return "".join(
            chars
        )

    @staticmethod
    def parse_amount_value(
        amount
    ):

        if amount is None:
            return None

        normalized = re.sub(
            r"(?i)(rs\.?|inr|\u20b9|\$|,|\s)",
            "",
            str(amount)
        ).lstrip(":")

        if not re.match(
            r"^\d+(\.\d{1,2})?$",
            normalized
        ):
            return None

        return float(
            normalized
        )

    @staticmethod
    def extract_invoice_line_items(
        text
    ):

        line_items = []

        for line in text.splitlines():

            match = re.match(
                r"^\s*(.+?)\s+(\d+(?:\.\d+)?)\s+(?:x\s+)?(?:Rs\.?|INR|\u20b9|\$)?\s*:?\s*([0-9,]+(?:\.\d{1,2})?)\s*$",
                line,
                re.IGNORECASE
            )

            if not match:
                continue

            description = match.group(1).strip()

            if description.lower() in {
                "item",
                "description",
                "total",
                "subtotal"
            }:
                continue

            quantity = float(
                match.group(2)
            )

            line_total = ExtractionService.parse_amount_value(
                match.group(3)
            )

            if line_total is None:
                continue

            unit_price = (
                line_total / quantity
                if quantity
                else line_total
            )

            line_items.append(
                {
                    "description": description,
                    "quantity": quantity,
                    "unit_price": f"{unit_price:.2f}",
                    "line_total": f"{line_total:.2f}",
                    "line_total_value": line_total
                }
            )

        return line_items

    @staticmethod
    def post_process_extraction(
        data,
        source_text
    ):

        processed = dict(
            data
        )

        processed["gst"] = ExtractionService.normalize_gst(
            processed.get(
                "gst",
                ""
            )
        )

        dynamic_fields = processed.get(
            "dynamic_fields",
            {}
        )

        if not isinstance(
            dynamic_fields,
            dict
        ):
            dynamic_fields = {}

        line_items = dynamic_fields.get(
            "line_items"
        )

        if not isinstance(
            line_items,
            list
        ) or not line_items:
            line_items = ExtractionService.extract_invoice_line_items(
                source_text
            )

        if line_items:

            cleaned_items = []

            for item in line_items:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                cleaned = dict(
                    item
                )

                if "line_total_value" not in cleaned:
                    amount_source = (
                        cleaned.get(
                            "line_total"
                        )
                        or cleaned.get(
                            "amount"
                        )
                        or cleaned.get(
                            "unit_price"
                        )
                    )

                    amount_value = ExtractionService.parse_amount_value(
                        amount_source
                    )

                    if amount_value is not None:
                        cleaned["line_total_value"] = amount_value

                cleaned_items.append(
                    cleaned
                )

            dynamic_fields["line_items"] = cleaned_items

            subtotal = sum(
                float(
                    item.get(
                        "line_total_value",
                        0
                    )
                    or 0
                )
                for item in cleaned_items
            )

            if subtotal:
                dynamic_fields.setdefault(
                    "subtotal",
                    f"{subtotal:.2f}"
                )

                total_value = ExtractionService.parse_amount_value(
                    processed.get(
                        "total_amount",
                        ""
                    )
                )

                if total_value is not None:
                    dynamic_fields[
                        "total_reconciliation"
                    ] = {
                        "line_items_total": f"{subtotal:.2f}",
                        "document_total": f"{total_value:.2f}",
                        "matches": abs(
                            subtotal
                            - total_value
                        ) < 0.01
                    }

        processed["dynamic_fields"] = dynamic_fields

        return processed
