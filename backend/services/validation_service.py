import re
from datetime import datetime
from decimal import Decimal, InvalidOperation


class ValidationService:

    OPTIONAL_FIELDS = [
        "name",
        "email",
        "phone",
        "date",
        "pan",
        "aadhaar",
        "gst",
        "total_amount"
    ]

    @staticmethod
    def validate_name(name):

        if not name:
            return False

        value = str(
            name
        ).strip()

        if len(value) < 2 or len(value) > 120:
            return False

        if not re.search(
            r"[A-Za-z]",
            value
        ):
            return False

        pattern = r"^[A-Za-z][A-Za-z .'-]*$"

        return bool(
            re.match(pattern, value)
        )

    @staticmethod
    def validate_email(email):

        if not email:
            return False

        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

        return bool(
            re.match(pattern, email)
        )

    @staticmethod
    def validate_phone(phone):

        if not phone:
            return False

        pattern = r'^[6-9]\d{9}$'

        return bool(
            re.match(pattern, phone)
        )

    @staticmethod
    def validate_pan(pan):

        if not pan:
            return False

        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'

        return bool(
            re.match(pattern, pan)
        )

    @staticmethod
    def validate_aadhaar(aadhaar):

        if not aadhaar:
            return False

        pattern = r'^\d{12}$'

        return bool(
            re.match(pattern, aadhaar)
        )

    @staticmethod
    def validate_gst(gst):

        if not gst:
            return False

        pattern = (
            r'^[0-9]{2}'
            r'[A-Z]{5}'
            r'[0-9]{4}'
            r'[A-Z]{1}'
            r'[A-Z0-9]{3}$'
        )

        return bool(
            re.match(pattern, gst)
        )

    @staticmethod
    def validate_amount(amount):

        if amount is None:
            return False

        amount_text = str(
            amount
        ).strip()

        if not amount_text:
            return False

        normalized = re.sub(
            r"(?i)(rs\.?|inr|\u20b9|\$|,|\s)",
            "",
            amount_text
        )

        normalized = normalized.lstrip(
            ":"
        )

        if not re.match(
            r"^\d+(\.\d{1,2})?$",
            normalized
        ):
            return False

        try:
            return Decimal(
                normalized
            ) >= 0
        except InvalidOperation:
            return False

    @staticmethod
    def validate_date(date_value):

        if not date_value:
            return False

        formats = [
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d"
        ]

        for fmt in formats:

            try:

                datetime.strptime(
                    date_value,
                    fmt
                )

                return True

            except ValueError:
                continue

        return False

    @classmethod
    def validate_dynamic_fields(
        cls,
        dynamic_fields
    ):

        if not isinstance(
            dynamic_fields,
            dict
        ):
            return {}

        validators = {
            "subtotal": cls.validate_amount,
            "tax_amount": cls.validate_amount,
            "opening_balance": cls.validate_amount,
            "closing_balance": cls.validate_amount,
            "due_date": cls.validate_date,
            "date_of_birth": cls.validate_date,
            "expiry_date": cls.validate_date,
            "issue_date": cls.validate_date,
            "statement_period": lambda value: bool(
                str(value).strip()
            ),
            "account_number": lambda value: bool(
                re.match(
                    r"^[0-9Xx* -]{6,24}$",
                    str(value).strip()
                )
            )
        }

        results = {}

        for field, value in dynamic_fields.items():

            if field in validators:
                results[
                    f"dynamic_fields.{field}"
                ] = validators[field](
                    value
                )

        return results

    @classmethod
    def validate_document(
        cls,
        data
    ):

        validators = {
            "name": cls.validate_name,
            "email": cls.validate_email,
            "phone": cls.validate_phone,
            "date": cls.validate_date,
            "pan": cls.validate_pan,
            "aadhaar": cls.validate_aadhaar,
            "gst": cls.validate_gst,
            "total_amount": cls.validate_amount
        }

        results = {}

        for field in cls.OPTIONAL_FIELDS:

            value = data.get(
                field,
                ""
            )

            if value in (None, ""):
                continue

            results[field] = validators[field](
                value
            )

        results.update(
            cls.validate_dynamic_fields(
                data.get(
                    "dynamic_fields",
                    {}
                )
            )
        )

        return results