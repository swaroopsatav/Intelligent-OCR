from backend.services.validation_service import ValidationService


def test_validate_document_marks_valid_indian_document_fields():
    result = ValidationService.validate_document(
        {
            "name": "Swaroop Satav",
            "email": "person@example.com",
            "phone": "9876543210",
            "date": "24/06/2026",
            "pan": "ABCDE1234F",
            "aadhaar": "123456789012",
            "gst": "27ABCDE1234F1Z5",
            "total_amount": "Rs. 1,200.50",
            "dynamic_fields": {
                "subtotal": "1000",
                "tax_amount": "200.50",
                "due_date": "24/06/2026"
            }
        }
    )

    assert result == {
        "name": True,
        "email": True,
        "phone": True,
        "date": True,
        "pan": True,
        "aadhaar": True,
        "gst": True,
        "total_amount": True,
        "dynamic_fields.subtotal": True,
        "dynamic_fields.tax_amount": True,
        "dynamic_fields.due_date": True
    }


def test_validate_document_marks_invalid_or_missing_fields():
    result = ValidationService.validate_document(
        {
            "name": "12345",
            "email": "not-an-email",
            "phone": "12345",
            "date": "2026/24/06",
            "pan": "BADPAN",
            "gst": "bad-gst",
            "total_amount": "twelve hundred",
            "dynamic_fields": {
                "subtotal": "bad",
                "due_date": "tomorrow"
            }
        }
    )

    assert result == {
        "name": False,
        "email": False,
        "phone": False,
        "date": False,
        "pan": False,
        "gst": False,
        "total_amount": False,
        "dynamic_fields.subtotal": False,
        "dynamic_fields.due_date": False
    }


def test_validate_date_accepts_supported_formats():
    assert ValidationService.validate_date("24-06-2026")
    assert ValidationService.validate_date("24/06/2026")
    assert ValidationService.validate_date("2026-06-24")


def test_validate_amount_accepts_common_currency_formats():
    assert ValidationService.validate_amount("Rs. 12,450.00")
    assert ValidationService.validate_amount("INR 12450")
    assert ValidationService.validate_amount("$99.99")
    assert not ValidationService.validate_amount("-99")
    assert not ValidationService.validate_amount("12.999")


def test_validate_document_skips_missing_optional_fields():
    result = ValidationService.validate_document(
        {
            "document_type": "Resume",
            "name": "Swaroop Satav",
            "email": "person@example.com"
        }
    )

    assert result == {
        "name": True,
        "email": True
    }