from backend.services.ocr_service import OCRService


def test_extract_table_rows_groups_boxes_by_vertical_position():
    boxes = [
        {
            "text": "Item",
            "confidence": 0.95,
            "box": [[10, 10], [40, 10], [40, 20], [10, 20]]
        },
        {
            "text": "Amount",
            "confidence": 0.92,
            "box": [[120, 12], [180, 12], [180, 22], [120, 22]]
        },
        {
            "text": "Total",
            "confidence": 0.9,
            "box": [[10, 50], [60, 50], [60, 60], [10, 60]]
        }
    ]

    rows = OCRService.extract_table_rows(
        boxes,
        y_tolerance=8
    )

    assert rows[0]["text"] == "Item | Amount"
    assert rows[0]["is_table_like"] is True
    assert rows[0]["cells"][0]["column_index"] == 1
    assert rows[0]["cells"][1]["column_index"] == 2
    assert rows[1]["text"] == "Total"
    assert rows[1]["is_table_like"] is False


def test_extract_tables_returns_structured_records():
    rows = [
        {
            "row_index": 1,
            "is_table_like": True,
            "cells": [
                {
                    "text": "Item",
                    "confidence": 0.98,
                    "box": [[10, 10], [50, 10], [50, 20], [10, 20]]
                },
                {
                    "text": "Amount",
                    "confidence": 0.96,
                    "box": [[120, 10], [180, 10], [180, 20], [120, 20]]
                }
            ]
        },
        {
            "row_index": 2,
            "is_table_like": True,
            "cells": [
                {
                    "text": "Consulting",
                    "confidence": 0.91,
                    "box": [[11, 40], [80, 40], [80, 50], [11, 50]]
                },
                {
                    "text": "1200",
                    "confidence": 0.93,
                    "box": [[122, 41], [160, 41], [160, 51], [122, 51]]
                }
            ]
        }
    ]

    tables = OCRService.extract_tables(
        rows,
        x_tolerance=10
    )

    assert tables[0]["headers"] == [
        "item",
        "amount"
    ]
    assert tables[0]["records"][1] == {
        "item": "Consulting",
        "amount": "1200"
    }
    assert tables[0]["rows"][1]["confidence"] == 0.92


def test_extract_text_returns_bounding_boxes_and_table_rows(monkeypatch, tmp_path):
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"fake image")

    class FakeReader:
        def readtext(self, path):
            return [
                (
                    [[0, 0], [40, 0], [40, 10], [0, 10]],
                    "Invoice",
                    0.9
                ),
                (
                    [[80, 1], [120, 1], [120, 11], [80, 11]],
                    "1200",
                    0.8
                )
            ]

    monkeypatch.setattr(
        OCRService,
        "get_reader",
        classmethod(lambda cls: FakeReader())
    )

    monkeypatch.setattr(
        OCRService,
        "get_cache_file",
        classmethod(lambda cls, path: str(tmp_path / "cache.json"))
    )

    result = OCRService.extract_text(
        str(image_path)
    )

    assert result["text"] == "Invoice\n1200"
    assert result["confidence"] == 0.85
    assert result["bounding_boxes"][0]["text"] == "Invoice"
    assert result["table_rows"][0]["text"] == "Invoice | 1200"
    assert result["tables"][0]["records"][0] == {
        "column_1": "Invoice",
        "column_2": "1200"
    }
    assert result["key_value_pairs"] == {}


def test_extract_key_value_pairs_from_colon_lines_and_two_cell_rows():
    pairs = OCRService.extract_key_value_pairs(
        [
            "Invoice No: INV-001",
            "Total Amount: Rs. 1200"
        ],
        [
            {
                "cells": [
                    {
                        "text": "Payment Status"
                    },
                    {
                        "text": "Paid"
                    }
                ]
            }
        ]
    )

    assert pairs == {
        "invoice_no": "INV-001",
        "total_amount": "Rs. 1200",
        "payment_status": "Paid"
    }
