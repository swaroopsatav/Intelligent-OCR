import os
import json
import time
import hashlib
import logging
import warnings

import cv2
from easyocr import Reader

from backend.config.settings import (
    HF_TOKEN,
    OCR_CACHE_ENABLED,
    OCR_LANGUAGES
)

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

logger = logging.getLogger(__name__)

CACHE_DIR = "data/cache"

os.makedirs(
    CACHE_DIR,
    exist_ok=True
)


class OCRService:

    _reader = None

    @classmethod
    def get_reader(cls):

        if cls._reader is None:

            logger.info(
                "Loading EasyOCR model..."
            )

            cls._reader = Reader(
                OCR_LANGUAGES,
                gpu=False
            )

        return cls._reader

    @classmethod
    def get_cache_file(
        cls,
        image_path
    ):

        os.makedirs(
            CACHE_DIR,
            exist_ok=True
        )

        cache_key = hashlib.md5(
            image_path.encode()
        ).hexdigest()

        return os.path.join(
            CACHE_DIR,
            f"{cache_key}.json"
        )

    @classmethod
    def extract_text(
        cls,
        image_path
    ):

        try:

            image_path = (
                image_path
                .strip()
                .strip('"')
                .replace("\\", "/")
            )

            cache_file = (
                cls.get_cache_file(
                    image_path
                )
            )

            if (
                OCR_CACHE_ENABLED
                and os.path.exists(
                    cache_file
                )
            ):

                logger.info(
                    "Using cached OCR result"
                )

                with open(
                    cache_file,
                    "r",
                    encoding="utf-8"
                ) as f:

                    cached_response = json.load(f)

                cached_response.setdefault(
                    "bounding_boxes",
                    []
                )

                cached_response.setdefault(
                    "table_rows",
                    []
                )

                cached_response.setdefault(
                    "overlay_image",
                    ""
                )

                return cached_response

            start_time = time.time()

            reader = (
                cls.get_reader()
            )

            results = reader.readtext(
                image_path
            )

            text_lines = []

            confidence_scores = []

            bounding_boxes = []

            for result in results:

                text_lines.append(
                    result[1]
                )

                confidence_scores.append(
                    float(result[2])
                )

                bounding_boxes.append(
                    {
                        "text": result[1],
                        "confidence": round(
                            float(result[2]),
                            4
                        ),
                        "box": [
                            [
                                float(point[0]),
                                float(point[1])
                            ]
                            for point in result[0]
                        ]
                    }
                )

            average_confidence = (
                sum(confidence_scores)
                / len(confidence_scores)
                if confidence_scores
                else 0
            )

            table_rows = cls.extract_table_rows(
                bounding_boxes
            )

            key_value_pairs = cls.extract_key_value_pairs(
                text_lines,
                table_rows
            )

            overlay_image = cls.create_overlay_image(
                image_path,
                bounding_boxes
            )

            processing_time = round(
                time.time() - start_time,
                2
            )

            response = {
                "text": "\n".join(
                    text_lines
                ),
                "confidence": round(
                    average_confidence,
                    4
                ),
                "line_count": len(
                    text_lines
                ),
                "bounding_boxes":
                    bounding_boxes,
                "table_rows":
                    table_rows,
                "key_value_pairs":
                    key_value_pairs,
                "overlay_image":
                    overlay_image,
                "processing_time":
                    processing_time,
                "status":
                    "success"
            }

            if OCR_CACHE_ENABLED:

                with open(
                    cache_file,
                    "w",
                    encoding="utf-8"
                ) as f:

                    json.dump(
                        response,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )

            logger.info(
                f"OCR completed in "
                f"{processing_time} sec"
            )

            return response

        except Exception as e:

            logger.exception(
                "OCR extraction failed"
            )

            return {
                "text": "",
                "confidence": 0,
                "line_count": 0,
                "bounding_boxes": [],
                "table_rows": [],
                "key_value_pairs": {},
                "overlay_image": "",
                "processing_time": 0,
                "status": "error",
                "message": str(e)
            }

    @staticmethod
    def extract_table_rows(
        bounding_boxes,
        y_tolerance=14
    ):

        if not bounding_boxes:
            return []

        row_groups = []

        for item in sorted(
            bounding_boxes,
            key=lambda box: (
                min(point[1] for point in box["box"]),
                min(point[0] for point in box["box"])
            )
        ):

            y_center = sum(
                point[1]
                for point in item["box"]
            ) / len(item["box"])

            matched_row = None

            for row in row_groups:

                if abs(row["y_center"] - y_center) <= y_tolerance:
                    matched_row = row
                    break

            if matched_row is None:

                matched_row = {
                    "y_center": y_center,
                    "cells": []
                }

                row_groups.append(
                    matched_row
                )

            matched_row["cells"].append(
                item
            )

            matched_row["y_center"] = (
                matched_row["y_center"]
                + y_center
            ) / 2

        table_rows = []

        for row_index, row in enumerate(
            row_groups,
            start=1
        ):

            cells = sorted(
                row["cells"],
                key=lambda cell: min(
                    point[0]
                    for point in cell["box"]
                )
            )

            table_rows.append(
                {
                    "row_index": row_index,
                    "column_count": len(
                        cells
                    ),
                    "text": " | ".join(
                        cell["text"]
                        for cell in cells
                    ),
                    "cells": [
                        {
                            "text": cell["text"],
                            "confidence": cell["confidence"],
                            "box": cell["box"]
                        }
                        for cell in cells
                    ]
                }
            )

        return table_rows

    @staticmethod
    def extract_key_value_pairs(
        text_lines,
        table_rows
    ):

        pairs = {}
        label_keywords = {
            "invoice no",
            "invoice number",
            "receipt no",
            "receipt number",
            "date",
            "name",
            "email",
            "phone",
            "gst",
            "pan",
            "aadhaar",
            "total",
            "total amount",
            "payment status",
            "account number",
            "opening balance",
            "closing balance",
            "statement period"
        }

        for line in text_lines:

            if ":" in line:
                key, value = line.split(
                    ":",
                    1
                )

                key = key.strip().lower().replace(
                    " ",
                    "_"
                )

                value = value.strip()

                if key and value:
                    pairs[key] = value

        for row in table_rows:

            cells = row.get(
                "cells",
                []
            )

            if len(cells) == 2:
                raw_key = cells[0]["text"].strip().lower()

                if raw_key not in label_keywords:
                    continue

                key = raw_key.replace(
                    " ",
                    "_"
                )

                value = cells[1]["text"].strip()

                if key and value:
                    pairs.setdefault(
                        key,
                        value
                    )

        return pairs

    @staticmethod
    def create_overlay_image(
        image_path,
        bounding_boxes
    ):

        if not bounding_boxes:
            return ""

        image = cv2.imread(
            image_path
        )

        if image is None:
            return ""

        for item in bounding_boxes:

            points = item.get(
                "box",
                []
            )

            if len(points) < 4:
                continue

            contour = [
                [
                    int(point[0]),
                    int(point[1])
                ]
                for point in points
            ]

            for index in range(
                len(contour)
            ):

                start = contour[index]
                end = contour[
                    (index + 1)
                    % len(contour)
                ]

                cv2.line(
                    image,
                    tuple(start),
                    tuple(end),
                    (0, 180, 0),
                    2
                )

            x = min(
                point[0]
                for point in contour
            )

            y = max(
                15,
                min(
                    point[1]
                    for point in contour
                )
                - 5
            )

            cv2.putText(
                image,
                f"{item.get('confidence', 0):.2f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 120, 255),
                1,
                cv2.LINE_AA
            )

        overlay_path = os.path.join(
            os.path.dirname(
                image_path
            ),
            f"overlay_{os.path.basename(image_path)}"
        )

        cv2.imwrite(
            overlay_path,
            image
        )

        return overlay_path.replace(
            "\\",
            "/"
        )
