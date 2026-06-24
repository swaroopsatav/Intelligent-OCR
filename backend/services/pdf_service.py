import os
import logging

import fitz

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

from backend.config.settings import (
    PDF_RENDER_SCALE
)


class PDFService:

    @staticmethod
    def pdf_to_images(
        pdf_path: str
    ):

        pdf_path = (
            pdf_path
            .strip()
            .strip('"')
            .replace("\\", "/")
        )

        if not os.path.exists(
            pdf_path
        ):

            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        os.makedirs(
            PROCESSED_DIR,
            exist_ok=True
        )

        logger.info(
            f"Processing PDF: {pdf_path}"
        )

        document = fitz.open(
            pdf_path
        )

        image_paths = []

        for page_index in range(
            len(document)
        ):

            page = document[
                page_index
            ]

            matrix = fitz.Matrix(
                PDF_RENDER_SCALE,
                PDF_RENDER_SCALE
            )

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            image_path = os.path.join(
                PROCESSED_DIR,
                f"page_{page_index + 1}.png"
            )

            pix.save(
                image_path
            )

            image_paths.append(
                image_path.replace(
                    "\\",
                    "/"
                )
            )

        document.close()

        logger.info(
            f"Generated {len(image_paths)} image(s)"
        )

        return image_paths