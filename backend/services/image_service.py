import os
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DIR = "data/processed"

from backend.config.settings import (
    ENABLE_BLUR,
    ENABLE_DESKEW,
    ENABLE_ENHANCEMENT,
    ENABLE_THRESHOLD
)

class ImageService:

    @staticmethod
    def deskew_image(
        image
    ):

        thresholded = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(
            np.where(thresholded > 0)
        )

        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(
            coords
        )[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return image

        height, width = image.shape[:2]

        center = (
            width // 2,
            height // 2
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    @staticmethod
    def enhance_image(
        image
    ):

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(
            image
        )

    @staticmethod
    def preprocess_image(
        image_path
    ):

        image_path = (
            image_path
            .strip()
            .strip('"')
            .replace("\\", "/")
        )

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        logger.info(
            f"Preprocessing: {image_path}"
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                f"Unable to load image: {image_path}"
            )

        # --------------------------
        # Convert to Grayscale
        # --------------------------

        processed = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        applied_steps = [
            "grayscale"
        ]

        # --------------------------
        # Noise Removal
        # --------------------------

        if ENABLE_BLUR:

            processed = cv2.GaussianBlur(
                processed,
                (5, 5),
                0
            )

            applied_steps.append(
                "noise_removal"
            )

        # --------------------------
        # Deskewing
        # --------------------------

        if ENABLE_DESKEW:

            processed = ImageService.deskew_image(
                processed
            )

            applied_steps.append(
                "deskew"
            )

        # --------------------------
        # Image Enhancement
        # --------------------------

        if ENABLE_ENHANCEMENT:

            processed = ImageService.enhance_image(
                processed
            )

            applied_steps.append(
                "enhancement"
            )

        # --------------------------
        # Thresholding
        # --------------------------

        if ENABLE_THRESHOLD:

            processed = cv2.threshold(
                processed,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU
            )[1]

            applied_steps.append(
                "threshold"
            )

        os.makedirs(
            PROCESSED_DIR,
            exist_ok=True
        )

        processed_path = os.path.join(
            PROCESSED_DIR,
            f"processed_{os.path.basename(image_path)}"
        )

        cv2.imwrite(
            processed_path,
            processed
        )

        processed_path = (
            processed_path.replace(
                "\\",
                "/"
            )
        )

        logger.info(
            f"Saved: {processed_path}"
        )

        return {
            "original_image":
                image_path,
            "processed_image":
                processed_path,
            "applied_steps":
                applied_steps
        }
