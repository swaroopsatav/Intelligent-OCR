import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "data",
    "uploads"
)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg"
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


load_dotenv()

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME"
)

BASE_URL = os.getenv(
    "BASE_URL"
)

ENABLE_THRESHOLD = (
    os.getenv(
        "ENABLE_THRESHOLD",
        "true"
    ).lower()
    == "true"
)

PDF_RENDER_SCALE = float(
    os.getenv(
        "PDF_RENDER_SCALE",
        "1.5"
    )
)

ENABLE_BLUR = (
    os.getenv(
        "ENABLE_BLUR",
        "true"
    ).lower()
    == "true"
)

ENABLE_DESKEW = (
    os.getenv(
        "ENABLE_DESKEW",
        "true"
    ).lower()
    == "true"
)

ENABLE_ENHANCEMENT = (
    os.getenv(
        "ENABLE_ENHANCEMENT",
        "true"
    ).lower()
    == "true"
)

OCR_CACHE_ENABLED = (
    os.getenv(
        "OCR_CACHE_ENABLED",
        "true"
    ).lower()
    == "true"
)

OCR_LANGUAGES = [
    language.strip()
    for language in os.getenv(
        "OCR_LANGUAGES",
        "en"
    ).split(",")
    if language.strip()
]

HF_LOCAL_FILES_ONLY = (
    os.getenv(
        "HF_LOCAL_FILES_ONLY",
        "false"
    ).lower()
    == "true"
)
