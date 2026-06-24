import os

import requests
from dotenv import load_dotenv

load_dotenv()


def _env_int(name, default):
    try:
        return int(
            os.getenv(
                name,
                str(default)
            )
        )
    except ValueError:
        return default


BASE_URL = (
    os.getenv(
        "BASE_URL"
    )
    or "http://127.0.0.1:8000"
)
CONNECT_TIMEOUT = _env_int(
    "BACKEND_CONNECT_TIMEOUT_SECONDS",
    10
)
REQUEST_TIMEOUT = _env_int(
    "BACKEND_REQUEST_TIMEOUT_SECONDS",
    120
)
LONG_REQUEST_TIMEOUT = _env_int(
    "BACKEND_LONG_REQUEST_TIMEOUT_SECONDS",
    900
)


class ApiClientError(Exception):
    def __init__(
        self,
        message,
        status_code=None,
        error_code=None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


def _extract_text_value(value):
    if isinstance(
        value,
        dict
    ):
        return value.get(
            "corrected_text",
            ""
        )

    return value or ""


def health_check():
    return _request("get", "/")


def _detail_from_response(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Backend returned an invalid response."

    detail = payload.get(
        "detail",
        payload
    )

    if isinstance(
        detail,
        dict
    ):
        return detail.get(
            "message",
            str(detail)
        )

    return str(detail)


def _error_code_from_status(status_code):
    if status_code == 400:
        return "invalid_file_format"

    if status_code == 503:
        return "model_loading_failed"

    if status_code >= 500:
        return "backend_error"

    return "request_failed"


def _request(method, endpoint, timeout_seconds=None, **kwargs):
    timeout = (
        CONNECT_TIMEOUT,
        timeout_seconds or REQUEST_TIMEOUT
    )

    try:
        response = requests.request(
            method,
            f"{BASE_URL}{endpoint}",
            timeout=timeout,
            **kwargs
        )
    except requests.exceptions.ConnectionError as e:
        raise ApiClientError(
            "Backend is unavailable. Start the backend server and try again.",
            error_code="backend_unavailable"
        ) from e
    except requests.exceptions.Timeout as e:
        raise ApiClientError(
            "The backend did not respond in time. Please try again.",
            error_code="backend_timeout"
        ) from e
    except requests.exceptions.RequestException as e:
        raise ApiClientError(
            f"Could not reach backend: {e}",
            error_code="backend_unavailable"
        ) from e

    if not response.ok:
        raise ApiClientError(
            _detail_from_response(
                response
            ),
            status_code=response.status_code,
            error_code=_error_code_from_status(
                response.status_code
            )
        )

    try:
        return response.json()
    except ValueError as e:
        raise ApiClientError(
            "Backend returned an invalid JSON response.",
            status_code=response.status_code,
            error_code="invalid_backend_response"
        ) from e


def upload_file(file):

    file.seek(0)

    files = {
        "file": (
            file.name,
            file,
            file.type
        )
    }

    return _request(
        "post",
        "/upload",
        files=files
    )


def preprocess_document(file_path):

    return _request(
        "post",
        "/preprocess",
        timeout_seconds=LONG_REQUEST_TIMEOUT,
        params={
            "file_path": file_path
        }
    )


def run_ocr(image_path):

    return _request(
        "post",
        "/ocr",
        timeout_seconds=LONG_REQUEST_TIMEOUT,
        params={
            "image_path": image_path
        }
    )


def correct_text(ocr_text):

    return _request(
        "post",
        "/correct",
        timeout_seconds=LONG_REQUEST_TIMEOUT,
        params={
            "ocr_text": ocr_text
        }
    )


def classify_document(text):

    return _request(
        "post",
        "/classify",
        timeout_seconds=LONG_REQUEST_TIMEOUT,
        params={
            "text": text
        }
    )

def extract_information(
    text
):

    return _request(
        "post",
        "/extract",
        timeout_seconds=LONG_REQUEST_TIMEOUT,
        json={
            "text": _extract_text_value(
                text
            )
        }
    )

def validate_document(data):

    return _request(
        "post",
        "/validate",
        json=data
    )

def ask_question(
    document_text,
    question
):

    return _request(
        "post",
        "/qa",
        timeout_seconds=LONG_REQUEST_TIMEOUT,
        params={
            "document_text": document_text,
            "question": question
        }
    )
