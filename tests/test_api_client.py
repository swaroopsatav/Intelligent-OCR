import importlib.util
from pathlib import Path

import pytest
import requests


API_CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "services"
    / "api_client.py"
)

spec = importlib.util.spec_from_file_location(
    "api_client",
    API_CLIENT_PATH
)
api_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_client)


class FakeResponse:
    def __init__(
        self,
        status_code,
        payload=None,
        text=""
    ):
        self.status_code = status_code
        self.payload = payload
        self.text = text
        self.ok = 200 <= status_code < 400

    def json(self):
        if self.payload is None:
            raise ValueError("no json")

        return self.payload


def test_request_maps_connection_error(monkeypatch):
    def fail_request(*args, **kwargs):
        raise requests.exceptions.ConnectionError(
            "refused"
        )

    monkeypatch.setattr(
        api_client.requests,
        "request",
        fail_request
    )

    with pytest.raises(api_client.ApiClientError) as exc:
        api_client.health_check()

    assert exc.value.error_code == "backend_unavailable"


def test_request_maps_invalid_file_format(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "request",
        lambda *args, **kwargs: FakeResponse(
            400,
            {
                "detail": "Invalid file format."
            }
        )
    )

    with pytest.raises(api_client.ApiClientError) as exc:
        api_client.health_check()

    assert exc.value.error_code == "invalid_file_format"
    assert exc.value.message == "Invalid file format."


def test_request_maps_model_loading_failure(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "request",
        lambda *args, **kwargs: FakeResponse(
            503,
            {
                "detail": "Model loading failed."
            }
        )
    )

    with pytest.raises(api_client.ApiClientError) as exc:
        api_client.health_check()

    assert exc.value.error_code == "model_loading_failed"


def test_health_check_uses_default_timeout(monkeypatch):
    captured = {}

    def fake_request(*args, **kwargs):
        captured.update(
            kwargs
        )

        return FakeResponse(
            200,
            {
                "status": "running"
            }
        )

    monkeypatch.setattr(
        api_client.requests,
        "request",
        fake_request
    )

    api_client.health_check()

    assert captured["timeout"] == (
        api_client.CONNECT_TIMEOUT,
        api_client.REQUEST_TIMEOUT
    )


def test_ocr_uses_long_timeout(monkeypatch):
    captured = {}

    def fake_request(*args, **kwargs):
        captured.update(
            kwargs
        )

        return FakeResponse(
            200,
            {
                "status": "success",
                "text": "hello"
            }
        )

    monkeypatch.setattr(
        api_client.requests,
        "request",
        fake_request
    )

    api_client.run_ocr(
        "data/processed/page_1.png"
    )

    assert captured["timeout"] == (
        api_client.CONNECT_TIMEOUT,
        api_client.LONG_REQUEST_TIMEOUT
    )
