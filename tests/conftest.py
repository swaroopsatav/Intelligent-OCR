import sys
import types

import pytest
from fastapi.testclient import TestClient


class DummyReader:
    def __init__(self, *args, **kwargs):
        pass

    def readtext(self, image_path):
        return []


easyocr_stub = types.ModuleType("easyocr")
easyocr_stub.Reader = DummyReader
sys.modules.setdefault("easyocr", easyocr_stub)


@pytest.fixture
def client():
    from backend.api.app import app

    return TestClient(app)
