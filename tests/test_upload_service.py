import asyncio

import pytest

from backend.services import upload_service
from backend.services.upload_service import UploadService


class FakeUpload:
    def __init__(self, filename, content):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def test_validate_extension_accepts_supported_files():
    UploadService.validate_extension("invoice.pdf")
    UploadService.validate_extension("scan.PNG")
    UploadService.validate_extension("photo.jpeg")


def test_validate_extension_rejects_unsupported_files():
    with pytest.raises(ValueError, match="Invalid file format"):
        UploadService.validate_extension("notes.txt")


def test_validate_size_rejects_files_over_limit():
    with pytest.raises(ValueError, match="File size exceeds"):
        UploadService.validate_size(upload_service.MAX_FILE_SIZE + 1)


def test_save_file_writes_upload_to_configured_directory(
    tmp_path,
    monkeypatch
):
    monkeypatch.setattr(
        upload_service,
        "UPLOAD_DIR",
        str(tmp_path)
    )

    result = asyncio.run(
        UploadService.save_file(
            FakeUpload(
                "sample.pdf",
                b"%PDF test"
            )
        )
    )

    saved_file = tmp_path / "sample.pdf"

    assert saved_file.read_bytes() == b"%PDF test"
    assert result["filename"] == "sample.pdf"
    assert result["file_path"] == str(saved_file)
    assert result["file_size"] == len(b"%PDF test")
