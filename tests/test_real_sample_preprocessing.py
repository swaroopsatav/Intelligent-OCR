from pathlib import Path

from backend.services.image_service import ImageService
from backend.services.pdf_service import PDFService


def test_sample_pdf_can_render_and_preprocess_first_page():
    sample_pdf = Path("samples/demo_invoice.pdf")

    assert sample_pdf.exists()

    page_images = PDFService.pdf_to_images(
        str(sample_pdf)
    )

    assert page_images

    result = ImageService.preprocess_image(
        page_images[0]
    )

    assert Path(
        result["processed_image"]
    ).exists()
    assert "grayscale" in result["applied_steps"]
