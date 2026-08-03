"""PDF extraction tests (text layer + scanned-page OCR)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytesseract = pytest.importorskip("pytesseract")


@pytest.fixture(scope="module")
def text_pdf(tmp_path_factory):
    import fitz

    path = tmp_path_factory.mktemp("pdf") / "text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Hello PDF World 42", fontsize=18, fontname="helv")
    doc.save(path)
    doc.close()
    return str(path)


@pytest.fixture(scope="module")
def scanned_pdf(tmp_path_factory):
    import cv2
    import fitz
    import numpy as np

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    img = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 60), "Scanned Page Text 99", fill="black", font=font)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    path = tmp_path_factory.mktemp("pdf") / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(50, 50, 545, 300), stream=cv2.imencode(".png", bgr)[1].tobytes())
    doc.save(path)
    doc.close()
    return str(path)


def test_run_pdf_text_layer(text_pdf, tmp_path):
    from src.pipeline import OCRPipeline

    result = OCRPipeline().run_pdf(text_pdf, out_dir=str(tmp_path))
    assert "hello pdf world 42" in result.text.lower()
    assert os.path.exists(os.path.join(tmp_path, "text_p1.txt"))


def test_run_pdf_scanned_ocr(scanned_pdf, tmp_path):
    from src.pipeline import OCRPipeline

    result = OCRPipeline().run_pdf(scanned_pdf, out_dir=str(tmp_path))
    assert result.text.strip()
    assert "scanned" in result.text.lower() or "page" in result.text.lower()
    assert os.path.exists(os.path.join(tmp_path, "scanned_p1.txt"))
