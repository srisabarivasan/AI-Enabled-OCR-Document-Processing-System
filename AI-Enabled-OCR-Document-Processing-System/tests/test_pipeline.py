"""End-to-end pipeline test using a generated image (Tesseract backend).

Skipped automatically if the tesseract binary is unavailable.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.make_sample import make_sample  # noqa: E402
from src.pipeline import OCRPipeline  # noqa: E402

TESSERACT = pytest.importorskip("pytesseract")


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory):
    path = tmp_path_factory.mktemp("ocr") / "sample.png"
    make_sample(str(path), text="Hello OCR World 42")
    return str(path)


def test_pipeline_runs_end_to_end(sample_image, tmp_path):
    pipeline = OCRPipeline()
    pipeline.output_cfg["save_viz"] = True
    pipeline.output_cfg["viz_path"] = str(tmp_path / "viz.jpg")
    result = pipeline.run(sample_image)
    assert result.text
    assert "hello" in result.text.lower()
    assert os.path.exists(pipeline.output_cfg["viz_path"])


def test_pipeline_validation(sample_image, tmp_path):
    gt = tmp_path / "gt.txt"
    gt.write_text("Hello OCR World 42", encoding="utf-8")
    pipeline = OCRPipeline()
    result = pipeline.run(sample_image, ground_truth=str(gt))
    assert result.confidence >= 0.0
