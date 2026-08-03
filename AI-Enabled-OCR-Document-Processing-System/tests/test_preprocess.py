import cv2
import numpy as np
import pytest

from src import preprocess
from src.detection import TextRegion, merge_regions


def _make_text_image(size=(400, 800), lines=("Hello World", "OCR test 123")):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    img = Image.new("RGB", (size[1], size[0]), "white")
    draw = ImageDraw.Draw(img)
    y = 10
    for ln in lines:
        draw.text((10, y), ln, fill="black", font=font)
        y += 20
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def test_to_grayscale():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    gray = preprocess.to_grayscale(image)
    assert gray.ndim == 2
    assert gray.shape == (50, 60)


def test_denoise_methods():
    image = _make_text_image()
    for method in ("none", "gaussian", "median", "bilateral", "nlm"):
        out = preprocess.denoise(image, method)
        assert out.shape == image.shape
    with pytest.raises(ValueError):
        preprocess.denoise(image, "bogus")


def test_threshold_otsu_is_binary():
    image = _make_text_image()
    binary = preprocess.threshold(preprocess.to_grayscale(image), method="otsu")
    unique = np.unique(binary)
    assert set(unique.tolist()) <= {0, 255}


def test_threshold_unknown_method():
    with pytest.raises(ValueError):
        preprocess.threshold(_make_text_image(), method="bogus")


def test_deskew_straight_image_unchanged_shape():
    image = _make_text_image()
    out = preprocess.deskew(image)
    assert out.shape == image.shape


def test_preprocess_image_disabled_returns_copy():
    image = _make_text_image()
    out = preprocess.preprocess_image(image, {"enabled": False})
    assert np.array_equal(out, image)


def test_preprocess_image_default_pipeline():
    image = _make_text_image()
    out = preprocess.preprocess_image(image)
    assert out.dtype == np.uint8


def test_grayscale_from_bgr_roundtrip_shape():
    image = _make_text_image()
    assert preprocess.to_grayscale(image).shape == image.shape[:2]


def test_merge_regions_same_line():
    regions = [
        TextRegion(points=np.array([[10, 10], [60, 10], [60, 30], [10, 30]], dtype=np.float32), confidence=0.9),
        TextRegion(points=np.array([[70, 10], [120, 10], [120, 30], [70, 30]], dtype=np.float32), confidence=0.8),
    ]
    merged = merge_regions(regions)
    assert len(merged) == 1
    assert merged[0].box == (10, 10, 110, 20)


def test_merge_regions_different_lines():
    regions = [
        TextRegion(points=np.array([[10, 10], [60, 10], [60, 30], [10, 30]], dtype=np.float32), confidence=0.9),
        TextRegion(points=np.array([[10, 60], [60, 60], [60, 80], [10, 80]], dtype=np.float32), confidence=0.8),
    ]
    merged = merge_regions(regions)
    assert len(merged) == 2
