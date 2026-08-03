"""Image preprocessing for OCR: enhance text clarity.

Provides a configurable chain of grayscale conversion, denoising,
binarization (thresholding) and skew correction. Every step is a small
pure function that can be used independently or composed via
:func:`preprocess_image`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

DEFAULT_PREPROCESS = {
    "enabled": True,
    "grayscale": True,
    "denoise": "gaussian",
    "denoise_kernel": 5,
    "threshold": "otsu",
    "adaptive_block_size": 31,
    "adaptive_c": 15,
    "global_thresh": 127,
    "invert": False,
    "deskew": True,
    "deskew_max_angle": 15,
}


@dataclass
class PreprocessConfig:
    """Preprocessing settings (mirrors config.yaml's ``preprocess`` block)."""

    enabled: bool = True
    grayscale: bool = True
    denoise: str = "gaussian"  # none | gaussian | median | bilateral | nlm
    denoise_kernel: int = 5
    threshold: str = "otsu"  # none | otsu | adaptive | global
    adaptive_block_size: int = 31
    adaptive_c: int = 15
    global_thresh: int = 127
    invert: bool = False
    deskew: bool = True
    deskew_max_angle: int = 15

    @classmethod
    def from_dict(cls, cfg: dict) -> "PreprocessConfig":
        if not cfg:
            return cls()
        known = {k: v for k, v in cls.__dataclass_fields__.items() if v.init}
        values = {k: cfg[k] for k in known if k in cfg}
        return cls(**values)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale if needed."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def denoise(image: np.ndarray, method: str = "gaussian", kernel: int = 5) -> np.ndarray:
    """Reduce noise while preserving text edges.

    Methods:
      - ``none``: return the image unchanged.
      - ``gaussian``: fast blur, good default for scanned documents.
      - ``median``: strong against salt-and-pepper noise.
      - ``bilateral``: preserves edges better than gaussian, slower.
      - ``nlm``: non-local means, best quality, slowest.
    """
    if method == "none":
        return image
    if method == "gaussian":
        k = kernel if kernel % 2 == 1 else kernel + 1
        return cv2.GaussianBlur(image, (k, k), 0)
    if method == "median":
        k = kernel if kernel % 2 == 1 else kernel + 1
        return cv2.medianBlur(image, k)
    if method == "bilateral":
        return cv2.bilateralFilter(image, kernel, 75, 75)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(image, None, h=15)
    raise ValueError(f"Unknown denoise method: {method!r}")


def threshold(image: np.ndarray, method: str = "otsu", **kwargs) -> np.ndarray:
    """Binarize a grayscale image to pure black/white.

    Methods:
      - ``none``: return the image unchanged.
      - ``otsu``: automatic global threshold (best default).
      - ``adaptive``: adaptive thresholding for uneven lighting.
      - ``global``: fixed threshold (see ``global_thresh``).
    """
    if method == "none":
        return image
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if method == "adaptive":
        block = kwargs.get("adaptive_block_size", 31)
        c = kwargs.get("adaptive_c", 15)
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c
        )
    if method == "global":
        level = kwargs.get("global_thresh", 127)
        _, binary = cv2.threshold(image, level, 255, cv2.THRESH_BINARY)
        return binary
    raise ValueError(f"Unknown threshold method: {method!r}")


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct small skew angles using text contours and a min-area rect.

    Returns the rotated image. If no reliable angle is found (low text
    coverage, blank image) the input is returned unchanged.
    """
    angle = _estimate_skew(image)
    if angle is None or abs(angle) > max_angle:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _estimate_skew(image: np.ndarray) -> Optional[float]:
    """Estimate skew angle in degrees from the dominant text contour."""
    gray = to_grayscale(image)
    binary = threshold(gray, method="otsu")
    binary = cv2.bitwise_not(binary)  # text becomes white

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles: list[float] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:
            continue
        _, (_, _), angle = cv2.minAreaRect(contour)
        # Normalize so the angle stays small regardless of text orientation.
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    if not angles:
        return None
    return float(np.mean(angles))


def preprocess_image(
    image: np.ndarray,
    cfg: PreprocessConfig | dict | None = None,
) -> np.ndarray:
    """Run the full preprocessing chain on an image.

    ``cfg`` may be a :class:`PreprocessConfig` or a raw dict (as loaded
    from config.yaml). Returns the processed image (BGR or grayscale,
    depending on the pipeline) or the input unchanged if disabled.
    """
    if isinstance(cfg, dict):
        cfg = PreprocessConfig.from_dict(cfg)
    elif cfg is None:
        cfg = PreprocessConfig()

    if not cfg.enabled:
        return image

    work = image.copy()
    if cfg.grayscale:
        work = to_grayscale(work)
    work = denoise(work, cfg.denoise, cfg.denoise_kernel)
    work = threshold(
        work,
        cfg.threshold,
        adaptive_block_size=cfg.adaptive_block_size,
        adaptive_c=cfg.adaptive_c,
        global_thresh=cfg.global_thresh,
    )
    if cfg.invert:
        work = cv2.bitwise_not(work)
    if cfg.deskew:
        work = deskew(work, cfg.deskew_max_angle)
    return work
