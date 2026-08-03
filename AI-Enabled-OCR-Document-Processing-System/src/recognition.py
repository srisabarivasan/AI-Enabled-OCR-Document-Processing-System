"""Text recognition: convert image regions into strings.

Supported backends:
  - ``tesseract``: classic OCR via ``pytesseract``. Works on a full
    image or per detected region.
  - ``easyocr``: deep-learning recognizer (optional dependency).
  - ``trocr``: Transformer-based recognition via HuggingFace
    ``TrOCRForCausalLM`` (optional, heavy).

Recognition results are returned as :class:`RecognitionResult` objects
and can optionally be grouped into paragraphs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Allow running this file directly (e.g. in an IDE debugger) as well as
# importing it as part of the `src` package.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

import numpy as np

from src.detection import TextRegion, crop_regions


@dataclass
class RecognitionResult:
    """Text recognized from one region."""

    text: str
    confidence: float
    box: tuple[int, int, int, int]  # (x, y, w, h)
    backend: str

    @property
    def cleaned(self) -> str:
        return self.text.strip()


@dataclass
class _Backend:
    """Runtime state for lazy-loaded backends (module-level singletons)."""

    tessdata_dir: Optional[str] = None
    easyocr_reader: Optional[object] = None
    trocr_processor: Optional[object] = None
    trocr_model: Optional[object] = None


_BACKENDS = _Backend()

_WINDOWS_TESSERACT_CANDIDATES = (
    "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
    "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
    "%LOCALAPPDATA%\\Programs\\Tesseract-OCR\\tesseract.exe",
)


def _resolve_tesseract_cmd() -> str:
    """Return a working path to the tesseract binary.

    Checks the OS PATH first (via shutil), then common Windows install
    locations. Raises RuntimeError if none is found.
    """
    import shutil
    from pathlib import Path

    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _WINDOWS_TESSERACT_CANDIDATES:
        path = Path(os.path.expandvars(candidate))
        if path.is_file():
            return str(path)
    raise RuntimeError(
        "Tesseract binary not found. Install it and add it to PATH:\n"
        "  Windows: winget install UB-Mannheim.TesseractOCR\n"
        "  Ubuntu : sudo apt install tesseract-ocr"
    )


def _tesseract_config(psm: int, lang: str) -> str:
    opts = f"--psm {psm}"
    if _BACKENDS.tessdata_dir:
        opts += f" --tessdata-dir {_BACKENDS.tessdata_dir}"
    return opts


def recognize_tesseract(
    image: np.ndarray,
    lang: str = "eng",
    psm: int = 6,
    tessdata_dir: Optional[str] = None,
) -> RecognitionResult:
    """Run Tesseract on the whole image."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd()
    if tessdata_dir:
        _BACKENDS.tessdata_dir = tessdata_dir
    cfg = _tesseract_config(psm, lang)
    data = pytesseract.image_to_data(
        image, lang=lang, config=cfg, output_type=pytesseract.Output.DICT
    )
    lines: list[str] = []
    confs: list[float] = []
    for i, text in enumerate(data["text"]):
        t = (text or "").strip()
        if not t:
            continue
        conf = data["conf"][i]
        if conf < 0:
            conf = 0.0
        lines.append(t)
        confs.append(float(conf) / 100.0)
    text = " ".join(lines)
    confidence = float(np.mean(confs)) if confs else 0.0
    h, w = image.shape[:2]
    return RecognitionResult(text=text, confidence=confidence, box=(0, 0, w, h), backend="tesseract")


def recognize_easyocr(
    image: np.ndarray,
    lang: str = "en",
    use_gpu: bool = False,
    paragraph: bool = False,
) -> RecognitionResult:
    """Run EasyOCR on the whole image (detection + recognition in one call)."""
    import easyocr

    if _BACKENDS.easyocr_reader is None:
        _BACKENDS.easyocr_reader = easyocr.Reader(
            [lang], gpu=use_gpu, verbose=False
        )
    output = _BACKENDS.easyocr_reader.readtext(
        image, detail=1, paragraph=paragraph
    )
    if paragraph:
        lines = [str(item) for item in output]
        text = "\n".join(lines)
        confidence = 0.0
    else:
        lines = [str(item[1]) for item in output]
        confs = [float(item[2]) for item in output]
        text = " ".join(lines)
        confidence = float(np.mean(confs)) if confs else 0.0
    h, w = image.shape[:2]
    return RecognitionResult(text=text, confidence=confidence, box=(0, 0, w, h), backend="easyocr")


def _load_trocr(model_name: str):
    from transformers import TrOCRForCausalLM, TrOCRProcessor

    if _BACKENDS.trocr_processor is None:
        _BACKENDS.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _BACKENDS.trocr_model = TrOCRForCausalLM.from_pretrained(model_name)
    return _BACKENDS.trocr_processor, _BACKENDS.trocr_model


def recognize_trocr_crop(
    crop: np.ndarray, model_name: str = "microsoft/trocr-base-printed", max_length: int = 128
) -> tuple[str, float]:
    """Recognize a single text-region crop with TrOCR."""
    from PIL import Image

    processor, model = _load_trocr(model_name)
    pil_image = Image.fromarray(crop[:, :, ::-1]) if crop.ndim == 3 else Image.fromarray(crop)
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated = model.generate(pixel_values, max_length=max_length)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text, 1.0


def recognize_regions(
    crops: Iterable[np.ndarray],
    regions: Iterable[TextRegion],
    backend: str = "tesseract",
    lang: str = "eng",
    psm: int = 6,
    tessdata_dir: Optional[str] = None,
    trocr_model: str = "microsoft/trocr-base-printed",
    trocr_max_length: int = 128,
) -> list[RecognitionResult]:
    """Recognize each detected region crop individually.

    Used when a separate detection stage produced regions (EAST or
    EasyOCR detection). Returns results in the same order as regions.
    """
    results: list[RecognitionResult] = []
    for crop, region in zip(crops, regions):
        if backend == "tesseract":
            res = recognize_tesseract(crop, lang=lang, psm=psm, tessdata_dir=tessdata_dir)
        elif backend == "easyocr":
            res = recognize_easyocr(crop, lang=lang)
        elif backend == "trocr":
            text, conf = recognize_trocr_crop(crop, trocr_model, trocr_max_length)
            res = RecognitionResult(text=text, confidence=conf, box=region.box, backend="trocr")
        else:
            raise ValueError(f"Unknown recognition backend: {backend!r}")
        # Carry over the region's box so results line up with detection.
        res.box = region.box
        results.append(res)
    return results


def recognize(
    image: np.ndarray,
    backend: str = "tesseract",
    lang: str = "eng",
    psm: int = 6,
    tessdata_dir: Optional[str] = None,
    regions: Optional[list[TextRegion]] = None,
    config: Optional[dict] = None,
) -> list[RecognitionResult]:
    """High-level recognition entry point.

    If ``regions`` are given, each crop is recognized and results are
    returned per region. Otherwise the full image is recognized in one
    pass (the common Tesseract / EasyOCR path).
    """
    config = config or {}
    if regions and backend in ("tesseract", "easyocr", "trocr"):
        crops = crop_regions(image, regions)
        return recognize_regions(
            crops,
            regions,
            backend=backend,
            lang=lang,
            psm=psm,
            tessdata_dir=tessdata_dir,
            trocr_model=config.get("trocr", {}).get("model", "microsoft/trocr-base-printed"),
            trocr_max_length=int(config.get("trocr", {}).get("max_length", 128)),
        )
    if backend == "tesseract":
        return [recognize_tesseract(image, lang=lang, psm=psm, tessdata_dir=tessdata_dir)]
    if backend == "easyocr":
        easy = config.get("easyocr", {})
        return [
            recognize_easyocr(
                image,
                lang=lang,
                use_gpu=easy.get("use_gpu", False),
                paragraph=easy.get("paragraph", False),
            )
        ]
    if backend == "trocr":
        # TrOCR is a pure recognizer: it needs regions to work on.
        raise ValueError(
            "TrOCR requires a detection stage (detection.method != none) "
            "so each region can be recognized independently."
        )
    raise ValueError(f"Unknown recognition backend: {backend!r}")
