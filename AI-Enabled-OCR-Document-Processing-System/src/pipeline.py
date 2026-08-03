"""End-to-end OCR pipeline orchestrator.

Loads configuration, runs preprocessing, text detection and
recognition, then optionally validates against ground truth and saves
visualizations / JSON output.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Allow running this file directly (e.g. in an IDE debugger) as well as
# importing it as part of the `src` package.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

import cv2
import numpy as np
import yaml

from src import preprocess
from src.detection import TextRegion, detect
from src.recognition import RecognitionResult, recognize
from src.validation import validate

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


@dataclass
class OCRResult:
    """Everything produced by one pipeline run."""

    text: str
    regions: list[TextRegion] = field(default_factory=list)
    recognitions: list[RecognitionResult] = field(default_factory=list)
    confidence: float = 0.0
    preprocessed: Optional[object] = None  # numpy array (kept out of repr)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "regions": [
                {
                    "box": list(map(int, r.box)),
                    "confidence": round(r.confidence, 4),
                    "text": rec.cleaned if rec else "",
                }
                for r, rec in zip(self.regions, self.recognitions)
            ],
        }

    def __repr__(self) -> str:
        return f"OCRResult(confidence={self.confidence:.3f}, regions={len(self.regions)})"


class OCRPipeline:
    """Configurable OCR pipeline.

    Parameters mirror ``config.yaml``. Any of them can be overridden
    after construction.
    """

    def __init__(self, config: dict | str | None = None):
        self.config = self._load_config(config)
        self.preprocess_cfg = preprocess.PreprocessConfig.from_dict(
            self.config.get("preprocess", {})
        )
        self.detection_cfg = self.config.get("detection", {})
        self.recognition_cfg = self.config.get("recognition", {})
        self.output_cfg = self.config.get("output", {})
        self.validation_cfg = self.config.get("validation", {})
        self._easyocr_reader = None

    @staticmethod
    def _load_config(config: dict | str | None) -> dict:
        if config is None:
            config = DEFAULT_CONFIG_PATH
        if isinstance(config, str):
            with open(config, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return config or {}

    # ------------------------------------------------------------------ steps

    def load_image(self, path: str) -> object:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            return image
        # OpenCV cannot decode some formats (e.g. AVIF, HEIC). Fall back
        # to Pillow and convert to a BGR numpy array.
        try:
            from PIL import Image

            pil_image = Image.open(path).convert("RGB")
            rgb = np.array(pil_image)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            raise FileNotFoundError(
                f"Could not read image (unsupported or corrupt format): {path}"
            ) from None

    def preprocess(self, image) -> object:
        return preprocess.preprocess_image(image, self.preprocess_cfg)

    def detect_text(self, image) -> list[TextRegion]:
        method = self.detection_cfg.get("method", "none")
        cfg = dict(self.detection_cfg)
        if method == "easyocr" and self._easyocr_reader is None:
            import easyocr  # lazy import so base installs stay light

            self._easyocr_reader = easyocr.Reader(
                [self.recognition_cfg.get("lang", "en")], gpu=False, verbose=False
            )
        return detect(image, method=method, config=cfg, easyocr_reader=self._easyocr_reader)

    def recognize_text(self, image, regions) -> list[RecognitionResult]:
        backend = self.recognition_cfg.get("backend", "tesseract")
        lang = self.recognition_cfg.get("lang", "eng")
        psm = int(self.recognition_cfg.get("psm", 6))
        tessdata_dir = self.recognition_cfg.get("tessdata_dir")
        if backend == "easyocr":
            lang = "en" if lang == "eng" else lang
        return recognize(
            image,
            backend=backend,
            lang=lang,
            psm=psm,
            tessdata_dir=tessdata_dir,
            regions=regions,
            config=self.recognition_cfg,
        )

    # ------------------------------------------------------------------- run

    def run(
        self,
        image_path: str,
        ground_truth: Optional[str] = None,
        output_prefix: Optional[str] = None,
    ) -> OCRResult:
        """Run the full pipeline on an image file.

        When ``output_prefix`` is given (batch mode), a plain-text file
        and a JSON file are written to ``{output_prefix}.txt`` and
        ``{output_prefix}.json``.
        """
        image = self.load_image(image_path)
        result = self._process(image)

        if output_prefix:
            self.save_outputs(result, output_prefix)
        if ground_truth:
            self._save_validation(result, ground_truth)
        if self.output_cfg.get("save_viz", False):
            viz_path = self.output_cfg.get("viz_path")
            if output_prefix:
                viz_path = f"{output_prefix}_viz.jpg"
            self.save_visualization(image_path, result, viz_path=viz_path)
        if not output_prefix and self.output_cfg.get("json_path"):
            self.save_json(result)

        return result

    def _process(self, image: np.ndarray) -> OCRResult:
        """Run preprocessing -> detection -> recognition on an in-memory image."""
        processed = self.preprocess(image) if self.preprocess_cfg.enabled else image

        # Detection runs on the preprocessed (grayscale/binary) image;
        # recognition gets the original image so it keeps full detail.
        regions = self.detect_text(processed)
        recognitions = self.recognize_text(image, regions)

        text = "\n".join(r.cleaned for r in recognitions if r.cleaned)
        confidences = [r.confidence for r in recognitions if r.confidence > 0]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=text,
            regions=regions,
            recognitions=recognitions,
            confidence=confidence,
            preprocessed=processed,
        )

    def run_pdf(
        self,
        pdf_path: str,
        out_dir: str = "output",
        dpi: int = 200,
    ) -> OCRResult:
        """Extract text from every page of a PDF.

        Pages with an embedded text layer are read directly (fast and
        exact); scanned pages are rendered to images and OCR'd. One
        ``{stem}_p{N}.txt`` + ``{stem}_p{N}.json`` file is saved per
        page. Returns a combined result for all pages.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is not installed. Run: pip install -r requirements.txt"
            ) from exc

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Could not read PDF: {pdf_path}")

        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        os.makedirs(out_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        page_results: list[OCRResult] = []
        for i, page in enumerate(doc, start=1):
            prefix = os.path.join(out_dir, f"{stem}_p{i}")
            text = (page.get_text() or "").strip()
            if text:
                h, w = page.rect.height, page.rect.width
                result = OCRResult(
                    text=text,
                    confidence=1.0,
                    preprocessed=None,
                )
                self.save_outputs(result, prefix)
                print(f"[page {i}] text layer extracted ({len(text.split())} words)")
            else:
                image = _render_pdf_page(page, dpi)
                result = self._process(image)
                self.save_outputs(result, prefix)
                if self.output_cfg.get("save_viz", False):
                    self.save_visualization(
                        image,
                        result,
                        viz_path=f"{prefix}_viz.jpg",
                    )
                print(f"[page {i}] scanned -> OCR conf={result.confidence:.3f}")
            page_results.append(result)

        combined_text = "\n".join(r.text for r in page_results)
        confs = [r.confidence for r in page_results]
        combined = OCRResult(
            text=combined_text,
            confidence=sum(confs) / len(confs) if confs else 0.0,
        )
        print(
            f"\nPDF complete: {len(page_results)} page(s) -> {out_dir}\\{stem}_p*.txt"
        )
        return combined

    def run_batch(
        self,
        image_paths: Iterable[str],
        out_dir: str = "output",
    ) -> list[OCRResult | None]:
        """Run the pipeline on every file (images and PDFs).

        PDFs are routed to :meth:`run_pdf`. One failing file does not
        stop the rest. Returns one entry per input file (``None`` for
        failures, keeping order aligned).
        """
        os.makedirs(out_dir, exist_ok=True)
        results: list[OCRResult | None] = []
        for path in image_paths:
            print(f"\n{'=' * 20} {path} {'=' * 20}")
            stem = os.path.splitext(os.path.basename(path))[0]
            prefix = os.path.join(out_dir, stem)
            try:
                if path.lower().endswith(".pdf"):
                    result = self.run_pdf(path, out_dir=out_dir)
                else:
                    result = self.run(path, output_prefix=prefix)
            except Exception as exc:
                print(f"[ERROR] {path}: {exc}")
                results.append(None)
                continue
            results.append(result)
            print(f"confidence={result.confidence:.3f}")
        return results

    # ------------------------------------------------------------ side effects

    def save_outputs(self, result: OCRResult, output_prefix: str) -> None:
        """Write ``{prefix}.txt`` and ``{prefix}.json`` side by side."""
        os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
        txt_path = f"{output_prefix}.txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(result.text)
        json_path = f"{output_prefix}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result.as_dict(), fh, indent=2)
        print(f"Saved {txt_path} and {json_path}")

    def _save_validation(self, result: OCRResult, ground_truth: str) -> None:
        with open(ground_truth, "r", encoding="utf-8") as fh:
            reference = fh.read().strip()
        report = validate(result.text, reference)
        print(f"\nValidation report:\n{json.dumps(report.as_dict(), indent=2)}\n")

    def save_visualization(
        self,
        image_or_path,
        result: OCRResult,
        viz_path: Optional[str] = None,
    ) -> None:
        if isinstance(image_or_path, str):
            image = self.load_image(image_or_path)
        else:
            image = image_or_path
        for rec in result.recognitions:
            x, y, w, h = rec.box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{rec.confidence:.2f}",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        path = viz_path or self.output_cfg.get("viz_path", "output/ocr_viz.jpg")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, image)
        print(f"Visualization saved to {path}")

    def save_json(self, result: OCRResult) -> None:
        path = self.output_cfg.get("json_path", "output/ocr_result.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result.as_dict(), fh, indent=2)
        print(f"JSON result saved to {path}")


def _render_pdf_page(page, dpi: int = 200) -> np.ndarray:
    """Render one PyMuPDF page to a BGR numpy array."""
    import fitz

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    # Allow `python src/pipeline.py <image>` (e.g. in an IDE debugger).
    from main import main

    sys.exit(main())
