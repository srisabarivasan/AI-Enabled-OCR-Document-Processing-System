#!/usr/bin/env python
"""Command-line interface for the OCR pipeline.

Examples:
  # Run Tesseract on an image with default settings
  python main.py path/to/image.png

  # Use a different backend and language
  python main.py image.jpg --backend easyocr --lang en

  # Enable EAST text detection (download the model first)
  python scripts/download_east.py
  python main.py image.jpg --detect east --backend tesseract

  # Validate against ground truth and save artifacts
  python main.py image.jpg --ground-truth reference.txt --save-viz --json-output out.json
"""

from __future__ import annotations

import argparse
import os
import sys

from src.pipeline import OCRPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCR: extract text from images (preprocessing -> detection -> recognition -> validation)."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to the input image (PNG/JPG/...). Defaults to samples/sample_printed.png.",
    )
    parser.add_argument("--config", default=None, help="Path to a config.yaml file.")
    parser.add_argument(
        "--backend",
        choices=["tesseract", "easyocr", "trocr"],
        default=None,
        help="Recognition backend (overrides config).",
    )
    parser.add_argument(
        "--detect",
        choices=["none", "east", "easyocr"],
        default=None,
        help="Text detection method (overrides config).",
    )
    parser.add_argument("--lang", default=None, help="Recognition language, e.g. eng/en.")
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to a .txt file with expected text for validation.",
    )
    parser.add_argument("--save-viz", action="store_true", help="Save box visualization.")
    parser.add_argument(
        "--json-output", default=None, help="Path to write structured JSON results."
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Disable image preprocessing."
    )
    parser.add_argument(
        "--dir",
        default=None,
        metavar="FOLDER",
        help="OCR every image in a folder; each result is saved separately to output/.",
    )
    parser.add_argument(
        "--out-dir",
        default="output",
        help="Where batch results are written (default: output/).",
    )
    return parser


_IMAGE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
    ".avif",
)
_DOCUMENT_EXTENSIONS = _IMAGE_EXTENSIONS + (".pdf",)


def _sample_files() -> list[str]:
    """All images/PDFs currently in samples/ (most recent last)."""
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    candidates = [
        os.path.join(samples_dir, name)
        for name in os.listdir(samples_dir)
        if name.lower().endswith(_DOCUMENT_EXTENSIONS)
    ]
    return sorted(candidates, key=os.path.getmtime)


def _files_in_folder(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise SystemExit(f"Not a folder: {folder}")
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(_DOCUMENT_EXTENSIONS)
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dir:
        images = _files_in_folder(args.dir)
        if not images:
            print(f"No images or PDFs found in {args.dir}")
            return 0
        return _run_batch(images, args, out_dir=args.out_dir)

    if not args.image:
        images = _sample_files()
        if not images:
            build_parser().print_help()
            return 2
        return _run_batch(images, args, out_dir=args.out_dir)

    return _run_single(args)


def _configure_pipeline(args, out_dir: str) -> OCRPipeline:
    pipeline = OCRPipeline(args.config)
    if args.backend:
        pipeline.recognition_cfg["backend"] = args.backend
    if args.detect:
        pipeline.detection_cfg["method"] = args.detect
    if args.lang:
        pipeline.recognition_cfg["lang"] = args.lang
    if args.no_preprocess:
        pipeline.preprocess_cfg.enabled = False
    if args.save_viz:
        pipeline.output_cfg["save_viz"] = True
    if args.json_output:
        pipeline.output_cfg["json_path"] = args.json_output
    pipeline.output_cfg["viz_path"] = os.path.join(out_dir, "ocr_viz.jpg")
    return pipeline


def _run_batch(images: list[str], args, out_dir: str) -> int:
    pipeline = _configure_pipeline(args, out_dir)
    print(f"\nProcessing {len(images)} file(s) -> {out_dir}\\")
    results = pipeline.run_batch(images, out_dir=out_dir)
    print("\nBatch summary:")
    for path, result in zip(images, results):
        status = "OK"
        if result is None:
            status = "FAILED"
            print(f"  {os.path.basename(path):40s} {status}")
            continue
        print(
            f"  {os.path.basename(path):40s} {status}  "
            f"conf={result.confidence:.3f}  words={len(result.text.split())}"
        )
    return 0


def _run_single(args) -> int:
    pipeline = _configure_pipeline(args, "output")
    if args.image.lower().endswith(".pdf"):
        result = pipeline.run_pdf(args.image, out_dir="output")
    else:
        result = pipeline.run(args.image, ground_truth=args.ground_truth)

    print("\nRecognized text:\n-----------------")
    print(result.text or "(no text found)")
    print(f"\nConfidence: {result.confidence:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
