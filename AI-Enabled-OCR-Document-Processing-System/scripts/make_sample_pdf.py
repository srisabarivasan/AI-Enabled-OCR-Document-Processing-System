#!/usr/bin/env python
"""Generate sample PDFs for testing.

Creates:
  - samples/sample_text.pdf  : a digital PDF with an embedded text layer.
  - samples/sample_scanned.pdf : a "scanned" PDF (rendered image, no text).

Usage:
    python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import argparse
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

TEXT = (
    "Optical Character Recognition\n"
    "PDF support added successfully.\n"
    "Invoice number: 482913\n"
    "Total amount: $1,234.56"
)


def make_text_pdf(out_path: str, text: str = TEXT) -> None:
    import fitz  # PyMuPDF

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=18, fontname="helv")
    doc.save(out_path)
    doc.close()
    print(f"Text-layer PDF saved to {out_path}")


def make_scanned_pdf(out_path: str, image_path: str = None) -> None:
    import fitz
    import cv2

    import numpy as np

    if image_path is None:
        image_path = os.path.join(BASE, "samples", "sample_printed.png")
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 portrait
    rect = fitz.Rect(50, 50, 545, 300)
    page.insert_image(rect, stream=cv2.imencode(".png", image)[1].tobytes())
    doc.save(out_path)
    doc.close()
    print(f"Scanned PDF saved to {out_path} (no text layer)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(BASE, "samples"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make_text_pdf(os.path.join(args.out_dir, "sample_text.pdf"))
    make_scanned_pdf(os.path.join(args.out_dir, "sample_scanned.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
