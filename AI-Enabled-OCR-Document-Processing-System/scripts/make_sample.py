#!/usr/bin/env python
"""Generate a synthetic sample image with text for demos and tests.

Usage:
    python scripts/make_sample.py                 # -> samples/sample_printed.png
    python scripts/make_sample.py --out custom.png

The image is rendered with Windows' default fonts via PIL, so it is a
clean, machine-printed sample that Tesseract recognizes reliably.
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

TEXT = (
    "Optical Character Recognition\n"
    "Digitizing documents is fun.\n"
    "Invoice number: 482913\n"
    "Total amount: $1,234.56"
)


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_sample(out_path: str, text: str = TEXT, padding: int = 40) -> None:
    size = 22
    font = _font(size)
    temp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = text.splitlines()
    w = max(temp.textbbox((0, 0), ln, font=font)[2] for ln in lines)
    h = size * len(lines) * 1.5
    image = Image.new("RGB", (w + 2 * padding, int(h) + 2 * padding), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for ln in lines:
        draw.text((padding, y), ln, fill="black", font=font)
        y += int(size * 1.5)
    image.save(out_path)
    print(f"Sample saved to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "samples",
        "sample_printed.png",
    )
    parser.add_argument("--out", default=default, help="Output image path.")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    make_sample(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
