#!/usr/bin/env python
"""Download the frozen EAST text-detection model used by OpenCV's DNN module.

Usage:
    python scripts/download_east.py

The model (~50 MB) is saved to models/frozen_east_text_detection.pb.
Source: https://github.com/oyyd/frozen_east_text_detection.pb
"""

from __future__ import annotations

import os
import sys
import urllib.request

URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
DEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "frozen_east_text_detection.pb",
)


def main() -> int:
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"Model already present: {DEST}")
        return 0
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    print(f"Downloading EAST model from {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved to {DEST} ({os.path.getsize(DEST):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
