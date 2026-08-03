"""Text detection: locate regions of interest containing text.

Supported detectors:
  - ``none``: no detection, recognition runs on the full image.
  - ``east``: OpenCV DNN with the EAST text detector (fast, accurate on
    scene text). Requires the frozen model file (see
    ``scripts/download_east.py``).
  - ``easyocr``: CRAFT-based detector bundled with EasyOCR (optional
    dependency).

Every detector returns a list of :class:`TextRegion` objects with the
corner points, confidence and a straightened bounding box in original
image coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

EAST_MODEL_URL = (
    "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/"
    "frozen_east_text_detection.pb"
)

_RECT_NAMES = ["topLeft", "topRight", "bottomRight", "bottomLeft"]


@dataclass
class TextRegion:
    """A detected text region in image coordinates."""

    points: np.ndarray  # (4, 2) corner points: TL, TR, BR, BL
    confidence: float
    box: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) axis-aligned

    def __post_init__(self) -> None:
        if self.box is None and self.points is not None:
            pts = np.asarray(self.points, dtype=np.float32)
            x0, y0 = int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min()))
            x1, y1 = int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))
            self.box = (x0, y0, x1 - x0, y1 - y0)


def detect_east(
    image: np.ndarray,
    model_path: str,
    min_confidence: float = 0.5,
    input_width: int = 320,
    input_height: int = 320,
    nms_threshold: float = 0.4,
) -> list[TextRegion]:
    """Detect text with the EAST model via OpenCV's DNN module.

    The input image is scaled to ``input_width x input_height`` (both
    multiples of 32) for a single forward pass. The resulting boxes are
    mapped back to the original image coordinates.
    """
    if not model_path:
        raise FileNotFoundError(
            "EAST model file not provided. Download it first:\n"
            f"  python scripts/download_east.py\nfrom {EAST_MODEL_URL}"
        )
    orig_h, orig_w = image.shape[:2]
    scale_w = orig_w / float(input_width)
    scale_h = orig_h / float(input_height)

    # EAST expects a 3-channel image; preprocessing may have produced grayscale.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    net = cv2.dnn.readNet(model_path)
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (input_width, input_height), (123.68, 116.78, 103.94), True, False
    )
    net.setInput(blob)
    # EAST's two outputs: scores then geometry. Using explicit names is
    # robust across OpenCV 4.x (1-based indices) and 5.x (0-based).
    output_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores_map, geometry_map) = net.forward(output_layers)

    # Decode scores/geometry into candidate boxes and confidence scores.
    num_rows, num_cols = scores_map.shape[2:4]
    rects: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    for y in range(num_rows):
        scores = scores_map[0, 0, y]
        geometry = geometry_map[0, :, y]
        if np.amax(scores) < min_confidence:
            continue
        for x in range(num_cols):
            if scores[x] < min_confidence:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            d_top = geometry[0, x]
            d_right = geometry[1, x]
            d_bottom = geometry[2, x]
            d_left = geometry[3, x]
            angle = geometry[4, x]
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            box_h = d_top + d_bottom
            box_w = d_left + d_right
            center_x = offset_x + cos_a * d_right + sin_a * d_bottom
            center_y = offset_y - sin_a * d_right + cos_a * d_bottom

            top_left_x = center_x - box_w / 2
            top_left_y = center_y - box_h / 2
            rects.append(
                (int(top_left_x), int(top_left_y), int(box_w), int(box_h))
            )
            confidences.append(float(scores[x]))

    indices = cv2.dnn.NMSBoxes(
        rects, confidences, min_confidence, nms_threshold
    )
    regions: list[TextRegion] = []
    for idx in _flatten(indices):
        x, y, w, h = rects[idx]
        x = int(x * scale_w)
        y = int(y * scale_h)
        w = int(w * scale_w)
        h = int(h * scale_h)
        points = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
        regions.append(TextRegion(points=points, confidence=confidences[idx]))

    regions.sort(key=lambda r: (r.box[1], r.box[0]))
    return merge_regions(regions)


def merge_regions(
    regions: list[TextRegion],
    y_overlap_ratio: float = 0.6,
    max_gap_ratio: float = 4.0,
) -> list[TextRegion]:
    """Merge word-level boxes that belong to the same text line.

    EAST and CRAFT often split a line into several word boxes. This
    post-processing groups boxes whose vertical ranges overlap and whose
    horizontal gap is small, producing line-level regions that OCR
    engines recognize much more accurately.

    - ``y_overlap_ratio``: minimum fraction of the smaller height that
      two boxes' vertical spans must share to be on the same line.
    - ``max_gap_ratio``: boxes are merged when the horizontal gap is at
      most this multiple of the smaller box height.
    """
    if not regions:
        return regions
    ordered = sorted(regions, key=lambda r: (r.box[1], r.box[0]))
    lines: list[list[TextRegion]] = [[ordered[0]]]
    for region in ordered[1:]:
        x, y, w, h = region.box
        last = lines[-1][-1].box
        lx, ly, lw, lh = last
        overlap = max(0, min(y + h, ly + lh) - max(y, ly))
        min_h = max(h, lh, 1)
        gap = x - (lx + lw)
        if (overlap / min_h) >= y_overlap_ratio and gap <= max_gap_ratio * min_h:
            lines[-1].append(region)
        else:
            lines.append([region])

    merged: list[TextRegion] = []
    for line in lines:
        xs0 = min(r.box[0] for r in line)
        ys0 = min(r.box[1] for r in line)
        xs1 = max(r.box[0] + r.box[2] for r in line)
        ys1 = max(r.box[1] + r.box[3] for r in line)
        conf = sum(r.confidence for r in line) / len(line)
        points = np.array(
            [[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]], dtype=np.float32
        )
        merged.append(TextRegion(points=points, confidence=conf))
    merged.sort(key=lambda r: (r.box[1], r.box[0]))
    return merged


def detect_easyocr(image: np.ndarray, reader=None, min_confidence: float = 0.25):
    """Detect text with EasyOCR's built-in detector (CRAFT-based)."""
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "EasyOCR is not installed. Run: pip install -r requirements-easyocr.txt"
        ) from exc
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    results = reader.detect(image, low_text=0.4, text_threshold=0.7)
    # results[0] -> list of polygons (each a 4x2 array), results[1] -> confidences
    regions: list[TextRegion] = []
    polygons, confs = results
    if polygons:
        for poly, conf in zip(polygons, confs):
            if conf < min_confidence:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] != 4:
                continue
            regions.append(TextRegion(points=pts, confidence=float(conf)))
    regions.sort(key=lambda r: (r.box[1], r.box[0]))
    return merge_regions(regions)


def detect(
    image: np.ndarray,
    method: str = "none",
    config: Optional[dict] = None,
    easyocr_reader=None,
) -> list[TextRegion]:
    """Dispatch to the configured detection method."""
    config = config or {}
    if method == "none":
        h, w = image.shape[:2]
        return [TextRegion(points=np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32), confidence=1.0)]
    if method == "east":
        east = config.get("east", {})
        return detect_east(
            image,
            model_path=east.get("model_path", "models/frozen_east_text_detection.pb"),
            min_confidence=east.get("min_confidence", 0.5),
            input_width=int(east.get("width", 320)),
            input_height=int(east.get("height", 320)),
            nms_threshold=float(east.get("nms_threshold", 0.4)),
        )
    if method == "easyocr":
        return detect_easyocr(
            image,
            reader=easyocr_reader,
            min_confidence=config.get("min_confidence", 0.25),
        )
    raise ValueError(f"Unknown detection method: {method!r}")


def crop_regions(
    image: np.ndarray,
    regions: Iterable[TextRegion],
    min_height: int = 40,
    pad: int = 8,
) -> list[np.ndarray]:
    """Extract the axis-aligned crop for each region.

    Small crops (shorter than ``min_height``) are scaled up ~2x so that
    OCR engines receive comfortably-sized text.
    """
    h, w = image.shape[:2]
    crops: list[np.ndarray] = []
    for region in regions:
        x, y, bw, bh = region.box
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = image[y0:y1, x0:x1]
        if crop.size and crop.shape[0] < min_height:
            scale = max(2.0, min_height / crop.shape[0])
            interpolation = cv2.INTER_NEAREST if crop.ndim == 2 else cv2.INTER_CUBIC
            crop = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=interpolation
            )
        crops.append(crop)
    return crops


def _flatten(indices) -> list[int]:
    if indices is None or len(indices) == 0:
        return []
    return [int(i) for i in np.asarray(indices).flatten()]
