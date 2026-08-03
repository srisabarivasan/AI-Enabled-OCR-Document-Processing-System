"""Validation: measure OCR quality against ground-truth text.

Metrics implemented:
  - Levenshtein distance (edit distance).
  - Character Error Rate (CER): edits per character.
  - Word Error Rate (WER): edits per word.
  - Character accuracy: ``1 - CER``.
  - Exact match: whether prediction equals ground truth.

Text is normalized before comparison (lowercase, whitespace collapse,
punctuation stripping optionally) so that formatting differences do not
count as errors.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalize(text: str, strip_punctuation: bool = True) -> str:
    """Normalize text for comparison: NFC + lowercase + collapse spaces."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.lower()
    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic-programming edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(
                min(
                    prev[j] + 1,  # deletion
                    cur[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = cur
    return prev[-1]


def cer(prediction: str, reference: str, strip_punctuation: bool = True) -> float:
    """Character error rate = edits / reference characters."""
    ref = normalize(reference, strip_punctuation)
    pred = normalize(prediction, strip_punctuation)
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / max(len(ref), 1)


def wer(prediction: str, reference: str, strip_punctuation: bool = True) -> float:
    """Word error rate = word edits / reference words.

    Computed via edit distance on the word sequences, not on the
    concatenated character string.
    """
    ref = normalize(reference, strip_punctuation).split()
    pred = normalize(prediction, strip_punctuation).split()
    if not ref:
        return 0.0 if not pred else 1.0
    distance = _word_edit_distance(pred, ref)
    return distance / len(ref)


def _word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over word tokens."""
    prev = list(range(len(b) + 1))
    for wa in a:
        cur = [prev[0] + 1]
        for j, wb in enumerate(b, 1):
            cost = 0 if wa == wb else 1
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = cur
    return prev[-1]


@dataclass
class ValidationReport:
    """Aggregated validation result."""

    cer: float
    wer: float
    char_accuracy: float
    exact_match: bool
    reference: str
    prediction: str

    def as_dict(self) -> dict:
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "char_accuracy": round(self.char_accuracy, 4),
            "exact_match": self.exact_match,
            "reference": self.reference,
            "prediction": self.prediction,
        }


def validate(prediction: str, reference: str, strip_punctuation: bool = True) -> ValidationReport:
    """Compare a predicted string against ground truth."""
    c = cer(prediction, reference, strip_punctuation)
    w = wer(prediction, reference, strip_punctuation)
    return ValidationReport(
        cer=c,
        wer=w,
        char_accuracy=1.0 - c,
        exact_match=normalize(prediction, strip_punctuation)
        == normalize(reference, strip_punctuation),
        reference=reference,
        prediction=prediction,
    )
