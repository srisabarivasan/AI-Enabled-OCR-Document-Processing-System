import pytest

from src.validation import cer, levenshtein, normalize, validate, wer


def test_normalize():
    assert normalize("  Hello,  World!  ") == "hello world"
    assert normalize("Caf\u0301e") == "cafe"  # NFC normalization


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("same", "same") == 0


def test_cer():
    assert cer("hello", "hello") == 0.0
    assert cer("helo", "hello") == pytest.approx(1 / 5)
    assert cer("xyz", "abc") == 1.0


def test_wer():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert wer("the cat", "the cat sat") == pytest.approx(1 / 3)
    assert wer("the dog ran", "the cat sat") == pytest.approx(2 / 3)


def test_validate_exact_match():
    report = validate("Invoice 123", "invoice 123")
    assert report.exact_match
    assert report.cer == 0.0
    assert report.char_accuracy == 1.0


def test_validate_imperfect():
    report = validate("Invoice 12", "Invoice 123")
    assert not report.exact_match
    assert report.cer > 0.0
