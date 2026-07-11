"""Unit tests for the Reading-mode OCR worker and OCRReader confidence path.

Covers (F-001 reading-mode-end-to-end):
- OCRReader.read_region → (text, mean_confidence) for a cropped ROI  [flec-7al]
- should_run_ocr settle-gate predicate                              [flec-7al]
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------
# OCRReader.read_region — text + confidence for the fingertip crop
# --------------------------------------------------------------------------


class _FakeReader:
    def __init__(self, results):
        self._results = results

    def readtext(self, frame, detail=1, paragraph=False):
        return self._results


def _frame():
    return np.full((60, 120, 3), 255, np.uint8)


def test_read_region_returns_text_and_confidence(monkeypatch):
    from flec.reading.ocr_reader import OCRReader

    reader = OCRReader()
    monkeypatch.setattr(
        reader, "_get_reader",
        lambda: _FakeReader([([[0, 0], [1, 0], [1, 1], [0, 1]], "cat", 0.92)]),
    )
    text, conf = reader.read_region(_frame())
    assert text == "cat"
    assert conf == pytest.approx(0.92, abs=1e-3)


def test_read_region_empty_when_no_detection(monkeypatch):
    from flec.reading.ocr_reader import OCRReader

    reader = OCRReader()
    monkeypatch.setattr(reader, "_get_reader", lambda: _FakeReader([]))
    text, conf = reader.read_region(_frame())
    assert text == ""
    assert conf == 0.0


def test_read_region_drops_low_confidence(monkeypatch):
    from flec.reading.ocr_reader import OCRReader

    reader = OCRReader()
    monkeypatch.setattr(
        reader, "_get_reader",
        lambda: _FakeReader([([[0, 0], [1, 0], [1, 1], [0, 1]], "noise", 0.1)]),
    )
    text, conf = reader.read_region(_frame())
    assert text == ""
    assert conf == 0.0


def test_read_region_never_raises_on_bad_frame(monkeypatch):
    from flec.reading.ocr_reader import OCRReader

    reader = OCRReader()
    text, conf = reader.read_region(None)  # degenerate
    assert (text, conf) == ("", 0.0)


# --------------------------------------------------------------------------
# should_run_ocr — settle-gate predicate (decision D1)
# --------------------------------------------------------------------------


def test_should_run_ocr_fires_when_detected_and_still():
    from flec.reading.ocr_worker import should_run_ocr

    assert should_run_ocr(detected=True, velocity=0.001, settle_threshold=0.02) is True


def test_should_run_ocr_silent_when_moving():
    from flec.reading.ocr_worker import should_run_ocr

    assert should_run_ocr(detected=True, velocity=0.5, settle_threshold=0.02) is False


def test_should_run_ocr_silent_when_no_finger():
    from flec.reading.ocr_worker import should_run_ocr

    assert should_run_ocr(detected=False, velocity=0.0, settle_threshold=0.02) is False
