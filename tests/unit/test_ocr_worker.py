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


# --------------------------------------------------------------------------
# crop_around_fingertip — bounded ROI around the fingertip  [flec-0ak]
# --------------------------------------------------------------------------


def test_crop_around_fingertip_is_bounded_subregion():
    from flec.reading.ocr_worker import crop_around_fingertip

    frame = np.zeros((100, 200, 3), np.uint8)
    crop = crop_around_fingertip(frame, 0.5, 0.5, frac=0.4)
    ch, cw = crop.shape[:2]
    assert 0 < cw <= 200 and 0 < ch <= 100
    assert cw <= int(200 * 0.4) + 2 and ch <= int(100 * 0.4) + 2


def test_crop_around_fingertip_clamps_at_edges():
    from flec.reading.ocr_worker import crop_around_fingertip

    frame = np.zeros((100, 200, 3), np.uint8)
    crop = crop_around_fingertip(frame, 0.0, 0.0, frac=0.4)  # top-left corner
    assert crop.size > 0  # never empty / out of bounds


# --------------------------------------------------------------------------
# resolve_orientation — normal vs mirror, confidence-delta gate  [flec-0ak]
# --------------------------------------------------------------------------


def _crop():
    return np.zeros((30, 60, 3), np.uint8)


def _reader(seq):
    it = iter(seq)
    return lambda img: next(it)


def test_resolve_picks_normal_when_higher_confidence():
    from flec.reading.ocr_worker import resolve_orientation

    text, conf, orient = resolve_orientation(
        _crop(), _reader([("cat", 0.9), ("tac", 0.2)]),
        cached=None, conf_gate=0.4, delta_gate=0.1,
    )
    assert (text, orient) == ("cat", "normal")


def test_resolve_picks_mirror_when_higher_confidence():
    from flec.reading.ocr_worker import resolve_orientation

    text, conf, orient = resolve_orientation(
        _crop(), _reader([("tac", 0.2), ("dog", 0.85)]),
        cached=None, conf_gate=0.4, delta_gate=0.1,
    )
    assert (text, orient) == ("dog", "mirror")


def test_resolve_silent_when_both_low_confidence():
    from flec.reading.ocr_worker import resolve_orientation

    text, conf, orient = resolve_orientation(
        _crop(), _reader([("x", 0.2), ("y", 0.25)]),
        cached=None, conf_gate=0.4, delta_gate=0.1,
    )
    assert text == "" and orient == ""


def test_resolve_silent_when_delta_too_small():
    from flec.reading.ocr_worker import resolve_orientation

    # both confident but ambiguous which orientation → silence (D2)
    text, conf, orient = resolve_orientation(
        _crop(), _reader([("cat", 0.82), ("tac", 0.80)]),
        cached=None, conf_gate=0.4, delta_gate=0.1,
    )
    assert text == "" and orient == ""


def test_resolve_speaks_when_both_orientations_agree():
    from flec.reading.ocr_worker import resolve_orientation

    # identical text both ways (e.g. symmetric) → orientation irrelevant, speak
    text, conf, orient = resolve_orientation(
        _crop(), _reader([("mom", 0.82), ("mom", 0.80)]),
        cached=None, conf_gate=0.4, delta_gate=0.1,
    )
    assert text == "mom"


def test_resolve_uses_cache_single_probe():
    from flec.reading.ocr_worker import resolve_orientation

    calls = {"n": 0}

    def one_probe(img):
        calls["n"] += 1
        return ("book", 0.9)

    text, conf, orient = resolve_orientation(
        _crop(), one_probe, cached="normal", conf_gate=0.4, delta_gate=0.1,
    )
    assert (text, orient) == ("book", "normal")
    assert calls["n"] == 1  # cached orientation → single OCR pass


def test_mirror_x_reflects_coordinate():
    from flec.reading.ocr_worker import mirror_x

    assert mirror_x(0.2) == pytest.approx(0.8)
    assert mirror_x(0.5) == pytest.approx(0.5)
