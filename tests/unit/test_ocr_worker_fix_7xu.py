"""Regression tests for flec-7xu: resolve_orientation cache fast-path early return.

Before the fix, a low-confidence cache hit would fall through to the full dual-probe,
causing 3 OCR calls instead of the intended 1. These tests pin the corrected behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from flec.reading.ocr_worker import resolve_orientation

CROP = np.zeros((10, 10, 3), dtype=np.uint8)
CONF_GATE = 0.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_reader(*results):
    """Return a callable that yields successive (text, conf) pairs and counts calls."""
    calls = []
    result_list = list(results)

    def reader(arr):
        calls.append(1)
        return result_list.pop(0)

    reader.call_count = calls
    return reader


# ---------------------------------------------------------------------------
# cached="normal" — low confidence: exactly 1 call, early return
# ---------------------------------------------------------------------------

def test_cached_normal_low_conf_single_call():
    """When cached=normal and conf < conf_gate, read_region is called exactly once."""
    reader = make_reader(("dog", 0.2))
    result = resolve_orientation(CROP, reader, cached="normal", conf_gate=CONF_GATE)
    assert len(reader.call_count) == 1, "Expected exactly 1 OCR call, got more"


def test_cached_normal_low_conf_returns_empty():
    """When cached=normal and conf < conf_gate, returns ('', conf, '')."""
    reader = make_reader(("dog", 0.2))
    text, conf, orient = resolve_orientation(CROP, reader, cached="normal", conf_gate=CONF_GATE)
    assert text == ""
    assert conf == pytest.approx(0.2)
    assert orient == ""


# ---------------------------------------------------------------------------
# cached="mirror" — low confidence: exactly 1 call, early return
# ---------------------------------------------------------------------------

def test_cached_mirror_low_conf_single_call():
    """When cached=mirror and conf < conf_gate, read_region is called exactly once."""
    reader = make_reader(("god", 0.15))
    result = resolve_orientation(CROP, reader, cached="mirror", conf_gate=CONF_GATE)
    assert len(reader.call_count) == 1, "Expected exactly 1 OCR call, got more"


def test_cached_mirror_low_conf_returns_empty():
    """When cached=mirror and conf < conf_gate, returns ('', conf, '')."""
    reader = make_reader(("god", 0.15))
    text, conf, orient = resolve_orientation(CROP, reader, cached="mirror", conf_gate=CONF_GATE)
    assert text == ""
    assert conf == pytest.approx(0.15)
    assert orient == ""


# ---------------------------------------------------------------------------
# cached="normal" — high confidence: returns (text, conf, "normal")
# ---------------------------------------------------------------------------

def test_cached_normal_high_conf_returns_normal():
    """When cached=normal and conf >= conf_gate, returns (text, conf, 'normal')."""
    reader = make_reader(("cat", 0.9))
    text, conf, orient = resolve_orientation(CROP, reader, cached="normal", conf_gate=CONF_GATE)
    assert text == "cat"
    assert conf == pytest.approx(0.9)
    assert orient == "normal"


def test_cached_normal_high_conf_single_call():
    """When cached=normal and conf >= conf_gate, read_region is called exactly once."""
    reader = make_reader(("cat", 0.9))
    resolve_orientation(CROP, reader, cached="normal", conf_gate=CONF_GATE)
    assert len(reader.call_count) == 1


# ---------------------------------------------------------------------------
# cached="mirror" — high confidence: returns (text, conf, "mirror")
# ---------------------------------------------------------------------------

def test_cached_mirror_high_conf_returns_mirror():
    """When cached=mirror and conf >= conf_gate, returns (text, conf, 'mirror')."""
    reader = make_reader(("tac", 0.85))
    text, conf, orient = resolve_orientation(CROP, reader, cached="mirror", conf_gate=CONF_GATE)
    assert text == "tac"
    assert conf == pytest.approx(0.85)
    assert orient == "mirror"


def test_cached_mirror_high_conf_single_call():
    """When cached=mirror and conf >= conf_gate, read_region is called exactly once."""
    reader = make_reader(("tac", 0.85))
    resolve_orientation(CROP, reader, cached="mirror", conf_gate=CONF_GATE)
    assert len(reader.call_count) == 1


# ---------------------------------------------------------------------------
# cached=None — no cache: full dual-probe (2 calls)
# ---------------------------------------------------------------------------

def test_no_cache_dual_probe_call_count():
    """When cached=None, both normal and mirror are probed (2 read_region calls)."""
    # Normal wins with high conf and good delta
    reader = make_reader(("cat", 0.85), ("tac", 0.3))
    resolve_orientation(CROP, reader, cached=None, conf_gate=CONF_GATE, delta_gate=0.1)
    assert len(reader.call_count) == 2, "Expected 2 OCR calls for uncached dual-probe"


def test_no_cache_best_orientation_returned():
    """When cached=None and normal wins, returns (text, conf, 'normal')."""
    reader = make_reader(("cat", 0.85), ("tac", 0.3))
    text, conf, orient = resolve_orientation(
        CROP, reader, cached=None, conf_gate=CONF_GATE, delta_gate=0.1
    )
    assert text == "cat"
    assert orient == "normal"
