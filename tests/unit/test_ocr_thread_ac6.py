"""AC-6 tests: OCR runs off the perception thread, timeout guard releases main thread.

These tests verify:
- OCR work is submitted to a background ThreadPoolExecutor (not run inline)
- The timeout guard in read_page fires when OCR is slow, returning "" promptly
- read_page never raises — it returns "" on timeout
- OCRReader._executor is a ThreadPoolExecutor (architectural assertion)
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from flec.reading.ocr_reader import OCRReader


# ---------------------------------------------------------------------------
# Test 1: OCR runs in a background thread, not the calling thread
# ---------------------------------------------------------------------------

def test_ocr_runs_off_thread():
    """OCR must execute in a background thread, not the caller's thread."""
    reader = OCRReader()
    calling_thread = threading.current_thread()
    ocr_thread_id = None

    def _capture_thread(*args, **kwargs):
        nonlocal ocr_thread_id
        ocr_thread_id = threading.current_thread().ident
        return ""

    reader._run_ocr = _capture_thread  # monkey-patch
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    reader.read_page(frame)

    assert ocr_thread_id is not None, "_capture_thread was never called"
    assert ocr_thread_id != calling_thread.ident, (
        "OCR must not run on the calling thread — it ran on ident "
        f"{ocr_thread_id} which matches the caller {calling_thread.ident}"
    )


# ---------------------------------------------------------------------------
# Test 2: Timeout guard releases the thread within budget
# ---------------------------------------------------------------------------

def test_ocr_timeout_releases_thread():
    """When OCR exceeds timeout, read_page must return within the budget (not hang)."""
    reader = OCRReader()

    def _slow_ocr(frame):
        time.sleep(5)  # simulate very slow OCR — much longer than test timeout
        return "never"

    reader._run_ocr = _slow_ocr

    import flec.reading.ocr_reader as ocr_module
    original_timeout = ocr_module._OCR_TIMEOUT_SECS
    ocr_module._OCR_TIMEOUT_SECS = 0.1  # 100 ms timeout for test speed

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    t0 = time.monotonic()
    result = reader.read_page(frame)
    elapsed = time.monotonic() - t0

    ocr_module._OCR_TIMEOUT_SECS = original_timeout  # restore

    assert result == "", f"Expected empty string on timeout, got: {result!r}"
    assert elapsed < 1.0, (
        f"read_page took {elapsed:.2f}s — timeout guard did not fire within budget"
    )


# ---------------------------------------------------------------------------
# Test 3: read_page returns empty string (not raises) on timeout
# ---------------------------------------------------------------------------

def test_ocr_timeout_returns_empty_string_not_raises():
    """read_page must return '' on timeout — never raise, never return None."""
    reader = OCRReader()

    def _slow_ocr(frame):
        time.sleep(5)
        return "never"

    reader._run_ocr = _slow_ocr

    import flec.reading.ocr_reader as ocr_module
    original_timeout = ocr_module._OCR_TIMEOUT_SECS
    ocr_module._OCR_TIMEOUT_SECS = 0.1  # 100 ms

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        result = reader.read_page(frame)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"read_page raised {type(exc).__name__}: {exc} — must never raise")
    finally:
        ocr_module._OCR_TIMEOUT_SECS = original_timeout

    assert result == "", (
        f"read_page must return '' on timeout, got {result!r}"
    )
    assert isinstance(result, str), (
        f"read_page must return str, got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 4: OCRReader._executor is a ThreadPoolExecutor (architectural assertion)
# ---------------------------------------------------------------------------

def test_ocr_uses_thread_pool():
    """OCRReader must back its async dispatch with a ThreadPoolExecutor."""
    reader = OCRReader()
    assert isinstance(reader._executor, ThreadPoolExecutor), (
        f"Expected ThreadPoolExecutor, got {type(reader._executor).__name__}"
    )
