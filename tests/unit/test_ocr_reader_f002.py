"""Tests for OCRReader.shutdown() and its wiring into FlecSession (flec-pxt)."""

from __future__ import annotations


def test_ocr_reader_has_shutdown():
    from flec.reading.ocr_reader import OCRReader
    r = OCRReader()
    r.shutdown()  # must not raise
    assert True


def test_flec_session_shutdown_calls_ocr_shutdown(monkeypatch):
    # Patch _ocr_reader.shutdown to verify it gets called
    import unittest.mock as mock
    monkeypatch.setenv("FLEC_READING_WEAR_OVERRIDE", "0")
    from flec.main import FlecSession
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    mock_shutdown = mock.Mock()
    session._ocr_reader.shutdown = mock_shutdown
    session.shutdown()
    mock_shutdown.assert_called_once()
