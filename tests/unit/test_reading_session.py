"""FlecSession Reading-mode wiring tests (F-001).

- Dev wear-state override so Reading activates without a wear sensor  [flec-d23]
- OCR wiring: settle-gated OCR drives update_ocr in process_frame    [flec-8vi]
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from flec.main import FlecSession
from flec.models import WearState


def test_dev_session_treats_webcam_as_on_head():
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        assert session.response_engine.wear_state == WearState.ON_HEAD
    finally:
        session.shutdown()


def test_wear_override_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FLEC_READING_WEAR_OVERRIDE", "0")
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        assert session.response_engine.wear_state == WearState.OFF_HEAD
    finally:
        session.shutdown()


# ---------------------------------------------------------------------------
# flec-8vi: OCR wiring — process_frame drives update_ocr when settled
# ---------------------------------------------------------------------------


def _blank_frame(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


class _FakeFingerState:
    """Minimal FingerTrackingState stand-in."""
    def __init__(self, detected=True, velocity=0.001, intent_name="READING"):
        from flec.models import ReadingIntent
        self.detected = detected
        self.velocity = velocity
        self.intent = ReadingIntent[intent_name]
        self.nearest_text = None
        self.position_x = 0.5
        self.position_y = 0.5


def test_process_frame_calls_update_ocr_with_confident_word(monkeypatch):
    """Given finger settled + confident OCR word → update_ocr is called with [word]."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        # Simulate a settled finger
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        # Simulate OCR returning "cat" with high confidence
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("cat", 0.9),
        )

        ocr_calls = []
        original_update_ocr = session._finger_tracker.update_ocr

        def capture_update_ocr(text_regions):
            ocr_calls.append(text_regions)
            original_update_ocr(text_regions)

        monkeypatch.setattr(session._finger_tracker, "update_ocr", capture_update_ocr)

        session.process_frame(_blank_frame())

        assert len(ocr_calls) == 1
        assert ocr_calls[0] == ["cat"]
    finally:
        session.shutdown()


def test_process_frame_does_not_call_update_ocr_when_confidence_low(monkeypatch):
    """Given finger settled + low confidence OCR result → update_ocr is NOT called."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        # Low confidence — below the silence gate
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("noise", 0.2),
        )

        ocr_calls = []
        monkeypatch.setattr(
            session._finger_tracker, "update_ocr",
            lambda text_regions: ocr_calls.append(text_regions),
        )

        session.process_frame(_blank_frame())

        assert ocr_calls == []
    finally:
        session.shutdown()


def test_process_frame_does_not_call_update_ocr_when_moving(monkeypatch):
    """Given fast finger sweep (velocity high) → OCR does not fire at all."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.9, intent_name="SCANNING"),
        )

        ocr_calls = []
        read_calls = []
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: read_calls.append(1) or ("word", 0.95),
        )
        monkeypatch.setattr(
            session._finger_tracker, "update_ocr",
            lambda text_regions: ocr_calls.append(text_regions),
        )

        session.process_frame(_blank_frame())

        # OCR should not even run when finger is moving fast
        assert read_calls == []
        assert ocr_calls == []
    finally:
        session.shutdown()


def test_process_frame_does_not_call_update_ocr_when_no_finger(monkeypatch):
    """Given no finger detected → OCR does not fire."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=False, velocity=0.0, intent_name="IDLE"),
        )

        read_calls = []
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: read_calls.append(1) or ("word", 0.95),
        )

        session.process_frame(_blank_frame())
        assert read_calls == []
    finally:
        session.shutdown()


def test_process_frame_legacy_ocr_result_param_still_works(monkeypatch):
    """Passing ocr_result explicitly still calls update_ocr (backward compat)."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )

        ocr_calls = []
        original_update_ocr = session._finger_tracker.update_ocr

        def capture_update_ocr(text_regions):
            ocr_calls.append(text_regions)
            original_update_ocr(text_regions)

        monkeypatch.setattr(session._finger_tracker, "update_ocr", capture_update_ocr)

        # Old API: pass ocr_result directly
        session.process_frame(_blank_frame(), ocr_result=["hello"])

        assert ["hello"] in ocr_calls
    finally:
        session.shutdown()


# ---------------------------------------------------------------------------
# flec-7dh: Illustration fallback — no confident word → describe the region
# ---------------------------------------------------------------------------


def test_process_frame_uses_illustration_fallback_when_no_confident_word(monkeypatch):
    """Settled finger + no confident OCR word → IllustrationDescriber.describe fires."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        # OCR returns nothing confident
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("", 0.0),
        )

        describe_calls = []
        monkeypatch.setattr(
            session._illustration_describer, "describe",
            lambda frame: describe_calls.append(1) or "a red duck",
        )

        session.process_frame(_blank_frame())

        assert len(describe_calls) == 1
    finally:
        session.shutdown()


def test_process_frame_illustration_description_reaches_response_engine(monkeypatch):
    """Illustration description is set as pending on the response engine."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("", 0.0),
        )
        monkeypatch.setattr(
            session._illustration_describer, "describe",
            lambda frame: "a blue cat",
        )

        session.process_frame(_blank_frame())

        assert session._response_engine._pending_illustration == "a blue cat"
    finally:
        session.shutdown()


def test_process_frame_no_illustration_when_describer_unavailable(monkeypatch):
    """When IllustrationDescriber returns empty, response engine gets nothing."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("", 0.0),
        )
        monkeypatch.setattr(
            session._illustration_describer, "describe",
            lambda frame: "",  # unavailable
        )

        session.process_frame(_blank_frame())

        assert session._response_engine._pending_illustration is None
    finally:
        session.shutdown()


def test_process_frame_illustration_not_called_when_ocr_succeeds(monkeypatch):
    """When OCR finds a confident word, IllustrationDescriber is not called."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("cat", 0.9),
        )

        describe_calls = []
        monkeypatch.setattr(
            session._illustration_describer, "describe",
            lambda frame: describe_calls.append(1) or "some picture",
        )

        session.process_frame(_blank_frame())

        assert describe_calls == []
    finally:
        session.shutdown()


# ---------------------------------------------------------------------------
# flec-jjb: Word-change flush — new word clears pending audio
# ---------------------------------------------------------------------------


def test_process_frame_flushes_pending_audio_when_word_changes(monkeypatch):
    """When OCR detects a word change, pending TTS narration is cleared (AC-4)."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        # Simulate finger in READING with "dog" as current nearest_text
        state = _FakeFingerState(detected=True, velocity=0.001, intent_name="READING")
        state.nearest_text = "dog"
        monkeypatch.setattr(session._finger_tracker, "update", lambda frame: state)

        # OCR now finds "cat" — a different word
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("cat", 0.9),
        )

        flush_calls = []
        monkeypatch.setattr(
            session._tts_engine, "clear_pending",
            lambda: flush_calls.append(1),
        )

        session.process_frame(_blank_frame())

        assert len(flush_calls) == 1
    finally:
        session.shutdown()


def test_process_frame_does_not_flush_when_same_word(monkeypatch):
    """When OCR finds the same word as current nearest_text, no flush occurs."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        state = _FakeFingerState(detected=True, velocity=0.001, intent_name="READING")
        state.nearest_text = "cat"
        monkeypatch.setattr(session._finger_tracker, "update", lambda frame: state)

        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("cat", 0.9),
        )

        flush_calls = []
        monkeypatch.setattr(
            session._tts_engine, "clear_pending",
            lambda: flush_calls.append(1),
        )

        session.process_frame(_blank_frame())

        assert flush_calls == []
    finally:
        session.shutdown()


def test_process_frame_does_not_flush_when_no_previous_word(monkeypatch):
    """When nearest_text is None (first read), no flush occurs."""
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        # nearest_text is None (default) — first recognition
        monkeypatch.setattr(
            session._finger_tracker, "update",
            lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
        )
        monkeypatch.setattr(
            session._ocr_reader, "read_region",
            lambda frame: ("hello", 0.85),
        )

        flush_calls = []
        monkeypatch.setattr(
            session._tts_engine, "clear_pending",
            lambda: flush_calls.append(1),
        )

        session.process_frame(_blank_frame())

        assert flush_calls == []
    finally:
        session.shutdown()
