"""Integration tests for finger-guided reading mode (US3).

Tests verify the full pipeline: FingerTracker → DetectionEvent → ResponseEngine → AudioResponse.

Also covers the FlecSession-level E2E pipeline added in F-001:
  - process_frame with mocked OCR drives update_ocr → narration
  - Orientation contract (normal vs mirror, confidence delta gate)
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

from flec.models import (
    AudioPriority,
    AudioResponse,
    DetectionEvent,
    DetectionType,
    FingerTrackingState,
    Mode,
    ReadingIntent,
)
from flec.perception.finger_tracker import FingerTracker
from flec.engine.response_engine import ResponseEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_finger_event(
    intent: ReadingIntent,
    nearest_text: str | None = None,
    is_illustration: bool = False,
) -> DetectionEvent:
    """Build a FINGER_TIP DetectionEvent with the given intent and context."""
    metadata: dict = {
        "intent": intent,
        "nearest_text": nearest_text,
        "is_illustration": is_illustration,
    }
    label = nearest_text or ("illustration" if is_illustration else "finger")
    return DetectionEvent(
        type=DetectionType.FINGER,
        label=label,
        confidence=0.95,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# T040-I1: Finger slows near word → that word read aloud
# ---------------------------------------------------------------------------


class TestFingerSlowsNearWord:
    """Integration: slow finger near text triggers audio narration of that word."""

    def test_slow_finger_near_word_triggers_narration(self) -> None:
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        event = _make_finger_event(intent=ReadingIntent.READING, nearest_text="cat")
        engine.on_event(event)

        assert not audio_queue.empty(), (
            "AudioResponse must be queued when finger slows near text"
        )
        response: AudioResponse = audio_queue.get_nowait()
        assert "cat" in response.text.lower(), (
            f"AudioResponse must contain the word 'cat'; got: {response.text!r}"
        )
        assert response.priority == AudioPriority.NORMAL, (
            "Reading narration must have NORMAL priority"
        )

    def test_single_word_not_repeated_immediately(self) -> None:
        """Same word should not trigger duplicate audio on consecutive frames."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        event = _make_finger_event(intent=ReadingIntent.READING, nearest_text="dog")
        engine.on_event(event)
        engine.on_event(event)  # Same event again

        # Only one response should be queued for the same word
        count = audio_queue.qsize()
        assert count == 1, (
            f"Same word must not be narrated twice consecutively; got {count} responses"
        )


# ---------------------------------------------------------------------------
# T040-I2: Finger moves L→R across multiple words → words read in sequence
# ---------------------------------------------------------------------------


class TestFingerMovesAcrossWords:
    """Integration: moving finger across multiple words reads each in sequence."""

    def test_multiple_words_read_in_sequence(self) -> None:
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        words = ["the", "big", "red", "dog"]
        for word in words:
            event = _make_finger_event(intent=ReadingIntent.READING, nearest_text=word)
            engine.on_event(event)

        responses: list[str] = []
        while not audio_queue.empty():
            responses.append(audio_queue.get_nowait().text.lower())

        assert len(responses) == len(words), (
            f"Expected {len(words)} audio responses for {len(words)} distinct words; "
            f"got {len(responses)}"
        )
        for word in words:
            assert any(word in r for r in responses), (
                f"Word '{word}' must appear in audio responses"
            )


# ---------------------------------------------------------------------------
# T040-I3: Fast movement → re-anchors (no mid-word reading while scanning)
# ---------------------------------------------------------------------------


class TestFastMovementReanchors:
    """Integration: SCANNING intent produces no audio narration."""

    def test_scanning_intent_produces_no_audio(self) -> None:
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        event = _make_finger_event(
            intent=ReadingIntent.SCANNING, nearest_text="hello"
        )
        engine.on_event(event)

        assert audio_queue.empty(), (
            "No AudioResponse must be queued when finger intent is SCANNING"
        )

    def test_idle_intent_produces_no_audio(self) -> None:
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        event = _make_finger_event(intent=ReadingIntent.IDLE, nearest_text=None)
        engine.on_event(event)

        assert audio_queue.empty(), (
            "No AudioResponse must be queued when intent is IDLE"
        )

    def test_reanchor_after_fast_then_slow(self) -> None:
        """After fast movement (SCANNING) followed by slow (READING), narration resumes."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        # Fast movement — no audio
        scan_event = _make_finger_event(
            intent=ReadingIntent.SCANNING, nearest_text="skip"
        )
        engine.on_event(scan_event)
        assert audio_queue.empty()

        # Slow down near new word — audio should resume
        read_event = _make_finger_event(
            intent=ReadingIntent.READING, nearest_text="sun"
        )
        engine.on_event(read_event)
        assert not audio_queue.empty(), (
            "After re-anchoring (READING intent), narration must resume"
        )
        response = audio_queue.get_nowait()
        assert "sun" in response.text.lower()


# ---------------------------------------------------------------------------
# T040-I4: Finger near illustration (no text) → illustration described
# ---------------------------------------------------------------------------


class TestIllustrationDescription:
    """Integration: finger near illustration triggers description, not word narration."""

    def test_illustration_triggers_description(self) -> None:
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        event = _make_finger_event(
            intent=ReadingIntent.READING,
            nearest_text=None,
            is_illustration=True,
        )
        # Inject a pending illustration description via the engine's illustration method
        engine.set_pending_illustration("a little yellow duck")
        engine.on_event(event)

        assert not audio_queue.empty(), (
            "AudioResponse must be queued when finger slows near illustration"
        )
        response = audio_queue.get_nowait()
        assert "duck" in response.text.lower(), (
            f"Description must reference the illustration; got: {response.text!r}"
        )

    def test_no_audio_when_no_illustration_and_no_text(self) -> None:
        """If no text and no illustration description available, no audio plays."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        event = _make_finger_event(
            intent=ReadingIntent.READING,
            nearest_text=None,
            is_illustration=False,
        )
        engine.on_event(event)

        assert audio_queue.empty(), (
            "No audio when finger is near empty area (no text, no illustration)"
        )


# ---------------------------------------------------------------------------
# T040-I5: FingerTracker pipeline — velocity-based intent feeds event correctly
# ---------------------------------------------------------------------------


class TestFingerTrackerPipeline:
    """Integration: FingerTracker simulate_finger → correct DetectionEvent → correct response."""

    def test_tracker_low_velocity_produces_reading_intent(self) -> None:
        tracker = FingerTracker()
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        state = tracker.current_state
        assert state.intent == ReadingIntent.READING

    def test_tracker_high_velocity_produces_scanning_intent(self) -> None:
        tracker = FingerTracker()
        threshold = tracker.velocity_threshold
        tracker.simulate_finger(position=(0.9, 0.5), velocity=threshold * 5)
        state = tracker.current_state
        assert state.intent == ReadingIntent.SCANNING

    def test_tracker_reset_then_reading_mode_narrates(self) -> None:
        """After tracker reset, re-entering READING state still produces narration."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.READING)

        tracker = FingerTracker()
        tracker.reset()

        # Drive back to reading
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        tracker.update_ocr(text_regions=["sky"])

        state = tracker.current_state
        event = _make_finger_event(
            intent=state.intent, nearest_text=state.nearest_text
        )
        engine.on_event(event)

        assert not audio_queue.empty(), (
            "After tracker reset and re-entry to READING, narration must work"
        )


# ---------------------------------------------------------------------------
# F-001 E2E: FlecSession.process_frame — OCR wired pipeline
# ---------------------------------------------------------------------------


def _blank_frame(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


class _FakeFingerState:
    def __init__(self, detected=True, velocity=0.001, intent_name="READING", nearest_text=None):
        self.detected = detected
        self.velocity = velocity
        self.intent = ReadingIntent[intent_name]
        self.nearest_text = nearest_text
        self.position_x = 0.5
        self.position_y = 0.5


class TestFlecSessionReadingPipeline:
    """E2E: FlecSession.process_frame drives full reading pipeline via mocked OCR."""

    def test_settled_finger_with_confident_word_wires_update_ocr(self, monkeypatch):
        """Settled finger + confident OCR word → update_ocr called with the word."""
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            monkeypatch.setattr(
                session._finger_tracker, "update",
                lambda frame: _FakeFingerState(detected=True, velocity=0.001, intent_name="READING"),
            )
            monkeypatch.setattr(
                session._ocr_reader, "read_region",
                lambda frame: ("sun", 0.88),
            )

            ocr_calls = []
            original = session._finger_tracker.update_ocr

            def capture(text_regions):
                ocr_calls.append(text_regions)
                original(text_regions)

            monkeypatch.setattr(session._finger_tracker, "update_ocr", capture)

            session.process_frame(_blank_frame())

            assert ocr_calls == [["sun"]], (
                f"Expected update_ocr([\"sun\"]); got: {ocr_calls}"
            )
        finally:
            session.shutdown()

    def test_fast_sweep_produces_no_narration(self, monkeypatch):
        """Fast finger sweep → no OCR fires → no narration."""
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            monkeypatch.setattr(
                session._finger_tracker, "update",
                lambda frame: _FakeFingerState(detected=True, velocity=0.9, intent_name="SCANNING"),
            )

            read_calls = []
            monkeypatch.setattr(
                session._ocr_reader, "read_region",
                lambda frame: read_calls.append(1) or ("word", 0.95),
            )

            session.process_frame(_blank_frame())

            assert read_calls == [], "OCR must not run during fast sweep"
        finally:
            session.shutdown()

    def test_no_confidence_falls_back_to_illustration(self, monkeypatch):
        """No confident OCR word → illustration fallback fires."""
        from flec.main import FlecSession

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

            describe_calls = []
            monkeypatch.setattr(
                session._illustration_describer, "describe",
                lambda frame: describe_calls.append(1) or "",
            )

            session.process_frame(_blank_frame())

            assert len(describe_calls) == 1, "IllustrationDescriber must be called as fallback"
        finally:
            session.shutdown()

    def test_word_change_clears_pending_audio(self, monkeypatch):
        """Moving from one word to another clears pending TTS narration."""
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            state = _FakeFingerState(
                detected=True, velocity=0.001, intent_name="READING", nearest_text="dog"
            )
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

            assert len(flush_calls) == 1, "Pending audio must be flushed on word change"
        finally:
            session.shutdown()
