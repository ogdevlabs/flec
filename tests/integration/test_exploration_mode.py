"""Integration tests for US1: Exploration Mode — Shape & Color Discovery.

Verifies end-to-end: shape enters frame → audio narration within 2s.
Tests the full pipeline: ShapeColorDetector → ResponseEngine → TTSEngine.

All audio feedback must be audio-complete (no visual-only paths).
No exceptions on blank frames.
Each detected shape/color is narrated individually.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Generator
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np
import pytest

from flec.models import (
    AudioPriority,
    AudioResponse,
    DetectionEvent,
    DetectionType,
    Mode,
    BoundingBox,
)
from flec.engine.response_engine import ResponseEngine
from flec.perception.shape_color_detector import ShapeColorDetector
from flec.audio.responses import narrate_detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_red_circle_frame() -> np.ndarray:
    """Return a BGR frame with a large red circle on white background."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 100, (0, 0, 220), -1)  # Red (BGR)
    return frame


def make_blue_square_frame() -> np.ndarray:
    """Return a BGR frame with a blue square on white background."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (220, 140), (420, 340), (220, 0, 0), -1)  # Blue (BGR)
    return frame


def make_multi_shape_frame() -> np.ndarray:
    """Return a BGR frame with two distinct shapes in separate regions."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    # Red circle on left
    cv2.circle(frame, (160, 240), 80, (0, 0, 200), -1)
    # Blue square on right
    cv2.rectangle(frame, (420, 160), (580, 320), (200, 0, 0), -1)
    return frame


# ---------------------------------------------------------------------------
# T027-A: Shape enters frame → audio narration within 2 seconds
# ---------------------------------------------------------------------------


class TestExplorationModeNarration:
    """Tests for shape/color detection triggering audio narration."""

    def test_red_circle_triggers_audio_narration(self) -> None:
        """When a red circle enters the frame, audio narration plays within 2s."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        detector = ShapeColorDetector()
        frame = make_red_circle_frame()

        start = time.monotonic()
        events = detector.detect(frame)
        assert events, "Detector must find something in a red circle frame"

        for event in events:
            engine.on_event(event)

        elapsed = time.monotonic() - start
        assert elapsed < 2.0, (
            f"Detection + routing took {elapsed:.2f}s — must be under 2s"
        )

        # At least one audio response must be queued
        assert not audio_queue.empty(), (
            "No audio response was queued after detecting a shape/color in exploration mode"
        )

        response = audio_queue.get_nowait()
        assert isinstance(response, AudioResponse)
        assert response.priority in (AudioPriority.NORMAL, AudioPriority.LOW)
        # Text must mention something detected
        assert response.text.strip(), "Audio response text must not be empty"

    def test_detection_text_mentions_detected_label(self) -> None:
        """Audio narration text must include the detected shape or color label."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        # Inject a known event directly
        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
            bounding_box=BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3),
        )
        engine.on_event(event)

        assert not audio_queue.empty(), "No audio response queued for shape event"
        response = audio_queue.get_nowait()
        assert "circle" in response.text.lower(), (
            f"Expected 'circle' in narration text, got: {response.text!r}"
        )

    def test_color_event_triggers_audio_narration(self) -> None:
        """A COLOR DetectionEvent in EXPLORATION mode must trigger audio narration."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        event = DetectionEvent(
            type=DetectionType.COLOR,
            label="red",
            confidence=0.85,
            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.4, height=0.4),
        )
        engine.on_event(event)

        assert not audio_queue.empty(), "No audio response queued for color event"
        response = audio_queue.get_nowait()
        assert "red" in response.text.lower(), (
            f"Expected 'red' in narration text, got: {response.text!r}"
        )


# ---------------------------------------------------------------------------
# T027-B: Multiple shapes narrated individually
# ---------------------------------------------------------------------------


class TestMultipleShapeNarration:
    """Multiple shapes in one frame must each be narrated without overlap."""

    def test_multiple_shapes_each_produce_audio_response(self) -> None:
        """Each distinct detection event must produce its own audio response."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        events = [
            DetectionEvent(
                type=DetectionType.SHAPE,
                label="circle",
                confidence=0.9,
                bounding_box=BoundingBox(x=0.0, y=0.2, width=0.3, height=0.4),
            ),
            DetectionEvent(
                type=DetectionType.SHAPE,
                label="square",
                confidence=0.88,
                bounding_box=BoundingBox(x=0.6, y=0.2, width=0.3, height=0.4),
            ),
        ]

        for event in events:
            engine.on_event(event)

        responses = []
        while not audio_queue.empty():
            responses.append(audio_queue.get_nowait())

        assert len(responses) == 2, (
            f"Expected 2 audio responses for 2 distinct shapes, got {len(responses)}"
        )

        texts_lower = [r.text.lower() for r in responses]
        assert any("circle" in t for t in texts_lower), "Circle not narrated"
        assert any("square" in t for t in texts_lower), "Square not narrated"

    def test_different_labels_do_not_overlap(self) -> None:
        """Two different labels must produce two separate audio responses, not one merged."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        event_a = DetectionEvent(
            type=DetectionType.SHAPE,
            label="triangle",
            confidence=0.9,
            bounding_box=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3),
        )
        event_b = DetectionEvent(
            type=DetectionType.COLOR,
            label="blue",
            confidence=0.9,
            bounding_box=BoundingBox(x=0.5, y=0.1, width=0.3, height=0.3),
        )
        engine.on_event(event_a)
        engine.on_event(event_b)

        responses = []
        while not audio_queue.empty():
            responses.append(audio_queue.get_nowait())

        assert len(responses) == 2, (
            f"Expected 2 separate responses for 2 events, got {len(responses)}"
        )


# ---------------------------------------------------------------------------
# T027-C: Audio is the complete experience — no visual-only path
# ---------------------------------------------------------------------------


class TestAudioCompleteExperience:
    """Spec FR-010: every detection must be communicated via audio."""

    def test_shape_detection_always_produces_audio_not_only_ar(self) -> None:
        """A SHAPE event must ALWAYS enqueue an AudioResponse — AR is enhancement only."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="heart",
            confidence=0.91,
            bounding_box=BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3),
        )
        engine.on_event(event)

        assert not audio_queue.empty(), (
            "Shape detection must always produce an audio response — "
            "audio is the complete experience, not an enhancement"
        )


# ---------------------------------------------------------------------------
# T027-D: Blank frame — no audio, no exception
# ---------------------------------------------------------------------------


class TestBlankFrameHandling:
    """No detections on blank frames — no audio queued, no exceptions raised."""

    def test_blank_frame_no_audio_queued(self) -> None:
        """A blank frame must produce no detections and no audio responses."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        detector = ShapeColorDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            events = detector.detect(blank)
        except Exception as exc:
            pytest.fail(f"detect() raised on blank frame: {exc}")

        assert events == [], f"Expected no events from blank frame, got: {events}"

        for event in events:
            engine.on_event(event)

        assert audio_queue.empty(), (
            "No audio should be queued when no detections occur on a blank frame"
        )

    def test_no_exception_on_black_frame(self) -> None:
        """Entire pipeline must never raise on a black frame."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        detector = ShapeColorDetector()
        black = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            events = detector.detect(black)
            for event in events:
                engine.on_event(event)
        except Exception as exc:
            pytest.fail(f"Pipeline raised on black frame: {exc}")


# ---------------------------------------------------------------------------
# T027-E: Deduplication — same label within 3s suppressed
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Same label detected repeatedly within 3s must not flood audio queue."""

    def test_same_label_within_3s_not_duplicated(self) -> None:
        """Sending the same label twice rapidly must produce only 1 audio response."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)

        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
            bounding_box=BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3),
        )

        engine.on_event(event)
        engine.on_event(event)  # Immediate repeat — should be suppressed

        responses = []
        while not audio_queue.empty():
            responses.append(audio_queue.get_nowait())

        assert len(responses) == 1, (
            f"Expected 1 response (dedup within 3s), got {len(responses)}"
        )


# ---------------------------------------------------------------------------
# T027-F: Mode isolation — non-EXPLORATION mode does NOT narrate shapes
# ---------------------------------------------------------------------------


class TestModeIsolation:
    """Shape/color narration must only happen in EXPLORATION mode."""

    def test_shape_event_in_standby_mode_produces_no_audio(self) -> None:
        """SHAPE events in STANDBY mode must not produce audio responses."""
        audio_queue: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.STANDBY)

        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
            bounding_box=BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3),
        )
        engine.on_event(event)

        assert audio_queue.empty(), (
            "Shape events in STANDBY mode must not produce audio narration"
        )
