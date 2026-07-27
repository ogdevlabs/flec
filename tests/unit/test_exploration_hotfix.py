"""Hotfix regression tests for exploration mode performance and narration issues.

Three bugs fixed:
1. FingerTracker ran MediaPipe on every frame regardless of mode (~13ms wasted/frame in Exploration)
2. HSV color detection fired on background (walls, floors) causing false narration
3. Went silent because background COLOR spam exhausted dedup windows for real objects
"""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from flec.engine.response_engine import ResponseEngine, _QueueTTS
from flec.models import (
    AudioPriority,
    AudioResponse,
    DetectionEvent,
    DetectionType,
    Mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(audio_q: queue.Queue) -> ResponseEngine:
    eng = ResponseEngine(audio_queue=audio_q)
    eng.set_mode(Mode.EXPLORATION)
    return eng


def _fire(eng: ResponseEngine, label: str, typ: DetectionType = DetectionType.COLOR,
          meta: dict | None = None) -> None:
    eng.on_event(DetectionEvent(type=typ, label=label, confidence=0.9, metadata=meta or {}))


def _drain(q: queue.Queue) -> list[str]:
    out = []
    while True:
        try:
            out.append(q.get_nowait().text)
        except queue.Empty:
            break
    return out


# ---------------------------------------------------------------------------
# Bug 1 — FingerTracker must not run in Exploration mode
# ---------------------------------------------------------------------------


class TestFingerTrackerGating:
    """FingerTracker.update() should only be called in READING mode."""

    def test_finger_tracker_not_called_in_exploration(self) -> None:
        """process_frame must skip FingerTracker.update() when mode is EXPLORATION."""
        import os
        os.environ["FLEC_READING_WEAR_OVERRIDE"] = "1"
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="log", voice=False, shapes=False)
        session._response_engine.set_mode(Mode.EXPLORATION)

        mock_tracker = MagicMock()
        mock_tracker.update.return_value = MagicMock(
            detected=False, intent=MagicMock(name="IDLE"), nearest_text=None,
            velocity=0.0, position_x=0.5, position_y=0.5,
        )
        mock_tracker.current_state = mock_tracker.update.return_value
        session._finger_tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame)

        mock_tracker.update.assert_not_called()

    def test_finger_tracker_called_in_reading(self) -> None:
        """process_frame must call FingerTracker.update() in READING mode."""
        import os
        os.environ["FLEC_READING_WEAR_OVERRIDE"] = "1"
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="log", voice=False, shapes=False)
        session._response_engine.set_mode(Mode.READING)

        mock_tracker = MagicMock()
        mock_tracker.update.return_value = MagicMock(
            detected=False, intent=MagicMock(name="IDLE"), nearest_text=None,
            velocity=0.0, position_x=0.5, position_y=0.5,
        )
        mock_tracker.current_state = mock_tracker.update.return_value
        session._finger_tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame)

        mock_tracker.update.assert_called_once()

    def test_finger_tracker_not_called_in_challenge(self) -> None:
        """FingerTracker must also be skipped in CHALLENGE mode."""
        import os
        os.environ["FLEC_READING_WEAR_OVERRIDE"] = "1"
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="log", voice=False, shapes=False)
        session._response_engine.set_mode(Mode.CHALLENGE)

        mock_tracker = MagicMock()
        mock_tracker.update.return_value = MagicMock(
            detected=False, intent=MagicMock(name="IDLE"), nearest_text=None,
            velocity=0.0, position_x=0.5, position_y=0.5,
        )
        session._finger_tracker = mock_tracker

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame)

        mock_tracker.update.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 2 — HSV color detection must not run in the live session
# ---------------------------------------------------------------------------


class TestHSVNotInLiveSession:
    """ShapeColorDetector with enable_contour_shapes=False must not emit COLOR events.

    In the live session, YOLO is the only source of truth. HSV fires on
    background walls/floors and causes false narration. When YOLO is absent
    the session should degrade gracefully (silence) not narrate the room's
    ambient colours.
    """

    def test_no_color_events_without_yolo(self) -> None:
        """With YOLO absent and contour shapes off, detect() returns empty list."""
        from flec.perception.shape_color_detector import ShapeColorDetector

        detector = ShapeColorDetector(
            model_path=None,
            enable_contour_shapes=False,
        )
        assert detector._yolo is None, "Expected YOLO to be absent for this test"

        # Solid red frame — previously would have triggered a COLOR event
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 220  # red channel
        events = detector.detect(frame)

        color_events = [e for e in events if e.type == DetectionType.COLOR]
        assert color_events == [], (
            f"Expected no COLOR events without YOLO in live mode; got {color_events}"
        )

    def test_color_events_emitted_with_contour_shapes_flag(self) -> None:
        """When enable_contour_shapes=True (--shapes mode), COLOR events still fire."""
        from flec.perception.shape_color_detector import ShapeColorDetector

        detector = ShapeColorDetector(enable_contour_shapes=True)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 2] = 220  # solid red
        events = detector.detect(frame)

        color_events = [e for e in events if e.type == DetectionType.COLOR]
        assert any(e.label == "red" for e in color_events), (
            "Expected 'red' COLOR event with contour shapes enabled"
        )


# ---------------------------------------------------------------------------
# Bug 3 — Background color spam must not silence real object narration
# ---------------------------------------------------------------------------


class TestExplorationDedup:
    """Background COLOR events must not eat the dedup window for OBJECT events."""

    def test_object_narrated_after_color_background(self) -> None:
        """An OBJECT event fires even if COLOR background events fired recently."""
        q: queue.Queue = queue.Queue()
        eng = _make_engine(q)

        # Background: white wall fires (this should not block object narration)
        _fire(eng, "white", DetectionType.COLOR)
        _drain(q)

        # A YOLO object detection — different type, different label — must fire
        _fire(eng, "cup", DetectionType.OBJECT, meta={"color": "red"})
        spoken = _drain(q)
        assert spoken, "OBJECT event was silenced after background COLOR event"

    def test_unique_objects_each_narrated(self) -> None:
        """Each distinct OBJECT label is narrated once within the dedup window."""
        q: queue.Queue = queue.Queue()
        eng = _make_engine(q)

        _fire(eng, "cup", DetectionType.OBJECT, meta={"color": "red"})
        _fire(eng, "book", DetectionType.OBJECT, meta={"color": "blue"})
        _fire(eng, "cup", DetectionType.OBJECT, meta={"color": "red"})  # dedup hit

        spoken = _drain(q)
        assert len(spoken) == 2, f"Expected 2 unique narrations, got {len(spoken)}: {spoken}"

    def test_same_object_re_narrated_after_dedup_window(self) -> None:
        """Same label fires again once the dedup window expires."""
        q: queue.Queue = queue.Queue()
        eng = _make_engine(q)

        _fire(eng, "cup", DetectionType.OBJECT, meta={"color": None})
        assert _drain(q)  # first narration fires

        _fire(eng, "cup", DetectionType.OBJECT, meta={"color": None})
        assert not _drain(q), "Should be deduped within window"

        # Dedup window expires — allow 5.1s for robustness
        time.sleep(5.1)
        _fire(eng, "cup", DetectionType.OBJECT, meta={"color": None})
        assert _drain(q), "Should narrate again after dedup window"
