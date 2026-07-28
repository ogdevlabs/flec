"""Unit tests for TrackingThread (flec-6hl).

Covers:
- Interface: subclass check, init state, queue drain, graceful no-model
- Output: track_id stamping, origin_mode stamping, graceful degradation
- Reset state: predictor.trackers cleared on reset
"""
from __future__ import annotations

import queue
import unittest.mock as mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def make_stub_result(label: str = "cup", conf: float = 0.8, track_id: int = 7):
    box = mock.MagicMock()
    box.conf = [conf]
    box.cls = [0]
    box.id = [track_id]
    boxes = mock.MagicMock()
    boxes.__iter__ = mock.Mock(return_value=iter([box]))
    result = mock.MagicMock()
    result.boxes = boxes
    result.names = {0: label}
    return result


def _make_tracking_thread_with_stub_model(label="cup", conf=0.8, track_id=7):
    from flec.perception.tracking_thread import TrackingThread

    model_mock = mock.MagicMock()
    model_mock.track.return_value = [make_stub_result(label, conf, track_id)]
    t = TrackingThread()
    t._model = model_mock
    t._load_failed = False
    return t


def _make_tagged_frame(origin_mode=None):
    from flec.models import TaggedFrame, Mode
    import time

    mode = origin_mode or Mode.EXPLORATION
    return TaggedFrame(
        frame=_blank_frame(),
        origin_mode=mode,
        timestamp_ns=time.monotonic_ns(),
    )


# ---------------------------------------------------------------------------
# TestTrackingThreadInterface
# ---------------------------------------------------------------------------


class TestTrackingThreadInterface:
    """Basic interface and initialisation checks."""

    def test_is_capability_thread(self) -> None:
        from flec.main import CapabilityThread
        from flec.perception.tracking_thread import TrackingThread

        assert issubclass(TrackingThread, CapabilityThread), (
            "TrackingThread must be a subclass of CapabilityThread"
        )

    def test_no_model_at_init(self) -> None:
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        assert t._model is None, "_model must be None before first frame"

    def test_reset_tracking_drains_queue(self) -> None:
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        # Put three dummy items into the input queue
        for _ in range(3):
            t._input_queue.put("dummy")
        assert not t._input_queue.empty()
        t.reset_tracking()
        assert t._input_queue.empty(), (
            "reset_tracking() must drain the input queue"
        )

    def test_reset_tracking_no_crash_without_model(self) -> None:
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        assert t._model is None
        # Must not raise even though no model is loaded
        t.reset_tracking()


# ---------------------------------------------------------------------------
# TestTrackingThreadOutput
# ---------------------------------------------------------------------------


class TestTrackingThreadOutput:
    """Output events: track_id, origin_mode, graceful degradation."""

    def test_track_id_on_event(self) -> None:
        from flec.models import DetectionEvent

        t = _make_tracking_thread_with_stub_model(track_id=42)
        tagged = _make_tagged_frame()
        t._process_tagged_frame(tagged)

        assert not t._output_queue.empty(), "Expected at least one DetectionEvent in output queue"
        event: DetectionEvent = t._output_queue.get_nowait()
        assert event.track_id == 42, f"Expected track_id=42, got {event.track_id}"

    def test_origin_mode_stamped(self) -> None:
        from flec.models import DetectionEvent, Mode

        t = _make_tracking_thread_with_stub_model()
        tagged = _make_tagged_frame(origin_mode=Mode.READING)
        t._process_tagged_frame(tagged)

        assert not t._output_queue.empty()
        event: DetectionEvent = t._output_queue.get_nowait()
        assert event.origin_mode == Mode.READING, (
            f"origin_mode must be stamped from tagged_frame; got {event.origin_mode}"
        )

    def test_no_model_graceful_degradation(self) -> None:
        """When model load fails, _process_tagged_frame must not raise and must emit nothing."""
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        t._load_failed = True  # simulate prior load failure
        tagged = _make_tagged_frame()

        t._process_tagged_frame(tagged)  # must not raise

        assert t._output_queue.empty(), (
            "No events should be emitted when model load failed"
        )


# ---------------------------------------------------------------------------
# TestTrackingThreadResetState
# ---------------------------------------------------------------------------


class TestTrackingThreadResetState:
    """reset_tracking() clears predictor tracker state."""

    def test_reset_clears_predictor_trackers(self) -> None:
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()

        # Build a mock model with a predictor that has a trackers attribute
        predictor_mock = mock.MagicMock()
        predictor_mock.trackers = ["some_tracker_state"]
        model_mock = mock.MagicMock()
        model_mock.predictor = predictor_mock
        t._model = model_mock

        t.reset_tracking()

        # After reset, trackers must be set to None
        assert predictor_mock.trackers is None, (
            "reset_tracking() must set model.predictor.trackers = None"
        )
