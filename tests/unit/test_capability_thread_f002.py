"""Unit tests for CapabilityThread interface and FlecSession non-blocking frame dispatch.

TDD RED phase — these tests must FAIL before CapabilityThread is added to main.py.

AC-7: thread isolation architecture
AC-10: mode-tag dispatch
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# CapabilityThread interface tests
# ---------------------------------------------------------------------------


class TestCapabilityThreadInterface:
    """AC-7: CapabilityThread exposes the required queue-isolation interface."""

    def test_capability_thread_importable(self) -> None:
        from flec.main import CapabilityThread
        assert CapabilityThread is not None

    def test_get_output_queue_returns_queue(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        out_q = t.get_output_queue()
        assert isinstance(out_q, queue.Queue), (
            "get_output_queue() must return a queue.Queue"
        )

    def test_get_output_queue_same_instance_each_call(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        assert t.get_output_queue() is t.get_output_queue()

    def test_reset_tracking_is_callable(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        t.reset_tracking()  # must not raise

    def test_stop_is_callable(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        t.stop()  # must not raise

    def test_is_daemon_thread(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        assert t.daemon is True, "CapabilityThread must be a daemon thread"

    def test_is_threading_thread_subclass(self) -> None:
        from flec.main import CapabilityThread
        assert issubclass(CapabilityThread, threading.Thread)

    def test_has_input_queue(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        assert hasattr(t, "_input_queue")
        assert isinstance(t._input_queue, queue.Queue)

    def test_has_stop_event(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        assert hasattr(t, "_stop_event")
        assert isinstance(t._stop_event, threading.Event)

    def test_stop_sets_stop_event(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        assert not t._stop_event.is_set()
        t.stop()
        assert t._stop_event.is_set()

    def test_reset_tracking_drains_input_queue(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        # Fill the input queue with dummy items
        for _ in range(3):
            t._input_queue.put("dummy")
        assert not t._input_queue.empty()
        t.reset_tracking()
        assert t._input_queue.empty(), (
            "reset_tracking() must drain the input queue"
        )

    def test_run_loop_stops_on_stop_event(self) -> None:
        from flec.main import CapabilityThread

        class _Stub(CapabilityThread):
            pass

        t = _Stub()
        t.start()
        t.stop()
        t.join(timeout=1.0)
        assert not t.is_alive(), "Thread must exit after stop() is called"


# ---------------------------------------------------------------------------
# TaggedFrame dispatch tests (AC-10)
# ---------------------------------------------------------------------------


class TestTaggedFrameDispatch:
    """AC-10: process_frame wraps frames in TaggedFrame with mode + timestamp."""

    def _make_session(self):
        """Build a FlecSession with voice disabled and mocked heavy deps."""
        with patch("flec.main._load_whisper", return_value=None):
            from flec.main import FlecSession
            session = FlecSession(mode="dev", tts_backend="off", voice=False)
        return session

    def test_process_frame_returns_immediately(self) -> None:
        session = self._make_session()
        frame = _blank_frame()

        start = time.monotonic()
        session.process_frame(frame)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, (
            f"process_frame should return quickly, took {elapsed:.3f}s"
        )

    def test_process_frame_dispatches_tagged_frame_to_capability_thread(self) -> None:
        from flec.main import CapabilityThread
        from flec.models import TaggedFrame

        session = self._make_session()
        frame = _blank_frame()

        class _CollectorThread(CapabilityThread):
            def __init__(self, input_queue_maxsize: int = 5):
                super().__init__(input_queue_maxsize=input_queue_maxsize)
                self.received: list = []

            def _process_tagged_frame(self, tagged_frame) -> None:
                self.received.append(tagged_frame)

        collector = _CollectorThread(input_queue_maxsize=10)
        session._capability_threads.append(collector)
        collector.start()

        session.process_frame(frame)
        time.sleep(0.1)  # let the thread drain
        collector.stop()
        collector.join(timeout=1.0)

        assert len(collector.received) >= 1, (
            "process_frame must dispatch a TaggedFrame to registered capability threads"
        )
        tf = collector.received[0]
        assert isinstance(tf, TaggedFrame), f"dispatched item must be TaggedFrame, got {type(tf)}"

    def test_dispatched_tagged_frame_has_origin_mode(self) -> None:
        from flec.main import CapabilityThread
        from flec.models import TaggedFrame, Mode

        session = self._make_session()
        frame = _blank_frame()

        received_frames: list = []

        class _CollectorThread(CapabilityThread):
            def _process_tagged_frame(self, tagged_frame) -> None:
                received_frames.append(tagged_frame)

        collector = _CollectorThread(input_queue_maxsize=10)
        session._capability_threads.append(collector)
        collector.start()

        session.process_frame(frame)
        time.sleep(0.1)
        collector.stop()
        collector.join(timeout=1.0)

        assert len(received_frames) >= 1
        tf = received_frames[0]
        assert isinstance(tf.origin_mode, Mode), (
            "TaggedFrame.origin_mode must be a Mode enum"
        )

    def test_dispatched_tagged_frame_has_timestamp_ns(self) -> None:
        from flec.main import CapabilityThread
        from flec.models import TaggedFrame

        session = self._make_session()
        frame = _blank_frame()
        before_ns = time.monotonic_ns()

        received_frames: list = []

        class _CollectorThread(CapabilityThread):
            def _process_tagged_frame(self, tagged_frame) -> None:
                received_frames.append(tagged_frame)

        collector = _CollectorThread(input_queue_maxsize=10)
        session._capability_threads.append(collector)
        collector.start()

        session.process_frame(frame)
        time.sleep(0.1)
        after_ns = time.monotonic_ns()
        collector.stop()
        collector.join(timeout=1.0)

        assert len(received_frames) >= 1
        tf = received_frames[0]
        assert isinstance(tf.timestamp_ns, int), "TaggedFrame.timestamp_ns must be int"
        assert before_ns <= tf.timestamp_ns <= after_ns, (
            "TaggedFrame.timestamp_ns must be within the call window"
        )

    def test_dispatched_tagged_frame_frame_matches_input(self) -> None:
        from flec.main import CapabilityThread

        session = self._make_session()
        frame = _blank_frame()
        frame[0, 0] = [42, 43, 44]  # sentinel pixel

        received_frames: list = []

        class _CollectorThread(CapabilityThread):
            def _process_tagged_frame(self, tagged_frame) -> None:
                received_frames.append(tagged_frame)

        collector = _CollectorThread(input_queue_maxsize=10)
        session._capability_threads.append(collector)
        collector.start()

        session.process_frame(frame)
        time.sleep(0.1)
        collector.stop()
        collector.join(timeout=1.0)

        assert len(received_frames) >= 1
        tf = received_frames[0]
        assert np.array_equal(tf.frame[0, 0], [42, 43, 44]), (
            "TaggedFrame.frame must be the same frame passed to process_frame"
        )

    def test_no_capability_threads_no_error(self) -> None:
        session = self._make_session()
        frame = _blank_frame()
        # With no registered capability threads, process_frame should not raise
        session.process_frame(frame)  # must not raise


# ---------------------------------------------------------------------------
# Drop-oldest on full queue
# ---------------------------------------------------------------------------


class TestDropOldestOnFullQueue:
    """Non-blocking dispatch: when input queue is full, oldest frame is dropped."""

    def _make_session(self):
        with patch("flec.main._load_whisper", return_value=None):
            from flec.main import FlecSession
            session = FlecSession(mode="dev", tts_backend="off", voice=False)
        return session

    def test_process_frame_does_not_block_when_queue_full(self) -> None:
        from flec.main import CapabilityThread

        session = self._make_session()

        class _SlowThread(CapabilityThread):
            """Never drains — simulate a full input queue."""

            def run(self) -> None:
                # Don't process anything — just block the queue
                self._stop_event.wait()

        slow = _SlowThread(input_queue_maxsize=2)
        session._capability_threads.append(slow)
        slow.start()

        frame = _blank_frame()
        start = time.monotonic()

        # Push enough frames to fill the queue
        for _ in range(10):
            session.process_frame(frame)

        elapsed = time.monotonic() - start
        slow.stop()
        slow.join(timeout=1.0)

        assert elapsed < 2.0, (
            f"process_frame must not block when capability thread queue is full; "
            f"took {elapsed:.3f}s for 10 frames"
        )

    def test_queue_not_full_after_overflow(self) -> None:
        from flec.main import CapabilityThread

        session = self._make_session()

        class _BlockedThread(CapabilityThread):
            def run(self) -> None:
                self._stop_event.wait()

        blocked = _BlockedThread(input_queue_maxsize=2)
        session._capability_threads.append(blocked)
        blocked.start()

        frame = _blank_frame()
        # Overflow the queue (maxsize=2, push 5)
        for _ in range(5):
            session.process_frame(frame)

        # After overflow the queue should still be drainable (not blocked)
        qsize = blocked._input_queue.qsize()
        assert qsize <= 2, (
            f"After overflow, queue size must be <= maxsize (2), got {qsize}"
        )

        blocked.stop()
        blocked.join(timeout=1.0)


# ---------------------------------------------------------------------------
# FlecSession backward-compatibility tests
# ---------------------------------------------------------------------------


class TestFlecSessionBackwardCompatibility:
    """FlecSession retains all methods integration tests depend on."""

    def _make_session(self):
        with patch("flec.main._load_whisper", return_value=None):
            from flec.main import FlecSession
            session = FlecSession(mode="dev", tts_backend="off", voice=False)
        return session

    def test_process_frame_method_exists(self) -> None:
        session = self._make_session()
        assert callable(getattr(session, "process_frame", None))

    def test_shutdown_method_exists(self) -> None:
        session = self._make_session()
        assert callable(getattr(session, "shutdown", None))

    def test_reset_reading_state_method_exists(self) -> None:
        session = self._make_session()
        assert callable(getattr(session, "reset_reading_state", None))

    def test_process_frame_signature_accepts_ocr_result(self) -> None:
        session = self._make_session()
        frame = _blank_frame()
        # Must accept ocr_result as optional kwarg
        session.process_frame(frame, ocr_result=["hello", "world"])

    def test_finger_tracker_attribute_exists(self) -> None:
        session = self._make_session()
        assert hasattr(session, "_finger_tracker")

    def test_shape_detector_attribute_exists(self) -> None:
        session = self._make_session()
        assert hasattr(session, "_shape_detector")

    def test_stabilizer_attribute_exists(self) -> None:
        session = self._make_session()
        assert hasattr(session, "_stabilizer")

    def test_capability_threads_list_initialized(self) -> None:
        session = self._make_session()
        assert hasattr(session, "_capability_threads"), (
            "FlecSession must have _capability_threads list"
        )
        assert isinstance(session._capability_threads, list)

    def test_response_engine_property(self) -> None:
        session = self._make_session()
        assert session.response_engine is not None

    def test_finger_tracker_property(self) -> None:
        session = self._make_session()
        assert session.finger_tracker is not None
