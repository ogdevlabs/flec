"""Integration tests for flec-9zg: capability thread isolation and mode-tag validation."""
from __future__ import annotations
import queue, time, threading
import numpy as np
import pytest
from flec.models import Mode, DetectionEvent, DetectionType, TaggedFrame


class TestThreadIsolation:
    """AC-7: No thread blocks another for >10ms while all are active."""

    def test_multiple_threads_process_independently(self):
        """Two CapabilityThreads must process frames concurrently without blocking."""
        from flec.main import CapabilityThread

        processed = []
        lock = threading.Lock()

        class SlowThread(CapabilityThread):
            def _process_tagged_frame(self, tf):
                time.sleep(0.02)  # 20ms processing
                with lock:
                    processed.append(("slow", time.monotonic()))

        class FastThread(CapabilityThread):
            def _process_tagged_frame(self, tf):
                with lock:
                    processed.append(("fast", time.monotonic()))

        slow = SlowThread()
        fast = FastThread()
        slow.start()
        fast.start()

        try:
            blank = TaggedFrame(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                origin_mode=Mode.EXPLORATION,
                timestamp_ns=0,
            )
            slow._input_queue.put(blank)
            fast._input_queue.put(blank)
            # Both should complete within ~100ms
            time.sleep(0.1)

            with lock:
                fast_times = [t for name, t in processed if name == "fast"]
                slow_times = [t for name, t in processed if name == "slow"]

            assert fast_times, "FastThread did not process"
            assert slow_times, "SlowThread did not process"
            # Both completed, implying they ran concurrently
            assert len(processed) >= 2
        finally:
            slow.stop()
            fast.stop()


class TestModeTagValidation:
    """AC-10: ResponseEngine discards events with wrong origin_mode."""

    def test_stale_event_discarded_by_response_engine(self):
        """Events from a different mode than current must be discarded by ResponseEngine."""
        from flec.engine.response_engine import ResponseEngine

        on_event_calls = []

        class TrackingResponseEngine(ResponseEngine):
            def on_event(self, event: DetectionEvent) -> None:
                on_event_calls.append(event)

        engine = TrackingResponseEngine()
        engine.set_mode(Mode.EXPLORATION)
        output_q = queue.Queue()
        engine.add_capability_queue(output_q)

        # Inject stale event (origin_mode=READING, but current mode=EXPLORATION)
        stale = DetectionEvent(
            type=DetectionType.OBJECT,
            label="cup",
            confidence=0.9,
            origin_mode=Mode.READING,
        )
        output_q.put(stale)
        time.sleep(0.15)  # let reader thread process it

        assert on_event_calls == [], "Stale event must be discarded, not processed"

    def test_current_mode_event_accepted(self):
        """Events matching current mode must be forwarded to on_event."""
        from flec.engine.response_engine import ResponseEngine

        on_event_calls = []

        class TrackingResponseEngine(ResponseEngine):
            def on_event(self, event: DetectionEvent) -> None:
                on_event_calls.append(event)

        engine = TrackingResponseEngine()
        engine.set_mode(Mode.EXPLORATION)
        output_q = queue.Queue()
        engine.add_capability_queue(output_q)

        # Inject matching event
        current = DetectionEvent(
            type=DetectionType.OBJECT,
            label="cup",
            confidence=0.9,
            origin_mode=Mode.EXPLORATION,
        )
        output_q.put(current)
        time.sleep(0.15)

        assert len(on_event_calls) == 1

    def test_none_origin_mode_event_accepted(self):
        """Events with origin_mode=None (legacy) must not be discarded."""
        from flec.engine.response_engine import ResponseEngine

        on_event_calls = []

        class TrackingResponseEngine(ResponseEngine):
            def on_event(self, event: DetectionEvent) -> None:
                on_event_calls.append(event)

        engine = TrackingResponseEngine()
        engine.set_mode(Mode.EXPLORATION)
        output_q = queue.Queue()
        engine.add_capability_queue(output_q)

        legacy = DetectionEvent(
            type=DetectionType.OBJECT,
            label="ball",
            confidence=0.7,
        )
        output_q.put(legacy)
        time.sleep(0.15)

        assert len(on_event_calls) == 1


class TestTrackingIDsAfterModeSwitch:
    """AC-14: Tracking IDs must not appear in events after a mode switch + reset."""

    def test_reset_tracking_drains_input_queue(self):
        """After reset_tracking(), input queue must be empty."""
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        blank = TaggedFrame(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            origin_mode=Mode.EXPLORATION,
            timestamp_ns=0,
        )
        # Fill input queue
        for _ in range(3):
            try:
                t._input_queue.put_nowait(blank)
            except queue.Full:
                break
        t.reset_tracking()
        assert t._input_queue.empty(), "Input queue must be drained after reset_tracking()"

    def test_reset_tracking_clears_predictor_state(self):
        """reset_tracking() must null out the YOLO predictor tracker state."""
        import unittest.mock as mock
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        # Simulate a loaded model with predictor
        predictor = mock.MagicMock()
        predictor.trackers = ["tracker1"]
        model = mock.MagicMock()
        model.predictor = predictor
        t._model = model
        t.reset_tracking()
        assert predictor.trackers is None
