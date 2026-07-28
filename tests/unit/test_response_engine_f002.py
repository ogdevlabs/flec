"""Unit tests for ResponseEngine multi-thread reader + mode-tag validation + bounded narration queue.

TDD: these tests are written BEFORE the implementation (RED phase).
They document the contract for flec-bj9.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from flec.models import (
    AudioPriority,
    AudioResponse,
    DetectionEvent,
    DetectionType,
    Mode,
)
from flec.engine.response_engine import ResponseEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_event(label: str = "circle", origin_mode: Mode | None = None) -> DetectionEvent:
    return DetectionEvent(
        type=DetectionType.SHAPE,
        label=label,
        confidence=0.9,
        origin_mode=origin_mode,
    )


def _text_event(label: str = "hello", origin_mode: Mode | None = None) -> DetectionEvent:
    return DetectionEvent(
        type=DetectionType.TEXT,
        label=label,
        confidence=0.9,
        origin_mode=origin_mode,
    )


class _MockTTS:
    """Fake TTS backend that records speak() calls."""

    def __init__(self) -> None:
        self.calls: list[AudioResponse] = []
        self._lock = threading.Lock()

    def speak(self, response: AudioResponse) -> None:
        with self._lock:
            self.calls.append(response)

    def stop_current(self) -> None:
        pass

    def clear_pending(self) -> None:
        pass

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self.calls)


# ---------------------------------------------------------------------------
# TestAddCapabilityQueue
# ---------------------------------------------------------------------------


class TestAddCapabilityQueue:
    """add_capability_queue(q) spawns a thread that drains events and calls on_event."""

    def test_add_capability_queue_method_exists(self) -> None:
        engine = ResponseEngine(audio_queue=queue.Queue())
        assert hasattr(engine, "add_capability_queue"), (
            "ResponseEngine must have add_capability_queue method"
        )

    def test_events_from_capability_queue_reach_on_event(self) -> None:
        """Events placed in a capability queue must be routed through on_event."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        event = _shape_event("triangle")
        cap_q.put(event)

        # Give the reader thread time to drain
        deadline = time.monotonic() + 2.0
        while audio_q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not audio_q.empty(), (
            "DetectionEvent placed in capability queue must reach the audio queue via on_event"
        )
        response = audio_q.get_nowait()
        assert "triangle" in response.text.lower()

    def test_add_capability_queue_spawns_daemon_thread(self) -> None:
        """Each call to add_capability_queue must spawn exactly one daemon thread."""
        engine = ResponseEngine(audio_queue=queue.Queue())
        before = threading.active_count()
        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)
        after = threading.active_count()
        assert after == before + 1, (
            f"Expected 1 new thread after add_capability_queue; "
            f"before={before}, after={after}"
        )

    def test_multiple_capability_queues_each_get_a_thread(self) -> None:
        """Two capability queues must each get their own reader thread."""
        engine = ResponseEngine(audio_queue=queue.Queue())
        before = threading.active_count()
        engine.add_capability_queue(queue.Queue())
        engine.add_capability_queue(queue.Queue())
        after = threading.active_count()
        assert after == before + 2, (
            f"Expected 2 new threads for 2 capability queues; "
            f"before={before}, after={after}"
        )

    def test_none_sentinel_stops_reader_thread(self) -> None:
        """Sending None to a capability queue must stop its reader thread."""
        engine = ResponseEngine(audio_queue=queue.Queue())
        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        # Record thread reference
        t, _ = engine._reader_threads[-1]
        assert t.is_alive(), "Thread must be alive initially"

        cap_q.put(None)  # sentinel
        t.join(timeout=2.0)
        assert not t.is_alive(), "Thread must stop after receiving None sentinel"


# ---------------------------------------------------------------------------
# TestModeTagValidation
# ---------------------------------------------------------------------------


class TestModeTagValidation:
    """Reader threads discard events whose origin_mode doesn't match current mode."""

    def test_stale_event_discarded_on_mode_mismatch(self) -> None:
        """Event with origin_mode=READING is discarded when engine is in EXPLORATION mode."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        # This event has origin_mode=READING but engine is in EXPLORATION
        stale_event = _shape_event("circle", origin_mode=Mode.READING)
        cap_q.put(stale_event)

        # Small wait to let the reader thread process
        time.sleep(0.15)

        assert audio_q.empty(), (
            "Stale event (origin_mode=READING, engine mode=EXPLORATION) must be discarded"
        )

    def test_matching_mode_event_not_discarded(self) -> None:
        """Event with origin_mode matching current mode must NOT be discarded."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        # origin_mode matches current mode
        live_event = _shape_event("square", origin_mode=Mode.EXPLORATION)
        cap_q.put(live_event)

        deadline = time.monotonic() + 2.0
        while audio_q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not audio_q.empty(), (
            "Event with matching origin_mode must NOT be discarded"
        )

    def test_event_without_origin_mode_not_discarded(self) -> None:
        """Event with origin_mode=None (legacy) must pass through (no filtering)."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        # No origin_mode set — legacy event
        legacy_event = _shape_event("hexagon", origin_mode=None)
        cap_q.put(legacy_event)

        deadline = time.monotonic() + 2.0
        while audio_q.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        assert not audio_q.empty(), (
            "Legacy event with origin_mode=None must NOT be discarded"
        )

    def test_stale_discarded_log_event_emitted(self, caplog) -> None:
        """Discarding a stale event must emit a structured log entry."""
        import logging

        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        stale_event = _shape_event("star", origin_mode=Mode.READING)

        with caplog.at_level(logging.DEBUG, logger="flec.engine.response_engine"):
            cap_q.put(stale_event)
            time.sleep(0.15)

        log_text = caplog.text
        assert "stale_discarded" in log_text, (
            f"Expected 'stale_discarded' in log output; got:\n{log_text}"
        )


# ---------------------------------------------------------------------------
# TestNarrationQueueBound
# ---------------------------------------------------------------------------


class TestNarrationQueueBound:
    """When using tts= constructor path, narration queue is bounded at 50."""

    def test_bounded_queue_drops_excess_events(self) -> None:
        """Injecting 60 events with slow TTS: only ≤50 speak() calls ever queued."""
        slow_tts = _MockTTS()
        # Wrap in a slow backend: block speak() so the queue fills up
        blocked = threading.Event()
        original_speak = slow_tts.speak

        call_count = [0]
        lock = threading.Lock()

        class SlowTTS:
            def speak(self, response: AudioResponse) -> None:
                with lock:
                    call_count[0] += 1
                # Block until test releases
                blocked.wait(timeout=5.0)

            def stop_current(self) -> None:
                pass

            def clear_pending(self) -> None:
                pass

        slow = SlowTTS()
        engine = ResponseEngine(tts=slow)
        engine.set_mode(Mode.READING)

        # Fire 60 TEXT events rapidly at the engine via the tts path
        for i in range(60):
            event = DetectionEvent(
                type=DetectionType.TEXT,
                label=f"word{i}",
                confidence=0.9,
            )
            engine.on_event(event)

        # Small pause for dispatcher to start processing first item
        time.sleep(0.05)

        # Release the block so dispatcher can drain
        blocked.set()

        # Shutdown to drain queue
        engine.shutdown()

        # At most 50 items could have been enqueued (first item is being processed,
        # remaining 49 in queue = 50 total before drops start)
        # The exact number depends on timing, but it must be ≤ 50+1 (one already dispatching)
        assert call_count[0] <= 51, (
            f"Expected ≤51 speak() calls (50 queued + 1 dispatching); got {call_count[0]}"
        )

    def test_narration_queue_full_log_emitted(self, caplog) -> None:
        """When narration queue is full, a structured log warning must be emitted."""
        import logging
        import json

        blocked = threading.Event()
        log_records: list[str] = []

        class BlockingTTS:
            def speak(self, response: AudioResponse) -> None:
                blocked.wait(timeout=5.0)

            def stop_current(self) -> None:
                pass

            def clear_pending(self) -> None:
                pass

        engine = ResponseEngine(tts=BlockingTTS())
        engine.set_mode(Mode.READING)

        with caplog.at_level(logging.WARNING, logger="flec.engine.response_engine"):
            # Send 60 events — queue can hold 50; rest must be dropped with log
            for i in range(60):
                event = DetectionEvent(
                    type=DetectionType.TEXT,
                    label=f"word{i}",
                    confidence=0.9,
                )
                engine.on_event(event)

            time.sleep(0.1)

        blocked.set()
        engine.shutdown()

        assert "narration_queue_full" in caplog.text, (
            f"Expected 'narration_queue_full' warning in logs; got:\n{caplog.text}"
        )

    def test_audio_queue_path_unaffected(self) -> None:
        """The audio_queue= path must NOT use the bounded narration queue."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        # Send 60 events — all must reach the queue (no 50-cap on this path)
        for i in range(60):
            event = DetectionEvent(
                type=DetectionType.SHAPE,
                label=f"shape{i}",
                confidence=0.9,
            )
            engine.on_event(event)

        assert audio_q.qsize() == 60, (
            f"audio_queue= path must not cap at 50; expected 60 items, got {audio_q.qsize()}"
        )


# ---------------------------------------------------------------------------
# TestShutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """shutdown() is callable and returns promptly."""

    def test_shutdown_exists(self) -> None:
        engine = ResponseEngine(audio_queue=queue.Queue())
        assert hasattr(engine, "shutdown"), "ResponseEngine must have a shutdown() method"

    def test_shutdown_returns_without_hanging(self) -> None:
        """shutdown() must return within 5 seconds (all threads must join)."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        start = time.monotonic()
        engine.shutdown()
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"shutdown() took {elapsed:.2f}s — must complete within 5s"

    def test_shutdown_with_tts_path(self) -> None:
        """shutdown() must also work when using the tts= path."""
        tts = _MockTTS()
        engine = ResponseEngine(tts=tts)

        cap_q: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q)

        start = time.monotonic()
        engine.shutdown()
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"shutdown() (tts= path) took {elapsed:.2f}s"

    def test_shutdown_stops_reader_threads(self) -> None:
        """After shutdown(), all reader threads must be stopped."""
        engine = ResponseEngine(audio_queue=queue.Queue())

        cap_q1: queue.Queue = queue.Queue()
        cap_q2: queue.Queue = queue.Queue()
        engine.add_capability_queue(cap_q1)
        engine.add_capability_queue(cap_q2)

        threads = [t for t, _ in engine._reader_threads]
        for t in threads:
            assert t.is_alive(), "Threads must be alive before shutdown"

        engine.shutdown()

        for t in threads:
            assert not t.is_alive(), "All reader threads must stop after shutdown()"

    def test_shutdown_idempotent(self) -> None:
        """Calling shutdown() twice must not raise."""
        engine = ResponseEngine(audio_queue=queue.Queue())
        engine.shutdown()
        engine.shutdown()  # must not raise


# ---------------------------------------------------------------------------
# TestBackwardCompat
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """ResponseEngine(audio_queue=q) must work unchanged after flec-bj9 changes."""

    def test_audio_queue_constructor_still_works(self) -> None:
        """ResponseEngine(audio_queue=q) instantiation must not raise."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        assert engine is not None

    def test_on_event_puts_audio_response_in_queue(self) -> None:
        """on_event with a valid DetectionEvent must put AudioResponse into audio_queue."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
        )
        engine.on_event(event)

        assert not audio_q.empty(), "on_event must put AudioResponse into audio_queue"
        response = audio_q.get_nowait()
        assert isinstance(response, AudioResponse)
        assert "circle" in response.text.lower()

    def test_on_event_callable_without_threads(self) -> None:
        """on_event must be directly callable (used by integration tests and FlecSession)."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        # Direct call — must not require reader threads
        event = DetectionEvent(
            type=DetectionType.COLOR,
            label="blue",
            confidence=0.88,
        )
        engine.on_event(event)
        assert not audio_q.empty()

    def test_on_event_no_mode_filtering(self) -> None:
        """on_event itself must NOT apply mode-tag filtering (only reader threads do)."""
        audio_q: queue.Queue[AudioResponse] = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_q)
        engine.set_mode(Mode.EXPLORATION)

        # Event has origin_mode=READING but we call on_event directly — must NOT be filtered
        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="star",
            confidence=0.9,
            origin_mode=Mode.READING,
        )
        engine.on_event(event)

        # on_event routes by event TYPE, not origin_mode — SHAPE in EXPLORATION → audio
        assert not audio_q.empty(), (
            "on_event must NOT apply mode-tag filtering — that is only the reader thread's job"
        )

    def test_set_mode_still_works(self) -> None:
        """set_mode() must still function correctly."""
        engine = ResponseEngine(audio_queue=queue.Queue())
        engine.set_mode(Mode.READING)
        assert engine.mode == Mode.READING

    def test_tts_constructor_still_works(self) -> None:
        """ResponseEngine(tts=...) constructor path must still work."""
        tts = _MockTTS()
        engine = ResponseEngine(tts=tts)
        engine.set_mode(Mode.EXPLORATION)

        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
        )
        engine.on_event(event)

        # Give dispatcher time to call speak()
        deadline = time.monotonic() + 2.0
        while tts.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert tts.call_count >= 1, "TTS speak() must be called via tts= constructor path"
        engine.shutdown()
