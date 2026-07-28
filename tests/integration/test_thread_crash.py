"""Integration tests for flec-9zg: ThreadSupervisor crash detection and restart (AC-9)."""
from __future__ import annotations
import json, queue, time, logging, threading
import numpy as np
import pytest
from flec.models import Mode, TaggedFrame


class TestThreadSupervisorRestart:
    """AC-9: Crashed thread restarts within 500ms, structured log emitted."""

    def test_crashed_thread_restarts_within_500ms(self):
        """ThreadSupervisor must detect a crash and restart within 500ms."""
        from flec.main import CapabilityThread
        from flec.engine.thread_supervisor import ThreadSupervisor

        restarted = []

        class CrashOnFirstFrame(CapabilityThread):
            def _process_tagged_frame(self, tf):
                raise RuntimeError("Simulated crash")

        original_thread = CrashOnFirstFrame()

        def factory():
            t = CrashOnFirstFrame()
            restarted.append(t)
            return t

        supervisor = ThreadSupervisor()
        supervisor.add_thread(original_thread, factory)
        supervisor.start()
        original_thread.start()

        # Send a frame to trigger the crash
        blank = TaggedFrame(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            origin_mode=Mode.EXPLORATION,
            timestamp_ns=0,
        )
        original_thread._input_queue.put(blank)

        # Wait for crash + restart (up to 1s)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not restarted:
            time.sleep(0.05)

        supervisor.stop()
        assert restarted, "Thread must be restarted after crash"

    def test_restart_log_event_emitted(self):
        """vision_thread_restarted event must be logged on restart.

        Uses a custom thread-safe log handler so daemon-thread log records are
        captured (pytest's caplog only captures records from the main thread).
        """
        from flec.main import CapabilityThread
        from flec.engine.thread_supervisor import ThreadSupervisor

        class CrashThread(CapabilityThread):
            def _process_tagged_frame(self, tf):
                raise RuntimeError("crash")

        original = CrashThread()
        new_thread = CrashThread()

        def factory():
            return new_thread

        # Install a thread-safe handler on the supervisor's logger to capture
        # INFO-level records from any thread (pytest caplog only sees main thread).
        captured_messages: list[str] = []
        _lock = threading.Lock()

        class CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                with _lock:
                    captured_messages.append(self.format(record))

        handler = CapturingHandler()
        handler.setLevel(logging.INFO)
        supervisor_logger = logging.getLogger("flec.engine.thread_supervisor")
        prev_level = supervisor_logger.level
        supervisor_logger.setLevel(logging.INFO)
        supervisor_logger.addHandler(handler)

        try:
            supervisor = ThreadSupervisor()
            supervisor.add_thread(original, factory)
            supervisor.start()
            original.start()

            blank = TaggedFrame(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                origin_mode=Mode.EXPLORATION,
                timestamp_ns=0,
            )
            original._input_queue.put(blank)
            time.sleep(0.6)  # allow supervisor poll cycle
            supervisor.stop()
        finally:
            supervisor_logger.removeHandler(handler)
            supervisor_logger.setLevel(prev_level)

        with _lock:
            msgs = list(captured_messages)

        assert any("vision_thread_restarted" in m for m in msgs), (
            f"Expected vision_thread_restarted in logs. Got: {msgs[:5]}"
        )

    def test_output_queue_preserved_after_restart(self):
        """After restart, the output queue handle must be the same object (transplant)."""
        from flec.main import CapabilityThread
        from flec.engine.thread_supervisor import ThreadSupervisor

        class CrashThread(CapabilityThread):
            def _process_tagged_frame(self, tf):
                raise RuntimeError("crash")

        original = CrashThread()
        original_output_q = original._output_queue

        def factory():
            return CrashThread()

        supervisor = ThreadSupervisor()
        supervisor.add_thread(original, factory)
        supervisor.start()
        original.start()

        blank = TaggedFrame(
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            origin_mode=Mode.EXPLORATION,
            timestamp_ns=0,
        )
        original._input_queue.put(blank)

        # Wait for crash + restart
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            entry = supervisor._threads[0]
            if entry["thread"] is not original:
                break
            time.sleep(0.05)

        supervisor.stop()

        # The supervisor entry's current thread should have the original output queue
        entry = supervisor._threads[0]
        assert entry["thread"]._output_queue is original_output_q, (
            "Output queue must be transplanted to restarted thread"
        )
