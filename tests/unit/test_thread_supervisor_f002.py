"""Unit tests for ThreadSupervisor — AC-9: auto-restart within 500ms.

TDD RED phase — all tests must FAIL before ThreadSupervisor is implemented.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_crash_thread(crash_after: float = 0.05):
    """Return a (thread_instance, factory_fn) pair where the thread crashes after crash_after seconds."""
    from flec.main import CapabilityThread

    class _CrashThread(CapabilityThread):
        def run(self) -> None:
            time.sleep(crash_after)
            # Crash by raising — thread exits without setting stop_event

    t = _CrashThread()
    factory = lambda: _CrashThread()  # noqa: E731
    return t, factory


def _make_stable_thread():
    """Return a (thread_instance, factory_fn) pair for a thread that runs until stopped."""
    from flec.main import CapabilityThread

    class _StableThread(CapabilityThread):
        pass  # uses default run() — loops until _stop_event

    t = _StableThread()
    factory = lambda: _StableThread()  # noqa: E731
    return t, factory


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


class TestThreadSupervisorInterface:
    """ThreadSupervisor exposes the required management interface."""

    def test_importable(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor
        assert ThreadSupervisor is not None

    def test_add_thread_method_exists(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor
        s = ThreadSupervisor()
        assert callable(getattr(s, "add_thread", None))

    def test_start_method_exists(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor
        s = ThreadSupervisor()
        assert callable(getattr(s, "start", None))

    def test_stop_method_exists(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor
        s = ThreadSupervisor()
        assert callable(getattr(s, "stop", None))

    def test_add_thread_accepts_thread_and_factory(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor
        s = ThreadSupervisor()
        t, factory = _make_stable_thread()
        s.add_thread(t, factory)  # must not raise


# ---------------------------------------------------------------------------
# Restart tests
# ---------------------------------------------------------------------------


class TestThreadSupervisorRestarts:
    """AC-9: crashed threads are restarted within 500ms with same output queue."""

    def test_crashed_thread_restarted_within_600ms(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        t, factory = _make_crash_thread(crash_after=0.05)
        t.start()

        s = ThreadSupervisor()
        s.add_thread(t, factory)
        s.start()

        # Wait up to 600ms for the thread to be replaced and new one be alive
        deadline = time.monotonic() + 0.6
        restarted = False
        while time.monotonic() < deadline:
            entry = s._entries[0]
            if entry["thread"] is not t and entry["thread"].is_alive():
                restarted = True
                break
            time.sleep(0.02)

        s.stop()
        assert restarted, "Supervisor must restart a crashed thread within 600ms"

    def test_restart_emits_structured_log(self, caplog) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        t, factory = _make_crash_thread(crash_after=0.05)
        t.start()

        s = ThreadSupervisor()
        s.add_thread(t, factory)

        with caplog.at_level(logging.INFO, logger="flec.engine.thread_supervisor"):
            s.start()
            # Wait long enough for crash + restart cycle
            deadline = time.monotonic() + 0.6
            while time.monotonic() < deadline:
                entry = s._entries[0]
                if entry["thread"] is not t:
                    break
                time.sleep(0.02)
            s.stop()

        restart_records = [
            r for r in caplog.records
            if "vision_thread_restarted" in r.getMessage()
        ]
        assert restart_records, "Must emit a log record containing 'vision_thread_restarted'"

        # Verify the message is valid JSON with expected keys
        msg = restart_records[0].getMessage()
        parsed = json.loads(msg)
        assert parsed.get("event") == "vision_thread_restarted"
        assert "thread" in parsed

    def test_restarted_thread_shares_same_output_queue(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        t, factory = _make_crash_thread(crash_after=0.05)
        original_queue = t.get_output_queue()
        t.start()

        s = ThreadSupervisor()
        s.add_thread(t, factory)
        s.start()

        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            entry = s._entries[0]
            if entry["thread"] is not t:
                break
            time.sleep(0.02)

        new_thread = s._entries[0]["thread"]
        s.stop()

        assert new_thread is not t, "Thread must have been replaced"
        assert new_thread.get_output_queue() is original_queue, (
            "Restarted thread must share the same output queue instance"
        )


# ---------------------------------------------------------------------------
# Clean shutdown tests
# ---------------------------------------------------------------------------


class TestThreadSupervisorCleanShutdown:
    """stop() halts monitoring without restarting intentionally-stopped threads."""

    def test_stop_does_not_restart_intentionally_stopped_threads(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        t, factory = _make_stable_thread()
        t.start()

        s = ThreadSupervisor()
        s.add_thread(t, factory)
        s.start()

        # Intentionally stop the supervised thread BEFORE stopping supervisor
        t.stop()
        t.join(timeout=1.0)

        # Now stop supervisor
        s.stop()

        # The slot should still point at the original thread (no restart)
        assert s._entries[0]["thread"] is t, (
            "Supervisor must not restart a thread that was intentionally stopped"
        )

    def test_stop_returns_promptly(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        t, factory = _make_stable_thread()
        t.start()

        s = ThreadSupervisor()
        s.add_thread(t, factory)
        s.start()

        start = time.monotonic()
        s.stop()
        elapsed = time.monotonic() - start

        t.stop()
        t.join(timeout=1.0)

        assert elapsed < 2.0, f"stop() must return promptly, took {elapsed:.3f}s"

    def test_stop_without_start_does_not_raise(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        s = ThreadSupervisor()
        t, factory = _make_stable_thread()
        t.start()
        s.add_thread(t, factory)
        s.stop()  # called before start() — must not raise
        t.stop()
        t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Non-interference tests
# ---------------------------------------------------------------------------


class TestThreadSupervisorNonInterference:
    """Crashing one thread does not affect other supervised threads."""

    def test_crashing_one_thread_does_not_stop_other_threads(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor

        crash_t, crash_factory = _make_crash_thread(crash_after=0.05)
        stable_t, stable_factory = _make_stable_thread()

        crash_t.start()
        stable_t.start()

        s = ThreadSupervisor()
        s.add_thread(crash_t, crash_factory)
        s.add_thread(stable_t, stable_factory)
        s.start()

        # Wait for the crash + restart cycle
        time.sleep(0.4)

        still_alive = stable_t.is_alive()

        s.stop()
        stable_t.stop()
        stable_t.join(timeout=1.0)

        assert still_alive, (
            "A crashing thread must not cause other supervised threads to be stopped"
        )

    def test_crashing_one_thread_does_not_call_stop_on_other_threads(self) -> None:
        from flec.engine.thread_supervisor import ThreadSupervisor
        from flec.main import CapabilityThread

        stop_called_on_stable = threading.Event()

        class _TrackedStable(CapabilityThread):
            def stop(self) -> None:
                stop_called_on_stable.set()
                super().stop()

        crash_t, crash_factory = _make_crash_thread(crash_after=0.05)
        stable_t = _TrackedStable()
        stable_factory = lambda: _TrackedStable()  # noqa: E731

        crash_t.start()
        stable_t.start()

        s = ThreadSupervisor()
        s.add_thread(crash_t, crash_factory)
        s.add_thread(stable_t, stable_factory)
        s.start()

        time.sleep(0.4)

        # stop_called_on_stable must NOT have been set by supervisor
        was_stopped_by_supervisor = stop_called_on_stable.is_set()

        s.stop()
        stable_t.stop()
        stable_t.join(timeout=1.0)

        assert not was_stopped_by_supervisor, (
            "Supervisor must not call stop() on other threads when one crashes"
        )
