"""Integration tests for flec-9zg: mode switch clears stale state (AC-14)."""
from __future__ import annotations
import queue, time
import numpy as np
import pytest
from flec.models import Mode, TaggedFrame


class TestModeSwitchClearsState:
    def test_reset_tracking_called_on_mode_switch(self):
        """FlecSession.set_mode() must call reset_tracking() on all capability threads."""
        from flec.main import FlecSession, CapabilityThread
        import unittest.mock as mock

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            # Register a mock capability thread
            mock_thread = mock.MagicMock(spec=CapabilityThread)
            session._capability_threads.append(mock_thread)

            # Switch mode via the session's set_mode wrapper
            session.set_mode(Mode.READING)

            mock_thread.reset_tracking.assert_called_once()
        finally:
            session.shutdown()

    def test_mode_switch_resets_all_capability_threads(self):
        """All registered capability threads must be reset on mode switch."""
        from flec.main import FlecSession, CapabilityThread
        import unittest.mock as mock

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            # Register multiple mock capability threads
            threads = [mock.MagicMock(spec=CapabilityThread) for _ in range(3)]
            for t in threads:
                session._capability_threads.append(t)

            session.set_mode(Mode.CHALLENGE)

            for t in threads:
                t.reset_tracking.assert_called_once()
        finally:
            session.shutdown()

    def test_real_tracking_thread_reset_drains_queue(self):
        """After mode switch, a real CapabilityThread input queue must be drained."""
        from flec.main import FlecSession, CapabilityThread

        class NoopThread(CapabilityThread):
            def _process_tagged_frame(self, tf):
                # Never process — just accumulate frames so we can verify drain
                time.sleep(10)

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            thread = NoopThread()
            session._capability_threads.append(thread)

            blank = TaggedFrame(
                frame=np.zeros((64, 64, 3), dtype=np.uint8),
                origin_mode=Mode.EXPLORATION,
                timestamp_ns=0,
            )
            # Fill queue without starting the thread so frames accumulate
            for _ in range(3):
                try:
                    thread._input_queue.put_nowait(blank)
                except queue.Full:
                    break

            assert not thread._input_queue.empty(), "Pre-condition: queue should have frames"

            session.set_mode(Mode.READING)

            assert thread._input_queue.empty(), (
                "Input queue must be drained after set_mode() (reset_tracking)"
            )
        finally:
            session.shutdown()

    def test_session_set_mode_updates_response_engine_mode(self):
        """session.set_mode() must propagate the new mode to ResponseEngine."""
        from flec.main import FlecSession

        session = FlecSession(mode="dev", tts_backend="off", voice=False)
        try:
            session.set_mode(Mode.READING)
            assert session._response_engine.mode == Mode.READING
        finally:
            session.shutdown()
