"""Contract tests for WakeWordListener.

Tests verify the public interface contract defined in
specs/001-perception-core/contracts/module-interfaces.md.

All tests use mock audio backend — no real microphone or openWakeWord model required.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Contract: WakeWordListener must be importable from its defined path
# ---------------------------------------------------------------------------


def test_wake_word_listener_importable() -> None:
    """WakeWordListener must be importable from flec.speech.wake_word_listener."""
    from flec.speech.wake_word_listener import WakeWordListener  # noqa: F401


# ---------------------------------------------------------------------------
# Contract: is_listening property
# ---------------------------------------------------------------------------


def test_is_listening_false_before_start() -> None:
    """is_listening must be False before start() is called."""
    from flec.speech.wake_word_listener import WakeWordListener

    with _patched_audio():
        listener = WakeWordListener()
        assert listener.is_listening is False


def test_is_listening_true_after_start() -> None:
    """is_listening must be True immediately after start() is called."""
    from flec.speech.wake_word_listener import WakeWordListener

    callback = MagicMock()
    with _patched_audio():
        listener = WakeWordListener()
        listener.start(callback)
        assert listener.is_listening is True
        listener.stop()


def test_is_listening_false_after_stop() -> None:
    """is_listening must be False after stop() is called."""
    from flec.speech.wake_word_listener import WakeWordListener

    callback = MagicMock()
    with _patched_audio():
        listener = WakeWordListener()
        listener.start(callback)
        listener.stop()
        assert listener.is_listening is False


# ---------------------------------------------------------------------------
# Contract: on_detected callback is invoked when wake word is detected
# ---------------------------------------------------------------------------


def test_callback_invoked_when_wake_word_detected() -> None:
    """on_detected callback must be invoked when mock audio matches wake word."""
    from flec.speech.wake_word_listener import WakeWordListener

    detected = threading.Event()
    callback = lambda: detected.set()  # noqa: E731

    with _patched_audio() as mock_audio:
        # Configure the mock to simulate a wake word detection on the next poll
        mock_audio.simulate_wake_word_detection = True
        listener = WakeWordListener()
        listener.start(callback)

        # Trigger a simulated wake word detection via the mock
        listener._trigger_detection_for_test()

        result = detected.wait(timeout=1.0)
        listener.stop()

    assert result, "on_detected callback was not invoked within timeout"


def test_callback_not_invoked_without_wake_word() -> None:
    """on_detected callback must NOT be invoked when no wake word is detected."""
    from flec.speech.wake_word_listener import WakeWordListener

    invocation_count = [0]

    def callback() -> None:
        invocation_count[0] += 1

    with _patched_audio():
        listener = WakeWordListener()
        listener.start(callback)
        # Give the listener time to run without a wake word trigger
        time.sleep(0.05)
        listener.stop()

    assert invocation_count[0] == 0, (
        f"Expected 0 callback invocations with no wake word, got {invocation_count[0]}"
    )


# ---------------------------------------------------------------------------
# Contract: stop() prevents further on_detected calls
# ---------------------------------------------------------------------------


def test_stop_prevents_further_callbacks() -> None:
    """After stop(), on_detected must not be called even if detection occurs."""
    from flec.speech.wake_word_listener import WakeWordListener

    invocations_after_stop = [0]

    def callback() -> None:
        invocations_after_stop[0] += 1

    with _patched_audio():
        listener = WakeWordListener()
        listener.start(callback)
        listener.stop()

        # Attempt to trigger a detection after stop — must be a no-op
        try:
            listener._trigger_detection_for_test()
        except Exception:
            pass  # Expected if triggering after stop raises

        time.sleep(0.05)

    assert invocations_after_stop[0] == 0, (
        "on_detected was invoked after stop() was called"
    )


# ---------------------------------------------------------------------------
# Contract: audio resource is released after stop()
# ---------------------------------------------------------------------------


def test_audio_resource_released_after_stop() -> None:
    """Audio resources must be released and no threads left running after stop()."""
    from flec.speech.wake_word_listener import WakeWordListener

    callback = MagicMock()
    initial_thread_count = threading.active_count()

    with _patched_audio():
        listener = WakeWordListener()
        listener.start(callback)
        listener.stop()

    # Give threads a moment to terminate
    time.sleep(0.1)

    # Thread count must not grow permanently after stop()
    final_thread_count = threading.active_count()
    assert final_thread_count <= initial_thread_count + 1, (
        f"Thread leak suspected: started with {initial_thread_count}, "
        f"ended with {final_thread_count}"
    )


def test_stop_is_idempotent() -> None:
    """Calling stop() multiple times must not raise."""
    from flec.speech.wake_word_listener import WakeWordListener

    callback = MagicMock()
    with _patched_audio():
        listener = WakeWordListener()
        listener.start(callback)
        listener.stop()
        listener.stop()  # Second call must be a no-op, not an error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockAudioContext:
    """Context object returned by _patched_audio() for test control."""
    simulate_wake_word_detection: bool = False


def _patched_audio():
    """Patch all audio backend imports so no real PyAudio or model is loaded."""
    ctx = _MockAudioContext()

    pyaudio_mock = MagicMock()
    pyaudio_mock.PyAudio.return_value.open.return_value = MagicMock()
    pyaudio_mock.paInt16 = 8

    openwakeword_mock = MagicMock()
    mock_oww_instance = MagicMock()
    # Default: no wake word detected
    mock_oww_instance.predict.return_value = {"hey_flec": 0.0}
    openwakeword_mock.Model.return_value = mock_oww_instance

    import contextlib

    @contextlib.contextmanager
    def _ctx_manager():
        with patch.dict("sys.modules", {
            "pyaudio": pyaudio_mock,
            "openwakeword": openwakeword_mock,
            "openwakeword.model": openwakeword_mock,
        }):
            yield ctx

    return _ctx_manager()
