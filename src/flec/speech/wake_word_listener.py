"""WakeWordListener — continuously listen for "Hey Flec" wake word.

Uses openWakeWord with a background PyAudio thread. Calls the on_detected
callback (thread-safe) when the wake word is recognized. Provides clean
resource release on stop().

Contract (module-interfaces.md):
    start(on_detected)  — begin listening; non-blocking
    stop()              — stop listening, release audio resource
    is_listening        — True if actively listening

Architecture:
    - Modular AI: no imports from other capability modules
    - Privacy: audio frames are processed in-memory and discarded immediately
    - Observability: structured JSON log on start, stop, and detection events
    - Toddler-First: callback errors are caught and logged, never raised
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from flec.logger import log_event

logger = logging.getLogger(__name__)

# Audio capture configuration
_SAMPLE_RATE: int = 16_000      # Hz — required by openWakeWord
_CHUNK_SIZE: int = 1_280        # Samples per chunk (~80ms at 16kHz)
_CHANNELS: int = 1              # Mono
_AUDIO_FORMAT_INT16: int = 8    # pyaudio.paInt16 value

# Wake word model name
_WAKE_WORD_MODEL: str = "hey_flec"

# Confidence threshold for wake word activation
_DETECTION_THRESHOLD: float = 0.5


class WakeWordListener:
    """Listen for the "Hey Flec" wake word on a background audio thread.

    Usage::

        listener = WakeWordListener()
        listener.start(on_detected=my_callback)
        # ... later ...
        listener.stop()
    """

    def __init__(self) -> None:
        self._callback: Optional[Callable[[], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._is_listening: bool = False

        log_event(
            module="WakeWordListener",
            event_type="initialized",
            data={},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self, on_detected: Callable[[], None]) -> None:
        """Begin listening for wake word.

        Non-blocking — spawns a background daemon thread.

        Args:
            on_detected: Thread-safe callable invoked when wake word is detected.
                         Called with no arguments.
        """
        if self._is_listening:
            log_event(
                module="WakeWordListener",
                event_type="start_ignored",
                data={"reason": "already_listening"},
            )
            return

        self._callback = on_detected
        self._stop_event.clear()
        self._is_listening = True

        self._thread = threading.Thread(
            target=self._listen_loop,
            name="WakeWordListener",
            daemon=True,
        )
        self._thread.start()

        log_event(
            module="WakeWordListener",
            event_type="listening_started",
            data={"model": _WAKE_WORD_MODEL, "threshold": _DETECTION_THRESHOLD},
        )

    def stop(self) -> None:
        """Stop listening and release the audio resource.

        Idempotent — safe to call multiple times.
        """
        if not self._is_listening:
            return

        self._is_listening = False
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        log_event(
            module="WakeWordListener",
            event_type="listening_stopped",
            data={},
        )

    @property
    def is_listening(self) -> bool:
        """True if actively listening for wake word."""
        return self._is_listening

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Audio capture + wake word inference loop (runs in background thread)."""
        import pyaudio
        import openwakeword

        audio: Optional[object] = None
        stream: Optional[object] = None
        model = None

        try:
            # Initialize openWakeWord model
            model = openwakeword.Model(
                wakeword_models=[_WAKE_WORD_MODEL],
                enable_speex_noise_suppression=False,
            )

            # Initialize PyAudio
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=_CHANNELS,
                rate=_SAMPLE_RATE,
                input=True,
                frames_per_buffer=_CHUNK_SIZE,
            )

            log_event(
                module="WakeWordListener",
                event_type="audio_stream_opened",
                data={"sample_rate": _SAMPLE_RATE, "chunk_size": _CHUNK_SIZE},
            )

            while not self._stop_event.is_set():
                try:
                    pcm_bytes = stream.read(_CHUNK_SIZE, exception_on_overflow=False)
                except Exception as read_err:
                    log_event(
                        module="WakeWordListener",
                        event_type="audio_read_error",
                        data={"error": str(read_err)},
                    )
                    continue

                predictions = model.predict(pcm_bytes)
                score = predictions.get(_WAKE_WORD_MODEL, 0.0)

                if score >= _DETECTION_THRESHOLD:
                    log_event(
                        module="WakeWordListener",
                        event_type="wake_word_detected",
                        data={"score": round(score, 3)},
                    )
                    self._invoke_callback()

        except Exception as exc:
            log_event(
                module="WakeWordListener",
                event_type="listen_loop_error",
                data={"error": str(exc)},
            )
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if audio is not None:
                try:
                    audio.terminate()
                except Exception:
                    pass
            log_event(
                module="WakeWordListener",
                event_type="audio_stream_closed",
                data={},
            )

    def _invoke_callback(self) -> None:
        """Invoke on_detected callback, catching any errors to prevent crash."""
        if self._callback is None or not self._is_listening:
            return
        try:
            self._callback()
        except Exception as exc:
            log_event(
                module="WakeWordListener",
                event_type="callback_error",
                data={"error": str(exc)},
            )

    # ------------------------------------------------------------------
    # Test support
    # ------------------------------------------------------------------

    def _trigger_detection_for_test(self) -> None:
        """Directly invoke the wake word callback for unit testing.

        Bypasses audio capture. MUST NOT be called in production code.
        Only fires if is_listening is True.
        """
        if not self._is_listening:
            return
        log_event(
            module="WakeWordListener",
            event_type="test_trigger",
            data={},
        )
        self._invoke_callback()
