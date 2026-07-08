"""TTSEngine — priority audio playback with pre-cached WAV support.

Converts text to speech using Coqui VITS and plays audio via sounddevice.
Pre-cached WAV keys play instantly from disk (no synthesis delay).
CRITICAL priority responses pre-empt lower-priority playback.

Contract:
    speak(response)           — enqueue AudioResponse (non-blocking)
    stop_current()            — halt playback, clear NORMAL and LOW queue
    preload_cache(responses)  — pre-render and cache {key: text} as WAV
    is_speaking               — True if audio is currently playing

Architecture:
    - All audio output is ephemeral (never persisted — Constitution Rule 4)
    - No direct imports from other capability modules
    - Background playback thread processes priority queue
"""

from __future__ import annotations

import io
import logging
import os
import queue
import tempfile
import threading
import time
from typing import Any, Optional

from flec.logger import log_event
from flec.models import AudioPriority, AudioResponse

logger = logging.getLogger(__name__)

# Optional heavy imports — guarded for test environments
try:
    import sounddevice  # type: ignore[import]
    import soundfile  # type: ignore[import]
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    from TTS.api import TTS  # type: ignore[import]
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False


class TTSEngine:
    """Priority audio playback engine with pre-cached WAV support.

    Args:
        model_name: Coqui TTS model to use (default: a lightweight VITS model).
        use_mock:   If True, skip actual TTS/audio device initialisation.
                    Used in tests.
    """

    _DEFAULT_MODEL = "tts_models/en/ljspeech/vits"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        use_mock: bool = False,
    ) -> None:
        self._use_mock = use_mock
        self._tts_model: Any = None
        self._wav_cache: dict[str, Any] = {}  # key → (data, samplerate)
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._speaking: bool = False
        self._speaking_lock: threading.Lock = threading.Lock()
        self._stop_event: threading.Event = threading.Event()
        self._current_stop: threading.Event = threading.Event()

        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="flec-tts-playback",
            daemon=True,
        )
        self._playback_thread.start()

        if not use_mock:
            self._init_tts(model_name)

        log_event(
            module="TTSEngine",
            event_type="engine_initialized",
            data={"use_mock": use_mock},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def speak(self, response: AudioResponse) -> None:
        """Enqueue an AudioResponse for playback (non-blocking).

        Higher priority responses pre-empt lower priority ones.
        """
        # Priority queue uses (priority_value, sequence, item) tuples.
        # Negate AudioPriority value so CRITICAL (4) → -4 (lowest in min-heap = highest priority)
        seq = time.monotonic()
        self._queue.put((-response.priority.value, seq, response))

        log_event(
            module="TTSEngine",
            event_type="audio_enqueued",
            data={
                "priority": response.priority.name,
                "pre_cached": response.pre_cached,
                "cache_key": response.cache_key,
                "text_preview": response.text[:40],
            },
        )

    def stop_current(self) -> None:
        """Immediately stop current playback. Clears NORMAL and LOW queue."""
        self._current_stop.set()

        if _SD_AVAILABLE and not self._use_mock:
            try:
                sounddevice.stop()
            except Exception:
                pass

        # Drain NORMAL and LOW priority items from the queue
        remaining = []
        while not self._queue.empty():
            try:
                neg_prio, seq, item = self._queue.get_nowait()
                priority_value = -neg_prio
                if priority_value >= AudioPriority.HIGH.value:
                    remaining.append((-neg_prio, seq, item))
            except queue.Empty:
                break

        for entry in remaining:
            self._queue.put(entry)

        self._set_speaking(False)

        log_event(
            module="TTSEngine",
            event_type="playback_stopped",
            data={"remaining_high_priority": len(remaining)},
        )

    def preload_cache(self, responses: dict[str, str]) -> None:
        """Pre-render and cache {key: text} pairs as audio data at startup.

        If TTS is not available (e.g. in tests), stores text as placeholder.
        """
        for key, text in responses.items():
            if self._use_mock or not _TTS_AVAILABLE:
                # In mock/test mode store placeholder data
                self._wav_cache[key] = ([0.0] * 1000, 22050)
            else:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        tmp_path = f.name

                    self._tts_model.tts_to_file(text=text, file_path=tmp_path)
                    data, samplerate = soundfile.read(tmp_path)
                    self._wav_cache[key] = (data, samplerate)
                    os.unlink(tmp_path)

                    log_event(
                        module="TTSEngine",
                        event_type="cache_preloaded",
                        data={"key": key, "text_preview": text[:40]},
                    )
                except Exception as e:
                    log_event(
                        module="TTSEngine",
                        event_type="cache_preload_failed",
                        data={"key": key, "error": str(e)},
                    )

    @property
    def is_speaking(self) -> bool:
        """True if audio is currently playing."""
        with self._speaking_lock:
            return self._speaking

    # ------------------------------------------------------------------
    # Test/introspection helpers (used by contract tests)
    # ------------------------------------------------------------------

    def has_cached(self, key: str) -> bool:
        """Return True if the given key is present in the WAV cache."""
        return key in self._wav_cache

    def peek_highest_priority(self) -> Optional[AudioPriority]:
        """Return the highest AudioPriority currently in the queue, or None."""
        if self._queue.empty():
            return None
        # Build a sorted snapshot without consuming the queue
        items = []
        while not self._queue.empty():
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break

        # Return all items to the queue
        for item in items:
            self._queue.put(item)

        if not items:
            return None

        # Highest priority = most negative neg_prio
        best = min(items, key=lambda t: t[0])
        return AudioPriority(-best[0])

    def queue_size(self) -> int:
        """Return the current number of items in the playback queue."""
        return self._queue.qsize()

    def _set_speaking(self, value: bool) -> None:
        """Set the is_speaking flag (callable from tests and background thread)."""
        with self._speaking_lock:
            self._speaking = value

    # ------------------------------------------------------------------
    # Internal playback loop
    # ------------------------------------------------------------------

    def _init_tts(self, model_name: str) -> None:
        """Initialise the Coqui TTS model (may take a few seconds)."""
        if not _TTS_AVAILABLE:
            log_event(
                module="TTSEngine",
                event_type="tts_library_unavailable",
                data={"reason": "TTS package not installed"},
            )
            return
        try:
            self._tts_model = TTS(model_name=model_name, progress_bar=False)
            log_event(
                module="TTSEngine",
                event_type="tts_model_loaded",
                data={"model": model_name},
            )
        except Exception as e:
            log_event(
                module="TTSEngine",
                event_type="tts_model_load_failed",
                data={"model": model_name, "error": str(e)},
            )

    def _playback_loop(self) -> None:
        """Background thread: dequeue and play AudioResponses in priority order."""
        while not self._stop_event.is_set():
            try:
                neg_prio, seq, response = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._current_stop.clear()
            self._set_speaking(True)

            try:
                if response.pre_cached and response.cache_key in self._wav_cache:
                    self._play_cached(response.cache_key)
                else:
                    self._synthesize_and_play(response.text)
            except Exception as e:
                log_event(
                    module="TTSEngine",
                    event_type="playback_error",
                    data={"error": str(e), "text_preview": response.text[:40]},
                )
            finally:
                self._set_speaking(False)

    def _play_cached(self, key: str) -> None:
        """Play pre-cached audio data for the given key."""
        data, samplerate = self._wav_cache[key]

        if self._use_mock or not _SD_AVAILABLE:
            return

        sounddevice.play(data, samplerate)
        # Wait for playback to finish or stop signal
        while sounddevice.get_stream().active:
            if self._current_stop.is_set():
                sounddevice.stop()
                break
            time.sleep(0.01)

        log_event(
            module="TTSEngine",
            event_type="cached_audio_played",
            data={"key": key},
        )

    def _synthesize_and_play(self, text: str) -> None:
        """Synthesize text via Coqui TTS and play immediately."""
        if self._use_mock or not _TTS_AVAILABLE or self._tts_model is None:
            return

        try:
            wav = self._tts_model.tts(text=text)

            if not _SD_AVAILABLE:
                return

            import numpy as np
            audio_data = np.array(wav, dtype=np.float32)
            samplerate = 22050

            sounddevice.play(audio_data, samplerate)
            while sounddevice.get_stream().active:
                if self._current_stop.is_set():
                    sounddevice.stop()
                    break
                time.sleep(0.01)

            log_event(
                module="TTSEngine",
                event_type="synthesized_audio_played",
                data={"text_preview": text[:40]},
            )
        except Exception as e:
            log_event(
                module="TTSEngine",
                event_type="synthesis_error",
                data={"error": str(e), "text_preview": text[:40]},
            )
