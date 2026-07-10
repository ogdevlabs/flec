"""TTSEngine — offline text-to-speech playback for Flec.

Satisfies the ``_TTSProtocol`` expected by ResponseEngine (``speak`` +
``stop_current``). Audio is synthesised and played on a background daemon
thread; higher-priority responses preempt whatever is currently playing.

Backends, tried in order (graceful degradation, never crashes the session):
  1. Coqui VITS  — offline neural TTS, the production path on ARM64 Linux and
     macOS. Needs the ``espeak-ng`` system binary for phonemisation
     (Linux: ``apt install espeak-ng``; macOS dev: ``brew install espeak-ng``).
  2. macOS ``say`` — zero-install dev fallback on Darwin only.
  3. log-only    — always available; logs the text so the pipeline still runs
     with no audio device.

Privacy: synthesised audio is played and discarded; nothing is persisted
(Constitution Rule 4).
"""

from __future__ import annotations

import itertools
import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Optional

from flec.models import AudioPriority, AudioResponse

logger = logging.getLogger(__name__)

# Narration (LOW/NORMAL) is coalesced and time-limited so the mask speaks about
# what is in view *now*, never a stale backlog. HIGH/CRITICAL responses (mode
# confirmations, wear prompts, challenge celebrations) are never dropped.
_NARRATION_MAX_PRIORITY = AudioPriority.NORMAL.value
_MAX_NARRATION_AGE_S = 1.5


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _LogBackend:
    """Last-resort backend: emits the narration text as a structured log."""

    name = "log"

    def load(self) -> bool:
        return True

    def play(self, text: str) -> None:
        logger.info(json.dumps({"event": "tts.audio_response", "text": text}))

    def stop(self) -> None:
        pass


class _SayBackend:
    """macOS ``say`` backend — dev fallback only (Darwin)."""

    name = "say"

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None

    def load(self) -> bool:
        return sys.platform == "darwin" and shutil.which("say") is not None

    def play(self, text: str) -> None:
        self._proc = subprocess.Popen(["say", text])
        self._proc.wait()
        self._proc = None

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


class _CoquiBackend:
    """Coqui VITS backend — offline neural TTS (production path)."""

    name = "coqui"

    def __init__(self, model_name: str = "tts_models/en/ljspeech/vits") -> None:
        self._model_name = model_name
        self._tts = None
        self._sd = None
        self._np = None
        self._sample_rate = 22050

    def load(self) -> bool:
        try:
            import numpy as np
            import sounddevice as sd
            from TTS.api import TTS  # heavy import, done once at load

            self._tts = TTS(self._model_name)
            self._sd = sd
            self._np = np
            self._sample_rate = int(
                getattr(self._tts.synthesizer, "output_sample_rate", 22050)
            )
            logger.info(json.dumps({
                "event": "tts.coqui_loaded",
                "model": self._model_name,
                "sample_rate": self._sample_rate,
            }))
            return True
        except Exception as exc:  # noqa: BLE001 — degrade to next backend
            logger.warning(json.dumps({
                "event": "tts.coqui_unavailable",
                "error": str(exc),
                "hint": "install espeak-ng and ensure a network path for first model fetch",
            }))
            return False

    def play(self, text: str) -> None:
        wav = self._tts.tts(text=text)  # type: ignore[union-attr]
        arr = self._np.asarray(wav, dtype="float32")  # type: ignore[union-attr]
        self._sd.play(arr, samplerate=self._sample_rate)  # type: ignore[union-attr]
        self._sd.wait()  # type: ignore[union-attr]

    def stop(self) -> None:
        try:
            if self._sd is not None:
                self._sd.stop()
        except Exception:  # noqa: BLE001
            pass


_BACKEND_CHAINS = {
    "coqui": (_CoquiBackend, _SayBackend, _LogBackend),
    "say": (_SayBackend, _LogBackend),
    "off": (_LogBackend,),
}


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------


class TTSEngine:
    """Priority-aware background TTS player.

    Usage::

        tts = TTSEngine(backend="coqui")
        engine = ResponseEngine(tts=tts)
        ...
        tts.shutdown()
    """

    def __init__(self, backend: str = "coqui") -> None:
        self._backend = self._select_backend(backend)
        # Pending items: (priority:int, seq:int, enqueued_at:float, response).
        self._pending: list = []
        self._counter = itertools.count()
        self._current_priority = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="flec-tts", daemon=True
        )
        self._thread.start()

    def _select_backend(self, name: str):
        chain = _BACKEND_CHAINS.get(name, _BACKEND_CHAINS["coqui"])
        for cls in chain:
            backend = cls()
            if backend.load():
                logger.info(json.dumps({
                    "event": "tts.backend_selected", "backend": backend.name,
                }))
                return backend
        # _LogBackend.load() is always True, so this is unreachable in practice.
        return _LogBackend()

    # -- _TTSProtocol -------------------------------------------------------

    def speak(self, response: AudioResponse) -> None:
        """Queue a response for playback.

        Narration (<= NORMAL) is *coalesced*: a new narration drops any pending
        narration so the mask only ever has the latest thing-in-view queued —
        it can't build a backlog that plays after the object has gone.
        HIGH/CRITICAL responses are always kept and preempt current playback.
        """
        priority = response.priority.value
        with self._lock:
            if priority <= _NARRATION_MAX_PRIORITY:
                # Drop stale narration; keep queued HIGH/CRITICAL.
                self._pending = [
                    e for e in self._pending if e[0] > _NARRATION_MAX_PRIORITY
                ]
            if priority > self._current_priority:
                self._backend.stop()  # cut current lower-priority playback
            self._pending.append(
                (priority, next(self._counter), time.monotonic(), response)
            )
        self._wake.set()

    def clear_pending(self) -> None:
        """Drop queued narration (e.g. on a mode switch). Keeps HIGH/CRITICAL."""
        with self._lock:
            dropped = [e for e in self._pending if e[0] <= _NARRATION_MAX_PRIORITY]
            self._pending = [
                e for e in self._pending if e[0] > _NARRATION_MAX_PRIORITY
            ]
        if dropped:
            logger.debug(json.dumps({
                "event": "tts.narration_cleared", "dropped": len(dropped),
            }))

    def stop_current(self) -> None:
        self._backend.stop()

    # -- lifecycle ----------------------------------------------------------

    def _next(self):
        """Pop the highest-priority pending item (FIFO within a priority)."""
        with self._lock:
            if not self._pending:
                return None
            self._pending.sort(key=lambda e: (-e[0], e[1]))
            return self._pending.pop(0)

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._next()
            if item is None:
                self._wake.wait(timeout=0.2)
                self._wake.clear()
                continue

            priority, _seq, enqueued_at, response = item
            # Drop narration that waited too long — the object is likely gone.
            if (
                priority <= _NARRATION_MAX_PRIORITY
                and (time.monotonic() - enqueued_at) > _MAX_NARRATION_AGE_S
            ):
                logger.debug(json.dumps({
                    "event": "tts.narration_stale_skipped", "text": response.text,
                }))
                continue

            with self._lock:
                self._current_priority = priority
            try:
                self._backend.play(response.text)
            except Exception as exc:  # noqa: BLE001 — never kill the TTS thread
                logger.warning(json.dumps({
                    "event": "tts.play_error", "error": str(exc),
                }))
            finally:
                with self._lock:
                    self._current_priority = 0

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        self._backend.stop()
