"""MicListener — background microphone capture → parsed VoiceCommands.

Captures audio from the default input device with ``sounddevice`` and uses a
simple energy-based VAD to segment speech. Each speech segment is transcribed
by CommandSTT (Whisper tiny) and, when it parses to a non-UNKNOWN intent, is
handed to the ``on_command`` callback.

This is the microphone front-end that feeds the queue-decoupled pipeline: it
does not import other capability modules — it only produces VoiceCommand
objects for the session to route.

Runs on a daemon thread. Never raises into the caller — degrades to a no-op if
sounddevice, the input device, or the Whisper model is unavailable (logs a
warning). Audio is transcribed and discarded; nothing is persisted.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Optional

from flec.models import CommandIntent, VoiceCommand

logger = logging.getLogger(__name__)

# 16 kHz mono is what Whisper expects; 100 ms blocks give responsive VAD.
_SAMPLE_RATE = 16000
_BLOCK = 1600                 # samples per read (~100 ms)
_ENERGY_THRESHOLD = 0.015     # RMS onset threshold (float32 [-1, 1] audio)
_SILENCE_BLOCKS = 8           # ~0.8 s of silence ends a segment
_MIN_SPEECH_BLOCKS = 3        # ignore < ~0.3 s blips
_MAX_SEGMENT_BLOCKS = 50      # ~5 s hard cap per utterance


class MicListener:
    """Continuously listen for spoken commands and dispatch parsed intents."""

    def __init__(
        self,
        command_stt,
        on_command: Callable[[VoiceCommand], None],
        device: Optional[int] = None,
        energy_threshold: float = _ENERGY_THRESHOLD,
    ) -> None:
        self._stt = command_stt
        self._on_command = on_command
        self._device = device
        self._threshold = energy_threshold
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sd = None
        self._np = None

    def start(self) -> bool:
        """Start the listener thread. Returns False (no-op) if unavailable."""
        try:
            import numpy as np
            import sounddevice as sd

            self._sd = sd
            self._np = np
        except Exception as exc:  # noqa: BLE001 — degrade, never crash session
            logger.warning(json.dumps({
                "event": "mic.unavailable", "error": str(exc),
            }))
            return False

        self._thread = threading.Thread(
            target=self._run, name="flec-mic", daemon=True
        )
        self._thread.start()
        logger.info(json.dumps({
            "event": "mic.started",
            "sample_rate": _SAMPLE_RATE,
            "threshold": self._threshold,
        }))
        return True

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            with self._sd.InputStream(  # type: ignore[union-attr]
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=_BLOCK,
                device=self._device,
            ) as stream:
                self._loop(stream)
        except Exception as exc:  # noqa: BLE001 — mic thread must never crash
            logger.warning(json.dumps({
                "event": "mic.stream_error", "error": str(exc),
            }))

    def _loop(self, stream) -> None:
        np = self._np
        segment = []
        silence = 0
        in_speech = False

        while not self._stop.is_set():
            data, _overflow = stream.read(_BLOCK)
            block = data[:, 0]
            rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0

            if rms >= self._threshold:
                in_speech = True
                silence = 0
                segment.append(block)
            elif in_speech:
                silence += 1
                segment.append(block)
                if silence >= _SILENCE_BLOCKS:
                    self._flush(segment)
                    segment, silence, in_speech = [], 0, False
                    continue
            # else: ambient silence before any speech → drop the block

            if len(segment) >= _MAX_SEGMENT_BLOCKS:
                self._flush(segment)
                segment, silence, in_speech = [], 0, False

    def _flush(self, segment) -> None:
        """Transcribe a completed speech segment and dispatch any command."""
        if len(segment) < _MIN_SPEECH_BLOCKS:
            return
        np = self._np
        audio = np.concatenate(segment)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        cmd = self._stt.transcribe(pcm)
        if cmd.intent == CommandIntent.UNKNOWN:
            return

        logger.info(json.dumps({
            "event": "mic.command",
            "intent": cmd.intent.name,
            "raw_text": cmd.raw_text,
        }))
        try:
            self._on_command(cmd)
        except Exception as exc:  # noqa: BLE001
            logger.warning(json.dumps({
                "event": "mic.callback_error", "error": str(exc),
            }))
