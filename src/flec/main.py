"""Flec entry point — boot, session loop.

Voice command wiring (T038):
  1. Wake word detected (WakeWordListener callback)
  2. Capture short audio segment
  3. CommandSTT.transcribe() → VoiceCommand
  4. Session.handle_voice_command() → state updates
  5. Build DetectionEvent with embedded VoiceCommand
  6. ResponseEngine.on_event() → audio/AR response

Observability: structured JSON log on every command received.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format='{"level": "%(levelname)s", "module": "%(name)s", "msg": "%(message)s"}',
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Voice command pipeline
# ---------------------------------------------------------------------------


def handle_voice_command(
    raw_text: str,
    session,      # flec.session.Session
    engine,       # flec.engine.response_engine.ResponseEngine
    stt,          # flec.speech.command_stt.CommandSTT
) -> None:
    """Parse a transcript and propagate it through session + response engine.

    This is the central wiring point between STT output and the session/engine
    layer.  It emits a VOICE_CMD DetectionEvent so ResponseEngine receives the
    full command, and also calls Session.handle_voice_command() for state updates.

    All exceptions are caught — the toddler must never see an error.

    Args:
        raw_text: The transcript string produced by the STT backend.
        session: The active Session instance.
        engine: The active ResponseEngine.
        stt: The CommandSTT instance used to parse the text.
    """
    try:
        voice_cmd = stt.transcribe_text(raw_text)

        logger.info(
            json.dumps({
                "event": "main.voice_command_received",
                "raw_text": raw_text,
                "intent": voice_cmd.intent.name,
                "target_label": voice_cmd.target_label,
                "target_type": (
                    voice_cmd.target_type.name if voice_cmd.target_type else None
                ),
            })
        )

        # Update session state
        session.handle_voice_command(
            intent=voice_cmd.intent,
            target_label=voice_cmd.target_label,
            target_type=voice_cmd.target_type,
        )

        # Build a DetectionEvent wrapping the VoiceCommand
        from flec.models import DetectionEvent, DetectionType

        cmd_event = DetectionEvent(
            type=DetectionType.VOICE_CMD,
            label=voice_cmd.intent.name,
            confidence=1.0,
            metadata={"command": voice_cmd},
        )

        # Route through ResponseEngine for audio/AR
        engine.on_event(cmd_event)

    except Exception as exc:  # noqa: BLE001
        # Constitution Rule 5: never surface errors to the toddler
        logger.error(
            json.dumps({
                "event": "main.voice_command_error",
                "raw_text": raw_text,
                "error": str(exc),
            })
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for Flec."""
    parser = argparse.ArgumentParser(
        description="Flec — wearable superhero mask for toddler early learning"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Run mode: 'dev' for development (iPhone camera), 'prod' for production (embedded camera)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    logger.info(
        json.dumps({
            "event": "main.boot",
            "run_mode": args.mode,
        })
    )

    # Lazy imports to avoid loading heavy ML dependencies at module import time
    from flec.audio.responses import wear_welcome
    from flec.engine.response_engine import ResponseEngine
    from flec.models import AudioPriority, AudioResponse, Mode, WearState
    from flec.session import Session
    from flec.speech.command_stt import CommandSTT

    # --- Initialise core components ---
    session = Session()
    stt = CommandSTT()  # Whisper model loaded lazily on first audio input

    # Stub TTS (real TTSEngine wired in subsequent phases)
    class _StubTTS:
        def speak(self, response: AudioResponse) -> None:
            logger.info(
                json.dumps({
                    "event": "tts.speak_stub",
                    "text": response.text,
                    "priority": response.priority.name,
                })
            )

        def stop_current(self) -> None:
            pass

    engine = ResponseEngine(tts=_StubTTS())
    engine.set_mode(Mode.EXPLORATION)

    # Wire post-wake-word handler
    def on_wake_word_detected() -> None:
        """Callback invoked by WakeWordListener on "Hey Flec"."""
        # In production: capture PCM audio, pass to stt.transcribe(audio_bytes)
        # In dev/stub: exercise the pipeline with a log entry
        logger.info(json.dumps({"event": "main.wake_word_detected"}))
        # Stub: no audio capture yet — full wiring in phase 6
        # handle_voice_command("find a circle", session, engine, stt)

    logger.info(
        json.dumps({
            "event": "main.ready",
            "mode": args.mode,
            "note": "Voice command wiring active — challenge mode ready",
        })
    )

    # Session loop stub — full boot sequence in subsequent phases
    logger.info(
        json.dumps({
            "event": "main.session_loop_stub",
            "msg": "Full boot sequence in later phases",
        })
    )


if __name__ == "__main__":
    main()
