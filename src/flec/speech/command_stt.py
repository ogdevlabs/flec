"""CommandSTT — transcribe post-wake-word audio into a VoiceCommand.

Uses OpenAI Whisper (tiny model) to transcribe a short PCM audio segment,
then applies rule-based intent parsing to produce a typed VoiceCommand.

Contract (module-interfaces.md):
    transcribe(pcm: bytes) → VoiceCommand  — never raises; returns UNKNOWN on failure

Architecture:
    - Modular AI: no imports from other capability modules
    - Privacy: audio bytes processed in-memory and discarded; never stored
    - Observability: structured JSON log on every transcription and parse result
    - Toddler-First: all errors silently return UNKNOWN — no error reaches the child
"""

from __future__ import annotations

import logging
import re
import struct
from typing import Optional

import numpy as np

from flec.logger import log_event
from flec.models import (
    ChallengeTargetType,
    CommandIntent,
    VoiceCommand,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary for intent parsing
# ---------------------------------------------------------------------------

# Challenge target shapes (matches spec vocabulary)
_SHAPES = frozenset(
    {
        "circle", "square", "triangle", "rectangle", "star",
        "heart", "diamond", "oval", "pentagon", "hexagon",
    }
)

# Challenge target colors (matches spec vocabulary)
_COLORS = frozenset(
    {
        "red", "blue", "yellow", "green", "orange",
        "purple", "black", "white",
    }
)

# Phrases that indicate a SHUTDOWN intent
_SHUTDOWN_PHRASES = frozenset({"off", "shut down", "shutdown", "turn off", "stop listening"})

# Phrases that indicate CANCEL_CHALLENGE
_CANCEL_PHRASES = frozenset({"stop", "cancel", "never mind", "nevermind", "quit", "end"})

# Trigger words for START_CHALLENGE
_FIND_VERBS = frozenset({"find", "look for", "show me", "get", "get me", "find me"})


class CommandSTT:
    """Transcribe short audio commands and parse them into VoiceCommands.

    Args:
        model_name: Whisper model size to load. Defaults to "tiny" for speed.
    """

    def __init__(self, model_name: str = "tiny") -> None:
        self._model_name = model_name
        self._model = None  # Lazy load on first transcription

        log_event(
            module="CommandSTT",
            event_type="initialized",
            data={"model": model_name},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transcribe(self, audio_segment: bytes) -> VoiceCommand:
        """Transcribe audio_segment (PCM 16kHz mono) to a VoiceCommand.

        Never raises. Returns VoiceCommand(intent=UNKNOWN) on any failure.

        Args:
            audio_segment: Raw PCM audio bytes (16-bit signed, 16kHz, mono).

        Returns:
            Parsed VoiceCommand with intent and optional target_label.
        """
        try:
            text = self._transcribe_pcm(audio_segment)
            command = self._parse_command(text)

            log_event(
                module="CommandSTT",
                event_type="transcription_complete",
                data={
                    "raw_text": text,
                    "intent": command.intent.name,
                    "target_label": command.target_label,
                },
            )
            return command

        except Exception as exc:
            log_event(
                module="CommandSTT",
                event_type="transcription_error",
                data={"error": str(exc)},
            )
            return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text="")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._model_name)
            log_event(
                module="CommandSTT",
                event_type="whisper_model_loaded",
                data={"model": self._model_name},
            )
        return self._model

    def _transcribe_pcm(self, pcm_bytes: bytes) -> str:
        """Convert raw PCM bytes to a normalised lowercase transcription."""
        if not pcm_bytes:
            return ""

        # Convert PCM bytes (16-bit signed LE, 16kHz mono) to float32 numpy array
        n_samples = len(pcm_bytes) // 2
        if n_samples == 0:
            return ""

        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        model = self._get_model()
        result = model.transcribe(audio_float32, language="en", fp16=False)
        raw_text: str = result.get("text", "")

        return raw_text.strip().lower()

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    def _parse_command(self, text: str) -> VoiceCommand:
        """Parse a normalised text string into a VoiceCommand.

        Priority order:
        1. SHUTDOWN ("off" / "turn off")
        2. CANCEL_CHALLENGE ("stop" / "cancel")
        3. START_CHALLENGE ("find a triangle" / "find something red")
        4. UNKNOWN (fallback)
        """
        text = text.strip().lower()

        # 1. Shutdown
        if self._matches_any(text, _SHUTDOWN_PHRASES):
            return VoiceCommand(intent=CommandIntent.SHUTDOWN, raw_text=text)

        # 2. Cancel challenge (check BEFORE start_challenge because "stop" is common)
        if self._matches_any(text, _CANCEL_PHRASES):
            return VoiceCommand(intent=CommandIntent.CANCEL_CHALLENGE, raw_text=text)

        # 3. Start challenge — look for find-verb + target
        start_cmd = self._parse_start_challenge(text)
        if start_cmd is not None:
            return start_cmd

        # 4. Unknown
        return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text=text)

    def _matches_any(self, text: str, phrases: frozenset) -> bool:
        """Return True if any phrase appears as a word boundary in text."""
        for phrase in phrases:
            # Whole-word match using word boundaries
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                return True
        return False

    def _parse_start_challenge(self, text: str) -> Optional[VoiceCommand]:
        """Try to extract a START_CHALLENGE command with a shape or color target.

        Patterns handled:
        - "find a triangle"
        - "find something red"
        - "find the blue one"
        - "look for a heart"
        - "show me a circle"
        """
        # Check for a find-verb in the text
        has_find_verb = self._matches_any(text, _FIND_VERBS)
        if not has_find_verb:
            return None

        # Extract target shape
        for shape in _SHAPES:
            if re.search(r"\b" + re.escape(shape) + r"\b", text):
                return VoiceCommand(
                    intent=CommandIntent.START_CHALLENGE,
                    target_label=shape,
                    target_type=ChallengeTargetType.SHAPE,
                    raw_text=text,
                )

        # Extract target color
        for color in _COLORS:
            if re.search(r"\b" + re.escape(color) + r"\b", text):
                return VoiceCommand(
                    intent=CommandIntent.START_CHALLENGE,
                    target_label=color,
                    target_type=ChallengeTargetType.COLOR,
                    raw_text=text,
                )

        return None
