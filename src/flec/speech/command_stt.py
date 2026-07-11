"""CommandSTT — speech-to-text command parser for Flec.

Transcribes a short audio segment (post-wake-word) into a VoiceCommand.

Never raises — returns VoiceCommand(intent=UNKNOWN) on any failure.
Observability: structured JSON log on every transcription.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from flec.models import ChallengeTargetType, CommandIntent, VoiceCommand

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword tables (spec §FR-006c, contracts/module-interfaces.md)
# ---------------------------------------------------------------------------

_COLOR_KEYWORDS: frozenset[str] = frozenset({
    "red", "blue", "yellow", "green", "orange", "purple", "pink", "white",
})

_SHAPE_KEYWORDS: frozenset[str] = frozenset({
    "circle", "triangle", "square", "rectangle", "pentagon",
    "hexagon", "star", "heart", "oval", "diamond",
})

# Patterns that signal intent to cancel / stop the current challenge
_CANCEL_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bstop\b", re.IGNORECASE),
    re.compile(r"\bcancel\b", re.IGNORECASE),
)

# "off" → shutdown (may appear alone or as trailing word after "hey flec")
_SHUTDOWN_PATTERN: re.Pattern = re.compile(
    r"(?:hey\s+flec\s+)?off\b", re.IGNORECASE
)

# "find [something|a|an|the|one]? <target>" — loosely matched
_FIND_PATTERN: re.Pattern = re.compile(
    r"\bfind\b(?:\s+(?:something|a|an|the|one))?\s+(\w+)",
    re.IGNORECASE,
)

# Spoken mode-switch commands: caregiver says the mode name to change modes
# on the fly ("exploration", "reading", "story", "challenge"). Checked after
# find/cancel/shutdown so "stop the challenge" and "find a red" keep their
# more-specific intents.
_MODE_PATTERNS: tuple[tuple[re.Pattern, CommandIntent], ...] = (
    (re.compile(r"\b(?:explore|exploration|explorer|look\s+around)\b", re.IGNORECASE),
     CommandIntent.SWITCH_EXPLORATION),
    (re.compile(r"\b(?:reading|read)\b", re.IGNORECASE),
     CommandIntent.SWITCH_READING),
    (re.compile(r"\b(?:story|storytime|story\s+time|book)\b", re.IGNORECASE),
     CommandIntent.SWITCH_STORY),
    (re.compile(r"\b(?:challenge|game|play)\b", re.IGNORECASE),
     CommandIntent.SWITCH_CHALLENGE),
)


# ---------------------------------------------------------------------------
# CommandSTT
# ---------------------------------------------------------------------------


class CommandSTT:
    """Parse caregiver voice commands into structured VoiceCommand objects.

    Primary entry points:
    - ``transcribe_text(text)``: parse a plain-text transcript (used in tests
      and by STT backends that already converted audio → text).
    - ``transcribe(audio_segment)``: accepts raw PCM bytes. Internally calls
      Whisper or a test stub; falls back gracefully to UNKNOWN on any error.
    """

    def __init__(self, whisper_model: Optional[object] = None) -> None:
        """Initialise the parser.

        Args:
            whisper_model: Optional pre-loaded Whisper model.  When ``None``
                (default) the module operates in text-only mode — suitable for
                tests and environments without a GPU.
        """
        self._whisper = whisper_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_segment: bytes) -> VoiceCommand:
        """Transcribe raw PCM audio (16 kHz mono) into a VoiceCommand.

        Never raises.  Returns ``VoiceCommand(intent=UNKNOWN)`` on any failure.
        """
        if not audio_segment:
            return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text="")

        if self._whisper is not None:
            try:
                # Whisper integration path (production)
                text = self._run_whisper(audio_segment)
                return self.transcribe_text(text)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    json.dumps({
                        "event": "command_stt.transcribe_error",
                        "error": str(exc),
                    })
                )
                return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text="")

        # No Whisper model available → UNKNOWN (test / stub mode)
        logger.debug(
            json.dumps({
                "event": "command_stt.no_whisper_model",
                "msg": "Returning UNKNOWN — no Whisper model loaded",
            })
        )
        return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text="")

    def transcribe_text(self, text: str) -> VoiceCommand:
        """Parse a plain-text transcript into a VoiceCommand.

        This is the primary logic entry point and is used directly in tests.
        Never raises.
        """
        raw = text  # preserve original for logging + raw_text
        try:
            cmd = self._parse(text)
            logger.info(
                json.dumps({
                    "event": "command_stt.parsed",
                    "raw_text": raw,
                    "intent": cmd.intent.name,
                    "target_label": cmd.target_label,
                    "target_type": cmd.target_type.name if cmd.target_type else None,
                })
            )
            return cmd
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({
                    "event": "command_stt.parse_error",
                    "raw_text": raw,
                    "error": str(exc),
                })
            )
            return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text=raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse(self, text: str) -> VoiceCommand:
        """Core parsing logic — maps text to CommandIntent."""
        stripped = text.strip()

        # 1. Shutdown check first — "off" is a standalone keyword
        if _SHUTDOWN_PATTERN.fullmatch(stripped) or _SHUTDOWN_PATTERN.search(stripped):
            # Only treat as shutdown if 'off' is the *main* intent (no find/cancel)
            if not any(p.search(stripped) for p in _CANCEL_PATTERNS):
                if not _FIND_PATTERN.search(stripped):
                    if re.search(r"\boff\b", stripped, re.IGNORECASE):
                        return VoiceCommand(
                            intent=CommandIntent.SHUTDOWN,
                            raw_text=text,
                        )

        # 2. Cancel / stop
        for pattern in _CANCEL_PATTERNS:
            if pattern.search(stripped):
                return VoiceCommand(
                    intent=CommandIntent.CANCEL_CHALLENGE,
                    raw_text=text,
                )

        # 3. Start challenge — "find [something|a|an] <keyword>"
        match = _FIND_PATTERN.search(stripped)
        if match:
            candidate = match.group(1).lower()
            target_label, target_type = self._resolve_target(candidate)
            if target_label is not None:
                return VoiceCommand(
                    intent=CommandIntent.START_CHALLENGE,
                    target_label=target_label,
                    target_type=target_type,
                    raw_text=text,
                )

        # 4. Mode switch — caregiver says a mode name to change modes on the fly
        for pattern, mode_intent in _MODE_PATTERNS:
            if pattern.search(stripped):
                return VoiceCommand(intent=mode_intent, raw_text=text)

        # 5. Unknown
        return VoiceCommand(intent=CommandIntent.UNKNOWN, raw_text=text)

    def _resolve_target(
        self, word: str
    ) -> tuple[Optional[str], Optional[ChallengeTargetType]]:
        """Classify a candidate "find X" word as COLOR, SHAPE, or OBJECT.

        Colors and shapes match the fixed spec vocabularies; anything else is
        treated as a real-world OBJECT target (matched against YOLO detections
        by the ResponseEngine). Returns (label, target_type); label is the
        lowercase canonical form.
        """
        word_lower = word.lower()
        if word_lower in _COLOR_KEYWORDS:
            return word_lower, ChallengeTargetType.COLOR
        if word_lower in _SHAPE_KEYWORDS:
            return word_lower, ChallengeTargetType.SHAPE
        return word_lower, ChallengeTargetType.OBJECT

    def _run_whisper(self, audio_segment: bytes) -> str:
        """Execute Whisper transcription. Called only when model is loaded."""
        # Import is deferred to avoid loading heavy dependencies at module init.
        # In production the model is pre-warmed at boot.
        import io
        import tempfile

        import numpy as np

        # PCM 16-bit LE → float32 normalised
        audio_np = np.frombuffer(audio_segment, dtype=np.int16).astype(np.float32) / 32768.0
        result = self._whisper.transcribe(audio_np, language="en")  # type: ignore[union-attr]
        return result.get("text", "") if isinstance(result, dict) else str(result)
