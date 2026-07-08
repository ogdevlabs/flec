"""ResponseEngine — single orchestration point for audio and AR responses.

Consumes DetectionEvents from all perception modules and emits AudioResponses
and AR overlay commands. No other module calls TTS or AR directly.

Constitution §III: Does NOT directly import capability modules (camera, CV, STT).
All perception results arrive as DetectionEvents.
"""

from __future__ import annotations

import json
import logging
import queue
from typing import Optional

from flec.models import (
    AudioPriority,
    AudioResponse,
    Challenge,
    ChallengeStatus,
    CommandIntent,
    DetectionEvent,
    DetectionType,
    Mode,
    ReadingIntent,
    WearState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum seconds between identical narration utterances (de-duplication).
_NARRATION_COOLDOWN_S: float = 3.0

#: Pre-cached response keys.
_CACHE_SHUTDOWN = "shutdown"
_CACHE_WEAR_OFF = "wear_off"
_CACHE_WELCOME = "welcome"


# ---------------------------------------------------------------------------
# ResponseEngine
# ---------------------------------------------------------------------------


class ResponseEngine:
    """Consume detection events and enqueue AudioResponses.

    Public interface (from module-interfaces.md):
        on_event(event: DetectionEvent) -> None
        set_mode(mode: Mode) -> None
        set_challenge(challenge: Challenge | None) -> None

    Extended interface for reading mode and testing:
        set_wear_state(state: WearState) -> None
        set_pending_illustration(description: str) -> None
    """

    def __init__(
        self,
        audio_queue: Optional[queue.Queue] = None,
    ) -> None:
        self._audio_queue: queue.Queue[AudioResponse] = (
            audio_queue if audio_queue is not None else queue.Queue()
        )
        self._mode: Mode = Mode.STANDBY
        self._wear_state: WearState = WearState.STANDBY
        self._challenge: Optional[Challenge] = None

        # De-duplication: track last narrated label per detection type.
        # key = (DetectionType, label), value = monotonic timestamp of last speak.
        self._last_spoken: dict[tuple, float] = {}

        # Reading mode: pending illustration description from IllustrationDescriber.
        self._pending_illustration: Optional[str] = None

        logger.info(json.dumps({"event": "response_engine_init"}))

    # ------------------------------------------------------------------
    # Public interface (module-interfaces.md)
    # ------------------------------------------------------------------

    def on_event(self, event: DetectionEvent) -> None:
        """Process a detection event and enqueue AudioResponse/AR commands as needed.

        State-aware: respects active mode, active challenge, and wear state.
        Never raises.
        """
        try:
            self._route(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps(
                    {
                        "event": "response_engine_route_error",
                        "error": str(exc),
                        "detection_type": event.type.name,
                        "label": event.label,
                    }
                )
            )

    def set_mode(self, mode: Mode) -> None:
        """Transition engine to a new operating mode."""
        prev = self._mode
        self._mode = mode
        logger.info(
            json.dumps(
                {"event": "mode_transition", "from": prev.name, "to": mode.name}
            )
        )

    def set_challenge(self, challenge: Optional[Challenge]) -> None:
        """Set or clear the active challenge."""
        self._challenge = challenge
        label = challenge.target_label if challenge else None
        logger.info(
            json.dumps({"event": "challenge_set", "target": label})
        )

    # ------------------------------------------------------------------
    # Extended interface
    # ------------------------------------------------------------------

    def set_wear_state(self, state: WearState) -> None:
        """Update the tracked wear state (used to gate shutdown commands)."""
        self._wear_state = state

    def set_pending_illustration(self, description: str) -> None:
        """Inject a pending illustration description for the next READING event."""
        self._pending_illustration = description

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route(self, event: DetectionEvent) -> None:  # noqa: PLR0912
        """Dispatch event to the appropriate handler based on type and mode."""
        etype = event.type

        if etype == DetectionType.WEAR:
            self._handle_wear(event)

        elif etype == DetectionType.VOICE_CMD:
            self._handle_voice_cmd(event)

        elif etype in (DetectionType.SHAPE, DetectionType.COLOR):
            self._handle_shape_color(event)

        elif etype == DetectionType.FINGER:
            self._handle_finger(event)

        elif etype == DetectionType.TEXT:
            self._handle_text(event)

        elif etype == DetectionType.ILLUSTRATION:
            self._handle_illustration(event)

        else:
            logger.debug(
                json.dumps(
                    {"event": "response_engine_unhandled", "type": etype.name}
                )
            )

    # ------------------------------------------------------------------
    # Wear event
    # ------------------------------------------------------------------

    def _handle_wear(self, event: DetectionEvent) -> None:
        """Handle wear state transitions."""
        label = event.label  # "on_head" or "off_head"
        if label == "off_head":
            self._enqueue(
                AudioResponse(
                    text="Put your mask back on, hero!",
                    priority=AudioPriority.CRITICAL,
                    pre_cached=True,
                    cache_key=_CACHE_WEAR_OFF,
                )
            )
            # Suspend all active modes.
            self.set_mode(Mode.STANDBY)
        elif label == "on_head":
            self._enqueue(
                AudioResponse(
                    text="Hero mask activated!",
                    priority=AudioPriority.CRITICAL,
                    pre_cached=True,
                    cache_key=_CACHE_WELCOME,
                )
            )
            self.set_mode(Mode.EXPLORATION)

        logger.info(
            json.dumps({"event": "wear_event", "label": label})
        )

    # ------------------------------------------------------------------
    # Voice command
    # ------------------------------------------------------------------

    def _handle_voice_cmd(self, event: DetectionEvent) -> None:
        """Handle parsed caregiver voice commands."""
        intent_str = event.metadata.get("intent", "UNKNOWN")
        # intent may be a CommandIntent enum or a string key.
        intent = (
            intent_str
            if isinstance(intent_str, CommandIntent)
            else CommandIntent[intent_str]
            if intent_str in CommandIntent.__members__
            else CommandIntent.UNKNOWN
        )

        if intent == CommandIntent.SHUTDOWN:
            if self._wear_state == WearState.ON_HEAD:
                self._enqueue(
                    AudioResponse(
                        text="See you next time, hero!",
                        priority=AudioPriority.CRITICAL,
                        pre_cached=True,
                        cache_key=_CACHE_SHUTDOWN,
                    )
                )
                self.set_mode(Mode.STANDBY)
            # Ignored if not worn (FR-001e).

        elif intent == CommandIntent.CANCEL_CHALLENGE:
            self.set_challenge(None)
            self.set_mode(Mode.EXPLORATION)

        elif intent == CommandIntent.START_CHALLENGE:
            target = event.metadata.get("target_label", "")
            self._enqueue(
                AudioResponse(
                    text=f"Ok! Let's find a {target}!",
                    priority=AudioPriority.HIGH,
                )
            )

    # ------------------------------------------------------------------
    # Shape / color detection (Exploration + Challenge modes)
    # ------------------------------------------------------------------

    def _handle_shape_color(self, event: DetectionEvent) -> None:
        """Handle shape/color detection events in EXPLORATION or CHALLENGE mode."""
        if self._mode not in (Mode.EXPLORATION, Mode.CHALLENGE):
            return

        label = event.label
        dedup_key = (event.type, label)

        # Challenge mode: check for match.
        if (
            self._mode == Mode.CHALLENGE
            and self._challenge is not None
            and self._challenge.status == ChallengeStatus.ACTIVE
        ):
            if label.lower() == self._challenge.target_label.lower():
                self._enqueue(
                    AudioResponse(
                        text=f"You found it! That's a {label}!",
                        priority=AudioPriority.HIGH,
                    )
                )
                return
            else:
                # Non-matching — play encouraging response (no cooldown needed; one per detection).
                self._enqueue(
                    AudioResponse(
                        text=f"Keep looking, hero! You're doing great!",
                        priority=AudioPriority.NORMAL,
                    )
                )
                return

        # Exploration mode — narrate if not recently spoken.
        if not self._is_recently_spoken(dedup_key):
            article = "an" if label[0].lower() in "aeiou" else "a"
            self._enqueue(
                AudioResponse(
                    text=f"I see {article} {label}!",
                    priority=AudioPriority.NORMAL,
                )
            )
            self._mark_spoken(dedup_key)

    # ------------------------------------------------------------------
    # Finger tracking (Reading mode)
    # ------------------------------------------------------------------

    def _handle_finger(self, event: DetectionEvent) -> None:
        """Handle FINGER_TIP detection events in READING mode.

        Routing rules (T043):
        - READING intent + nearest_text  → NORMAL narration AudioResponse
        - READING intent + no text + is_illustration=True → illustration description
        - SCANNING/IDLE intent → no audio (AR trail only)
        """
        if self._mode != Mode.READING:
            return

        intent = event.metadata.get("intent", ReadingIntent.IDLE)
        nearest_text: Optional[str] = event.metadata.get("nearest_text")
        is_illustration: bool = event.metadata.get("is_illustration", False)

        # Only act on READING intent.
        if intent != ReadingIntent.READING:
            # SCANNING or IDLE — no audio; AR trail is handled externally.
            logger.debug(
                json.dumps(
                    {
                        "event": "finger_scanning_no_audio",
                        "intent": intent.name if hasattr(intent, "name") else str(intent),
                    }
                )
            )
            return

        # READING intent.
        if nearest_text:
            dedup_key = (DetectionType.FINGER, nearest_text)
            if not self._is_recently_spoken(dedup_key):
                self._enqueue(
                    AudioResponse(
                        text=nearest_text,
                        priority=AudioPriority.NORMAL,
                    )
                )
                self._mark_spoken(dedup_key)
                logger.info(
                    json.dumps({"event": "reading_narrate", "word": nearest_text})
                )
            return

        if is_illustration and self._pending_illustration:
            description = self._pending_illustration
            self._pending_illustration = None  # Consume once used.
            self._enqueue(
                AudioResponse(
                    text=description,
                    priority=AudioPriority.NORMAL,
                )
            )
            logger.info(
                json.dumps({"event": "illustration_narrate", "description": description})
            )
            return

        # READING intent but no text and no illustration — silent.

    # ------------------------------------------------------------------
    # Text / Illustration events (Story mode or reading pipeline)
    # ------------------------------------------------------------------

    def _handle_text(self, event: DetectionEvent) -> None:
        """Handle TEXT detection in STORY or READING mode."""
        if self._mode not in (Mode.STORY, Mode.READING):
            return
        self._enqueue(
            AudioResponse(
                text=event.label,
                priority=AudioPriority.NORMAL,
            )
        )

    def _handle_illustration(self, event: DetectionEvent) -> None:
        """Handle ILLUSTRATION detection — store as pending for next READING event."""
        self._pending_illustration = event.label

    # ------------------------------------------------------------------
    # Queue helpers
    # ------------------------------------------------------------------

    def _enqueue(self, response: AudioResponse) -> None:
        """Add an AudioResponse to the queue. Logs every enqueue."""
        self._audio_queue.put_nowait(response)
        logger.debug(
            json.dumps(
                {
                    "event": "audio_enqueued",
                    "text": response.text,
                    "priority": response.priority.name,
                }
            )
        )

    # ------------------------------------------------------------------
    # De-duplication helpers
    # ------------------------------------------------------------------

    def _is_recently_spoken(self, key: tuple) -> bool:
        """Return True if key was narrated within the cooldown window."""
        import time
        last = self._last_spoken.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < _NARRATION_COOLDOWN_S

    def _mark_spoken(self, key: tuple) -> None:
        """Record that key was just narrated."""
        import time
        self._last_spoken[key] = time.monotonic()
