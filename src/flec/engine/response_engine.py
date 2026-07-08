"""ResponseEngine — single orchestration point for audio and AR outputs.

Consumes DetectionEvents from all perception modules and emits AudioResponses.
No other module calls TTS or AR directly — all routing goes through here.

Observability: every routing decision is logged as structured JSON.
Constitution compliance:
- Rule 3 (Modular AI): does not import from perception modules
- Rule 5 (Toddler-First UX): all audio is positive / no error messages to child
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional, Protocol

from flec.models import (
    AudioPriority,
    AudioResponse,
    Challenge,
    ChallengeStatus,
    ChallengeTargetType,
    CommandIntent,
    DetectionEvent,
    DetectionType,
    Mode,
    WearState,
)

logger = logging.getLogger(__name__)

# Throttle encouraging responses so the toddler isn't bombarded every frame
_ENCOURAGE_THROTTLE_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# TTS protocol (duck-typed to avoid importing TTSEngine at module level)
# ---------------------------------------------------------------------------


class _TTSProtocol(Protocol):
    def speak(self, response: AudioResponse) -> None:
        ...

    def stop_current(self) -> None:
        ...


# ---------------------------------------------------------------------------
# ResponseEngine
# ---------------------------------------------------------------------------


class ResponseEngine:
    """Stateful event → audio router for the Flec session.

    Usage::

        engine = ResponseEngine(tts=tts_engine)
        engine.set_mode(Mode.EXPLORATION)
        engine.on_event(detection_event)
    """

    def __init__(self, tts: _TTSProtocol) -> None:
        self._tts = tts
        self._mode: Mode = Mode.STANDBY
        self._wear_state: WearState = WearState.STANDBY
        self._challenge: Optional[Challenge] = None
        # Timestamp of the last encouraging response to throttle flooding
        self._last_encourage_at: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def wear_state(self) -> WearState:
        return self._wear_state

    @property
    def active_challenge(self) -> Optional[Challenge]:
        return self._challenge

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        """Explicitly set the active mode (called from session/main)."""
        previous = self._mode
        self._mode = mode
        logger.info(
            json.dumps({
                "event": "response_engine.mode_changed",
                "from": previous.name,
                "to": mode.name,
            })
        )

    def set_challenge(
        self,
        target_label: str,
        target_type: ChallengeTargetType,
        issued_at_override: Optional[float] = None,
    ) -> None:
        """Create and activate a Challenge (called from session dispatch).

        Args:
            target_label: Human-readable target (e.g. "red", "triangle").
            target_type: COLOR or SHAPE.
            issued_at_override: Optional monotonic timestamp for testing.
        """
        issued_at = issued_at_override if issued_at_override is not None else time.monotonic()
        self._challenge = Challenge(
            target_type=target_type,
            target_label=target_label,
            issued_at=issued_at,
            status=ChallengeStatus.ACTIVE,
        )
        self._last_encourage_at = 0.0
        logger.info(
            json.dumps({
                "event": "response_engine.challenge_set",
                "target_label": target_label,
                "target_type": target_type.name,
            })
        )

    def _cancel_challenge(self) -> None:
        if self._challenge is not None:
            self._challenge = Challenge(
                target_type=self._challenge.target_type,
                target_label=self._challenge.target_label,
                issued_at=self._challenge.issued_at,
                status=ChallengeStatus.CANCELLED,
            )
        self.set_mode(Mode.EXPLORATION)
        logger.info(json.dumps({"event": "response_engine.challenge_cancelled"}))

    def _complete_challenge(self) -> None:
        if self._challenge is not None:
            self._challenge = Challenge(
                target_type=self._challenge.target_type,
                target_label=self._challenge.target_label,
                issued_at=self._challenge.issued_at,
                status=ChallengeStatus.COMPLETED,
            )
        logger.info(json.dumps({"event": "response_engine.challenge_completed"}))

    # ------------------------------------------------------------------
    # Main event router
    # ------------------------------------------------------------------

    def on_event(self, event: DetectionEvent) -> None:
        """Route a DetectionEvent to the appropriate audio/AR response.

        State-aware: respects active mode, active challenge, and wear state.
        Never raises — logs and silently continues on unexpected events.
        """
        try:
            self._route(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({
                    "event": "response_engine.routing_error",
                    "detection_type": event.type.name,
                    "error": str(exc),
                })
            )

    def _route(self, event: DetectionEvent) -> None:
        """Internal routing — may raise (caught by on_event)."""
        if event.type == DetectionType.VOICE_CMD:
            self._handle_voice_cmd(event)
        elif event.type in (DetectionType.SHAPE, DetectionType.COLOR):
            self._handle_perception(event)
        elif event.type == DetectionType.WEAR:
            self._handle_wear(event)
        else:
            logger.debug(
                json.dumps({
                    "event": "response_engine.event_unhandled",
                    "type": event.type.name,
                })
            )

    # ------------------------------------------------------------------
    # Voice command handling
    # ------------------------------------------------------------------

    def _handle_voice_cmd(self, event: DetectionEvent) -> None:
        """Dispatch VOICE_CMD event to challenge/shutdown handlers."""
        cmd = event.metadata.get("command")
        if cmd is None:
            logger.warning(
                json.dumps({
                    "event": "response_engine.voice_cmd_missing_metadata",
                    "label": event.label,
                })
            )
            return

        intent = cmd.intent

        if intent == CommandIntent.START_CHALLENGE:
            self._start_challenge_flow(cmd)
        elif intent == CommandIntent.CANCEL_CHALLENGE:
            self._cancel_challenge_flow()
        elif intent == CommandIntent.SHUTDOWN:
            self._shutdown_flow()
        elif intent == CommandIntent.REPEAT_CHALLENGE:
            self._repeat_challenge_flow()
        else:
            logger.info(
                json.dumps({
                    "event": "response_engine.unknown_intent",
                    "intent": intent.name,
                })
            )

    def _start_challenge_flow(self, cmd) -> None:
        """Acknowledge and activate a new challenge."""
        from flec.audio.responses import challenge_acknowledgment

        target = cmd.target_label or "something"
        target_type = cmd.target_type or ChallengeTargetType.COLOR

        self.set_challenge(target_label=target, target_type=target_type)
        self.set_mode(Mode.CHALLENGE)

        ack_text = challenge_acknowledgment(target)
        self._tts.speak(
            AudioResponse(text=ack_text, priority=AudioPriority.HIGH)
        )
        logger.info(
            json.dumps({
                "event": "response_engine.challenge_acknowledged",
                "target": target,
                "ack_text": ack_text,
            })
        )

    def _cancel_challenge_flow(self) -> None:
        """Cancel active challenge and return to exploration."""
        self._cancel_challenge()

    def _shutdown_flow(self) -> None:
        """Handle SHUTDOWN intent — only when mask is worn."""
        from flec.audio.responses import session_farewell

        if self._wear_state != WearState.ON_HEAD:
            # Constitution rule: ignore shutdown when mask is not worn
            logger.info(
                json.dumps({
                    "event": "response_engine.shutdown_ignored",
                    "reason": "mask_not_worn",
                })
            )
            return
        self._tts.speak(
            AudioResponse(text=session_farewell(), priority=AudioPriority.CRITICAL)
        )

    def _repeat_challenge_flow(self) -> None:
        """Repeat the current challenge hint."""
        if self._challenge and self._challenge.status == ChallengeStatus.ACTIVE:
            from flec.audio.responses import challenge_hint
            self._tts.speak(
                AudioResponse(
                    text=challenge_hint(self._challenge.target_label),
                    priority=AudioPriority.HIGH,
                )
            )

    # ------------------------------------------------------------------
    # Perception event handling (SHAPE / COLOR)
    # ------------------------------------------------------------------

    def _handle_perception(self, event: DetectionEvent) -> None:
        """Route a SHAPE/COLOR detection based on current mode."""
        if self._mode == Mode.CHALLENGE:
            self._handle_challenge_detection(event)
        elif self._mode == Mode.EXPLORATION:
            self._handle_exploration_detection(event)
        else:
            # In READING, STORY, STANDBY — ignore shape/color events
            logger.debug(
                json.dumps({
                    "event": "response_engine.perception_ignored",
                    "mode": self._mode.name,
                    "label": event.label,
                })
            )

    def _handle_challenge_detection(self, event: DetectionEvent) -> None:
        """Match detection against active challenge target."""
        from flec.audio.responses import (
            challenge_celebration,
            challenge_encouraging,
            challenge_hint,
        )

        challenge = self._challenge
        if challenge is None or challenge.status != ChallengeStatus.ACTIVE:
            return

        # Check hint timer first
        now = time.monotonic()
        if now - challenge.issued_at >= 30.0 and self._hint_not_recently_given(now):
            hint_text = challenge_hint(challenge.target_label)
            self._tts.speak(
                AudioResponse(text=hint_text, priority=AudioPriority.HIGH)
            )
            logger.info(
                json.dumps({
                    "event": "response_engine.hint_played",
                    "target": challenge.target_label,
                })
            )
            return

        if self._is_match(event, challenge):
            # Target found — celebrate!
            celebration_text = challenge_celebration(challenge.target_label)
            self._tts.speak(
                AudioResponse(text=celebration_text, priority=AudioPriority.CRITICAL)
            )
            self._complete_challenge()
            logger.info(
                json.dumps({
                    "event": "response_engine.challenge_match",
                    "label": event.label,
                    "target": challenge.target_label,
                    "confidence": event.confidence,
                })
            )
        else:
            # No match — encourage (throttled)
            if self._should_encourage(now):
                encourage_text = challenge_encouraging()
                self._tts.speak(
                    AudioResponse(text=encourage_text, priority=AudioPriority.NORMAL)
                )
                self._last_encourage_at = now
                logger.info(
                    json.dumps({
                        "event": "response_engine.encourage_played",
                        "label": event.label,
                        "target": challenge.target_label,
                    })
                )

    def _handle_exploration_detection(self, event: DetectionEvent) -> None:
        """Narrate a shape or color in Exploration mode."""
        from flec.audio.responses import exploration_narration

        narration = exploration_narration(event.label)
        self._tts.speak(
            AudioResponse(text=narration, priority=AudioPriority.NORMAL)
        )
        logger.info(
            json.dumps({
                "event": "response_engine.exploration_narrated",
                "label": event.label,
                "confidence": event.confidence,
            })
        )

    # ------------------------------------------------------------------
    # Wear event handling
    # ------------------------------------------------------------------

    def _handle_wear(self, event: DetectionEvent) -> None:
        """Update wear state and react."""
        from flec.audio.responses import wear_off_prompt, wear_welcome
        from flec.models import WearState as WS

        if event.label == WS.ON_HEAD.name:
            self._wear_state = WS.ON_HEAD
            if self._mode == Mode.STANDBY:
                self.set_mode(Mode.EXPLORATION)
                self._tts.speak(
                    AudioResponse(text=wear_welcome(), priority=AudioPriority.HIGH)
                )
        elif event.label == WS.OFF_HEAD.name:
            self._wear_state = WS.OFF_HEAD
            self._tts.speak(
                AudioResponse(text=wear_off_prompt(), priority=AudioPriority.CRITICAL)
            )
            self.set_mode(Mode.STANDBY)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_match(self, event: DetectionEvent, challenge: Challenge) -> bool:
        """Return True if detection label matches challenge target."""
        return event.label.lower() == challenge.target_label.lower()

    def _should_encourage(self, now: float) -> bool:
        """Throttle encouraging messages to at most once per 5 seconds."""
        return (now - self._last_encourage_at) >= _ENCOURAGE_THROTTLE_SECONDS

    def _hint_not_recently_given(self, now: float) -> bool:
        """Prevent hint from firing on every frame once 30s has elapsed.

        Uses the same encourage throttle bucket for simplicity — hints also
        wait for the throttle window to expire.
        """
        return (now - self._last_encourage_at) >= _ENCOURAGE_THROTTLE_SECONDS
