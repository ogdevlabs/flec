"""Session — Flec session state machine.

Manages wear state, active mode, and challenge lifecycle.
All state is ephemeral (never persisted).

Observability: every state transition emits a structured JSON log.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from flec.models import (
    Challenge,
    ChallengeStatus,
    ChallengeTargetType,
    Mode,
    WearState,
)

logger = logging.getLogger(__name__)

# How many seconds before the system delivers a hint for an active challenge.
HINT_AFTER_SECONDS: float = 30.0


class Session:
    """Ephemeral session state for a single wear period.

    Thread-safety note: this class is designed to be accessed from a single
    orchestration thread. If accessed from multiple threads, callers are
    responsible for external locking.
    """

    def __init__(self) -> None:
        self._wear_state: WearState = WearState.STANDBY
        self._mode: Mode = Mode.STANDBY
        self._challenge: Optional[Challenge] = None
        # hint_at tracks when we should remind the toddler about the challenge
        self._hint_at: Optional[float] = None
        self._hint_given: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wear_state(self) -> WearState:
        return self._wear_state

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def active_challenge(self) -> Optional[Challenge]:
        return self._challenge

    # ------------------------------------------------------------------
    # Wear state
    # ------------------------------------------------------------------

    def set_wear_state(self, state: WearState) -> None:
        """Update wear state and log the transition."""
        if state == self._wear_state:
            return
        previous = self._wear_state
        self._wear_state = state
        logger.info(
            json.dumps({
                "event": "session.wear_state_changed",
                "from": previous.name,
                "to": state.name,
            })
        )

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        """Transition to a new mode and log the change."""
        if mode == self._mode:
            return
        previous = self._mode
        self._mode = mode
        logger.info(
            json.dumps({
                "event": "session.mode_changed",
                "from": previous.name,
                "to": mode.name,
            })
        )

    # ------------------------------------------------------------------
    # Challenge lifecycle
    # ------------------------------------------------------------------

    def start_challenge(
        self,
        target: str,
        target_type: ChallengeTargetType,
        issued_at_override: Optional[float] = None,
    ) -> Challenge:
        """Create a new Challenge, set hint timer, transition to CHALLENGE mode.

        Args:
            target: The human-readable target label (e.g. "red", "triangle").
            target_type: Whether the target is a COLOR or SHAPE.
            issued_at_override: Optional monotonic timestamp override (for tests).

        Returns:
            The newly created Challenge.
        """
        issued_at = issued_at_override if issued_at_override is not None else time.monotonic()
        self._challenge = Challenge(
            target_type=target_type,
            target_label=target,
            issued_at=issued_at,
            status=ChallengeStatus.ACTIVE,
        )
        self._hint_at = issued_at + HINT_AFTER_SECONDS
        self._hint_given = False
        self.set_mode(Mode.CHALLENGE)

        logger.info(
            json.dumps({
                "event": "session.challenge_started",
                "target_label": target,
                "target_type": target_type.name,
                "hint_at_offset_seconds": HINT_AFTER_SECONDS,
            })
        )
        return self._challenge

    def cancel_challenge(self) -> None:
        """Cancel the active challenge (if any) and return to EXPLORATION mode."""
        if self._challenge is None:
            return

        # Replace frozen Challenge with updated status by rebuilding
        self._challenge = Challenge(
            target_type=self._challenge.target_type,
            target_label=self._challenge.target_label,
            issued_at=self._challenge.issued_at,
            status=ChallengeStatus.CANCELLED,
        )
        self._hint_at = None
        self._hint_given = False

        logger.info(
            json.dumps({
                "event": "session.challenge_cancelled",
                "target_label": self._challenge.target_label,
            })
        )
        self.set_mode(Mode.EXPLORATION)

    def complete_challenge(self) -> None:
        """Mark the active challenge as COMPLETED."""
        if self._challenge is None or self._challenge.status != ChallengeStatus.ACTIVE:
            return

        self._challenge = Challenge(
            target_type=self._challenge.target_type,
            target_label=self._challenge.target_label,
            issued_at=self._challenge.issued_at,
            status=ChallengeStatus.COMPLETED,
        )
        self._hint_at = None

        logger.info(
            json.dumps({
                "event": "session.challenge_completed",
                "target_label": self._challenge.target_label,
            })
        )

    def should_hint(self, now: Optional[float] = None) -> bool:
        """Return True if a hint should be played for the active challenge.

        True when all of:
        - A challenge is active
        - hint_at timestamp has elapsed
        - Hint has not already been given for this challenge cycle

        After returning True, resets the hint timer so the next hint fires
        another HINT_AFTER_SECONDS later (repeated prompts).

        Args:
            now: Optional monotonic timestamp override (for tests).
        """
        if self._challenge is None or self._challenge.status != ChallengeStatus.ACTIVE:
            return False
        if self._hint_at is None:
            return False
        t = now if now is not None else time.monotonic()
        if t >= self._hint_at:
            # Reset for next cycle so hints repeat
            self._hint_at = t + HINT_AFTER_SECONDS
            logger.info(
                json.dumps({
                    "event": "session.hint_due",
                    "target_label": self._challenge.target_label,
                    "next_hint_in_seconds": HINT_AFTER_SECONDS,
                })
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Voice command dispatch
    # ------------------------------------------------------------------

    def handle_voice_command(
        self,
        intent,  # CommandIntent
        target_label: Optional[str] = None,
        target_type: Optional[ChallengeTargetType] = None,
        issued_at_override: Optional[float] = None,
    ) -> None:
        """Dispatch a parsed VoiceCommand to the appropriate session method.

        Called by main.py after CommandSTT.transcribe(); produces a
        DetectionEvent that ResponseEngine will consume.
        """
        from flec.models import CommandIntent  # local import to avoid circular deps

        if intent == CommandIntent.START_CHALLENGE and target_label and target_type:
            self.start_challenge(
                target=target_label,
                target_type=target_type,
                issued_at_override=issued_at_override,
            )
        elif intent == CommandIntent.CANCEL_CHALLENGE:
            self.cancel_challenge()
        elif intent == CommandIntent.SHUTDOWN:
            self.set_wear_state(WearState.STANDBY)
            self.set_mode(Mode.STANDBY)
        else:
            logger.debug(
                json.dumps({
                    "event": "session.unhandled_intent",
                    "intent": str(intent),
                })
            )
