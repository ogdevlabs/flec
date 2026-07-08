"""Session state machine for Flec.

Manages the lifecycle of a single active use period. A session is bounded by
wear detection events (put on / taken off) or a voice shutdown command.

All session state is ephemeral — never persisted (Constitution Rule 4).

State transitions are validated; invalid transitions are logged and rejected.
Every transition emits a structured JSON log event (Constitution Rule 2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from flec.logger import log_event
from flec.models import (
    Challenge,
    ChallengeStatus,
    CommandIntent,
    DetectionEvent,
    Mode,
    StoryContext,
    VoiceCommand,
    WearState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid mode transitions
# ---------------------------------------------------------------------------

# Maps (from_mode, to_mode) → True if the transition is valid.
# Mode.STANDBY is always a valid destination (wear removal or shutdown).
_VALID_TRANSITIONS: set[tuple[Mode, Mode]] = {
    # From STANDBY (mask off or idle)
    (Mode.STANDBY, Mode.EXPLORATION),    # Mask put on → auto-start exploration

    # From EXPLORATION (default active mode)
    (Mode.EXPLORATION, Mode.STANDBY),    # Mask removed
    (Mode.EXPLORATION, Mode.CHALLENGE),  # Caregiver issues a challenge
    (Mode.EXPLORATION, Mode.READING),    # Finger near text detected
    (Mode.EXPLORATION, Mode.STORY),      # Book detected

    # From CHALLENGE
    (Mode.CHALLENGE, Mode.STANDBY),      # Mask removed
    (Mode.CHALLENGE, Mode.EXPLORATION),  # Challenge cancelled or completed

    # From READING
    (Mode.READING, Mode.STANDBY),        # Mask removed
    (Mode.READING, Mode.EXPLORATION),    # Finger retracted / no text

    # From STORY
    (Mode.STORY, Mode.STANDBY),          # Mask removed
    (Mode.STORY, Mode.EXPLORATION),      # Book removed
}


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """Active session state.

    Ephemeral — created on wear detection, discarded on removal/shutdown.
    All fields are in-memory only.
    """

    wear_state: WearState = WearState.OFF_HEAD
    mode: Mode = Mode.STANDBY
    active_challenge: Optional[Challenge] = None
    story_context: Optional[StoryContext] = None
    started_at: float = field(default_factory=time.monotonic)
    _transition_count: int = field(default=0, repr=False, init=False)

    # ------------------------------------------------------------------
    # Transition methods
    # ------------------------------------------------------------------

    def set_wear_state(self, new_state: WearState) -> bool:
        """Update wear state and auto-transition mode.

        Returns True if the state actually changed.
        """
        if new_state == self.wear_state:
            return False

        prev = self.wear_state
        self.wear_state = new_state

        log_event(
            module="Session",
            event_type="wear_state_transition",
            data={
                "from": prev.name,
                "to": new_state.name,
                "transition_count": self._transition_count,
            },
        )

        # Auto-transition mode based on wear state
        if new_state == WearState.ON_HEAD and self.mode == Mode.STANDBY:
            self.transition_mode(Mode.EXPLORATION)
        elif new_state == WearState.OFF_HEAD and self.mode != Mode.STANDBY:
            self.transition_mode(Mode.STANDBY)

        return True

    def transition_mode(self, new_mode: Mode) -> bool:
        """Transition to a new mode.

        Returns True if the transition was valid and applied.
        Returns False if the transition is invalid (logs the rejection).
        """
        if new_mode == self.mode:
            return True  # No-op, already in target mode

        if (self.mode, new_mode) not in _VALID_TRANSITIONS:
            log_event(
                module="Session",
                event_type="invalid_mode_transition",
                data={
                    "from": self.mode.name,
                    "to": new_mode.name,
                    "reason": "not in valid_transitions table",
                },
            )
            return False

        prev = self.mode
        self.mode = new_mode
        self._transition_count += 1

        log_event(
            module="Session",
            event_type="mode_transition",
            data={
                "from": prev.name,
                "to": new_mode.name,
                "transition_count": self._transition_count,
            },
        )
        return True

    def start_challenge(self, challenge: Challenge) -> None:
        """Set an active challenge and transition to CHALLENGE mode."""
        self.active_challenge = challenge
        self.transition_mode(Mode.CHALLENGE)

        log_event(
            module="Session",
            event_type="challenge_started",
            data={
                "target_type": challenge.target_type.name,
                "target_label": challenge.target_label,
            },
        )

    def complete_challenge(self) -> None:
        """Mark active challenge as completed and return to EXPLORATION."""
        if self.active_challenge is not None:
            log_event(
                module="Session",
                event_type="challenge_completed",
                data={"target": self.active_challenge.target_label},
            )
        self.active_challenge = None
        self.transition_mode(Mode.EXPLORATION)

    def cancel_challenge(self) -> None:
        """Cancel active challenge and return to EXPLORATION."""
        if self.active_challenge is not None:
            log_event(
                module="Session",
                event_type="challenge_cancelled",
                data={"target": self.active_challenge.target_label},
            )
        self.active_challenge = None
        self.transition_mode(Mode.EXPLORATION)

    def handle_wear_event(self, state: WearState) -> bool:
        """Handle a wear state event from WearDetector.

        Transitions:
        - ON_HEAD while STANDBY → EXPLORATION
        - OFF_HEAD while active → STANDBY (suspend all modes)
        - ON_HEAD while already ON_HEAD → no-op (idempotent)

        Returns True if the wear state actually changed.
        """
        changed = self.set_wear_state(state)

        log_event(
            module="Session",
            event_type="wear_event_handled",
            data={
                "state": state.name,
                "changed": changed,
                "mode": self.mode.name,
            },
        )
        return changed

    def handle_voice_command(self, cmd: VoiceCommand) -> bool:
        """Handle a parsed voice command.

        Rules:
        - SHUTDOWN is only processed when mask is ON_HEAD (FR-001e)
        - CANCEL_CHALLENGE cancels active challenge and returns to EXPLORATION
        - REPEAT_CHALLENGE is a no-op on the session (routed by ResponseEngine)
        - START_CHALLENGE is handled by the session loop (not here)
        - UNKNOWN is silently ignored

        Returns True if a session-level state change occurred.
        """
        if cmd.intent == CommandIntent.SHUTDOWN:
            if self.wear_state != WearState.ON_HEAD:
                log_event(
                    module="Session",
                    event_type="shutdown_command_ignored",
                    data={"reason": "mask_not_worn", "wear_state": self.wear_state.name},
                )
                return False

            log_event(
                module="Session",
                event_type="shutdown_command_accepted",
                data={},
            )
            self.transition_mode(Mode.STANDBY)
            return True

        if cmd.intent == CommandIntent.CANCEL_CHALLENGE:
            self.cancel_challenge()
            return True

        log_event(
            module="Session",
            event_type="voice_command_received",
            data={"intent": cmd.intent.name, "target": cmd.target_label},
        )
        return False

    def expire_challenge_if_needed(self, now: Optional[float] = None) -> bool:
        """Check and expire challenge if past EXPIRY_SECONDS.

        Returns True if the challenge was expired.
        """
        if self.active_challenge is None:
            return False
        if self.active_challenge.is_expired(now):
            log_event(
                module="Session",
                event_type="challenge_expired",
                data={"target": self.active_challenge.target_label},
            )
            self.active_challenge = None
            self.transition_mode(Mode.EXPLORATION)
            return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if the session is in an active (non-standby) mode."""
        return self.mode != Mode.STANDBY

    @property
    def is_worn(self) -> bool:
        """True if the mask is currently detected as on a head."""
        return self.wear_state == WearState.ON_HEAD

    @property
    def uptime_seconds(self) -> float:
        """Seconds since the session was created."""
        return time.monotonic() - self.started_at
