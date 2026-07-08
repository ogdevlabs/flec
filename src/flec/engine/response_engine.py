"""ResponseEngine — event orchestration and routing.

The single entry point for all audio and AR output decisions. Consumes
DetectionEvents from perception modules and emits AudioResponses and AR
overlay commands. No other module calls TTS or AR directly.

Contract:
    on_event(event)           — process a detection event
    set_mode(mode)            — transition to a new mode
    set_challenge(challenge)  — set or clear the active challenge

Architecture:
    - State-aware: routes events based on current mode and wear state
    - Never raises — all errors are logged, Constitution Rule 2
    - No direct imports from other capability modules
    - Wear detection gates all session modes
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from flec.audio.responses import CACHE_MANIFEST, CacheKey
from flec.logger import log_event
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


class ResponseEngine:
    """Orchestrate DetectionEvents → AudioResponses + AR overlay commands.

    Args:
        tts_engine: TTSEngine instance for audio output.
        ar_overlay: AROverlay instance for visual output (optional — AR is
                    enhancement only, not required for audio-complete operation).
    """

    def __init__(
        self,
        tts_engine: Any,
        ar_overlay: Any = None,
    ) -> None:
        self._tts = tts_engine
        self._ar = ar_overlay
        self._mode: Mode = Mode.STANDBY
        self._wear_state: WearState = WearState.OFF_HEAD
        self._challenge: Optional[Challenge] = None

        log_event(
            module="ResponseEngine",
            event_type="engine_initialized",
            data={"mode": self._mode.name},
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def on_event(self, event: DetectionEvent) -> None:
        """Process a detection event and route to audio/AR output.

        State-aware: respects active mode, wear state, and active challenge.
        Never raises — all errors are logged.
        """
        try:
            self._dispatch(event)
        except Exception as e:
            log_event(
                module="ResponseEngine",
                event_type="dispatch_error",
                data={"event_type": event.type.name, "error": str(e)},
            )

    def set_mode(self, mode: Mode) -> None:
        """Transition the engine to a new mode."""
        prev = self._mode
        self._mode = mode
        log_event(
            module="ResponseEngine",
            event_type="mode_transition",
            data={"from": prev.name, "to": mode.name},
        )

    def set_challenge(self, challenge: Optional[Challenge]) -> None:
        """Set or clear the active challenge."""
        self._challenge = challenge
        log_event(
            module="ResponseEngine",
            event_type="challenge_updated",
            data={
                "challenge": (
                    {"target": challenge.target_label, "type": challenge.target_type.name}
                    if challenge
                    else None
                )
            },
        )

    @property
    def current_mode(self) -> Mode:
        """Current active mode."""
        return self._mode

    @property
    def active_challenge(self) -> Optional[Challenge]:
        """Currently active challenge, or None."""
        return self._challenge

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: DetectionEvent) -> None:
        """Route an event to the appropriate handler based on type."""
        handlers = {
            DetectionType.WEAR: self._handle_wear,
            DetectionType.VOICE_CMD: self._handle_voice_cmd,
            DetectionType.SHAPE: self._handle_shape_color,
            DetectionType.COLOR: self._handle_shape_color,
            DetectionType.FINGER: self._handle_finger,
            DetectionType.TEXT: self._handle_text,
            DetectionType.ILLUSTRATION: self._handle_illustration,
        }

        handler = handlers.get(event.type)
        if handler:
            handler(event)
        else:
            log_event(
                module="ResponseEngine",
                event_type="unhandled_event_type",
                data={"event_type": event.type.name},
            )

    # ------------------------------------------------------------------
    # Individual event handlers
    # ------------------------------------------------------------------

    def _handle_wear(self, event: DetectionEvent) -> None:
        """Handle WearState transition events."""
        try:
            new_state = WearState[event.label]
        except KeyError:
            log_event(
                module="ResponseEngine",
                event_type="invalid_wear_label",
                data={"label": event.label},
            )
            return

        prev_state = self._wear_state
        self._wear_state = new_state

        log_event(
            module="ResponseEngine",
            event_type="wear_state_change",
            data={"from": prev_state.name, "to": new_state.name},
        )

        if new_state == WearState.ON_HEAD:
            self.set_mode(Mode.EXPLORATION)
            # Welcome audio is handled by the Session, not ResponseEngine
            # (boot sequence in main.py). Only play on re-wear during a session.
            if prev_state == WearState.OFF_HEAD:
                # Resume — no explicit audio needed; session resumes silently
                pass

        elif new_state == WearState.OFF_HEAD:
            self.set_mode(Mode.STANDBY)
            self._speak(
                text=CACHE_MANIFEST[CacheKey.MASK_OFF],
                priority=AudioPriority.CRITICAL,
                pre_cached=True,
                cache_key=CacheKey.MASK_OFF,
            )

    def _handle_voice_cmd(self, event: DetectionEvent) -> None:
        """Handle parsed voice command events."""
        try:
            intent = CommandIntent[event.label]
        except KeyError:
            log_event(
                module="ResponseEngine",
                event_type="invalid_command_intent",
                data={"label": event.label},
            )
            return

        # CRITICAL: shutdown commands are ONLY processed when mask is worn
        if intent == CommandIntent.SHUTDOWN:
            if self._wear_state != WearState.ON_HEAD:
                log_event(
                    module="ResponseEngine",
                    event_type="shutdown_ignored",
                    data={"reason": "mask not on head", "wear_state": self._wear_state.name},
                )
                return

            self._speak(
                text=CACHE_MANIFEST[CacheKey.SHUTDOWN],
                priority=AudioPriority.CRITICAL,
                pre_cached=True,
                cache_key=CacheKey.SHUTDOWN,
            )
            self.set_mode(Mode.STANDBY)
            log_event(
                module="ResponseEngine",
                event_type="shutdown_initiated",
                data={},
            )
            return

        if intent == CommandIntent.CANCEL_CHALLENGE:
            self.set_challenge(None)
            self.set_mode(Mode.EXPLORATION)
            return

        if intent == CommandIntent.REPEAT_CHALLENGE and self._challenge:
            target = self._challenge.target_label
            self._speak(
                text=f"Let's find a {target}!",
                priority=AudioPriority.HIGH,
            )

    def _handle_shape_color(self, event: DetectionEvent) -> None:
        """Handle shape and color detection events."""
        # Gate: only narrate when mask is on head
        if self._mode == Mode.STANDBY:
            log_event(
                module="ResponseEngine",
                event_type="detection_suppressed",
                data={"reason": "standby_mode", "label": event.label},
            )
            return

        det_type_word = "a" if event.type == DetectionType.SHAPE else ""

        if self._mode == Mode.CHALLENGE and self._challenge:
            if (
                self._challenge.status == ChallengeStatus.ACTIVE
                and event.label.lower() == self._challenge.target_label.lower()
            ):
                # Match! Celebration.
                self._speak(
                    text=CACHE_MANIFEST[CacheKey.CELEBRATION],
                    priority=AudioPriority.HIGH,
                    pre_cached=True,
                    cache_key=CacheKey.CELEBRATION,
                )
                log_event(
                    module="ResponseEngine",
                    event_type="challenge_match",
                    data={
                        "target": self._challenge.target_label,
                        "detected": event.label,
                        "confidence": event.confidence,
                    },
                )
                return
            else:
                # Non-match in challenge — encouraging audio
                self._speak(
                    text=CACHE_MANIFEST[CacheKey.ENCOURAGE],
                    priority=AudioPriority.NORMAL,
                    pre_cached=True,
                    cache_key=CacheKey.ENCOURAGE,
                )
                log_event(
                    module="ResponseEngine",
                    event_type="challenge_no_match",
                    data={
                        "target": self._challenge.target_label,
                        "detected": event.label,
                    },
                )
                return

        if self._mode == Mode.EXPLORATION:
            # Narrate the detection
            article = "a" if event.type == DetectionType.SHAPE else ""
            text = f"I see {article} {event.label}!".replace("  ", " ")
            self._speak(text=text, priority=AudioPriority.NORMAL)

            log_event(
                module="ResponseEngine",
                event_type="detection_narrated",
                data={
                    "type": event.type.name,
                    "label": event.label,
                    "confidence": event.confidence,
                },
            )

            # AR overlay — enhancement only, no impact on audio-complete operation
            if self._ar and event.bounding_box:
                try:
                    self._ar.highlight(event.bounding_box, label=event.label)
                except Exception as e:
                    log_event(
                        module="ResponseEngine",
                        event_type="ar_overlay_error",
                        data={"error": str(e)},
                    )

    def _handle_finger(self, event: DetectionEvent) -> None:
        """Handle finger tracking state updates (reading mode)."""
        if self._mode not in (Mode.READING, Mode.EXPLORATION):
            return
        # Finger tracking narration is handled by ReadingModule — ResponseEngine
        # receives the final AudioResponse to play here if needed.

    def _handle_text(self, event: DetectionEvent) -> None:
        """Handle OCR text detection events."""
        if self._mode not in (Mode.READING, Mode.STORY):
            return
        self._speak(text=event.label, priority=AudioPriority.NORMAL)

    def _handle_illustration(self, event: DetectionEvent) -> None:
        """Handle illustration description events."""
        if self._mode not in (Mode.READING, Mode.STORY):
            return
        text = f"I see {event.label}!"
        self._speak(text=text, priority=AudioPriority.NORMAL)

    # ------------------------------------------------------------------
    # Audio output helper
    # ------------------------------------------------------------------

    def _speak(
        self,
        text: str,
        priority: AudioPriority = AudioPriority.NORMAL,
        pre_cached: bool = False,
        cache_key: Optional[str] = None,
    ) -> None:
        """Enqueue an AudioResponse with the TTS engine."""
        response = AudioResponse(
            text=text,
            priority=priority,
            pre_cached=pre_cached,
            cache_key=cache_key,
        )
        self._tts.speak(response)

        log_event(
            module="ResponseEngine",
            event_type="audio_queued",
            data={
                "text_preview": text[:40],
                "priority": priority.name,
                "pre_cached": pre_cached,
            },
        )
