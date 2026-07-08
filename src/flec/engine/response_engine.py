"""ResponseEngine — single orchestration point for audio and AR responses.

Responsibility: Consume DetectionEvents from all perception modules and
emit AudioResponses and AR overlay commands. No other module calls TTS or
AR directly.

Contract: see specs/001-perception-core/contracts/module-interfaces.md

Architecture notes:
- Does NOT import capability modules directly (Principle III)
- All audio responses flow through this engine — it is the sole gatekeeper
- Mode-aware: same event has different responses in different modes
- Wear-state-aware: VOICE_CMD(SHUTDOWN) is ignored when mask is off
- Story mode: TEXT events → cursor-gated narration; ILLUSTRATION → insert-point description
- StoryContext=None (book removed) → silent pause, no error audio
- Emits structured JSON logs for every routing decision (Principle II)
- Never raises — logs any error internally (Principle V)
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
    CommandIntent,
    DetectionEvent,
    DetectionType,
    Mode,
    StoryContext,
    WearState,
)

logger = logging.getLogger(__name__)


class ResponseEngine:
    """Single orchestration point for audio responses and AR commands.

    Usage::

        engine = ResponseEngine(audio_queue=my_queue)
        engine.set_mode(Mode.STORY)
        engine.on_event(detection_event)

    Args:
        audio_queue: Queue where :class:`AudioResponse` objects are placed for
            the TTS engine to consume.
        ar_queue: Optional queue for AR overlay commands (not used in story mode).
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        ar_queue: Optional[queue.Queue] = None,
    ) -> None:
        self._audio_queue: queue.Queue = audio_queue
        self._ar_queue: Optional[queue.Queue] = ar_queue
        self._mode: Mode = Mode.EXPLORATION
        self._wear_state: WearState = WearState.OFF_HEAD
        self._challenge: Optional[Challenge] = None
        self._story_context: Optional[StoryContext] = None

    # ------------------------------------------------------------------
    # Public: state management
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        """Current active mode."""
        return self._mode

    @property
    def wear_state(self) -> WearState:
        """Current wear state."""
        return self._wear_state

    @property
    def story_context(self) -> Optional[StoryContext]:
        """Current StoryContext (STORY mode), or None."""
        return self._story_context

    def set_mode(self, mode: Mode) -> None:
        """Transition to *mode*. Resets mode-specific state."""
        old = self._mode
        self._mode = mode
        if mode == Mode.STORY and self._story_context is None:
            self._story_context = StoryContext()
        elif old == Mode.STORY and mode != Mode.STORY:
            self._story_context = None
        logger.info(
            json.dumps({
                "event": "response_engine_mode_change",
                "module": "ResponseEngine",
                "from": old.name,
                "to": mode.name,
            })
        )

    def set_wear_state(self, state: WearState) -> None:
        """Update wear state."""
        self._wear_state = state

    def set_challenge(self, challenge: Optional[Challenge]) -> None:
        """Set or clear the active challenge."""
        self._challenge = challenge

    def set_story_context(self, ctx: Optional[StoryContext]) -> None:
        """Set or clear the active StoryContext (e.g. on page turn or book removal)."""
        self._story_context = ctx

    # ------------------------------------------------------------------
    # Public: event routing
    # ------------------------------------------------------------------

    def on_event(self, event: DetectionEvent) -> None:
        """Process a detection event and enqueue resulting audio/AR responses.

        State-aware: respects active mode, wear state, challenge, and StoryContext.
        Never raises — logs any error internally.
        """
        try:
            self._dispatch(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({
                    "event": "response_engine_error",
                    "module": "ResponseEngine",
                    "error": str(exc),
                    "detection_type": event.type.name,
                })
            )

    def _dispatch(self, event: DetectionEvent) -> None:
        """Internal dispatch — route event by type and mode."""
        etype = event.type

        # ------------------------------------------------------------------
        # Wear detection — CRITICAL priority, always handled
        # ------------------------------------------------------------------
        if etype == DetectionType.WEAR:
            self._handle_wear_event(event)
            return

        # ------------------------------------------------------------------
        # Voice commands — mode-independent, wear-state-gated
        # ------------------------------------------------------------------
        if etype == DetectionType.VOICE_CMD:
            self._handle_voice_command(event)
            return

        # ------------------------------------------------------------------
        # Mode-specific routing
        # ------------------------------------------------------------------
        if self._mode == Mode.STORY:
            self._handle_story_event(event)
        elif self._mode == Mode.EXPLORATION:
            self._handle_exploration_event(event)
        elif self._mode == Mode.CHALLENGE:
            self._handle_challenge_event(event)
        # READING mode handled in a later phase

    # ------------------------------------------------------------------
    # Event handlers by category
    # ------------------------------------------------------------------

    def _handle_wear_event(self, event: DetectionEvent) -> None:
        """Process a WEAR detection event."""
        from flec.models import WearState as WS  # local import to avoid circular ref

        label = event.label.upper()
        if "OFF" in label:
            self._enqueue(AudioResponse(
                text="Put your mask back on, hero!",
                priority=AudioPriority.CRITICAL,
            ))
            logger.info(json.dumps({
                "event": "wear_off_head_response",
                "module": "ResponseEngine",
            }))
        elif "ON" in label:
            self._enqueue(AudioResponse(
                text="Hero mask activated! Let's explore!",
                priority=AudioPriority.CRITICAL,
                pre_cached=True,
                cache_key="welcome",
            ))
            logger.info(json.dumps({
                "event": "wear_on_head_response",
                "module": "ResponseEngine",
            }))

    def _handle_voice_command(self, event: DetectionEvent) -> None:
        """Process a VOICE_CMD detection event (wake word + parsed intent)."""
        intent_name = event.metadata.get("intent", "")

        # SHUTDOWN command: only respond if mask is being worn
        if intent_name == CommandIntent.SHUTDOWN.name:
            if self._wear_state == WearState.ON_HEAD:
                self._enqueue(AudioResponse(
                    text="See you next time, hero!",
                    priority=AudioPriority.CRITICAL,
                    pre_cached=True,
                    cache_key="farewell",
                ))
                logger.info(json.dumps({
                    "event": "shutdown_command_accepted",
                    "module": "ResponseEngine",
                    "wear_state": self._wear_state.name,
                }))
            else:
                # Mask not worn — ignore command per FR-001e
                logger.info(json.dumps({
                    "event": "shutdown_command_ignored",
                    "module": "ResponseEngine",
                    "reason": "mask_not_worn",
                    "wear_state": self._wear_state.name,
                }))

    def _handle_story_event(self, event: DetectionEvent) -> None:
        """Route events in STORY mode.

        TEXT events → NORMAL narration (cursor-gated via StoryContext).
        ILLUSTRATION events → description at narrative insert position.
        StoryContext=None (book removed) → silent pause, no error.
        """
        if self._story_context is None:
            # Book removed — silent pause per FR-013d
            logger.debug(json.dumps({
                "event": "story_event_dropped_no_context",
                "module": "ResponseEngine",
                "detection_type": event.type.name,
            }))
            return

        if event.type == DetectionType.TEXT:
            text = event.label.strip()
            if not text:
                return

            # Cursor gate: only narrate text after the current narrative position
            # (prevents re-reading already-narrated content on same page)
            ctx = self._story_context
            start_pos = ctx.narrative_position
            words = text.split()

            if start_pos < len(words):
                remaining = " ".join(words[start_pos:])
                if remaining.strip():
                    self._enqueue(AudioResponse(
                        text=remaining,
                        priority=AudioPriority.NORMAL,
                    ))
                    logger.info(json.dumps({
                        "event": "story_text_narrated",
                        "module": "ResponseEngine",
                        "words_remaining": len(remaining.split()),
                        "confidence": round(event.confidence, 3),
                    }))

        elif event.type == DetectionType.ILLUSTRATION:
            description = event.label.strip()
            if not description:
                return

            self._enqueue(AudioResponse(
                text=description,
                priority=AudioPriority.NORMAL,
            ))
            logger.info(json.dumps({
                "event": "story_illustration_described",
                "module": "ResponseEngine",
                "word_count": len(description.split()),
                "confidence": round(event.confidence, 3),
            }))

    def _handle_exploration_event(self, event: DetectionEvent) -> None:
        """Route events in EXPLORATION mode."""
        if event.type == DetectionType.SHAPE:
            self._enqueue(AudioResponse(
                text=f"I see a {event.label}!",
                priority=AudioPriority.NORMAL,
            ))
            logger.info(json.dumps({
                "event": "exploration_shape_narrated",
                "module": "ResponseEngine",
                "label": event.label,
            }))
        elif event.type == DetectionType.COLOR:
            self._enqueue(AudioResponse(
                text=f"I see something {event.label}!",
                priority=AudioPriority.NORMAL,
            ))

    def _handle_challenge_event(self, event: DetectionEvent) -> None:
        """Route events in CHALLENGE mode."""
        if self._challenge is None:
            return

        if event.type in (DetectionType.SHAPE, DetectionType.COLOR):
            if event.label.lower() == self._challenge.target_label.lower():
                # Match!
                self._enqueue(AudioResponse(
                    text=f"You found it! That's a {event.label}! Amazing!",
                    priority=AudioPriority.HIGH,
                ))
                logger.info(json.dumps({
                    "event": "challenge_match",
                    "module": "ResponseEngine",
                    "label": event.label,
                }))
            else:
                # Non-match — encouraging, not discouraging (FR-007)
                self._enqueue(AudioResponse(
                    text="Keep looking, you're doing great!",
                    priority=AudioPriority.NORMAL,
                ))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enqueue(self, response: AudioResponse) -> None:
        """Place *response* on the audio queue. Drops silently if queue is full."""
        try:
            self._audio_queue.put_nowait(response)
        except queue.Full:
            logger.warning(json.dumps({
                "event": "audio_queue_full",
                "module": "ResponseEngine",
                "dropped_text": response.text[:40],
                "priority": response.priority.name,
            }))
