"""Session — Flec session state machines.

Contains:
- Session: phase 5 wear/challenge/voice-command state machine
- FlecSession: phase 7 story-mode state machine with StoryContext lifecycle

All state is ephemeral (never persisted).
Observability: every state transition emits a structured JSON log.
"""

from __future__ import annotations

import json
import logging
import queue
import time
from dataclasses import replace
from typing import Optional

from flec.models import (
    AudioPriority,
    AudioResponse,
    Challenge,
    ChallengeStatus,
    ChallengeTargetType,
    DetectionEvent,
    DetectionType,
    Mode,
    StoryContext,
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
        """Create a new Challenge, set hint timer, transition to CHALLENGE mode."""
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
        """Return True if a hint should be played for the active challenge."""
        if self._challenge is None or self._challenge.status != ChallengeStatus.ACTIVE:
            return False
        if self._hint_at is None:
            return False
        t = now if now is not None else time.monotonic()
        if t >= self._hint_at:
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
        intent,
        target_label: Optional[str] = None,
        target_type: Optional[ChallengeTargetType] = None,
        issued_at_override: Optional[float] = None,
    ) -> None:
        """Dispatch a parsed VoiceCommand to the appropriate session method."""
        from flec.models import CommandIntent

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


# ---------------------------------------------------------------------------
# Book detection heuristics
# ---------------------------------------------------------------------------

_MIN_TEXT_WORDS_FOR_BOOK = 5
_MIN_TEXT_CONFIDENCE = 0.5


def detect_book_frame(
    page_text: str,
    has_illustration: bool = False,
    text_confidence: float = 1.0,
) -> bool:
    """Return True if the current frame looks like a picture-book page."""
    words = page_text.strip().split() if page_text else []
    word_count = len(words)

    if text_confidence < _MIN_TEXT_CONFIDENCE and word_count < _MIN_TEXT_WORDS_FOR_BOOK:
        return False

    if word_count >= _MIN_TEXT_WORDS_FOR_BOOK:
        return True

    if has_illustration and word_count >= 1:
        return True

    return False


class FlecSession:
    """Story-mode session state machine.

    Consumers push :class:`~flec.models.DetectionEvent` objects via
    :meth:`on_detection_event`; the session routes them to the
    ``audio_queue`` as :class:`~flec.models.AudioResponse` objects.

    Args:
        audio_queue: Queue where :class:`AudioResponse` objects are placed for
            the TTS engine to consume.
        event_queue: Optional queue for raw :class:`DetectionEvent` objects.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        event_queue: Optional[queue.Queue] = None,
    ) -> None:
        self._audio_queue: queue.Queue = audio_queue
        self._event_queue: Optional[queue.Queue] = event_queue
        self._mode: Mode = Mode.EXPLORATION
        self._wear_state: WearState = WearState.OFF_HEAD
        self._story_context: Optional[StoryContext] = None

    # ------------------------------------------------------------------
    # Public: mode / wear state
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def wear_state(self) -> WearState:
        return self._wear_state

    @property
    def story_context(self) -> Optional[StoryContext]:
        return self._story_context

    def set_mode(self, mode: Mode) -> None:
        """Transition session to *mode*. Resets mode-specific state."""
        old_mode = self._mode
        self._mode = mode

        if mode == Mode.STORY:
            if self._story_context is None:
                self._story_context = StoryContext()
        elif old_mode == Mode.STORY and mode != Mode.STORY:
            self._story_context = None

        logger.info(
            json.dumps({
                "event": "mode_transition",
                "module": "FlecSession",
                "from": old_mode.name,
                "to": mode.name,
            })
        )

    def set_wear_state(self, state: WearState) -> None:
        old = self._wear_state
        self._wear_state = state
        logger.info(
            json.dumps({
                "event": "wear_state_transition",
                "module": "FlecSession",
                "from": old.name,
                "to": state.name,
            })
        )

    # ------------------------------------------------------------------
    # Public: detection event routing
    # ------------------------------------------------------------------

    def on_detection_event(self, event: DetectionEvent) -> None:
        """Route a DetectionEvent to the appropriate audio response. Never raises."""
        try:
            if self._event_queue is not None:
                try:
                    self._event_queue.put_nowait(event)
                except queue.Full:
                    pass

            if self._mode == Mode.STORY:
                self._route_story_event(event)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({
                    "event": "session_routing_error",
                    "module": "FlecSession",
                    "error": str(exc),
                    "detection_type": event.type.name,
                })
            )

    def _route_story_event(self, event: DetectionEvent) -> None:
        if self._story_context is None:
            return

        if event.type == DetectionType.TEXT:
            self._story_context = StoryContext(
                page_text=event.label,
                illustrations=self._story_context.illustrations,
                narrative_position=self._story_context.narrative_position,
                page_stable=True,
            )
            if event.label.strip():
                response = AudioResponse(text=event.label, priority=AudioPriority.NORMAL)
                self._enqueue_audio(response)
                logger.info(
                    json.dumps({
                        "event": "story_text_narrated",
                        "module": "FlecSession",
                        "char_count": len(event.label),
                        "confidence": round(event.confidence, 3),
                    })
                )

        elif event.type == DetectionType.ILLUSTRATION:
            updated_illustrations = list(self._story_context.illustrations) + [event.label]
            self._story_context = StoryContext(
                page_text=self._story_context.page_text,
                illustrations=updated_illustrations,
                narrative_position=self._story_context.narrative_position,
                page_stable=self._story_context.page_stable,
            )
            if event.label.strip():
                response = AudioResponse(text=event.label, priority=AudioPriority.NORMAL)
                self._enqueue_audio(response)
                logger.info(
                    json.dumps({
                        "event": "story_illustration_described",
                        "module": "FlecSession",
                        "word_count": len(event.label.split()),
                        "confidence": round(event.confidence, 3),
                    })
                )

    def _enqueue_audio(self, response: AudioResponse) -> None:
        try:
            self._audio_queue.put_nowait(response)
        except queue.Full:
            logger.warning(
                json.dumps({
                    "event": "audio_queue_full",
                    "module": "FlecSession",
                    "dropped_text": response.text[:40],
                })
            )

    # ------------------------------------------------------------------
    # Public: story mode trigger
    # ------------------------------------------------------------------

    def process_frame_for_story_mode(
        self,
        page_text: str,
        has_illustration: bool = False,
        text_confidence: float = 1.0,
    ) -> bool:
        """Evaluate whether the current frame should trigger STORY mode."""
        is_book = detect_book_frame(
            page_text=page_text,
            has_illustration=has_illustration,
            text_confidence=text_confidence,
        )

        if is_book:
            if self._mode != Mode.STORY:
                self.set_mode(Mode.STORY)
                logger.info(
                    json.dumps({
                        "event": "story_mode_triggered",
                        "module": "FlecSession",
                        "word_count": len(page_text.split()),
                        "has_illustration": has_illustration,
                    })
                )
            return True
        else:
            if self._mode == Mode.STORY:
                self.on_book_removed()
                self.set_mode(Mode.EXPLORATION)
                logger.info(
                    json.dumps({
                        "event": "story_mode_exited",
                        "module": "FlecSession",
                        "reason": "no_book_layout",
                    })
                )
            return False

    # ------------------------------------------------------------------
    # Public: story lifecycle
    # ------------------------------------------------------------------

    def on_book_removed(self) -> None:
        """Clear StoryContext and silently pause narration (no error audio)."""
        self._story_context = None
        logger.info(json.dumps({"event": "book_removed", "module": "FlecSession"}))

    def detect_page_turn(self, old_text: str, new_text: str) -> bool:
        """Return True if a page turn occurred; resets StoryContext if so."""
        turned = _is_page_turn(old_text, new_text)
        if turned:
            self._story_context = StoryContext()
            logger.info(
                json.dumps({
                    "event": "page_turn_detected",
                    "module": "FlecSession",
                    "old_word_count": len(old_text.split()),
                    "new_word_count": len(new_text.split()),
                })
            )
        return turned

    def advance_narrative(self, word_count: int) -> None:
        """Advance the narrative cursor by *word_count* words."""
        if self._story_context is None:
            return
        new_pos = self._story_context.narrative_position + max(0, word_count)
        self._story_context = StoryContext(
            page_text=self._story_context.page_text,
            illustrations=self._story_context.illustrations,
            narrative_position=new_pos,
            page_stable=self._story_context.page_stable,
        )
        logger.debug(
            json.dumps({
                "event": "narrative_advanced",
                "module": "FlecSession",
                "word_count": word_count,
                "new_position": new_pos,
            })
        )

    def set_illustration_insert(self, position: int) -> None:
        """Mark *position* as the illustration description insertion point."""
        if self._story_context is None:
            return
        self._story_context = StoryContext(
            page_text=self._story_context.page_text,
            illustrations=self._story_context.illustrations,
            narrative_position=self._story_context.narrative_position,
            page_stable=self._story_context.page_stable,
        )
        object.__setattr__(self._story_context, "_illustration_insert", position)
        logger.debug(
            json.dumps({
                "event": "illustration_insert_set",
                "module": "FlecSession",
                "position": position,
            })
        )


# ---------------------------------------------------------------------------
# Page-turn detection helpers
# ---------------------------------------------------------------------------


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _is_page_turn(old_text: str, new_text: str, similarity_threshold: float = 0.3) -> bool:
    if old_text.strip() == new_text.strip():
        return False
    similarity = _word_overlap_ratio(old_text, new_text)
    return similarity < similarity_threshold
