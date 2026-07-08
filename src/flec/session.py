"""FlecSession — session state machine for Flec.

Manages:
- Active Mode (EXPLORATION, CHALLENGE, READING, STORY, STANDBY)
- WearState tracking
- StoryContext lifecycle (page text, illustrations, narrative cursor)
- Detection event routing to the audio response queue
- Page turn detection

Architecture notes:
- Does NOT import capability modules directly (Principle III)
- All state is ephemeral / in-memory (Principle IV)
- All exceptions are caught and logged — no errors reach the toddler (Principle V)
- Emits structured JSON logs for every state transition (Principle II)
"""

from __future__ import annotations

import json
import logging
import queue
from dataclasses import replace
from typing import Optional

from flec.models import (
    AudioPriority,
    AudioResponse,
    DetectionEvent,
    DetectionType,
    Mode,
    StoryContext,
    WearState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Book detection heuristics
# ---------------------------------------------------------------------------

# Minimum word count on a page to consider it "book-like" (dense text)
_MIN_TEXT_WORDS_FOR_BOOK = 5

# Minimum text confidence (average) to consider a page text reliable
_MIN_TEXT_CONFIDENCE = 0.5


def detect_book_frame(
    page_text: str,
    has_illustration: bool = False,
    text_confidence: float = 1.0,
) -> bool:
    """Return True if the current frame looks like a picture-book page.

    A frame is considered "book-like" when:
    1. It contains at least ``_MIN_TEXT_WORDS_FOR_BOOK`` words of readable text, OR
    2. It contains an illustration AND at least 1 word of text.

    This heuristic is intentionally permissive — false positives are acceptable
    (brief exploration-mode narration of non-book text) while false negatives would
    silently fail to enter story mode.

    Args:
        page_text: OCR-extracted text from the current frame.
        has_illustration: True if an illustration region was detected.
        text_confidence: Average OCR confidence for the text (0.0–1.0).

    Returns:
        True if the frame should trigger STORY mode.
    """
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
    """Active session state machine.

    Consumers push :class:`~flec.models.DetectionEvent` objects via
    :meth:`on_detection_event`; the session routes them to the
    ``audio_queue`` as :class:`~flec.models.AudioResponse` objects.

    All public methods are thread-safe (they acquire the GIL; short ops only).

    Args:
        audio_queue: Queue where :class:`AudioResponse` objects are placed for
            the TTS engine to consume. Must be a ``queue.Queue`` instance.
        event_queue: Optional queue where raw :class:`DetectionEvent` objects
            are placed for external consumers (e.g. AR overlay, logging).
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
        """Current active mode."""
        return self._mode

    @property
    def wear_state(self) -> WearState:
        """Current wear state."""
        return self._wear_state

    @property
    def story_context(self) -> Optional[StoryContext]:
        """Current StoryContext, or None if not in STORY mode or book removed."""
        return self._story_context

    def set_mode(self, mode: Mode) -> None:
        """Transition session to *mode*.

        Resets mode-specific state (e.g. StoryContext when leaving STORY).
        Logs the transition as a structured event.
        """
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
        """Update the current wear state."""
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
        """Route a :class:`DetectionEvent` to the appropriate audio response.

        In STORY mode:
          - TEXT events → NORMAL priority narration queued
          - ILLUSTRATION events → NORMAL priority description queued
        Other modes: no story routing (future modes handled separately).

        Never raises — logs any error internally.
        """
        try:
            # Forward to raw event queue if wired
            if self._event_queue is not None:
                try:
                    self._event_queue.put_nowait(event)
                except queue.Full:
                    pass  # Non-critical — drop if queue is full

            if self._mode == Mode.STORY:
                self._route_story_event(event)
            # Other mode routing (EXPLORATION, CHALLENGE, READING) handled
            # by their respective response engines in later phases.

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
        """Route a detection event for STORY mode."""
        if self._story_context is None:
            # Book was removed — no routing
            return

        if event.type == DetectionType.TEXT:
            # Store latest page text in context
            self._story_context = StoryContext(
                page_text=event.label,
                illustrations=self._story_context.illustrations,
                narrative_position=self._story_context.narrative_position,
                page_stable=True,
            )
            # Queue narration
            if event.label.strip():
                response = AudioResponse(
                    text=event.label,
                    priority=AudioPriority.NORMAL,
                )
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
            # Append description to illustration list
            updated_illustrations = list(self._story_context.illustrations) + [event.label]
            self._story_context = StoryContext(
                page_text=self._story_context.page_text,
                illustrations=updated_illustrations,
                narrative_position=self._story_context.narrative_position,
                page_stable=self._story_context.page_stable,
            )
            # Queue description
            if event.label.strip():
                response = AudioResponse(
                    text=event.label,
                    priority=AudioPriority.NORMAL,
                )
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
        """Place *response* on the audio queue. Drops silently if queue is full."""
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
        """Evaluate whether the current frame should trigger STORY mode.

        Transitions the session to STORY mode if a book-like frame is detected,
        or returns to EXPLORATION if the book layout clears.

        Args:
            page_text: OCR text extracted from the current frame (empty string if none).
            has_illustration: True if an illustration region was detected in the frame.
            text_confidence: Average OCR confidence for the page text (0.0–1.0).

        Returns:
            True if STORY mode is now active (either newly triggered or already active).
        """
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
                # Book layout cleared — return to exploration, no error audio
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
        """Called when the book disappears from the camera view.

        Clears StoryContext and silently pauses narration.
        No error audio is played — toddler-first UX (Principle V).
        """
        self._story_context = None
        logger.info(
            json.dumps({
                "event": "book_removed",
                "module": "FlecSession",
            })
        )
        # Intentionally: no audio queued — silent pause per FR-013d

    def detect_page_turn(self, old_text: str, new_text: str) -> bool:
        """Determine whether a page turn has occurred by comparing text layouts.

        A page turn is detected when the new text differs meaningfully from the
        old text (>30% of words changed). On detection, StoryContext is reset
        so the new page reads from position 0.

        Args:
            old_text: Text from the previous stable frame.
            new_text: Text from the current frame.

        Returns:
            True if a page turn was detected; False otherwise.
        """
        turned = _is_page_turn(old_text, new_text)

        if turned:
            # Reset StoryContext for fresh page
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
        """Advance the narrative cursor by *word_count* words.

        Updates ``StoryContext.narrative_position``. No-op if no StoryContext.
        """
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
        """Mark *position* (word index) as the illustration description insertion point.

        The ResponseEngine uses this to interleave illustration descriptions at
        the natural narrative point rather than all at the start or end.

        Currently stored in StoryContext metadata for future ResponseEngine use.
        No-op if no StoryContext.
        """
        if self._story_context is None:
            return
        # Store in a new StoryContext with the insert position in page metadata
        # (StoryContext is a dataclass so we recreate it with updated fields)
        self._story_context = StoryContext(
            page_text=self._story_context.page_text,
            illustrations=self._story_context.illustrations,
            narrative_position=self._story_context.narrative_position,
            page_stable=self._story_context.page_stable,
        )
        # Attach illustration insert position as an attribute
        # (StoryContext doesn't have this field yet — store on instance)
        object.__setattr__(self._story_context, "_illustration_insert", position)
        logger.debug(
            json.dumps({
                "event": "illustration_insert_set",
                "module": "FlecSession",
                "position": position,
            })
        )


# ---------------------------------------------------------------------------
# Page-turn detection helper
# ---------------------------------------------------------------------------


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    """Return the Jaccard similarity (overlap ratio) between words in two texts.

    Returns 0.0 if both texts are empty.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0  # Both empty — same "page"
    if not words_a or not words_b:
        return 0.0  # One empty, one not — definite change
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def _is_page_turn(old_text: str, new_text: str, similarity_threshold: float = 0.3) -> bool:
    """Return True if the text change indicates a page turn.

    A page turn is assumed when word overlap drops below *similarity_threshold*
    (default: 30% Jaccard similarity). This is intentionally permissive to
    avoid false negatives from OCR noise between frames on the same page.
    """
    if old_text.strip() == new_text.strip():
        return False
    similarity = _word_overlap_ratio(old_text, new_text)
    return similarity < similarity_threshold
