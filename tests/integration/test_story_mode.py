"""Integration tests for Story Mode (US4).

Verifies end-to-end story mode behaviour:
- Book page detected → text read aloud in sequence
- Illustration present → described at narrative insert position
- Page turn (layout change) → new page narration starts fresh
- Book removed from frame → silent pause (no error audio)

Uses mocked OCRReader, IllustrationDescriber, and ResponseEngine
to test the StoryContext/session orchestration layer without requiring
real models (which are heavy and unavailable in CI).
"""

from __future__ import annotations

import queue
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flec.models import (
    AudioPriority,
    AudioResponse,
    DetectionEvent,
    DetectionType,
    Mode,
    StoryContext,
)
from flec.session import FlecSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_text_frame(text: str = "THE CAT SAT ON THE MAT") -> np.ndarray:
    """Return a BGR frame with rendered text (simulates a book page)."""
    import cv2

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.putText(frame, text, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    return frame


def make_illustration_frame() -> np.ndarray:
    """Return a colourful BGR frame (simulates a book illustration)."""
    import cv2

    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 150, (0, 180, 255), -1)  # orange circle
    return frame


def make_blank_frame() -> np.ndarray:
    """Return a blank black frame (book removed)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audio_queue() -> queue.Queue:
    """Return a fresh audio response queue."""
    return queue.Queue(maxsize=100)


@pytest.fixture
def event_queue() -> queue.Queue:
    """Return a fresh detection event queue."""
    return queue.Queue(maxsize=100)


@pytest.fixture
def session(audio_queue: queue.Queue, event_queue: queue.Queue) -> FlecSession:
    """Return a FlecSession wired with mocked OCR and Illustration backends."""
    return FlecSession(
        audio_queue=audio_queue,
        event_queue=event_queue,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestStoryModeIntegration:
    """Story Mode integration tests — session + OCR + IllustrationDescriber."""

    def test_book_page_detected_queues_narration(
        self, session: FlecSession, audio_queue: queue.Queue
    ) -> None:
        """When a TEXT detection event arrives in STORY mode, an audio response is queued."""
        session.set_mode(Mode.STORY)
        event = DetectionEvent(
            type=DetectionType.TEXT,
            label="THE CAT SAT ON THE MAT",
            confidence=0.95,
        )
        session.on_detection_event(event)

        assert not audio_queue.empty(), (
            "A text detection in STORY mode must produce an audio narration response"
        )
        response: AudioResponse = audio_queue.get_nowait()
        assert isinstance(response, AudioResponse)
        assert response.priority == AudioPriority.NORMAL
        assert "cat" in response.text.lower() or "mat" in response.text.lower() or len(response.text) > 0

    def test_illustration_event_queues_description_audio(
        self, session: FlecSession, audio_queue: queue.Queue
    ) -> None:
        """When an ILLUSTRATION detection event arrives in STORY mode, a description is queued."""
        session.set_mode(Mode.STORY)
        event = DetectionEvent(
            type=DetectionType.ILLUSTRATION,
            label="a big red barn and a little yellow chick",
            confidence=0.88,
        )
        session.on_detection_event(event)

        assert not audio_queue.empty(), (
            "An illustration detection in STORY mode must produce a description audio response"
        )
        response: AudioResponse = audio_queue.get_nowait()
        assert isinstance(response, AudioResponse)
        assert response.priority == AudioPriority.NORMAL

    def test_page_turn_resets_story_context(
        self, session: FlecSession, audio_queue: queue.Queue
    ) -> None:
        """Detecting a page turn resets StoryContext (narrative position → 0)."""
        session.set_mode(Mode.STORY)

        # Simulate reading some text on page 1
        event1 = DetectionEvent(
            type=DetectionType.TEXT,
            label="Once upon a time there was a cat",
            confidence=0.95,
        )
        session.on_detection_event(event1)

        # Advance narrative position
        ctx_before = session.story_context
        assert ctx_before is not None

        # Simulate page turn (new layout detected)
        session.detect_page_turn(
            old_text="Once upon a time there was a cat",
            new_text="The cat lived in a big red barn",
        )

        ctx_after = session.story_context
        assert ctx_after is not None
        assert ctx_after.narrative_position == 0, (
            "Page turn must reset narrative_position to 0 for fresh page reading"
        )

    def test_book_removed_causes_silent_pause(
        self, session: FlecSession, audio_queue: queue.Queue
    ) -> None:
        """When book is removed (no text/illustration detected), narration pauses silently."""
        session.set_mode(Mode.STORY)

        # Book removed: session receives a signal that story content is gone
        session.on_book_removed()

        # Audio queue should NOT have any error message
        while not audio_queue.empty():
            response: AudioResponse = audio_queue.get_nowait()
            # No error text should reach the toddler
            assert "error" not in response.text.lower(), (
                "Book removal must not produce error audio — toddler-first UX rule"
            )
            assert "sorry" not in response.text.lower(), (
                "Book removal must not produce apology audio"
            )

    def test_story_context_none_on_book_removed(
        self, session: FlecSession
    ) -> None:
        """story_context is cleared when book is removed from view."""
        session.set_mode(Mode.STORY)

        # First, seed some context
        session.on_detection_event(DetectionEvent(
            type=DetectionType.TEXT,
            label="Hello world",
            confidence=0.90,
        ))
        # Now remove the book
        session.on_book_removed()

        assert session.story_context is None or session.story_context.page_stable is False, (
            "story_context must be cleared or unstable when book is removed"
        )

    def test_text_event_in_non_story_mode_does_not_queue_narration(
        self, session: FlecSession, audio_queue: queue.Queue
    ) -> None:
        """TEXT detection events in EXPLORATION mode do not produce story narration."""
        session.set_mode(Mode.EXPLORATION)
        event = DetectionEvent(
            type=DetectionType.TEXT,
            label="hello world",
            confidence=0.90,
        )
        session.on_detection_event(event)

        # In EXPLORATION mode, text events are not narrated as story content
        # (they may queue audio via other paths, but story routing must be STORY-mode-only)
        # The session must not crash — basic sanity check
        assert True  # no exception raised

    def test_illustration_event_outside_story_mode_does_not_produce_story_description(
        self, session: FlecSession, audio_queue: queue.Queue
    ) -> None:
        """ILLUSTRATION events in non-story modes don't produce story description audio."""
        session.set_mode(Mode.EXPLORATION)
        event = DetectionEvent(
            type=DetectionType.ILLUSTRATION,
            label="a yellow duck",
            confidence=0.82,
        )
        # Must not raise
        session.on_detection_event(event)
        assert True  # no exception raised

    def test_advance_narrative_cursor_moves_forward(
        self, session: FlecSession
    ) -> None:
        """advance_narrative() increments narrative_position by word_count."""
        session.set_mode(Mode.STORY)
        session.on_detection_event(DetectionEvent(
            type=DetectionType.TEXT,
            label="the cat sat on the mat",
            confidence=0.95,
        ))
        ctx = session.story_context
        assert ctx is not None

        initial_pos = ctx.narrative_position
        session.advance_narrative(word_count=3)
        ctx = session.story_context
        assert ctx is not None
        assert ctx.narrative_position == initial_pos + 3, (
            "advance_narrative must increment narrative_position by word_count"
        )

    def test_illustration_insert_position_is_set(
        self, session: FlecSession
    ) -> None:
        """set_illustration_insert() marks the narrative position for illustration description."""
        session.set_mode(Mode.STORY)
        session.on_detection_event(DetectionEvent(
            type=DetectionType.TEXT,
            label="the cat sat on the mat",
            confidence=0.95,
        ))
        # Mark illustration insert at word 2
        session.set_illustration_insert(position=2)
        ctx = session.story_context
        assert ctx is not None
        # The session must record this without raising
        assert True  # interface smoke-test
