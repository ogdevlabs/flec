"""Integration tests for US2 — Challenge Mode: Ask & Verify Game.

Tests the full flow: voice command → acknowledgment → detection → celebration.

All tests use mocked perception and TTS dependencies — no hardware required.
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from flec.models import (
    AudioPriority,
    AudioResponse,
    ChallengeStatus,
    ChallengeTargetType,
    CommandIntent,
    DetectionEvent,
    DetectionType,
    Mode,
    VoiceCommand,
    WearState,
)
from flec.engine.response_engine import ResponseEngine
from flec.session import Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tts_mock():
    """Mock TTSEngine that records all speak() calls."""
    mock = MagicMock()
    mock.responses: List[AudioResponse] = []

    def _record_speak(response: AudioResponse) -> None:
        mock.responses.append(response)

    mock.speak.side_effect = _record_speak
    return mock


@pytest.fixture
def session() -> Session:
    return Session()


@pytest.fixture
def engine(tts_mock) -> ResponseEngine:
    return ResponseEngine(tts=tts_mock)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_shape_event(label: str, confidence: float = 0.95) -> DetectionEvent:
    return DetectionEvent(
        type=DetectionType.SHAPE,
        label=label,
        confidence=confidence,
    )


def make_color_event(label: str, confidence: float = 0.95) -> DetectionEvent:
    return DetectionEvent(
        type=DetectionType.COLOR,
        label=label,
        confidence=confidence,
    )


def make_voice_cmd_event(
    intent: CommandIntent,
    target_label: str | None = None,
    target_type: ChallengeTargetType | None = None,
    raw_text: str = "",
) -> DetectionEvent:
    cmd = VoiceCommand(
        intent=intent,
        target_label=target_label,
        target_type=target_type,
        raw_text=raw_text,
    )
    return DetectionEvent(
        type=DetectionType.VOICE_CMD,
        label=intent.name,
        confidence=1.0,
        metadata={"command": cmd},
    )


# ---------------------------------------------------------------------------
# Test: voice command → acknowledgment audio plays
# ---------------------------------------------------------------------------


class TestChallengeAcknowledgment:
    """AC1: caregiver voice command triggers acknowledgment audio."""

    def test_start_challenge_voice_cmd_queues_acknowledgment(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """START_CHALLENGE event → acknowledgment AudioResponse enqueued."""
        engine.set_mode(Mode.EXPLORATION)
        event = make_voice_cmd_event(
            CommandIntent.START_CHALLENGE,
            target_label="triangle",
            target_type=ChallengeTargetType.SHAPE,
            raw_text="find a triangle",
        )
        engine.on_event(event)

        assert tts_mock.speak.called, "TTS speak() was not called for acknowledgment"
        ack_responses = [r for r in tts_mock.responses]
        assert len(ack_responses) >= 1
        # Acknowledgment must be HIGH priority or above (not NORMAL/LOW)
        assert any(
            r.priority.value >= AudioPriority.HIGH.value for r in ack_responses
        ), f"Acknowledgment must be HIGH+ priority, got: {[r.priority for r in ack_responses]}"

    def test_acknowledgment_text_contains_target(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Acknowledgment text should mention the target."""
        engine.set_mode(Mode.EXPLORATION)
        event = make_voice_cmd_event(
            CommandIntent.START_CHALLENGE,
            target_label="red",
            target_type=ChallengeTargetType.COLOR,
            raw_text="find something red",
        )
        engine.on_event(event)

        assert tts_mock.speak.called
        texts = [r.text.lower() for r in tts_mock.responses]
        assert any("red" in t for t in texts), f"Target 'red' not mentioned in {texts}"

    def test_start_challenge_transitions_mode_to_challenge(
        self, engine: ResponseEngine
    ) -> None:
        """After START_CHALLENGE event, engine should be in CHALLENGE mode."""
        engine.set_mode(Mode.EXPLORATION)
        event = make_voice_cmd_event(
            CommandIntent.START_CHALLENGE,
            target_label="circle",
            target_type=ChallengeTargetType.SHAPE,
        )
        engine.on_event(event)
        assert engine.mode == Mode.CHALLENGE

    def test_start_challenge_sets_active_challenge(
        self, engine: ResponseEngine
    ) -> None:
        """After START_CHALLENGE, engine has an active Challenge with correct target."""
        engine.set_mode(Mode.EXPLORATION)
        event = make_voice_cmd_event(
            CommandIntent.START_CHALLENGE,
            target_label="blue",
            target_type=ChallengeTargetType.COLOR,
        )
        engine.on_event(event)
        challenge = engine.active_challenge
        assert challenge is not None
        assert challenge.target_label == "blue"
        assert challenge.status == ChallengeStatus.ACTIVE


# ---------------------------------------------------------------------------
# Test: target detected in frame → celebration audio
# ---------------------------------------------------------------------------


class TestChallengeMatchCelebration:
    """AC2: when target detected, celebration audio plays."""

    def test_matching_shape_triggers_celebration(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Detection matching challenge target → CRITICAL celebration response."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="triangle",
            target_type=ChallengeTargetType.SHAPE,
        )
        tts_mock.responses.clear()

        engine.on_event(make_shape_event("triangle"))

        celebration = [
            r for r in tts_mock.responses
            if r.priority == AudioPriority.CRITICAL or r.priority == AudioPriority.HIGH
        ]
        assert len(celebration) >= 1, (
            f"Expected HIGH/CRITICAL celebration, got: {[r.priority for r in tts_mock.responses]}"
        )

    def test_matching_color_triggers_celebration(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="red",
            target_type=ChallengeTargetType.COLOR,
        )
        tts_mock.responses.clear()

        engine.on_event(make_color_event("red"))

        celebration = [
            r for r in tts_mock.responses
            if r.priority.value >= AudioPriority.HIGH.value
        ]
        assert len(celebration) >= 1

    def test_celebration_text_contains_target(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Celebration text should name what was found."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="star",
            target_type=ChallengeTargetType.SHAPE,
        )
        tts_mock.responses.clear()

        engine.on_event(make_shape_event("star"))

        texts = [r.text.lower() for r in tts_mock.responses]
        assert any("star" in t for t in texts), f"'star' not in celebration texts: {texts}"

    def test_challenge_marked_completed_after_match(
        self, engine: ResponseEngine
    ) -> None:
        """Active challenge transitions to COMPLETED after target match."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="circle",
            target_type=ChallengeTargetType.SHAPE,
        )
        engine.on_event(make_shape_event("circle"))

        assert engine.active_challenge is not None
        assert engine.active_challenge.status == ChallengeStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test: non-matching detection → encouraging "keep looking" audio
# ---------------------------------------------------------------------------


class TestChallengeMismatchEncouragement:
    """AC3: non-matching detection plays encouraging (not negative) audio."""

    def test_non_matching_shape_triggers_encouraging_response(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Detecting wrong shape → NORMAL priority encouraging audio."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="triangle",
            target_type=ChallengeTargetType.SHAPE,
        )
        tts_mock.responses.clear()

        engine.on_event(make_shape_event("circle"))

        assert tts_mock.speak.called
        responses = tts_mock.responses
        assert len(responses) >= 1
        # Must be NORMAL priority (not CRITICAL/HIGH — reserved for matches)
        assert any(r.priority == AudioPriority.NORMAL for r in responses)

    def test_non_matching_response_is_encouraging_not_negative(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Encouraging text must not contain negative words."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="blue",
            target_type=ChallengeTargetType.COLOR,
        )
        tts_mock.responses.clear()

        engine.on_event(make_color_event("red"))

        negative_words = ["wrong", "no!", "bad", "incorrect", "error", "fail"]
        for response in tts_mock.responses:
            lower = response.text.lower()
            for word in negative_words:
                assert word not in lower, (
                    f"Negative word '{word}' found in response: {response.text!r}"
                )

    def test_non_matching_responses_are_throttled(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Encouraging responses must be throttled (not every frame)."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="triangle",
            target_type=ChallengeTargetType.SHAPE,
        )
        tts_mock.responses.clear()

        # Fire 10 non-matching events in quick succession
        for _ in range(10):
            engine.on_event(make_shape_event("square"))

        # Should be far fewer than 10 responses (throttled to once per ~5s)
        assert len(tts_mock.responses) <= 3, (
            f"Expected throttling, but got {len(tts_mock.responses)} responses for 10 events"
        )


# ---------------------------------------------------------------------------
# Test: 30s elapsed without match → hint audio
# ---------------------------------------------------------------------------


class TestChallengeHint:
    """AC4: hint plays when 30 seconds elapse without a match."""

    def test_hint_plays_when_challenge_expired(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """If challenge has been active > 30s, a hint AudioResponse is queued."""
        engine.set_mode(Mode.CHALLENGE)
        # Set challenge with issued_at in the past
        engine.set_challenge(
            target_label="triangle",
            target_type=ChallengeTargetType.SHAPE,
            issued_at_override=time.monotonic() - 35.0,
        )
        tts_mock.responses.clear()

        # Trigger hint check via a non-matching detection
        engine.on_event(make_shape_event("square"))

        hint_responses = [r for r in tts_mock.responses if "triangle" in r.text.lower()]
        assert len(hint_responses) >= 1, (
            f"Expected hint mentioning 'triangle', got: {[r.text for r in tts_mock.responses]}"
        )

    def test_hint_text_mentions_target(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="red",
            target_type=ChallengeTargetType.COLOR,
            issued_at_override=time.monotonic() - 40.0,
        )
        tts_mock.responses.clear()

        engine.on_event(make_color_event("blue"))

        texts = [r.text.lower() for r in tts_mock.responses]
        assert any("red" in t for t in texts), f"'red' not in hint texts: {texts}"


# ---------------------------------------------------------------------------
# Test: "Hey Flec, stop" → challenge cancelled, back to EXPLORATION
# ---------------------------------------------------------------------------


class TestChallengeCancellation:
    """AC5: CANCEL_CHALLENGE command stops challenge and returns to EXPLORATION."""

    def test_cancel_challenge_cmd_clears_challenge(
        self, engine: ResponseEngine
    ) -> None:
        """CANCEL_CHALLENGE event clears active challenge."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="square",
            target_type=ChallengeTargetType.SHAPE,
        )
        assert engine.active_challenge is not None

        engine.on_event(make_voice_cmd_event(CommandIntent.CANCEL_CHALLENGE))

        assert engine.active_challenge is None or (
            engine.active_challenge.status == ChallengeStatus.CANCELLED
        )

    def test_cancel_challenge_returns_to_exploration_mode(
        self, engine: ResponseEngine
    ) -> None:
        """After CANCEL_CHALLENGE, engine returns to EXPLORATION mode."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="red",
            target_type=ChallengeTargetType.COLOR,
        )

        engine.on_event(make_voice_cmd_event(CommandIntent.CANCEL_CHALLENGE))

        assert engine.mode == Mode.EXPLORATION

    def test_detection_after_cancel_is_not_celebration(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """After challenge cancelled, matching detection is NOT a celebration."""
        engine.set_mode(Mode.CHALLENGE)
        engine.set_challenge(
            target_label="triangle",
            target_type=ChallengeTargetType.SHAPE,
        )
        engine.on_event(make_voice_cmd_event(CommandIntent.CANCEL_CHALLENGE))
        tts_mock.responses.clear()

        # This would have been a match, but challenge is now cancelled
        engine.on_event(make_shape_event("triangle"))

        celebration = [
            r for r in tts_mock.responses
            if r.priority == AudioPriority.CRITICAL
        ]
        assert len(celebration) == 0, (
            "CRITICAL celebration should not fire after challenge is cancelled"
        )


# ---------------------------------------------------------------------------
# Test: no challenge in EXPLORATION mode — shape events narrated normally
# ---------------------------------------------------------------------------


class TestExplorationModePassthrough:
    """In EXPLORATION mode, shape events should narrate normally (not challenge routing)."""

    def test_shape_event_in_exploration_mode_queues_normal_audio(
        self, engine: ResponseEngine, tts_mock
    ) -> None:
        """Shape detection in EXPLORATION mode → NORMAL audio narration."""
        engine.set_mode(Mode.EXPLORATION)
        engine.on_event(make_shape_event("circle"))

        assert tts_mock.speak.called
        responses = tts_mock.responses
        # In exploration mode, narration is NORMAL priority
        assert any(r.priority == AudioPriority.NORMAL for r in responses)
