"""Contract tests for ResponseEngine.

Tests verify the public interface contract defined in
specs/001-perception-core/contracts/module-interfaces.md.

All tests use mock TTSEngine and AROverlay to verify routing logic only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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


# ---------------------------------------------------------------------------
# Contract: ResponseEngine must be importable
# ---------------------------------------------------------------------------


def test_response_engine_importable() -> None:
    """ResponseEngine must be importable from flec.engine.response_engine."""
    from flec.engine.response_engine import ResponseEngine  # noqa: F401


# ---------------------------------------------------------------------------
# Fixture: ResponseEngine with mocked TTS and AR
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tts():
    """Return a mock TTSEngine."""
    engine = MagicMock()
    engine.is_speaking = False
    return engine


@pytest.fixture
def mock_ar():
    """Return a mock AROverlay."""
    return MagicMock()


@pytest.fixture
def response_engine(mock_tts, mock_ar):
    """Return a ResponseEngine wired to mock TTS and AR backends."""
    from flec.engine.response_engine import ResponseEngine

    engine = ResponseEngine(tts_engine=mock_tts, ar_overlay=mock_ar)
    return engine


# ---------------------------------------------------------------------------
# Contract: WEAR(ON_HEAD) → mode transitions to EXPLORATION
# ---------------------------------------------------------------------------


def test_wear_on_head_transitions_to_exploration(response_engine, mock_tts) -> None:
    """WEAR(ON_HEAD) DetectionEvent must transition mode to EXPLORATION."""
    from flec.engine.response_engine import ResponseEngine

    event = DetectionEvent(
        type=DetectionType.WEAR,
        label=WearState.ON_HEAD.name,
        confidence=1.0,
    )
    response_engine.on_event(event)

    assert response_engine.current_mode == Mode.EXPLORATION, (
        f"WEAR(ON_HEAD) must transition to EXPLORATION, got {response_engine.current_mode}"
    )


# ---------------------------------------------------------------------------
# Contract: WEAR(OFF_HEAD) → mode transitions to STANDBY, CRITICAL response queued
# ---------------------------------------------------------------------------


def test_wear_off_head_transitions_to_standby(response_engine, mock_tts) -> None:
    """WEAR(OFF_HEAD) DetectionEvent must transition mode to STANDBY."""
    # First put it on
    response_engine.on_event(
        DetectionEvent(type=DetectionType.WEAR, label=WearState.ON_HEAD.name, confidence=1.0)
    )

    # Now take it off
    event = DetectionEvent(
        type=DetectionType.WEAR,
        label=WearState.OFF_HEAD.name,
        confidence=1.0,
    )
    response_engine.on_event(event)

    assert response_engine.current_mode == Mode.STANDBY, (
        f"WEAR(OFF_HEAD) must transition to STANDBY, got {response_engine.current_mode}"
    )


def test_wear_off_head_queues_critical_audio(response_engine, mock_tts) -> None:
    """WEAR(OFF_HEAD) must queue a CRITICAL AudioResponse ('put mask back on')."""
    # Put on first
    response_engine.on_event(
        DetectionEvent(type=DetectionType.WEAR, label=WearState.ON_HEAD.name, confidence=1.0)
    )

    event = DetectionEvent(
        type=DetectionType.WEAR,
        label=WearState.OFF_HEAD.name,
        confidence=1.0,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_called()
    call_args = mock_tts.speak.call_args_list
    critical_calls = [
        c for c in call_args
        if c.args and isinstance(c.args[0], AudioResponse)
        and c.args[0].priority == AudioPriority.CRITICAL
    ]
    assert len(critical_calls) >= 1, (
        "WEAR(OFF_HEAD) must queue at least one CRITICAL AudioResponse"
    )


# ---------------------------------------------------------------------------
# Contract: VOICE_CMD(SHUTDOWN) + ON_HEAD → CRITICAL shutdown response
# ---------------------------------------------------------------------------


def test_shutdown_cmd_while_on_head_queues_critical_response(
    response_engine, mock_tts
) -> None:
    """VOICE_CMD(SHUTDOWN) while ON_HEAD must queue a CRITICAL shutdown response."""
    # Wear mask
    response_engine.on_event(
        DetectionEvent(type=DetectionType.WEAR, label=WearState.ON_HEAD.name, confidence=1.0)
    )
    mock_tts.speak.reset_mock()

    # Issue shutdown command
    event = DetectionEvent(
        type=DetectionType.VOICE_CMD,
        label=CommandIntent.SHUTDOWN.name,
        confidence=1.0,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_called()
    call_args = mock_tts.speak.call_args_list
    critical_calls = [
        c for c in call_args
        if c.args and isinstance(c.args[0], AudioResponse)
        and c.args[0].priority == AudioPriority.CRITICAL
    ]
    assert len(critical_calls) >= 1, (
        "VOICE_CMD(SHUTDOWN) while ON_HEAD must queue CRITICAL shutdown response"
    )


# ---------------------------------------------------------------------------
# Contract: VOICE_CMD(SHUTDOWN) + OFF_HEAD → ignored (no response)
# ---------------------------------------------------------------------------


def test_shutdown_cmd_while_off_head_is_ignored(response_engine, mock_tts) -> None:
    """VOICE_CMD(SHUTDOWN) while OFF_HEAD must be ignored — no audio queued."""
    # Start in STANDBY (mask off)
    assert response_engine.current_mode == Mode.STANDBY

    event = DetectionEvent(
        type=DetectionType.VOICE_CMD,
        label=CommandIntent.SHUTDOWN.name,
        confidence=1.0,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_not_called(), (
        "VOICE_CMD(SHUTDOWN) while OFF_HEAD must be ignored"
    )


# ---------------------------------------------------------------------------
# Contract: SHAPE event in EXPLORATION mode → narration response queued
# ---------------------------------------------------------------------------


def test_shape_event_in_exploration_queues_narration(response_engine, mock_tts) -> None:
    """SHAPE event in EXPLORATION mode must queue an AudioResponse with the shape label."""
    # Put mask on to enter EXPLORATION
    response_engine.on_event(
        DetectionEvent(type=DetectionType.WEAR, label=WearState.ON_HEAD.name, confidence=1.0)
    )
    mock_tts.speak.reset_mock()

    event = DetectionEvent(
        type=DetectionType.SHAPE,
        label="circle",
        confidence=0.9,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_called()
    # The narration must mention the shape label
    call_args = mock_tts.speak.call_args_list
    narration_calls = [
        c for c in call_args
        if c.args and isinstance(c.args[0], AudioResponse)
        and "circle" in c.args[0].text.lower()
    ]
    assert len(narration_calls) >= 1, (
        "SHAPE event in EXPLORATION must queue AudioResponse mentioning 'circle'"
    )


# ---------------------------------------------------------------------------
# Contract: SHAPE event in STANDBY mode → ignored
# ---------------------------------------------------------------------------


def test_shape_event_in_standby_is_ignored(response_engine, mock_tts) -> None:
    """SHAPE event in STANDBY mode must be ignored (no audio queued)."""
    assert response_engine.current_mode == Mode.STANDBY

    event = DetectionEvent(
        type=DetectionType.SHAPE,
        label="circle",
        confidence=0.9,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_not_called(), (
        "SHAPE events in STANDBY must be ignored"
    )


# ---------------------------------------------------------------------------
# Contract: SHAPE event matching active Challenge → HIGH priority celebration
# ---------------------------------------------------------------------------


def test_shape_matches_challenge_queues_celebration(response_engine, mock_tts) -> None:
    """SHAPE event matching active challenge target → HIGH priority celebration response."""
    # Put on, enter EXPLORATION, then set challenge
    response_engine.on_event(
        DetectionEvent(type=DetectionType.WEAR, label=WearState.ON_HEAD.name, confidence=1.0)
    )
    challenge = Challenge(
        target_type=ChallengeTargetType.SHAPE,
        target_label="triangle",
        status=ChallengeStatus.ACTIVE,
    )
    response_engine.set_challenge(challenge)
    response_engine.set_mode(Mode.CHALLENGE)
    mock_tts.speak.reset_mock()

    event = DetectionEvent(
        type=DetectionType.SHAPE,
        label="triangle",
        confidence=0.95,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_called()
    call_args = mock_tts.speak.call_args_list
    celebration_calls = [
        c for c in call_args
        if c.args and isinstance(c.args[0], AudioResponse)
        and c.args[0].priority in (AudioPriority.HIGH, AudioPriority.CRITICAL)
    ]
    assert len(celebration_calls) >= 1, (
        "Matching challenge target must queue HIGH/CRITICAL celebration response"
    )


# ---------------------------------------------------------------------------
# Contract: SHAPE event NOT matching active Challenge → encouraging NORMAL response
# ---------------------------------------------------------------------------


def test_shape_does_not_match_challenge_queues_encouraging(
    response_engine, mock_tts
) -> None:
    """SHAPE event not matching active challenge → encouraging NORMAL response."""
    response_engine.on_event(
        DetectionEvent(type=DetectionType.WEAR, label=WearState.ON_HEAD.name, confidence=1.0)
    )
    challenge = Challenge(
        target_type=ChallengeTargetType.SHAPE,
        target_label="triangle",
        status=ChallengeStatus.ACTIVE,
    )
    response_engine.set_challenge(challenge)
    response_engine.set_mode(Mode.CHALLENGE)
    mock_tts.speak.reset_mock()

    event = DetectionEvent(
        type=DetectionType.SHAPE,
        label="circle",  # Not the target
        confidence=0.9,
    )
    response_engine.on_event(event)

    mock_tts.speak.assert_called()
    call_args = mock_tts.speak.call_args_list
    # Must not be a critical/celebration response
    critical_calls = [
        c for c in call_args
        if c.args and isinstance(c.args[0], AudioResponse)
        and c.args[0].priority == AudioPriority.CRITICAL
    ]
    assert len(critical_calls) == 0, (
        "Non-matching detection in challenge must NOT queue CRITICAL response"
    )
    # Must queue something (encouraging audio)
    assert len(call_args) >= 1, (
        "Non-matching detection in challenge must queue at least one response"
    )


# ---------------------------------------------------------------------------
# Contract: set_mode() transitions mode
# ---------------------------------------------------------------------------


def test_set_mode_transitions_mode(response_engine) -> None:
    """set_mode() must update current_mode."""
    response_engine.set_mode(Mode.EXPLORATION)
    assert response_engine.current_mode == Mode.EXPLORATION

    response_engine.set_mode(Mode.STANDBY)
    assert response_engine.current_mode == Mode.STANDBY


# ---------------------------------------------------------------------------
# Contract: set_challenge() sets and clears challenge
# ---------------------------------------------------------------------------


def test_set_challenge_and_clear(response_engine) -> None:
    """set_challenge() must set and clear the active challenge."""
    challenge = Challenge(
        target_type=ChallengeTargetType.COLOR,
        target_label="red",
    )
    response_engine.set_challenge(challenge)
    assert response_engine.active_challenge is challenge

    response_engine.set_challenge(None)
    assert response_engine.active_challenge is None
