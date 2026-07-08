"""Integration test: Wear Lifecycle (US5).

Tests the end-to-end session lifecycle triggered by wear and voice events,
verifying Session state machine and ResponseEngine routing behave correctly.

All tests use mocked TTS and no real hardware.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch
from typing import Optional

import pytest

from flec.models import (
    AudioPriority,
    AudioResponse,
    CommandIntent,
    DetectionEvent,
    DetectionType,
    Mode,
    VoiceCommand,
    WearState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tts() -> MagicMock:
    """Return a mock TTSEngine that records speak() calls."""
    tts = MagicMock()
    tts.is_speaking = False
    tts.spoken: list[AudioResponse] = []

    def record_speak(response: AudioResponse) -> None:
        tts.spoken.append(response)

    tts.speak.side_effect = record_speak
    return tts


@pytest.fixture
def session():
    """Return a fresh Session instance."""
    from flec.session import Session
    return Session()


@pytest.fixture
def response_engine(mock_tts):
    """Return a ResponseEngine backed by mock TTS."""
    from flec.engine.response_engine import ResponseEngine
    return ResponseEngine(tts_engine=mock_tts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wear_event(state: WearState) -> DetectionEvent:
    return DetectionEvent(
        type=DetectionType.WEAR,
        label=state.name,
        confidence=1.0,
    )


def _voice_event(intent: CommandIntent, label: Optional[str] = None) -> DetectionEvent:
    lbl = label if label is not None else intent.name
    return DetectionEvent(
        type=DetectionType.VOICE_CMD,
        label=intent.name,
        confidence=1.0,
        metadata={"intent": intent.name},
    )


# ---------------------------------------------------------------------------
# Test: ON_HEAD event → Session EXPLORATION, ResponseEngine routes
# ---------------------------------------------------------------------------


def test_on_head_transitions_session_to_exploration(session) -> None:
    """ON_HEAD wear event must transition session from STANDBY to EXPLORATION."""
    assert session.mode == Mode.STANDBY

    session.handle_wear_event(WearState.ON_HEAD)

    assert session.mode == Mode.EXPLORATION
    assert session.wear_state == WearState.ON_HEAD
    assert session.is_worn is True


def test_on_head_event_routes_through_response_engine(mock_tts, response_engine) -> None:
    """ON_HEAD DetectionEvent routed through ResponseEngine must not raise and set mode."""
    event = _wear_event(WearState.ON_HEAD)
    response_engine.on_event(event)

    # ResponseEngine should be in EXPLORATION mode after ON_HEAD
    assert response_engine.current_mode == Mode.EXPLORATION


# ---------------------------------------------------------------------------
# Test: OFF_HEAD event → Session STANDBY, mask-off audio queued
# ---------------------------------------------------------------------------


def test_off_head_transitions_session_to_standby(session) -> None:
    """OFF_HEAD event must suspend all modes and set STANDBY."""
    # Put on first
    session.handle_wear_event(WearState.ON_HEAD)
    assert session.mode == Mode.EXPLORATION

    # Now remove
    session.handle_wear_event(WearState.OFF_HEAD)
    assert session.mode == Mode.STANDBY
    assert session.wear_state == WearState.OFF_HEAD
    assert session.is_active is False


def test_off_head_queues_mask_off_audio(mock_tts, response_engine) -> None:
    """OFF_HEAD event must queue the CRITICAL 'put mask back on' AudioResponse."""
    # First put on
    response_engine.on_event(_wear_event(WearState.ON_HEAD))
    mock_tts.spoken.clear()

    # Now remove
    response_engine.on_event(_wear_event(WearState.OFF_HEAD))

    critical_responses = [r for r in mock_tts.spoken if r.priority == AudioPriority.CRITICAL]
    assert len(critical_responses) >= 1
    mask_off_text = critical_responses[0].text.lower()
    assert any(
        phrase in mask_off_text
        for phrase in ("mask", "hero", "put")
    ), f"Expected mask-off audio, got: {mask_off_text!r}"


# ---------------------------------------------------------------------------
# Test: Re-wear (ON_HEAD again) → Session resumes EXPLORATION
# ---------------------------------------------------------------------------


def test_rewear_resumes_exploration(session) -> None:
    """After removal, re-wearing must resume EXPLORATION without interaction."""
    session.handle_wear_event(WearState.ON_HEAD)
    session.handle_wear_event(WearState.OFF_HEAD)
    assert session.mode == Mode.STANDBY

    # Re-wear
    session.handle_wear_event(WearState.ON_HEAD)
    assert session.mode == Mode.EXPLORATION
    assert session.is_worn is True
    assert session.is_active is True


def test_rewear_multiple_cycles(session) -> None:
    """Multiple on/off/on cycles must all succeed."""
    for _ in range(3):
        session.handle_wear_event(WearState.ON_HEAD)
        assert session.mode == Mode.EXPLORATION
        session.handle_wear_event(WearState.OFF_HEAD)
        assert session.mode == Mode.STANDBY


# ---------------------------------------------------------------------------
# Test: VOICE_CMD(SHUTDOWN) while ON_HEAD → farewell + session ends
# ---------------------------------------------------------------------------


def test_shutdown_while_worn_plays_farewell_and_ends_session(
    session, mock_tts, response_engine
) -> None:
    """SHUTDOWN command while ON_HEAD must play farewell audio and enter STANDBY."""
    # Put on
    response_engine.on_event(_wear_event(WearState.ON_HEAD))
    session.handle_wear_event(WearState.ON_HEAD)
    assert session.mode == Mode.EXPLORATION
    mock_tts.spoken.clear()

    # Issue voice shutdown via session
    cmd = VoiceCommand(intent=CommandIntent.SHUTDOWN)
    session.handle_voice_command(cmd)

    # Session should be in STANDBY
    assert session.mode == Mode.STANDBY

    # Response engine should queue farewell audio (CRITICAL priority) when routed
    response_engine.on_event(_voice_event(CommandIntent.SHUTDOWN))
    critical = [r for r in mock_tts.spoken if r.priority == AudioPriority.CRITICAL]
    assert len(critical) >= 1
    farewell_text = critical[0].text.lower()
    assert any(
        phrase in farewell_text
        for phrase in ("see you", "next time", "hero", "bye")
    ), f"Expected farewell audio, got: {farewell_text!r}"
    assert response_engine.current_mode == Mode.STANDBY


# ---------------------------------------------------------------------------
# Test: VOICE_CMD(SHUTDOWN) while OFF_HEAD → ignored
# ---------------------------------------------------------------------------


def test_shutdown_while_not_worn_is_ignored(mock_tts, response_engine) -> None:
    """SHUTDOWN command while OFF_HEAD must produce no audio response."""
    # Do NOT put on — starts OFF_HEAD
    assert response_engine.current_mode == Mode.STANDBY
    mock_tts.spoken.clear()

    # Issue shutdown event while off head
    response_engine.on_event(_voice_event(CommandIntent.SHUTDOWN))

    critical = [r for r in mock_tts.spoken if r.priority == AudioPriority.CRITICAL]
    assert len(critical) == 0, (
        f"Expected no CRITICAL audio while off-head, got: {critical}"
    )


def test_session_handle_voice_command_shutdown_while_off_head(session) -> None:
    """Session.handle_voice_command(SHUTDOWN) must be ignored when mask is off."""
    assert session.wear_state == WearState.OFF_HEAD
    # Session starts in STANDBY
    assert session.mode == Mode.STANDBY

    cmd = VoiceCommand(intent=CommandIntent.SHUTDOWN)
    session.handle_voice_command(cmd)

    # Mode must remain STANDBY (no change)
    assert session.mode == Mode.STANDBY
