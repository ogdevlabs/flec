"""Contract tests for CommandSTT — intent parsing interface.

These tests verify the contract defined in
specs/001-perception-core/contracts/module-interfaces.md for CommandSTT.

Tests use text-input mocking (no actual audio hardware required).
All tests must FAIL before T034 implements CommandSTT.
"""

from __future__ import annotations

import pytest

from flec.models import CommandIntent, ChallengeTargetType, VoiceCommand
from flec.speech.command_stt import CommandSTT


class TestCommandSTTIntentParsing:
    """Contract: transcribe() maps transcript text to correct CommandIntent."""

    @pytest.fixture
    def stt(self) -> CommandSTT:
        return CommandSTT()

    # ------------------------------------------------------------------
    # START_CHALLENGE — color targets
    # ------------------------------------------------------------------

    def test_find_something_red_returns_start_challenge(self, stt: CommandSTT) -> None:
        """'find something red' → START_CHALLENGE, target_label='red'."""
        result = stt.transcribe_text("find something red")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "red"
        assert result.target_type == ChallengeTargetType.COLOR

    def test_find_something_blue_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find something blue")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "blue"
        assert result.target_type == ChallengeTargetType.COLOR

    def test_find_something_yellow_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find something yellow")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "yellow"
        assert result.target_type == ChallengeTargetType.COLOR

    def test_find_something_green_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find something green")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "green"
        assert result.target_type == ChallengeTargetType.COLOR

    def test_find_something_orange_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find something orange")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "orange"
        assert result.target_type == ChallengeTargetType.COLOR

    def test_find_something_purple_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find something purple")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "purple"
        assert result.target_type == ChallengeTargetType.COLOR

    def test_find_something_white_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find something white")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "white"
        assert result.target_type == ChallengeTargetType.COLOR

    # ------------------------------------------------------------------
    # START_CHALLENGE — shape targets
    # ------------------------------------------------------------------

    def test_find_a_triangle_returns_start_challenge(self, stt: CommandSTT) -> None:
        """'find a triangle' → START_CHALLENGE, target_label='triangle'."""
        result = stt.transcribe_text("find a triangle")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "triangle"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_circle_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a circle")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "circle"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_square_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a square")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "square"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_rectangle_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a rectangle")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "rectangle"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_star_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a star")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "star"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_heart_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a heart")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "heart"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_diamond_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a diamond")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "diamond"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_an_oval_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find an oval")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "oval"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_pentagon_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a pentagon")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "pentagon"
        assert result.target_type == ChallengeTargetType.SHAPE

    def test_find_a_hexagon_returns_start_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("find a hexagon")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "hexagon"
        assert result.target_type == ChallengeTargetType.SHAPE

    # ------------------------------------------------------------------
    # START_CHALLENGE — case-insensitive / alternate phrasings
    # ------------------------------------------------------------------

    def test_find_command_case_insensitive(self, stt: CommandSTT) -> None:
        """Commands should be recognised regardless of case."""
        result = stt.transcribe_text("Find Something RED")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "red"

    def test_find_a_color_variant_phrasing(self, stt: CommandSTT) -> None:
        """'find a red' should also work."""
        result = stt.transcribe_text("find a red")
        assert result.intent == CommandIntent.START_CHALLENGE
        assert result.target_label == "red"
        assert result.target_type == ChallengeTargetType.COLOR

    # ------------------------------------------------------------------
    # CANCEL_CHALLENGE
    # ------------------------------------------------------------------

    def test_stop_returns_cancel_challenge(self, stt: CommandSTT) -> None:
        """'stop' → CANCEL_CHALLENGE."""
        result = stt.transcribe_text("stop")
        assert result.intent == CommandIntent.CANCEL_CHALLENGE

    def test_cancel_returns_cancel_challenge(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("cancel")
        assert result.intent == CommandIntent.CANCEL_CHALLENGE

    def test_stop_challenge_phrase_returns_cancel(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("stop the challenge")
        assert result.intent == CommandIntent.CANCEL_CHALLENGE

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def test_off_returns_shutdown(self, stt: CommandSTT) -> None:
        """'off' → SHUTDOWN."""
        result = stt.transcribe_text("off")
        assert result.intent == CommandIntent.SHUTDOWN

    def test_shutdown_phrase_returns_shutdown(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("hey flec off")
        assert result.intent == CommandIntent.SHUTDOWN

    # ------------------------------------------------------------------
    # UNKNOWN — no exception
    # ------------------------------------------------------------------

    def test_unknown_phrase_returns_unknown(self, stt: CommandSTT) -> None:
        """Unrecognised speech → UNKNOWN intent, no exception."""
        result = stt.transcribe_text("banana tornado elevator")
        assert result.intent == CommandIntent.UNKNOWN

    def test_empty_string_returns_unknown_no_exception(self, stt: CommandSTT) -> None:
        """Empty transcript → UNKNOWN, no exception."""
        result = stt.transcribe_text("")
        assert result.intent == CommandIntent.UNKNOWN

    def test_garbled_text_returns_unknown_no_exception(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("xzqwerty 12345 !!!")
        assert result.intent == CommandIntent.UNKNOWN

    # ------------------------------------------------------------------
    # VoiceCommand contract: raw_text preserved, no raises
    # ------------------------------------------------------------------

    def test_raw_text_is_preserved(self, stt: CommandSTT) -> None:
        """raw_text on VoiceCommand should echo the input transcript."""
        text = "find something blue"
        result = stt.transcribe_text(text)
        assert result.raw_text == text

    def test_transcribe_returns_voice_command_type(self, stt: CommandSTT) -> None:
        result = stt.transcribe_text("anything")
        assert isinstance(result, VoiceCommand)

    # ------------------------------------------------------------------
    # audio bytes path (calls transcribe() with PCM bytes)
    # ------------------------------------------------------------------

    def test_transcribe_bytes_never_raises_on_empty(self, stt: CommandSTT) -> None:
        """transcribe(b'') should return UNKNOWN without raising."""
        result = stt.transcribe(b"")
        assert result.intent == CommandIntent.UNKNOWN

    def test_transcribe_bytes_never_raises_on_garbage(self, stt: CommandSTT) -> None:
        """transcribe() with random bytes should return UNKNOWN without raising."""
        result = stt.transcribe(b"\x00\xff\xab\xcd" * 100)
        assert result.intent == CommandIntent.UNKNOWN
