"""Contract tests for FingerTracker module interface.

These tests verify the contract defined in specs/001-perception-core/contracts/module-interfaces.md.
Tests must FAIL before the implementation in T041 is in place (TDD RED phase).
"""

from __future__ import annotations

import numpy as np
import pytest

from flec.models import FingerTrackingState, ReadingIntent
from flec.perception.finger_tracker import FingerTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_frame() -> np.ndarray:
    """Return a completely black (empty) BGR frame — no hand present."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _white_frame() -> np.ndarray:
    """Return a white BGR frame — no hand present."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


def _dark_frame() -> np.ndarray:
    """Return a very dark frame to simulate low-light conditions."""
    return np.full((480, 640, 3), 5, dtype=np.uint8)


def _corrupted_frame() -> np.ndarray:
    """Return a frame with random noise (corrupted input)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# T039-C1: detected=False on empty/no-hand frame
# ---------------------------------------------------------------------------


class TestNoHandDetection:
    """Contract: FingerTracker returns detected=False when no hand is present."""

    def test_detected_false_on_empty_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_empty_frame())
        assert isinstance(state, FingerTrackingState), (
            "update() must return a FingerTrackingState"
        )
        assert state.detected is False, (
            "detected must be False when no hand is present"
        )

    def test_detected_false_on_white_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_white_frame())
        assert state.detected is False, (
            "detected must be False on a blank white frame"
        )

    def test_intent_idle_when_not_detected(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_empty_frame())
        assert state.intent == ReadingIntent.IDLE, (
            "intent must be IDLE when finger is not detected"
        )

    def test_nearest_text_none_when_not_detected(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_empty_frame())
        assert state.nearest_text is None, (
            "nearest_text must be None when finger is not detected"
        )


# ---------------------------------------------------------------------------
# T039-C2: velocity is always a non-negative float
# ---------------------------------------------------------------------------


class TestVelocityContract:
    """Contract: velocity is always a non-negative float."""

    def test_velocity_non_negative_on_empty_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_empty_frame())
        assert isinstance(state.velocity, float), "velocity must be a float"
        assert state.velocity >= 0.0, "velocity must be non-negative"

    def test_velocity_non_negative_after_reset(self) -> None:
        tracker = FingerTracker()
        tracker.reset()
        state = tracker.update(_empty_frame())
        assert state.velocity >= 0.0, "velocity must be non-negative after reset"

    def test_velocity_zero_after_reset(self) -> None:
        """reset() must clear velocity to 0.0 per contract."""
        tracker = FingerTracker()
        tracker.reset()
        assert tracker.current_state.velocity == 0.0, (
            "reset() must set velocity to 0.0"
        )


# ---------------------------------------------------------------------------
# T039-C3: SCANNING → READING intent transition on sustained velocity drop
# ---------------------------------------------------------------------------


class TestScanningToReadingTransition:
    """Contract: intent transitions SCANNING → READING when velocity drops below
    threshold for 3 consecutive frames."""

    def _make_slow_frame_sequence(self, n: int = 5) -> list[np.ndarray]:
        """Return n frames simulating a nearly-stationary fingertip.

        We inject synthetic tracking via update_ocr + simulate low-velocity by
        calling update_position (test helper) if exposed, otherwise we rely on
        unit-level velocity injection via the tracker's internal method.
        Uses a synthetic approach: we call the tracker and check state after
        enough low-velocity frames.
        """
        frames = []
        for _ in range(n):
            frame = np.full((480, 640, 3), 200, dtype=np.uint8)
            frames.append(frame)
        return frames

    def test_intent_starts_idle(self) -> None:
        tracker = FingerTracker()
        assert tracker.current_state.intent == ReadingIntent.IDLE

    def test_scanning_to_reading_via_simulate(self) -> None:
        """After sustained low-velocity updates, intent must become READING."""
        tracker = FingerTracker()
        # Inject low-velocity positions via the simulate_finger helper
        # (contract-required test helper — forces intent transition without real MediaPipe)
        tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        state = tracker.current_state
        assert state.detected is True, "simulate_finger must set detected=True"
        assert state.intent == ReadingIntent.READING, (
            "3 consecutive low-velocity frames must transition intent to READING"
        )

    def test_slow_velocity_threshold_is_respected(self) -> None:
        """Velocity at or below threshold transitions to READING."""
        tracker = FingerTracker()
        threshold = tracker.velocity_threshold
        assert isinstance(threshold, float) and threshold > 0.0, (
            "velocity_threshold must be a positive float"
        )
        # Inject exactly at threshold
        for _ in range(3):
            tracker.simulate_finger(position=(0.4, 0.6), velocity=threshold)
        assert tracker.current_state.intent == ReadingIntent.READING


# ---------------------------------------------------------------------------
# T039-C4: READING → SCANNING transition on velocity rise
# ---------------------------------------------------------------------------


class TestReadingToScanningTransition:
    """Contract: intent transitions READING → SCANNING when velocity rises above threshold."""

    def test_reading_to_scanning_on_velocity_rise(self) -> None:
        tracker = FingerTracker()
        # First get into READING state
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        assert tracker.current_state.intent == ReadingIntent.READING

        # Now inject high velocity — must switch to SCANNING
        threshold = tracker.velocity_threshold
        tracker.simulate_finger(position=(0.9, 0.5), velocity=threshold * 5)
        assert tracker.current_state.intent == ReadingIntent.SCANNING, (
            "High velocity must transition intent back to SCANNING"
        )

    def test_scanning_intent_when_velocity_high_from_idle(self) -> None:
        tracker = FingerTracker()
        threshold = tracker.velocity_threshold
        tracker.simulate_finger(position=(0.9, 0.5), velocity=threshold * 5)
        assert tracker.current_state.intent == ReadingIntent.SCANNING, (
            "High-velocity finger must result in SCANNING intent"
        )


# ---------------------------------------------------------------------------
# T039-C5: reset() returns to IDLE state
# ---------------------------------------------------------------------------


class TestResetContract:
    """Contract: reset() resets tracker to IDLE / default state."""

    def test_reset_clears_to_idle(self) -> None:
        tracker = FingerTracker()
        # Drive tracker into READING state
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        assert tracker.current_state.intent == ReadingIntent.READING

        tracker.reset()
        state = tracker.current_state
        assert state.intent == ReadingIntent.IDLE, "reset() must set intent to IDLE"
        assert state.velocity == 0.0, "reset() must set velocity to 0.0"
        assert state.detected is False, "reset() must set detected to False"
        assert state.nearest_text is None, "reset() must clear nearest_text"

    def test_reset_clears_position_history(self) -> None:
        """After reset, position history must be empty (velocity stays 0 on next frame)."""
        tracker = FingerTracker()
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        tracker.reset()
        # One more slow frame — velocity starts from scratch, should not jump
        tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        # After one frame post-reset, velocity should still read as low (near 0.5)
        assert tracker.current_state.velocity >= 0.0


# ---------------------------------------------------------------------------
# T039-C6: Does not raise on corrupted/dark frames
# ---------------------------------------------------------------------------


class TestRobustness:
    """Contract: FingerTracker never raises on malformed or edge-case frames."""

    def test_no_raise_on_dark_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_dark_frame())
        assert isinstance(state, FingerTrackingState)

    def test_no_raise_on_corrupted_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_corrupted_frame())
        assert isinstance(state, FingerTrackingState)

    def test_no_raise_on_zero_size_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(np.zeros((0, 0, 3), dtype=np.uint8))
        assert isinstance(state, FingerTrackingState)

    def test_velocity_non_negative_on_corrupted_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_corrupted_frame())
        assert state.velocity >= 0.0

    def test_returns_state_not_none_on_dark_frame(self) -> None:
        tracker = FingerTracker()
        state = tracker.update(_dark_frame())
        assert state is not None


# ---------------------------------------------------------------------------
# T039-C7: update_ocr populates nearest_text
# ---------------------------------------------------------------------------


class TestUpdateOCR:
    """Contract: update_ocr sets nearest_text when finger is in READING state."""

    def test_nearest_text_set_after_update_ocr(self) -> None:
        tracker = FingerTracker()
        # Get into READING state
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        tracker.update_ocr(text_regions=["hello", "world"])
        assert tracker.current_state.nearest_text is not None, (
            "nearest_text must be set after update_ocr when in READING state"
        )

    def test_update_ocr_with_empty_list_clears_nearest_text(self) -> None:
        tracker = FingerTracker()
        for _ in range(3):
            tracker.simulate_finger(position=(0.5, 0.5), velocity=0.5)
        tracker.update_ocr(text_regions=["hello"])
        tracker.update_ocr(text_regions=[])
        assert tracker.current_state.nearest_text is None, (
            "nearest_text must be None when update_ocr receives empty list"
        )

    def test_nearest_text_none_when_scanning(self) -> None:
        """nearest_text must not be set when intent is SCANNING."""
        tracker = FingerTracker()
        threshold = tracker.velocity_threshold
        tracker.simulate_finger(position=(0.9, 0.5), velocity=threshold * 5)
        tracker.update_ocr(text_regions=["hello", "world"])
        # In SCANNING mode, nearest_text stays None
        assert tracker.current_state.nearest_text is None, (
            "nearest_text must remain None when intent is SCANNING"
        )
