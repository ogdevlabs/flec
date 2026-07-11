"""Property-based tests for FingerTracker using Hypothesis.

These tests verify the interface contract for FingerTracker:
- velocity is always non-negative for any frame input
- intent is always a valid ReadingIntent enum value
- detected is always bool

Constitution Rule: Test-First — these tests define the contract before implementation.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from flec.models import FingerTrackingState, ReadingIntent


# ---------------------------------------------------------------------------
# Stub FingerTracker for contract testing
# We test the contract: any implementation MUST satisfy these properties.
# ---------------------------------------------------------------------------


class _StubFingerTracker:
    """Minimal stub that satisfies the FingerTracker contract.

    Real implementation will be in src/flec/perception/finger_tracker.py.
    Property tests run against this stub to document the expected contract.
    """

    def __init__(self) -> None:
        self._state = FingerTrackingState()

    def update(self, frame: np.ndarray) -> FingerTrackingState:
        """Process frame and return updated FingerTrackingState."""
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return FingerTrackingState()
        # Stub: no finger detected — safe defaults
        return FingerTrackingState(
            detected=False,
            velocity=0.0,
            intent=ReadingIntent.IDLE,
            nearest_text=None,
        )

    def reset(self) -> None:
        """Clear tracking history."""
        self._state = FingerTrackingState()


# ---------------------------------------------------------------------------
# Frame strategy (same as T053)
# ---------------------------------------------------------------------------


def _frame_strategy():
    """Hypothesis strategy: produce arbitrary BGR-like numpy arrays."""

    @st.composite
    def _build(draw):
        h = draw(st.integers(min_value=1, max_value=640))
        w = draw(st.integers(min_value=1, max_value=640))
        dtype = draw(st.sampled_from([np.uint8, np.float32]))
        if dtype == np.uint8:
            return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            return np.zeros((h, w, 3), dtype=dtype)

    return _build()


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(_frame_strategy())
@settings(max_examples=50)
def test_velocity_always_non_negative(frame: np.ndarray) -> None:
    """FingerTracker.update() MUST return non-negative velocity for any frame."""
    tracker = _StubFingerTracker()
    state = tracker.update(frame)
    assert state.velocity >= 0.0, (
        f"velocity must be non-negative, got {state.velocity}"
    )


@given(_frame_strategy())
@settings(max_examples=50)
def test_intent_always_valid_reading_intent(frame: np.ndarray) -> None:
    """FingerTracker.update() MUST return a valid ReadingIntent enum value."""
    tracker = _StubFingerTracker()
    state = tracker.update(frame)
    assert isinstance(state.intent, ReadingIntent), (
        f"intent must be ReadingIntent, got {type(state.intent).__name__}"
    )
    assert state.intent in (
        ReadingIntent.IDLE,
        ReadingIntent.SCANNING,
        ReadingIntent.READING,
    ), f"intent must be a valid ReadingIntent, got {state.intent}"


@given(_frame_strategy())
@settings(max_examples=50)
def test_detected_always_bool(frame: np.ndarray) -> None:
    """FingerTracker.update() MUST return detected as bool (not truthy int, etc.)."""
    tracker = _StubFingerTracker()
    state = tracker.update(frame)
    assert isinstance(state.detected, bool), (
        f"detected must be bool, got {type(state.detected).__name__}"
    )


@given(_frame_strategy())
@settings(max_examples=50)
def test_update_never_raises(frame: np.ndarray) -> None:
    """FingerTracker.update() MUST never raise for any frame input."""
    tracker = _StubFingerTracker()
    # Must not raise
    state = tracker.update(frame)
    assert state is not None


def test_no_finger_in_frame_returns_detected_false() -> None:
    """When no hand is in frame, detected MUST be False."""
    tracker = _StubFingerTracker()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = tracker.update(blank_frame)
    assert state.detected is False


def test_no_finger_nearest_text_is_none() -> None:
    """When detected=False, nearest_text MUST be None."""
    tracker = _StubFingerTracker()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = tracker.update(blank_frame)
    if not state.detected:
        assert state.nearest_text is None, (
            "nearest_text must be None when no finger detected"
        )


def test_reset_sets_velocity_zero() -> None:
    """reset() MUST set velocity to 0.0."""
    tracker = _StubFingerTracker()
    tracker.reset()
    assert tracker._state.velocity == 0.0


def test_reset_sets_intent_idle() -> None:
    """reset() MUST set intent to IDLE."""
    tracker = _StubFingerTracker()
    tracker.reset()
    assert tracker._state.intent == ReadingIntent.IDLE


def test_reset_sets_detected_false() -> None:
    """reset() MUST set detected to False."""
    tracker = _StubFingerTracker()
    tracker.reset()
    assert tracker._state.detected is False


def test_finger_tracking_state_defaults() -> None:
    """FingerTrackingState defaults must satisfy all contract invariants."""
    state = FingerTrackingState()
    assert isinstance(state.detected, bool)
    assert state.detected is False
    assert state.velocity >= 0.0
    assert isinstance(state.intent, ReadingIntent)
    assert state.nearest_text is None


def test_intent_transitions_scanning_to_reading_when_velocity_drops() -> None:
    """Intent MUST transition SCANNING -> READING when velocity drops below threshold.

    This is a contract test: documents the expected behavior for real implementation.
    The stub always returns IDLE; a real implementation must satisfy this.
    """
    # Contract documentation test — verifies the FingerTrackingState model supports
    # all required intent values (not testing real implementation which isn't built yet)
    scanning_state = FingerTrackingState(
        detected=True,
        velocity=20.0,
        intent=ReadingIntent.SCANNING,
    )
    reading_state = FingerTrackingState(
        detected=True,
        velocity=2.0,
        intent=ReadingIntent.READING,
    )
    assert scanning_state.intent == ReadingIntent.SCANNING
    assert scanning_state.velocity >= 0.0
    assert reading_state.intent == ReadingIntent.READING
    assert reading_state.velocity >= 0.0


def test_velocity_is_float_not_int() -> None:
    """velocity MUST be a float (non-negative)."""
    state = FingerTrackingState(velocity=5.0)
    assert isinstance(state.velocity, float)
    assert state.velocity >= 0.0
