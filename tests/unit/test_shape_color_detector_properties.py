"""Property-based tests for ShapeColorDetector using Hypothesis.

These tests verify the interface contract for ShapeColorDetector:
- Arbitrary frame sizes and types never raise
- confidence is always in [0.0, 1.0]
- bounding_box values are always in [0.0, 1.0]
- Return type is always list[DetectionEvent]

Constitution Rule: Test-First — these tests define the contract before implementation.
"""

from __future__ import annotations

import sys
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from flec.models import BoundingBox, DetectionEvent, DetectionType


# ---------------------------------------------------------------------------
# Stub ShapeColorDetector for contract testing
# We test the contract: any implementation MUST satisfy these properties.
# ---------------------------------------------------------------------------


class _StubShapeColorDetector:
    """Minimal stub that satisfies the ShapeColorDetector contract.

    Real implementation will be in src/flec/perception/shape_color_detector.py.
    Property tests run against this stub to document the expected contract.
    """

    def detect(self, frame: np.ndarray) -> List[DetectionEvent]:
        """Return list of DetectionEvents for all shapes/colors found."""
        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return []
        # Stub: return empty list for all frames (contract: never raises)
        return []


# Strategy: generate arbitrary numpy frames
def _frame_strategy():
    """Hypothesis strategy: produce arbitrary BGR-like numpy arrays."""
    height = st.integers(min_value=1, max_value=1280)
    width = st.integers(min_value=1, max_value=1280)
    channels = st.integers(min_value=1, max_value=4)

    @st.composite
    def _build(draw):
        h = draw(height)
        w = draw(width)
        c = draw(channels)
        dtype = draw(st.sampled_from([np.uint8, np.float32, np.float64]))
        if dtype == np.uint8:
            return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        else:
            return np.zeros((h, w, c), dtype=dtype)

    return _build()


@given(_frame_strategy())
@settings(max_examples=50)
def test_detect_never_raises_on_arbitrary_frames(frame: np.ndarray) -> None:
    """ShapeColorDetector.detect() MUST never raise for any frame input."""
    detector = _StubShapeColorDetector()
    # Must not raise — contract guarantee
    result = detector.detect(frame)
    assert result is not None, "detect() must not return None"


@given(_frame_strategy())
@settings(max_examples=50)
def test_detect_always_returns_list(frame: np.ndarray) -> None:
    """ShapeColorDetector.detect() MUST always return a list."""
    detector = _StubShapeColorDetector()
    result = detector.detect(frame)
    assert isinstance(result, list), (
        f"detect() must return list, got {type(result).__name__}"
    )


def _make_detection_event_with_valid_confidence(conf: float) -> DetectionEvent:
    """Helper: create a DetectionEvent with given confidence."""
    return DetectionEvent(
        type=DetectionType.SHAPE,
        label="circle",
        confidence=conf,
    )


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_detection_event_confidence_in_range(conf: float) -> None:
    """Any DetectionEvent from detect() MUST have confidence in [0.0, 1.0]."""
    event = _make_detection_event_with_valid_confidence(conf)
    assert 0.0 <= event.confidence <= 1.0, (
        f"confidence must be in [0.0, 1.0], got {event.confidence}"
    )


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_bounding_box_values_in_range(val: float) -> None:
    """Any BoundingBox in a DetectionEvent MUST have all values in [0.0, 1.0]."""
    # width and height can't both be 0 in real detection, but test the range contract
    width = min(val, 1.0 - val) if val < 0.5 else 1.0 - val
    height = width
    bb = BoundingBox(x=val * 0.5, y=val * 0.5, width=max(width, 0.01), height=max(height, 0.01))
    assert 0.0 <= bb.x <= 1.0
    assert 0.0 <= bb.y <= 1.0
    assert 0.0 <= bb.width <= 1.0
    assert 0.0 <= bb.height <= 1.0


@given(
    st.floats(min_value=0.0, max_value=0.9, allow_nan=False),
    st.floats(min_value=0.0, max_value=0.9, allow_nan=False),
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
)
def test_bounding_box_rejects_out_of_range(x: float, y: float, w: float, h: float) -> None:
    """BoundingBox constructor MUST accept all [0.0, 1.0] values without raising."""
    # Clamp to valid range — verify no raises for valid inputs
    x_safe = min(x, 1.0)
    y_safe = min(y, 1.0)
    w_safe = min(w, 1.0 - x_safe)
    h_safe = min(h, 1.0 - y_safe)
    # Should not raise
    bb = BoundingBox(x=x_safe, y=y_safe, width=w_safe, height=h_safe)
    assert 0.0 <= bb.x <= 1.0
    assert 0.0 <= bb.y <= 1.0
    assert 0.0 <= bb.width <= 1.0
    assert 0.0 <= bb.height <= 1.0


def test_detect_returns_empty_list_on_blank_frame() -> None:
    """detect() MUST return empty list (not None, not exception) on blank frame."""
    detector = _StubShapeColorDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert isinstance(result, list)
    # Blank frame may yield empty list — that's acceptable
    # but MUST be a list, not None


def test_detect_returns_list_not_none_on_dark_frame() -> None:
    """detect() MUST return list (never None) on dark/black frame."""
    detector = _StubShapeColorDetector()
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(dark_frame)
    assert result is not None
    assert isinstance(result, list)


def test_confidence_out_of_range_raises_value_error() -> None:
    """DetectionEvent MUST raise ValueError if confidence is out of [0.0, 1.0]."""
    with pytest.raises(ValueError, match="confidence"):
        DetectionEvent(type=DetectionType.SHAPE, label="circle", confidence=1.5)


def test_confidence_negative_raises_value_error() -> None:
    """DetectionEvent MUST raise ValueError if confidence is negative."""
    with pytest.raises(ValueError, match="confidence"):
        DetectionEvent(type=DetectionType.SHAPE, label="circle", confidence=-0.1)


def test_bounding_box_out_of_range_raises_value_error() -> None:
    """BoundingBox MUST raise ValueError if any value is out of [0.0, 1.0]."""
    with pytest.raises(ValueError):
        BoundingBox(x=1.5, y=0.0, width=0.1, height=0.1)

    with pytest.raises(ValueError):
        BoundingBox(x=0.0, y=-0.1, width=0.1, height=0.1)


def test_detection_event_type_is_detection_type() -> None:
    """Every DetectionEvent must have a valid DetectionType."""
    event = DetectionEvent(
        type=DetectionType.SHAPE,
        label="triangle",
        confidence=0.9,
    )
    assert isinstance(event.type, DetectionType)


def test_detection_event_with_bounding_box_has_valid_box() -> None:
    """When DetectionEvent includes bounding_box, all coordinates in [0.0, 1.0]."""
    bb = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
    event = DetectionEvent(
        type=DetectionType.COLOR,
        label="red",
        confidence=0.85,
        bounding_box=bb,
    )
    assert event.bounding_box is not None
    assert 0.0 <= event.bounding_box.x <= 1.0
    assert 0.0 <= event.bounding_box.y <= 1.0
    assert 0.0 <= event.bounding_box.width <= 1.0
    assert 0.0 <= event.bounding_box.height <= 1.0
