"""Contract tests for ShapeColorDetector.

Verifies the interface contract defined in:
  specs/001-perception-core/contracts/module-interfaces.md

All 10 shapes and 8 colors from the spec vocabulary must be detectable.
The detector must never raise — it must return an empty list on blank input.
All bounding_box and confidence values must be in [0.0, 1.0].
"""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from flec.models import DetectionEvent, DetectionType, BoundingBox
from flec.perception.shape_color_detector import ShapeColorDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> ShapeColorDetector:
    """Return a fresh ShapeColorDetector instance."""
    return ShapeColorDetector()


def make_color_frame(bgr_color: tuple[int, int, int]) -> np.ndarray:
    """Return a 480x640 frame filled with a solid color (as a large rectangle)."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    # Draw a large solid rectangle of the given color
    cv2.rectangle(frame, (100, 80), (540, 400), bgr_color, -1)
    return frame


def make_shape_frame(shape: str) -> np.ndarray:
    """Return a synthetic frame with a red-filled shape for testing detection."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    color = (0, 0, 200)  # Red in BGR
    cx, cy = 320, 240

    if shape == "circle":
        cv2.circle(frame, (cx, cy), 100, color, -1)
    elif shape == "square":
        cv2.rectangle(frame, (cx - 90, cy - 90), (cx + 90, cy + 90), color, -1)
    elif shape == "rectangle":
        cv2.rectangle(frame, (cx - 130, cy - 70), (cx + 130, cy + 70), color, -1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - 110], [cx - 110, cy + 90], [cx + 110, cy + 90]], np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "pentagon":
        pts = []
        import math
        for i in range(5):
            angle = math.radians(90 + i * 72)
            pts.append([int(cx + 100 * math.cos(angle)), int(cy - 100 * math.sin(angle))])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "hexagon":
        pts = []
        import math
        for i in range(6):
            angle = math.radians(30 + i * 60)
            pts.append([int(cx + 100 * math.cos(angle)), int(cy - 100 * math.sin(angle))])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "star":
        # 5-pointed star via inner/outer radius
        import math
        outer, inner = 100, 40
        pts = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r = outer if i % 2 == 0 else inner
            pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "heart":
        # Approximate heart using two circles + triangle
        cv2.circle(frame, (cx - 45, cy - 20), 55, color, -1)
        cv2.circle(frame, (cx + 45, cy - 20), 55, color, -1)
        pts = np.array([[cx - 95, cy + 15], [cx + 95, cy + 15], [cx, cy + 100]], np.int32)
        cv2.fillPoly(frame, [pts], color)
    elif shape == "oval":
        cv2.ellipse(frame, (cx, cy), (130, 80), 0, 0, 360, color, -1)
    elif shape == "diamond":
        pts = np.array([[cx, cy - 110], [cx + 90, cy], [cx, cy + 110], [cx - 90, cy]], np.int32)
        cv2.fillPoly(frame, [pts], color)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return frame


# ---------------------------------------------------------------------------
# T026-A: Shape detection — all 10 spec shapes
# ---------------------------------------------------------------------------

SPEC_SHAPES = [
    "circle",
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "hexagon",
    "star",
    "heart",
    "oval",
    "diamond",
]


@pytest.mark.parametrize("shape", SPEC_SHAPES)
def test_detects_shape(detector: ShapeColorDetector, shape: str) -> None:
    """Detector must return at least one SHAPE event matching each spec shape."""
    frame = make_shape_frame(shape)
    events = detector.detect(frame)
    shape_labels = [e.label for e in events if e.type == DetectionType.SHAPE]
    assert shape in shape_labels, (
        f"Expected shape '{shape}' in detected labels, got: {shape_labels}"
    )


# ---------------------------------------------------------------------------
# T026-B: Color detection — all 8 spec colors
# ---------------------------------------------------------------------------

# BGR tuples for each spec color — use distinct, saturated values
SPEC_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (0, 0, 220)),
    ("blue", (220, 0, 0)),
    ("yellow", (0, 220, 220)),
    ("green", (0, 180, 0)),
    ("orange", (0, 140, 220)),
    ("purple", (180, 0, 180)),
    ("pink", (180, 100, 220)),
    ("white", (240, 240, 240)),
]


@pytest.mark.parametrize("color,bgr", SPEC_COLORS)
def test_detects_color(
    detector: ShapeColorDetector, color: str, bgr: tuple[int, int, int]
) -> None:
    """Detector must return at least one COLOR event matching each spec color."""
    frame = make_color_frame(bgr)
    events = detector.detect(frame)
    color_labels = [e.label for e in events if e.type == DetectionType.COLOR]
    assert color in color_labels, (
        f"Expected color '{color}' in detected labels, got: {color_labels}"
    )


# ---------------------------------------------------------------------------
# T026-C: Robustness — blank and dark frames
# ---------------------------------------------------------------------------


def test_blank_frame_returns_empty_list(detector: ShapeColorDetector) -> None:
    """detect() must return an empty list on a blank (black) frame — never raise."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    events = detector.detect(blank)
    assert isinstance(events, list), "detect() must return a list, not None"
    assert events == [], f"Expected empty list on blank frame, got: {events}"


def test_dark_frame_returns_empty_list(detector: ShapeColorDetector) -> None:
    """detect() must return an empty list on a very dark frame — never raise."""
    dark = np.full((480, 640, 3), 5, dtype=np.uint8)
    events = detector.detect(dark)
    assert isinstance(events, list)
    assert events == [], f"Expected empty list on dark frame, got: {events}"


def test_white_frame_returns_list_not_exception(detector: ShapeColorDetector) -> None:
    """detect() must return a list (possibly non-empty) on a pure white frame — never raise."""
    white = np.full((480, 640, 3), 255, dtype=np.uint8)
    try:
        events = detector.detect(white)
    except Exception as exc:
        pytest.fail(f"detect() raised on white frame: {exc}")
    assert isinstance(events, list)


# ---------------------------------------------------------------------------
# T026-D: Bounding box values in [0.0, 1.0]
# ---------------------------------------------------------------------------


def test_bounding_box_values_normalized(detector: ShapeColorDetector) -> None:
    """All bounding_box attributes must be in [0.0, 1.0] for every returned event."""
    frame = make_shape_frame("circle")
    events = detector.detect(frame)
    assert events, "Expected at least one event from circle frame"
    for event in events:
        if event.bounding_box is not None:
            bb = event.bounding_box
            for attr in ("x", "y", "width", "height"):
                val = getattr(bb, attr)
                assert 0.0 <= val <= 1.0, (
                    f"bounding_box.{attr} out of range: {val!r} in event {event}"
                )


def test_bounding_box_present_on_shape_event(detector: ShapeColorDetector) -> None:
    """SHAPE events must include a bounding_box (not None)."""
    frame = make_shape_frame("square")
    events = detector.detect(frame)
    shape_events = [e for e in events if e.type == DetectionType.SHAPE]
    assert shape_events, "Expected at least one SHAPE event from square frame"
    for event in shape_events:
        assert event.bounding_box is not None, (
            f"SHAPE event missing bounding_box: {event}"
        )


# ---------------------------------------------------------------------------
# T026-E: Confidence values in [0.0, 1.0]
# ---------------------------------------------------------------------------


def test_confidence_values_in_range(detector: ShapeColorDetector) -> None:
    """All confidence values must be in [0.0, 1.0]."""
    for shape in ["circle", "square", "triangle"]:
        frame = make_shape_frame(shape)
        events = detector.detect(frame)
        for event in events:
            assert 0.0 <= event.confidence <= 1.0, (
                f"confidence out of range: {event.confidence!r} in event {event}"
            )


# ---------------------------------------------------------------------------
# T026-F: Return type guarantees
# ---------------------------------------------------------------------------


def test_returns_list_type(detector: ShapeColorDetector) -> None:
    """detect() must always return a list, never None."""
    frame = make_shape_frame("circle")
    result = detector.detect(frame)
    assert isinstance(result, list), f"detect() returned {type(result)}, expected list"


def test_all_events_are_detection_events(detector: ShapeColorDetector) -> None:
    """Every item in the returned list must be a DetectionEvent."""
    frame = make_shape_frame("triangle")
    events = detector.detect(frame)
    for item in events:
        assert isinstance(item, DetectionEvent), (
            f"Expected DetectionEvent, got {type(item)}: {item}"
        )


def test_event_type_is_shape_or_color(detector: ShapeColorDetector) -> None:
    """Every returned event must have type SHAPE or COLOR."""
    frame = make_shape_frame("circle")
    events = detector.detect(frame)
    for event in events:
        assert event.type in (DetectionType.SHAPE, DetectionType.COLOR), (
            f"Unexpected DetectionType: {event.type} in event {event}"
        )
