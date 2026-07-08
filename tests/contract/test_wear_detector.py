"""Contract tests for WearDetector.

Tests verify the public interface contract defined in
specs/001-perception-core/contracts/module-interfaces.md.

All tests use mock frames — no real camera or MediaPipe model required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flec.models import DetectionEvent, DetectionType, WearState


# ---------------------------------------------------------------------------
# Contract: WearDetector must be importable from its defined path
# ---------------------------------------------------------------------------


def test_wear_detector_importable() -> None:
    """WearDetector must be importable from flec.perception.wear_detector."""
    from flec.perception.wear_detector import WearDetector  # noqa: F401


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Return a completely black BGR frame (no face, no skin)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def face_frame() -> np.ndarray:
    """Return a BGR frame simulating close-range skin-tone content (face proxy).

    We fill the centre with a skin-tone HSV colour converted to BGR.
    This avoids needing a real face photo while still exercising the skin
    detection fallback path.
    """
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Add a large skin-tone region in the centre (HSV approx 10°, 50%, 80%)
    # BGR value for that HSV: approximately (102, 163, 204)
    frame[160:320, 213:427] = (102, 163, 204)
    return frame


# ---------------------------------------------------------------------------
# Contract: current_state property returns WearState
# ---------------------------------------------------------------------------


def test_initial_state_is_off_head() -> None:
    """current_state is OFF_HEAD before any frame is processed."""
    from flec.perception.wear_detector import WearDetector

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector()
        assert detector.current_state == WearState.OFF_HEAD


# ---------------------------------------------------------------------------
# Contract: update() returns WearState
# ---------------------------------------------------------------------------


def test_update_returns_wear_state_type(blank_frame: np.ndarray) -> None:
    """update() must return a WearState enum value."""
    from flec.perception.wear_detector import WearDetector

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector()
        result = detector.update(blank_frame)
        assert isinstance(result, WearState)


def test_update_returns_off_head_on_blank_frame(blank_frame: np.ndarray) -> None:
    """update() returns OFF_HEAD when frame has no face or skin content."""
    from flec.perception.wear_detector import WearDetector

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector()
        result = detector.update(blank_frame)
        assert result == WearState.OFF_HEAD


def test_update_returns_on_head_when_face_detected(face_frame: np.ndarray) -> None:
    """update() returns ON_HEAD when MediaPipe reports a face detection."""
    from flec.perception.wear_detector import WearDetector

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[_make_face_detection()])
        # debounce_seconds=0 so transition is immediate in the test
        detector = WearDetector(debounce_seconds=0.0)
        result = detector.update(face_frame)
        assert result == WearState.ON_HEAD


# ---------------------------------------------------------------------------
# Contract: no DetectionEvent spam — exactly one event per state transition
# ---------------------------------------------------------------------------


def test_no_event_emitted_when_state_unchanged(blank_frame: np.ndarray) -> None:
    """Repeated updates with the same state must not emit duplicate events."""
    from flec.perception.wear_detector import WearDetector

    events: list[DetectionEvent] = []

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector(on_event=events.append)

        # Call update 5 times with the same (OFF_HEAD) result
        for _ in range(5):
            detector.update(blank_frame)

    # Should be at most the initial state announcement (or nothing)
    # The key invariant: repeated same-state calls produce 0 additional events
    initial_count = len(events)
    events.clear()

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector(on_event=events.append)
        detector.update(blank_frame)
        detector.update(blank_frame)
        detector.update(blank_frame)

    assert len(events) == 0, (
        f"Expected 0 events for repeated same-state updates, got {len(events)}"
    )


def test_exactly_one_event_on_state_transition() -> None:
    """Exactly one DetectionEvent(type=WEAR) is emitted per state transition."""
    from flec.perception.wear_detector import WearDetector

    events: list[DetectionEvent] = []
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    face = np.full((480, 640, 3), 200, dtype=np.uint8)
    face[160:320, 213:427] = (102, 163, 204)

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        # Use debounce_seconds=0 so transitions are immediate without time.sleep
        detector = WearDetector(on_event=events.append, debounce_seconds=0.0)

        # First call: no face → OFF_HEAD (starts OFF, no transition event)
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector.update(blank)
        assert len(events) == 0

        # Transition to ON_HEAD (1 event)
        _setup_mediapipe_mock(mock_mp, detections=[_make_face_detection()])
        detector.update(face)
        assert len(events) == 1
        assert events[0].type == DetectionType.WEAR
        assert events[0].label == WearState.ON_HEAD.name

        # Still ON_HEAD (no new event)
        detector.update(face)
        assert len(events) == 1

        # Transition to OFF_HEAD (1 more event)
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector.update(blank)
        assert len(events) == 2
        assert events[1].type == DetectionType.WEAR
        assert events[1].label == WearState.OFF_HEAD.name


# ---------------------------------------------------------------------------
# Contract: does not raise on dark or corrupted frames
# ---------------------------------------------------------------------------


def test_no_raise_on_dark_frame() -> None:
    """update() must not raise on a completely dark (all-zero) frame."""
    from flec.perception.wear_detector import WearDetector

    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector()
        # Must not raise
        result = detector.update(dark_frame)
        assert isinstance(result, WearState)


def test_no_raise_on_tiny_frame() -> None:
    """update() must not raise on an unusually small frame."""
    from flec.perception.wear_detector import WearDetector

    tiny_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector()
        result = detector.update(tiny_frame)
        assert isinstance(result, WearState)


def test_no_raise_on_corrupted_frame() -> None:
    """update() must not raise on a frame with unexpected dtype."""
    from flec.perception.wear_detector import WearDetector

    # Float32 frame (unusual dtype)
    weird_frame = np.random.rand(480, 640, 3).astype(np.float32)

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[])
        detector = WearDetector()
        result = detector.update(weird_frame)
        assert isinstance(result, WearState)


# ---------------------------------------------------------------------------
# Contract: emitted events have correct DetectionType and valid fields
# ---------------------------------------------------------------------------


def test_wear_event_has_correct_type_and_confidence() -> None:
    """Emitted DetectionEvent must have type=WEAR and confidence in [0.0, 1.0]."""
    from flec.perception.wear_detector import WearDetector

    events: list[DetectionEvent] = []
    face = np.full((480, 640, 3), 200, dtype=np.uint8)
    face[160:320, 213:427] = (102, 163, 204)

    with patch("flec.perception.wear_detector.mp") as mock_mp:
        _setup_mediapipe_mock(mock_mp, detections=[_make_face_detection()])
        detector = WearDetector(on_event=events.append, debounce_seconds=0.0)
        detector.update(face)

    assert len(events) == 1
    event = events[0]
    assert event.type == DetectionType.WEAR
    assert 0.0 <= event.confidence <= 1.0
    assert event.label in (WearState.ON_HEAD.name, WearState.OFF_HEAD.name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_mediapipe_mock(mock_mp: MagicMock, detections: list) -> None:
    """Configure a mediapipe mock to return specified face detections."""
    mock_results = MagicMock()
    mock_results.detections = detections

    mock_face_detection = MagicMock()
    mock_face_detection.FaceDetection.return_value.__enter__ = MagicMock(
        return_value=MagicMock(process=MagicMock(return_value=mock_results))
    )
    mock_face_detection.FaceDetection.return_value.__exit__ = MagicMock(return_value=False)
    mock_mp.solutions.face_detection = mock_face_detection

    # Configure FaceDetection constructor for direct instantiation
    mock_detector_instance = MagicMock()
    mock_detector_instance.process.return_value = mock_results
    mock_face_detection.FaceDetection.return_value = mock_detector_instance


def _make_face_detection(score: float = 0.95) -> MagicMock:
    """Return a mock MediaPipe face detection result with given score."""
    detection = MagicMock()
    detection.score = [score]
    return detection
