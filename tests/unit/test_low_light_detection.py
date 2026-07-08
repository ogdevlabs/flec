"""Unit tests for LowLightDetector in CameraModule.

Verifies:
- Per-frame brightness check (mean pixel value < threshold)
- Below threshold -> emit DetectionEvent triggering "I can't see very well" audio
- Threshold configurable via env var FLEC_LOW_LIGHT_THRESHOLD (default: 40)
- Debounced: one event per 10s maximum
"""

from __future__ import annotations

import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from flec.camera.camera_module import LowLightDetector, _DEFAULT_LOW_LIGHT_THRESHOLD
from flec.models import DetectionEvent, DetectionType


class TestLowLightDetectorBasic:
    """Basic threshold and return value tests."""

    def test_bright_frame_returns_none(self) -> None:
        """Frame with mean brightness >= threshold MUST return None."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        bright_frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = detector.check(bright_frame)
        assert result is None

    def test_dark_frame_returns_detection_event(self) -> None:
        """Frame with mean brightness < threshold MUST return a DetectionEvent."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)  # mean=0
        result = detector.check(dark_frame)
        assert result is not None
        assert isinstance(result, DetectionEvent)

    def test_dark_frame_event_label_is_low_light(self) -> None:
        """Low-light DetectionEvent label MUST be 'low_light'."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.check(dark_frame)
        assert result is not None
        assert result.label == "low_light"

    def test_dark_frame_event_type(self) -> None:
        """Low-light DetectionEvent must have a valid DetectionType."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.check(dark_frame)
        assert result is not None
        assert isinstance(result.type, DetectionType)

    def test_exactly_at_threshold_returns_none(self) -> None:
        """Frame with mean brightness == threshold is NOT low light."""
        threshold = 40.0
        detector = LowLightDetector(threshold=threshold, debounce_secs=0.0)
        # Mean brightness exactly at threshold
        frame = np.full((100, 100, 3), int(threshold), dtype=np.uint8)
        result = detector.check(frame)
        assert result is None  # >= threshold => not low light

    def test_one_below_threshold_returns_event(self) -> None:
        """Frame with mean brightness 1 below threshold MUST return event."""
        threshold = 40.0
        detector = LowLightDetector(threshold=threshold, debounce_secs=0.0)
        frame = np.full((100, 100, 3), int(threshold) - 1, dtype=np.uint8)
        result = detector.check(frame)
        assert result is not None

    def test_event_metadata_contains_mean_brightness(self) -> None:
        """DetectionEvent metadata MUST contain mean_brightness."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.check(dark_frame)
        assert result is not None
        assert "mean_brightness" in result.metadata

    def test_event_metadata_contains_threshold(self) -> None:
        """DetectionEvent metadata MUST contain threshold value."""
        threshold = 40.0
        detector = LowLightDetector(threshold=threshold, debounce_secs=0.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.check(dark_frame)
        assert result is not None
        assert "threshold" in result.metadata
        assert result.metadata["threshold"] == threshold

    def test_event_confidence_is_1_0(self) -> None:
        """Low-light event MUST have confidence=1.0 (deterministic detection)."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.check(dark_frame)
        assert result is not None
        assert result.confidence == 1.0


class TestLowLightDetectorDebouncing:
    """Debounce behavior: at most one event per debounce_secs."""

    def test_second_call_within_debounce_returns_none(self) -> None:
        """Second low-light detection within debounce window MUST return None."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=10.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # First call — should emit event
        result1 = detector.check(dark_frame)
        assert result1 is not None, "First call should return event"
        # Second call immediately — should be debounced
        result2 = detector.check(dark_frame)
        assert result2 is None, "Second call within debounce window should return None"

    def test_call_after_debounce_window_emits_event(self) -> None:
        """Call after debounce window expires MUST emit a new event."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.05)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # First call
        result1 = detector.check(dark_frame)
        assert result1 is not None

        # Wait for debounce to expire
        time.sleep(0.1)

        # Should emit again
        result2 = detector.check(dark_frame)
        assert result2 is not None, "Should emit event after debounce window expires"

    def test_many_calls_within_debounce_emit_only_one_event(self) -> None:
        """Multiple consecutive calls within debounce window MUST emit exactly one event."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=10.0)
        dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        events = [detector.check(dark_frame) for _ in range(20)]
        non_none = [e for e in events if e is not None]
        assert len(non_none) == 1, (
            f"Expected exactly 1 event in debounce window, got {len(non_none)}"
        )


class TestLowLightDetectorEdgeCases:
    """Edge cases and invalid inputs."""

    def test_none_frame_returns_none(self) -> None:
        """None input MUST return None (never raises)."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        result = detector.check(None)  # type: ignore[arg-type]
        assert result is None

    def test_1d_array_returns_none(self) -> None:
        """1D array MUST return None (never raises)."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        result = detector.check(np.array([1, 2, 3]))
        assert result is None

    def test_non_ndarray_returns_none(self) -> None:
        """Non-ndarray input MUST return None (never raises)."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        result = detector.check("not a frame")  # type: ignore[arg-type]
        assert result is None

    def test_float32_frame_works(self) -> None:
        """Float32 frames MUST be handled without errors."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_float_frame = np.zeros((100, 100, 3), dtype=np.float32)
        result = detector.check(dark_float_frame)
        assert result is not None

    def test_grayscale_frame_works(self) -> None:
        """2D grayscale frames MUST be handled without errors."""
        detector = LowLightDetector(threshold=40.0, debounce_secs=0.0)
        dark_gray = np.zeros((100, 100), dtype=np.uint8)
        result = detector.check(dark_gray)
        assert result is not None  # mean=0 < 40 => low light


class TestLowLightThresholdFromEnv:
    """Threshold configurable via FLEC_LOW_LIGHT_THRESHOLD env var."""

    def test_default_threshold_is_40(self) -> None:
        """Default threshold MUST be 40."""
        assert _DEFAULT_LOW_LIGHT_THRESHOLD == 40

    def test_env_var_overrides_threshold(self) -> None:
        """FLEC_LOW_LIGHT_THRESHOLD env var MUST override the default threshold."""
        with patch.dict(os.environ, {"FLEC_LOW_LIGHT_THRESHOLD": "80"}):
            from flec.camera.camera_module import _get_low_light_threshold
            threshold = _get_low_light_threshold()
            assert threshold == 80.0

    def test_env_var_60_accepts_frame_at_50(self) -> None:
        """When threshold=60, frame with mean=50 should be low-light."""
        detector = LowLightDetector(threshold=60.0, debounce_secs=0.0)
        frame = np.full((50, 50, 3), 50, dtype=np.uint8)
        result = detector.check(frame)
        assert result is not None, "mean=50 < threshold=60 should trigger low-light"

    def test_threshold_property(self) -> None:
        """LowLightDetector.threshold property MUST reflect configured threshold."""
        detector = LowLightDetector(threshold=55.0)
        assert detector.threshold == 55.0
