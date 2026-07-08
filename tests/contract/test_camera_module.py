"""Contract tests for CameraModule.

Tests verify the public interface contract defined in
specs/001-perception-core/contracts/module-interfaces.md.

All tests use a mock camera (no real hardware required).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Contract: CameraModule must be importable from its defined path
# ---------------------------------------------------------------------------


def test_camera_module_importable() -> None:
    """CameraModule must be importable from flec.camera.camera_module."""
    from flec.camera.camera_module import CameraModule  # noqa: F401


# ---------------------------------------------------------------------------
# Contract: get_frame() returns None before start() is called
# ---------------------------------------------------------------------------


def test_get_frame_returns_none_before_start() -> None:
    """get_frame() returns None when capture has not been started."""
    from flec.camera.camera_module import CameraModule

    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.isOpened.return_value = False
        cam = CameraModule()
        result = cam.get_frame()
        assert result is None, "get_frame() must return None before start() is called"


# ---------------------------------------------------------------------------
# Contract: is_running is False before start() is called
# ---------------------------------------------------------------------------


def test_is_running_false_before_start() -> None:
    """is_running property must be False before start() is called."""
    from flec.camera.camera_module import CameraModule

    with patch("cv2.VideoCapture"):
        cam = CameraModule()
        assert cam.is_running is False


# ---------------------------------------------------------------------------
# Contract: start() makes is_running True within 2 seconds
# ---------------------------------------------------------------------------


def test_start_makes_is_running_true_within_2s() -> None:
    """start() must set is_running=True within 2 seconds."""
    from flec.camera.camera_module import CameraModule

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap_class.return_value = mock_cap

        cam = CameraModule()
        cam.start()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cam.is_running:
                break
            time.sleep(0.05)

        assert cam.is_running is True, "is_running must be True within 2 seconds of start()"

        cam.stop()


# ---------------------------------------------------------------------------
# Contract: stop() makes is_running False and releases camera resource
# ---------------------------------------------------------------------------


def test_stop_makes_is_running_false_and_releases_resource() -> None:
    """stop() must set is_running=False and release the camera resource."""
    from flec.camera.camera_module import CameraModule

    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap_class.return_value = mock_cap

        cam = CameraModule()
        cam.start()

        # Wait for start to take effect
        deadline = time.monotonic() + 2.0
        while not cam.is_running and time.monotonic() < deadline:
            time.sleep(0.05)

        cam.stop()

        # Give the background thread time to stop
        deadline = time.monotonic() + 2.0
        while cam.is_running and time.monotonic() < deadline:
            time.sleep(0.05)

        assert cam.is_running is False, "is_running must be False after stop()"
        mock_cap.release.assert_called(), "camera resource must be released after stop()"


# ---------------------------------------------------------------------------
# Contract: get_frame() returns BGR numpy array of consistent shape after start()
# ---------------------------------------------------------------------------


def test_get_frame_returns_bgr_numpy_array_after_start() -> None:
    """get_frame() returns a BGR numpy array of consistent shape after start()."""
    from flec.camera.camera_module import CameraModule

    expected_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, expected_frame)
        mock_cap_class.return_value = mock_cap

        cam = CameraModule()
        cam.start()

        # Wait for at least one frame to be captured
        deadline = time.monotonic() + 2.0
        frame = None
        while time.monotonic() < deadline:
            frame = cam.get_frame()
            if frame is not None:
                break
            time.sleep(0.05)

        cam.stop()

        assert frame is not None, "get_frame() must return a frame after start()"
        assert isinstance(frame, np.ndarray), "frame must be a numpy array"
        assert frame.ndim == 3, "frame must have 3 dimensions (H, W, C)"
        assert frame.shape[2] == 3, "frame must have 3 channels (BGR)"
        assert frame.dtype == np.uint8, "frame must be uint8"


# ---------------------------------------------------------------------------
# Contract: frames have consistent shape across multiple get_frame() calls
# ---------------------------------------------------------------------------


def test_frames_have_consistent_shape() -> None:
    """Multiple get_frame() calls must return arrays of the same shape."""
    from flec.camera.camera_module import CameraModule

    base_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("cv2.VideoCapture") as mock_cap_class:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, base_frame)
        mock_cap_class.return_value = mock_cap

        cam = CameraModule()
        cam.start()

        # Collect a few frames
        frames = []
        deadline = time.monotonic() + 2.0
        while len(frames) < 3 and time.monotonic() < deadline:
            f = cam.get_frame()
            if f is not None:
                frames.append(f)
            time.sleep(0.05)

        cam.stop()

        assert len(frames) >= 1, "Must capture at least one frame"
        shapes = {f.shape for f in frames}
        assert len(shapes) == 1, f"All frames must have consistent shape; got {shapes}"
