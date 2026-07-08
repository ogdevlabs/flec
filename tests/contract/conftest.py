"""Contract test fixtures — shared mocks and frame helpers for interface testing."""

import numpy as np
import pytest


@pytest.fixture
def red_circle_frame() -> np.ndarray:
    """Return a BGR frame containing a red circle on white background."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    import cv2
    cv2.circle(frame, (320, 240), 100, (0, 0, 255), -1)  # Red circle (BGR)
    return frame


@pytest.fixture
def blue_square_frame() -> np.ndarray:
    """Return a BGR frame containing a blue square on white background."""
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    import cv2
    cv2.rectangle(frame, (220, 140), (420, 340), (255, 0, 0), -1)  # Blue square (BGR)
    return frame
