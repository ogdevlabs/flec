"""Top-level test configuration and shared fixtures."""

import queue
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Return a blank 480x640x3 BGR frame (all zeros / black)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    """Return a white 480x640x3 BGR frame."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def frame_queue() -> queue.Queue:
    """Return an empty in-memory frame queue."""
    return queue.Queue(maxsize=10)


@pytest.fixture
def event_queue() -> queue.Queue:
    """Return an empty in-memory detection event queue."""
    return queue.Queue(maxsize=100)
