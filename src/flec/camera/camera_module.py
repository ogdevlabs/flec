"""CameraModule — frame capture from camera source.

Captures frames from the embedded camera or iPhone Bluetooth proxy (dev phase).
Publishes frames to an in-memory queue consumed by perception modules.

Contract:
    start()     — begin capture in background thread
    stop()      — gracefully stop capture and release camera resource
    get_frame() — return latest BGR frame (HxWx3) or None
    is_running  — True if capture is active
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from flec.logger import log_event

logger = logging.getLogger(__name__)


class CameraModule:
    """Capture frames from a camera source in a background thread.

    Thread-safe: get_frame() can be called from any thread while capture runs.
    """

    def __init__(self, camera_index: Optional[int] = None) -> None:
        """Initialise the CameraModule.

        Args:
            camera_index: OpenCV camera index. Defaults to FLEC_CAMERA_INDEX env var
                          (default 0).
        """
        if camera_index is None:
            camera_index = int(os.environ.get("FLEC_CAMERA_INDEX", "0"))

        self._camera_index: int = camera_index
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin frame capture. Non-blocking — runs in background thread."""
        if self._running:
            logger.warning("CameraModule.start() called while already running")
            return

        self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            log_event(
                module="CameraModule",
                event_type="camera_open_failed",
                data={"camera_index": self._camera_index},
            )
            logger.error(
                '{"module": "CameraModule", "event": "camera_open_failed", '
                '"camera_index": %d}',
                self._camera_index,
            )
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="flec-camera-capture",
            daemon=True,
        )
        self._thread.start()

        log_event(
            module="CameraModule",
            event_type="capture_started",
            data={"camera_index": self._camera_index},
        )

    def stop(self) -> None:
        """Gracefully stop capture and release camera resource."""
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        with self._frame_lock:
            self._latest_frame = None

        log_event(
            module="CameraModule",
            event_type="capture_stopped",
            data={"camera_index": self._camera_index},
        )

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame (HxWx3 uint8) or None if unavailable."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    @property
    def is_running(self) -> bool:
        """True if capture is active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames from camera."""
        consecutive_failures = 0
        max_failures = 10

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                log_event(
                    module="CameraModule",
                    event_type="capture_error",
                    data={"reason": "VideoCapture not open"},
                )
                break

            ret, frame = self._cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                log_event(
                    module="CameraModule",
                    event_type="frame_read_failed",
                    data={"consecutive_failures": consecutive_failures},
                )
                if consecutive_failures >= max_failures:
                    logger.error(
                        '{"module": "CameraModule", "event": "capture_abandoned", '
                        '"reason": "too many consecutive read failures"}'
                    )
                    break
                time.sleep(0.033)  # ~30fps retry interval
                continue

            consecutive_failures = 0
            with self._frame_lock:
                self._latest_frame = frame

        # Ensure is_running reflects actual stopped state
        self._running = False
