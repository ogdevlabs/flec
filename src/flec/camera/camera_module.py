"""CameraModule — frame capture from embedded camera or iPhone Bluetooth proxy.

Includes per-frame brightness check with debounced low-light detection.

Constitution:
- Rule 2: Emits structured JSON logs on all state changes and errors.
- Rule 4: Zero persistence — frames are never written to disk.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import numpy as np

from flec.models import DetectionEvent, DetectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment-configurable constants
# ---------------------------------------------------------------------------

_DEFAULT_LOW_LIGHT_THRESHOLD = 40          # Mean pixel value below this = low light
_DEFAULT_LOW_LIGHT_DEBOUNCE_SECS = 10.0   # Minimum seconds between low-light events


def _get_low_light_threshold() -> float:
    """Return low-light threshold from env or default."""
    raw = os.environ.get("FLEC_LOW_LIGHT_THRESHOLD", str(_DEFAULT_LOW_LIGHT_THRESHOLD))
    try:
        val = float(raw)
        if val < 0 or val > 255:
            logger.warning(
                "FLEC_LOW_LIGHT_THRESHOLD out of valid range [0–255]: %s — using default %s",
                raw,
                _DEFAULT_LOW_LIGHT_THRESHOLD,
            )
            return float(_DEFAULT_LOW_LIGHT_THRESHOLD)
        return val
    except (ValueError, TypeError):
        logger.warning(
            "FLEC_LOW_LIGHT_THRESHOLD is not a valid number: %r — using default %s",
            raw,
            _DEFAULT_LOW_LIGHT_THRESHOLD,
        )
        return float(_DEFAULT_LOW_LIGHT_THRESHOLD)


class LowLightDetector:
    """Per-frame brightness check with debouncing.

    Emits a DetectionEvent when the scene is too dark for reliable perception.
    Debounced to at most one event per `debounce_secs` seconds.

    Privacy: Never stores or logs frame pixel data — only the computed mean.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        debounce_secs: float = _DEFAULT_LOW_LIGHT_DEBOUNCE_SECS,
    ) -> None:
        self._threshold: float = (
            threshold if threshold is not None else _get_low_light_threshold()
        )
        self._debounce_secs = debounce_secs
        self._last_event_time: float = 0.0
        self._lock = threading.Lock()

        logger.info(
            "LowLightDetector initialised with threshold=%.1f debounce_secs=%.1f",
            self._threshold,
            self._debounce_secs,
        )

    @property
    def threshold(self) -> float:
        """Mean pixel brightness threshold below which a frame is considered low-light."""
        return self._threshold

    def check(self, frame: np.ndarray) -> Optional[DetectionEvent]:
        """Check a single frame for low-light conditions.

        Args:
            frame: BGR numpy array (any dtype, any shape with at least 2 dims).

        Returns:
            A DetectionEvent(type=ILLUSTRATION, label="low_light") if low-light
            is detected and the debounce window has passed, else None.
            Returns None if frame is invalid (never raises).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            logger.warning("LowLightDetector.check() received invalid frame — skipping")
            return None

        try:
            mean_brightness = float(np.mean(frame))
        except Exception as exc:  # pragma: no cover — defensive only
            logger.error("Failed to compute frame brightness: %s", exc)
            return None

        if mean_brightness >= self._threshold:
            return None  # Frame is bright enough

        # Low-light condition detected — apply debounce
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_event_time
            if elapsed < self._debounce_secs:
                logger.debug(
                    "Low-light detected (mean=%.2f < threshold=%.1f) — debounced "
                    "(%.1fs since last event, need %.1fs)",
                    mean_brightness,
                    self._threshold,
                    elapsed,
                    self._debounce_secs,
                )
                return None

            self._last_event_time = now

        logger.info(
            "Low-light detected: mean_brightness=%.2f threshold=%.1f",
            mean_brightness,
            self._threshold,
        )

        # Emit a DetectionEvent — the ResponseEngine will trigger the audio response
        # "I can't see very well, can we find more light?"
        return DetectionEvent(
            type=DetectionType.ILLUSTRATION,
            label="low_light",
            confidence=1.0,
            metadata={
                "mean_brightness": round(mean_brightness, 3),
                "threshold": self._threshold,
            },
        )


class CameraModule:
    """Capture frames from camera source.

    In dev phase: wraps OpenCV VideoCapture (laptop/iPhone camera).
    In prod phase: wraps the embedded mask camera hardware.

    Publishes raw frames to a shared frame queue (in-memory only).
    Privacy: Frames are NEVER written to disk at any point.
    """

    def __init__(
        self,
        device_index: int = 0,
        low_light_threshold: Optional[float] = None,
    ) -> None:
        self._device_index = device_index
        self._capture = None  # OpenCV VideoCapture, initialised on start()
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._low_light = LowLightDetector(threshold=low_light_threshold)

        logger.info(
            "CameraModule initialised: device_index=%d", device_index
        )

    def start(self) -> None:
        """Begin frame capture. Non-blocking — runs in background thread."""
        if self._running:
            logger.warning("CameraModule.start() called while already running")
            return

        logger.info("CameraModule starting on device %d", self._device_index)
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="flec-camera",
        )
        self._thread.start()

    def stop(self) -> None:
        """Gracefully stop capture and release camera resource."""
        logger.info("CameraModule stopping")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        with self._lock:
            self._latest_frame = None
        logger.info("CameraModule stopped and camera resource released")

    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest frame (BGR, HxWx3). Returns None if no frame available."""
        with self._lock:
            return self._latest_frame

    @property
    def is_running(self) -> bool:
        """True if capture is active."""
        return self._running

    @property
    def low_light_detector(self) -> LowLightDetector:
        """Expose low-light detector for testing and external event subscription."""
        return self._low_light

    def _capture_loop(self) -> None:
        """Background thread: capture frames and check for low-light conditions."""
        try:
            import cv2  # Deferred to avoid hard dependency in tests
            self._capture = cv2.VideoCapture(self._device_index)
            if not self._capture.isOpened():
                logger.error(
                    "CameraModule: could not open device %d", self._device_index
                )
                self._running = False
                return
        except ImportError:
            logger.error("OpenCV (cv2) not available — camera capture unavailable")
            self._running = False
            return

        logger.info("CameraModule capture loop started on device %d", self._device_index)

        while self._running:
            ret, frame = self._capture.read()
            if not ret or frame is None:
                logger.warning("CameraModule: frame read failed — retrying")
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame

            # Check for low-light and emit event if debounce allows
            event = self._low_light.check(frame)
            if event is not None:
                logger.info(
                    "CameraModule: low-light event emitted (mean_brightness=%.2f)",
                    event.metadata.get("mean_brightness", 0),
                )
                # In the full system, this event would be pushed to the event_queue.
                # For now, we log it; ResponseEngine integration happens in session.py.

        logger.info("CameraModule capture loop ended")
