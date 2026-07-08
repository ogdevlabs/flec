"""WearDetector — determines whether the Flec mask is on a head.

Uses MediaPipe Face Detection as the primary signal and HSV skin-tone
proximity as a fallback. A 2-second debounce prevents spurious transitions.

Contract (module-interfaces.md):
    update(frame) → WearState   — process one frame, return current state
    current_state              — most recent WearState (property)
    on_event callback          — emits DetectionEvent(WEAR) on each transition

Architecture:
    - Modular AI: imports only models.py and logger (no capability cross-imports)
    - Privacy: frame is processed in-memory and discarded; never stored
    - Observability: structured JSON log on every state transition
    - Toddler-First: no error messages; silently handles bad frames
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

import mediapipe as mp

from flec.logger import log_event
from flec.models import DetectionEvent, DetectionType, WearState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Seconds a new state must persist before it is accepted (debounce).
# Prevents rapid ON/OFF flicker from momentary occlusions.
_DEBOUNCE_SECONDS: float = 2.0

# MediaPipe detection confidence threshold.
_FACE_DETECTION_CONFIDENCE: float = 0.5

# Minimum fraction of frame pixels that must be skin-tone for fallback.
_SKIN_PIXEL_FRACTION_THRESHOLD: float = 0.04

# HSV skin-tone range (covers a broad range of skin tones under normal lighting)
_SKIN_LOWER_1 = np.array([0, 48, 80], dtype=np.uint8)    # Hue 0-20
_SKIN_UPPER_1 = np.array([20, 255, 255], dtype=np.uint8)
_SKIN_LOWER_2 = np.array([170, 48, 80], dtype=np.uint8)   # Hue 170-180 (red wrap)
_SKIN_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)


class WearDetector:
    """Determine whether the Flec mask is currently on a head.

    Args:
        on_event: Optional callback receiving DetectionEvent on state transition.
                  Called synchronously from update() — must be thread-safe if
                  update() is called from a background thread.
        debounce_seconds: Minimum duration (s) a state must persist before
                          transitioning. Defaults to 2.0 per FR-001b / SC-003a.
    """

    def __init__(
        self,
        on_event: Optional[Callable[[DetectionEvent], None]] = None,
        debounce_seconds: float = _DEBOUNCE_SECONDS,
    ) -> None:
        self._on_event = on_event
        self._debounce = debounce_seconds

        # State tracking
        self._state: WearState = WearState.OFF_HEAD
        self._candidate_state: WearState = WearState.OFF_HEAD
        self._candidate_since: float = 0.0

        # Lazily initialize MediaPipe Face Detection (deferred to first update()
        # call so that test patches applied at construction time take effect).
        self._face_detection = None

        log_event(
            module="WearDetector",
            event_type="initialized",
            data={"debounce_seconds": debounce_seconds},
        )

    def _get_face_detector(self):
        """Lazy accessor for MediaPipe FaceDetection instance.

        Deferred initialization ensures that patches applied in tests (which
        replace the module-level `mp` reference) are honoured.
        """
        if self._face_detection is None:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=_FACE_DETECTION_CONFIDENCE,
            )
        return self._face_detection

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> WearState:
        """Process a frame and return the current (debounced) WearState.

        Emits a DetectionEvent(type=WEAR) via on_event callback on each
        confirmed state transition. Never raises.

        Args:
            frame: BGR numpy array. Handles unusual dtypes and sizes gracefully.

        Returns:
            Current confirmed WearState (not the candidate).
        """
        try:
            raw_state = self._detect(frame)
        except Exception as exc:
            log_event(
                module="WearDetector",
                event_type="detection_error",
                data={"error": str(exc)},
            )
            raw_state = self._state  # Keep current state on error

        self._maybe_transition(raw_state)
        return self._state

    @property
    def current_state(self) -> WearState:
        """Most recently confirmed WearState."""
        return self._state

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> WearState:
        """Run face detection and skin-tone fallback on one frame.

        Returns the raw (non-debounced) WearState for this frame.
        """
        # Normalise to uint8 BGR for OpenCV / MediaPipe
        if frame.dtype != np.uint8:
            # Clip and cast without storing an intermediate persistent copy
            frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(
                np.uint8
            )

        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            return WearState.OFF_HEAD

        # Primary: MediaPipe Face Detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._get_face_detector().process(rgb)

        if results.detections:
            best_score = max(d.score[0] for d in results.detections)
            log_event(
                module="WearDetector",
                event_type="face_detected",
                data={"score": round(best_score, 3), "count": len(results.detections)},
            )
            return WearState.ON_HEAD

        # Fallback: HSV skin-tone proximity check
        if self._has_skin_tone(frame):
            log_event(
                module="WearDetector",
                event_type="skin_tone_fallback",
                data={"method": "hsv"},
            )
            return WearState.ON_HEAD

        return WearState.OFF_HEAD

    def _has_skin_tone(self, frame: np.ndarray) -> bool:
        """Return True if a sufficient fraction of pixels fall in skin-tone HSV ranges."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except Exception:
            return False

        mask1 = cv2.inRange(hsv, _SKIN_LOWER_1, _SKIN_UPPER_1)
        mask2 = cv2.inRange(hsv, _SKIN_LOWER_2, _SKIN_UPPER_2)
        skin_mask = cv2.bitwise_or(mask1, mask2)

        total_pixels = frame.shape[0] * frame.shape[1]
        skin_pixels = int(np.count_nonzero(skin_mask))
        fraction = skin_pixels / total_pixels if total_pixels > 0 else 0.0

        return fraction >= _SKIN_PIXEL_FRACTION_THRESHOLD

    # ------------------------------------------------------------------
    # Debounced state transition
    # ------------------------------------------------------------------

    def _maybe_transition(self, raw_state: WearState) -> None:
        """Apply debounce logic and emit events on confirmed transitions."""
        now = time.monotonic()

        if raw_state != self._candidate_state:
            # New candidate — reset timer
            self._candidate_state = raw_state
            self._candidate_since = now

            # When debounce is zero, confirm immediately on first occurrence
            if self._debounce == 0.0 and raw_state != self._state:
                self._confirm_transition(raw_state)
            return

        # Same candidate persisting — check if debounce elapsed
        if raw_state == self._state:
            return  # No change needed

        if (now - self._candidate_since) >= self._debounce:
            self._confirm_transition(raw_state)

    def _confirm_transition(self, new_state: WearState) -> None:
        """Apply a confirmed state transition and emit DetectionEvent."""
        prev = self._state
        self._state = new_state

        confidence = 1.0 if new_state == WearState.ON_HEAD else 0.9

        event = DetectionEvent(
            type=DetectionType.WEAR,
            label=new_state.name,
            confidence=confidence,
        )

        log_event(
            module="WearDetector",
            event_type="state_transition",
            data={
                "from": prev.name,
                "to": new_state.name,
                "confidence": confidence,
            },
        )

        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception as exc:
                log_event(
                    module="WearDetector",
                    event_type="callback_error",
                    data={"error": str(exc)},
                )

    # ------------------------------------------------------------------
    # Test support
    # ------------------------------------------------------------------

    def _trigger_detection_for_test(self) -> None:
        """Force an immediate state transition for unit testing.

        This method bypasses debounce and directly confirms the opposite state.
        MUST NOT be called in production code.
        """
        target = (
            WearState.ON_HEAD
            if self._state == WearState.OFF_HEAD
            else WearState.OFF_HEAD
        )
        self._confirm_transition(target)
