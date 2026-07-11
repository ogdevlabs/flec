"""FingerTracker — adaptive finger-position and velocity tracker for reading intent.

Uses MediaPipe Hands to detect the index fingertip (landmark 8).
Computes a rolling-average velocity over the last N frames to infer reading intent:
  - READING  when velocity stays below threshold for READING_FRAMES consecutive frames
  - SCANNING when velocity rises above threshold
  - IDLE     when no finger is detected

Communication principle (Constitution §III): This module does NOT import other
capability modules. Text regions are injected from outside via update_ocr().

All data is ephemeral — no frames or positions are persisted.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from flec.models import FingerTrackingState, ReadingIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (tunable)
# ---------------------------------------------------------------------------

#: Landmark index for the index fingertip in MediaPipe Hands.
_FINGERTIP_LANDMARK = 8

#: Number of recent velocities to keep for the rolling average.
_VELOCITY_WINDOW = 5

#: Number of consecutive low-velocity frames required to transition SCANNING → READING.
_READING_FRAMES = 3

#: Default velocity threshold (normalised coords/frame).
#: Velocity below this value for _READING_FRAMES frames triggers READING intent.
#: At 30 fps, 0.01 normalised units/frame = ~3% of frame width per frame.
#: A value of 1.0 means "one full frame width per frame" — anything slower is READING.
#: In practice, children's finger movements are 2–10% of frame width per frame
#: when reading deliberately; 1.0 is conservative and safe for v1.
_DEFAULT_VELOCITY_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# FingerTracker
# ---------------------------------------------------------------------------


class FingerTracker:
    """Track index fingertip position and infer reading intent from velocity.

    Public interface (from module-interfaces.md):
        update(frame) -> FingerTrackingState
        reset() -> None

    Test-helper interface (required for contract tests without real MediaPipe):
        simulate_finger(position, velocity) -> None
        update_ocr(text_regions) -> None
        current_state: FingerTrackingState (property)
        velocity_threshold: float (property)
    """

    def __init__(
        self,
        velocity_threshold: float = _DEFAULT_VELOCITY_THRESHOLD,
        reading_frames: int = _READING_FRAMES,
    ) -> None:
        self._threshold = velocity_threshold
        #: Consecutive low-velocity frames required for SCANNING → READING.
        #: Instance-configurable so the live session can use realistic values
        #: (a deliberate hold reads; a fast sweep stays silent) without changing
        #: the contract-test defaults.
        self._reading_frames = reading_frames

        # Rolling window of raw per-frame velocities (normalised coords/frame).
        self._velocity_window: deque[float] = deque(maxlen=_VELOCITY_WINDOW)

        # Count of consecutive frames below threshold (drives SCANNING→READING).
        self._low_velocity_streak: int = 0

        # Mutable state — updated each frame.
        self._state = FingerTrackingState()

        # Last detected normalised position (x, y); None if never seen.
        self._last_pos: Optional[tuple[float, float]] = None

        # Attempt to initialise MediaPipe Hands lazily (not at import time).
        self._mp_hands = None
        self._hands_model = None
        self._mp_available = False
        self._init_mediapipe()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_mediapipe(self) -> None:
        """Try to initialise MediaPipe Hands. Silently degrade if unavailable."""
        try:
            import mediapipe as mp  # type: ignore[import-untyped]
            self._mp_hands = mp.solutions.hands
            self._hands_model = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_available = True
            logger.info(
                json.dumps({"event": "finger_tracker_init", "mediapipe": "ok"})
            )
        except Exception as exc:  # noqa: BLE001
            self._mp_available = False
            logger.warning(
                json.dumps(
                    {"event": "finger_tracker_init", "mediapipe": "unavailable", "reason": str(exc)}
                )
            )

    def _compute_velocity(self, pos: tuple[float, float]) -> float:
        """Compute Euclidean distance from the last known position (normalised).

        Returns 0.0 if there is no previous position.
        """
        if self._last_pos is None:
            return 0.0
        dx = pos[0] - self._last_pos[0]
        dy = pos[1] - self._last_pos[1]
        return math.sqrt(dx * dx + dy * dy)

    def _rolling_velocity(self) -> float:
        """Return the rolling average velocity (mean of the velocity window)."""
        if not self._velocity_window:
            return 0.0
        return sum(self._velocity_window) / len(self._velocity_window)

    def _determine_intent(self, detected: bool, rolling_vel: float) -> ReadingIntent:
        """Map velocity + detection state to a ReadingIntent."""
        if not detected:
            return ReadingIntent.IDLE

        if rolling_vel > self._threshold:
            # Velocity is high — finger is moving fast (scanning / repositioning).
            self._low_velocity_streak = 0
            return ReadingIntent.SCANNING

        # Velocity is at or below threshold — increment streak.
        self._low_velocity_streak += 1
        if self._low_velocity_streak >= self._reading_frames:
            return ReadingIntent.READING

        # Not yet enough consecutive slow frames — remain SCANNING (or stay READING).
        current = self._state.intent
        return current if current == ReadingIntent.READING else ReadingIntent.SCANNING

    def _update_nearest_text(self, intent: ReadingIntent) -> None:
        """Clear nearest_text if not in READING state."""
        if intent != ReadingIntent.READING:
            self._state.nearest_text = None

    # ------------------------------------------------------------------
    # Public interface (from module-interfaces.md)
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> FingerTrackingState:
        """Process a camera frame and return the updated FingerTrackingState.

        Never raises — degrades gracefully on corrupt or empty frames.
        """
        try:
            return self._update_internal(frame)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps(
                    {"event": "finger_tracker_update_error", "error": str(exc)}
                )
            )
            # Return last known state unchanged
            return self._state

    def _update_internal(self, frame: np.ndarray) -> FingerTrackingState:
        """Internal update — may raise; wrapped by update()."""
        if not self._mp_available or self._hands_model is None:
            # MediaPipe not available — return IDLE state.
            self._state = FingerTrackingState()
            return self._state

        if frame is None or frame.size == 0:
            self._state = FingerTrackingState()
            return self._state

        # MediaPipe expects RGB.
        import cv2  # type: ignore[import-untyped]
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            self._state = FingerTrackingState()
            return self._state

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands_model.process(rgb)

        if not results.multi_hand_landmarks:
            # No hand detected.
            self._velocity_window.append(0.0)
            intent = ReadingIntent.IDLE
            self._low_velocity_streak = 0
            self._last_pos = None
            nearest = None
            self._state = FingerTrackingState(
                detected=False,
                position_x=0.0,
                position_y=0.0,
                velocity=self._rolling_velocity(),
                intent=intent,
                nearest_text=nearest,
            )
            return self._state

        # Extract landmark 8 (index fingertip) from first hand.
        landmark = results.multi_hand_landmarks[0].landmark[_FINGERTIP_LANDMARK]
        pos = (landmark.x, landmark.y)  # normalised [0.0, 1.0]

        raw_vel = self._compute_velocity(pos)
        self._velocity_window.append(raw_vel)
        rolling_vel = self._rolling_velocity()
        intent = self._determine_intent(detected=True, rolling_vel=rolling_vel)
        self._last_pos = pos

        # Preserve nearest_text only in READING intent.
        nearest = self._state.nearest_text if intent == ReadingIntent.READING else None

        self._state = FingerTrackingState(
            detected=True,
            position_x=pos[0],
            position_y=pos[1],
            velocity=rolling_vel,
            intent=intent,
            nearest_text=nearest,
        )

        logger.debug(
            json.dumps(
                {
                    "event": "finger_tracked",
                    "x": round(pos[0], 4),
                    "y": round(pos[1], 4),
                    "velocity": round(rolling_vel, 6),
                    "intent": intent.name,
                }
            )
        )
        return self._state

    def reset(self) -> None:
        """Clear tracking history and return to IDLE state.

        Call on mode transitions.
        """
        self._velocity_window.clear()
        self._low_velocity_streak = 0
        self._last_pos = None
        self._state = FingerTrackingState()
        logger.info(json.dumps({"event": "finger_tracker_reset"}))

    # ------------------------------------------------------------------
    # Test-helper interface (required by contract tests)
    # ------------------------------------------------------------------

    def simulate_finger(
        self,
        position: tuple[float, float],
        velocity: float,
    ) -> None:
        """Inject a synthetic finger position and velocity without a real frame.

        Used by contract tests that cannot provide real MediaPipe input.
        velocity is the raw per-frame speed (normalised pixels/frame).
        """
        self._velocity_window.append(abs(velocity))
        rolling_vel = self._rolling_velocity()
        intent = self._determine_intent(detected=True, rolling_vel=rolling_vel)
        self._last_pos = position

        nearest = self._state.nearest_text if intent == ReadingIntent.READING else None

        self._state = FingerTrackingState(
            detected=True,
            position_x=position[0],
            position_y=position[1],
            velocity=rolling_vel,
            intent=intent,
            nearest_text=nearest,
        )

    def update_ocr(self, text_regions: list[str]) -> None:
        """Inject OCR results. Sets nearest_text only when intent is READING.

        Called by the session loop when an OCR result is available.
        Passes an empty list to clear nearest_text.
        """
        if self._state.intent == ReadingIntent.READING and text_regions:
            # Use the first text region as the nearest readable word.
            self._state.nearest_text = text_regions[0]
        else:
            self._state.nearest_text = None

        logger.debug(
            json.dumps(
                {
                    "event": "finger_tracker_ocr_update",
                    "intent": self._state.intent.name,
                    "nearest_text": self._state.nearest_text,
                    "regions": text_regions,
                }
            )
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> FingerTrackingState:
        """Most recent FingerTrackingState."""
        return self._state

    @property
    def velocity_threshold(self) -> float:
        """Velocity threshold (normalised coords/frame) for READING intent transition."""
        return self._threshold
