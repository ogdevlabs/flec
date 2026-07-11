"""AR Overlay — renders detection bounding boxes and fingertip trails on the companion dev screen.

In dev mode (FLEC_DEV_MODE=1): renders colored bounding boxes, labels, and
fingertip trails onto an OpenCV frame for display on the companion screen mirror.

In production mode: all methods are no-ops. The AR projection is handled by
dedicated mask hardware; this module only controls the dev-phase companion screen.

Privacy: no frames are persisted. All rendering is in-memory and ephemeral.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Optional

import cv2
import numpy as np

from flec.models import BoundingBox, DetectionEvent, DetectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette for AR borders — maps detection label to BGR color
# ---------------------------------------------------------------------------

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "circle":    (0, 255, 255),
    "square":    (255, 200, 0),
    "rectangle": (0, 200, 255),
    "triangle":  (100, 255, 0),
    "pentagon":  (255, 0, 200),
    "hexagon":   (0, 100, 255),
    "star":      (255, 255, 0),
    "heart":     (0, 0, 255),
    "oval":      (200, 0, 255),
    "diamond":   (255, 150, 0),
    "red":       (0, 0, 255),
    "blue":      (255, 0, 0),
    "yellow":    (0, 255, 255),
    "green":     (0, 200, 0),
    "orange":    (0, 140, 255),
    "purple":    (200, 0, 200),
    "pink":      (180, 100, 255),
    "white":     (200, 200, 200),
}

_DEFAULT_COLOR: tuple[int, int, int] = (0, 255, 100)
_BORDER_THICKNESS = 3
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.7
_LABEL_THICKNESS = 2

_TRAIL_LENGTH = 5
_DOT_RADIUS = 10
_TRAIL_COLOR_BGR: tuple[int, int, int] = (255, 220, 50)
_DOT_COLOR_BGR: tuple[int, int, int] = (255, 255, 255)


def _is_dev_mode() -> bool:
    val = os.environ.get("FLEC_DEV_MODE", "0")
    return val.strip() in ("1", "true", "True", "yes")


class AROverlay:
    """Renders AR detection overlays and fingertip trails onto a companion dev screen frame.

    In production mode (dev_mode=False), all methods are no-ops.
    """

    def __init__(self, dev_mode: Optional[bool] = None) -> None:
        if dev_mode is None:
            self._dev_mode = _is_dev_mode()
        else:
            self._dev_mode = dev_mode

        self._trail: deque[tuple[float, float]] = deque(maxlen=_TRAIL_LENGTH)

        logger.info(json.dumps({
            "event": "ar_overlay_init",
            "dev_mode": self._dev_mode,
        }))

    @property
    def is_dev_mode(self) -> bool:
        return self._dev_mode

    @property
    def dev_mode(self) -> bool:
        return self._dev_mode

    @property
    def trail_length(self) -> int:
        return len(self._trail)

    # ------------------------------------------------------------------
    # Shape/color detection overlays (Exploration / Challenge Mode)
    # ------------------------------------------------------------------

    def draw_detection(
        self, frame: np.ndarray, event: DetectionEvent
    ) -> np.ndarray:
        """Draw a single detection event's bounding box onto the frame."""
        if not self._dev_mode:
            return frame
        if event.bounding_box is None:
            return frame
        try:
            return self._draw_bbox(frame, event)
        except Exception as exc:
            logger.error(json.dumps({"event": "ar_draw_error", "error": str(exc), "label": event.label}))
            return frame

    def update(
        self, frame: np.ndarray, events: list[DetectionEvent]
    ) -> np.ndarray:
        """Batch-draw all detection events onto the frame."""
        if not self._dev_mode:
            return frame
        for event in events:
            frame = self.draw_detection(frame, event)
        return frame

    def draw_shape_border(
        self,
        frame: np.ndarray,
        bbox_normalised: tuple[float, float, float, float],
        color_bgr: tuple[int, int, int] = _DEFAULT_COLOR,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw a bounding-box border around a detected shape (normalised coords)."""
        if not self._dev_mode:
            return frame
        if frame is None or frame.size == 0:
            return frame
        try:
            return self._draw_shape_border_internal(frame, bbox_normalised, color_bgr, label)
        except Exception as exc:
            logger.error(json.dumps({"event": "ar_draw_border_error", "error": str(exc)}))
            return frame

    # ------------------------------------------------------------------
    # Fingertip trail (Reading Mode)
    # ------------------------------------------------------------------

    def draw_fingertip(
        self,
        frame: np.ndarray,
        position: tuple[float, float],
    ) -> np.ndarray:
        """Draw a glowing fingertip dot and trail at the given normalised position."""
        if not self._dev_mode:
            return frame
        if frame is None or frame.size == 0:
            return frame
        try:
            return self._draw_fingertip_internal(frame, position)
        except Exception as exc:
            logger.error(json.dumps({"event": "ar_draw_fingertip_error", "error": str(exc)}))
            return frame

    def clear_trail(self) -> None:
        """Clear fingertip trail history. Call on mode transitions."""
        self._trail.clear()
        logger.debug(json.dumps({"event": "ar_trail_cleared"}))

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _draw_bbox(self, frame: np.ndarray, event: DetectionEvent) -> np.ndarray:
        h, w = frame.shape[:2]
        bb = event.bounding_box
        assert bb is not None

        x = max(0, min(int(bb.x * w), w - 1))
        y = max(0, min(int(bb.y * h), h - 1))
        bw = max(1, min(int(bb.width * w), w - x))
        bh = max(1, min(int(bb.height * h), h - y))

        color = _LABEL_COLORS.get(event.label, _DEFAULT_COLOR)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, _BORDER_THICKNESS)

        conf_pct = int(event.confidence * 100)
        label_text = f"{event.label} {conf_pct}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS)

        pad = 4
        label_y = max(y - text_h - pad * 2, 0)
        cv2.rectangle(frame, (x, label_y), (x + text_w + pad * 2, label_y + text_h + baseline + pad * 2), color, -1)
        cv2.putText(frame, label_text, (x + pad, label_y + text_h + pad),
                    _LABEL_FONT, _LABEL_SCALE, (0, 0, 0), _LABEL_THICKNESS, cv2.LINE_AA)

        logger.debug(json.dumps({"event": "ar_overlay_drawn", "label": event.label}))
        return frame

    def _draw_shape_border_internal(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        color_bgr: tuple[int, int, int],
        label: Optional[str],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame
        bx, by, bw, bh = bbox
        x1, y1 = int(bx * w), int(by * h)
        x2, y2 = int((bx + bw) * w), int((by + bh) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, _BORDER_THICKNESS)
        if label:
            font_scale, thickness = 0.8, 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color_bgr, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness)
        return frame

    def _draw_fingertip_internal(
        self,
        frame: np.ndarray,
        position: tuple[float, float],
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame

        nx = max(0.0, min(1.0, position[0]))
        ny = max(0.0, min(1.0, position[1]))
        px, py = int(nx * w), int(ny * h)

        self._trail.append((nx, ny))
        trail_list = list(self._trail)
        n_trail = len(trail_list)

        for i, (tx, ty) in enumerate(trail_list[:-1]):
            alpha = (i + 1) / n_trail
            radius = max(2, int(_DOT_RADIUS * alpha * 0.7))
            tpx, tpy = int(tx * w), int(ty * h)
            overlay = frame.copy()
            cv2.circle(overlay, (tpx, tpy), radius, _TRAIL_COLOR_BGR, -1)
            cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)

        glow_overlay = frame.copy()
        cv2.circle(glow_overlay, (px, py), _DOT_RADIUS * 2, _TRAIL_COLOR_BGR, -1)
        cv2.addWeighted(glow_overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, (px, py), _DOT_RADIUS, _TRAIL_COLOR_BGR, -1)
        cv2.circle(frame, (px, py), _DOT_RADIUS // 2, _DOT_COLOR_BGR, -1)

        logger.debug(json.dumps({"event": "ar_fingertip_drawn", "x": round(nx, 4), "y": round(ny, 4)}))
        return frame
