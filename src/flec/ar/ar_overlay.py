"""AR Overlay — renders detection bounding boxes on the companion dev screen.

In dev mode (FLEC_DEV_MODE=1 or mode='dev'): renders colored bounding boxes
and labels onto an OpenCV frame for display on the companion screen mirror.

In production mode: all methods are no-ops. The AR projection is handled by
dedicated mask hardware; this module only controls the dev-phase companion screen.

Usage:
    overlay = AROverlay(dev_mode=True)
    frame = overlay.draw_detection(frame, event)
    overlay.update(frame, events)

Privacy: no frames are persisted. All rendering is in-memory and ephemeral.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import cv2
import numpy as np

from flec.models import BoundingBox, DetectionEvent, DetectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette for AR borders — maps detection label to BGR color
# ---------------------------------------------------------------------------

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    # Shape colors — each shape gets a distinct neon-style border
    "circle": (0, 255, 255),        # Cyan
    "square": (255, 200, 0),        # Yellow-blue
    "rectangle": (0, 200, 255),     # Orange
    "triangle": (100, 255, 0),      # Lime
    "pentagon": (255, 0, 200),      # Magenta
    "hexagon": (0, 100, 255),       # Orange-red
    "star": (255, 255, 0),          # Bright cyan
    "heart": (0, 0, 255),           # Red
    "oval": (200, 0, 255),          # Violet
    "diamond": (255, 150, 0),       # Sky blue
    # Color detections — use a bright white border
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "green": (0, 200, 0),
    "orange": (0, 140, 255),
    "purple": (200, 0, 200),
    "pink": (180, 100, 255),
    "white": (200, 200, 200),
}

_DEFAULT_COLOR: tuple[int, int, int] = (0, 255, 100)  # Bright green fallback
_BORDER_THICKNESS = 3
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.7
_LABEL_THICKNESS = 2


def _is_dev_mode() -> bool:
    """Return True if running in dev mode (env var or default)."""
    val = os.environ.get("FLEC_DEV_MODE", "0")
    return val.strip() in ("1", "true", "True", "yes")


class AROverlay:
    """Renders AR detection overlays onto a companion dev screen frame.

    In production mode (dev_mode=False), all methods are no-ops.
    """

    def __init__(self, dev_mode: Optional[bool] = None) -> None:
        """Construct overlay renderer.

        Args:
            dev_mode: Force dev or production mode. If None, reads FLEC_DEV_MODE env var.
        """
        if dev_mode is None:
            self._dev_mode = _is_dev_mode()
        else:
            self._dev_mode = dev_mode

        logger.info(json.dumps({
            "event": "ar_overlay_init",
            "dev_mode": self._dev_mode,
        }))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def draw_detection(
        self, frame: np.ndarray, event: DetectionEvent
    ) -> np.ndarray:
        """Draw a single detection event's bounding box onto the frame.

        Args:
            frame: BGR frame (modified in-place and returned).
            event: The detection event to render.

        Returns:
            The frame with the overlay drawn (same array, modified in-place).
            In production mode, returns frame unmodified.
        """
        if not self._dev_mode:
            return frame
        if event.bounding_box is None:
            return frame
        try:
            return self._draw_bbox(frame, event)
        except Exception as exc:
            logger.error(json.dumps({
                "event": "ar_draw_error",
                "error": str(exc),
                "label": event.label,
            }))
            return frame

    def update(
        self, frame: np.ndarray, events: list[DetectionEvent]
    ) -> np.ndarray:
        """Batch-draw all detection events onto the frame.

        Args:
            frame: BGR frame (modified in-place and returned).
            events: List of detection events to render.

        Returns:
            The frame with all overlays drawn.
            In production mode, returns frame unmodified.
        """
        if not self._dev_mode:
            return frame
        for event in events:
            frame = self.draw_detection(frame, event)
        return frame

    @property
    def is_dev_mode(self) -> bool:
        """True if overlay is rendering in dev mode."""
        return self._dev_mode

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _draw_bbox(self, frame: np.ndarray, event: DetectionEvent) -> np.ndarray:
        """Draw bounding box and label for a single event."""
        h, w = frame.shape[:2]
        bb = event.bounding_box
        assert bb is not None  # guarded by caller

        # Convert normalised coords to pixel coords
        x = int(bb.x * w)
        y = int(bb.y * h)
        bw = int(bb.width * w)
        bh = int(bb.height * h)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        color = _LABEL_COLORS.get(event.label, _DEFAULT_COLOR)

        # Draw border rectangle
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, _BORDER_THICKNESS)

        # Draw label + confidence
        conf_pct = int(event.confidence * 100)
        label_text = f"{event.label} {conf_pct}%"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS
        )

        # Background pill for readability
        pad = 4
        label_y = max(y - text_h - pad * 2, 0)
        cv2.rectangle(
            frame,
            (x, label_y),
            (x + text_w + pad * 2, label_y + text_h + baseline + pad * 2),
            color,
            -1,  # filled
        )
        cv2.putText(
            frame,
            label_text,
            (x + pad, label_y + text_h + pad),
            _LABEL_FONT,
            _LABEL_SCALE,
            (0, 0, 0),  # Black text on colored background
            _LABEL_THICKNESS,
            cv2.LINE_AA,
        )

        logger.debug(json.dumps({
            "event": "ar_overlay_drawn",
            "label": event.label,
            "bbox_px": [x, y, bw, bh],
            "color_bgr": list(color),
        }))

        return frame
