"""AR overlay rendering for Flec mask.

In dev mode: renders AR annotations (shape borders, fingertip trails) onto a
BGR frame that can be displayed on the companion phone screen.

In production mode: all draw_* methods are no-ops — the AR hardware handles
projection directly. This module never imports other capability modules.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of historical positions kept for the fingertip trail.
_TRAIL_LENGTH = 5

#: Base radius (pixels) of the glowing fingertip dot.
_DOT_RADIUS = 10

#: BGR colour for the glowing fingertip trail (cyan-white glow).
_TRAIL_COLOR_BGR = (255, 220, 50)  # Light cyan-yellow

#: BGR colour for the solid tip of the dot.
_DOT_COLOR_BGR = (255, 255, 255)   # White

#: Border thickness for shape detection overlays.
_BORDER_THICKNESS = 3

#: Default border colour (green, toddler-friendly).
_DEFAULT_BORDER_COLOR_BGR = (0, 255, 100)


# ---------------------------------------------------------------------------
# AROverlay
# ---------------------------------------------------------------------------


class AROverlay:
    """Renders AR annotations onto BGR frames (dev mode only).

    All draw_* methods are safe no-ops in production mode.
    """

    def __init__(self, dev_mode: bool = True) -> None:
        self._dev_mode = dev_mode

        # Circular buffer of recent fingertip positions (normalised [0.0, 1.0]).
        self._trail: deque[tuple[float, float]] = deque(maxlen=_TRAIL_LENGTH)

        logger.info(
            json.dumps({"event": "ar_overlay_init", "dev_mode": dev_mode})
        )

    # ------------------------------------------------------------------
    # Fingertip trail
    # ------------------------------------------------------------------

    def draw_fingertip(
        self,
        frame: np.ndarray,
        position: tuple[float, float],
    ) -> np.ndarray:
        """Draw a glowing fingertip dot and trail at the given normalised position.

        Args:
            frame:    BGR frame to annotate (modified in-place and returned).
            position: Normalised (x, y) in [0.0, 1.0] — fraction of frame width/height.

        Returns:
            The annotated frame. If dev_mode is False, the original frame is
            returned unchanged.
        """
        if not self._dev_mode:
            return frame

        if frame is None or frame.size == 0:
            return frame

        try:
            return self._draw_fingertip_internal(frame, position)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({"event": "ar_draw_fingertip_error", "error": str(exc)})
            )
            return frame

    def _draw_fingertip_internal(
        self,
        frame: np.ndarray,
        position: tuple[float, float],
    ) -> np.ndarray:
        """Internal drawing — may raise; wrapped by draw_fingertip()."""
        import cv2  # type: ignore[import-untyped]

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame

        # Clamp position to valid frame coordinates.
        nx = max(0.0, min(1.0, position[0]))
        ny = max(0.0, min(1.0, position[1]))
        px = int(nx * w)
        py = int(ny * h)

        # Add current position to trail.
        self._trail.append((nx, ny))

        # Draw trail: older positions get smaller radius and lower opacity.
        trail_list = list(self._trail)
        n_trail = len(trail_list)

        for i, (tx, ty) in enumerate(trail_list[:-1]):  # Skip the last (drawn as main dot)
            alpha = (i + 1) / n_trail  # Opacity increases towards tip
            radius = max(2, int(_DOT_RADIUS * alpha * 0.7))
            tpx = int(tx * w)
            tpy = int(ty * h)

            # Blend trail dot onto frame using alpha.
            overlay = frame.copy()
            cv2.circle(overlay, (tpx, tpy), radius, _TRAIL_COLOR_BGR, -1)
            cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, frame)

        # Draw primary glowing dot at current position (multi-layer for glow effect).
        # Outer glow (large, low opacity).
        glow_overlay = frame.copy()
        cv2.circle(glow_overlay, (px, py), _DOT_RADIUS * 2, _TRAIL_COLOR_BGR, -1)
        cv2.addWeighted(glow_overlay, 0.3, frame, 0.7, 0, frame)

        # Inner solid dot.
        cv2.circle(frame, (px, py), _DOT_RADIUS, _TRAIL_COLOR_BGR, -1)
        # Bright centre.
        cv2.circle(frame, (px, py), _DOT_RADIUS // 2, _DOT_COLOR_BGR, -1)

        logger.debug(
            json.dumps(
                {
                    "event": "ar_fingertip_drawn",
                    "x": round(nx, 4),
                    "y": round(ny, 4),
                    "trail_length": n_trail,
                }
            )
        )
        return frame

    def clear_trail(self) -> None:
        """Clear fingertip trail history. Call on mode transitions."""
        self._trail.clear()
        logger.debug(json.dumps({"event": "ar_trail_cleared"}))

    # ------------------------------------------------------------------
    # Shape/color border overlays (Exploration Mode)
    # ------------------------------------------------------------------

    def draw_shape_border(
        self,
        frame: np.ndarray,
        bbox_normalised: tuple[float, float, float, float],
        color_bgr: tuple[int, int, int] = _DEFAULT_BORDER_COLOR_BGR,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """Draw a bounding-box border around a detected shape.

        Args:
            frame:             BGR frame to annotate.
            bbox_normalised:   (x, y, width, height) all in [0.0, 1.0].
            color_bgr:         BGR border colour.
            label:             Optional text label to render above the box.

        Returns:
            Annotated frame. No-op if dev_mode is False.
        """
        if not self._dev_mode:
            return frame
        if frame is None or frame.size == 0:
            return frame

        try:
            return self._draw_shape_border_internal(frame, bbox_normalised, color_bgr, label)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({"event": "ar_draw_border_error", "error": str(exc)})
            )
            return frame

    def _draw_shape_border_internal(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        color_bgr: tuple[int, int, int],
        label: Optional[str],
    ) -> np.ndarray:
        import cv2  # type: ignore[import-untyped]

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame

        bx, by, bw, bh = bbox
        x1 = int(bx * w)
        y1 = int(by * h)
        x2 = int((bx + bw) * w)
        y2 = int((by + bh) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, _BORDER_THICKNESS)

        if label:
            font_scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Background pill behind text.
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color_bgr, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return frame

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dev_mode(self) -> bool:
        """True if dev-mode rendering is active."""
        return self._dev_mode

    @property
    def trail_length(self) -> int:
        """Number of positions currently in the trail buffer."""
        return len(self._trail)
