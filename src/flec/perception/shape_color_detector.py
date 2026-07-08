"""ShapeColorDetector — detects shapes and colors in a camera frame.

Strategy:
  1. HSV-based color detection: identifies dominant color regions using
     configurable HSV range tables for the 8 spec colors.
  2. Contour-based shape classification: approximates polygons from large
     contours and classifies by vertex count and aspect ratio for 10 shapes.
  3. YOLOv8n (optional): if a model file is present at `.models/yolov8n.pt`,
     it is loaded once at construction and used to improve detection accuracy
     for real-world frames. Gracefully skipped when ultralytics is not available
     or the model file is absent.

Emits structured JSON logs on every detection and on errors (never raises).

Contract: detect(frame) always returns list[DetectionEvent]. Never raises.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from flec.models import BoundingBox, DetectionEvent, DetectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HSV color range table
# Color name → list of (lower_hsv, upper_hsv) tuples (multiple ranges per color)
# OpenCV HSV: H in [0,179], S in [0,255], V in [0,255]
# ---------------------------------------------------------------------------

_HSV_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (np.array([0, 100, 80]), np.array([10, 255, 255])),
        (np.array([160, 100, 80]), np.array([179, 255, 255])),
    ],
    "orange": [
        (np.array([10, 100, 80]), np.array([25, 255, 255])),
    ],
    "yellow": [
        (np.array([22, 80, 80]), np.array([38, 255, 255])),
    ],
    "green": [
        (np.array([38, 80, 40]), np.array([82, 255, 255])),
    ],
    "blue": [
        (np.array([95, 80, 40]), np.array([135, 255, 255])),
    ],
    "purple": [
        (np.array([125, 60, 40]), np.array([160, 255, 255])),
    ],
    "pink": [
        (np.array([145, 40, 80]), np.array([175, 255, 255])),
        (np.array([0, 40, 150]), np.array([10, 100, 255])),
    ],
    "white": [
        (np.array([0, 0, 180]), np.array([179, 60, 255])),
    ],
}

# Minimum pixel area for a color region to be considered a detection
_MIN_COLOR_AREA_FRACTION = 0.005  # 0.5% of frame area

# Minimum contour area (as fraction of frame) for shape detection
_MIN_SHAPE_AREA_FRACTION = 0.005


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_bbox(x: int, y: int, w: int, h: int, frame_h: int, frame_w: int) -> BoundingBox:
    """Convert pixel rect to normalised BoundingBox, clamped to [0, 1]."""
    nx = float(np.clip(x / frame_w, 0.0, 1.0))
    ny = float(np.clip(y / frame_h, 0.0, 1.0))
    nw = float(np.clip(w / frame_w, 0.0, 1.0))
    nh = float(np.clip(h / frame_h, 0.0, 1.0))
    # Ensure x+w and y+h don't exceed 1.0
    nw = min(nw, 1.0 - nx)
    nh = min(nh, 1.0 - ny)
    return BoundingBox(x=nx, y=ny, width=nw, height=nh)


def _classify_shape(contour: np.ndarray) -> Optional[str]:
    """Classify a contour into one of the 10 spec shapes.

    Uses polygon approximation (vertex count) and aspect ratio as discriminators.
    Returns None if the contour does not match any known shape.
    """
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1.0
    area = cv2.contourArea(contour)
    rect_area = w * h if (w * h) > 0 else 1
    extent = area / rect_area  # how much of bounding rect is filled

    # Circularity: 4π·area / perimeter²  → 1.0 for perfect circle
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    if vertices == 3:
        return "triangle"

    if vertices == 4:
        # Diamond: low extent (rhombus shape — lots of empty corners in bounding rect)
        if extent < 0.60:
            return "diamond"
        # Distinguish square vs rectangle by aspect ratio
        if 0.85 <= aspect_ratio <= 1.18:
            return "square"
        return "rectangle"

    if vertices == 5:
        return "pentagon"

    if vertices == 6:
        return "hexagon"

    if vertices >= 10:
        # Star: low extent (lots of empty space in bounding rect), low circularity
        if extent < 0.65 and circularity < 0.55:
            return "star"
        # High circularity and extent → circle or oval
        if circularity > 0.80:
            return "circle"
        return "oval"

    if vertices == 7 or vertices == 8 or vertices == 9:
        # Could be oval/circle approximation or heart
        if circularity > 0.80:
            if 0.85 <= aspect_ratio <= 1.15:
                return "circle"
            return "oval"
        # Heart: aspect ratio 0.7–1.4, moderate circularity (0.55–0.80), good extent
        # Heart has a concave top (dip between the two lobes) → lower circularity than oval
        if 0.70 <= aspect_ratio <= 1.40 and 0.55 <= circularity <= 0.80 and extent > 0.55:
            return "heart"
        return "oval"

    # High-vertex polygon (>= 11 vertices) — treat as circle or oval
    if circularity > 0.82:
        if 0.85 <= aspect_ratio <= 1.15:
            return "circle"
        return "oval"

    # Diamond (rhombus): 4 vertices with aspect_ratio ~1, rotated 45° → looks like 4 pts
    # Already handled in vertices==4; but tight epsilon may give more vertices
    # If extent is moderate and circularity low → diamond
    if 0.5 <= aspect_ratio <= 1.5 and circularity < 0.70 and extent < 0.70:
        return "diamond"

    return None


def _classify_shape_strict(contour: np.ndarray, frame_h: int, frame_w: int) -> Optional[str]:
    """Stricter shape classifier that also handles diamond via multiple epsilons."""
    # First pass: standard epsilon
    result = _classify_shape(contour)
    if result is not None:
        return result

    # Second pass: tighter epsilon (catches star and diamond better)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1.0
    area = cv2.contourArea(contour)
    rect_area = w * h if (w * h) > 0 else 1
    extent = area / rect_area
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    if vertices == 4 and 0.5 <= aspect_ratio <= 1.5 and extent < 0.65:
        return "diamond"

    if 8 <= vertices <= 14 and circularity > 0.60:
        return "star"

    # Heart approximation via tighter polygon
    if 7 <= vertices <= 12 and 0.7 <= aspect_ratio <= 1.3 and extent > 0.50:
        return "heart"

    return None


# ---------------------------------------------------------------------------
# YOLOv8n optional loader
# ---------------------------------------------------------------------------

def _try_load_yolo(model_path: Path):  # type: ignore[return]
    """Attempt to load YOLOv8n from model_path. Return model or None."""
    if not model_path.exists():
        return None
    try:
        from ultralytics import YOLO  # type: ignore[import]
        model = YOLO(str(model_path))
        logger.info(json.dumps({
            "event": "yolo_loaded",
            "model_path": str(model_path),
        }))
        return model
    except ImportError:
        logger.debug(json.dumps({
            "event": "yolo_unavailable",
            "reason": "ultralytics not installed",
        }))
        return None
    except Exception as exc:
        logger.warning(json.dumps({
            "event": "yolo_load_error",
            "error": str(exc),
        }))
        return None


# ---------------------------------------------------------------------------
# ShapeColorDetector
# ---------------------------------------------------------------------------

class ShapeColorDetector:
    """Detects shapes and colors from camera frames.

    Usage:
        detector = ShapeColorDetector()
        events: list[DetectionEvent] = detector.detect(frame)

    Always returns a list (empty if nothing detected). Never raises.
    """

    _MODELS_DIR = Path(".models")

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._yolo = _try_load_yolo(
            model_path if model_path is not None else self._MODELS_DIR / "yolov8n.pt"
        )
        logger.info(json.dumps({
            "event": "shape_color_detector_init",
            "yolo_available": self._yolo is not None,
        }))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[DetectionEvent]:
        """Process frame and return all detected shapes and colors.

        Returns an empty list on blank/dark/unreadable frames.
        Never raises.
        """
        try:
            return self._detect_impl(frame)
        except Exception as exc:
            logger.error(json.dumps({
                "event": "shape_color_detector_error",
                "error": str(exc),
                "timestamp": time.time(),
            }))
            return []

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _detect_impl(self, frame: np.ndarray) -> list[DetectionEvent]:
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        min_area = frame_h * frame_w * _MIN_SHAPE_AREA_FRACTION

        events: list[DetectionEvent] = []

        # Step 1: Color detection via HSV ranges
        color_events = self._detect_colors(frame, frame_h, frame_w)
        events.extend(color_events)

        # Step 2: Contour-based shape classification
        shape_events = self._detect_shapes(frame, frame_h, frame_w, min_area)
        events.extend(shape_events)

        # Step 3: YOLOv8n augmentation (if model available)
        if self._yolo is not None:
            yolo_events = self._detect_yolo(frame, frame_h, frame_w)
            events.extend(yolo_events)

        # Log all detections
        if events:
            logger.info(json.dumps({
                "event": "detections",
                "count": len(events),
                "labels": [e.label for e in events],
                "timestamp": time.time(),
            }))

        return events

    def _detect_colors(
        self, frame: np.ndarray, frame_h: int, frame_w: int
    ) -> list[DetectionEvent]:
        """Detect dominant color regions using HSV range matching."""
        events: list[DetectionEvent] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_area = frame_h * frame_w
        min_area = frame_area * _MIN_COLOR_AREA_FRACTION

        for color_name, ranges in _HSV_RANGES.items():
            # Combine masks for multi-range colors (e.g. red spans 0-10 and 160-179)
            combined_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            # Find connected regions
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            largest_area = 0
            best_contour = None
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if a > min_area and a > largest_area:
                    largest_area = a
                    best_contour = cnt

            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                bbox = _to_bbox(x, y, w, h, frame_h, frame_w)
                confidence = min(1.0, largest_area / (frame_area * 0.3))
                confidence = float(np.clip(confidence, 0.1, 1.0))

                event = DetectionEvent(
                    type=DetectionType.COLOR,
                    label=color_name,
                    confidence=confidence,
                    bounding_box=bbox,
                )
                events.append(event)
                logger.debug(json.dumps({
                    "event": "color_detected",
                    "label": color_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x, y, w, h],
                }))

        return events

    def _detect_shapes(
        self, frame: np.ndarray, frame_h: int, frame_w: int, min_area: float
    ) -> list[DetectionEvent]:
        """Detect shapes using contour analysis + polygon approximation."""
        events: list[DetectionEvent] = []

        # Convert to grayscale and threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Also try Canny for better edge-based detection
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        shape_events: list[DetectionEvent] = []

        for src in [thresh, edges]:
            contours, _ = cv2.findContours(
                src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue

                # Skip contours that fill almost the entire frame (likely background)
                if area > (frame_h * frame_w * 0.80):
                    continue

                shape_label = _classify_shape_strict(cnt, frame_h, frame_w)
                if shape_label is None:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                bbox = _to_bbox(x, y, w, h, frame_h, frame_w)

                # Confidence: based on area relative to frame + shape regularity
                area_score = min(1.0, area / (frame_h * frame_w * 0.20))
                confidence = float(np.clip(0.5 + area_score * 0.5, 0.3, 0.95))

                event = DetectionEvent(
                    type=DetectionType.SHAPE,
                    label=shape_label,
                    confidence=confidence,
                    bounding_box=bbox,
                )
                shape_events.append(event)
                logger.debug(json.dumps({
                    "event": "shape_detected",
                    "label": shape_label,
                    "confidence": round(confidence, 3),
                    "bbox": [x, y, w, h],
                    "area": int(area),
                }))

        # Deduplicate shape events by label (keep highest confidence)
        seen: dict[str, DetectionEvent] = {}
        for e in shape_events:
            if e.label not in seen or e.confidence > seen[e.label].confidence:
                seen[e.label] = e

        events.extend(seen.values())
        return events

    def _detect_yolo(
        self, frame: np.ndarray, frame_h: int, frame_w: int
    ) -> list[DetectionEvent]:
        """Augment detection using YOLOv8n for real-world robustness."""
        events: list[DetectionEvent] = []
        try:
            results = self._yolo(frame, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < 0.3:
                        continue
                    label = result.names.get(cls_id, "unknown")
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = _to_bbox(
                        int(x1), int(y1),
                        int(x2 - x1), int(y2 - y1),
                        frame_h, frame_w
                    )
                    # Map YOLO labels to spec vocabulary where possible
                    normalized_label = _map_yolo_label(label)
                    if normalized_label is None:
                        continue
                    event = DetectionEvent(
                        type=DetectionType.SHAPE,
                        label=normalized_label,
                        confidence=min(1.0, conf),
                        bounding_box=bbox,
                    )
                    events.append(event)
        except Exception as exc:
            logger.warning(json.dumps({
                "event": "yolo_inference_error",
                "error": str(exc),
            }))
        return events


def _map_yolo_label(label: str) -> Optional[str]:
    """Map a YOLO class label to a spec shape/color vocabulary word, or None."""
    # YOLO detects objects; map common objects to shape associations
    # This is a best-effort mapping — primarily used for real-world scenes
    mapping = {
        "circle": "circle",
        "square": "square",
        "rectangle": "rectangle",
        "triangle": "triangle",
    }
    return mapping.get(label.lower())
