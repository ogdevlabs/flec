"""OCRWorker — settle-gated, background Reading-mode OCR (F-001).

Runs OCR on the fingertip crop only when the finger has *settled* (decision D1),
resolves the mirror/normal orientation by confidence delta (decision D2/D3), and
feeds the recognized word to ``FingerTracker.update_ocr`` so ``ResponseEngine``
narrates it. Falls back to the illustration describer when there is no confident
word. Runs off the 30 fps loop so perception/preview never stalls.

Queue-only contract: this is session-level orchestration using the *public*
interfaces of OCRReader / IllustrationDescriber / FingerTracker — the capability
modules still never import one another. Heavy deps are lazy; every path degrades
gracefully (log + no-op) and nothing is persisted.

This module is built incrementally across the F-001 tasks. Pure, thread-free
helpers live at module scope so they are unit-testable without a camera or models.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

#: ROI side length as a fraction of the frame, centered on the fingertip.
_DEFAULT_CROP_FRAC = 0.35


def should_run_ocr(detected: bool, velocity: float, settle_threshold: float) -> bool:
    """Return True when an OCR pass should fire this tick (settle gate, D1).

    Fires only when a fingertip is detected *and* moving slowly enough to be a
    deliberate point (velocity at or below ``settle_threshold``). A fast sweep or
    an absent finger produces no OCR — this is also what keeps browsing silent.
    """
    if not detected:
        return False
    return velocity <= settle_threshold


def mirror_x(x_norm: float) -> float:
    """Reflect a normalized x-coordinate for a horizontally-flipped frame (R8)."""
    return 1.0 - x_norm


def crop_around_fingertip(
    frame: np.ndarray, x_norm: float, y_norm: float, frac: float = _DEFAULT_CROP_FRAC
) -> np.ndarray:
    """Return a bounded ROI centered on the fingertip (clamped to frame bounds).

    Reading only the word under the finger bounds OCR cost on the ARM64 target
    (decision D4) and naturally selects the pointed word.
    """
    h, w = frame.shape[:2]
    half_w = max(1, int(w * frac / 2))
    half_h = max(1, int(h * frac / 2))
    cx = int(min(max(x_norm, 0.0), 1.0) * w)
    cy = int(min(max(y_norm, 0.0), 1.0) * h)
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    if x1 <= x0:
        x0, x1 = 0, w
    if y1 <= y0:
        y0, y1 = 0, h
    return frame[y0:y1, x0:x1]


def resolve_orientation(
    crop: np.ndarray,
    read_region: Callable[[np.ndarray], "tuple[str, float]"],
    *,
    cached: Optional[str] = None,
    conf_gate: float = 0.4,
    delta_gate: float = 0.1,
) -> "tuple[str, float, str]":
    """Resolve mirror/normal orientation for a fingertip crop (decisions D2/D3).

    OCRs the crop as-captured and horizontally flipped via ``read_region`` (which
    returns ``(text, confidence)``), and returns ``(text, confidence, orientation)``
    where orientation is ``"normal"`` / ``"mirror"`` / ``""`` (empty = stay silent).

    Silence wins over a guess: nothing is returned unless the top confidence clears
    ``conf_gate`` and the normal-vs-mirror confidence delta clears ``delta_gate``
    (unless both orientations agree on the text, in which case orientation is moot).
    When ``cached`` names a known orientation, only that one is probed (single OCR).
    """
    # Cache fast-path — probe only the known orientation.
    if cached == "normal":
        text, conf = read_region(crop)
        if conf >= conf_gate:
            return text, conf, "normal"
    elif cached == "mirror":
        text, conf = read_region(crop[:, ::-1])
        if conf >= conf_gate:
            return text, conf, "mirror"

    n_text, n_conf = read_region(crop)
    m_text, m_conf = read_region(crop[:, ::-1])

    if n_conf >= m_conf:
        best_text, best_conf, orient, other = n_text, n_conf, "normal", m_conf
    else:
        best_text, best_conf, orient, other = m_text, m_conf, "mirror", n_conf

    if best_conf < conf_gate:
        return "", best_conf, ""
    # Both orientations agree on the text → orientation is irrelevant, speak it.
    if n_text and n_text == m_text:
        return best_text, best_conf, orient
    # Ambiguous which orientation is correct → stay silent (correctness > coverage).
    if (best_conf - other) < delta_gate:
        return "", best_conf, ""
    return best_text, best_conf, orient
