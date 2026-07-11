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


def should_run_ocr(detected: bool, velocity: float, settle_threshold: float) -> bool:
    """Return True when an OCR pass should fire this tick (settle gate, D1).

    Fires only when a fingertip is detected *and* moving slowly enough to be a
    deliberate point (velocity at or below ``settle_threshold``). A fast sweep or
    an absent finger produces no OCR — this is also what keeps browsing silent.
    """
    if not detected:
        return False
    return velocity <= settle_threshold
