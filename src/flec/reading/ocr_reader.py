"""OCRReader — extract text from book page frames using EasyOCR.

Responsibility: Read printed text from camera frames.
Contract: see specs/001-perception-core/contracts/module-interfaces.md

Architecture notes:
- EasyOCR model loaded once at construction from .models/ directory
- Processing is synchronous; caller is responsible for thread dispatch
- Never raises — returns empty string on any error
- Emits structured JSON logs for every recognition event (Principle II)
- Does NOT import other capability modules (Principle III)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Processing timeout in seconds — prevents blocking the frame pipeline
_OCR_TIMEOUT_SECS = 10.0


def _strip_garbage(text: str) -> str:
    """Remove non-printable, non-whitespace characters from OCR output.

    EasyOCR can occasionally emit replacement characters or noise bytes
    from low-confidence detections. This strips them while preserving
    normal printable text and standard whitespace.
    """
    cleaned_chars: list[str] = []
    for char in text:
        # Keep printable characters and normal whitespace
        if char.isprintable() or char in ("\n", "\t", " "):
            # Also filter out Unicode control/format categories
            category = unicodedata.category(char)
            if not category.startswith("C"):
                cleaned_chars.append(char)
    return "".join(cleaned_chars)


def _normalize_whitespace(text: str) -> str:
    """Collapse internal whitespace runs and strip outer whitespace."""
    # Collapse multiple spaces/tabs into single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple newlines into single newline
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _is_degenerate_frame(frame: np.ndarray) -> bool:
    """Return True if frame is too small or malformed to process."""
    if frame is None:
        return True
    if frame.ndim < 2:
        return True
    h, w = frame.shape[:2]
    return h < 8 or w < 8


class OCRReader:
    """Extract readable text from a book-page camera frame using EasyOCR.

    Usage::

        reader = OCRReader()
        text = reader.read_page(frame)  # returns str, never raises

    The EasyOCR reader is loaded lazily on first call to avoid slowing
    startup time, and then cached for subsequent calls.
    """

    def __init__(self) -> None:
        self._reader: Optional[object] = None
        self._load_error: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")

    def _get_reader(self) -> Optional[object]:
        """Lazy-load and return the EasyOCR reader, or None if unavailable."""
        if self._load_error is not None:
            return None
        if self._reader is not None:
            return self._reader

        try:
            import easyocr  # type: ignore[import-untyped]

            # Load English model; disable GPU for ARM64 embedded target
            self._reader = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=".models/easyocr",
                verbose=False,
            )
            logger.info(
                json.dumps({
                    "event": "ocr_reader_loaded",
                    "module": "OCRReader",
                    "status": "ok",
                })
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.error(
                json.dumps({
                    "event": "ocr_reader_load_failed",
                    "module": "OCRReader",
                    "error": str(exc),
                })
            )
            return None

        return self._reader

    def _run_ocr(self, frame: np.ndarray) -> str:
        """Run EasyOCR on *frame* and return concatenated text."""
        reader = self._get_reader()
        if reader is None:
            return ""

        # EasyOCR returns: list of (bbox, text, confidence)
        results = reader.readtext(frame, detail=1, paragraph=False)  # type: ignore[union-attr]

        if not results:
            return ""

        # Sort results top-to-bottom, then left-to-right (reading order)
        def _sort_key(item: tuple) -> tuple[float, float]:
            bbox = item[0]  # 4 corner points [[x,y], ...]
            top_y = min(pt[1] for pt in bbox)
            left_x = min(pt[0] for pt in bbox)
            return (top_y, left_x)

        results.sort(key=_sort_key)

        # Collect text from detections with reasonable confidence
        fragments: list[str] = []
        for _bbox, text, confidence in results:
            logger.debug(
                json.dumps({
                    "event": "ocr_detection",
                    "module": "OCRReader",
                    "text": text,
                    "confidence": round(float(confidence), 3),
                })
            )
            # Low-confidence detections are noise — skip below 0.3
            if confidence >= 0.3:
                fragments.append(text)

        return " ".join(fragments)

    def _run_ocr_detailed(self, frame: np.ndarray) -> tuple[str, float]:
        """Run EasyOCR and return (joined_text, mean_confidence) over accepted detections."""
        reader = self._get_reader()
        if reader is None:
            return "", 0.0

        results = reader.readtext(frame, detail=1, paragraph=False)  # type: ignore[union-attr]
        if not results:
            return "", 0.0

        def _sort_key(item: tuple) -> tuple[float, float]:
            bbox = item[0]
            return (min(pt[1] for pt in bbox), min(pt[0] for pt in bbox))

        results.sort(key=_sort_key)
        fragments: list[str] = []
        confidences: list[float] = []
        for _bbox, text, confidence in results:
            if confidence >= 0.3:
                fragments.append(text)
                confidences.append(float(confidence))

        if not fragments:
            return "", 0.0
        return " ".join(fragments), sum(confidences) / len(confidences)

    def read_region(self, frame: np.ndarray) -> tuple[str, float]:
        """OCR a (typically cropped) region and return ``(text, mean_confidence)``.

        Used by the Reading-mode OCR worker: it needs the confidence, not just the
        text, to gate narration (silence when unsure) and to decide orientation by
        the normal-vs-mirror confidence delta. Returns ``("", 0.0)`` when nothing
        readable is found. Never raises.
        """
        if _is_degenerate_frame(frame):
            return "", 0.0
        try:
            future = self._executor.submit(self._run_ocr_detailed, frame)
            text, confidence = future.result(timeout=_OCR_TIMEOUT_SECS)
        except FuturesTimeoutError:
            logger.warning(json.dumps({"event": "ocr_timeout", "module": "OCRReader"}))
            return "", 0.0
        except Exception as exc:  # noqa: BLE001
            logger.error(json.dumps({"event": "ocr_error", "module": "OCRReader", "error": str(exc)}))
            return "", 0.0

        return _normalize_whitespace(_strip_garbage(text)), confidence

    def read_page(self, frame: np.ndarray) -> str:
        """Extract and return all readable text from *frame*.

        Returns empty string (not None) if no text detected.
        Never raises — returns empty string on any error.

        Args:
            frame: BGR numpy array (H x W x 3). Accepts any height/width.

        Returns:
            Cleaned, normalised string of recognised text, or ``""`` if none found.
        """
        if _is_degenerate_frame(frame):
            logger.debug(
                json.dumps({
                    "event": "ocr_skipped",
                    "module": "OCRReader",
                    "reason": "degenerate_frame",
                })
            )
            return ""

        try:
            future = self._executor.submit(self._run_ocr, frame)
            raw_text: str = future.result(timeout=_OCR_TIMEOUT_SECS)
        except FuturesTimeoutError:
            logger.warning(
                json.dumps({
                    "event": "ocr_timeout",
                    "module": "OCRReader",
                    "timeout_secs": _OCR_TIMEOUT_SECS,
                })
            )
            return ""
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({
                    "event": "ocr_error",
                    "module": "OCRReader",
                    "error": str(exc),
                })
            )
            return ""

        cleaned = _strip_garbage(raw_text)
        normalised = _normalize_whitespace(cleaned)

        if normalised:
            logger.info(
                json.dumps({
                    "event": "ocr_result",
                    "module": "OCRReader",
                    "char_count": len(normalised),
                    "word_count": len(normalised.split()),
                })
            )

        return normalised
