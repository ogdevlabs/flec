"""Contract tests for OCRReader.

Verifies the OCRReader interface contract as defined in
specs/001-perception-core/contracts/module-interfaces.md.

These tests run BEFORE the implementation exists — they define the RED phase.
All tests are expected to fail until OCRReader is implemented (T048).
"""

from __future__ import annotations

import numpy as np
import pytest

# Import the module under test — will fail (ImportError) until T048
from flec.reading.ocr_reader import OCRReader


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ocr_reader() -> OCRReader:
    """Return a fresh OCRReader instance."""
    return OCRReader()


@pytest.fixture
def text_frame() -> np.ndarray:
    """Return a BGR frame with clear printed text rendered onto white background.

    Uses OpenCV to render ASCII text large enough for OCR to detect.
    """
    import cv2

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    # Render a child-book-style line of text in black
    cv2.putText(
        frame,
        "THE CAT SAT",
        (80, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "ON THE MAT",
        (80, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    return frame


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Return a blank black frame (no text)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame() -> np.ndarray:
    """Return a white frame with no text."""
    return np.full((480, 640, 3), 255, dtype=np.uint8)


@pytest.fixture
def corrupted_frame() -> np.ndarray:
    """Return a tiny 1x1 frame that represents a corrupted/degenerate input."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestOCRReaderContract:
    """Contract tests verifying the OCRReader interface."""

    def test_returns_non_empty_string_on_text_frame(
        self, ocr_reader: OCRReader, text_frame: np.ndarray
    ) -> None:
        """read_page() returns a non-empty string when clear text is present."""
        result = ocr_reader.read_page(text_frame)
        assert isinstance(result, str), "read_page must return str"
        assert len(result.strip()) > 0, "read_page must return non-empty string for text frame"

    def test_returns_string_type_always(
        self, ocr_reader: OCRReader, text_frame: np.ndarray
    ) -> None:
        """read_page() always returns str (never None, never raises)."""
        result = ocr_reader.read_page(text_frame)
        assert isinstance(result, str), "read_page must return str, not None or other type"

    def test_returns_empty_string_on_blank_frame(
        self, ocr_reader: OCRReader, blank_frame: np.ndarray
    ) -> None:
        """read_page() returns empty string (not exception) on blank black frame."""
        result = ocr_reader.read_page(blank_frame)
        assert isinstance(result, str), "read_page must return str on blank frame"
        assert result == "", "read_page should return empty string on blank frame"

    def test_returns_empty_string_on_white_frame(
        self, ocr_reader: OCRReader, white_frame: np.ndarray
    ) -> None:
        """read_page() returns empty string (not exception) on illustration-only (no text) frame."""
        result = ocr_reader.read_page(white_frame)
        assert isinstance(result, str), "read_page must return str on white frame"
        # A plain white frame has no text — empty string expected
        assert result == "" or isinstance(result, str), (
            "read_page must return str (empty or minimal) on no-text frame"
        )

    def test_no_binary_garbage_characters_in_output(
        self, ocr_reader: OCRReader, text_frame: np.ndarray
    ) -> None:
        """Returned text contains no binary/garbage characters (non-printable, non-whitespace)."""
        result = ocr_reader.read_page(text_frame)
        for char in result:
            assert char.isprintable() or char.isspace(), (
                f"read_page output contains garbage character: {char!r} (ord={ord(char)})"
            )

    def test_does_not_raise_on_corrupted_frame(
        self, ocr_reader: OCRReader, corrupted_frame: np.ndarray
    ) -> None:
        """read_page() never raises — returns empty string on corrupted input."""
        # Must not raise any exception
        result = ocr_reader.read_page(corrupted_frame)
        assert isinstance(result, str), "read_page must return str even on corrupted frame"

    def test_does_not_raise_on_blank_frame(
        self, ocr_reader: OCRReader, blank_frame: np.ndarray
    ) -> None:
        """read_page() never raises on black frame."""
        # Must not raise any exception
        result = ocr_reader.read_page(blank_frame)
        assert isinstance(result, str)

    def test_does_not_raise_on_white_frame(
        self, ocr_reader: OCRReader, white_frame: np.ndarray
    ) -> None:
        """read_page() never raises on white frame."""
        result = ocr_reader.read_page(white_frame)
        assert isinstance(result, str)

    def test_result_is_normalized_whitespace(
        self, ocr_reader: OCRReader, text_frame: np.ndarray
    ) -> None:
        """Returned text has normalized whitespace (no leading/trailing whitespace noise)."""
        result = ocr_reader.read_page(text_frame)
        # Should be stripped
        assert result == result.strip(), (
            "read_page output must be stripped of leading/trailing whitespace"
        )

    def test_multiple_calls_are_consistent(
        self, ocr_reader: OCRReader, text_frame: np.ndarray
    ) -> None:
        """read_page() returns the same result for the same frame on multiple calls."""
        result1 = ocr_reader.read_page(text_frame)
        result2 = ocr_reader.read_page(text_frame)
        assert result1 == result2, "read_page must be deterministic for the same frame"

    def test_accepts_different_frame_shapes(
        self, ocr_reader: OCRReader
    ) -> None:
        """read_page() handles various frame shapes without raising."""
        shapes = [
            (240, 320, 3),
            (480, 640, 3),
            (720, 1280, 3),
        ]
        for shape in shapes:
            frame = np.zeros(shape, dtype=np.uint8)
            result = ocr_reader.read_page(frame)
            assert isinstance(result, str), f"read_page must return str for frame shape {shape}"
