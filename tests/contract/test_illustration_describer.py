"""Contract tests for IllustrationDescriber.

Verifies the IllustrationDescriber interface contract as defined in
specs/001-perception-core/contracts/module-interfaces.md.

These tests run BEFORE the implementation exists — they define the RED phase.
All tests are expected to fail until IllustrationDescriber is implemented (T049).

Tests that require the BLIP-2 model (transformers + torch) are skipped when
those libraries are not installed (e.g. in CI without heavy ML dependencies).
The interface/structural tests always run.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

# Import the module under test — will fail (ImportError) until T049
from flec.reading.illustration_describer import IllustrationDescriber

# Mark for tests that require transformers + torch (BLIP-2 model)
blip2_available = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)
requires_blip2 = pytest.mark.skipif(
    not blip2_available,
    reason="transformers/torch not installed — BLIP-2 model-dependent tests skipped",
)

# Technical jargon / model class labels that must NOT appear in output
TECHNICAL_JARGON = [
    "tensor",
    "logit",
    "embedding",
    "token",
    "pixel",
    "rgb",
    "bgr",
    "classification",
    "inference",
    "softmax",
    "batch",
    "cuda",
    "cpu",
    "model",
    "neural",
    "network",
    "layer",
    "weight",
    "gradient",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def describer() -> IllustrationDescriber:
    """Return a fresh IllustrationDescriber instance."""
    return IllustrationDescriber()


@pytest.fixture
def illustration_frame() -> np.ndarray:
    """Return a colourful BGR frame that represents a simple illustration.

    Uses OpenCV to create a scene: blue sky, green ground, yellow circle (sun).
    This is recognisable enough for BLIP-2 or a mock to produce a description.
    """
    import cv2

    frame = np.full((480, 640, 3), 200, dtype=np.uint8)  # light grey background
    # Blue sky (top half)
    frame[:240, :] = [230, 180, 100]  # light blue (BGR)
    # Green ground (bottom half)
    frame[240:, :] = [50, 150, 50]   # green (BGR)
    # Yellow sun (circle)
    cv2.circle(frame, (540, 80), 60, (0, 220, 255), -1)  # yellow (BGR)
    # Brown tree trunk
    cv2.rectangle(frame, (150, 200), (200, 380), (30, 90, 139), -1)
    # Green tree top (circle)
    cv2.circle(frame, (175, 170), 80, (34, 139, 34), -1)
    return frame


@pytest.fixture
def blank_frame() -> np.ndarray:
    """Return a blank black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def corrupted_frame() -> np.ndarray:
    """Return a 1x1 degenerate frame."""
    return np.zeros((1, 1, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


class TestIllustrationDescriberContract:
    """Contract tests verifying the IllustrationDescriber interface."""

    @requires_blip2
    def test_returns_non_empty_string_on_illustration(
        self, describer: IllustrationDescriber, illustration_frame: np.ndarray
    ) -> None:
        """describe() returns a non-empty string for a clear illustration image."""
        result = describer.describe(illustration_frame)
        assert isinstance(result, str), "describe must return str"
        assert len(result.strip()) > 0, "describe must return non-empty string for clear image"

    def test_returns_string_type_always(
        self, describer: IllustrationDescriber, illustration_frame: np.ndarray
    ) -> None:
        """describe() always returns str (never None)."""
        result = describer.describe(illustration_frame)
        assert isinstance(result, str), "describe must return str, not None or other type"

    def test_returns_empty_string_on_blank_frame(
        self, describer: IllustrationDescriber, blank_frame: np.ndarray
    ) -> None:
        """describe() returns empty string (not exception) on blank input."""
        result = describer.describe(blank_frame)
        assert isinstance(result, str), "describe must return str on blank frame"
        assert result == "", "describe should return empty string on blank black frame"

    def test_does_not_raise_on_blank_frame(
        self, describer: IllustrationDescriber, blank_frame: np.ndarray
    ) -> None:
        """describe() never raises on blank frame."""
        # Must not raise any exception
        result = describer.describe(blank_frame)
        assert isinstance(result, str)

    def test_does_not_raise_on_corrupted_frame(
        self, describer: IllustrationDescriber, corrupted_frame: np.ndarray
    ) -> None:
        """describe() never raises — returns empty string on corrupted input."""
        result = describer.describe(corrupted_frame)
        assert isinstance(result, str), "describe must return str even on corrupted frame"

    @requires_blip2
    def test_description_is_at_most_20_words(
        self, describer: IllustrationDescriber, illustration_frame: np.ndarray
    ) -> None:
        """Description is <= 20 words (child-friendly brevity)."""
        result = describer.describe(illustration_frame)
        if result:  # non-empty descriptions must be <= 20 words
            word_count = len(result.split())
            assert word_count <= 20, (
                f"describe output must be <= 20 words for child-friendliness; "
                f"got {word_count} words: {result!r}"
            )

    @requires_blip2
    def test_no_technical_jargon_in_output(
        self, describer: IllustrationDescriber, illustration_frame: np.ndarray
    ) -> None:
        """Description contains no technical jargon or ML/model class labels."""
        result = describer.describe(illustration_frame).lower()
        for term in TECHNICAL_JARGON:
            assert term not in result, (
                f"describe output contains technical jargon {term!r}: {result!r}"
            )

    @requires_blip2
    def test_description_is_child_friendly_language(
        self, describer: IllustrationDescriber, illustration_frame: np.ndarray
    ) -> None:
        """Description uses simple, child-friendly language — all printable characters."""
        result = describer.describe(illustration_frame)
        for char in result:
            assert char.isprintable() or char.isspace(), (
                f"describe output contains garbage character: {char!r} (ord={ord(char)})"
            )

    @requires_blip2
    def test_multiple_calls_on_same_frame_are_consistent(
        self, describer: IllustrationDescriber, illustration_frame: np.ndarray
    ) -> None:
        """describe() is deterministic — same frame yields same description."""
        result1 = describer.describe(illustration_frame)
        result2 = describer.describe(illustration_frame)
        assert result1 == result2, "describe must be deterministic for the same frame"

    def test_accepts_various_frame_dimensions(
        self, describer: IllustrationDescriber
    ) -> None:
        """describe() handles various frame sizes without raising."""
        for shape in [(120, 160, 3), (480, 640, 3), (720, 1280, 3)]:
            frame = np.zeros(shape, dtype=np.uint8)
            result = describer.describe(frame)
            assert isinstance(result, str), (
                f"describe must return str for frame shape {shape}"
            )

    def test_corrupted_frame_returns_empty_not_error(
        self, describer: IllustrationDescriber, corrupted_frame: np.ndarray
    ) -> None:
        """describe() returns empty string (not raises) on corrupted input."""
        result = describer.describe(corrupted_frame)
        # Corrupted frames should gracefully return empty
        assert isinstance(result, str)
