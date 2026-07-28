"""Unit tests for F-002 models.py additions.

Tests are written in Given/When/Then style per the PDLC TDD protocol.
Red phase: all tests should fail before implementation.
"""

import time

import numpy as np
import pytest

from flec.models import (
    DetectionEvent,
    DetectionType,
    FingerTrackingState,
    Mode,
    TaggedFrame,
)


# ---------------------------------------------------------------------------
# TaggedFrame
# ---------------------------------------------------------------------------


class TestTaggedFrame:
    """Given a raw camera frame and dispatch-time metadata,
    When TaggedFrame is constructed,
    Then it wraps frame + origin_mode + timestamp_ns."""

    def test_tagged_frame_holds_all_fields(self):
        """TaggedFrame must expose frame, origin_mode, and timestamp_ns."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tf = TaggedFrame(frame=frame, origin_mode=Mode.READING, timestamp_ns=123456789)
        assert tf.frame is frame
        assert tf.origin_mode is Mode.READING
        assert tf.timestamp_ns == 123456789

    def test_tagged_frame_accepts_all_modes(self):
        """TaggedFrame must accept every Mode enum value."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for mode in Mode:
            tf = TaggedFrame(frame=frame, origin_mode=mode, timestamp_ns=0)
            assert tf.origin_mode is mode

    def test_tagged_frame_timestamp_ns_is_int(self):
        """timestamp_ns must be an integer (monotonic nanoseconds)."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        tf = TaggedFrame(frame=frame, origin_mode=Mode.EXPLORATION, timestamp_ns=time.monotonic_ns())
        assert isinstance(tf.timestamp_ns, int)


# ---------------------------------------------------------------------------
# DetectionEvent new optional fields — backward compatibility
# ---------------------------------------------------------------------------


class TestDetectionEventBackwardCompat:
    """Given existing callers that construct DetectionEvent with only the original fields,
    When the 6 new optional fields are added,
    Then existing construction must succeed unchanged and new fields default to None."""

    def test_existing_construction_still_works(self):
        """DetectionEvent constructed with only original fields must not raise."""
        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
        )
        assert event.type is DetectionType.SHAPE
        assert event.label == "circle"
        assert event.confidence == 0.9

    def test_origin_mode_defaults_to_none(self):
        """origin_mode must default to None for backward compatibility."""
        event = DetectionEvent(type=DetectionType.COLOR, label="red", confidence=0.8)
        assert event.origin_mode is None

    def test_mask_defaults_to_none(self):
        """mask must default to None."""
        event = DetectionEvent(type=DetectionType.SHAPE, label="square", confidence=0.7)
        assert event.mask is None

    def test_track_id_defaults_to_none(self):
        """track_id must default to None."""
        event = DetectionEvent(type=DetectionType.OBJECT, label="cup", confidence=0.85)
        assert event.track_id is None

    def test_obb_angle_defaults_to_none(self):
        """obb_angle must default to None."""
        event = DetectionEvent(type=DetectionType.SHAPE, label="book", confidence=0.9)
        assert event.obb_angle is None

    def test_depth_map_defaults_to_none(self):
        """depth_map must default to None."""
        event = DetectionEvent(type=DetectionType.SHAPE, label="box", confidence=0.6)
        assert event.depth_map is None

    def test_semantic_mask_defaults_to_none(self):
        """semantic_mask must default to None."""
        event = DetectionEvent(type=DetectionType.SHAPE, label="floor", confidence=0.5)
        assert event.semantic_mask is None


# ---------------------------------------------------------------------------
# DetectionEvent new optional fields — set and read
# ---------------------------------------------------------------------------


class TestDetectionEventNewFields:
    """Given a DetectionEvent created with the new optional fields populated,
    When the fields are read back,
    Then they must match the supplied values."""

    def test_origin_mode_can_be_set(self):
        """origin_mode can be set to any Mode value."""
        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="circle",
            confidence=0.9,
            origin_mode=Mode.EXPLORATION,
        )
        assert event.origin_mode is Mode.EXPLORATION

    def test_mask_can_be_set(self):
        """mask can be set to an ndarray."""
        mask = np.ones((480, 640), dtype=bool)
        event = DetectionEvent(
            type=DetectionType.SHAPE,
            label="cup",
            confidence=0.8,
            mask=mask,
        )
        assert event.mask is mask

    def test_track_id_can_be_set(self):
        """track_id can be set to a positive integer."""
        event = DetectionEvent(
            type=DetectionType.OBJECT,
            label="ball",
            confidence=0.95,
            track_id=42,
        )
        assert event.track_id == 42

    def test_obb_angle_can_be_set(self):
        """obb_angle can be set to a float."""
        event = DetectionEvent(
            type=DetectionType.DOCUMENT,
            label="book",
            confidence=0.9,
            obb_angle=15.5,
        )
        assert event.obb_angle == pytest.approx(15.5)

    def test_depth_map_can_be_set(self):
        """depth_map can be set to a float32 ndarray."""
        depth = np.random.rand(480, 640).astype(np.float32)
        event = DetectionEvent(
            type=DetectionType.DEPTH,
            label="depth_map",
            confidence=1.0,
            depth_map=depth,
        )
        assert event.depth_map is depth

    def test_semantic_mask_can_be_set(self):
        """semantic_mask can be set to an int ndarray."""
        sem = np.zeros((480, 640), dtype=np.int32)
        event = DetectionEvent(
            type=DetectionType.SEMANTIC,
            label="semantic_mask",
            confidence=1.0,
            semantic_mask=sem,
        )
        assert event.semantic_mask is sem


# ---------------------------------------------------------------------------
# New DetectionType values
# ---------------------------------------------------------------------------


class TestDetectionTypeNewValues:
    """Given the new DetectionType enum values added for F-002,
    When they are accessed,
    Then they must be valid DetectionType members."""

    def test_document_detection_type_exists(self):
        """DetectionType.DOCUMENT must be a valid enum member."""
        assert DetectionType.DOCUMENT in DetectionType

    def test_depth_detection_type_exists(self):
        """DetectionType.DEPTH must be a valid enum member."""
        assert DetectionType.DEPTH in DetectionType

    def test_semantic_detection_type_exists(self):
        """DetectionType.SEMANTIC must be a valid enum member."""
        assert DetectionType.SEMANTIC in DetectionType

    def test_new_types_are_distinct(self):
        """DOCUMENT, DEPTH, and SEMANTIC must be distinct from each other and from existing types."""
        new_types = {DetectionType.DOCUMENT, DetectionType.DEPTH, DetectionType.SEMANTIC}
        assert len(new_types) == 3
        # None of the new types should equal existing ones
        existing = {DetectionType.SHAPE, DetectionType.COLOR, DetectionType.OBJECT,
                    DetectionType.WEAR, DetectionType.FINGER, DetectionType.TEXT,
                    DetectionType.ILLUSTRATION, DetectionType.VOICE_CMD}
        assert new_types.isdisjoint(existing)


# ---------------------------------------------------------------------------
# FingerTrackingState — frozen (no changes)
# ---------------------------------------------------------------------------


class TestFingerTrackingStatePreserved:
    """Given the FingerTrackingState dataclass,
    When F-002 models changes are applied,
    Then all existing field names and types must be unchanged (AC-4)."""

    def test_all_original_fields_present(self):
        """All original FingerTrackingState fields must exist with correct types."""
        state = FingerTrackingState()
        assert hasattr(state, "detected") and isinstance(state.detected, bool)
        assert hasattr(state, "position_x") and isinstance(state.position_x, float)
        assert hasattr(state, "position_y") and isinstance(state.position_y, float)
        assert hasattr(state, "velocity") and isinstance(state.velocity, float)
        assert hasattr(state, "intent")
        assert hasattr(state, "nearest_text")

    def test_no_new_fields_added(self):
        """FingerTrackingState must have exactly the same fields as before F-002."""
        expected_fields = {"detected", "position_x", "position_y", "velocity", "intent", "nearest_text"}
        actual_fields = {f.name for f in FingerTrackingState.__dataclass_fields__.values()}
        assert actual_fields == expected_fields
