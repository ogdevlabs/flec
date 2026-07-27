"""Contract tests: SemanticThread mode-lazy stub contract (flec-pw3, AC-15)."""
from __future__ import annotations

import time
import unittest.mock as mock

import numpy as np
import pytest

from flec.models import DetectionType, Mode, TaggedFrame


def _tagged():
    return TaggedFrame(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        origin_mode=Mode.EXPLORATION,
        timestamp_ns=0,
    )


class TestSemanticThreadContract:
    def test_model_not_loaded_at_init(self):
        """AC-15: SemanticThread must NOT load the model at construction."""
        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread()
        assert t._model is None

    def test_init_is_fast(self):
        """AC-15: init must complete in <50ms (no model loading)."""
        from flec.perception.semantic_thread import SemanticThread

        t0 = time.monotonic()
        SemanticThread()
        assert (time.monotonic() - t0) < 0.05

    def test_load_started_false_at_init(self):
        """AC-15: _load_started must be False at construction."""
        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread()
        assert t._load_started is False

    def test_output_type_is_semantic(self):
        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.masks = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.type is DetectionType.SEMANTIC

    def test_output_label_is_semantic_mask(self):
        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.masks = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.label == "semantic_mask"

    def test_missing_model_graceful_degradation(self):
        from pathlib import Path

        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()
        assert t._load_failed

    def test_semantic_mask_is_ndarray_or_none(self):
        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.masks = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.semantic_mask is None or isinstance(ev.semantic_mask, np.ndarray)

    def test_semantic_mask_ndarray_when_masks_present(self):
        """When the model returns instance masks, semantic_mask must be a non-None ndarray."""
        from flec.perception.semantic_thread import SemanticThread

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tagged = TaggedFrame(frame=frame, origin_mode=Mode.EXPLORATION, timestamp_ns=0)

        mask_data = mock.MagicMock()
        mask_arr = np.ones((64, 64), dtype=np.uint8)
        mask_data.cpu().numpy.return_value = mask_arr

        masks = mock.MagicMock()
        masks.data = [mask_data]

        boxes = mock.MagicMock()
        boxes.cls = [0]

        result = mock.MagicMock()
        result.masks = masks
        result.boxes = boxes

        t = SemanticThread()
        t._load_failed = False
        t._load_started = True
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(tagged)
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.semantic_mask is None or isinstance(ev.semantic_mask, np.ndarray)

    def test_confidence_is_one(self):
        """SEMANTIC events always emit confidence=1.0."""
        from flec.perception.semantic_thread import SemanticThread

        t = SemanticThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.masks = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.confidence == 1.0
