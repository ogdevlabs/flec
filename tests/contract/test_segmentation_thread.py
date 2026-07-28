"""Contract tests: SegmentationThread output schema (flec-dyv, AC-11, AC-17)."""
from __future__ import annotations

import queue
import unittest.mock as mock

import numpy as np
import pytest

from flec.models import DetectionType, Mode, TaggedFrame


def _tagged(mode=Mode.EXPLORATION):
    return TaggedFrame(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        origin_mode=mode,
        timestamp_ns=0,
    )


class TestSegmentationThreadOutputContract:
    def _stub_result(self, conf=0.8, label="table", mask_nonzero=True):
        box = mock.MagicMock()
        box.conf = [conf]
        box.cls = [0]
        boxes = mock.MagicMock()
        boxes.__iter__ = mock.Mock(return_value=iter([box]))
        mask_arr = (
            np.ones((64, 64), dtype=np.uint8)
            if mask_nonzero
            else np.zeros((64, 64), dtype=np.uint8)
        )
        mask_data = mock.MagicMock()
        mask_data.cpu().numpy.return_value = mask_arr
        masks = mock.MagicMock()
        masks.data = [mask_data]
        result = mock.MagicMock()
        result.boxes = boxes
        result.masks = masks
        result.names = {0: label}
        return result

    def test_output_event_has_detection_type_shape(self):
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result()])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.type is DetectionType.SHAPE

    def test_output_event_has_confidence_float(self):
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(conf=0.75)])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert isinstance(ev.confidence, float)
        assert 0.0 <= ev.confidence <= 1.0

    def test_mask_is_ndarray_or_none(self):
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(mask_nonzero=True)])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.mask is None or isinstance(ev.mask, np.ndarray)

    def test_empty_mask_becomes_none(self):
        """AC-17: empty (all-zero) mask must become None, not an ndarray."""
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(mask_nonzero=False)])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.mask is None

    def test_origin_mode_matches_tagged_frame(self):
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result()])
        t._process_tagged_frame(_tagged(Mode.READING))
        ev = t.get_output_queue().get_nowait()
        assert ev.origin_mode is Mode.READING

    def test_label_is_str(self):
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(label="chair")])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert isinstance(ev.label, str)
        assert ev.label == "chair"

    def test_missing_model_no_output_no_crash(self):
        from pathlib import Path

        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()

    def test_load_failed_flag_set_on_missing_model(self):
        from pathlib import Path

        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t._load_failed

    def test_confidence_gate_filters_low_confidence(self):
        """Detections below _SEG_CONFIDENCE (0.5) must not appear in output."""
        from flec.perception.segmentation_thread import SegmentationThread

        t = SegmentationThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(conf=0.1)])
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()
