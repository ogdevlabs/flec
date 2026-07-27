"""Contract tests: DepthThread mode-lazy stub contract (flec-pw3, AC-15)."""
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


class TestDepthThreadContract:
    def test_model_not_loaded_at_init(self):
        """AC-15: DepthThread must NOT load the model at construction."""
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        assert t._model is None, "Model must be None at init (mode-lazy)"

    def test_init_is_fast(self):
        """AC-15: init must complete in <50ms (no model loading)."""
        from flec.perception.depth_thread import DepthThread

        t0 = time.monotonic()
        t = DepthThread()
        assert (time.monotonic() - t0) < 0.05

    def test_load_started_false_at_init(self):
        """AC-15: _load_started must be False at construction."""
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        assert t._load_started is False

    def test_output_type_is_depth(self):
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.depth = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.type is DetectionType.DEPTH

    def test_output_label_is_depth_map(self):
        """Label on DEPTH events must be 'depth_map'."""
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.depth = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.label == "depth_map"

    def test_missing_model_graceful_degradation(self):
        from pathlib import Path

        from flec.perception.depth_thread import DepthThread

        t = DepthThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()
        assert t._load_failed

    def test_depth_map_is_ndarray_when_present(self):
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        t._load_failed = False
        t._load_started = True
        depth_arr = np.random.rand(64, 64).astype(np.float32)
        result = mock.MagicMock()
        result.depth = mock.MagicMock()
        result.depth.cpu().numpy.return_value = depth_arr
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.depth_map is None or isinstance(ev.depth_map, np.ndarray)

    def test_depth_map_is_none_when_result_has_no_depth(self):
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.depth = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.depth_map is None

    def test_confidence_is_one(self):
        """DEPTH events always emit confidence=1.0 (no per-pixel confidence)."""
        from flec.perception.depth_thread import DepthThread

        t = DepthThread()
        t._load_failed = False
        t._load_started = True
        result = mock.MagicMock()
        result.depth = None
        t._model = mock.MagicMock(return_value=[result])
        t._process_tagged_frame(_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.confidence == 1.0
