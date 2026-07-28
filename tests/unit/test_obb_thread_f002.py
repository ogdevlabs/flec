"""Unit tests for flec-56p: OBBThread + OBB-angle OCR rotation."""
from __future__ import annotations
import queue, numpy as np
import unittest.mock as mock
from flec.models import Mode, DetectionType

class TestOBBThreadInterface:
    def test_is_capability_thread(self):
        from flec.main import CapabilityThread
        from flec.perception.obb_thread import OBBThread
        assert issubclass(OBBThread, CapabilityThread)

    def test_no_model_at_init(self):
        from flec.perception.obb_thread import OBBThread
        assert OBBThread()._model is None

    def test_graceful_degradation(self):
        from flec.perception.obb_thread import OBBThread
        from flec.models import TaggedFrame
        t = OBBThread(model_path=__import__("pathlib").Path("/nonexistent.pt"))
        tagged = TaggedFrame(frame=np.zeros((100,100,3),dtype=np.uint8), origin_mode=Mode.READING, timestamp_ns=0)
        t._process_tagged_frame(tagged)  # no raise
        assert t.get_output_queue().empty()

class TestOBBThreadOutput:
    def _make_stub(self, angle=15.0, conf=0.8, label="book"):
        obb = mock.MagicMock()
        obb.conf = [conf]; obb.cls = [0]
        import numpy as np
        obb.xywhr = np.array([[100, 100, 50, 30, angle]])
        obbs = mock.MagicMock()
        obbs.__iter__ = mock.Mock(return_value=iter([obb]))
        result = mock.MagicMock()
        result.obb = obbs; result.names = {0: label}
        return result

    def test_obb_angle_on_event(self):
        from flec.perception.obb_thread import OBBThread
        from flec.models import TaggedFrame
        t = OBBThread(); t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._make_stub(angle=20.0)])
        tagged = TaggedFrame(frame=np.zeros((100,100,3),dtype=np.uint8), origin_mode=Mode.EXPLORATION, timestamp_ns=0)
        t._process_tagged_frame(tagged)
        ev = t.get_output_queue().get_nowait()
        assert abs(ev.obb_angle - 20.0) < 0.01
        assert ev.type is DetectionType.DOCUMENT

    def test_90_degree_clamped_to_zero(self):
        from flec.perception.obb_thread import OBBThread
        from flec.models import TaggedFrame
        t = OBBThread(); t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._make_stub(angle=90.0)])
        tagged = TaggedFrame(frame=np.zeros((100,100,3),dtype=np.uint8), origin_mode=Mode.READING, timestamp_ns=0)
        t._process_tagged_frame(tagged)
        ev = t.get_output_queue().get_nowait()
        assert ev.obb_angle == 0.0, "90° must be clamped to 0°"

class TestRotateCrop:
    def test_zero_angle_returns_same(self):
        from flec.reading.ocr_worker import rotate_crop
        crop = np.zeros((50,50,3), dtype=np.uint8)
        result = rotate_crop(crop, 0.0)
        assert result is crop  # no copy when angle < 1°

    def test_nonzero_angle_returns_ndarray(self):
        from flec.reading.ocr_worker import rotate_crop
        crop = np.zeros((50,50,3), dtype=np.uint8)
        result = rotate_crop(crop, 15.0)
        assert isinstance(result, np.ndarray)

class TestResolveOrientationOBBAngle:
    def test_obb_angle_sets_cached_normal(self):
        """When obb_angle provided, should resolve without mirror probe."""
        from flec.reading.ocr_worker import resolve_orientation
        call_count = [0]
        def read_region(crop):
            call_count[0] += 1
            return ("hello", 0.9)
        crop = np.zeros((50,100,3), dtype=np.uint8)
        text, conf, orient = resolve_orientation(crop, read_region, obb_angle=15.0, conf_gate=0.4)
        assert text == "hello"
        assert call_count[0] == 1  # only one probe, not two
