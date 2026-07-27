"""Unit tests for flec-dyv: DetectionThread + SegmentationThread."""
from __future__ import annotations
import queue
import numpy as np
import pytest
from flec.models import Mode, DetectionEvent, DetectionType


class TestDetectionThreadInterface:
    def test_is_capability_thread(self):
        from flec.main import CapabilityThread
        from flec.perception.detection_thread import DetectionThread
        assert issubclass(DetectionThread, CapabilityThread)

    def test_get_output_queue_returns_queue(self):
        from flec.perception.detection_thread import DetectionThread
        t = DetectionThread()
        assert isinstance(t.get_output_queue(), queue.Queue)

    def test_no_model_load_at_init(self):
        """Model must not be loaded at construction — lazy-load only."""
        from flec.perception.detection_thread import DetectionThread
        t = DetectionThread()
        assert t._detector is None

    def test_process_frame_with_no_model_gracefully_degrades(self):
        """When model unavailable, thread must not raise — just no-op."""
        from flec.perception.detection_thread import DetectionThread
        from flec.models import TaggedFrame
        t = DetectionThread(model_path=__import__("pathlib").Path("/nonexistent/model.pt"))
        t._load_failed = False
        tagged = TaggedFrame(frame=np.zeros((100, 100, 3), dtype=np.uint8),
                             origin_mode=Mode.EXPLORATION, timestamp_ns=0)
        t._process_tagged_frame(tagged)  # must not raise
        assert t.get_output_queue().empty()

    def test_origin_mode_stamped_on_output(self, monkeypatch):
        """Events emitted must carry origin_mode from the TaggedFrame."""
        from flec.perception.detection_thread import DetectionThread
        from flec.models import TaggedFrame, BoundingBox

        # Provide a stub detector
        stub_event = DetectionEvent(type=DetectionType.OBJECT, label="cup", confidence=0.9)

        class StubDetector:
            def detect(self, frame):
                return [stub_event]

        t = DetectionThread()
        t._detector = StubDetector()
        t._load_failed = False
        tagged = TaggedFrame(frame=np.zeros((100, 100, 3), dtype=np.uint8),
                             origin_mode=Mode.READING, timestamp_ns=42)
        t._process_tagged_frame(tagged)
        event = t.get_output_queue().get_nowait()
        assert event.origin_mode is Mode.READING


class TestSegmentationThreadInterface:
    def test_is_capability_thread(self):
        from flec.main import CapabilityThread
        from flec.perception.segmentation_thread import SegmentationThread
        assert issubclass(SegmentationThread, CapabilityThread)

    def test_no_model_load_at_init(self):
        from flec.perception.segmentation_thread import SegmentationThread
        t = SegmentationThread()
        assert t._model is None

    def test_graceful_degradation_on_missing_model(self):
        from flec.perception.segmentation_thread import SegmentationThread
        from flec.models import TaggedFrame
        t = SegmentationThread(model_path=__import__("pathlib").Path("/nonexistent.pt"))
        tagged = TaggedFrame(frame=np.zeros((100, 100, 3), dtype=np.uint8),
                             origin_mode=Mode.EXPLORATION, timestamp_ns=0)
        t._process_tagged_frame(tagged)  # must not raise
        assert t.get_output_queue().empty()

    def test_mask_none_on_empty_mask(self):
        """Empty masks (all zeros) must result in mask=None on the event."""
        from flec.perception.segmentation_thread import SegmentationThread
        from flec.models import TaggedFrame
        import unittest.mock as mock

        t = SegmentationThread()
        # Stub model returning a box with empty mask
        box_mock = mock.MagicMock()
        box_mock.conf = [0.9]
        box_mock.cls = [0]
        boxes_mock = mock.MagicMock()
        boxes_mock.__iter__ = mock.Mock(return_value=iter([box_mock]))
        boxes_mock.__len__ = mock.Mock(return_value=1)
        mask_data = mock.MagicMock()
        mask_data.cpu().numpy.return_value = np.zeros((100, 100), dtype=np.uint8)
        masks_mock = mock.MagicMock()
        masks_mock.data = [mask_data]
        result_mock = mock.MagicMock()
        result_mock.boxes = boxes_mock
        result_mock.masks = masks_mock
        result_mock.names = {0: "table"}
        t._model = mock.MagicMock(return_value=[result_mock])
        t._load_failed = False
        tagged = TaggedFrame(frame=np.zeros((100, 100, 3), dtype=np.uint8),
                             origin_mode=Mode.EXPLORATION, timestamp_ns=0)
        t._process_tagged_frame(tagged)
        if not t.get_output_queue().empty():
            event = t.get_output_queue().get_nowait()
            assert event.mask is None  # empty mask → None
