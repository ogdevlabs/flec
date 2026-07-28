"""Contract tests: TrackingThread output schema (flec-6hl, AC-13, AC-14)."""
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


class TestTrackingThreadOutputContract:
    def _stub_result(self, label="cup", conf=0.8, track_id=7):
        box = mock.MagicMock()
        box.conf = [conf]
        box.cls = [0]
        box.id = [track_id]
        boxes = mock.MagicMock()
        boxes.__iter__ = mock.Mock(return_value=iter([box]))
        result = mock.MagicMock()
        result.boxes = boxes
        result.names = {0: label}
        return result

    def test_output_type_is_object(self):
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        t._load_failed = False
        t._model = mock.MagicMock()
        t._model.track.return_value = [self._stub_result()]
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.type is DetectionType.OBJECT

    def test_track_id_is_int_when_present(self):
        """AC-13: track_id must be int when the tracker assigns one."""
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        t._load_failed = False
        t._model = mock.MagicMock()
        t._model.track.return_value = [self._stub_result(track_id=42)]
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        if ev.track_id is not None:
            assert isinstance(ev.track_id, int)

    def test_track_id_matches_stub_value(self):
        """AC-13: emitted track_id must equal the value the tracker returned."""
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        t._load_failed = False
        t._model = mock.MagicMock()
        t._model.track.return_value = [self._stub_result(track_id=99)]
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.track_id == 99

    def test_track_id_is_none_when_box_id_is_none(self):
        """AC-13: track_id must be None when the tracker hasn't assigned an ID yet."""
        from flec.perception.tracking_thread import TrackingThread

        box = mock.MagicMock()
        box.conf = [0.8]
        box.cls = [0]
        box.id = None
        boxes = mock.MagicMock()
        boxes.__iter__ = mock.Mock(return_value=iter([box]))
        result = mock.MagicMock()
        result.boxes = boxes
        result.names = {0: "cup"}

        t = TrackingThread()
        t._load_failed = False
        t._model = mock.MagicMock()
        t._model.track.return_value = [result]
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.track_id is None

    def test_confidence_in_range(self):
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        t._load_failed = False
        t._model = mock.MagicMock()
        t._model.track.return_value = [self._stub_result(conf=0.9)]
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert 0.0 <= ev.confidence <= 1.0

    def test_origin_mode_matches_tagged_frame(self):
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        t._load_failed = False
        t._model = mock.MagicMock()
        t._model.track.return_value = [self._stub_result()]
        t._process_tagged_frame(_tagged(Mode.CHALLENGE))
        ev = t.get_output_queue().get_nowait()
        assert ev.origin_mode is Mode.CHALLENGE

    def test_reset_tracking_drains_input_queue(self):
        """AC-14: reset_tracking() must drain the input queue."""
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        for _ in range(3):
            try:
                t._input_queue.put_nowait(_tagged())
            except queue.Full:
                break
        t.reset_tracking()
        assert t._input_queue.empty()

    def test_reset_tracking_does_not_crash_without_model(self):
        """AC-14: reset_tracking() must not crash when no model is loaded."""
        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread()
        assert t._model is None
        t.reset_tracking()  # must not raise

    def test_missing_model_no_output_no_crash(self):
        from pathlib import Path

        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()

    def test_load_failed_flag_set_on_missing_model(self):
        from pathlib import Path

        from flec.perception.tracking_thread import TrackingThread

        t = TrackingThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t._load_failed
