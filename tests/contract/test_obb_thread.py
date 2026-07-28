"""Contract tests: OBBThread output schema (flec-56p, AC-12, AC-19)."""
from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from flec.models import DetectionType, Mode, TaggedFrame


def _tagged(mode=Mode.READING):
    return TaggedFrame(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        origin_mode=mode,
        timestamp_ns=0,
    )


class TestOBBThreadOutputContract:
    def _stub_result(self, angle=15.0, conf=0.8, label="book"):
        obb = mock.MagicMock()
        obb.conf = [conf]
        obb.cls = [0]
        obb.xywhr = np.array([[100, 100, 50, 30, angle]])
        obbs = mock.MagicMock()
        obbs.__iter__ = mock.Mock(return_value=iter([obb]))
        result = mock.MagicMock()
        result.obb = obbs
        result.names = {0: label}
        return result

    def test_output_type_is_document(self):
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result()])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.type is DetectionType.DOCUMENT

    def test_obb_angle_is_float(self):
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(angle=20.0)])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert isinstance(ev.obb_angle, float)

    def test_obb_angle_value_preserved(self):
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(angle=30.0)])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert abs(ev.obb_angle - 30.0) < 0.01

    def test_90_degree_clamped_to_zero(self):
        """AC-19: 90° OBB angle must be clamped to 0° (portrait vs landscape ambiguity)."""
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(angle=90.0)])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert ev.obb_angle == 0.0

    def test_angle_near_90_also_clamped(self):
        """AC-19: angles within 5° of 90° must also be clamped to 0°."""
        from flec.perception.obb_thread import OBBThread

        for angle in (86.0, 87.5, 89.9, 90.1, 92.0, 94.9):
            t = OBBThread()
            t._load_failed = False
            t._model = mock.MagicMock(return_value=[self._stub_result(angle=angle)])
            t._process_tagged_frame(_tagged())
            ev = t.get_output_queue().get_nowait()
            assert ev.obb_angle == 0.0, f"angle={angle} expected clamped to 0.0, got {ev.obb_angle}"

    def test_origin_mode_matches_tagged_frame(self):
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result()])
        t._process_tagged_frame(_tagged(Mode.READING))
        ev = t.get_output_queue().get_nowait()
        assert ev.origin_mode is Mode.READING

    def test_confidence_in_range(self):
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(conf=0.65)])
        t._process_tagged_frame(_tagged())
        ev = t.get_output_queue().get_nowait()
        assert 0.0 <= ev.confidence <= 1.0

    def test_missing_model_no_output_no_crash(self):
        from pathlib import Path

        from flec.perception.obb_thread import OBBThread

        t = OBBThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()

    def test_load_failed_flag_set_on_missing_model(self):
        from pathlib import Path

        from flec.perception.obb_thread import OBBThread

        t = OBBThread(model_path=Path("/no/model.pt"))
        t._process_tagged_frame(_tagged())
        assert t._load_failed

    def test_confidence_gate_filters_low_confidence(self):
        """Detections below _OBB_CONFIDENCE (0.4) must not appear in output."""
        from flec.perception.obb_thread import OBBThread

        t = OBBThread()
        t._load_failed = False
        t._model = mock.MagicMock(return_value=[self._stub_result(conf=0.1)])
        t._process_tagged_frame(_tagged())
        assert t.get_output_queue().empty()
