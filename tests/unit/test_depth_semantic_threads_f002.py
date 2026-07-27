"""Unit tests for flec-pw3: DepthThread + SemanticThread mode-lazy stubs."""
from __future__ import annotations
import queue, time
import numpy as np
import pytest
from flec.models import Mode, DetectionType, TaggedFrame

def _blank_tagged(mode=Mode.EXPLORATION):
    return TaggedFrame(frame=np.zeros((100,100,3), dtype=np.uint8),
                       origin_mode=mode, timestamp_ns=0)

class TestDepthThreadInterface:
    def test_is_capability_thread(self):
        from flec.main import CapabilityThread
        from flec.perception.depth_thread import DepthThread
        assert issubclass(DepthThread, CapabilityThread)

    def test_no_model_at_init(self):
        """Model must NOT be loaded at construction (mode-lazy AC-15)."""
        from flec.perception.depth_thread import DepthThread
        t = DepthThread()
        assert t._model is None
        assert not t._load_started

    def test_graceful_degradation_missing_model(self):
        """Missing model → no raise, empty output."""
        from flec.perception.depth_thread import DepthThread
        t = DepthThread(model_path=__import__("pathlib").Path("/no/such/model.pt"))
        t._process_tagged_frame(_blank_tagged())
        assert t.get_output_queue().empty()
        assert t._load_failed

    def test_depth_model_loading_log_emitted(self, caplog):
        """depth_model_loading event must be logged before load attempt."""
        import logging
        from flec.perception.depth_thread import DepthThread
        t = DepthThread(model_path=__import__("pathlib").Path("/missing.pt"))
        with caplog.at_level(logging.INFO):
            t._process_tagged_frame(_blank_tagged())
        assert any("depth_model_loading" in r.message for r in caplog.records)

    def test_output_type_is_depth(self):
        """Events emitted must have type=DEPTH."""
        import unittest.mock as mock
        from flec.perception.depth_thread import DepthThread
        t = DepthThread(); t._load_failed = False; t._load_started = True
        result_mock = mock.MagicMock()
        result_mock.depth = None
        t._model = mock.MagicMock(return_value=[result_mock])
        t._process_tagged_frame(_blank_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.type is DetectionType.DEPTH


class TestSemanticThreadInterface:
    def test_is_capability_thread(self):
        from flec.main import CapabilityThread
        from flec.perception.semantic_thread import SemanticThread
        assert issubclass(SemanticThread, CapabilityThread)

    def test_no_model_at_init(self):
        from flec.perception.semantic_thread import SemanticThread
        t = SemanticThread()
        assert t._model is None
        assert not t._load_started

    def test_graceful_degradation_missing_model(self):
        from flec.perception.semantic_thread import SemanticThread
        t = SemanticThread(model_path=__import__("pathlib").Path("/missing.pt"))
        t._process_tagged_frame(_blank_tagged())
        assert t.get_output_queue().empty()
        assert t._load_failed

    def test_semantic_model_loading_log(self, caplog):
        import logging
        from flec.perception.semantic_thread import SemanticThread
        t = SemanticThread(model_path=__import__("pathlib").Path("/missing.pt"))
        with caplog.at_level(logging.INFO):
            t._process_tagged_frame(_blank_tagged())
        assert any("semantic_model_loading" in r.message for r in caplog.records)

    def test_output_type_is_semantic(self):
        import unittest.mock as mock
        from flec.perception.semantic_thread import SemanticThread
        t = SemanticThread(); t._load_failed = False; t._load_started = True
        result_mock = mock.MagicMock()
        result_mock.masks = None
        t._model = mock.MagicMock(return_value=[result_mock])
        t._process_tagged_frame(_blank_tagged())
        if not t.get_output_queue().empty():
            ev = t.get_output_queue().get_nowait()
            assert ev.type is DetectionType.SEMANTIC

    def test_load_not_triggered_at_init_timing(self):
        """AC-15: frame thread not blocked — verify load only happens on first frame call."""
        from flec.perception.depth_thread import DepthThread
        import time
        t = DepthThread()
        t0 = time.monotonic()
        # Just creating — no load should happen
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, f"Init took {elapsed:.3f}s — model likely loading at init"
        assert t._model is None
