"""DepthThread — yolo26n-depth monocular depth estimation (mode-lazy stub).

Model loads on first frame, not at boot — avoids blocking startup.
Outputs DetectionEvent(type=DEPTH, depth_map=np.ndarray|None).
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Optional
from flec.main import CapabilityThread

logger = logging.getLogger(__name__)


class DepthThread(CapabilityThread):
    """Mode-lazy depth estimation thread.

    Loads model on first _process_tagged_frame call. Gracefully no-ops
    if model file is absent.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__()
        self._model_path = model_path or Path(".models/yolo26n-depth.pt")
        self._model = None
        self._load_failed = False
        self._load_started = False

    def _process_tagged_frame(self, tagged_frame) -> None:
        if self._load_failed:
            return
        if self._model is None and not self._load_started:
            self._load_started = True
            self._load_model()
            if self._load_failed:
                return
        if self._model is None:
            return  # still loading (shouldn't happen — load is sync)
        try:
            from flec.models import DetectionEvent, DetectionType
            import numpy as np
            results = self._model(tagged_frame.frame, verbose=False)
            for result in results:
                depth_map = None
                if hasattr(result, "depth") and result.depth is not None:
                    depth_map = result.depth.cpu().numpy() if hasattr(result.depth, "cpu") else result.depth
                event = DetectionEvent(
                    type=DetectionType.DEPTH,
                    label="depth_map",
                    confidence=1.0,
                    depth_map=depth_map,
                    origin_mode=tagged_frame.origin_mode,
                )
                self._output_queue.put_nowait(event)
        except Exception as exc:
            logger.error(json.dumps({"event": "depth_thread_error", "error": str(exc)}))

    def _load_model(self) -> None:
        logger.info(json.dumps({"event": "depth_model_loading", "model": str(self._model_path)}))
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            logger.info(json.dumps({"event": "depth_model_ready", "model": str(self._model_path)}))
        except Exception as exc:
            self._load_failed = True
            logger.warning(json.dumps({"event": "depth_model_unavailable", "error": str(exc)}))
