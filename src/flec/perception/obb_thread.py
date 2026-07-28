"""OBBThread — yolo26n-obb oriented bounding box detection for document orientation."""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
from flec.main import CapabilityThread

logger = logging.getLogger(__name__)
_OBB_CONFIDENCE = 0.4
_DOCUMENT_CLASSES = {"book", "magazine", "card", "paper", "page", "document"}

class OBBThread(CapabilityThread):
    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__()
        self._model_path = model_path or Path(".models/yolo26n-obb.pt")
        self._model = None
        self._load_failed = False

    def _process_tagged_frame(self, tagged_frame) -> None:
        if self._load_failed:
            return
        if self._model is None:
            self._load_model()
            if self._load_failed:
                return
        try:
            from flec.models import DetectionEvent, DetectionType
            results = self._model(tagged_frame.frame, verbose=False)
            for result in results:
                if result.obb is None:
                    continue
                for i, obb in enumerate(result.obb):
                    conf = float(obb.conf[0])
                    if conf < _OBB_CONFIDENCE:
                        continue
                    cls_id = int(obb.cls[0])
                    label = result.names.get(cls_id, str(cls_id)).lower()
                    angle = float(obb.xywhr[0][4]) if obb.xywhr is not None else 0.0
                    # 90° ambiguity clamp: if angle ≈ 90°, clamp to 0° (prefer upright)
                    if abs(angle - 90.0) < 5.0:
                        logger.info(json.dumps({"event": "obb_angle_clamped", "raw_angle": angle}))
                        angle = 0.0
                    event = DetectionEvent(
                        type=DetectionType.DOCUMENT,
                        label=label,
                        confidence=conf,
                        obb_angle=angle,
                        origin_mode=tagged_frame.origin_mode,
                    )
                    self._output_queue.put_nowait(event)
        except Exception as exc:
            logger.error(json.dumps({"event": "obb_thread_error", "error": str(exc)}))

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            logger.info(json.dumps({"event": "obb_thread_loaded", "model": str(self._model_path)}))
        except Exception as exc:
            self._load_failed = True
            logger.warning(json.dumps({"event": "obb_model_unavailable", "error": str(exc)}))
