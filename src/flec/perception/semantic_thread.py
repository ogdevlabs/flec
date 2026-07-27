"""SemanticThread — yolo26n-seg semantic segmentation (mode-lazy stub).

Model loads on first frame, not at boot. Outputs DetectionEvent(type=SEMANTIC, semantic_mask=np.ndarray|None).
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
from flec.main import CapabilityThread

logger = logging.getLogger(__name__)


class SemanticThread(CapabilityThread):
    """Mode-lazy semantic segmentation thread."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__()
        self._model_path = model_path or Path(".models/yolo26n-seg.pt")
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
            return
        try:
            from flec.models import DetectionEvent, DetectionType
            import numpy as np
            results = self._model(tagged_frame.frame, verbose=False)
            for result in results:
                semantic_mask = None
                if hasattr(result, "masks") and result.masks is not None:
                    # Combine all instance masks into a single semantic mask (class-id per pixel)
                    if result.masks.data is not None and len(result.masks.data) > 0:
                        combined = np.zeros(tagged_frame.frame.shape[:2], dtype=np.int32)
                        for i, mask_data in enumerate(result.masks.data):
                            cls_id = int(result.boxes.cls[i]) if result.boxes is not None else 0
                            mask_arr = mask_data.cpu().numpy().astype(bool)
                            combined[mask_arr] = cls_id + 1  # 0 = background
                        semantic_mask = combined if combined.any() else None
                event = DetectionEvent(
                    type=DetectionType.SEMANTIC,
                    label="semantic_mask",
                    confidence=1.0,
                    semantic_mask=semantic_mask,
                    origin_mode=tagged_frame.origin_mode,
                )
                self._output_queue.put_nowait(event)
        except Exception as exc:
            logger.error(json.dumps({"event": "semantic_thread_error", "error": str(exc)}))

    def _load_model(self) -> None:
        logger.info(json.dumps({"event": "semantic_model_loading", "model": str(self._model_path)}))
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            logger.info(json.dumps({"event": "semantic_model_ready", "model": str(self._model_path)}))
        except Exception as exc:
            self._load_failed = True
            logger.warning(json.dumps({"event": "semantic_model_unavailable", "error": str(exc)}))
