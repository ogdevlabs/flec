"""SegmentationThread — CapabilityThread wrapping YOLOv8n-seg inference.

Lazy-loads the segmentation model on the first frame; degrades gracefully
when the model file or ultralytics are unavailable.  Results are posted to
the output queue as DetectionEvent(type=SHAPE) with an optional boolean mask.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from flec.main import CapabilityThread

logger = logging.getLogger(__name__)

_SEG_CONFIDENCE = 0.5


class SegmentationThread(CapabilityThread):
    """Runs YOLOv8n-seg inference on a background thread.

    Construction is cheap — model loading is deferred to the first call of
    ``_process_tagged_frame`` so the session boot path stays fast.

    Each detected instance yields a ``DetectionEvent`` with:
    - ``type=DetectionType.SHAPE``
    - ``label`` — YOLO class name
    - ``confidence`` — box confidence
    - ``mask`` — boolean ``np.ndarray`` of the instance mask, or ``None``
      when the mask array is entirely zero (empty / no foreground pixels).
    - ``origin_mode`` — copied from the incoming ``TaggedFrame``
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__()
        self._model_path = model_path or Path(".models/yolo26n-seg.pt")
        self._model = None
        self._load_failed = False

    # ------------------------------------------------------------------
    # CapabilityThread implementation
    # ------------------------------------------------------------------

    def _process_tagged_frame(self, tagged_frame) -> None:
        if self._load_failed:
            return
        if self._model is None:
            self._load_model()
            if self._load_failed:
                return
        try:
            import numpy as np  # noqa: PLC0415
            from flec.models import DetectionEvent, DetectionType  # noqa: PLC0415

            results = self._model(tagged_frame.frame, verbose=False)
            for result in results:
                boxes = result.boxes
                masks = result.masks  # may be None
                if boxes is None:
                    continue
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf < _SEG_CONFIDENCE:
                        continue
                    cls_id = int(box.cls[0])
                    label = result.names.get(cls_id, str(cls_id))

                    mask: Optional[np.ndarray] = None
                    if masks is not None and i < len(masks.data):
                        mask_arr = masks.data[i].cpu().numpy().astype(bool)
                        mask = mask_arr if mask_arr.any() else None

                    event = DetectionEvent(
                        type=DetectionType.SHAPE,
                        label=label,
                        confidence=conf,
                        mask=mask,
                        origin_mode=tagged_frame.origin_mode,
                    )
                    self._output_queue.put_nowait(event)
        except Exception as exc:  # noqa: BLE001
            logger.error(json.dumps({"event": "seg_thread_error", "error": str(exc)}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            self._model = YOLO(str(self._model_path))
            logger.info(json.dumps({"event": "seg_thread_loaded", "model": str(self._model_path)}))
        except Exception as exc:  # noqa: BLE001
            self._load_failed = True
            logger.warning(json.dumps({"event": "seg_model_unavailable", "error": str(exc)}))
