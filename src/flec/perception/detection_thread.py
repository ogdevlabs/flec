"""DetectionThread — CapabilityThread wrapping ShapeColorDetector.

Lazy-loads YOLOv8n on the first frame; degrades gracefully when the model
or ultralytics are unavailable.  Results are stamped with the TaggedFrame's
origin_mode and posted to the output queue.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import Optional

from flec.main import CapabilityThread

logger = logging.getLogger(__name__)


class DetectionThread(CapabilityThread):
    """Runs ShapeColorDetector inference on a background thread.

    Construction is cheap — model loading is deferred to the first call of
    ``_process_tagged_frame`` so the session boot path stays fast.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__()
        self._model_path = model_path
        self._detector = None
        self._load_failed = False

    # ------------------------------------------------------------------
    # CapabilityThread implementation
    # ------------------------------------------------------------------

    def _process_tagged_frame(self, tagged_frame) -> None:
        if self._load_failed:
            return
        if self._detector is None:
            self._load_detector()
            if self._load_failed:
                return
        try:
            events = self._detector.detect(tagged_frame.frame)
            for event in events:
                stamped = replace(event, origin_mode=tagged_frame.origin_mode)
                self._output_queue.put_nowait(stamped)
        except Exception as exc:  # noqa: BLE001
            logger.error(json.dumps({"event": "det_thread_error", "error": str(exc)}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_detector(self) -> None:
        try:
            from flec.perception.shape_color_detector import ShapeColorDetector  # noqa: PLC0415

            model_path = self._model_path
            if model_path is None:
                env = os.environ.get("FLEC_YOLO_MODEL")
                model_path = Path(env) if env else Path(".models/yolo26n.pt")
            self._detector = ShapeColorDetector(model_path=model_path)
            logger.info(json.dumps({"event": "det_thread_loaded", "model": str(model_path)}))
        except Exception as exc:  # noqa: BLE001
            self._load_failed = True
            logger.warning(json.dumps({"event": "det_model_unavailable", "error": str(exc)}))
