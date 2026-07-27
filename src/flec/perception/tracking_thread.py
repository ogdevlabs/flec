"""TrackingThread — yolo26n model.track(persist=True) for stable object IDs.

Wraps ultralytics YOLO tracking with persist=True (BoTSORT) to maintain stable
track_id across consecutive frames. reset_tracking() clears internal tracker
state so IDs restart from scratch on mode switch or wear-detect.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from flec.main import CapabilityThread

logger = logging.getLogger(__name__)

_TRACK_CONFIDENCE = 0.5


class TrackingThread(CapabilityThread):
    """YOLO26n-based object tracking thread.

    Outputs DetectionEvents with track_id set (never None for tracked objects).
    On mode switch, reset_tracking() must be called to clear tracker state.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        super().__init__()
        self._model_path = model_path or Path(".models/yolo26n.pt")
        self._model = None
        self._load_failed = False

    def reset_tracking(self) -> None:
        """Drain input queue AND reset tracker internal state."""
        super().reset_tracking()  # drain input queue
        # Reset YOLO tracker state so IDs restart
        if self._model is not None:
            try:
                # ultralytics tracker state lives on the predictor
                if hasattr(self._model, "predictor") and self._model.predictor is not None:
                    if hasattr(self._model.predictor, "trackers"):
                        self._model.predictor.trackers = None
                    if hasattr(self._model.predictor, "tracker"):
                        self._model.predictor.tracker = None
                logger.info(json.dumps({"event": "tracking_thread_reset"}))
            except Exception as exc:
                logger.warning(json.dumps({"event": "tracking_reset_warning", "error": str(exc)}))

    def _process_tagged_frame(self, tagged_frame) -> None:
        if self._load_failed:
            return
        if self._model is None:
            self._load_model()
            if self._load_failed:
                return
        try:
            from flec.models import DetectionEvent, DetectionType
            results = self._model.track(
                tagged_frame.frame,
                persist=True,
                verbose=False,
                conf=_TRACK_CONFIDENCE,
            )
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < _TRACK_CONFIDENCE:
                        continue
                    cls_id = int(box.cls[0])
                    label = result.names.get(cls_id, str(cls_id))
                    # track_id: None from tracker means object not yet associated
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                    event = DetectionEvent(
                        type=DetectionType.OBJECT,
                        label=label,
                        confidence=conf,
                        track_id=track_id,
                        origin_mode=tagged_frame.origin_mode,
                    )
                    self._output_queue.put_nowait(event)
        except Exception as exc:
            logger.error(json.dumps({"event": "tracking_thread_error", "error": str(exc)}))

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._model_path))
            logger.info(json.dumps({"event": "tracking_thread_loaded", "model": str(self._model_path)}))
        except Exception as exc:
            self._load_failed = True
            logger.warning(json.dumps({"event": "tracking_model_unavailable", "error": str(exc)}))
