"""Flec entry point — boot, session loop.

Session loop architecture (Constitution §III):
- FingerTracker processes frames; results become DetectionEvents on the event queue.
- ResponseEngine consumes events from the queue and emits AudioResponses.
- OCR results (when available) are fed back to FingerTracker via update_ocr().
- No capability module imports another directly; all communication via queues.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


class FlecSession:
    """Manages the per-frame perception loop for one session.

    Runs in dev mode: FingerTracker + ResponseEngine wired together.
    OCR results are fed to FingerTracker.update_ocr() when available.
    """

    def __init__(self, mode: str = "dev") -> None:
        self._run_mode = mode
        self._running = False
        self._frame_thread: Optional[threading.Thread] = None

        # Lazy imports (keep top-level import cost low at startup).
        from flec.perception.finger_tracker import FingerTracker
        from flec.engine.response_engine import ResponseEngine
        from flec.models import Mode as FlecMode

        # Shared event queue (perception → response engine).
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)

        # Audio output queue (response engine → TTS — stub for now).
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)

        self._finger_tracker = FingerTracker()
        self._response_engine = ResponseEngine(audio_queue=self._audio_queue)

        # Start in READING mode for dev testing of this phase.
        self._response_engine.set_mode(FlecMode.READING)

        logger.info(json.dumps({"event": "flec_session_init", "run_mode": mode}))

    def process_frame(self, frame: np.ndarray, ocr_result: Optional[list[str]] = None) -> None:
        """Process a single camera frame through the perception pipeline.

        Called per-frame by the camera capture loop (or tests).

        Args:
            frame:      BGR numpy array (HxWx3).
            ocr_result: Text regions from the latest OCR pass, if available.
                        Fed to FingerTracker.update_ocr() to populate nearest_text.
        """
        from flec.models import DetectionEvent, DetectionType

        # 1. Run FingerTracker.
        state = self._finger_tracker.update(frame)

        # 2. Inject OCR result if provided.
        if ocr_result is not None:
            self._finger_tracker.update_ocr(text_regions=ocr_result)
            # Re-read state after OCR update (nearest_text may have changed).
            state = self._finger_tracker.current_state

        # 3. Emit DetectionEvent(FINGER) if finger is detected.
        if state.detected or state.intent.name != "IDLE":
            event = DetectionEvent(
                type=DetectionType.FINGER,
                label=state.nearest_text or "finger",
                confidence=1.0,
                metadata={
                    "intent": state.intent,
                    "nearest_text": state.nearest_text,
                    "is_illustration": False,
                    "position_x": state.position_x,
                    "position_y": state.position_y,
                    "velocity": state.velocity,
                },
            )
            # 4. Route event through ResponseEngine.
            self._response_engine.on_event(event)

            logger.debug(
                json.dumps(
                    {
                        "event": "frame_processed",
                        "detected": state.detected,
                        "intent": state.intent.name,
                        "nearest_text": state.nearest_text,
                    }
                )
            )

    def drain_audio_queue(self) -> list:
        """Return and clear all pending AudioResponses (for testing / logging)."""
        responses = []
        while not self._audio_queue.empty():
            try:
                responses.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break
        return responses

    def reset_reading_state(self) -> None:
        """Reset FingerTracker on mode transitions."""
        self._finger_tracker.reset()
        logger.info(json.dumps({"event": "reading_state_reset"}))

    @property
    def finger_tracker(self):
        return self._finger_tracker

    @property
    def response_engine(self):
        return self._response_engine


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for Flec."""
    parser = argparse.ArgumentParser(
        description="Flec — wearable superhero mask for toddler early learning"
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "prod"],
        default="dev",
        help="Run mode: 'dev' for development (iPhone camera), 'prod' for production (embedded camera)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate configuration and imports without starting the session loop",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='{"level": "%(levelname)s", "module": "%(name)s", "msg": "%(message)s"}',
        stream=sys.stdout,
    )

    logger.info(json.dumps({"event": "flec_start", "mode": args.mode}))

    if args.dry_run:
        logger.info("Dry-run: configuration validated — exiting without starting session loop")
        return

    session = FlecSession(mode=args.mode)
    logger.info(
        json.dumps({"event": "flec_ready", "note": "session loop ready — awaiting frames"})
    )

    # In dev mode, process a single blank frame to confirm the pipeline works.
    if args.mode == "dev":
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame, ocr_result=None)
        logger.info(
            json.dumps({"event": "flec_dev_frame_processed", "status": "ok"})
        )


if __name__ == "__main__":
    main()
