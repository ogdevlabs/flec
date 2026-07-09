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
import os
import queue
import sys
import threading
import time
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
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Open a live dev window showing the camera feed with AR overlay (dev only). "
        "Press 'q' or Esc in the window to quit.",
    )
    parser.add_argument(
        "--camera",
        choices=["integrated", "iphone"],
        default="integrated",
        help="Camera source: 'integrated' for the built-in webcam (index 0), "
        "'iphone' to auto-detect a Continuity Camera. Overridden by --camera-index "
        "or FLEC_CAMERA_INDEX.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Explicit OpenCV device index. Takes precedence over --camera. "
        "Run scripts/list_cameras.py to discover indices.",
    )
    args = parser.parse_args()

    # macOS: OpenCV opens the camera on a background thread and cannot show the
    # permission prompt from there. Skipping its in-thread auth request relies on
    # the terminal already holding Camera permission (Privacy & Security → Camera).
    os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

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

    device_index = _resolve_device_index(args.camera, args.camera_index)
    _run_session_loop(session, preview=args.preview, device_index=device_index)


def _resolve_device_index(camera: str, camera_index: Optional[int]) -> int:
    """Resolve the OpenCV device index from the chosen camera options.

    Precedence: FLEC_CAMERA_INDEX env > --camera-index > --camera preset.
    For 'iphone', probes indices 1..5 for the first Continuity Camera that opens,
    falling back to the built-in webcam (0) with a warning if none respond.
    """
    env_index = os.environ.get("FLEC_CAMERA_INDEX")
    if env_index is not None:
        idx = int(env_index)
        logger.info(json.dumps({"event": "camera_selected", "source": "env", "index": idx}))
        return idx

    if camera_index is not None:
        logger.info(
            json.dumps({"event": "camera_selected", "source": "flag", "index": camera_index})
        )
        return camera_index

    if camera == "integrated":
        logger.info(
            json.dumps({"event": "camera_selected", "source": "integrated", "index": 0})
        )
        return 0

    # camera == "iphone": probe for a Continuity Camera (usually index >= 1).
    try:
        import cv2

        for idx in range(1, 6):
            cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            cap.release()
            if opened:
                logger.info(
                    json.dumps({"event": "camera_selected", "source": "iphone", "index": idx})
                )
                return idx
    except Exception as exc:  # noqa: BLE001 — degrade to built-in webcam
        logger.warning(json.dumps({"event": "iphone_probe_error", "error": str(exc)}))

    logger.warning(
        json.dumps(
            {
                "event": "iphone_not_found",
                "note": "no Continuity Camera detected — falling back to integrated webcam (0). "
                "Wake/unlock the iPhone, enable Continuity Camera, then retry.",
            }
        )
    )
    return 0


def _run_session_loop(
    session: "FlecSession", preview: bool = False, device_index: int = 0
) -> None:
    """Continuously capture camera frames and drive the perception pipeline.

    Runs until interrupted (Ctrl+C / SIGINT, or 'q'/Esc in the preview window).
    Frames are captured in a background thread by CameraModule; this loop reads
    the latest frame, feeds it through the session, and drains any resulting
    AudioResponses.

    When ``preview`` is True, a live OpenCV window is shown with the finger-tip
    AR overlay and a status HUD. GUI calls (imshow/waitKey) run on the main
    thread, as macOS requires. Frames are never persisted (Constitution Rule 4).
    """
    from flec.camera.camera_module import CameraModule

    target_fps = float(os.environ.get("FLEC_TARGET_FPS", "30"))
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

    # Preview window setup (dev only). Degrades gracefully if cv2 GUI is absent.
    overlay = None
    cv2 = None
    window_name = "Flec — dev preview (press q to quit)"
    if preview:
        try:
            import cv2 as _cv2

            from flec.ar.ar_overlay import AROverlay

            cv2 = _cv2
            overlay = AROverlay(dev_mode=True)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            logger.info(json.dumps({"event": "flec_preview_enabled"}))
        except Exception as exc:  # noqa: BLE001 — degrade, never crash the session
            logger.warning(
                json.dumps({"event": "flec_preview_unavailable", "error": str(exc)})
            )
            cv2 = None
            overlay = None

    camera = CameraModule(device_index=device_index)
    camera.start()

    logger.info(
        json.dumps(
            {
                "event": "flec_loop_start",
                "device_index": device_index,
                "target_fps": target_fps,
                "preview": bool(cv2),
            }
        )
    )

    frame_count = 0
    try:
        while True:
            frame = camera.get_frame()

            # Camera thread died (e.g. device could not be opened) — stop cleanly.
            if frame is None and not camera.is_running:
                logger.error(
                    json.dumps(
                        {
                            "event": "flec_loop_abort",
                            "reason": "camera_unavailable",
                            "device_index": device_index,
                        }
                    )
                )
                break

            if frame is None:
                time.sleep(frame_interval or 0.03)
                continue

            session.process_frame(frame, ocr_result=None)
            frame_count += 1

            for response in session.drain_audio_queue():
                logger.info(
                    json.dumps(
                        {
                            "event": "audio_response",
                            "text": response.text,
                            "priority": response.priority.name,
                            "pre_cached": response.pre_cached,
                        }
                    )
                )

            # Render the live preview window with AR overlay + HUD.
            if cv2 is not None:
                display = _render_preview(frame, session, overlay, frame_count, cv2)
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or Esc
                    logger.info(
                        json.dumps({"event": "flec_shutdown", "reason": "preview_quit"})
                    )
                    break
            elif frame_interval:
                time.sleep(frame_interval)
    except KeyboardInterrupt:
        logger.info(json.dumps({"event": "flec_shutdown", "reason": "keyboard_interrupt"}))
    finally:
        camera.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
        logger.info(
            json.dumps({"event": "flec_loop_end", "frames_processed": frame_count})
        )


def _render_preview(frame, session, overlay, frame_count, cv2):
    """Return a copy of ``frame`` annotated with the fingertip overlay + HUD.

    Never mutates the source frame; all drawing is on an in-memory copy.
    """
    display = frame.copy()

    state = session.finger_tracker.current_state
    if overlay is not None and getattr(state, "detected", False):
        display = overlay.draw_fingertip(
            display, (state.position_x, state.position_y)
        )

    mode = session.response_engine.mode.name
    intent = getattr(getattr(state, "intent", None), "name", "IDLE")
    hud = f"mode={mode}  finger={'YES' if state.detected else 'no'}  intent={intent}  frame={frame_count}"
    cv2.putText(
        display, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        display, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1, cv2.LINE_AA
    )
    return display


if __name__ == "__main__":
    main()
