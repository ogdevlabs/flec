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


def _load_whisper():
    """Load the cached Whisper tiny model, or return None if unavailable.

    Degrades gracefully: any failure (missing package/model, no network for a
    first fetch) leaves voice transcription disabled rather than crashing.
    """
    try:
        import whisper  # heavy import; done once at boot

        model = whisper.load_model("tiny", download_root=".models/whisper-tiny")
        logger.info(json.dumps({"event": "whisper_loaded", "model": "tiny"}))
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning(json.dumps({"event": "whisper_unavailable", "error": str(exc)}))
        return None


class FlecSession:
    """Manages the per-frame perception loop for one dev session.

    Wires the queue-decoupled pipeline for dev use:
      - ShapeColorDetector → SHAPE/COLOR events (object identification).
      - FingerTracker       → FINGER events (Reading mode).
      - MicListener + CommandSTT → VoiceCommands, routed as VOICE_CMD events so
        the caregiver can switch modes on the fly by speaking the mode name.
      - ResponseEngine      → single output gatekeeper, driving a real TTSEngine.

    Voice commands are captured on the mic thread but *routed* on the main
    (frame) thread via ``drain_voice_commands()`` so ResponseEngine.on_event is
    only ever called from one thread.
    """

    def __init__(
        self, mode: str = "dev", tts_backend: str = "coqui", voice: bool = True
    ) -> None:
        self._run_mode = mode
        self._running = False

        # Lazy imports (keep top-level import cost low at startup).
        from flec.perception.finger_tracker import FingerTracker
        from flec.perception.shape_color_detector import ShapeColorDetector
        from flec.engine.response_engine import ResponseEngine
        from flec.audio.tts import TTSEngine
        from flec.models import Mode as FlecMode

        # Shared event queue (reserved for the queue-decoupled contract) and a
        # thread-safe hand-off queue for mic-captured voice commands.
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)
        self._voice_cmd_queue: queue.Queue = queue.Queue(maxsize=50)

        # Perception modules.
        self._finger_tracker = FingerTracker()
        self._shape_detector = ShapeColorDetector()

        # Real audio output (Coqui VITS → say → log, per backend availability).
        self._tts_engine = TTSEngine(backend=tts_backend)
        self._response_engine = ResponseEngine(tts=self._tts_engine)

        # Boot into Exploration so the mask narrates objects it sees right away.
        self._response_engine.set_mode(FlecMode.EXPLORATION)

        # Microphone front-end (optional). Only started if voice is requested
        # and a Whisper model + mic device are actually available.
        self._mic: Optional[object] = None
        if voice:
            self._start_mic()

        logger.info(json.dumps({
            "event": "flec_session_init",
            "run_mode": mode,
            "tts_backend": tts_backend,
            "voice": bool(self._mic),
        }))

    def _start_mic(self) -> None:
        from flec.speech.command_stt import CommandSTT
        from flec.speech.mic_listener import MicListener

        whisper_model = _load_whisper()
        if whisper_model is None:
            logger.warning(json.dumps({
                "event": "voice_disabled",
                "reason": "whisper_model_unavailable",
            }))
            return

        stt = CommandSTT(whisper_model=whisper_model)
        mic = MicListener(command_stt=stt, on_command=self._voice_cmd_queue.put)
        if mic.start():
            self._mic = mic
        else:
            logger.warning(json.dumps({
                "event": "voice_disabled", "reason": "mic_unavailable",
            }))

    def process_frame(self, frame: np.ndarray, ocr_result: Optional[list[str]] = None) -> None:
        """Process a single camera frame through the perception pipeline.

        Runs object detection (shapes/colors) and finger tracking, routing each
        resulting DetectionEvent through the ResponseEngine. Called per-frame by
        the camera capture loop (or tests).
        """
        from flec.models import DetectionEvent, DetectionType

        # 1. Object identification — shapes & colors (Exploration / Challenge).
        for event in self._shape_detector.detect(frame):
            self._response_engine.on_event(event)

        # 2. Finger tracking (Reading mode).
        state = self._finger_tracker.update(frame)

        if ocr_result is not None:
            self._finger_tracker.update_ocr(text_regions=ocr_result)
            state = self._finger_tracker.current_state

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
            self._response_engine.on_event(event)

    def drain_voice_commands(self) -> None:
        """Route any mic-captured VoiceCommands as VOICE_CMD events.

        Called on the main loop thread so on_event stays single-threaded.
        """
        from flec.models import DetectionEvent, DetectionType

        while not self._voice_cmd_queue.empty():
            try:
                cmd = self._voice_cmd_queue.get_nowait()
            except queue.Empty:
                break
            self._response_engine.on_event(
                DetectionEvent(
                    type=DetectionType.VOICE_CMD,
                    label=cmd.intent.name,
                    confidence=1.0,
                    metadata={"command": cmd},
                )
            )

    def drain_audio_queue(self) -> list:
        """Deprecated no-op: audio now plays via TTSEngine, not a queue.

        Retained so the dev loop's optional logging block stays valid.
        """
        return []

    def shutdown(self) -> None:
        """Stop the mic thread and TTS playback thread cleanly."""
        if self._mic is not None:
            self._mic.stop()  # type: ignore[attr-defined]
        self._tts_engine.shutdown()

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
    parser.add_argument(
        "--tts",
        choices=["coqui", "say", "off"],
        default="coqui",
        help="TTS backend: 'coqui' (offline VITS, production; needs espeak-ng), "
        "'say' (macOS dev fallback), or 'off' (log narration text only). "
        "'coqui' auto-falls back to say/log when unavailable.",
    )
    parser.add_argument(
        "--voice",
        dest="voice",
        action="store_true",
        default=True,
        help="Enable microphone voice commands / mode switching (default).",
    )
    parser.add_argument(
        "--no-voice",
        dest="voice",
        action="store_false",
        help="Disable the microphone listener (perception + audio out only).",
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

    session = FlecSession(mode=args.mode, tts_backend=args.tts, voice=args.voice)
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

            # Route any voice commands captured since the last frame (mode
            # switches, challenges) before processing perception.
            session.drain_voice_commands()

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
        session.shutdown()
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
