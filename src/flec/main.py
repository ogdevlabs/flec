"""Flec entry point — boot, session loop.

Startup sequence:
1. Parse CLI args
2. Configure structured JSON logging
3. Pre-warm all AI models (to meet <10s ready-state requirement)
4. Pre-cache critical audio WAVs
5. Play Iron Man-style boot audio
6. Start camera capture
7. Enter session loop (wear detection → mode routing → response engine)

Usage:
    python -m flec.main --mode dev
    python -m flec.main --mode dev --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boot sequence
# ---------------------------------------------------------------------------


def _configure_logging(log_level: str) -> None:
    """Configure structured JSON logging from FLEC_LOG_LEVEL or CLI arg."""
    # Allow env var to override CLI arg
    effective_level = os.environ.get("FLEC_LOG_LEVEL", log_level).upper()
    os.environ["FLEC_LOG_LEVEL"] = effective_level

    from flec.logger import configure_logging
    configure_logging()


def _prewarm_modules(dry_run: bool = False) -> dict:
    """Import and initialise all capability modules.

    Pre-warming at boot ensures all models are loaded into memory before
    the first detection event, meeting the <10s ready-state requirement (SC-003).

    Returns a dict of initialized module instances.
    """
    from flec.logger import log_event

    log_event(module="main", event_type="prewarm_start", data={"dry_run": dry_run})
    t0 = time.monotonic()

    modules = {}

    # CameraModule — always initialise (OpenCV is fast)
    from flec.camera.camera_module import CameraModule
    modules["camera"] = CameraModule()
    log_event(module="main", event_type="module_prewarmed", data={"module": "CameraModule"})

    # TTSEngine — initialise with mock in dry-run mode (no Coqui model load)
    from flec.audio.tts_engine import TTSEngine
    modules["tts"] = TTSEngine(use_mock=dry_run)
    log_event(module="main", event_type="module_prewarmed", data={"module": "TTSEngine"})

    # ResponseEngine — needs tts_engine
    from flec.engine.response_engine import ResponseEngine
    modules["response"] = ResponseEngine(tts_engine=modules["tts"])
    log_event(module="main", event_type="module_prewarmed", data={"module": "ResponseEngine"})

    # Session state machine
    from flec.session import Session
    modules["session"] = Session()
    log_event(module="main", event_type="module_prewarmed", data={"module": "Session"})

    elapsed = time.monotonic() - t0
    log_event(
        module="main",
        event_type="prewarm_complete",
        data={"elapsed_s": round(elapsed, 3), "module_count": len(modules)},
    )

    if elapsed > 10.0:
        log_event(
            module="main",
            event_type="prewarm_slow",
            data={"elapsed_s": round(elapsed, 3), "threshold_s": 10.0},
        )

    return modules


def _preload_audio_cache(tts_engine, dry_run: bool = False) -> None:
    """Pre-render critical audio responses to WAV cache.

    In dry_run mode, registers the cache keys without actual synthesis.
    """
    from flec.audio.responses import CACHE_MANIFEST
    from flec.logger import log_event

    log_event(module="main", event_type="audio_cache_preload_start", data={})
    tts_engine.preload_cache(CACHE_MANIFEST)
    log_event(
        module="main",
        event_type="audio_cache_preload_complete",
        data={"keys": list(CACHE_MANIFEST.keys())},
    )


def _play_boot_sequence(tts_engine, dry_run: bool = False) -> None:
    """Play the Iron Man-style startup sequence."""
    from flec.audio.responses import CacheKey
    from flec.models import AudioPriority, AudioResponse
    from flec.logger import log_event

    if dry_run:
        log_event(
            module="main",
            event_type="boot_audio_skipped",
            data={"reason": "dry_run"},
        )
        return

    boot_response = AudioResponse(
        text="Hero mask activated!",
        priority=AudioPriority.CRITICAL,
        pre_cached=True,
        cache_key=CacheKey.BOOT_READY,
    )
    tts_engine.speak(boot_response)
    log_event(module="main", event_type="boot_audio_queued", data={})


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def _session_loop(
    modules: dict,
    dry_run: bool = False,
    max_iterations: Optional[int] = None,
) -> None:
    """Main session loop.

    Polls the camera, runs wear detection, and routes events through the
    ResponseEngine. Runs until interrupted or max_iterations reached.

    In dry-run mode, simulates one iteration then exits cleanly.
    """
    from flec.logger import log_event
    from flec.models import DetectionEvent, DetectionType, WearState

    camera = modules["camera"]
    response_engine = modules["response"]
    session = modules["session"]

    # Start camera capture
    if not dry_run:
        camera.start()
        log_event(module="main", event_type="camera_started", data={})

    log_event(module="main", event_type="session_loop_start", data={"dry_run": dry_run})

    iteration = 0
    try:
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            if dry_run:
                # In dry-run, simulate a single "mask on" event and exit
                event = DetectionEvent(
                    type=DetectionType.WEAR,
                    label=WearState.ON_HEAD.name,
                    confidence=1.0,
                )
                response_engine.on_event(event)
                session.set_wear_state(WearState.ON_HEAD)
                log_event(
                    module="main",
                    event_type="dry_run_simulated_event",
                    data={"event_type": "WEAR", "label": "ON_HEAD"},
                )
                break

            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.033)  # ~30fps
                iteration += 1
                continue

            # Wear detection will be wired in Phase 3 (wear-detection phase).
            # For now, the session loop runs frame capture and routes to response engine.
            # This stub keeps the loop alive and the camera warm.
            time.sleep(0.033)
            iteration += 1

    except KeyboardInterrupt:
        log_event(module="main", event_type="session_interrupted", data={})
    finally:
        if not dry_run:
            camera.stop()
        log_event(
            module="main",
            event_type="session_loop_end",
            data={"iterations": iteration},
        )


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


def _setup_signal_handlers(modules: dict) -> None:
    """Register SIGINT/SIGTERM handlers for graceful shutdown."""
    def _shutdown(sig, frame):
        from flec.logger import log_event
        from flec.audio.responses import CacheKey
        from flec.models import AudioPriority, AudioResponse

        log_event(module="main", event_type="signal_received", data={"signal": sig})

        tts = modules.get("tts")
        if tts and not getattr(tts, "_use_mock", False):
            shutdown_response = AudioResponse(
                text="See you next time, hero!",
                priority=AudioPriority.CRITICAL,
                pre_cached=True,
                cache_key=CacheKey.SHUTDOWN,
            )
            tts.speak(shutdown_response)
            # Give audio a moment to start
            time.sleep(0.5)

        camera = modules.get("camera")
        if camera and camera.is_running:
            camera.stop()

        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


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
        help="Run mode: 'dev' (iPhone camera proxy) or 'prod' (embedded camera)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (overridden by FLEC_LOG_LEVEL env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialise all modules, play boot sequence, then exit immediately. "
             "Used for CI verification.",
    )
    args = parser.parse_args()

    # Step 1: Configure logging
    _configure_logging(args.log_level)

    from flec.logger import log_event
    log_event(
        module="main",
        event_type="startup",
        data={"mode": args.mode, "dry_run": args.dry_run},
    )

    # Step 2: Pre-warm all modules
    modules = _prewarm_modules(dry_run=args.dry_run)

    # Step 3: Pre-cache critical audio
    _preload_audio_cache(modules["tts"], dry_run=args.dry_run)

    # Step 4: Register signal handlers
    if not args.dry_run:
        _setup_signal_handlers(modules)

    # Step 5: Play boot sequence
    _play_boot_sequence(modules["tts"], dry_run=args.dry_run)

    log_event(module="main", event_type="boot_complete", data={"mode": args.mode})

    # Step 6: Enter session loop
    _session_loop(modules, dry_run=args.dry_run)

    log_event(module="main", event_type="shutdown", data={})


if __name__ == "__main__":
    main()
