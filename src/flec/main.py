"""Flec entry point — boot, session loop.

Story Mode integration (US4, Phase 7):
  The session loop calls ``session.process_frame_for_story_mode(page_text, has_illustration)``
  on each processed frame. This method transitions the session to STORY mode when a
  book-like layout is detected (dense OCR text + optional illustration) and back to
  EXPLORATION when the book leaves the frame.

  Full session loop wired in a later boot phase.
"""

import argparse
import logging
import queue
import sys

logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='{"level": "%(levelname)s", "module": "%(name)s", "msg": "%(message)s"}',
        stream=sys.stdout,
    )

    logger.info("Flec starting in %s mode", args.mode)

    # Initialise queues and session — story mode pipeline is ready.
    # Full session loop (camera capture → OCR → IllustrationDescriber → ResponseEngine)
    # is wired in the boot/session-loop phase.
    audio_q: queue.Queue = queue.Queue(maxsize=50)
    event_q: queue.Queue = queue.Queue(maxsize=200)

    from flec.session import FlecSession  # noqa: PLC0415
    _session = FlecSession(audio_queue=audio_q, event_queue=event_q)

    logger.info("Flec ready — Story Mode pipeline initialised (full boot in later phases)")


if __name__ == "__main__":
    main()
