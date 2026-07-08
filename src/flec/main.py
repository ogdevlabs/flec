"""Flec entry point — boot, session loop."""

import argparse
import logging
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

    logger.info("Flec starting in %s mode", args.mode)
    # Boot sequence and session loop will be implemented in subsequent phases.
    if args.dry_run:
        logger.info("Dry-run: configuration validated — exiting without starting session loop")
        return
    logger.info("Flec ready (stub — full boot in later phases)")


if __name__ == "__main__":
    main()
