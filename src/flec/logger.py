"""Structured JSON logger for Flec modules.

Every module emits structured JSON log events (Constitution Rule 2: Observability).
No silent failures — all detection events, state transitions, and errors are logged.

Usage:
    from flec.logger import log_event

    log_event(
        module="CameraModule",
        event_type="capture_started",
        data={"camera_index": 0},
    )

Log level is controlled by the FLEC_LOG_LEVEL environment variable (default: INFO).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Session ID — generated once per process, included in all log events
# ---------------------------------------------------------------------------

_SESSION_ID: str = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
            "session_id": _SESSION_ID,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Include any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in (
                "args", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message",
                "module", "msecs", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "taskName",
                "thread", "threadName",
            ):
                if not key.startswith("_"):
                    payload[key] = val

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure root logger with JSON formatting.

    Call once at application startup (main.py). Safe to call multiple times
    — subsequent calls are no-ops if already configured.
    """
    level_name = os.environ.get("FLEC_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Structured event helper
# ---------------------------------------------------------------------------


def log_event(
    module: str,
    event_type: str,
    data: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    """Emit a structured JSON log event.

    Args:
        module:     Name of the emitting module (e.g. "CameraModule").
        event_type: Machine-readable event type (e.g. "capture_started").
        data:       Optional dict of event-specific fields.
        level:      Python logging level (default: INFO).

    Example output::

        {
            "timestamp": "2026-07-07T10:00:00",
            "level": "INFO",
            "module": "CameraModule",
            "event_type": "capture_started",
            "data": {"camera_index": 0},
            "session_id": "abc-123"
        }
    """
    _logger = logging.getLogger(module)
    record = _logger.makeRecord(
        name=module,
        level=level,
        fn="",
        lno=0,
        msg=event_type,
        args=(),
        exc_info=None,
    )
    record.event_type = event_type
    record.data = data or {}
    record.session_id = _SESSION_ID

    # Emit as structured JSON via the root handler
    event_logger = logging.getLogger("flec.events")
    event_logger.log(
        level,
        json.dumps({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "module": module,
            "event_type": event_type,
            "data": data or {},
            "session_id": _SESSION_ID,
        }),
    )
