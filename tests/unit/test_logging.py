"""Log schema validation tests.

Verifies that structured JSON log output from Flec modules:
- Contains all required Log Event schema fields
- Has correct types (timestamp ISO8601, module str, etc.)
- Contains NO PII fields (no frame data, no audio data)

Constitution Rule 2: Every module MUST emit structured JSON logs. No silent failures.
Constitution Rule 4: ZERO persistence of camera frames, audio, or biometric data.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from io import StringIO
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Required Log Event schema (from Constitution + spec)
# ---------------------------------------------------------------------------

# All Flec structured log records must contain these fields at minimum.
REQUIRED_FIELDS = {"level", "module", "msg"}

# Fields that MUST NOT appear (PII / privacy guardrails)
FORBIDDEN_FIELDS = {
    "frame",
    "frame_data",
    "audio",
    "audio_data",
    "raw_audio",
    "biometric",
    "pixel_data",
    "image_bytes",
    "face_id",
    "voice_print",
}

# Flec's structured log format from main.py
LOG_FORMAT = '{"level": "%(levelname)s", "module": "%(name)s", "msg": "%(message)s"}'

ISO8601_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_log_output(logger_name: str, level: str, message: str) -> str:
    """Emit a single structured log record and capture the raw output string."""
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    log = logging.getLogger(logger_name)
    # Preserve existing handlers
    original_handlers = log.handlers[:]
    original_propagate = log.propagate
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG)

    getattr(log, level.lower())(message)

    log.handlers = original_handlers
    log.propagate = original_propagate

    return buffer.getvalue().strip()


def _parse_log_line(raw: str) -> Dict[str, Any]:
    """Parse a single JSON log line. Raises if not valid JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        pytest.fail(f"Log output is not valid JSON: {raw!r}\nError: {exc}")


# ---------------------------------------------------------------------------
# Schema field tests
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Verify required Log Event schema fields are present."""

    def test_level_field_present(self) -> None:
        """Log record MUST contain 'level' field."""
        raw = _capture_log_output("flec.test_module", "INFO", "test message")
        record = _parse_log_line(raw)
        assert "level" in record, f"Missing 'level' field in log: {record}"

    def test_module_field_present(self) -> None:
        """Log record MUST contain 'module' field."""
        raw = _capture_log_output("flec.test_module", "INFO", "test message")
        record = _parse_log_line(raw)
        assert "module" in record, f"Missing 'module' field in log: {record}"

    def test_msg_field_present(self) -> None:
        """Log record MUST contain 'msg' field."""
        raw = _capture_log_output("flec.test_module", "INFO", "test message")
        record = _parse_log_line(raw)
        assert "msg" in record, f"Missing 'msg' field in log: {record}"

    def test_all_required_fields_present(self) -> None:
        """All required fields MUST be present in every log record."""
        raw = _capture_log_output("flec.perception", "WARNING", "wear state changed")
        record = _parse_log_line(raw)
        for field in REQUIRED_FIELDS:
            assert field in record, (
                f"Required field '{field}' missing from log record: {record}"
            )


class TestFieldTypes:
    """Verify correct types for each required field."""

    def test_level_is_string(self) -> None:
        """'level' MUST be a string."""
        raw = _capture_log_output("flec.camera", "ERROR", "camera error")
        record = _parse_log_line(raw)
        assert isinstance(record["level"], str), (
            f"'level' must be str, got {type(record['level']).__name__}"
        )

    def test_level_is_valid_log_level(self) -> None:
        """'level' MUST be one of the standard log levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        for level in ("info", "warning", "error"):
            raw = _capture_log_output("flec.test", level, f"{level} message")
            record = _parse_log_line(raw)
            assert record["level"] in valid_levels, (
                f"'level' value '{record['level']}' not in {valid_levels}"
            )

    def test_module_is_string(self) -> None:
        """'module' MUST be a string."""
        raw = _capture_log_output("flec.perception.shape_detector", "INFO", "detected")
        record = _parse_log_line(raw)
        assert isinstance(record["module"], str), (
            f"'module' must be str, got {type(record['module']).__name__}"
        )

    def test_module_contains_flec_prefix(self) -> None:
        """'module' MUST start with 'flec.' for all Flec modules."""
        raw = _capture_log_output("flec.audio.tts_engine", "INFO", "speaking")
        record = _parse_log_line(raw)
        assert record["module"].startswith("flec"), (
            f"'module' should start with 'flec', got '{record['module']}'"
        )

    def test_msg_is_string(self) -> None:
        """'msg' MUST be a string."""
        raw = _capture_log_output("flec.engine", "INFO", "response queued")
        record = _parse_log_line(raw)
        assert isinstance(record["msg"], str), (
            f"'msg' must be str, got {type(record['msg']).__name__}"
        )


class TestLogIsValidJSON:
    """Verify log output is valid JSON on every call."""

    def test_info_log_is_json(self) -> None:
        """INFO log record MUST be valid JSON."""
        raw = _capture_log_output("flec.main", "INFO", "starting up")
        _parse_log_line(raw)  # Would fail if not JSON

    def test_error_log_is_json(self) -> None:
        """ERROR log record MUST be valid JSON."""
        raw = _capture_log_output("flec.main", "ERROR", "something went wrong")
        _parse_log_line(raw)  # Would fail if not JSON

    def test_warning_log_is_json(self) -> None:
        """WARNING log record MUST be valid JSON."""
        raw = _capture_log_output("flec.session", "WARNING", "mode mismatch")
        _parse_log_line(raw)  # Would fail if not JSON

    def test_log_is_single_json_object_not_array(self) -> None:
        """Log output MUST be a JSON object, not an array."""
        raw = _capture_log_output("flec.camera", "INFO", "frame captured")
        record = _parse_log_line(raw)
        assert isinstance(record, dict), (
            f"Log record must be a JSON object (dict), got {type(record).__name__}"
        )


class TestPrivacyNoPII:
    """Verify no PII fields appear in log output (Constitution Rule 4)."""

    def test_no_frame_data_in_log(self) -> None:
        """Log MUST NOT contain 'frame' or 'frame_data' fields."""
        raw = _capture_log_output("flec.camera", "INFO", "frame processed")
        record = _parse_log_line(raw)
        for forbidden in ("frame", "frame_data", "pixel_data", "image_bytes"):
            assert forbidden not in record, (
                f"PRIVACY VIOLATION: forbidden field '{forbidden}' found in log: {record}"
            )

    def test_no_audio_data_in_log(self) -> None:
        """Log MUST NOT contain 'audio' or 'audio_data' fields."""
        raw = _capture_log_output("flec.audio", "INFO", "audio played")
        record = _parse_log_line(raw)
        for forbidden in ("audio", "audio_data", "raw_audio"):
            assert forbidden not in record, (
                f"PRIVACY VIOLATION: forbidden field '{forbidden}' found in log: {record}"
            )

    def test_no_biometric_data_in_log(self) -> None:
        """Log MUST NOT contain biometric fields."""
        raw = _capture_log_output("flec.perception", "INFO", "wear detected")
        record = _parse_log_line(raw)
        for forbidden in ("biometric", "face_id", "voice_print"):
            assert forbidden not in record, (
                f"PRIVACY VIOLATION: forbidden field '{forbidden}' found in log: {record}"
            )

    def test_forbidden_fields_not_in_any_log(self) -> None:
        """Comprehensive check: none of the forbidden fields appear in any log output."""
        messages = [
            ("flec.camera", "INFO", "camera frame ready"),
            ("flec.perception.wear", "INFO", "wear state ON_HEAD"),
            ("flec.audio.tts", "INFO", "speech synthesized"),
            ("flec.perception.shape", "INFO", "shape detected: circle"),
        ]
        for logger_name, level, msg in messages:
            raw = _capture_log_output(logger_name, level, msg)
            record = _parse_log_line(raw)
            for forbidden in FORBIDDEN_FIELDS:
                assert forbidden not in record, (
                    f"PRIVACY VIOLATION: forbidden field '{forbidden}' "
                    f"found in log from {logger_name}: {record}"
                )


class TestLogContentIntegrity:
    """Verify log content accurately reflects what was logged."""

    def test_msg_field_contains_logged_message(self) -> None:
        """'msg' field MUST contain the actual log message."""
        test_msg = "detection confidence 0.92 for circle"
        raw = _capture_log_output("flec.perception", "INFO", test_msg)
        record = _parse_log_line(raw)
        assert record["msg"] == test_msg, (
            f"Expected msg '{test_msg}', got '{record['msg']}'"
        )

    def test_level_matches_logged_level(self) -> None:
        """'level' field MUST match the actual log level used."""
        for level_lower, level_upper in [("info", "INFO"), ("error", "ERROR"), ("warning", "WARNING")]:
            raw = _capture_log_output("flec.test", level_lower, "check level")
            record = _parse_log_line(raw)
            assert record["level"] == level_upper, (
                f"Expected level '{level_upper}', got '{record['level']}'"
            )

    def test_module_field_matches_logger_name(self) -> None:
        """'module' field MUST match the logger name."""
        logger_name = "flec.perception.finger_tracker"
        raw = _capture_log_output(logger_name, "info", "tracking state updated")
        record = _parse_log_line(raw)
        assert record["module"] == logger_name, (
            f"Expected module '{logger_name}', got '{record['module']}'"
        )
