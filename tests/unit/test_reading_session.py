"""FlecSession Reading-mode wiring tests (F-001).

- Dev wear-state override so Reading activates without a wear sensor  [flec-d23]
"""

from __future__ import annotations

import os

from flec.main import FlecSession
from flec.models import WearState


def test_dev_session_treats_webcam_as_on_head():
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        assert session.response_engine.wear_state == WearState.ON_HEAD
    finally:
        session.shutdown()


def test_wear_override_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FLEC_READING_WEAR_OVERRIDE", "0")
    session = FlecSession(mode="dev", tts_backend="off", voice=False)
    try:
        assert session.response_engine.wear_state == WearState.OFF_HEAD
    finally:
        session.shutdown()
