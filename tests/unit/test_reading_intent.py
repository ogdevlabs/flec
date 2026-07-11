"""Reading-intent tuning tests (F-001 flec-4yx).

A deliberate steady point must register READING; a fast sweep must stay SCANNING
(silent). Thresholds are instance-configurable so the live session can use
realistic values without changing the contract-test defaults.
"""

from __future__ import annotations

from flec.models import ReadingIntent
from flec.perception.finger_tracker import FingerTracker


def test_steady_point_reads_at_realistic_threshold():
    ft = FingerTracker(velocity_threshold=0.05, reading_frames=3)
    for _ in range(3):
        ft.simulate_finger((0.5, 0.5), velocity=0.01)  # deliberate hold
    assert ft.current_state.intent == ReadingIntent.READING


def test_fast_sweep_stays_scanning_at_realistic_threshold():
    ft = FingerTracker(velocity_threshold=0.05, reading_frames=3)
    for _ in range(4):
        ft.simulate_finger((0.5, 0.5), velocity=0.3)  # browsing sweep
    assert ft.current_state.intent == ReadingIntent.SCANNING


def test_reading_frames_is_configurable():
    ft = FingerTracker(velocity_threshold=0.05, reading_frames=2)
    ft.simulate_finger((0.5, 0.5), velocity=0.01)
    ft.simulate_finger((0.5, 0.5), velocity=0.01)
    assert ft.current_state.intent == ReadingIntent.READING


def test_default_threshold_unchanged_for_contract_compat():
    ft = FingerTracker()
    assert ft.velocity_threshold == 1.0  # contract-test default preserved
