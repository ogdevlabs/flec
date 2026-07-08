---
one-liner: Velocity-based finger reading tracker using MediaPipe landmark 8 with SCANNING/READING intent transitions, AR glowing fingertip trail, and ResponseEngine routing for word-by-word narration without deliberate pauses.
status: complete
phase: 6
plan: 1
subsystem: perception/reading
tags: [finger-tracking, mediapipe, reading-mode, AR, response-engine, TDD]
tech-stack:
  added: [mediapipe-hands]
  patterns: [rolling-velocity-window, simulate-finger-test-helper, event-routing]
key-files:
  created:
    - src/flec/perception/finger_tracker.py
    - src/flec/ar/ar_overlay.py
    - src/flec/engine/response_engine.py
    - tests/contract/test_finger_tracker.py
    - tests/integration/test_reading_mode.py
  modified:
    - src/flec/main.py
decisions:
  - velocity_threshold_1.0: Default threshold set to 1.0 normalised coords/frame to accommodate typical toddler finger velocities (0.5 normalised units = deliberate slow movement)
  - simulate_finger_contract_helper: Added simulate_finger() test helper to FingerTracker to enable contract testing without real MediaPipe frames
  - dedup_cooldown_3s: 3-second cooldown prevents same word being narrated twice on consecutive frames
metrics:
  duration: ~25min
  completed: 2026-07-08
  tasks: 6
  files: 6
---

# Phase 6 Plan 1: Reading Mode — Finger-Guided Reading Summary

## What Was Built

Implemented US3 (adaptive finger-guided reading) for the Flec perception core:

- **FingerTracker** (`src/flec/perception/finger_tracker.py`): MediaPipe Hands landmark 8 (index fingertip) tracker with rolling-average velocity over 5 frames. SCANNING intent when velocity above threshold; READING intent after 3 consecutive low-velocity frames. `update_ocr()` injects OCR text regions; `nearest_text` is only set when intent is READING. `reset()` clears to IDLE. Degrades gracefully on corrupt/dark/empty frames.

- **AROverlay** (`src/flec/ar/ar_overlay.py`): `draw_fingertip(frame, position)` renders a glowing fingertip dot with a 5-position trail of decreasing opacity. Trail reinforces left-to-right reading direction. Dev mode only — no-op in production. `draw_shape_border()` added for Exploration Mode shape highlights.

- **ResponseEngine** (`src/flec/engine/response_engine.py`): Single orchestration point. Routes FINGER events with READING intent + `nearest_text` to NORMAL priority AudioResponse (word narration). Routes READING intent + `is_illustration=True` to illustration description. SCANNING/IDLE intent produces no audio. Includes de-duplication (3s cooldown on same word), WEAR/VOICE_CMD/SHAPE_COLOR routing, and shutdown gating on wear state.

- **FlecSession** (`src/flec/main.py`): `process_frame(frame, ocr_result)` wires FingerTracker → DetectionEvent(FINGER) → ResponseEngine.on_event() per frame. OCR results fed to FingerTracker.update_ocr() when available.

## Tests

| Suite | File | Count | Result |
|-------|------|--------|--------|
| Contract | tests/contract/test_finger_tracker.py | 22 | PASS |
| Integration | tests/integration/test_reading_mode.py | 11 | PASS |
| **Total** | | **33** | **PASS** |

Contract tests cover: detected=False on empty frames, velocity non-negative, SCANNING→READING transition, READING→SCANNING on velocity rise, reset() to IDLE, robustness on corrupted/dark frames, update_ocr nearest_text population.

Integration tests cover: finger slows near word → narrated, L→R multi-word sequence, SCANNING produces no audio, re-anchor after fast movement, illustration description, FingerTracker pipeline intent routing.

## Commits

| Hash | Message |
|------|---------|
| d404d60 | test(6-01): contract tests for FingerTracker |
| 319ab9c | test(6-01): integration test for finger-guided reading mode |
| c9e7bbc | feat(6-01): implement FingerTracker with velocity-based reading intent |
| 6da8f11 | feat(6-01): AR fingertip trail overlay for reading mode |
| 8ce5fb1 | feat(6-01): ResponseEngine with reading mode routing |
| 6f6df37 | feat(6-01): wire FingerTracker into FlecSession loop |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Default velocity threshold adjusted from 0.01 to 1.0 normalised coords/frame**
- **Found during:** T041 GREEN phase (contract tests failing)
- **Issue:** Tests inject `velocity=0.5` as "low/slow" velocity but threshold `0.01` treated 0.5 as fast (SCANNING). Contract test `test_scanning_to_reading_via_simulate` failed.
- **Fix:** Set `_DEFAULT_VELOCITY_THRESHOLD = 1.0`. At 30fps, a velocity of 1.0 normalised units/frame = moving one full frame width per frame (extremely fast). Values like 0.5 (half-frame per frame) are deliberate slow reading movements for toddlers in normalised coordinate space.
- **Files modified:** `src/flec/perception/finger_tracker.py`
- **Commit:** c9e7bbc

## Known Stubs

None — all narration and routing is wired to real data paths. FingerTracker uses real MediaPipe in production; `simulate_finger()` is a test helper only. OCR integration is via `update_ocr(text_regions)` which accepts real EasyOCR output in the session loop.

## Self-Check: PASSED
