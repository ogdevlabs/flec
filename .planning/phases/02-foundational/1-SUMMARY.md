---
one-liner: "Core infrastructure built: CameraModule (OpenCV capture), TTSEngine (priority audio + pre-cached WAVs), ResponseEngine (event routing state machine), Session (mode lifecycle), structured JSON logger, and main.py boot sequence with dry-run CI verification."
status: complete
phase: 2
plan: 1
subsystem: core-infrastructure
tags: [camera, tts, response-engine, session, logger, boot]
dependency_graph:
  requires: [phase/1-setup]
  provides: [camera-frames, audio-playback, event-routing, session-lifecycle, structured-logging, entry-point]
  affects: [phase/3-wear-detection, phase/4-exploration-mode, phase/5-challenge-mode]
tech_stack:
  added: [sounddevice, soundfile, TTS (Coqui VITS — optional)]
  patterns: [priority-queue-audio, background-thread-capture, in-memory-event-routing, test-first-TDD]
key_files:
  created:
    - src/flec/camera/camera_module.py
    - src/flec/audio/tts_engine.py
    - src/flec/audio/responses.py
    - src/flec/engine/response_engine.py
    - src/flec/session.py
    - src/flec/logger.py
    - src/flec/main.py
    - tests/contract/test_camera_module.py
    - tests/contract/test_tts_engine.py
    - tests/contract/test_response_engine.py
  modified: []
decisions:
  - "TTSEngine uses use_mock=True in tests to avoid Coqui model loading; guarded imports for TTS/sounddevice/soundfile allow the package to import cleanly without heavy deps installed"
  - "ResponseEngine requires tts_engine injection (not import) to satisfy modular AI principle (Constitution Rule 3)"
  - "T015 (logger) was implemented before T011 (camera) to unblock the import dependency; committed together as feat(2-01): implement-camera-module-and-logger"
  - "stop_current() only clears NORMAL and LOW priority items; HIGH and CRITICAL remain queued to preserve safety-critical audio"
metrics:
  duration_minutes: 60
  completed_date: "2026-07-08"
  tasks_completed: 10
  tests_written: 28
  tests_passing: 28
---

# Phase 2 Plan 1: Core Infrastructure Summary

## What Was Built

All core infrastructure modules required as the foundation for all subsequent perception phases:

**CameraModule** (`src/flec/camera/camera_module.py`)
- OpenCV VideoCapture with background capture thread
- `FLEC_CAMERA_INDEX` env var support (default 0)
- Thread-safe `get_frame()` returns BGR uint8 copy or None
- Structured JSON log on start/stop/error

**TTSEngine** (`src/flec/audio/tts_engine.py`)
- Coqui VITS synthesis with guarded optional import
- `preload_cache()` pre-renders text to in-memory audio data (zero synthesis delay for critical responses)
- Priority queue: CRITICAL (4) > HIGH (3) > NORMAL (2) > LOW (1)
- Background playback thread; `stop_current()` interrupts and drains NORMAL/LOW queue
- `use_mock=True` for test environments without model loading

**Pre-cached Audio Responses** (`src/flec/audio/responses.py`)
- `CacheKey` constants: BOOT_READY, MASK_OFF, SHUTDOWN, CELEBRATION, THINKING, etc.
- `CACHE_MANIFEST` dict for all responses pre-rendered at startup
- All text is kid-friendly, encouraging, audio-complete (Constitution Rule 5)

**ResponseEngine** (`src/flec/engine/response_engine.py`)
- State-aware event routing: WEAR, VOICE_CMD, SHAPE, COLOR, FINGER, TEXT, ILLUSTRATION
- WEAR(ON_HEAD) → EXPLORATION mode; WEAR(OFF_HEAD) → STANDBY + CRITICAL "put mask on" audio
- VOICE_CMD(SHUTDOWN) gated by wear state (ignored when OFF_HEAD per FR-001e)
- SHAPE/COLOR in EXPLORATION → narration AudioResponse
- SHAPE/COLOR in CHALLENGE → celebration (match) or encouraging (non-match) response
- AR overlay integration for shape highlights (enhancement only)
- All dispatch errors logged, never raised

**Session State Machine** (`src/flec/session.py`)
- `Session` dataclass: wear_state, mode, active_challenge, story_context
- Valid transition table for all mode pairs
- `set_wear_state()` auto-transitions mode (ON_HEAD→EXPLORATION, OFF_HEAD→STANDBY)
- Challenge lifecycle: start, complete, cancel, expire (30s timeout)
- Structured log on every transition

**Structured Logger** (`src/flec/logger.py`)
- `_JSONFormatter` for Python logging
- `log_event(module, event_type, data)` helper with session_id
- `configure_logging()` for application startup
- `FLEC_LOG_LEVEL` env var controls level

**Entry Point** (`src/flec/main.py`)
- `argparse`: `--mode dev|prod`, `--log-level`, `--dry-run`
- 6-step boot sequence: configure logging → pre-warm modules → pre-cache audio → register signals → play boot audio → session loop
- `--dry-run`: boots, pre-warms all modules, simulates WEAR(ON_HEAD), verifies EXPLORATION mode, exits cleanly (CI verification)
- SIGINT/SIGTERM handlers: graceful shutdown with farewell audio

## Tests

**28 contract tests written and passing:**

| Module | Tests | Status |
|--------|-------|--------|
| CameraModule | 7 | PASSED |
| TTSEngine | 9 | PASSED |
| ResponseEngine | 12 | PASSED |
| **Total** | **28** | **28/28 PASSED** |

All tests use mock backends (no real camera, no TTS model loading) and run in ~0.4s.

## Commits

| Hash | Message |
|------|---------|
| c107adf | test(2-01): contract-tests-camera-module |
| a56f44a | test(2-01): contract-tests-tts-engine |
| 869c801 | test(2-01): contract-tests-response-engine |
| 35108d3 | feat(2-01): implement-camera-module-and-logger |
| 693f36c | feat(2-01): implement-tts-engine |
| 0ad9c6a | feat(2-01): pre-cached-audio-responses |
| d1f6fd9 | feat(2-01): implement-response-engine |
| 7e01ed1 | feat(2-01): session-state-machine |
| 6bc675d | feat(2-01): entry-point-boot-sequence |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] T015 (logger) implemented before T011 (camera)**
- **Found during:** T011 implementation
- **Issue:** CameraModule imports `flec.logger.log_event` but T015 (logger) was planned after T011. The import would fail at module load time.
- **Fix:** Implemented logger.py first, committed both together as a single implementation commit.
- **Files modified:** `src/flec/logger.py`, `src/flec/camera/camera_module.py`
- **Commit:** 35108d3

**2. [Rule 2 - Missing Critical] Added introspection helpers to TTSEngine**
- **Found during:** T009 (test writing)
- **Issue:** Contract tests need to introspect engine state to verify priority ordering and queue clearing. The interface contract specifies behavior but tests require observable state.
- **Fix:** Added `has_cached()`, `peek_highest_priority()`, `queue_size()`, `_set_speaking()` to TTSEngine — used exclusively by tests.
- **Files modified:** `src/flec/audio/tts_engine.py`
- **Commit:** 693f36c

## Known Stubs

- `_session_loop()` in `main.py`: The wear detection integration (frame → WearDetector → event) is stubbed with a comment. The loop correctly starts the camera and accepts frames but does not yet call WearDetector (Phase 3).
- `_handle_finger()` in `response_engine.py`: Pass-through stub — full finger tracking wiring is Phase 6 (reading-mode).

These stubs are intentional and do not block Phase 2's goal. Each is resolved in the phase that implements the corresponding capability.

## Self-Check: PASSED

Files created:
- `src/flec/camera/camera_module.py` — FOUND
- `src/flec/audio/tts_engine.py` — FOUND
- `src/flec/audio/responses.py` — FOUND
- `src/flec/engine/response_engine.py` — FOUND
- `src/flec/session.py` — FOUND
- `src/flec/logger.py` — FOUND
- `src/flec/main.py` — FOUND
- `tests/contract/test_camera_module.py` — FOUND
- `tests/contract/test_tts_engine.py` — FOUND
- `tests/contract/test_response_engine.py` — FOUND

Commits verified in git log: c107adf, a56f44a, 869c801, 35108d3, 693f36c, 0ad9c6a, d1f6fd9, 7e01ed1, 6bc675d

Verification commands:
- `python -m flec.main --mode dev --dry-run` — exits 0 with structured JSON logs
- `pytest tests/contract/ -v` — 28 passed, 0 failed
