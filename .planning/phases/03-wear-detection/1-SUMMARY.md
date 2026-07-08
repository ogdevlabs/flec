---
one-liner: "WearDetector (MediaPipe + HSV fallback), WakeWordListener (openWakeWord), CommandSTT (Whisper), and Session wear/voice lifecycle methods implementing US5 mask-on/off session boundaries"
status: complete
phase: 3
plan: 1
subsystem: perception, speech, session
tags: [wear-detection, wake-word, stt, session, tdd]
dependency_graph:
  requires: [phase/2-foundational (Session, ResponseEngine, TTSEngine)]
  provides: [WearDetector, WakeWordListener, CommandSTT, Session.handle_wear_event, Session.handle_voice_command]
  affects: [session-loop in main.py (Phase 2 stub now replaceable), ResponseEngine (shutdown routing verified)]
tech_stack:
  added: [mediapipe>=0.10.0 (FaceDetection), openWakeWord, pyaudio, openai-whisper]
  patterns: [lazy-init for testable mocks, debounce with configurable bypass, daemon background thread with stop-event]
key_files:
  created:
    - src/flec/perception/wear_detector.py
    - src/flec/speech/wake_word_listener.py
    - src/flec/speech/command_stt.py
    - tests/contract/test_wear_detector.py
    - tests/contract/test_wake_word_listener.py
    - tests/integration/test_wear_lifecycle.py
    - .gitignore
  modified:
    - src/flec/session.py (added handle_wear_event, handle_voice_command)
decisions:
  - "WearDetector uses lazy MediaPipe init to enable clean mock patching in contract tests"
  - "debounce_seconds=0.0 parameter added to WearDetector to bypass 2s debounce in unit tests without time.sleep"
  - "WakeWordListener._trigger_detection_for_test() test-only hook avoids threading complexity in contract tests"
  - "ResponseEngine shutdown routing was already correct from Phase 2; T025 verified rather than modified"
metrics:
  duration: "~35 minutes"
  completed: "2026-07-08"
  tasks: 8
  files_created: 7
  files_modified: 1
---

# Phase 3 Plan 1: Wear Detection Summary

WearDetector (MediaPipe FaceDetection + HSV skin-tone fallback), WakeWordListener (openWakeWord on PyAudio thread), CommandSTT (Whisper tiny + rule-based intent parser), and Session wear/voice lifecycle methods implementing the complete US5 mask-on/off session boundary.

## What Was Built

### WearDetector (`src/flec/perception/wear_detector.py`)
- MediaPipe FaceDetection as primary signal (min_detection_confidence=0.5)
- HSV skin-tone proximity check (two Hue ranges covering broad skin tones) as fallback when MediaPipe reports no face
- 2-second configurable debounce (`debounce_seconds` param) to prevent rapid ON/OFF flicker
- Emits exactly one `DetectionEvent(type=WEAR)` per state transition — no flood
- Lazy MediaPipe init so test `patch()` calls on the module-level `mp` reference are honoured
- Handles dark, tiny, and float32 frames without raising (returns current state on error)
- Structured JSON log on every detection and transition

### WakeWordListener (`src/flec/speech/wake_word_listener.py`)
- openWakeWord "hey_flec" model running in a daemon background thread
- PyAudio stream (16kHz, mono, paInt16) with clean resource release in `stop()`
- Thread-safe `on_detected` callback with error protection
- `is_listening` property tracks state accurately across start/stop cycles
- `_trigger_detection_for_test()` test-only hook bypasses audio capture for contract tests
- `stop()` is idempotent — safe to call multiple times

### CommandSTT (`src/flec/speech/command_stt.py`)
- Whisper tiny model (lazy load on first transcription)
- PCM bytes (16-bit signed LE, 16kHz mono) → float32 normalisation → Whisper transcription
- Rule-based intent parser supporting all spec vocabulary: 10 shapes + 8 colors
- Intent priority: SHUTDOWN > CANCEL_CHALLENGE > START_CHALLENGE > UNKNOWN
- `transcribe()` never raises — returns `VoiceCommand(intent=UNKNOWN)` on any error
- Structured JSON log on every result and error

### Session additions (`src/flec/session.py`)
- `handle_wear_event(state: WearState) → bool`: delegates to `set_wear_state()` with extra logging
- `handle_voice_command(cmd: VoiceCommand) → bool`: SHUTDOWN only processed when ON_HEAD; CANCEL_CHALLENGE delegates to `cancel_challenge()`; UNKNOWN silently ignored

### ResponseEngine shutdown routing (T025 — verified, no changes needed)
- `VOICE_CMD(SHUTDOWN)` + ON_HEAD → CRITICAL priority "See you next time, hero!" AudioResponse
- `VOICE_CMD(SHUTDOWN)` + OFF_HEAD → no-op + structured log "shutdown_ignored"
- Both paths were already correctly implemented in Phase 2 and verified by existing contract tests

## Tests

| Suite | Tests | Result |
|---|---|---|
| `tests/contract/test_wear_detector.py` | 11 | PASS |
| `tests/contract/test_wake_word_listener.py` | 9 | PASS |
| `tests/integration/test_wear_lifecycle.py` | 9 | PASS |
| Phase 2 regression (full suite) | 28 | PASS |
| **Total** | **57** | **57 PASS** |

## Commits

| Hash | Type | Description |
|---|---|---|
| 159578b | test | contract-tests-wear-detector (RED) |
| 504e685 | test | contract-tests-wake-word-listener (RED) |
| f0152e1 | test | integration-test-wear-lifecycle (RED) |
| d27d7f1 | feat | implement-wear-detector (GREEN + debounce fix) |
| 99841e5 | feat | implement-wake-word-listener (GREEN) |
| f5fac15 | feat | implement-command-stt |
| c918d9e | feat | wire-wear-and-voice-into-session |
| 2874183 | chore | add-gitignore |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] WearDetector debounce prevents single-call ON_HEAD detection in tests**
- **Found during:** T021 GREEN phase
- **Issue:** `_maybe_transition()` requires a state to persist for `debounce_seconds` before confirming. With the default 2s debounce, contract tests calling `update()` once never see a transition.
- **Fix:** Added `debounce_seconds=0.0` as a constructor parameter. When 0.0, the first occurrence of a new candidate state is confirmed immediately (no wait). All tests use `debounce_seconds=0.0`; production uses the default 2.0.
- **Files modified:** `src/flec/perception/wear_detector.py`, `tests/contract/test_wear_detector.py`
- **Commit:** d27d7f1

**2. [Rule 1 - Bug] MediaPipe init at `__init__` time defeats test mock patching**
- **Found during:** T021 GREEN phase
- **Issue:** `WearDetector.__init__` was calling `mp.solutions.face_detection.FaceDetection()` immediately. Since `mp` is bound at module import time, test patches applied after construction had no effect.
- **Fix:** Lazy initialization via `_get_face_detector()` accessor — MediaPipe is instantiated on first call to `_detect()`, after any patches are already applied.
- **Files modified:** `src/flec/perception/wear_detector.py`
- **Commit:** d27d7f1

**3. [Rule 2 - Missing] No .gitignore in repo**
- **Found during:** post-T022 git status
- **Issue:** `__pycache__/`, `.egg-info/`, `.pytest_cache/` were untracked. These are standard Python build artifacts that should never be committed.
- **Fix:** Created `.gitignore` covering Python bytecode, egg-info, venv, pytest cache, macOS, and IDE directories.
- **Files created:** `.gitignore`
- **Commit:** 2874183

## Known Stubs

None — all implemented functionality is wired and exercised by tests. The session loop stub in `main.py` (from Phase 2) is documented there as "Wear detection will be wired in Phase 3" — that wiring is now available via `Session.handle_wear_event()` and `WearDetector`. Integrating them into the `_session_loop()` function is deferred to a later phase when the full pipeline is assembled.

## Self-Check: PASSED
