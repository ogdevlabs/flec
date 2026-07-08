---
one-liner: Voice-driven challenge mode with regex intent parsing, session lifecycle, throttled encouragement, 30s hint timer, and ResponseEngine routing — all wired end-to-end through main.py.
status: complete
phase: 5
plan: 1
subsystem: challenge-mode
tags: [tdd, voice-commands, challenge-mode, session-state, response-engine]
requires: [phase/1-setup]
provides: [CommandSTT, Session challenge lifecycle, ResponseEngine challenge routing, audio templates, voice command wiring]
affects: [main.py, session.py, engine/response_engine.py, speech/command_stt.py, audio/responses.py]
tech-stack:
  added: []
  patterns: [regex-keyword-extraction, throttled-events, frozen-dataclass-rebuild-pattern]
key-files:
  created:
    - src/flec/speech/command_stt.py
    - src/flec/session.py
    - src/flec/engine/response_engine.py
    - src/flec/audio/responses.py
    - tests/contract/test_command_stt.py
    - tests/integration/test_challenge_mode.py
  modified:
    - src/flec/main.py
decisions:
  - Frozen Challenge dataclass rebuilt (not mutated) for status updates — maintains immutability invariant while enabling lifecycle transitions
  - Encouraging responses throttled to once per 5s to prevent audio flooding on every CV frame
  - Hint timer resets after firing so hints repeat every 30s if toddler still hasn't found the target
  - transcribe_text() exposed as separate method from transcribe(bytes) to enable test-only path without audio hardware
  - ResponseEngine.set_challenge() accepts issued_at_override for deterministic hint testing
metrics:
  duration: ~25 minutes
  completed: 2026-07-08
  tasks_completed: 7
  files_created: 6
  files_modified: 1
---

# Phase 5 Plan 1: Challenge Mode Summary

## What Was Built

Full implementation of US2 (Challenge Mode: Ask & Verify Game):

1. **CommandSTT** (`src/flec/speech/command_stt.py`) — Regex + keyword intent parser that maps caregiver voice transcripts to `VoiceCommand` objects. Supports START_CHALLENGE (color + shape targets), CANCEL_CHALLENGE, SHUTDOWN, and UNKNOWN. Never raises.

2. **Session challenge lifecycle** (`src/flec/session.py`) — `start_challenge()`, `cancel_challenge()`, `complete_challenge()`, and `should_hint()` (fires after 30s, resets for repeat prompts). All state transitions emit structured JSON logs.

3. **ResponseEngine** (`src/flec/engine/response_engine.py`) — Single orchestration point routing VOICE_CMD, SHAPE, and COLOR events to AudioResponses. START_CHALLENGE → HIGH acknowledgment; target match → CRITICAL celebration; no match → NORMAL encouraging (throttled 5s); 30s elapsed → repeat hint.

4. **Audio templates** (`src/flec/audio/responses.py`) — `challenge_acknowledgment`, `challenge_celebration`, `challenge_encouraging` (variants), `challenge_hint`, plus exploration narration and session lifecycle phrases. All messages are positive — no negative words.

5. **Voice command wiring** (`src/flec/main.py`) — `handle_voice_command()` connects STT → Session → ResponseEngine. Emits `DetectionEvent(VOICE_CMD)` with embedded `VoiceCommand` for engine routing.

## Tests

- **Contract tests**: `tests/contract/test_command_stt.py` — 31 tests covering all color/shape keywords, cancel, shutdown, unknown, raw_text preservation, bytes interface
- **Integration tests**: `tests/integration/test_challenge_mode.py` — 17 tests covering acknowledgment, celebration, encouraging throttling, hint on expiry, cancellation, exploration passthrough
- **Total**: 48 tests, all passing

```
pytest tests/contract/test_command_stt.py -v  → 31 passed
pytest tests/integration/test_challenge_mode.py -v  → 17 passed
pytest tests/ -x  → 48 passed
```

## Commits

| Hash    | Type   | Description                              |
|---------|--------|------------------------------------------|
| 0609d6d | test   | contract-tests-command-stt-intents       |
| a7cfe5d | test   | integration-test-challenge-mode          |
| 03cb6a6 | feat   | command-stt-intent-parsing               |
| fcce5f7 | feat   | challenge-lifecycle-in-session           |
| e10dafb | feat   | challenge-audio-templates                |
| 72dab83 | feat   | response-engine-challenge-routing        |
| 3c42e80 | feat   | voice-command-session-wiring             |

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Implementation Notes

- T037 (audio templates) was implemented before T036 (ResponseEngine) to resolve the dependency order — ResponseEngine imports from `audio.responses`.
- The frozen `Challenge` dataclass uses a rebuild pattern for status updates (e.g., `ACTIVE → COMPLETED`) since Python frozen dataclasses cannot be mutated. This was the design intent from the models.py contract.
- `set_challenge()` on `ResponseEngine` accepts an `issued_at_override` parameter to enable deterministic 30s hint testing without actual time.sleep().

## Known Stubs

- `main.py` has a stub TTS (`_StubTTS`) — the real `TTSEngine` will be wired in the audio phase.
- Wake word callback (`on_wake_word_detected`) logs but does not capture audio — `WakeWordListener` integration is deferred to phase 6.
- `CommandSTT.transcribe(bytes)` returns UNKNOWN when no Whisper model is loaded — production wiring deferred to phase 6.

These stubs are intentional scaffolding; challenge mode logic is fully tested via `transcribe_text()` which exercises the complete intent-parsing pipeline.

## Self-Check: PASSED

All 7 files verified present on disk. All 7 task commits verified in git history.
