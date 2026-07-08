---
one-liner: "Bootstrapped the Flec Python package: src/flec/ package tree, pinned requirements, .env.example, pytest scaffolding, model download/camera-list scripts, and all shared data models (enums + dataclasses)."
status: complete
phase: 1
plan: 1
subsystem: infrastructure
tags: [setup, package-structure, models, testing, scripts]
dependency-graph:
  requires: []
  provides: [flec-package, shared-models, test-scaffold, model-scripts]
  affects: [all-subsequent-phases]
tech-stack:
  added: [pytest>=7.4.0, hypothesis>=6.90.0]
  patterns: [dataclass-models, enum-states, editable-install-pyproject]
key-files:
  created:
    - pyproject.toml
    - requirements.txt
    - .env.example
    - src/flec/__init__.py
    - src/flec/main.py
    - src/flec/models.py
    - src/flec/camera/__init__.py
    - src/flec/perception/__init__.py
    - src/flec/speech/__init__.py
    - src/flec/audio/__init__.py
    - src/flec/reading/__init__.py
    - src/flec/engine/__init__.py
    - src/flec/ar/__init__.py
    - tests/conftest.py
    - tests/contract/__init__.py
    - tests/contract/conftest.py
    - tests/integration/__init__.py
    - tests/integration/conftest.py
    - tests/unit/__init__.py
    - tests/unit/conftest.py
    - scripts/download_models.py
    - scripts/list_cameras.py
  modified: []
decisions:
  - "Used pyproject.toml (not setup.py) with setuptools build backend for PEP 517 compliance"
  - "Challenge.EXPIRY_SECONDS set to 30s matching FR-006 hint timeout from spec"
  - "IllustrationDescription enforces <=20 word limit in __post_init__ to uphold contract spec"
  - "BoundingBox validates [0.0,1.0] coordinates at construction time, fail fast"
metrics:
  duration: "~15 minutes"
  completed: "2026-07-08"
  tasks: 7
  files_created: 23
---

# Phase 1 Plan 1: Setup Summary

## What Was Built

Seven tasks bootstrapped the complete project skeleton for Flec:

1. **Package structure** (`src/flec/` + 7 subpackages): All directories from the architecture spec created with `__init__.py` files. `main.py` entry point supports `--mode dev|prod` and `--log-level`, satisfying `python -m flec.main --help`. `pyproject.toml` installed package in editable mode.

2. **requirements.txt**: All AI/CV dependencies pinned with minimum versions — `opencv-python`, `ultralytics` (YOLOv8n), `mediapipe`, `openwakeword`, `openai-whisper`, `TTS` (Coqui), `easyocr`, `transformers`, `torch`, `pytest`, `hypothesis`.

3. **.env.example**: Documents all five environment variables (`FLEC_CAMERA_INDEX`, `FLEC_DEV_MODE`, `FLEC_LOG_LEVEL`, `FLEC_TTS_VOICE`, `FLEC_WAKE_WORD`) with descriptions.

4. **Test directory structure**: `tests/contract/`, `tests/integration/`, `tests/unit/` with `conftest.py` at each level. Top-level conftest provides shared frame fixtures (`blank_frame`, `white_frame`) and in-memory queues (`frame_queue`, `event_queue`). Contract conftest provides `red_circle_frame` and `blue_square_frame` helpers for interface tests.

5. **scripts/download_models.py**: Downloads YOLOv8n, Whisper tiny, Coqui VITS, EasyOCR latin, and BLIP-2 INT8 to `.models/`. Skips already-downloaded files (idempotent). Handles missing packages gracefully with WARN output (no fatal crashes).

6. **scripts/list_cameras.py**: Probes OpenCV indices 0–9, reports accessible cameras with index, backend name, resolution, and FPS. Exits with guidance if `opencv-python` not installed.

7. **src/flec/models.py**: All shared data models from contracts:
   - 8 enums: `WearState`, `Mode`, `DetectionType`, `AudioPriority`, `CommandIntent`, `ChallengeTargetType`, `ChallengeStatus`, `ReadingIntent`
   - 8 dataclasses: `BoundingBox` (validates [0.0,1.0]), `DetectionEvent`, `FingerTrackingState`, `Challenge` (with `is_expired()`), `StoryContext`, `IllustrationDescription` (enforces ≤20 words), `AudioResponse`, `VoiceCommand`

## Tests

```
pytest tests/ --collect-only
collected 0 items  (no tests yet — scaffolding phase only)
```

Verification checks all pass:
- `python -c "import flec; print('ok')"` → ok
- `python -m flec.main --help` → argparse help shown
- `python -c "from flec.models import WearState, Mode, DetectionEvent; print('ok')"` → ok
- `pytest tests/ --collect-only` → collected 0 items (no failures)

## Commits

| Hash    | Message |
|---------|---------|
| 7ca6a3a | feat(1-01): package-structure |
| 22df19a | feat(1-01): requirements-txt |
| 7c6cba2 | feat(1-01): env-example |
| 5836a31 | feat(1-01): test-directory-structure |
| 16f8c7b | feat(1-01): download-models-script |
| 7df5e37 | feat(1-01): list-cameras-script |
| fe7d3cf | feat(1-01): shared-data-models |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pyproject.toml backend syntax corrected**
- **Found during:** T001
- **Issue:** Initial `build-backend = "setuptools.backends.legacy:build"` caused `ModuleNotFoundError: No module named 'setuptools.backends'` on Python 3.12
- **Fix:** Changed to `build-backend = "setuptools.build_meta"` (correct PEP 517 backend string)
- **Files modified:** `pyproject.toml`
- **Commit:** 7ca6a3a

**2. [Rule 2 - Missing critical functionality] main.py added to satisfy verification goal**
- **Found during:** T001
- **Issue:** Plan's verification step requires `python -m flec.main --help` to run without import errors, but no `main.py` was listed in T001 files
- **Fix:** Added `src/flec/main.py` with argparse entry point as part of T001
- **Files modified:** `src/flec/main.py`
- **Commit:** 7ca6a3a

## Known Stubs

- `src/flec/main.py` boot sequence is a stub (logs "Flec ready (stub — full boot in later phases)"). This is intentional — full session loop is in-scope for phase 2.

## Self-Check: PASSED

All key files verified present and all commits confirmed in git log.
