# Flec

Wearable superhero mask for toddler early learning. Flec uses computer vision and
audio to help toddlers recognize shapes, colors, and words — without a screen.

**Platform**: macOS (dev) | ARM64 embedded Linux (production)  
**Python**: 3.11+

---

## What It Does

A toddler wears Flec like a superhero mask. The embedded camera watches the world
in front of them. Flec speaks back:

- **Exploration mode** — narrates shapes and colors it sees ("I see a red triangle!")
- **Challenge mode** — voice-commanded games ("find something blue")
- **Reading mode** — tracks a fingertip across a page and reads words aloud
- **Story mode** — autonomously reads picture books, describing illustrations

All feedback is audio. No screen, no error messages to the child.

---

## Quickstart

### 1. Clone and set up

```bash
git clone <repo-url>
cd flec
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download models

```bash
python scripts/download_models.py
```

Downloads MediaPipe, YOLOv8n, EasyOCR, and BLIP-2 (~10 GB total). BLIP-2 is ~3.2 GB
and resumable — re-run if interrupted.

### 3. Run in dev mode

```bash
python -m flec.main --mode dev
```

### 4. Validate without running

```bash
python -m flec.main --dry-run
```

---

## Running Tests

```bash
# All tests
pytest

# Skip slow performance benchmarks
pytest -m "not slow"

# Contract tests only
pytest tests/contract/

# Integration tests only
pytest tests/integration/
```

---

## Project Structure

```
src/flec/
├── main.py                        # Entry point, boot, session loop
├── models.py                      # Shared data models and enums
├── session.py                     # Session state machines
├── camera/camera_module.py        # Frame capture + low-light detection
├── perception/
│   ├── shape_color_detector.py    # YOLOv8n shape/color detection
│   └── finger_tracker.py          # MediaPipe fingertip tracking
├── speech/command_stt.py          # Wake word + voice command STT
├── audio/responses.py             # TTS audio response builders
├── reading/
│   ├── ocr_reader.py              # EasyOCR page text extraction
│   └── illustration_describer.py  # BLIP-2 illustration descriptions
├── engine/response_engine.py      # Event routing (single audio/AR gatekeeper)
└── ar/ar_overlay.py               # AR border overlay (dev companion screen)

tests/
├── contract/    # Per-module interface tests
├── integration/ # Mode-level end-to-end tests
└── unit/        # Property tests, benchmarks, log schema validation
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLEC_CAMERA_INDEX` | `0` | OpenCV camera device index |
| `FLEC_DEV_MODE` | `1` | Dev mode (laptop camera instead of mask camera) |
| `FLEC_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |
| `FLEC_LOW_LIGHT_THRESHOLD` | `40` | Low-light mean pixel threshold (0–255) |
| `FLEC_TTS_VOICE` | `default` | TTS voice character |
| `FLEC_WAKE_WORD` | `hey_flec` | Wake word string |

Copy `.env.example` to `.env` to configure locally.

---

## Architecture

- **No cross-module imports** — capability modules communicate via in-memory queues only
- **All models pre-warmed at boot** — <10s ready-state target
- **Wear detection gates everything** — all modes suspend if the mask is removed
- **Zero persistence** — no frames, audio, or biometric data ever written to disk
- **Toddler-first UX** — no error messages reach the child; all feedback is audio-complete

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'flec'`** — run `pip install -e .`

**`mediapipe` install fails on macOS ARM** — try `pip install mediapipe --no-binary mediapipe`

**Camera not found** — check available devices with `python scripts/list_cameras.py`,
then set `FLEC_CAMERA_INDEX` in `.env`

**Low-light too sensitive** — raise the threshold: `FLEC_LOW_LIGHT_THRESHOLD=60 python -m flec.main`

**`hypothesis` not installed** — run `pip install hypothesis`

See `specs/001-perception-core/quickstart.md` for the full setup guide.
