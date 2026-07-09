# Flec

Screenless wearable "superhero mask" for toddlers (~2–4). An on-device camera plus
computer vision and audio help children recognize shapes, colors, and words — narrating
the real world back to them with no screen and no data ever leaving the device.

## Tech Stack

- **Language:** Python 3.11 (3.12 OK; not 3.13/3.14 — Coqui TTS caps at <3.12)
- **Platform:** macOS (dev) · ARM64 embedded Linux (production)
- **Computer vision:** OpenCV, MediaPipe (finger tracking), YOLOv8n / Ultralytics (shape + color)
- **Speech:** openWakeWord + openai-whisper tiny (in), Coqui TTS / VITS (out, needs `espeak-ng`)
- **Reading:** EasyOCR (text), BLIP-2 INT8 / Transformers (illustration description)
- **ML runtime:** PyTorch (MPS on macOS, no CUDA)
- **Testing:** pytest + hypothesis

## Project Structure

- `src/flec/` — package (`src/` layout, installed editable)
  - `main.py` (entry point + session loop), `models.py` (enums + dataclasses), `session.py` (state machines)
  - `camera/` frame capture + low-light · `perception/` shape/color + finger tracking
  - `speech/` wake word + STT · `audio/` TTS response builders
  - `reading/` OCR + illustration description · `engine/` ResponseEngine (output gatekeeper) · `ar/` dev overlay
- `tests/` — `contract/` (per-module interface), `integration/` (mode-level E2E), `unit/` (property tests, benchmarks, log schema)
- `scripts/` — `download_models.py`, `list_cameras.py`
- `specs/001-perception-core/` — feature spec + quickstart · `.models/` — downloaded model cache (gitignored)

## Development

- **Python env:** `uv venv --python 3.11 .venv` then `source .venv/bin/activate` (3.14 will not resolve the ML deps)
- **Install:** `pip install -r requirements.txt` then `pip install -e .`
  - `openai-whisper` builds from sdist and needs `setuptools<81` present with `--no-build-isolation` (its `setup.py` imports the removed `pkg_resources`)
  - `mediapipe` is pinned `<0.10.30` — newer releases dropped the legacy `mp.solutions` API that `finger_tracker.py` uses
- **Models:** `python scripts/download_models.py` (YOLOv8n, Whisper tiny, EasyOCR ship cleanly; Coqui VITS needs `espeak-ng`; BLIP-2 8-bit needs `accelerate`/`bitsandbytes`, unavailable on macOS MPS)
- **Run (dev):** `python -m flec.main --mode dev` · **Validate only:** `python -m flec.main --dry-run`
- **Test:** `pytest` · fast: `pytest -m "not slow"` · `pytest tests/contract/` · `pytest tests/integration/`
- **Deploy:** Not configured (on-device wearable; provisioning flow TBD)
- **Config:** copy `.env.example` → `.env`; all env vars use the `FLEC_` prefix

## Architecture

Queue-decoupled pipeline: capability modules **never import each other** — they communicate
only via in-memory queues (perception → event queue → `ResponseEngine` → audio queue).
`engine/response_engine.py` is the single gatekeeper for all audio/AR output and routes events
by the current `Mode`. Heavy ML libraries are lazy-imported inside methods and degrade
gracefully when missing (log a warning, no-op). All inference is on-device with zero
persistence of frames/audio/biometrics; models are pre-warmed at boot (<10s target).

The four child-facing modes: **Exploration** (narrate shapes/colors), **Challenge**
(voice-commanded find-it games), **Reading** (fingertip tracking reads words), **Story**
(autonomous picture-book read-aloud). Wear detection suspends everything when the mask is off.

## Coding Conventions

- Modules/functions `snake_case`; classes `PascalCase`; enum members `SCREAMING_SNAKE`; env vars `FLEC_`-prefixed
- Structured JSON logging to stdout: `logger.info(json.dumps({"event": ..., ...}))`
- Lazy-import heavy ML deps inside methods; never at module top level
- Every capability module must degrade gracefully when its model/dependency is unavailable
- No error messages ever reach the child — all user-facing feedback is audio
- New capability modules must obey the queue-only contract (no cross-module imports)

## Key Files

- `src/flec/main.py` — entry point, arg parsing, `FlecSession` loop wiring
- `src/flec/models.py` — shared enums (Mode, WearState, DetectionType, …) and dataclasses (DetectionEvent, AudioResponse, …)
- `src/flec/engine/response_engine.py` — single audio/AR output gatekeeper, mode routing
- `src/flec/perception/finger_tracker.py` — MediaPipe fingertip tracking + reading intent
- `src/flec/perception/shape_color_detector.py` — YOLOv8n shape/color detection
- `src/flec/reading/ocr_reader.py` / `illustration_describer.py` — EasyOCR / BLIP-2
- `src/flec/camera/camera_module.py` — frame capture + low-light detection
- `requirements.txt` / `pyproject.toml` — deps and packaging
- `specs/001-perception-core/quickstart.md` — full setup guide
