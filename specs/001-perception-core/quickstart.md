# Flec Perception Core — Quickstart Guide

**Feature**: 001-perception-core (Shapes, Colors & Reading Enabler)
**Platform**: macOS (dev) | ARM64 embedded Linux (production)
**Python**: 3.11+

---

## Prerequisites

- macOS 12+ or Ubuntu 22.04+ (ARM64 for production)
- Python 3.11 or 3.12
- At least 8 GB RAM (BLIP-2 INT8 model requires ~4 GB)
- ~10 GB free disk space for all models

---

## 1. Clone and Navigate

```bash
git clone <repo-url>
cd flec
```

---

## 2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Verify Python version:

```bash
python --version
# Expected: Python 3.11.x or 3.12.x
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Expected output: all packages installed without errors.

**Known issues:**

- `mediapipe` on macOS ARM: if install fails, try `pip install mediapipe --no-binary mediapipe`
- `openwakeword` may require `pip install openwakeword[full]` on some systems
- `TTS` (Coqui): requires `espeak-ng` on Linux. Install with `sudo apt install espeak-ng`
- `torch` on macOS ARM: the default torch wheel supports MPS; no CUDA required

---

## 4. Install Flec Package (Editable)

```bash
pip install -e .
```

Verify:

```bash
python -c "from flec.models import DetectionEvent; print('OK')"
# Expected: OK
```

---

## 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` for your setup. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLEC_CAMERA_INDEX` | `0` | Camera device index (0 = built-in, 1+ = external) |
| `FLEC_DEV_MODE` | `true` | Use laptop/iPhone camera instead of embedded camera |
| `FLEC_LOG_LEVEL` | `INFO` | Logging verbosity |
| `FLEC_LOW_LIGHT_THRESHOLD` | `40` | Mean pixel brightness below which low-light is detected |

---

## 6. Download Models

Downloads all required AI model weights. Run once before first use.

```bash
python scripts/download_models.py
```

**Model sizes (approximate):**

| Model | Size | Purpose |
|-------|------|---------|
| YOLOv8n | ~6 MB | Shape detection |
| Whisper tiny | ~75 MB | Speech-to-text (caregiver commands) |
| Coqui VITS | ~82 MB | Text-to-speech (kid-friendly voice) |
| EasyOCR (latin) | ~90 MB | OCR for book reading |
| BLIP-2 INT8 | ~3.2 GB | Illustration description |

**Total: approximately 3.5 GB**

**Expected output:**

```
Flec model downloader
Target directory: /path/to/flec/.models
--------------------------------------------------

[YOLOv8n]
  [OK] YOLOv8n saved to .models/yolov8n.pt

[Whisper tiny]
  [OK] Whisper tiny saved to .models/whisper-tiny

[Coqui VITS]
  [OK] Coqui VITS downloaded (cached by TTS library)

[EasyOCR latin]
  [OK] EasyOCR latin models saved to .models/easyocr

[BLIP-2 INT8]
  [OK] BLIP-2 INT8 saved to .models/blip2-int8

--------------------------------------------------
Done. Run 'python -m flec.main --mode dev' to start.
```

---

## 7. Run Tests

```bash
# All tests
pytest

# Contract tests only
pytest tests/contract/

# Unit tests only
pytest tests/unit/

# Performance benchmarks (slow)
pytest tests/unit/test_performance.py -v -m slow
```

Expected: all tests pass (some skip if models not downloaded).

---

## 8. Run Dev Mode

```bash
python -m flec.main --mode dev
```

Expected output:

```
{"level": "INFO", "module": "flec.main", "msg": "Flec starting in dev mode"}
{"level": "INFO", "module": "flec.main", "msg": "Flec ready (stub — full boot in later phases)"}
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'flec'`

The package is not installed. Run:

```bash
pip install -e .
```

### `ImportError: cannot import name 'cv2'`

OpenCV is not installed. Run:

```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'mediapipe'`

```bash
pip install mediapipe
```

If that fails on macOS ARM:

```bash
pip install mediapipe --no-binary mediapipe
```

### Camera not found (`Could not open device 0`)

Check available cameras:

```bash
python scripts/list_cameras.py
```

Set the correct device index in `.env`:

```bash
FLEC_CAMERA_INDEX=1
```

### Low-light threshold too sensitive

Adjust via environment variable:

```bash
FLEC_LOW_LIGHT_THRESHOLD=20 python -m flec.main --mode dev
```

Higher values = more sensitive (detects low light sooner).

### Tests fail with `hypothesis not installed`

```bash
pip install hypothesis
```

### BLIP-2 download very slow

BLIP-2 INT8 is ~3.2 GB. Allow 10–30 minutes depending on connection.
The download is resumable — re-run `python scripts/download_models.py` if interrupted.

### `WARN` messages during model download

Missing packages are skipped gracefully. Install any missing packages from
`requirements.txt` and re-run `python scripts/download_models.py`.

---

## Environment Variables Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FLEC_CAMERA_INDEX` | int | `0` | OpenCV camera device index |
| `FLEC_DEV_MODE` | bool | `true` | Dev mode (laptop camera instead of mask camera) |
| `FLEC_LOG_LEVEL` | str | `INFO` | Log verbosity (DEBUG/INFO/WARNING/ERROR) |
| `FLEC_TTS_VOICE` | str | `default` | TTS voice character |
| `FLEC_WAKE_WORD` | str | `hey_flec` | Wake word string |
| `FLEC_LOW_LIGHT_THRESHOLD` | float | `40` | Low-light mean pixel threshold (0–255) |
