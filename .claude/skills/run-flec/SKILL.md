---
name: "run-flec"
description: "Launch and drive the Flec app locally on macOS — dev session loop, live preview window, and integrated-webcam vs iPhone Continuity Camera selection. Use when asked to run, start, preview, or verify the Flec app."
metadata:
  author: "flec"
  platform: "macOS (dev)"
---

# Running Flec locally (dev)

Verified launch path for the Flec dev app on macOS. Flec is normally **headless**
(captures frames, runs perception, emits JSON logs). Use `--preview` to open a
window and see the camera feed + fingertip AR overlay.

Full user-facing guide: **`docs/RUNNING.md`**.

## Preconditions (verified)

- Venv exists at `.venv` (Python 3.11), `flec` installed editable. Confirm:
  ```bash
  .venv/bin/python --version          # 3.11.x
  .venv/bin/pip show flec | head -2   # Name: flec
  ```
- **Camera permission is required.** System Settings → Privacy & Security →
  Camera → enable the terminal app. Without it the camera opens but returns
  black frames and the low-light detector fires every cycle
  (`mean_brightness≈0`).
- The app **auto-sets `OPENCV_AVFOUNDATION_SKIP_AUTH=1`** internally (see
  `main.py`), so no env prefix is needed. This is mandatory on macOS because
  OpenCV opens the capture on a background thread and cannot show the permission
  prompt from there.

## Validate first (no camera needed)

```bash
.venv/bin/python -m flec.main --dry-run
```
Expect `flec_start` then "configuration validated". Exit 0.

## Launch

```bash
# Integrated webcam + live preview window (default dev path — most reliable)
.venv/bin/python -m flec.main --mode dev --camera integrated --preview

# iPhone Continuity Camera (auto-probes indices 1–5; falls back to webcam)
.venv/bin/python -m flec.main --mode dev --camera iphone --preview

# Explicit device index (overrides --camera)
.venv/bin/python -m flec.main --mode dev --camera-index 1 --preview

# Headless (JSON logs only)
.venv/bin/python -m flec.main --mode dev --camera integrated
```

Device-index precedence: `FLEC_CAMERA_INDEX` env → `--camera-index` → `--camera`.
Discover indices with `.venv/bin/python scripts/list_cameras.py` (iPhone must be
awake/unlocked with Continuity Camera enabled to appear).

## Drive it (don't just launch)

- **With `--preview`:** a window "Flec — dev preview" opens. Wave a finger at the
  camera — the glowing fingertip trail appears and the HUD flips
  `finger=no` → `YES`. **Look at the window**; a black frame = permission missing
  or camera asleep, not success. Quit with `q`/`Esc` in the window or `Ctrl+C`.
- **Headless:** watch stdout JSON for the success markers below.

## Success markers (JSON log events)

- `flec_ready` — session initialized
- `camera_selected` — chosen index + source (env/flag/integrated/iphone)
- `flec_loop_start` — capture loop running (`"preview": true` when window is up)
- `CameraModule capture loop started` — camera opened
- `audio_response` — a response was generated

## Gotchas (all hit during bring-up)

- **Runs foreground, blocks:** the dev loop runs continuously until quit. When
  running via a tool, background it and tail the log; the GUI window only appears
  when launched in the user's own foreground terminal.
- `flec_loop_abort` / `camera_unavailable` → camera couldn't open: grant Camera
  permission, check the index.
- `iphone_not_found` → wake/unlock iPhone, enable Continuity Camera, retry;
  OpenCV's AVFoundation backend is flaky with Continuity Camera — integrated
  webcam is the reliable fallback.
- Benign stderr noise on macOS: `absl::InitializeLog`,
  `inference_feedback_manager`, `landmark_projection_calculator` (MediaPipe/TFLite).
