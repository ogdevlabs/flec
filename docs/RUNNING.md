# Running Flec Locally (dev)

How to launch Flec on macOS, view the live camera feed with the AR overlay, and
choose between the **integrated webcam** and an **iPhone Continuity Camera**.

> Flec is normally headless — it captures frames, runs perception, and emits JSON
> logs to stdout. Use `--preview` to open a window and actually *see* what the
> camera captures and what the finger tracker detects.

---

## 1. Prerequisites (one time)

```bash
# From the repo root
uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Grant camera permission:** System Settings → Privacy & Security → **Camera** →
enable your terminal app (Terminal / iTerm / VS Code). Without this, the camera
opens but delivers black frames (the low-light detector will fire constantly).

> The app auto-sets `OPENCV_AVFOUNDATION_SKIP_AUTH=1` internally, so you no longer
> need to prefix commands with it. (OpenCV can't show the macOS permission prompt
> from its capture thread — hence the manual grant above.)

---

## 2. Quick start — integrated webcam + live window

```bash
.venv/bin/python -m flec.main --mode dev --camera integrated --preview
```

- A window titled **"Flec — dev preview"** opens showing the live feed.
- Wave a finger in front of the camera — you'll see the glowing fingertip trail
  and the HUD flip `finger=no` → `YES`.
- **Press `q` or `Esc`** in the window (or `Ctrl+C` in the terminal) to quit.

Headless (no window, JSON logs only):

```bash
.venv/bin/python -m flec.main --mode dev --camera integrated
```

---

## 3. Launch with the iPhone (Continuity Camera)

**Enable Continuity Camera first:**

1. iPhone → Settings → General → AirPlay & Continuity → **Continuity Camera** = on.
2. Mac and iPhone signed into the **same Apple ID**; Wi-Fi + Bluetooth on both.
3. Keep the iPhone **near the Mac, unlocked/awake**, camera unobstructed.

**Auto-detect the iPhone** (probes device indices 1–5 for a Continuity Camera):

```bash
.venv/bin/python -m flec.main --mode dev --camera iphone --preview
```

If none is found, Flec logs `iphone_not_found` and falls back to the integrated
webcam (index 0).

**Find the exact index** (with the iPhone awake and nearby):

```bash
.venv/bin/python scripts/list_cameras.py
```

**Pin a specific index** (overrides `--camera`):

```bash
.venv/bin/python -m flec.main --mode dev --camera-index 1 --preview
```

> Note: OpenCV's AVFoundation backend can be flaky with Continuity Camera. If an
> iPhone index won't open, the integrated webcam (`--camera integrated`) is the
> reliable dev path.

---

## 4. All options

| Flag / Env | Values | Default | Purpose |
|---|---|---|---|
| `--mode` | `dev`, `prod` | `dev` | Run mode |
| `--camera` | `integrated`, `iphone` | `integrated` | Camera source preset |
| `--camera-index` | integer | — | Explicit OpenCV device index (overrides `--camera`) |
| `--preview` | flag | off | Open the live AR preview window |
| `--log-level` | `DEBUG`…`ERROR` | `INFO` | Logging verbosity |
| `--dry-run` | flag | off | Validate config + imports, then exit |
| `FLEC_CAMERA_INDEX` | integer | — | Env override; **highest** precedence for device index |
| `FLEC_TARGET_FPS` | number | `30` | Frame-processing rate cap |
| `FLEC_LOW_LIGHT_THRESHOLD` | 0–255 | `40` | Mean brightness below which low-light fires |

**Device-index precedence:** `FLEC_CAMERA_INDEX` → `--camera-index` → `--camera` preset.

---

## 5. Verify it's working

Watch the JSON logs for these events:

- `flec_ready` — session initialized
- `camera_selected` — which index was chosen (`source` = env / flag / integrated / iphone)
- `flec_loop_start` — capture loop running (`"preview": true` when the window is up)
- `CameraModule capture loop started` — camera opened successfully
- `audio_response` — a response was generated (e.g. reading a word)

**Troubleshooting**

| Symptom | Cause / Fix |
|---|---|
| `flec_loop_abort` `camera_unavailable` | Camera couldn't open — grant Camera permission; check the index |
| `Low-light detected: mean_brightness≈0` | Camera permission missing, lens covered, or iPhone asleep |
| `iphone_not_found` | Wake/unlock iPhone, enable Continuity Camera, retry |
| No window appears | Run in the **foreground** in your terminal (not backgrounded); ensure `--preview` is set |

---

## 6. Validate without a camera

```bash
.venv/bin/python -m flec.main --dry-run
```

Confirms configuration and imports resolve, then exits — no camera needed.
