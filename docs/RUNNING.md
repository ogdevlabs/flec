# Running Flec Locally (dev)

How to launch Flec on macOS, see the live camera feed, hear it narrate the real
world, and **switch between the four child-facing modes by voice**.

> In dev, Flec captures frames, runs YOLO object recognition + finger tracking,
> speaks through your speakers, and listens on the microphone for mode commands.
> Use `--preview` to open a window showing the feed and the **active-mode banner**.

---

## 1. Prerequisites (one time)

```bash
# From the repo root
uv venv --python 3.11 .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Text-to-speech (Coqui VITS) needs the espeak-ng system binary:
brew install espeak-ng            # macOS dev
# (Linux/production: sudo apt install espeak-ng)

# Download models (YOLOv8n, Whisper tiny, Coqui VITS, ...)
python scripts/download_models.py
```

**Grant permissions:** System Settings → Privacy & Security →
- **Camera** → enable your terminal app (Terminal / iTerm / VS Code / PyCharm).
- **Microphone** → enable the same app (required for voice mode-switching).

> On first run Flec lets OpenCV *request* Camera authorization
> (`OPENCV_AVFOUNDATION_SKIP_AUTH=0`), so the macOS prompt appears. If it doesn't,
> grant the app manually above and relaunch it. Export
> `OPENCV_AVFOUNDATION_SKIP_AUTH=1` to force-skip the request.

---

## 2. Quick start — integrated webcam + live window

```bash
.venv/bin/python -m flec.main --mode dev --camera integrated --preview
```

- A window opens with a **colored banner showing the active mode**.
- It boots in **Exploration** and narrates real objects it recognizes
  ("Look, a red cup!"). It only speaks what's in view *now* — it won't keep
  naming something after it leaves the frame.
- Say a mode name to switch (see §3). **Press `q` / `Esc`** (or `Ctrl+C`) to quit.

Headless (no window, JSON logs only):

```bash
.venv/bin/python -m flec.main --mode dev --camera integrated
```

---

## 3. Modes — one at a time, switch by voice

Flec runs **exactly one mode at a time**. Switching modes replaces the current
one (and clears any pending narration), so you never have two modes active at
once. Speak clearly; the mic listener (Whisper) transcribes short phrases.

| Mode | Say one of… | What it does |
|---|---|---|
| **Exploration** | "exploration", "explore", "look around" | Narrates real objects it sees (+ their color) |
| **Reading** | "reading", "read" | Reads words your fingertip points at |
| **Story** | "story", "story time", "book" | Autonomous picture-book read-aloud |
| **Challenge** | "challenge", "let's play a game" | Find-it game (see below) |

**Starting a Challenge directly** — you can name the target in one phrase; it
enters Challenge mode and celebrates when it sees a match:

- `"find a cup"` / `"find the ball"` — a real **object** (matched by YOLO)
- `"find something red"` — a **color** (matches any object of that color)
- `"find a circle"` — a **shape** (needs `--shapes`, see §5)

**During a Challenge:** say `"stop"` or `"cancel"` to end it (returns to
Exploration). Say `"off"` or `"hey flec off"` to shut the session down.

The **preview banner color** tells you the active mode at a glance:
🟢 Exploration · 🔵 Reading · 🟣 Story · 🟠 Challenge · ⚪ Standby.

> Voice switching needs the microphone (`--voice`, on by default) and a working
> Whisper model. Run with `--no-voice` to disable the mic (perception + audio
> only, no mode switching).

---

## 4. Reading mode — point to read words

Switch to Reading mode by saying **"reading"** or **"read"**. Then hold a book or
printed page in front of the camera and rest your index fingertip steadily under
a word.

**How it works:**
- Flec tracks your fingertip with MediaPipe and waits until it slows to a halt
  (settle gate — configurable, see §5).
- Once settled, it crops the region around the fingertip and runs EasyOCR.
- The crop is probed in both the camera's natural orientation and mirrored; the
  reading with higher confidence wins. If neither is confident enough, Flec stays
  silent (no gibberish).
- A confident word is spoken once. Move to a new word and the old narration is
  dropped immediately.
- If no word is found but there is a picture, the IllustrationDescriber (BLIP-2)
  describes the region instead, where the model is available.

**Quick test on the integrated webcam:**

```bash
.venv/bin/python -m flec.main --mode dev --preview
# Say "reading" to activate, then point at any printed text on screen
```

> The integrated webcam is mirrored by default on macOS. Flec auto-detects the
> mirror orientation by comparing confidence scores, so printed text works as-is.

**Troubleshooting Reading mode:**

| Symptom | Cause / Fix |
|---|---|
| Words not read / silent | Say "reading" first; check `FLEC_READING_VELOCITY_THRESHOLD` — lower = stricter settle required |
| Wrong word / gibberish | Raise `FLEC_OCR_CONF_GATE` (default 0.4) to require higher confidence before speaking |
| Stale word spoken after moving finger | Lower `FLEC_OCR_SETTLE_THRESHOLD` so the settle gate resets faster |
| "reading_ocr_unavailable" in logs | Run `python scripts/download_models.py` — EasyOCR model missing |
| Illustration never described | BLIP-2 unavailable on macOS MPS (expected); silent fallback is by design |

---

## 5. Launch with the iPhone (Continuity Camera)

**Enable Continuity Camera first:**

1. iPhone → Settings → General → AirPlay & Continuity → **Continuity Camera** = on.
2. Mac and iPhone on the **same Apple ID**; Wi-Fi + Bluetooth on both.
3. Keep the iPhone **near the Mac, unlocked/awake**, camera unobstructed.

```bash
.venv/bin/python -m flec.main --mode dev --camera iphone --preview
```

Probes device indices 1–5 for a Continuity Camera; if none is found, logs
`iphone_not_found` and falls back to the integrated webcam (index 0).

```bash
.venv/bin/python scripts/list_cameras.py            # find the exact index
.venv/bin/python -m flec.main --camera-index 1 --preview   # pin an index
```

> OpenCV's AVFoundation backend can be flaky with Continuity Camera. If an iPhone
> index won't open, `--camera integrated` is the reliable dev path.

---

## 6. All options

| Flag / Env | Values | Default | Purpose |
|---|---|---|---|
| `--mode` | `dev`, `prod` | `dev` | Run mode |
| `--camera` | `integrated`, `iphone` | `integrated` | Camera source preset |
| `--camera-index` | integer | — | Explicit OpenCV device index (overrides `--camera`) |
| `--preview` | flag | off | Open the live window with the mode banner |
| `--tts` | `coqui`, `say`, `off` | `coqui` | Speech backend (`coqui` falls back to `say`→log) |
| `--voice` / `--no-voice` | flag | voice **on** | Enable/disable mic voice mode-switching |
| `--shapes` | flag | off | Also run contour/HSV geometric-shape + color learning |
| `--log-level` | `DEBUG`…`ERROR` | `INFO` | Logging verbosity |
| `--dry-run` | flag | off | Validate config + imports, then exit |
| `FLEC_CAMERA_INDEX` | integer | — | Env override; **highest** precedence for device index |
| `FLEC_YOLO_MODEL` | path | `.models/yolov8n.pt` | Use a larger/custom trained YOLO model |
| `FLEC_TARGET_FPS` | number | `30` | Frame-processing rate cap |
| `FLEC_READING_VELOCITY_THRESHOLD` | float | `0.08` | Max fingertip velocity (normalised/frame) to enter READING intent |
| `FLEC_READING_FRAMES` | int | `3` | Consecutive low-velocity frames required before intent becomes READING |
| `FLEC_OCR_SETTLE_THRESHOLD` | float | `0.02` | Max velocity to run OCR (settle gate — stricter than READING intent) |
| `FLEC_OCR_CONF_GATE` | float | `0.4` | Minimum OCR confidence to speak a word (lower = more permissive but noisier) |
| `FLEC_READING_WEAR_OVERRIDE` | `0`/`1` | `1` in dev | Treat integrated webcam as ON_HEAD so Reading activates without a wear sensor |

**Device-index precedence:** `FLEC_CAMERA_INDEX` → `--camera-index` → `--camera` preset.

> YOLO is the source of truth for what's in frame. `--shapes` adds the geometric
> shape/color heuristics (for flashcard-style learning); leave it off for
> real-world object narration.

---

## 7. Verify it's working

Watch the JSON logs for these events:

- `flec_ready` / `flec_loop_start` — session + capture loop running
- `camera_selected` — which index was chosen
- `tts.backend_selected` — `coqui` (or the `say`/`log` fallback)
- `detections` — objects recognized this frame (`labels: [...]`)
- `tts.audio_response` — narration text spoken
- `mic.command` / `response_engine.mode_changed` — a voice command switched mode

**Troubleshooting**

| Symptom | Cause / Fix |
|---|---|
| No audio | Install `espeak-ng` (Coqui needs it); or try `--tts say`; check `tts.backend_selected` |
| Keeps naming things not in view | Fixed — narration is coalesced + expires; if seen, file a bug with logs |
| Inaccurate object names | Low light or partial view; try a larger model via `FLEC_YOLO_MODEL` |
| Mode won't switch by voice | Grant Microphone permission; check for `mic.started`; speak the exact phrase (§3) |
| `not authorized to capture video` | Grant Camera permission to your terminal app; relaunch it |
| `iphone_not_found` | Wake/unlock iPhone, enable Continuity Camera, retry |
| No window appears | Run in the **foreground**; ensure `--preview` is set |

---

## 8. Validate without a camera

```bash
.venv/bin/python -m flec.main --dry-run
```

Confirms configuration and imports resolve, then exits — no camera needed.
