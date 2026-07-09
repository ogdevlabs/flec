# Overview
<!-- pdlc-template-version: 2.1.0 -->
<!-- This file is the living, aggregated record of everything this product does and has shipped.
     It is updated automatically by PDLC after every successful merge to main (during Reflect sub-phase).
     Use it to orient yourself after time away, onboard a new teammate, or brief Claude in a fresh session.
     Do not edit manually — let PDLC maintain it. If you need to correct something, update and note the reason. -->

**Project:** Flec
**Last updated:** 2026-07-08T00:00:00Z

---

## Project Summary

Flec is a screenless wearable "superhero mask" for toddlers (~2–4) that uses an on-device
camera plus computer vision and audio to help children recognize shapes, colors, and words —
narrating the real world back to them without any screen.

---

## Active Functionality

<!-- Pre-PDLC functionality, reverse-engineered from the codebase during the /setup scan. -->

- **Exploration mode** — passively narrates shapes and colors it sees ("I see a red triangle!")
- **Challenge mode** — voice-commanded find-it games ("find something blue"), with match/celebration feedback
- **Reading mode** — tracks a fingertip across a book page and reads the nearest word aloud
- **Story mode** — autonomously reads picture books, describing illustrations (BLIP-2)
- **Wake word + voice commands** — "Hey Flec" wake word and command parsing (start/cancel/repeat challenge, shutdown)
- **Wear detection** — suspends all modes when the mask is removed from the head
- **Low-light detection** — detects poor lighting from mean pixel brightness
- **AR dev overlay** — companion developer screen drawing detection boxes and fingertip trails (not shown to the child)

---

## Shipped Features

<!-- Auto-populated by PDLC after each successful merge to main.
     Pre-PDLC functionality documented above. -->

| # | Feature | Date Shipped | Episode | PR |
|---|---------|-------------|---------|-----|
| — | — | — | — | — |

---

## Architecture Summary

- **Queue-decoupled pipeline**: capability modules never import each other; they communicate via in-memory queues (perception → event queue → `ResponseEngine` → audio queue). Constitution §III, cited in `main.py`.
- **Single output gatekeeper**: `engine/response_engine.py` is the only component that emits audio/AR output, routing events by the current `Mode`.
- **Capability modules**: `camera/` (frame capture + low-light), `perception/` (YOLOv8n shape/color, MediaPipe finger tracking), `speech/` (wake word + STT), `audio/` (TTS response builders), `reading/` (EasyOCR + BLIP-2), `ar/` (dev overlay).
- **Lazy ML imports + graceful degradation**: heavy libraries load inside methods; a missing model logs a warning and the module no-ops rather than crashing.
- **On-device, zero-persistence**: all inference is local; no frames/audio/biometrics are written to disk. Models pre-warmed at boot (<10s target).
- **Structured JSON logging** to stdout throughout.

---

## Known Tech Debt

<!-- Reverse-engineered signals from the /setup scan. -->

- [2026-07-08] No CI/CD pipeline configured — set up at first `/ship`.
- [2026-07-08] No linter/formatter config (ruff/black) despite consistent style — add to lock in conventions.
- [2026-07-08] `requirements.txt` had unpinned `mediapipe>=0.10.0` that now resolves to a release without the legacy `mp.solutions` API used by `finger_tracker.py`; pinned `<0.10.30` during setup. Longer term, migrate FingerTracker to the MediaPipe Tasks API.
- [2026-07-08] Story mode (BLIP-2 INT8) needs `accelerate`/`bitsandbytes`, unsupported on macOS MPS — Story mode is degraded in the dev environment.
- [2026-07-08] Coqui TTS requires the `espeak-ng` system package; without it, TTS voice download/synthesis fails.
- [2026-07-08] `.planning/` is empty and `specs/` holds a single feature spec — planning history lives mostly in git.

---

## Decision Log Summary

- **Queue-only module decoupling** (no cross-module imports) is the defining architectural rule — see ADR-001.
- **On-device, zero-persistence** stance for child privacy — see ADR-003.
- Full ADR log: `docs/pdlc/memory/DECISIONS.md`.
