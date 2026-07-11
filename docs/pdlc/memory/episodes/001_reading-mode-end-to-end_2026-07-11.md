# Episode 001 — reading-mode-end-to-end

**Date:** 2026-07-11  
**Feature:** reading-mode-end-to-end  
**Status:** Draft  
**Branch:** `feature/reading-mode-end-to-end`

---

## What Was Built

Reading mode end-to-end (F-001): a child can point a steady fingertip at a word printed on a book or card, and Flec speaks it aloud. The pipeline runs entirely on-device: MediaPipe finger tracking detects the fingertip and measures velocity; a settle gate (`should_run_ocr`) fires OCR only when the finger is still; `crop_around_fingertip` extracts a region-of-interest around the fingertip; `resolve_orientation` probes both the native and horizontally-mirrored crop and picks the higher-confidence read, staying silent when ambiguous; a confidence gate suppresses low-quality OCR; and `ResponseEngine` narrates the word via TTS. A word-change flush drops stale narration the instant the finger moves to a new word. When no confident word is found, `IllustrationDescriber` (BLIP-2) describes any picture under the fingertip as a fallback, gracefully degrading to silence when the model is unavailable (macOS/MPS).

The feature also shipped: dev wear-state override (`FLEC_READING_WEAR_OVERRIDE`) so Reading mode activates without a physical sensor during development; a `OnceWarner` mechanism for one-time dev signals; and §4 of `docs/RUNNING.md` with how-it-works explanation, quick-test command, and troubleshooting table.

---

## Links

- **PRD:** `docs/pdlc/prds/PRD_reading-mode-end-to-end_2026-07-10.md`
- **Plan:** `docs/pdlc/prds/plans/plan_reading-mode-end-to-end_2026-07-10.md`
- **Architecture:** `docs/pdlc/design/reading-mode-end-to-end/ARCHITECTURE.md`
- **Review:** `docs/pdlc/reviews/REVIEW_reading-mode-end-to-end_2026-07-11.md`
- **PR:** _(not yet merged)_

---

## Key Decisions & Rationale

1. **OCR runs as pure helper functions, not a daemon thread** — `ocr_worker.py` exposes module-level pure functions called from `FlecSession.process_frame`. The architecture doc originally described a "background daemon thread" but the inline approach was chosen to avoid the complexity of a separate thread mailbox while still keeping OCR compute off the UI thread via `OCRReader`'s internal `ThreadPoolExecutor`. See `ARCHITECTURE.md` and the D1–D4 decisions in `ocr_worker.py`'s module docstring.

2. **Silence over gibberish** — both the confidence gate (`FLEC_OCR_CONF_GATE`) and the orientation delta gate in `resolve_orientation` default to silence rather than speaking a low-confidence or ambiguous read. Established in the PRD: a toddler hearing a wrong word is worse than no word at all.

3. **Crop-only OCR** — only the fingertip region is passed to EasyOCR, not the full frame, to stay within the ARM64 inference budget (decision D4 in `ocr_worker.py`).

4. **Orientation probe with cache** — both normal and mirrored crops are probed on first use; the winning orientation is cached in `_ocr_cached_orient` to make subsequent calls single-probe. Cache is cleared on `reset_reading_state()`.

5. **Word-change flush via `tts_engine.clear_pending()`** — when OCR detects a different word from what the finger tracker has stored, pending TTS narration is flushed before calling `update_ocr`. This is the only location in the system that has both the old word (from tracker state) and the new word (from OCR) simultaneously.

---

## Files Created

**Source:**
- `src/flec/reading/ocr_worker.py` — settle gate, crop, orientation resolution, OnceWarner
- `src/flec/reading/ocr_reader.py` — EasyOCR wrapper with lazy-load and off-thread executor
- `src/flec/reading/illustration_describer.py` — BLIP-2 illustration fallback
- `src/flec/perception/finger_tracker.py` — MediaPipe fingertip tracking + reading intent
- `src/flec/engine/response_engine.py` — single audio/AR output gatekeeper
- `src/flec/audio/tts.py`, `audio/responses.py` — TTS engine + audio response types
- `src/flec/camera/camera_module.py` — frame capture + low-light detection
- `src/flec/perception/shape_color_detector.py` — YOLOv8n shape/color detection
- `src/flec/ar/ar_overlay.py` — dev overlay
- `src/flec/session.py` — mode state machine
- `src/flec/speech/command_stt.py`, `speech/mic_listener.py` — voice command pipeline

**Tests:**
- `tests/unit/test_ocr_worker.py` (19 tests)
- `tests/unit/test_reading_session.py` (12 tests)
- `tests/unit/test_reading_intent.py` (4 tests)
- `tests/integration/test_reading_mode.py` (19 tests)
- `tests/contract/test_ocr_reader.py`
- `tests/contract/test_finger_tracker.py`
- `tests/contract/test_illustration_describer.py`
- `tests/contract/test_shape_color_detector.py`
- `tests/contract/test_command_stt.py`
- `tests/unit/test_finger_tracker_properties.py`
- `tests/unit/test_low_light_detection.py`
- `tests/unit/test_shape_color_detector_properties.py`
- `tests/unit/test_logging.py`
- `tests/unit/test_performance.py`
- `tests/integration/test_challenge_mode.py`
- `tests/integration/test_exploration_mode.py`
- `tests/integration/test_story_mode.py`

**Docs:**
- `docs/RUNNING.md` — §4 Reading mode how-to
- `docs/pdlc/design/reading-mode-end-to-end/` — ARCHITECTURE.md, api-contracts.md, data-model.md, threat-model.md, ux-review.md
- `docs/pdlc/reviews/REVIEW_reading-mode-end-to-end_2026-07-11.md`

---

## Files Modified

- `src/flec/main.py` — added OCR pipeline wiring to `process_frame`; new instance vars `_ocr_reader`, `_illustration_describer`, `_ocr_once_warner`, `_ocr_settle_threshold`, `_ocr_conf_gate`, `_ocr_cached_orient`
- `src/flec/models.py` — added `ReadingIntent` enum, `FingerTrackingState` dataclass fields
- `pyproject.toml` — dependency additions
- `requirements.txt` — dependency additions

---

## Test Summary

| Layer | Command | Result |
|-------|---------|--------|
| Unit | `pytest tests/unit/` | 109 passed, 1 skipped |
| Integration | `pytest tests/integration/` | 51 passed |
| Contract | `pytest tests/contract/` | 93 passed, 9 skipped |
| **Total** | `pytest` | **253 passed, 10 skipped, 2 warnings** |
| E2E (Playwright) | N/A — not a web UI | Skipped (N/A) |
| Performance | embedded in unit (test_performance.py) | 8 passed |
| Security scan | secret grep on diff | Clean |
| Dependency audit | pre-existing deps; no new packages | Clean |

---

## Known Tradeoffs & Tech Debt

Issues filed as beads after Party Review (CONDITIONAL PASS):

**Critical — filed for next sprint:**
- `flec-ois` — AC-6 test coverage (OCR off-thread / timeout guard)
- `flec-b0e` — AC-9 test coverage (zero disk-write assertion)
- `flec-7xu` — `resolve_orientation` cache fast-path falls through to 3 probes on confidence miss

**Important — filed for next sprint:**
- `flec-uwk` — OCR block runs in all modes; should be gated on `Mode.READING`
- `flec-pxt` — `ThreadPoolExecutor` in `OCRReader` never shut down on session teardown
- `flec-70r` — `OnceWarner` instantiated but `warn_once` never called; `reading_ocr_unavailable` event silent
- `flec-p7t` — `_ocr_cached_orient` not cleared in `reset_reading_state()`
- `flec-2fb` — `_pending_illustration` accessed as private attr; needs `has_pending_illustration` property
- `flec-fmx` — RUNNING.md §4 troubleshooting directional errors in two rows

**Advisory (logged, not filed):**
- Async non-blocking OCR dispatch needed before ARM64 production (current `future.result()` blocks frame thread)
- `IllustrationDescriber.describe()` synchronous on hot path — latency risk when BLIP-2 available on ARM64

---

## Agent Team

| Agent | Role | Contribution |
|-------|------|-------------|
| Neo | Architect | Architecture & PRD conformance review |
| Echo | QA | Test coverage review |
| Phantom | Security | Security review (PASS — no blocking findings) |
| Jarvis | Tech Writer | Docs review + CHANGELOG draft |

---

## Reflect Notes

_(Filled during Reflect sub-phase in `/ship`)_
