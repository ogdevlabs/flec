# Party Review — reading-mode-end-to-end (F-001)

**Date:** 2026-07-11  
**Branch:** `feature/reading-mode-end-to-end`  
**Commits reviewed:** 11 (from `feat(flec-7al)` through `feat(flec-fpj)`)  
**Reviewers:** Neo (Architecture), Echo (QA), Phantom (Security), Jarvis (Docs)  
**MOM author:** Orchestrator  

---

## Verdict

**CONDITIONAL PASS** — 5 Critical/Important findings require resolution before merge. No blocking security vulnerabilities. Docs corrections needed.

---

## Cross-Talk Synthesis

Three independent reviewers (Neo, Echo, Jarvis) converged on the same two root issues from different angles:

**Cross-talk A — OnceWarner is dead code (Neo #5, Echo implicit, Jarvis #3)**
Neo flagged it as an Important architecture issue (AC-7 partially unmet). Jarvis flagged the doc/runtime mismatch (the troubleshooting table says `reading_ocr_unavailable` is emitted; it never is). Both confirm: `_ocr_once_warner` is instantiated but `warn_once` is never called in `process_frame`. The fix is a one-liner in the OCR low-confidence branch.

**Cross-talk B — OCR mode guard missing (Neo #1, Echo #5)**
Neo found it as a CPU/architecture correctness issue: OCR runs in all modes. Echo found the same gap from the test side: no integration test exercises non-READING mode with the guard. Both point to the same fix: gate the OCR block on `Mode.READING`.

**Cross-talk C — ARCHITECTURE.md drift (Neo #6, Jarvis #10)**
Both note independently that ARCHITECTURE.md describes a "background daemon thread" but the implementation runs inline on the frame thread. This is a doc-to-code drift, not a code bug — but the ARCHITECTURE.md needs a correction note.

---

## Findings by Severity

### Critical

| # | Reviewer | File | Issue |
|---|----------|------|-------|
| C-1 | Echo | — | **AC-6 (30fps) has zero test coverage.** No test verifies OCR runs off-thread or that `future.result(timeout=10)` doesn't block the perception loop. On ARM64, EasyOCR latency can be 300–1000ms. |
| C-2 | Echo | — | **AC-9 (no disk writes) has zero test coverage.** A debug `cv2.imwrite` added by any future contributor would not be caught. |
| C-3 | Neo | `ocr_worker.py:112–123` | **`resolve_orientation` cache fast-path logic bug.** When cached orientation returns `conf < conf_gate`, the function falls through and calls `read_region(crop)` twice more — 3 probes total instead of 2. Intent was single-probe optimisation once orientation is known. |

### Important

| # | Reviewer | File | Issue |
|---|----------|------|-------|
| I-1 | Neo | `main.py:227–249` | **OCR pipeline runs in all modes, not just READING.** EasyOCR fires on every settled fingertip during Exploration/Challenge/Story, burning CPU and calling `set_pending_illustration` in modes that discard it immediately. Fix: gate on `Mode.READING`. |
| I-2 | Neo | `ocr_reader.py:83`, `main.py:295` | **`ThreadPoolExecutor` never shut down.** `FlecSession.shutdown()` doesn't call `OCRReader.shutdown()`. On repeated test instantiation or long dev sessions, threads leak silently. |
| I-3 | Neo+Jarvis | `main.py:138–142` | **`OnceWarner` instantiated but never called.** The `reading_ocr_unavailable` event (documented in RUNNING.md troubleshooting) is never emitted. AC-7 dev signal is silent. Fix: one call in the low-confidence/no-result branch. |
| I-4 | Neo | `main.py:151`, `:301–304` | **`_ocr_cached_orient` not cleared on mode transition or finger reset.** `reset_reading_state()` resets `FingerTracker` but not `_ocr_cached_orient`. Stale orientation cache from a previous book position will mis-orient the first probe on a new position. Fix: add `self._ocr_cached_orient = None` to `reset_reading_state()`. |
| I-5 | Neo | `main.py:259` | **`_pending_illustration` accessed as private attribute across module boundary.** `process_frame` reads `self._response_engine._pending_illustration` to set `is_illustration`. Add `has_pending_illustration: bool` property to `ResponseEngine`. |
| I-6 | Jarvis | `RUNNING.md:124` | **RUNNING.md §4 troubleshooting directional guidance is backwards.** "lower = stricter settle required" is correct but placed under "Words not read" where raising is the fix. Misleads the troubleshooter. |
| I-7 | Jarvis | `RUNNING.md:126` | **Stale word / `FLEC_OCR_SETTLE_THRESHOLD` advice is wrong.** Lowering the threshold makes OCR fire less often; it does not prevent stale narration. The staleness mechanism is word-change flush, not the settle gate. |

### Advisory

| # | Reviewer | Issue |
|---|----------|-------|
| A-1 | Echo | AC-1 timing (2s budget) verified for correctness, not latency |
| A-2 | Echo | Mode guard in `_handle_finger` only tested for READING; non-READING path untested |
| A-3 | Echo | Illustration fallback uses fingertip crop; dark crops silently filtered as degenerate |
| A-4 | Echo | `resolve_orientation` cache fall-through path has no test |
| A-5 | Echo | `test_process_frame_legacy_ocr_result_param_still_works` doesn't assert safety gates bypassed |
| A-6 | Echo | Duplicate test suites between `test_reading_session.py` and `test_reading_mode.py` |
| A-7 | Neo | Confidence double-check in `process_frame` is redundant (caller re-checks what `resolve_orientation` already enforced) |
| A-8 | Neo | No structured log for `reset_reading_state` trigger |
| A-9 | Neo | `IllustrationDescriber.describe()` called synchronously on hot path — latency bomb on prod ARM64 when BLIP-2 is available |
| A-10 | Neo | `_strip_garbage` Unicode category filtering undocumented English-only assumption |
| A-11 | Jarvis | `process_frame` docstring doesn't describe OCR pipeline path |
| A-12 | Jarvis | ARCHITECTURE.md describes "background daemon thread"; implementation is inline — doc drift |
| A-13 | Phantom | `say` backend SSML: `[` and `]` pass through `_strip_garbage`; cheap to harden |
| A-14 | Phantom | `_run_ocr` DEBUG log emits raw text; inconsistent with `read_page` char_count-only pattern |

---

## Security

**Phantom verdict: PASS — No blocking security findings.**

The implementation is consistent with the zero-egress, zero-persistence, on-device security model. Two low-confidence advisories noted (say SSML hardening, debug log OCR text) but neither rises to a vulnerability.

---

## Phantom Sign-off

Security review: **SIGNED OFF**. No Critical or Important security findings. Advisories A-13 and A-14 are noted for hardening in a follow-up.

---

## Pre-Merge Action Items

Before merging `feature/reading-mode-end-to-end` → `phase/1-setup`:

**Required (Critical):**
- [ ] Add test for AC-6: mock OCR timeout, assert `process_frame` returns within deadline
- [ ] Add test for AC-9: assert no new files created in CWD/tmp after `process_frame`
- [ ] Fix `resolve_orientation` cache fast-path: return early on confidence miss instead of double-probing

**Required (Important):**
- [ ] Gate OCR block in `process_frame` on `Mode.READING`
- [ ] Add `OCRReader.shutdown()` and wire into `FlecSession.shutdown()`
- [ ] Call `self._ocr_once_warner.warn_once(...)` in the no-word branch (activate AC-7 signal)
- [ ] Add `self._ocr_cached_orient = None` to `reset_reading_state()`
- [ ] Add `has_pending_illustration` property to `ResponseEngine`
- [ ] Fix RUNNING.md §4 troubleshooting directional errors (I-6, I-7)

**Deferred (Advisory):**
- Async non-blocking OCR dispatch (A-9 — needed before ARM64 production)
- Test: non-READING mode guard path (A-2)
- ARCHITECTURE.md sync to reflect inline-vs-thread design (A-12)

---

## CHANGELOG Entry (Jarvis draft)

```markdown
## [Unreleased] — 2026-07-11

### Added
- **Reading mode end-to-end (F-001):** pointing a steady fingertip at a word causes Flec
  to speak it aloud. Covers the full pipeline from fingertip detection through OCR to audio
  output.
- `reading/ocr_worker.py`: settle-gated OCR helpers (`should_run_ocr`,
  `crop_around_fingertip`, `resolve_orientation`, `mirror_x`) and `OnceWarner` for
  one-time dev warnings.
- `OCRReader.read_region`: confidence-gated word extraction with orientation resolution.
- Auto-orientation: EasyOCR probed in both native and mirrored orientations; higher
  confidence wins. Orientation cached per session to avoid redundant re-probing.
- Confidence silence gate (`FLEC_OCR_CONF_GATE`, default `0.4`).
- Illustration fallback: BLIP-2 describes the fingertip region when no confident word
  is found; silent no-op when unavailable.
- Word-change flush: pending TTS cleared when the pointed word changes.
- Dev wear-state override (`FLEC_READING_WEAR_OVERRIDE`, default `1`).
- `docs/RUNNING.md` §4 Reading mode how-to, quick-test command, troubleshooting table.
- New env vars: `FLEC_READING_VELOCITY_THRESHOLD`, `FLEC_READING_FRAMES`,
  `FLEC_OCR_SETTLE_THRESHOLD`, `FLEC_OCR_CONF_GATE`, `FLEC_READING_WEAR_OVERRIDE`.
```
