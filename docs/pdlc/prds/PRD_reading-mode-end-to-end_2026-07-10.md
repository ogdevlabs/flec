# PRD: Reading Mode — End-to-End (finger → OCR → speak)
<!-- pdlc-template-version: 2.1.0 -->

**Date:** 2026-07-10
**Status:** Approved
**Feature slug:** reading-mode-end-to-end
**Episode:** <!-- assigned after delivery -->

---

## Overview

Reading mode is one of Flec's four child-facing modes, but it currently produces no
audio: the fingertip is tracked, yet OCR is never run in the live loop, so there is
nothing to read aloud. This feature wires Reading end-to-end — detect the fingertip,
OCR the word the child points at (correcting the dev webcam's mirrored feed), and
speak it — delivering on the INTENT goal of narrating the words a toddler encounters
in the real world, hands-free and screen-free.

---

## Problem Statement

In Reading mode today, a child can point at a word and see the fingertip highlighted,
but Flec never speaks. The cause is structural: `main.py` calls
`process_frame(frame, ocr_result=None)` and instantiates no `OCRReader`, so
`FingerTracker`'s `nearest_text` is always empty and `_handle_finger` can never
narrate — regardless of finger detection. Compounding this, the integrated
development webcam delivers a horizontally **mirrored** feed, so even once OCR is
wired, mirror-image text would fail to recognize. The result: the reading capability
is inert, and the child gets no spoken feedback when they point at words.

---

## Target User

Primary: the **Curious Toddler (~2–4)** from INTENT.md — pre/early-literate, learning
first words, who points at words in a physical book and expects encouraging spoken
feedback. Secondary: the **caregiver/developer** operating Flec in dev on a laptop's
integrated (mirrored) webcam.

---

## Requirements

1. The system MUST run OCR in the live Reading loop: when a fingertip is detected, crop the frame to a region around the fingertip, OCR that crop, and speak the recognized word.
2. The system MUST auto-correct horizontal mirroring: OCR the crop both as-captured and horizontally flipped, select the orientation with the higher recognition confidence (decided by the confidence **delta**, not an absolute floor), and cache the winning orientation for the session.
3. The system MUST gate narration on confidence: speak a word only when recognition confidence clears the gate; otherwise stay silent (no gibberish).
4. The system MUST fall back to illustration description: when no confident word is under the fingertip but the region is describable, route the region to the illustration describer and speak the description.
5. The system MUST run OCR on a throttled background worker thread so the 30 fps perception/preview loop is not blocked; OCR SHOULD run only when a fingertip is detected and reasonably stable.
6. The system MUST flush any pending/queued reading narration when the fingertip's target word changes, so a word the child has moved off is not spoken afterward.
7. The system MUST register a deliberate, steady point as READING intent (holding still on a word narrates it), while a fast sweep stays silent (browsing).
8. The system MUST map the fingertip coordinate into the same orientation as the OCR'd crop (mirror the x-coordinate when the mirrored orientation is chosen).
9. The system SHOULD log a clear structured warning when the OCR model is unavailable, and degrade to silent reading without crashing.
10. In dev (integrated webcam, no wear sensor), the system MUST treat wear-state as `ON_HEAD` so Reading mode activates.
11. The system MUST NOT persist frames, crops, audio, or recognized text.

---

## Assumptions

- Fingertip detection (MediaPipe) works at reading distance — confirmed in dev (the fingertip is highlighted in the preview).
- EasyOCR returns per-region text with a bounding box and a confidence score, usable for nearest-word selection and the silence gate.
- The integrated dev webcam feed is horizontally mirrored; the production mask camera is not — auto-orientation keeps the pipeline camera-agnostic.
- The child points at one word at a time (a single primary fingertip).
- EasyOCR models are downloaded/cached at `.models/easyocr`.

---

## Acceptance Criteria

1. On the integrated (mirrored) webcam, pointing at a clearly printed word results in the correct word being spoken within ~2s. *(R1, R2)*
2. When no confident word is under the fingertip, nothing is spoken — no gibberish. *(R3)*
3. Pointing at a picture (no word) speaks a short description of that region where the describer is available; where BLIP-2 is unavailable it is a silent no-op with a logged warning. *(R4)*
4. Moving the fingertip to a different word replaces narration with the new word; the previous, no-longer-pointed word is not spoken after the move. *(R6)*
5. Holding the fingertip steady on a word (~1s) narrates it (registers as READING, not IDLE); a fast sweep across the line does not narrate. *(R7)*
6. The 30 fps perception/preview loop stays responsive while OCR runs (no visible stall). *(R5)*
7. With the OCR model absent, Reading mode does not crash, logs a clear warning, and stays silent. *(R9)*
8. In a dev run on the integrated webcam, Reading mode activates without a physical wear sensor. *(R10)*
9. No frames, crops, or recognized text are written to disk. *(R11)*

---

## User Stories

**US-001: Read the word under the fingertip**
*Acceptance criteria: 1, 5*
Given the child is in Reading mode holding a book to the (mirrored) webcam
When they rest their fingertip steadily under a printed word
Then Flec speaks that word correctly within ~2s

**US-002: Silence instead of gibberish**
*Acceptance criteria: 2*
Given the fingertip is over whitespace or unreadable text
When OCR finds no confident word
Then Flec stays silent

**US-003: Picture fallback**
*Acceptance criteria: 3*
Given the child points at an illustration rather than a word
When no confident word is found under the fingertip
Then Flec describes that region (where the describer is available) or stays silently no-op with a logged warning (where it is not)

**US-004: Word-change flush**
*Acceptance criteria: 4*
Given Flec has just read (or is about to read) a word
When the child moves their fingertip to a different word
Then any pending narration of the previous word is dropped and the new word is read instead

**US-005: Graceful OCR-unavailable**
*Acceptance criteria: 7*
Given the EasyOCR model is not installed/loaded
When the child points at a word in Reading mode
Then Flec logs a clear warning, does not crash, and simply stays silent

---

## Non-Functional Requirements

- The point→speak round-trip SHOULD target ~2s on the ARM64 embedded device; OCR MUST be cropped, downscaled, throttled, and orientation-cached to meet it.
- On-device only: no network egress; zero persistence of frames, crops, audio, or recognized text (privacy-by-design for minors).
- No error message or text ever reaches the child — all feedback is audio; every failure path degrades to silence.
- The queue-only architecture MUST be preserved: no cross-module imports; OCR feeds the pipeline via `FingerTracker.update_ocr` / the event queue.
- Every new dependency (EasyOCR, BLIP-2) MUST degrade gracefully when unavailable (log a warning, no-op).

---

## Out of Scope

- 90° / 180° rotation correction — v1 handles horizontal mirror only (the dev-webcam reality).
- Full-page OCR / reading multiple words, lines, or paragraphs.
- Multi-language OCR.
- Multi-hand ROI ownership — v1 uses a single primary fingertip (edge case #8, triaged out).
- Hardware prism / optical mirror correction in the mask.

---

## Known Risks

- **Crop-edge / between-words selection** (edge #4): a fingertip between two words may read the wrong one. Accepted for v1 — refine ROI heuristics later.
- **Tiny / distant text** (edge #6): text below OCR resolution fails silently. Deferred: a "bring it closer" cue is a follow-up.
- **Page-flip staleness** (edge #7): a fast page turn mid-OCR could read the previous page. Partially mitigated by the word-change flush (R6); residual accepted.
- **BLIP-2 MPS-degraded in dev** (edge #10): picture fallback no-ops locally, active on the ARM64 target. Accepted per the graceful-degradation contract.
- **Glare / low-light on glossy pages** (edge #12): only the existing low-light detector applies; specular handling deferred.
- **OCR latency vs ~2s on ARM64** (adversarial #2): mitigated by crop + downscale + throttle + orientation cache; validate on target.
- **Mirror confidence-delta may be small for short words** (adversarial #4/#5): risk of wrong-orientation reads; monitor and tune.
- **EasyOCR confidence calibration** (adversarial #3): the silence-gate threshold needs tuning to balance false silence vs gibberish.
- **Field accuracy unmeasurable** (adversarial #11): zero persistence means no in-field ground truth; only words/session is observable.

---

## Design Docs

<!-- Auto-populated after the Design sub-phase. -->

- Architecture: [ARCHITECTURE.md](../design/reading-mode-end-to-end/ARCHITECTURE.md)
- Data model: [data-model.md](../design/reading-mode-end-to-end/data-model.md) — no persistence
- API contracts: [api-contracts.md](../design/reading-mode-end-to-end/api-contracts.md) — internal module interfaces (no network API)
- Threat model: [threat-model.md](../design/reading-mode-end-to-end/threat-model.md) — triage: **Lite**
- UX review: [ux-review.md](../design/reading-mode-end-to-end/ux-review.md) — triage: **Skip** (non-visual)

---

## Related Episodes

<!-- None yet — this is F-001. -->

---

## Approval

**Approved by:** Oscar Paul Garcia
**Date approved:** 2026-07-10
**Notes:** Approved as drafted. Scope reframed from "mirrored-text" to end-to-end Reading during Discover.
