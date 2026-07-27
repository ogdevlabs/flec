---
feature: reading-mode-end-to-end
date: 2026-07-10
status: inception-complete
last-updated: 2026-07-11T05:22:36Z
approved-by: Oscar Paul Garcia
approved-date: 2026-07-11T05:22:36Z
prd: docs/pdlc/prds/PRD_reading-mode-end-to-end_2026-07-10.md
---

# Brainstorm Log: Reading Mode — End-to-End (finger → OCR → speak)

> **Scope reframed 2026-07-11** (was `reading-mode-mirrored-text`). Socratic Q1
> revealed the live symptom is "finger not detected, no audio" — and the code
> shows Reading mode's OCR is not wired into the live loop at all
> (`main.py` calls `process_frame(..., ocr_result=None)`; no OCRReader runs), so
> `nearest_text` is always empty and `_handle_finger` can never speak. Mirror /
> orientation correction is a necessary *sub-part*, downstream of wiring the
> pipeline. Feature now = make Reading mode work end-to-end.

## Divergent Ideation

**Technique:** Structured Domain Rotation (option B)
**Total ideas generated:** 105
**Completed:** 2026-07-10T21:24:56Z

### Raw Ideas

_Technical (1–10)_
1. Horizontal-flip the frame before OCR (`cv2.flip`, axis 1) in Reading mode.
2. Run EasyOCR on raw + mirrored; keep whichever has higher mean confidence.
3. Try 4 rotations × mirror (8 variants); pick the max-confidence result.
4. Detect mirror by comparing dictionary-hit rate of raw vs flipped OCR.
5. Use EasyOCR per-box confidence to auto-select the correct orientation.
6. Add a `FLEC_CAMERA_MIRROR` config flag applied per device.
7. Auto-detect integrated/front cameras (index 0) and default them to mirrored.
8. Cache the resolved orientation per session after first successful read.
9. Tesseract OSD (orientation & script detection) as a lightweight rotation pre-pass.
10. Downscale for a fast orientation probe, then OCR full-res at the chosen orientation.

_User Experience (11–20)_
11. Audio cue when no readable text ("Turn the page toward me!").
12. "Hold it steady" prompt when the page is moving/unstable.
13. Confidence-gated narration — read only when confident; else stay silent (no gibberish).
14. Narrate a word only after it's stable across N frames (debounce, like objects).
15. Dev preview overlay: detected text boxes + orientation arrow.
16. Dev badge "MIRROR CORRECTED" when a flip was applied.
17. Calibration gesture: point at a known word to auto-learn orientation.
18. Distinct reading-mode voice tone so the child knows the mode.
19. If no text for a while: "Point at a word with your finger!".
20. Read in the child's finger-scan direction (left-to-right tracking).

_Business & Viability (21–30)_
21. Headline "reading accuracy" in the parent-facing pitch.
22. Multi-language later (EasyOCR) — orientation fix is language-agnostic groundwork.
23. "Works with any book, held any way" as a robustness selling point.
24. Reduce frustration/returns — misreads are the #1 trust-killer for a reading toy.
25. Log an on-device "reading confidence" metric for QA.
26. Recommended-books list tuned to the camera's reading range.
27. Differentiator vs screen apps: reads the real physical book the child holds.
28. Ship a synthetic mirrored-text test corpus (no child data).
29. "Parent read-along" variant where the caregiver holds the book.
30. Reinforce "no image ever leaves the device" in reading, too.

_Edge Cases & Failure Modes (31–40)_
31. Upside-down book (180°), not just mirror.
32. Sideways book (90°) — OSD rotation.
33. Partial text at frame edge — don't narrate fragments.
34. Glare/reflection on glossy pages.
35. Curved page near the spine — text distortion.
36. Two pages visible — read the one nearest the fingertip.
37. Handwriting vs print — confidence-gate (EasyOCR weaker on handwriting).
38. Mixed mirror: front cam mirrored, iPhone Continuity not — per-device.
39. Small/far text — resolution floor; "bring it closer".
40. Fast page flip — stale OCR narrated after the turn (queue-flush link).

_Technical, 2nd pass (41–50)_
41. Put orientation resolution inside `ocr_reader` preprocessing (module-local, no cross-import).
42. Expose OCR confidence in DetectionEvent metadata for engine gating.
43. Run the expensive 8-variant probe only on mode-entry or when confidence drops.
44. Crop the OCR ROI around the MediaPipe fingertip location.
45. Temporal voting: accumulate OCR across frames; narrate the majority word.
46. Perspective-warp via detected page quad (`getPerspectiveTransform`).
47. Binarize/threshold before OCR for contrast.
48. Frame-diff for page stability before spending an OCR pass.
49. Batch the orientation variants in one EasyOCR/MPS call.
50. Persist per-device mirror default in `.env` (`FLEC_CAMERA_MIRROR=auto/on/off`).

_User Experience, 2nd pass (51–60)_
51. "Found it!" chime before reading a word.
52. Read syllable-by-syllable for early readers.
53. Repeat-on-request ("say it again"), reusing the challenge repeat intent.
54. Spell it out ("C-A-T, cat!").
55. When mirror is detected, silently correct — never tell the child the book is "backwards".
56. Dev HUD: raw vs corrected thumbnail side by side.
57. Slower, clearer TTS cadence in reading vs upbeat exploration.
58. One word at a time to avoid overwhelming the toddler.
59. Encourage tracing: "Move your finger along the word".
60. "You're reading!" celebration after a few words.

_Business & Viability, 2nd pass (61–70)_
61. On-device reading-accuracy telemetry to guide model choice.
62. Swappable OCR model via `FLEC_OCR_MODEL` (mirrors `FLEC_YOLO_MODEL`).
63. Multi-language content packs.
64. Partner with publishers for camera-friendly print.
65. "Literacy pillar" framing for education grants/funding.
66. Accessibility angle: helps low-vision kids hear text.
67. Words-read-per-session KPI (fills the INTENT success-metrics gap).
68. On-device parent report of words encountered (no cloud).
69. Robust reading → fewer support tickets → lower cost.
70. "Reads any orientation" retail demo wow-moment.

_Edge Cases, 2nd pass (71–80)_
71. Blank page → stay silent, don't hallucinate.
72. Non-text illustration in reading mode → route to illustration describer, not OCR garbage.
73. Mirror + upside-down at once (180° + flip).
74. Camera flips mid-session (Continuity handoff) → re-detect orientation.
75. Wear-off during reading → suspend (already gated).
76. Mode switch mid-read → flush queued word narration (verification item).
77. Multiple hands in frame → pick the fingertip-owner's ROI.
78. Backlit page (window behind) → exposure correction.
79. Confident garbage on mirror text → dictionary/lexicon sanity check.
80. Long paragraph → read only the pointed word, not the whole block.

_Analogies & Adjacent Domains (81–90)_
81. Document scanners auto-rotate via OSD — steal the pre-pass.
82. QR readers brute-force all orientations instantly — same small-variant trick.
83. Selfie cams mirror the *preview* but save the *raw* — mirror preview, OCR raw.
84. Teleprompters read mirrored text through glass — model mask optics if a mirror is in the path.
85. Da Vinci mirror-writing — flipping to decode is the exact operation.
86. Ambulance reversed lettering — mirroring is a known, fully reversible transform.
87. Dyslexia reading aids reflow/focus text — confidence-gate + one-word focus mirrors their UX.
88. POS scanner "beep" on success — our "found it" chime.
89. Live-camera translation overlays corrected text — our audio-only equivalent.
90. Robot vacuums map once then cache — cache orientation once per session.

_Wild & Combinatorial (91–105)_
91. (Worst→invert) Don't flip audio to match mirrored text — flip the frame.
92. Train a tiny one-shot classifier that outputs mirror/rotation instead of brute force.
93. Physical prism/mirror in the mask so software never sees mirrored text (hardware track).
94. Infer page orientation from the child's hand pose ("show me the words").
95. Detect text-baseline slope to infer rotation angle continuously.
96. Anchor orientation using the page-number position.
97. (Worst→invert) Narrating every OCR attempt incl. garbage → strict confidence gate + silence.
98. (Exaggerate) Assume every integrated webcam is mirrored; flip index-0 by default, verify empirically.
99. If YOLO sees a "book," trigger orientation calibration.
100. Reverse the OCR string; if the reversal is a dictionary word, it was mirrored.
101. 3-second parent calibration card at first run locks device orientation.
102. Lazy correction: only auto-mirror when the raw-frame dictionary test fails.
103. Prefer the iPhone Continuity Camera (non-mirrored) for reading when available.
104. A "reading reticle" the child centers the word in (dev/parent aid).
105. Printable "Flec reading card" whose known text auto-calibrates orientation on boot.

### Clusters

- **A. Orientation detection & correction (core):** #1, #2, #3, #5, #9, #10, #31, #32, #41, #43, #46, #49, #73, #81, #82, #85, #86, #92, #95, #96, #98, #100
- **B. Confidence-gating & anti-gibberish:** #4, #13, #37, #47, #71, #72, #79, #80, #97, #102
- **C. Fingertip-guided ROI & stability:** #14, #20, #33, #36, #44, #45, #48, #77
- **D. Child-facing reading UX:** #11, #12, #17, #18, #19, #51, #52, #53, #54, #55, #57, #58, #59, #60, #87, #89, #104
- **E. Camera/device config & calibration:** #6, #7, #8, #38, #50, #74, #83, #84, #93, #101, #103, #105
- **F. Product, robustness & extensibility:** #21–#30, #34, #35, #39, #61–#70, #78

### Standouts

1. **Confidence-select raw vs mirrored (#2/#4)** — the simplest robust core fix; OCR both, keep the higher-confidence read.
2. **Small-variant brute force (#3/#82)** — flip + 4 rotations, pick max confidence; also solves upside-down/sideways, not just mirror.
3. **Confidence + dictionary gate (#13/#79)** — kills gibberish; directly fixes the "inaccurate speech" complaint too.
4. **Cache/persist resolved orientation (#8/#50)** — resolve once, avoid a per-frame 8× OCR cost; embedded-friendly.
5. **Default-flip integrated (index-0) cameras (#7/#98)** — pragmatic default matching the reported symptom; verify empirically.
6. **Fingertip-guided OCR ROI (#44)** — read what the child points at; less noise, faster, on-brand for reading mode.
7. **Temporal voting / debounce (#14/#45)** — stable words, mirrors the existing object stabilizer pattern.
8. **Prefer iPhone Continuity for reading (#103)** — non-mirrored, higher-res; sidesteps the problem when available.
9. **Flush queued word narration on mode switch (#76)** — the user's second concern; verify the existing `clear_pending` covers it.
10. **Route non-text to illustration describer (#72)** — avoids OCR garbage on pictures in reading/story.
11. **Silent correction UX (#55)** — never tell the child the book is backwards; correct invisibly.
12. **OSD rotation pre-pass (#9/#81)** — scanner-style orientation detection instead of brute force.
13. **Printable calibration card (#101/#105)** — known text auto-locks device orientation at first run.
14. **Hardware prism (#93)** — long-term: fix it in the optics so software never sees mirrored text.

## Socratic Discovery

**Completed:** 2026-07-11T03:25:00Z
**Interaction mode:** Socratic

### Round 1 — Problem Statement

**Q1:** In Reading mode with the integrated webcam, what happens on the audio side, and how often does it read correctly vs fail?
**A:** "Finger was not detected, no audio output." → Reframed: the live symptom is upstream of OCR. Code review confirms Reading mode's OCR is not wired into the live loop (`main.py` passes `ocr_result=None`; no OCRReader runs), so `nearest_text` is always empty and `_handle_finger` can never speak — regardless of finger detection or mirroring. Feature reframed from "mirrored-text" to **end-to-end Reading mode**.

**Q2 (who/context):** Not re-asked — covered by INTENT.md (primary: curious toddler ~2–4 pointing at words in a physical book; secondary: caregiver/dev). Dev surface is the integrated webcam, which is mirrored.

### Round 2 — Future State / Key Capabilities

**Q:** When a child points at a page, what should Flec read?
**A:** **A — the single word under the fingertip.** Flec crops OCR to the region around the detected fingertip and speaks only that word. Pointing is the intentional "read this" gesture. Bounds OCR cost (ARM64 target) and avoids reading a paragraph at a screen-free toddler. Hard-depends on finger detection working (the reported failure).

### Round 3 — Acceptance Criteria

**Q:** What bar proves it works, and how is orientation handled?
**A:** **A — auto-orientation, correctness-gated.** On the integrated (mirrored) webcam, a child holds a printed word under their fingertip → Flec speaks the correct word within ~2s; silent (no gibberish) when there's no clear word. Orientation is **auto-detected**: OCR the fingertip crop normal + mirrored, keep the higher-confidence read, cache the winning orientation per session (standouts #2/#4/#8). No config flag required.

## Progressive Thinking (Agent Team Meeting)

**MOM:** `docs/pdlc/mom/reading-mode-end-to-end_progressive-thinking_mom_2026_07_10.md`

### Confirmed Facts
- Reading narration exists (`_handle_finger`) but OCR is **unwired** in the live loop (`process_frame(ocr_result=None)`, no OCRReader) → `nearest_text` always empty → no audio today.
- Building blocks present: `OCRReader` (EasyOCR, models cached), `FingerTracker.update_ocr(text_regions=…)`, queue-only orchestration in `main.py`.
- Capability = single word under the fingertip (user choice A). On-device, zero-persistence, ARM64 target.

### Accepted Inferences
- Mirror unlikely to break finger detection (framing/lighting/distance more likely).
- OCR must be throttled + cropped + downscaled (30 fps on ARM64 can't OCR every frame).
- EasyOCR bbox+confidence supports nearest-word selection and a "silent when unsure" gate.
- Auto-orientation = OCR crop normal vs horizontally-mirrored, keep higher confidence, cache per session.

### Key Consequences
- Wire `OCRReader` on a throttled background worker thread → feed regions to `FingerTracker.update_ocr` (keeps the 30 fps loop responsive).
- New tests: OCR-orientation contract (normal + synthetic mirrored) + Reading end-to-end integration (stubbed OCR).
- Docs: RUNNING.md Reading section + docstrings. No new security/infra surface.

### Risks & Unknowns
1. **Fingertip-detection reliability at reading distance** — linchpin for model A (this is what actually failed for the user).
2. OCR latency vs the ~2s budget on ARM64.
3. Confident-wrong reads on mirrored text → gate on the normal/mirror confidence **delta**, not an absolute floor.
4. Fingertip x-coordinate must be mirrored to match a flipped OCR crop (coordinate-space bug risk).
5. Crop-size sensitivity (miss the word vs read neighbors).
6. Multi-hand ROI ownership.

### Conflicts Resolved
- **Where OCR runs:** background worker thread (throttled), not inline — avoids stalling the loop (Neo). No user escalation needed.
- **Orientation breadth:** v1 = horizontal-mirror only; 90°/180° rotation deferred.

### Design Priorities
1. Fingertip-detection reliability (investigate/mitigate first).
2. Wire OCRReader on a throttled worker thread → `FingerTracker.update_ocr`.
3. Auto horizontal-mirror orientation (confidence-select, cached per session).
4. Correctness gate (confidence + normal/mirror delta) → silent when unsure.
5. Bound OCR cost (crop, downscale, throttle, cache).

## Adversarial Review

**Completed:** 2026-07-11T03:35:00Z

### Findings
1. Finger-detection root cause unknown — model A depends 100% on it. **[RESOLVED: finger IS detected/highlighted; failure is OCR, not tracking.]**
2. "~2s" bar unvalidated on ARM64 (set on dev Mac + MPS).
3. Confidence-gate calibration hand-wavy; EasyOCR confidence poorly calibrated.
4. Mirror detection foolable — short words yield plausible mirrored tokens; delta may be tiny.
5. Fingertip→word association unspecified (coord space; off-by-mirror bug on flipped crop).
6. Reading-INTENT gate is a second silent-failure point (still→IDLE, moving→SCANNING).
7. OCR worker-thread staleness/races — word arrives after child moved off it.
8. Dev vs prod camera divergence (integrated mirrored webcam vs mask camera). **[RESOLVED at team level: auto-orientation makes pipeline camera-agnostic; validate on the integrated webcam we have.]**
9. No text size/distance floor — tiny/distant text fails silently.
10. Illustration-vs-text ambiguity. **[RESOLVED: v1 falls back to illustration describer when no word — user choice B.]**
11. Accuracy unmeasurable in-field (zero persistence); only words/session observable.
12. Wear-state gate — dev webcam has no wear sensor → may suppress reading.
13. Two-variant OCR doubles per-pass cost (compounds latency, #2).

### Follow-up Q&A
**Q1 (finger detection):** In the preview, was the finger visible/highlighted, and how far was the book?
**A1:** "Finger is highlighted, words are not being identified." → Finger tracking works; confirmed the blocker is OCR (unwired + mirror), not detection. Finding #1 resolved; linchpin risk downgraded.

**Q2 (illustration vs text scope):** When the child points at a picture, what should v1 do?
**A2:** **B — describe the picture too.** OCR the crop; if no confident word, route the region to the illustration describer. Known risk: BLIP-2 is MPS-degraded in dev (graceful no-op locally; active on ARM64 target).

_Follow-up 3 not spent — finding #8 resolved at team level (auto-orientation → camera-agnostic). Remaining findings (#2–7, #9, #11–13) feed Edge Case Analysis and the PRD known-risks section._

## Edge Case Analysis

**Completed:** 2026-07-11T03:40:00Z

### Findings

| # | Category | Scenario | Trigger | Addressed? | Risk if unhandled |
|---|----------|----------|---------|-----------|-------------------|
| 1 | Concurrency/timing | OCR word arrives after finger moved to another word/page | OCR latency > finger movement | Partial | Reads stale/wrong word |
| 2 | User-flow | Finger held still → `IDLE` intent → no narration | Child points and holds | Partial | Silent despite correct point |
| 3 | User-flow | Finger moving → `SCANNING` → no narration | Child sweeps finger | Partial | Never reads while browsing |
| 4 | Boundary | Finger between two words / word at crop edge | Ambiguous fingertip position | No | Partial/wrong word |
| 5 | Invalid input | Mirror yields confident wrong token | Mirror + short word | Partial | Confidently wrong word |
| 6 | Scale/load | Text too small/distant for OCR | Book at arm's length | No | Silent failure |
| 7 | Timing | Page turned mid-OCR pass | Fast page flip | No | Reads previous page |
| 8 | Concurrency | Two hands/fingers in frame | Adult + child | No | Ambiguous ROI |
| 9 | Integration | EasyOCR model missing/unloaded | Models not downloaded | Partial | Reading silently dead |
| 10 | Integration | BLIP-2 unavailable on MPS (picture fallback) | Point at picture in dev | Partial | Picture-point no-ops in dev |
| 11 | Boundary | Wear `OFF_HEAD` in dev suppresses reading | Dev webcam has no wear sensor | No | Reading never activates in dev |
| 12 | Environment | Glare/low-light on glossy page | Overhead reflection | Partial | OCR fails silently |

### Triage Decisions

| # | Decision | Notes (→ PRD) |
|---|----------|---------------|
| 1 | In scope | Flush pending reading word when the fingertip's target word changes / finger moves — extends the mode-switch flush to intra-reading. |
| 2 | In scope | Tune reading-intent so a steady, deliberate point registers as READING (not IDLE). |
| 3 | In scope | Ensure a slow scan-then-settle reads; fast sweeps stay silent (browsing). |
| 5 | In scope | Orientation decided by normal-vs-mirror confidence **delta**, not absolute floor. |
| 9 | In scope | When EasyOCR is unavailable, log a clear structured warning (dev signal); reading degrades to silent. |
| 11 | In scope | In dev, treat the webcam as `ON_HEAD` so reading activates without a wear sensor. |
| 4 | Known risk | Crop-edge / between-words word — accept imperfect selection for v1. |
| 6 | Known risk | Tiny/distant text — defer a "bring it closer" cue. |
| 7 | Known risk | Page-flip staleness — partially mitigated by #1; accept residual. |
| 10 | Known risk | BLIP-2 MPS-degraded in dev — graceful no-op; active on ARM64. |
| 12 | Known risk | Glare/low-light — existing low-light detector only; defer specular handling. |
| 8 | Out of scope | Multi-hand ROI — v1 uses a single primary fingertip. |

## UX Discovery
_Skipped: non-visual feature (backend CV/OCR); no visual companion. Only surface is a dev HUD badge._

## External Context
_None ingested._

## Design Discovery (Bloom's Taxonomy)

### Round 1 — Mechanics
Skipped as clear from PRD/MOM: finger→crop→OCR(normal+mirror)→word-or-illustration→speak; OCR on a throttled worker thread feeding `FingerTracker.update_ocr`; orientation by confidence-delta, cached per session.

### Round 2 — Apply
Skipped as clear: EasyOCR (`reading/ocr_reader.py`) for text; BLIP-2 (`reading/illustration_describer.py`) for the picture fallback; MediaPipe fingertip via `perception/finger_tracker.py`; narration via `engine/response_engine.py` `_handle_finger`; queue-only orchestration in `main.py`/`FlecSession`.

### Round 3 — Trade-offs and Judgments
**Q1 (OCR trigger cadence):** run OCR continuously while a finger is present, or only when it settles?
**A1:** **Settle-gated** — OCR fires only when the fingertip is detected and stable (low velocity ~300–500 ms). Bounds cost, doubles as the READING-intent signal, keeps the 30 fps loop free. Accepts a ~½s settle delay (within the ~2s budget).

**Q2 (uncertainty handling):** best-guess or silence when orientation/word confidence is low?
**A2:** **Silence over best-guess** — if the normal/mirror confidence delta is small or top confidence is below the gate, say nothing rather than speak a wrong word to a toddler. Correctness over coverage.

### Synthesis
Neo's sketch: a settle-gated OCR worker thread crops around the fingertip, OCRs normal + horizontally-mirrored, picks the higher-confidence read (delta-gated), caches the winning orientation, and feeds text regions to `FingerTracker.update_ocr`; `_handle_finger` narrates the nearest word, or falls back to the illustration describer when no confident word exists; silence otherwise. User validated: **both trade-offs accepted, no pushback.**

## Discovery Summary

**Confirmed by Oscar Paul Garcia, 2026-07-11.**

**Feature:** Reading Mode — End-to-End (finger → OCR → speak)

**Problem:** Reading mode produces no audio. The fingertip is detected (highlighted), but OCR is never wired into the live loop (`process_frame(ocr_result=None)`, no OCRReader), so `nearest_text` is always empty and nothing is spoken. Once wired, the integrated webcam's mirrored feed further breaks recognition.

**User:** Curious toddler (~2–4) pointing at a word in a physical book; caregiver/dev on the integrated (mirrored) webcam.

**Success metric:** On the integrated (mirrored) webcam, pointing at a printed word → correct word spoken within ~2s; silent (no gibberish) when no confident word; if a picture (no word), describe the region.

**Technical approach / constraints:**
- Single word under the fingertip — OCR cropped to the fingertip ROI.
- Auto horizontal-mirror orientation — OCR normal + mirrored, pick higher confidence via delta, cache per session.
- OCR on a throttled background worker thread → `FingerTracker.update_ocr` (keeps 30 fps loop responsive).
- Picture fallback → illustration describer (BLIP-2; MPS-degraded in dev, active on ARM64).
- On-device, zero-persistence, ARM64 (<10s boot); queue-only architecture; EasyOCR (cached).

**In-scope acceptance items:** wire finger→OCR→speak; flush pending word when target word changes; steady point reads as READING; mirror confidence-delta gate; clear warning when OCR unavailable; dev treats webcam as `ON_HEAD`.

**Out of scope:** 90°/180° rotation; full-page OCR; multi-language; multi-hand ROI; hardware prism.

**Key risks / assumptions:** OCR latency vs ~2s on ARM64; EasyOCR confidence calibration; mirror delta small for short words; fingertip-coordinate mirroring bug on flipped crop; reading-intent thresholds; stale-read races; BLIP-2 MPS-degraded in dev; field accuracy unmeasurable.

---

## Seed context (from the /brainstorm request)

Original request (verbatim):

> "when use reading mode, be aware that possible computer integrated camera, is
> observing text in inverse mode, and is not able to identify text, also make
> sure that when switching modes, remove exploration items from the queue"

**Scope decision (confirmed with user, 2026-07-10):** feature is focused on the
**mirrored/inverse-text OCR problem in Reading mode**. The second concern —
removing exploration items from the queue on a mode switch — is already
implemented in commit `a60d33d` (`ResponseEngine.set_mode()` →
`TTSEngine.clear_pending()`). Carried here as a **verification item**, not new scope.

**Pre-flight:** Beads unavailable (no db) → legacy single-dev mode. Local main
rebased onto origin/main (PR #16 merge); low conflict risk. Interaction mode: Socratic.
