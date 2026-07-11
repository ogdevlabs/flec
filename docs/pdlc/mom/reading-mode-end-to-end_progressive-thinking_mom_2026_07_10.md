# MOM — Progressive Thinking: reading-mode-end-to-end

**Called by:** Atlas (Product Manager)
**Participants:** Neo, Echo, Phantom, Bolt, Friday, Muse, Pulse, Jarvis (Atlas facilitates)
**Date:** 2026-07-10
**Purpose:** Progressive-thinking refinement of Reading-mode end-to-end discovery before Define.

---

## Discussion

### Round 1 — Concrete (facts)
- **Neo/Bolt:** `_handle_finger` (response_engine.py) narrates only when `mode==READING`, `intent==READING`, and `nearest_text` is set. OCR is **not wired** in the live loop — `main.py` calls `process_frame(frame, ocr_result=None)` and instantiates no `OCRReader`. So `nearest_text` is always empty; Reading mode cannot speak today.
- **Bolt:** `OCRReader` exists (`reading/ocr_reader.py`, EasyOCR); EasyOCR models cached in `.models/easyocr`. `FingerTracker.update_ocr(text_regions=...)` is the documented path to populate `nearest_text`.
- **Neo:** Queue-only contract — capability modules never import each other; `main.py` orchestrates. OCR→FingerTracker handoff already sketched in `main.py` docstring.
- **Muse:** Confirmed capability = read the **single word under the fingertip** (user choice A); pointing is the "read this" gesture.
- **Echo:** Contract test pattern exists for detectors (synthetic frames, type-filtered assertions); OCR contract tests exist.
- **Phantom:** On-device, zero-persistence; the fingertip crop is ephemeral — no new egress or storage.
- **Pulse:** ARM64 embedded target; <10s boot; models pre-warmed. `FLEC_*` env-var config convention.
- **Reported symptom:** integrated webcam (index 0, mirrored); "finger not detected, no audio."

### Round 2 — Inferential
- **Muse/Neo:** MediaPipe hand tracking is robust to horizontal mirroring → finger-not-detected is likely framing/lighting/reading-distance, not the mirror. (inference)
- **Bolt:** OCR every frame at 30 fps is infeasible on ARM64 → must throttle (run only when a finger is detected + stable), crop to the fingertip ROI, and downscale. (inference)
- **Bolt:** EasyOCR returns (bbox, text, confidence) → confidence enables a "silent when unsure" gate; bbox enables fingertip-nearest word selection. (inference)
- **Neo:** Auto-orientation = OCR the crop normal + horizontally-mirrored, keep higher confidence, cache the winner per session. (inference, from user choice A)

### Round 3 — Consequential
- **Bolt:** Instantiate `OCRReader` in `FlecSession`; run it on a **throttled worker thread** (not inline) so the 30 fps perception loop never stalls; feed regions via `finger_tracker.update_ocr`.
- **Echo:** New tests — OCR-orientation contract (normal + synthetic mirrored crop) and a Reading end-to-end integration test with a stubbed OCR returning a known word at a fingertip coordinate.
- **Muse:** Reading UX — one word at a time; silent when unsure; no "your book is backwards" messaging.
- **Phantom:** No new attack surface; confirm crop/frames remain un-persisted.
- **Pulse:** Possibly a `FLEC_OCR_*` throttle/confidence env knob; no infra change.
- **Jarvis:** Update RUNNING.md Reading section + docstrings.

### Round 4 — Speculative (what might we miss?)
- **Muse:** If finger detection is unreliable at book-reading distance, model A yields no audio → the whole feature feels broken. Linchpin risk.
- **Bolt:** OCR latency could blow the ~2s budget on ARM64 even with cropping.
- **Echo:** Mirrored text can produce a *confident* wrong word → an absolute confidence threshold is insufficient; use the **delta** between normal vs mirrored reads, not just a floor.
- **Neo:** If we OCR a mirror-corrected crop, the **fingertip x-coordinate must be mirrored to match** — coordinate-space bug risk.
- **Phantom:** Multiple hands/people in frame → whose fingertip owns the ROI?
- **Muse:** Crop size sensitivity — too small misses the word, too large reads neighbors.

### Round 5 — Conflicting
- **Bolt vs Neo — where does OCR run?** Bolt: simplest is inline in `process_frame` via the existing `ocr_result` param. Neo: inline OCR stalls the 30 fps loop. **Resolved (Neo):** OCR runs on a throttled background worker thread that produces text regions consumed by `FingerTracker.update_ocr`; the loop stays responsive. Detailed design deferred to Neo in Design.
- **Orientation breadth.** 8-variant (4 rotations × mirror) vs cost. **Resolved:** v1 handles **horizontal mirror only** (the integrated-webcam reality); 90°/180° rotation deferred to a follow-up. Keeps the auto-probe to 2 variants.
- No conflict required user escalation.

### Round 6 — Strategic (priorities)
Ranked (see Conclusion).

---

## Conclusion

**Confirmed facts:** Reading narration path exists but OCR is unwired in the live loop (`ocr_result=None`, no OCRReader); `OCRReader`/EasyOCR + `FingerTracker.update_ocr` are the building blocks; capability = single word under fingertip; on-device/zero-persistence; ARM64 target.

**Accepted inferences:** mirror unlikely to break finger detection (framing/lighting more likely); OCR must be throttled + cropped + downscaled; EasyOCR bbox+confidence supports nearest-word + silence gate; auto-orientation = normal-vs-mirror confidence-select, cached per session.

**Key consequences:** wire OCRReader on a background worker thread; feed regions to FingerTracker; add OCR-orientation contract + Reading e2e integration tests; RUNNING.md + docstring updates; no new security/infra surface.

**Risks & unknowns:** (1) finger-detection reliability at reading distance = linchpin; (2) OCR latency vs ~2s budget on ARM64; (3) confident-wrong reads on mirrored text → use normal/mirror confidence *delta*, not absolute floor; (4) fingertip-coordinate mirroring bug when OCR'ing a flipped crop; (5) crop-size sensitivity; (6) multi-hand ROI ownership.

**Resolved conflicts:** OCR runs on a throttled worker thread (not inline) — Neo; v1 = horizontal-mirror only, rotation deferred.

**User escalation answers:** none required.

**Design priorities (ranked):**
1. Fingertip-detection reliability (investigate/mitigate first — the linchpin for model A).
2. Wire OCRReader into the live loop on a throttled worker thread → `FingerTracker.update_ocr`.
3. Auto horizontal-mirror orientation via confidence-select on the fingertip crop; cache per session.
4. Correctness gate: confidence threshold + normal/mirror delta → silent when unsure (no gibberish).
5. Bound OCR cost: crop to ROI, downscale, throttle, cache orientation.

**Deferred/simplified:** 90°/180° rotation, full-page OCR, multi-language, hardware prism.
