# API Contracts: Reading Mode — End-to-End

**No new HTTP/network API endpoints.** Flec is an on-device application with no network
egress. The "contracts" here are the **internal module interfaces** the OCR worker composes.
They matter because the queue-only architecture depends on these public signatures staying stable.

---

## Internal module contracts

### `reading/ocr_reader.py` — `OCRReader`

Existing/extended. Recognizes text in a frame or crop.

- `detect(frame: np.ndarray) -> list[TextRegion]`
  - `TextRegion`: `{ text: str, bbox: BoundingBox (normalized), confidence: float [0,1] }`
  - Returns `[]` on blank/unreadable input. **Never raises** (graceful degradation).
  - If the EasyOCR model is unavailable: log a structured warning once and return `[]` (R9).

### OCR worker → `perception/finger_tracker.py`

- `FingerTracker.update_ocr(text_regions: list[str] | list[TextRegion]) -> None`
  - Populates `FingerTrackingState.nearest_text` with the region nearest the current fingertip.
  - Called from the worker thread after orientation resolution.

### Orientation resolution (new helper, inside the worker / `ocr_reader`)

- `resolve_orientation(crop: np.ndarray, cached: Orientation | None) -> tuple[list[TextRegion], Orientation]`
  - Runs `detect` on `crop` and on `cv2.flip(crop, 1)`; returns the higher-mean-confidence
    result **only if** `confidence_delta >= DELTA_GATE` and `top_confidence >= CONF_GATE`;
    otherwise returns `([], cached)` (silence, decision D2).
  - When `cached` is set, probe the cached orientation first and skip the second pass unless confidence drops.

### Picture fallback → `reading/illustration_describer.py`

- `IllustrationDescriber.describe(region: np.ndarray) -> IllustrationDescription | None`
  - Returns `None` (and logs a warning) when BLIP-2 is unavailable (MPS-degraded in dev, R4/known-risk).

### Narration → `engine/response_engine.py` (unchanged contract)

- FINGER `DetectionEvent` metadata gains a populated `nearest_text` (and/or a pending
  illustration). `_handle_finger` narrates only when `mode==READING` and `intent==READING`.

---

## Config (env, `FLEC_`-prefixed)

| Var | Purpose | Default |
|-----|---------|---------|
| `FLEC_OCR_CONF_GATE` | Min confidence to speak a word | ~0.4 (tune) |
| `FLEC_OCR_DELTA_GATE` | Min normal-vs-mirror confidence delta to decide orientation | ~0.1 (tune) |
| `FLEC_OCR_SETTLE_MS` | Fingertip-stable duration before OCR fires | ~400 |
| `FLEC_READING_WEAR_OVERRIDE` | Dev: treat webcam as ON_HEAD | on in dev |

Thresholds are tunable and validated during Construction/Test; defaults are starting points, not commitments.
