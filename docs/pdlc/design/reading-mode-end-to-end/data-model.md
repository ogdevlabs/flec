# Data Model: Reading Mode — End-to-End

**No persistent data model changes. This feature operates entirely on in-memory,
ephemeral state — consistent with the Constitution's zero-persistence invariant
(privacy-by-design for minors).**

## Why no persistence

Flec never writes frames, audio, biometrics, or recognized text to disk. Reading mode
adds only transient, per-session in-memory state, all discarded each frame or at session end:

| State | Type | Lifetime | Persisted? |
|-------|------|----------|-----------|
| Current frame / fingertip crop | `np.ndarray` (BGR) | per frame | No |
| OCR text regions `(text, bbox, confidence)` | list | until next OCR pass | No |
| `FingerTrackingState.nearest_text` | `str \| None` | until next update | No |
| Cached orientation (normal vs mirrored) | bool/enum | per session | No (memory only) |
| Pending illustration description | `str \| None` | until narrated | No |

## Deliberately NOT persisted (and why)

- **Camera frames / crops** — privacy-by-design for minors; on-device, zero retention.
- **Recognized text & descriptions** — could constitute a record of what a child reads; never stored.
- **Orientation cache** — a runtime optimization only; re-derived each session (cheap, and camera may change).

No migrations. No new tables, collections, or schema. No ERD (no entities persisted).
