# Decision Registry

**Project:** Flec
**Last updated:** 2026-07-08

<!-- PDLC Decision Registry (ADR format).
     Entries are appended by:
     - User: via /decide <text>
     - Agents: during Construction/Review (Step 14) and Reflect (Step 7)
     Each entry records: what was decided, who decided, why, what was considered,
     and what cross-cutting impacts were applied.
     This file is append-only. Mark superseded decisions as [SUPERSEDED by ADR-NNN]. -->

---

## ADR-001 — Queue-decoupled modules, no cross-module imports *(pre-PDLC, inferred)*

**Date:** 2026-07-07
**Status:** Accepted

**Decision:** Capability modules never import one another. All inter-module communication flows through in-memory queues (perception → event queue → ResponseEngine → audio queue).

**Context:** Keeps modules independently testable and swappable, prevents tight coupling between perception, reading, speech, and output, and lets `ResponseEngine` act as the single output gatekeeper. Cited as "Constitution §III" in `main.py`.

**Inferred from:** `src/flec/main.py` docstring, `engine/response_engine.py`, queue wiring in `FlecSession`.

---

## ADR-002 — Lazy ML imports with graceful degradation *(pre-PDLC, inferred)*

**Date:** 2026-07-07
**Status:** Accepted

**Decision:** Heavy ML libraries (mediapipe, torch, transformers, easyocr, whisper, TTS) are imported inside methods, not at module top level. A missing/failed dependency logs a warning and the module no-ops instead of crashing.

**Context:** Keeps startup import cost low, allows the app to boot and run partial functionality on machines missing an optional model, and supports the <10s boot target.

**Inferred from:** `perception/finger_tracker.py` (`_init_mediapipe` try/except), top-level import scan (only `cv2`/`numpy` imported at module level).

---

## ADR-003 — On-device inference, zero persistence *(pre-PDLC, inferred)*

**Date:** 2026-07-07
**Status:** Accepted

**Decision:** All inference runs on-device with lightweight models (YOLOv8n, Whisper tiny, Coqui VITS, EasyOCR latin, BLIP-2 INT8). No frames, audio, or biometric data are ever written to disk, and nothing is uploaded.

**Context:** The product is used by young children; privacy-by-design and offline operation are non-negotiable. Model choices target constrained ARM64 embedded hardware.

**Inferred from:** README Architecture section, `scripts/download_models.py`, absence of any file/network write of captured media.

---

## ADR-004 — Python 3.11/3.12 runtime pin *(pre-PDLC, inferred)*

**Date:** 2026-07-07
**Status:** Accepted

**Decision:** Target Python 3.11 (3.12 acceptable). Not 3.13/3.14.

**Context:** The ML dependency set (notably Coqui `TTS`, which requires `<3.12`, plus mediapipe/whisper wheels) does not support newer interpreters. During `/setup` the dev venv was rebuilt on Python 3.11.15 after 3.14 failed dependency resolution.

**Inferred from:** `specs/001-perception-core/quickstart.md` ("Python 3.11 or 3.12"), `pyproject.toml` `requires-python = ">=3.11"`, dependency resolution during setup.

---

<!-- New decisions appended below. -->
