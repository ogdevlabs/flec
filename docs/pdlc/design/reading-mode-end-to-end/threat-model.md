# Threat Model — reading-mode-end-to-end
<!-- pdlc-template-version: 1.0.0 -->

**Triage:** Lite
**Convened:** 2026-07-10
**Lead:** Phantom (Security Reviewer)
**Participants:** Phantom (solo — Lite mode)
**Status:** Approved (2026-07-11, Oscar Paul Garcia)

---

## Triage Record

| Question | Answer | Evidence |
|---|---|---|
| Does this feature introduce or modify a trust boundary? | no | ARCHITECTURE.md — camera→OCR→TTS is fully on-device; no network/auth boundary added |
| Does this feature touch regulated data (PII, payment, health, biometric, children's)? | no (children's context, but nothing retained) | data-model.md — zero persistence, no egress |
| Does this feature add a new attack surface (endpoint, consumer, upload, query, LLM tool, mobile handler)? | no | No network endpoint; only new input is camera pixels → spoken text (not executed) |

**Triage outcome:** Lite — no full party. One LOW availability note; content-safety flagged for the human as a non-security product consideration.

---

## Trust Boundaries

| ID | Boundary | What crosses | Trust direction | Diagram reference |
|---|---|---|---|---|
| TB-1 | Physical environment → camera sensor | printed text / images (uncontrolled) | untrusted → on-device | ARCHITECTURE.md data-flow |

No process, network, or persistence boundary is added. TB-1 already exists for every camera-driven mode.

---

## Threats Identified

### T-001 — Pathological scene degrades OCR latency (availability)
- **STRIDE category:** Denial of Service
- **Trust boundary:** TB-1
- **Asset affected:** Availability / responsiveness of the reading loop
- **Attack vector:** A visually dense/high-entropy scene under the fingertip could make an OCR pass slow; without bounds, a background worker could back up.
- **Severity:** LOW
- **DREAD:** Damage L · Reproducibility M · Exploitability L · Affected users single local device · Discoverability L
- **Mapped frameworks:** CWE-400 (Uncontrolled Resource Consumption)
- **Current mitigation status:** Mitigated by design — settle-gated trigger, fingertip-cropped ROI, downscale, throttle, and orientation cache (ARCHITECTURE.md D1/D4); worker runs off the 30 fps loop so the UI never stalls.
- **Proposed action (party recommendation):** Accept (already mitigated by design). Add an OCR per-pass time budget during Construction as defense-in-depth.
- **Decision (human, at Step 12 approval):** *[blank]*

---

## Threats Noted but Not Prioritized

| ID | Title | STRIDE | Boundary | Why deprioritized |
|---|---|---|---|---|
| T-NL-1 | Physical tampering with the device/camera | Tampering | TB-1 | Requires physical possession of the wearable; out of software scope |

---

## Open Questions for Human

1. **Content safety (product, not security):** OCR will read aloud *whatever printed text* is under the fingertip, and the BLIP-2 fallback will describe whatever image it sees — including potentially inappropriate text/imagery in a child's environment. This is inherent to "narrate the real world." Do you want a content-safety filter on spoken output as a follow-up feature, or is caregiver supervision (INTENT.md secondary persona) the accepted control for v1?

---

## Approval Outcomes (filled in at Step 12)

| Threat ID | Party recommendation | Human decision | Rationale |
|---|---|---|---|
| T-001 | Accept (mitigated by design) | Accept ✓ | Latency bounded by settle-gate + crop + throttle; add a per-pass OCR time budget in Construction as defense-in-depth |

**Content-safety open question:** human approved without mandating a spoken-output filter → **caregiver supervision is the accepted control for v1** (per INTENT.md secondary persona). A content-safety filter on spoken output is tracked as a possible follow-up feature, not v1 scope.

**ADR registry updates required:** none (no accepted security debt beyond design-mitigated T-001).

---

## Revision History

| Date | Author | Change |
|---|---|---|
| 2026-07-10 | Phantom (initial draft) | Created at Step 10.5 — Lite triage |
