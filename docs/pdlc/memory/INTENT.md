# Intent
<!-- pdlc-template-version: 2.1.0 -->
<!-- This file defines the core purpose of the product.
     It is set during /pdlc init and should rarely change.
     If the fundamental problem or user changes, update this file and record why in docs/pdlc/memory/DECISIONS.md.
     Claude reads this at the start of every Inception phase to anchor the Discover conversation. -->

**Project:** Flec
**Created:** 2026-07-08
**Last updated:** 2026-07-08

---

## Project Name

Flec — the screenless superhero mask that teaches toddlers shapes, colors, and words.

---

## Problem Statement

Toddlers learn shapes, colors, and early words best through play in the physical world,
but most "educational" tools put a screen between the child and their environment. Screens
introduce passive consumption, eye-strain concerns, and parental guilt, and they pull a
toddler's attention away from the real objects and books in front of them. There is no
hands-free, screen-free way for a young child to get real-time, encouraging narration about
the shapes, colors, and words they encounter as they explore.

---

## Target User (Persona)

**Primary: The Curious Toddler (age ~2–4)**
- Pre-literate or early-literate; learning to name shapes, colors, and first words
- Explores by looking, pointing, and handling real objects and picture books
- Cannot read error messages and has no patience for setup or screens
- Delighted by playful voice feedback and a "superhero" wearable they control

**Secondary users (if any):**
Parents/caregivers who set up and supervise the device and want a screen-free,
privacy-respecting learning aid (no camera data leaves the device).

---

## Core Value Proposition

Only Flec lets a toddler learn shapes, colors, and words hands-free and screen-free —
by wearing a mask whose camera watches the real world and speaks back encouragingly,
with zero data ever leaving the device.

---

## What Success Looks Like

<!-- Needs human input — placeholders below reflect inferred product goals, not committed targets. -->

| Metric | Target | Timeframe |
|--------|--------|-----------|
| <!-- metric --> | <!-- target --> | <!-- e.g. 30 days post-launch --> |
| <!-- metric --> | <!-- target --> | <!-- timeframe --> |
| <!-- metric --> | <!-- target --> | <!-- timeframe --> |

---

## Out of Scope

<!-- Inferred from architecture; confirm with the team. -->

- No screen or visual UI for the child (the AR overlay is a dev-only companion screen)
- No cloud inference or media upload — all models run on-device
- No persistence of frames, audio, or biometric data
- No accounts, profiles, or personalization storage in v1

---

## Key Constraints

- Runs on ARM64 embedded Linux hardware in production; developed on macOS
- On-device only: models must fit the embedded target (YOLOv8n, Whisper tiny, VITS, EasyOCR, BLIP-2 INT8)
- <10s boot-to-ready target; all models pre-warmed at boot
- Python 3.11/3.12 (Coqui TTS caps at <3.12)
- Coqui TTS requires `espeak-ng` on the host; BLIP-2 8-bit needs `accelerate`/`bitsandbytes` (unavailable on macOS MPS — Story mode degraded in dev)
