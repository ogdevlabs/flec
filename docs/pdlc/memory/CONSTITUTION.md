# Constitution
<!-- pdlc-template-version: 2.5.0 -->
<!-- This file is the single source of truth for how this project is built.
     PDLC reads it before every phase. Strong defaults are already set.
     Override only what your team explicitly agrees to change.
     Edits to this file are logged by the guardrails hook for `/diagnose`
     reconciliation. -->


**Version:** 1.0.0
**Last updated:** 2026-07-08
**Project:** Flec

---

## 1. Tech Stack Decisions

<!-- Reverse-engineered from the codebase during brownfield /setup scan.
     Marked (inferred) where derived from code rather than an explicit decision doc. -->

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11 (spec allows 3.11/3.12) | ML ecosystem support; Coqui TTS caps at <3.12 |
| Runtime / Platform | macOS (dev) · ARM64 embedded Linux (prod) | Wearable target device |
| Computer vision | OpenCV, MediaPipe (finger tracking), YOLOv8n / Ultralytics (shape + color) | Lightweight, on-device inference |
| Speech (in) | openWakeWord (wake word) + openai-whisper tiny (STT) | Small models for embedded latency |
| Speech (out) | Coqui TTS / VITS (LJSpeech) | Kid-friendly local voice; requires `espeak-ng` |
| Reading | EasyOCR (text), BLIP-2 INT8 / Transformers (illustration) | On-device OCR + image captioning |
| ML runtime | PyTorch (MPS on macOS, no CUDA required) | Standard tensor backend |
| Testing | pytest + hypothesis (property tests) | Contract / integration / unit + benchmarks |
| Packaging | setuptools (`src/` layout), editable install | Standard Python packaging |
| CI/CD | none yet | Set up at `/ship` (Pulse) |

---

## 2. Coding Standards & Style

<!-- Observed conventions from the existing codebase. -->

### Linting & Formatting

- Linter: none configured (no ruff/flake8/pylint config found) — consider adding
- Formatter: none configured (no black/isort config found) — consider adding
- Pre-commit hook: none

### Naming Conventions

| Construct | Convention | Example |
|-----------|-----------|---------|
| Modules / files | snake_case | `finger_tracker.py` |
| Classes | PascalCase | `ResponseEngine`, `FingerTracker` |
| Functions / variables | snake_case | `process_frame`, `nearest_text` |
| Enums | PascalCase type, SCREAMING_SNAKE members | `Mode.EXPLORATION` |
| Env variables | `FLEC_` prefix, SCREAMING_SNAKE | `FLEC_CAMERA_INDEX` |
| Branch names | `phase/[n]-[slug]` (observed) | `phase/8-polish` |

### General Rules

- All logging is structured JSON (`json.dumps({"event": ...})`) written to stdout
- Heavy ML libraries are lazy-imported inside methods, never at module top level
- Modules must degrade gracefully when an ML dependency is unavailable (log + continue)
- No error messages ever surface to the child — all user-facing feedback is audio

---

## 3. Architectural Constraints

<!-- These are load-bearing invariants observed in the code (Constitution §III cited in main.py).
     Treat as guardrails — flag any deviation. -->

- **No cross-module imports** — capability modules communicate only via in-memory queues (perception → event queue → ResponseEngine → audio queue)
- **Single output gatekeeper** — `ResponseEngine` is the only component that emits audio/AR output
- **All models pre-warmed at boot** — <10s ready-state target
- **Wear detection gates everything** — all modes suspend when the mask is removed
- **Zero persistence** — no frames, audio, or biometric data are ever written to disk
- **Toddler-first UX** — no error messages reach the child; all feedback is audio-complete

---

## 4. Security & Compliance Requirements

<!-- Baseline for a product used by children. Phantom verifies during Review. -->

- Zero persistence of camera frames, audio, or biometric data (privacy-by-design for minors)
- No network egress of captured media — all inference is on-device
- Secrets never in source or logs — configuration via `FLEC_*` env vars / `.env` (gitignored)
- Dependency + secret scan required before each ship

---

## 5. Definition of Done

- [ ] Code is committed on the feature branch with a conventional commit message
- [ ] All unit tests pass (`pytest -m "not slow"`)
- [ ] All integration tests pass (`pytest tests/integration/`)
- [ ] All contract tests pass (`pytest tests/contract/`)
- [ ] Code has been reviewed per PDLC review flow
- [ ] Review file (`docs/pdlc/reviews/REVIEW_*.md`) exists and is human-approved
- [ ] No debug/print statements left in committed code (structured logging only)
- [ ] All public functions/methods have docstrings
- [ ] Graceful degradation preserved for any new ML dependency
- [ ] PR description is complete and references the Beads task ID
- [ ] Episode file drafted and human-approved

---

## 6. Git Workflow Rules

### Branch Strategy

- **Feature branch model (default)**: one branch per feature, single PR to `main`.
- Observed historical pattern: `phase/[n]-[slug]` branches merged via PR.

**Default branch:** `main`
**Feature branch naming:** `feature/[kebab-case-feature-name]`
**Merge strategy:** Merge commit (preserves full branch history)

### Commit Message Format

Format: `<type>(<scope>): <description>`

Types: `feat` | `fix` | `chore` | `docs` | `test` | `refactor` | `perf` | `ci`

Examples (from history):
- `feat(6-01): implement FingerTracker with velocity-based reading intent`
- `test(8-01): performance-benchmarks`

**Breaking changes:** append `!` after type.

### Protected Branches

- `main` — requires PR + human approval

---

## 7. Test Gates

<!-- Checked layers hard-block the ship. This project already has strong contract +
     integration + unit coverage (206 test functions). -->

- [x] Unit tests
- [x] Integration tests
- [ ] E2E tests (real Chromium)
- [ ] Performance / load tests
- [ ] Accessibility checks
- [ ] Visual regression tests
- [x] Security scan (dependency audit + secret scan — always required, cannot be unchecked)

<!-- Performance benchmarks exist but are marked `slow` and excluded from the default gate. -->

### Custom Test Layers

| Name | Command | Required |
|------|---------|----------|
| Contract tests | `pytest tests/contract/` | yes |
| Performance benchmarks | `pytest -m slow` | no |

---

## 8. Context & Model Configuration

**Context window (tokens):** 1000000

**Warning threshold:** 60
**Critical threshold:** 75

**Distill threshold (tokens):** 800

**Interaction Mode:** Socratic

---

## 9. Additional Rules

- **Dangerous-action norms:** none flagged during init (user answered via `/setup`). Claude Code's permission system still prompts on destructive actions in default mode.
- Any new capability module must follow the queue-only contract in §3 — no direct imports between capability modules.
