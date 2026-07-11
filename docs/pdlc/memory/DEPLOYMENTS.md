# Deployments
<!-- pdlc-template-version: 1.1.0 -->
<!-- Canonical register of deployment environments for this project.
     Maintained by Pulse during the Ship and Verify sub-phases; read by the
     team on every ship to understand the current deployment surface.

     This file answers: where do we deploy, how, with what secrets, how do we
     roll back, and how is each environment identified in our cloud/catalog?

     Claude never writes secret *values* here — only variable names. Real
     values live in your secret store (Vault, AWS Secrets Manager, etc.). -->

**Project:** Flec
**Last updated:** 2026-07-08

---

## Environments

<!-- No deployment environment configured yet. Flec is an on-device wearable
     (ARM64 embedded Linux in production, macOS for dev) rather than a hosted
     service — Pulse will define the device provisioning / flashing flow as an
     "environment" during the first /ship. -->

### Environment: <!-- e.g. production -->

**Purpose:** <!-- One sentence on what this environment is for -->
**URL:** <!-- n/a — on-device wearable, no hosted URL -->
**Status:** planned

#### Deploy

- **Method:** <!-- device image / flash | manual -->
- **Command:** <!-- TBD -->
- **Workflow file:** <!-- none yet -->
- **Custom deploy artifact:** none — default pipeline
- **Latest Deployment Review MOM:** n/a
- **Triggered by:** <!-- TBD -->
- **Typical duration:** <!-- TBD -->

#### Verification

- **Smoke test URL:** <!-- n/a -->
- **Required smoke checks:** <!-- e.g. boot-to-ready <10s, all models pre-warmed -->

#### Rollback

- **Method:** <!-- TBD -->
- **Command:** <!-- TBD -->
- **Reversibility window:** <!-- TBD -->
- **Last successful rollback:** <!-- none -->

#### Required secrets / env vars

| Name | Purpose | Source |
|------|---------|--------|
| FLEC_CAMERA_INDEX | OpenCV camera device index | device config / `.env` |
| FLEC_DEV_MODE | dev vs mask camera | device config / `.env` |

#### Tags

| Key | Value | Notes |
|-----|-------|-------|
| tier | <!-- dev | test | staging | pre-production | production --> | **Required.** Assign before first ship. |
| cloud-provider | n/a (on-device) | Wearable hardware, not cloud-hosted |

#### Deployment History

| Date | Version | Deployed by | Episode | Notes |
|------|---------|-------------|---------|-------|
<!-- none yet -->

#### Notes

On-device product: "deployment" means provisioning/flashing the embedded ARM64 device,
not deploying to a hosted environment. Pulse formalizes this at the first `/ship`.

---

<!-- ═══════════════════════════════════════════════════════════════════
     END OF ENVIRONMENT BLOCK — clone above for every environment you run.
     ═══════════════════════════════════════════════════════════════════ -->

## Cross-environment references

- **Promotion path:** <!-- e.g. dev (macOS) → device staging → production device image -->
- **Shared infrastructure:** <!-- none -->
- **Data migration policy:** <!-- n/a — zero persistence -->
- **Smoke test dependencies:** <!-- none -->

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
<!-- none yet -->
