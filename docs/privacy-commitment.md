# Flec Privacy Commitment

Flec is a wearable learning tool for toddlers. This document explains what data Flec processes and what it does not do.

## What Flec Does

- **On-device vision**: Flec uses the mask's built-in camera to recognize shapes, colors, and objects. All recognition runs locally on the device using pre-downloaded AI models.
- **On-device speech**: Flec listens for caregiver voice commands (e.g. "reading mode") using a local Whisper model. Nothing is sent to a server.
- **On-device text-to-speech**: Flec speaks back using the Coqui VITS model running locally.

## What Flec Does NOT Do

- **No cloud processing**: Camera frames are never transmitted to any server.
- **No storage**: Frames, audio clips, OCR results, and biometric data (fingertip positions) are processed in memory and immediately discarded. Nothing is written to disk during a session. See: [`src/flec/models.py`](../src/flec/models.py) — all dataclasses are marked `Ephemeral — never persisted`.
- **No analytics**: No usage data, no telemetry, no crash reports leave the device.
- **No network access during inference**: Capability threads (perception, OCR, TTS) are verified by CI to make no outbound network connections. See: [`tests/integration/test_no_network_egress.py`](../tests/integration/test_no_network_egress.py).

## COPPA / GDPR-K Note

Flec processes fingertip position data from toddlers to support the Reading mode. This data is biometric-adjacent (real-time body landmark). It is processed entirely on-device and never leaves the device. Whether this processing requires a parental consent notice under COPPA or GDPR-K is an **open legal question** (see threat-model.md T-006). Flec's current architecture is designed to make such consent straightforward to add if required.

## Evidence

- Data model: [`docs/pdlc/design/ultralytics-integration/data-model.md`](pdlc/design/ultralytics-integration/data-model.md)
- Privacy test: [`tests/unit/test_privacy_no_disk_write.py`](../tests/unit/test_privacy_no_disk_write.py)
- No-network CI test: [`tests/integration/test_no_network_egress.py`](../tests/integration/test_no_network_egress.py)
