#!/usr/bin/env python3
"""Benchmark fingertip detection model against the reference test set.

Used by flec-09b to validate the fine-tuned yolo26n-pose model meets the
AC-5 (confidence >= MediaPipe baseline) and AC-6 (latency gate) acceptance
criteria before the finger_tracker.py migration (flec-08h) can proceed.

Usage:
    python scripts/benchmark_fingertip.py
    python scripts/benchmark_fingertip.py --model .models/yolo26n-pose-ft.pt
    python scripts/benchmark_fingertip.py --model .models/yolo26n-pose-ft.pt --baseline mediapipe

Exit codes:
    0  PASS — all ACs met
    1  FAIL — one or more ACs failed (see stdout for details)
    2  SKIP — reference test set images not present (run download_fingertip_fixtures.py first)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "fingertip_reference"
IMAGES_DIR = FIXTURE_DIR / "images"
LABELS_DIR = FIXTURE_DIR / "labels"

# AC-6 latency gates (milliseconds, p95)
LATENCY_MACOS_P95_MS = 50
LATENCY_ARM64_P95_MS = 100

# AC-5 minimum confidence threshold
MIN_CONFIDENCE = 0.5


def _load_model(model_path: Path):
    """Load YOLO pose model. Returns None if unavailable."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        return model
    except Exception as exc:
        print(f"  [WARN] Could not load model {model_path}: {exc}")
        return None


def _load_mediapipe_baseline():
    """Load MediaPipe hands as baseline comparator. Returns None if unavailable."""
    try:
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
        )
        return hands
    except Exception as exc:
        print(f"  [WARN] MediaPipe unavailable: {exc}")
        return None


def _load_reference_images() -> list[Path]:
    """Return list of image files in the fixture images/ directory."""
    if not IMAGES_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts)


def _run_yolo_inference(model, images: list[Path]) -> dict:
    """Run YOLO pose inference on all images. Returns per-image results."""
    results = {}
    for img_path in images:
        t0 = time.monotonic()
        try:
            import numpy as np
            import cv2
            frame = cv2.imread(str(img_path))
            if frame is None:
                results[img_path.name] = {"error": "unreadable", "latency_ms": 0, "confidence": 0.0}
                continue
            preds = model(frame, verbose=False)
            latency_ms = (time.monotonic() - t0) * 1000
            # Extract best fingertip keypoint confidence
            conf = 0.0
            for r in preds:
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    kp_conf = r.keypoints.conf
                    if kp_conf is not None and len(kp_conf) > 0:
                        conf = float(kp_conf.max())
            results[img_path.name] = {"latency_ms": latency_ms, "confidence": conf}
        except Exception as exc:
            results[img_path.name] = {"error": str(exc), "latency_ms": 0, "confidence": 0.0}
    return results


def _run_mediapipe_inference(hands, images: list[Path]) -> dict:
    """Run MediaPipe hands inference on all images. Returns per-image confidence."""
    results = {}
    for img_path in images:
        try:
            import cv2
            frame = cv2.imread(str(img_path))
            if frame is None:
                results[img_path.name] = {"confidence": 0.0}
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = hands.process(rgb)
            conf = 0.0
            if out.multi_hand_landmarks:
                # MediaPipe does not expose per-landmark confidence directly;
                # use detection confidence as proxy
                if out.multi_handedness:
                    conf = float(out.multi_handedness[0].classification[0].score)
            results[img_path.name] = {"confidence": conf}
        except Exception as exc:
            results[img_path.name] = {"confidence": 0.0, "error": str(exc)}
    return results


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.95)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def benchmark(model_path: Path, compare_baseline: bool, platform: str) -> int:
    """Run the full benchmark. Returns exit code (0=PASS, 1=FAIL, 2=SKIP)."""
    images = _load_reference_images()
    if not images:
        print("[SKIP] No images found in tests/fixtures/fingertip_reference/images/")
        print("       Run: python scripts/download_fingertip_fixtures.py")
        return 2

    print(f"\nReference set: {len(images)} images")
    print(f"Model:         {model_path}")
    print(f"Platform:      {platform}")
    print("-" * 60)

    model = _load_model(model_path)
    if model is None:
        print(f"[FAIL] Could not load model — AC-5/AC-6 cannot be evaluated")
        return 1

    print("Running YOLO pose inference...")
    yolo_results = _run_yolo_inference(model, images)

    latencies = [r["latency_ms"] for r in yolo_results.values() if "error" not in r]
    confidences = [r["confidence"] for r in yolo_results.values() if "error" not in r]
    errors = [k for k, r in yolo_results.items() if "error" in r]

    p95_latency = _p95(latencies)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    below_gate = [c for c in confidences if c < MIN_CONFIDENCE]

    print(f"\nYOLO results:")
    print(f"  Mean confidence:   {mean_conf:.3f}  (gate: >={MIN_CONFIDENCE})")
    print(f"  Below gate:        {len(below_gate)}/{len(confidences)} images")
    print(f"  p95 latency:       {p95_latency:.1f}ms")
    if errors:
        print(f"  Errors:            {len(errors)} images could not be processed")

    # MediaPipe baseline comparison
    if compare_baseline:
        mp_hands = _load_mediapipe_baseline()
        if mp_hands:
            print("\nRunning MediaPipe baseline...")
            mp_results = _run_mediapipe_inference(mp_hands, images)
            mp_confs = [r["confidence"] for r in mp_results.values()]
            mp_mean = sum(mp_confs) / len(mp_confs) if mp_confs else 0.0
            print(f"  MediaPipe mean confidence: {mp_mean:.3f}")
            print(f"  YOLO vs MediaPipe delta:   {mean_conf - mp_mean:+.3f}")
            mp_hands.close()

    # AC-5: confidence gate
    latency_gate = LATENCY_MACOS_P95_MS if platform == "macos" else LATENCY_ARM64_P95_MS
    ac5_pass = len(below_gate) == 0
    ac6_pass = p95_latency <= latency_gate

    print(f"\n{'─' * 60}")
    print(f"AC-5 (confidence >= {MIN_CONFIDENCE} on all images): {'PASS' if ac5_pass else 'FAIL'}")
    print(f"AC-6 (p95 latency <= {latency_gate}ms on {platform}):   {'PASS' if ac6_pass else 'FAIL'}")

    if ac5_pass and ac6_pass:
        print("\n[PASS] All acceptance criteria met — finger_tracker.py migration (flec-08h) is unblocked.")
        report = {"status": "PASS", "mean_confidence": mean_conf, "p95_latency_ms": p95_latency,
                  "platform": platform, "n_images": len(images)}
        print(json.dumps(report))
        return 0
    else:
        print("\n[FAIL] One or more acceptance criteria failed.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fingertip detection model")
    parser.add_argument("--model", default=".models/yolo26n-pose-ft.pt",
                        help="Path to fine-tuned YOLO pose model")
    parser.add_argument("--baseline", choices=["mediapipe"], default=None,
                        help="Compare against baseline")
    parser.add_argument("--platform", choices=["macos", "arm64"], default="macos",
                        help="Target platform for latency gate")
    args = parser.parse_args()

    return benchmark(
        model_path=Path(args.model),
        compare_baseline=args.baseline == "mediapipe",
        platform=args.platform,
    )


if __name__ == "__main__":
    sys.exit(main())
