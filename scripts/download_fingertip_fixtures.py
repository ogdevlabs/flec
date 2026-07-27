#!/usr/bin/env python3
"""
download_fingertip_fixtures.py — flec-atl
==========================================
Download 60 images from the HaGRID "point" gesture class for the
fingertip reference fixture set used to gate the yolo26n-pose fine-tune chain.

HaGRID license: CC BY 4.0  https://github.com/hukenovs/hagrid
Attribution: Kapitanov et al., "HaGRID — HAnd Gesture Recognition Image Dataset",
             arXiv:2206.08219, 2022.

Usage
-----
Option A — HuggingFace Hub (recommended):
    pip install huggingface_hub
    export HUGGING_FACE_HUB_TOKEN=<your_token>   # (or huggingface-cli login)
    python scripts/download_fingertip_fixtures.py

Option B — Direct (no auth required for public repos):
    python scripts/download_fingertip_fixtures.py --source direct

If neither source is reachable, placeholder synthetic images are generated
so the fixture structure is in place for CI.
"""

import argparse
import hashlib
import io
import json
import os
import pathlib
import random
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

# Load .env before reading tokens
sys.path.insert(0, str(Path(__file__).parent))
from _env import load_dotenv  # noqa: E402
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "fingertip_reference"
IMG_DIR = FIXTURE_DIR / "images"
LBL_DIR = FIXTURE_DIR / "labels"

TARGET_COUNT = 60
GESTURE_CLASS = "point"

# HaGRID sample dataset on HuggingFace (hagrid-sample-30k-384p, CC BY 4.0)
HF_DATASET_ID = "hagrid/hagrid-sample-30k-384p"
HF_SUBFOLDER = f"hagrid_dataset_sample/{GESTURE_CLASS}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)


def _make_placeholder_label(
    stem: str,
    tip_x: float = 0.5,
    tip_y: float = 0.25,
    vis: int = 0,
) -> None:
    """Write a YOLO-pose label with a centre bounding box and given fingertip."""
    label = f"0 0.500000 0.500000 0.800000 0.800000  {tip_x:.6f} {tip_y:.6f} {vis}\n"
    (LBL_DIR / f"{stem}.txt").write_text(label)


def _label_from_hagrid_annotation(
    ann: dict,
    stem: str,
    img_w: int,
    img_h: int,
) -> None:
    """
    Convert a HaGRID bounding-box annotation to YOLO-pose format.

    HaGRID bbox format: [x_min, y_min, width, height] normalised.
    We use the midpoint of the top edge of the bbox as a proxy for the
    index fingertip (kp_vis=1, i.e. estimated / not human-verified).
    """
    labels = ann.get("labels", [])
    bboxes = ann.get("bboxes", [])
    if not bboxes:
        _make_placeholder_label(stem)
        return

    lines = []
    for label, bbox in zip(labels, bboxes):
        if label != GESTURE_CLASS:
            continue
        x_min, y_min, bw, bh = bbox
        cx = x_min + bw / 2
        cy = y_min + bh / 2
        # Proxy fingertip: top-centre of the bounding box
        tip_x = cx
        tip_y = y_min  # top edge ≈ fingertip for pointing gestures
        lines.append(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {tip_x:.6f} {tip_y:.6f} 1\n"
        )

    if not lines:
        _make_placeholder_label(stem)
        return

    (LBL_DIR / f"{stem}.txt").write_text("".join(lines))


# ---------------------------------------------------------------------------
# Strategy A: HuggingFace Hub
# ---------------------------------------------------------------------------


def _download_via_hf_hub(n: int = TARGET_COUNT) -> int:
    """
    Download up to *n* images using the huggingface_hub library.
    Returns the number of images actually downloaded.
    """
    try:
        import huggingface_hub as hf  # lazy import
    except ImportError:
        print("  huggingface_hub not installed. pip install huggingface_hub")
        return 0

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    api = hf.HfApi()

    print(f"  Listing files in {HF_DATASET_ID}/{HF_SUBFOLDER} …")
    try:
        files = list(
            api.list_repo_files(
                repo_id=HF_DATASET_ID,
                repo_type="dataset",
                token=token,
            )
        )
    except Exception as exc:
        print(f"  HF listing failed: {exc}")
        return 0

    # Filter to images in the point subfolder
    point_files = [
        f for f in files
        if f.startswith(HF_SUBFOLDER) and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not point_files:
        print(f"  No images found under {HF_SUBFOLDER}")
        return 0

    rng = random.Random(42)
    rng.shuffle(point_files)
    selected = point_files[:n]

    # Try to load annotations
    ann_path = f"hagrid_dataset_sample/ann_test_{GESTURE_CLASS}.json"
    annotations: dict = {}
    if ann_path in files:
        try:
            local_ann = hf.hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=ann_path,
                repo_type="dataset",
                token=token,
            )
            with open(local_ann) as fh:
                annotations = json.load(fh)
            print(f"  Loaded {len(annotations)} annotations from {ann_path}")
        except Exception as exc:
            print(f"  Could not load annotations: {exc}")

    downloaded = 0
    for i, hf_path in enumerate(selected, 1):
        stem = pathlib.Path(hf_path).stem
        dest = IMG_DIR / pathlib.Path(hf_path).name
        if dest.exists():
            print(f"  [{i:02d}/{len(selected)}] already exists: {dest.name}")
            downloaded += 1
            continue
        try:
            local = hf.hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=hf_path,
                repo_type="dataset",
                token=token,
            )
            import shutil

            shutil.copy2(local, dest)
            # Label
            ann = annotations.get(stem, {})
            if ann:
                _label_from_hagrid_annotation(ann, stem, 384, 384)
            else:
                _make_placeholder_label(stem)
            downloaded += 1
            print(f"  [{i:02d}/{len(selected)}] {dest.name}")
        except Exception as exc:
            print(f"  [{i:02d}/{len(selected)}] FAILED {hf_path}: {exc}")
        time.sleep(0.05)  # polite rate-limiting

    return downloaded


# ---------------------------------------------------------------------------
# Strategy B: Direct download via requests
# ---------------------------------------------------------------------------


def _download_via_direct(n: int = TARGET_COUNT) -> int:
    """
    Attempt to download images from the HaGRID public HTTP mirror.
    HaGRID v1 annotations (JSON) are fetched first to discover image names.
    Returns the number of images downloaded.
    """
    try:
        import requests
    except ImportError:
        print("  requests not installed.")
        return 0

    # HaGRID v1 annotation file for the test split (small JSON ~2 MB)
    ANN_URL = (
        "https://raw.githubusercontent.com/hukenovs/hagrid/master/"
        "hagrid_annotations/ann_test.json"
    )
    print(f"  Fetching annotation index: {ANN_URL}")
    try:
        resp = requests.get(ANN_URL, timeout=30)
        resp.raise_for_status()
        all_ann = resp.json()
    except Exception as exc:
        print(f"  Could not fetch annotation index: {exc}")
        return 0

    # Filter to "point" gesture samples
    point_items = [
        (img_id, ann)
        for img_id, ann in all_ann.items()
        if GESTURE_CLASS in ann.get("labels", [])
    ]
    rng = random.Random(42)
    rng.shuffle(point_items)
    selected = point_items[:n]
    print(f"  Found {len(point_items)} point samples; downloading {len(selected)}")

    BASE_URL = (
        "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
        "datasets/hagrid/hagrid_dataset_512/test/point"
    )
    downloaded = 0
    for i, (img_id, ann) in enumerate(selected, 1):
        filename = f"{img_id}.jpg"
        dest = IMG_DIR / filename
        if dest.exists():
            print(f"  [{i:02d}/{len(selected)}] already exists: {filename}")
            downloaded += 1
            continue
        url = f"{BASE_URL}/{filename}"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            dest.write_bytes(r.content)
            _label_from_hagrid_annotation(ann, img_id, 512, 512)
            downloaded += 1
            print(f"  [{i:02d}/{len(selected)}] {filename}")
        except Exception as exc:
            print(f"  [{i:02d}/{len(selected)}] FAILED {filename}: {exc}")
        time.sleep(0.1)

    return downloaded


# ---------------------------------------------------------------------------
# Fallback: generate synthetic placeholders
# ---------------------------------------------------------------------------


def _generate_synthetic_placeholders(n: int = 10) -> int:
    """
    Generate *n* synthetic 384x384 hand-schematic images with placeholder
    YOLO-pose labels (kp_vis=0).  Used when no network source is reachable.
    """
    print(f"  Generating {n} synthetic placeholder images …")
    try:
        from PIL import Image, ImageDraw  # lazy import
    except ImportError:
        print("  Pillow not installed — cannot generate synthetic images.")
        return 0

    rng = random.Random(42)
    W, H = 384, 384
    generated = 0
    for i in range(1, n + 1):
        stem = f"point_placeholder_{i:04d}"
        dest = IMG_DIR / f"{stem}.jpg"
        if dest.exists():
            generated += 1
            continue

        bg_r = rng.randint(200, 230)
        bg_g = rng.randint(160, 200)
        bg_b = rng.randint(130, 170)
        img = Image.new("RGB", (W, H), color=(bg_r, bg_g, bg_b))
        draw = ImageDraw.Draw(img)

        palm_cx = rng.uniform(0.35, 0.65)
        palm_cy = rng.uniform(0.60, 0.75)
        palm_w = rng.uniform(0.30, 0.45)
        palm_h = rng.uniform(0.30, 0.40)

        skin = (bg_r - 20, bg_g - 20, bg_b - 20)
        draw.rectangle(
            [
                int((palm_cx - palm_w / 2) * W),
                int((palm_cy - palm_h / 2) * H),
                int((palm_cx + palm_w / 2) * W),
                int((palm_cy + palm_h / 2) * H),
            ],
            fill=skin,
            outline=(100, 70, 50),
            width=2,
        )

        finger_cx = palm_cx + rng.uniform(-0.05, 0.05)
        finger_w = rng.uniform(0.07, 0.10)
        finger_h = rng.uniform(0.28, 0.38)
        finger_cy = palm_cy - palm_h / 2 - finger_h / 2

        draw.rectangle(
            [
                int((finger_cx - finger_w / 2) * W),
                int((finger_cy - finger_h / 2) * H),
                int((finger_cx + finger_w / 2) * W),
                int((finger_cy + finger_h / 2) * H),
            ],
            fill=skin,
            outline=(100, 70, 50),
            width=2,
        )

        tip_x = max(0.0, min(1.0, finger_cx))
        tip_y = max(0.0, min(1.0, finger_cy - finger_h / 2))
        tr = 6
        draw.ellipse(
            [
                int(tip_x * W) - tr,
                int(tip_y * H) - tr,
                int(tip_x * W) + tr,
                int(tip_y * H) + tr,
            ],
            fill=(220, 50, 50),
        )

        img.save(dest, quality=90)

        bb_cx = (palm_cx + finger_cx) / 2
        bb_top = tip_y - 0.02
        bb_bot = palm_cy + palm_h / 2
        bb_cy = (bb_top + bb_bot) / 2
        bb_w = max(palm_w, finger_w) + 0.10
        bb_h = bb_bot - bb_top + 0.04
        for v in (bb_cx, bb_cy, bb_w, bb_h, tip_x, tip_y):
            v = max(0.0, min(1.0, v))

        _make_placeholder_label(stem, tip_x=tip_x, tip_y=tip_y, vis=0)
        generated += 1
        print(f"  [{i:02d}/{n}] {stem}.jpg  tip=({tip_x:.3f},{tip_y:.3f})")

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["hf", "direct", "synthetic", "auto"],
        default="auto",
        help=(
            "Download strategy: 'hf' = HuggingFace Hub, 'direct' = sbercloud mirror, "
            "'synthetic' = generate placeholder images, 'auto' = try hf then direct "
            "then synthetic (default)"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=TARGET_COUNT,
        help=f"Number of images to download (default: {TARGET_COUNT})",
    )
    args = parser.parse_args()

    _ensure_dirs()
    print(f"\nFingertip reference fixture downloader — target: {args.count} images")
    print(f"Destination: {IMG_DIR}\n")

    n = 0
    if args.source in ("hf", "auto"):
        print("Strategy: HuggingFace Hub")
        n = _download_via_hf_hub(args.count)

    if n < args.count and args.source in ("direct", "auto"):
        remaining = args.count - n
        print(f"\nStrategy: direct HTTP download ({remaining} remaining)")
        n += _download_via_direct(remaining)

    if n < 1 and args.source in ("synthetic", "auto"):
        print("\nStrategy: synthetic placeholders (no network source reachable)")
        n = _generate_synthetic_placeholders(min(10, args.count))

    print(f"\nDone — {n} images in {IMG_DIR}")
    label_count = len(list(LBL_DIR.glob("*.txt")))
    print(f"     — {label_count} label files in {LBL_DIR}")

    if n < args.count:
        print(
            f"\nNote: only {n}/{args.count} images downloaded.\n"
            "To download the full set, install huggingface_hub and set\n"
            "HUGGING_FACE_HUB_TOKEN, then re-run with --source hf"
        )


if __name__ == "__main__":
    main()
