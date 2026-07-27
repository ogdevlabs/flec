#!/usr/bin/env python3
"""
annotate_fingertip_fixtures.py — flec-atl
==========================================
Estimate index-fingertip positions for images in the fingertip reference
fixture set and write YOLO-pose annotation files (kp_vis=1 = estimated).

Heuristic used
--------------
For images without an existing annotation (or with kp_vis=0 placeholder):

1. Load the image and convert to YCrCb colour space.
2. Threshold the Cr channel to create a rough skin mask.
3. Find the topmost skin pixel — that is the fingertip proxy.
4. Estimate the hand bounding box from the skin-mask contours.

This produces noisy but useful bootstrap annotations.  Human review is
recommended before fine-tuning; change kp_vis to 2 after verification.

Usage
-----
    python scripts/annotate_fingertip_fixtures.py [--overwrite]

Options
-------
--overwrite   Re-annotate even files that already have kp_vis != 0.
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "fingertip_reference"
IMG_DIR = FIXTURE_DIR / "images"
LBL_DIR = FIXTURE_DIR / "labels"


def _skin_mask(bgr_img):
    """Return a binary skin mask via YCrCb thresholding."""
    import cv2
    import numpy as np

    ycr = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([235, 173, 127], dtype=np.uint8)
    return cv2.inRange(ycr, lower, upper)


def _estimate_annotation(img_path: pathlib.Path) -> str | None:
    """
    Return a YOLO-pose annotation line or None if estimation fails.
    kp_vis=1 (estimated, not human-verified).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("  OpenCV not installed — cannot run heuristic annotation.")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    H, W = img.shape[:2]
    mask = _skin_mask(img)

    # Find the topmost non-zero pixel (fingertip)
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return None

    pts = nonzero.reshape(-1, 2)  # (N, 2) in (x, y)
    topmost = pts[pts[:, 1].argmin()]
    tip_x = float(topmost[0]) / W
    tip_y = float(topmost[1]) / H

    # Bounding box from all skin pixels
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    pad = 0.03
    bx0 = max(0.0, float(x_min) / W - pad)
    by0 = max(0.0, float(y_min) / H - pad)
    bx1 = min(1.0, float(x_max) / W + pad)
    by1 = min(1.0, float(y_max) / H + pad)
    cx = (bx0 + bx1) / 2
    cy = (by0 + by1) / 2
    bw = bx1 - bx0
    bh = by1 - by0

    return (
        f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}  {tip_x:.6f} {tip_y:.6f} 1\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-annotate even files that already have kp_vis != 0",
    )
    args = parser.parse_args()

    LBL_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

    if not images:
        print(f"No images found in {IMG_DIR}")
        print("Run  python scripts/download_fingertip_fixtures.py  first.")
        sys.exit(1)

    updated = 0
    skipped = 0
    for img_path in images:
        stem = img_path.stem
        lbl_path = LBL_DIR / f"{stem}.txt"

        if lbl_path.exists() and not args.overwrite:
            text = lbl_path.read_text().strip()
            # Check if already annotated (kp_vis != 0)
            try:
                parts = text.split()
                kp_vis = int(parts[-1])
                if kp_vis != 0:
                    skipped += 1
                    continue
            except (IndexError, ValueError):
                pass

        ann = _estimate_annotation(img_path)
        if ann is None:
            print(f"  SKIP  {img_path.name}  (estimation failed)")
            continue

        lbl_path.write_text(ann)
        updated += 1
        print(f"  OK    {img_path.name}")

    print(f"\nAnnotated {updated} images. Skipped {skipped} already-annotated.")
    print("Tip: review labels and change kp_vis to 2 after human verification.")


if __name__ == "__main__":
    main()
