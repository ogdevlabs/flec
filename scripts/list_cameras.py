"""List all available camera devices with their index and name.

Uses OpenCV to probe camera indices and reports which ones are accessible.

Usage:
    python scripts/list_cameras.py
"""

import sys


def list_cameras(max_index: int = 10) -> list[dict]:
    """Enumerate available cameras by probing OpenCV indices 0..max_index-1.

    Returns a list of dicts with 'index', 'name', 'width', 'height' keys
    for each accessible camera.
    """
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python is not installed. Run: pip install opencv-python")
        sys.exit(1)

    cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, _ = cap.read()
        if not ret:
            cap.release()
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # OpenCV does not expose camera display names on all platforms;
        # use backend name as a best-effort label.
        backend = cap.getBackendName()
        name = f"Camera {index} ({backend})"
        cameras.append(
            {
                "index": index,
                "name": name,
                "width": width,
                "height": height,
                "fps": fps,
            }
        )
        cap.release()

    return cameras


def main() -> None:
    print("Flec camera enumerator")
    print("-" * 40)
    cameras = list_cameras()
    if not cameras:
        print("No cameras found (indices 0–9 probed).")
        print("Tip: ensure camera permissions are granted in System Settings → Privacy → Camera.")
        return

    for cam in cameras:
        print(
            f"  Index {cam['index']:2d}: {cam['name']}  "
            f"{cam['width']}x{cam['height']} @ {cam['fps']:.0f}fps"
        )

    print(f"\n{len(cameras)} camera(s) found.")
    print("Set FLEC_CAMERA_INDEX=<index> in your .env file.")


if __name__ == "__main__":
    main()
