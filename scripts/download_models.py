"""Download all required Flec AI models to the .models/ directory.

Run once before first use:
    python scripts/download_models.py

Models downloaded:
  - YOLOv8n       (ultralytics auto-download, object detection)
  - Whisper tiny  (openai-whisper, speech-to-text)
  - Coqui VITS    (TTS, text-to-speech)
  - EasyOCR latin (easyocr, optical character recognition)
  - BLIP-2 INT8   (transformers, illustration description)
"""

import os
import sys
from pathlib import Path

# Load .env before anything else so HUGGING_FACE_HUB_TOKEN is available
sys.path.insert(0, str(Path(__file__).parent))
from _env import load_dotenv  # noqa: E402
load_dotenv()

MODELS_DIR = Path(__file__).parent.parent / ".models"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def already_downloaded(path: Path) -> bool:
    """Return True if path exists and is non-empty."""
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(path.iterdir())
    return False


def download_yolov8n() -> None:
    """Download YOLOv8n model via ultralytics."""
    dest = MODELS_DIR / "yolov8n.pt"
    if already_downloaded(dest):
        print(f"  [SKIP] YOLOv8n already at {dest}")
        return
    print("  [DOWNLOAD] YOLOv8n (ultralytics)...")
    try:
        from ultralytics import YOLO
        # ultralytics downloads to its own cache by default; export to .models/
        model = YOLO("yolov8n.pt")
        src = Path("yolov8n.pt")
        if src.exists():
            src.rename(dest)
        print(f"  [OK] YOLOv8n saved to {dest}")
    except ImportError:
        print("  [WARN] ultralytics not installed — skipping YOLOv8n download")


def download_whisper_tiny() -> None:
    """Download Whisper tiny model via openai-whisper."""
    dest_dir = MODELS_DIR / "whisper-tiny"
    if already_downloaded(dest_dir):
        print(f"  [SKIP] Whisper tiny already at {dest_dir}")
        return
    print("  [DOWNLOAD] Whisper tiny (openai-whisper)...")
    try:
        import whisper
        ensure_dir(dest_dir)
        # whisper.load_model downloads to its cache; we set download_root
        model = whisper.load_model("tiny", download_root=str(dest_dir))
        print(f"  [OK] Whisper tiny saved to {dest_dir}")
        del model
    except ImportError:
        print("  [WARN] openai-whisper not installed — skipping Whisper download")


def download_coqui_vits() -> None:
    """Download Coqui TTS VITS model."""
    dest_dir = MODELS_DIR / "coqui-vits"
    if already_downloaded(dest_dir):
        print(f"  [SKIP] Coqui VITS already at {dest_dir}")
        return
    print("  [DOWNLOAD] Coqui VITS (TTS)...")
    try:
        from TTS.api import TTS
        ensure_dir(dest_dir)
        # Use a lightweight English VITS model suitable for kid-friendly voice
        tts = TTS("tts_models/en/ljspeech/vits")
        print(f"  [OK] Coqui VITS downloaded (cached by TTS library)")
    except ImportError:
        print("  [WARN] TTS not installed — skipping Coqui VITS download")


def download_easyocr_latin() -> None:
    """Download EasyOCR Latin detection and recognition models."""
    dest_dir = MODELS_DIR / "easyocr"
    if already_downloaded(dest_dir):
        print(f"  [SKIP] EasyOCR latin already at {dest_dir}")
        return
    print("  [DOWNLOAD] EasyOCR latin models...")
    try:
        import easyocr
        ensure_dir(dest_dir)
        reader = easyocr.Reader(["en"], model_storage_directory=str(dest_dir), download_enabled=True)
        print(f"  [OK] EasyOCR latin models saved to {dest_dir}")
        del reader
    except ImportError:
        print("  [WARN] easyocr not installed — skipping EasyOCR download")


def download_blip2() -> None:
    """Download BLIP-2 model via HuggingFace Hub snapshot_download.

    Uses snapshot_download with local_dir= so files land flat in .models/blip2/
    (not in the nested models--Salesforce--blip2-opt-2.7b-coco/snapshots/...
    cache structure).  illustration_describer.py loads from that flat path with
    local_files_only=True, which requires the flat layout.

    Set FLEC_SKIP_BLIP2=1 to skip entirely (used in CI where bitsandbytes is
    unavailable and the 4 GB download is impractical).
    """
    if os.environ.get("FLEC_SKIP_BLIP2"):
        print("  [SKIP] BLIP-2 — FLEC_SKIP_BLIP2 is set")
        return

    dest_dir = MODELS_DIR / "blip2"

    # Only consider it downloaded if the flat config.json is present at root
    # (not nested inside a HF cache subdirectory).
    if (dest_dir / "config.json").exists():
        print(f"  [SKIP] BLIP-2 already at {dest_dir}")
        return

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None
    if not token:
        print("  [INFO] HUGGING_FACE_HUB_TOKEN not set — attempting unauthenticated download")
        print("         Set it in .env to avoid rate-limiting on large model files")

    print("  [DOWNLOAD] BLIP-2 (HuggingFace Hub snapshot, ~4 GB — may take several minutes)...")
    try:
        from huggingface_hub import snapshot_download
        ensure_dir(dest_dir)
        model_id = "Salesforce/blip2-opt-2.7b-coco"
        hf_kwargs = {"token": token} if token else {}
        snapshot_download(
            repo_id=model_id,
            local_dir=str(dest_dir),
            **hf_kwargs,
        )
        print(f"  [OK] BLIP-2 saved to {dest_dir}")
    except ImportError:
        print("  [WARN] huggingface_hub not installed — skipping BLIP-2 download")
        print("         Install: pip install huggingface_hub")
    except Exception as e:
        print(f"  [WARN] BLIP-2 download failed: {e}")


# Registry of YOLO26n model variants needed for F-002
# SHA-256 checksums are placeholder values until official weights are released
YOLO26_MODELS = {
    "yolo26n": {
        "path": ".models/yolo26n.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "sha256": "PLACEHOLDER_VERIFY_BEFORE_PRODUCTION",
        "description": "YOLO26n detection (2.4M params)",
    },
    "yolo26n-pose": {
        "path": ".models/yolo26n-pose.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
        "sha256": "PLACEHOLDER_VERIFY_BEFORE_PRODUCTION",
        "description": "YOLO26n pose estimation",
    },
    "yolo26n-seg": {
        "path": ".models/yolo26n-seg.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "sha256": "PLACEHOLDER_VERIFY_BEFORE_PRODUCTION",
        "description": "YOLO26n instance segmentation",
    },
    "yolo26n-obb": {
        "path": ".models/yolo26n-obb.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
        "sha256": "PLACEHOLDER_VERIFY_BEFORE_PRODUCTION",
        "description": "YOLO26n oriented bounding box (document detection)",
    },
}


def download_yolo26_models() -> None:
    """Download YOLO26n model variants for F-002 multi-task perception."""
    for name, spec in YOLO26_MODELS.items():
        dest = Path(spec["path"])
        if already_downloaded(dest):
            print(f"  [SKIP] {name} already at {dest}")
            continue
        print(f"  [DOWNLOAD] {name} ({spec['description']})...")
        try:
            from ultralytics import YOLO
            ensure_dir(dest.parent)
            model = YOLO(dest.name)
            src = Path(dest.name)
            if src.exists():
                src.rename(dest)
            print(f"  [OK] {name} saved to {dest}")
        except ImportError:
            print(f"  [WARN] ultralytics not installed — skipping {name}")
        except Exception as e:
            print(f"  [WARN] {name} download failed: {e}")


def verify_checksums(checksums_path: Path) -> None:
    """Verify model SHA-256 checksums against model_checksums.json.

    Stub implementation — full verification requires minisign Python bindings
    on ARM64 (not yet available). When implemented, this function should:
      1. Load the public key from checksums_path["public_key"].
      2. Verify checksums_path.minisig using minisign -V.
      3. For each model entry, compute sha256(model_path) and compare.
      4. Raise RuntimeError on any mismatch.

    See docs/KEYS.md for the signing workflow.
    """
    # TODO: verify_checksums(MODELS_DIR / "model_checksums.json")
    pass


def main() -> None:
    print(f"Flec model downloader")
    print(f"Target directory: {MODELS_DIR.resolve()}")
    print("-" * 50)
    ensure_dir(MODELS_DIR)

    # TODO: verify_checksums(MODELS_DIR / "model_checksums.json")
    # Uncomment once minisign Python bindings are available on ARM64.

    steps = [
        ("YOLOv8n", download_yolov8n),
        ("Whisper tiny", download_whisper_tiny),
        ("Coqui VITS", download_coqui_vits),
        ("EasyOCR latin", download_easyocr_latin),
        ("BLIP-2 INT8", download_blip2),
        ("YOLO26n variants", download_yolo26_models),
    ]

    for name, fn in steps:
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")

    print("\n" + "-" * 50)
    print("Done. Run 'python -m flec.main --mode dev' to start.")


if __name__ == "__main__":
    main()
