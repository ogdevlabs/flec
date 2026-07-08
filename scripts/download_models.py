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
    """Download BLIP-2 INT8 model via HuggingFace transformers."""
    dest_dir = MODELS_DIR / "blip2-int8"
    if already_downloaded(dest_dir):
        print(f"  [SKIP] BLIP-2 INT8 already at {dest_dir}")
        return
    print("  [DOWNLOAD] BLIP-2 INT8 (transformers / HuggingFace)...")
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        ensure_dir(dest_dir)
        model_id = "Salesforce/blip2-opt-2.7b-coco"
        print(f"    Downloading processor from {model_id}...")
        processor = Blip2Processor.from_pretrained(model_id, cache_dir=str(dest_dir))
        print(f"    Downloading model weights (INT8, may take several minutes)...")
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto",
            cache_dir=str(dest_dir),
        )
        print(f"  [OK] BLIP-2 INT8 saved to {dest_dir}")
        del processor, model
    except ImportError as e:
        print(f"  [WARN] Required package not installed ({e}) — skipping BLIP-2 download")
    except Exception as e:
        print(f"  [WARN] BLIP-2 download failed ({e}) — skipping")


def main() -> None:
    print(f"Flec model downloader")
    print(f"Target directory: {MODELS_DIR.resolve()}")
    print("-" * 50)
    ensure_dir(MODELS_DIR)

    steps = [
        ("YOLOv8n", download_yolov8n),
        ("Whisper tiny", download_whisper_tiny),
        ("Coqui VITS", download_coqui_vits),
        ("EasyOCR latin", download_easyocr_latin),
        ("BLIP-2 INT8", download_blip2),
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
