"""IllustrationDescriber — generate child-friendly descriptions of book illustrations.

Responsibility: Describe images in simple language for toddlers.
Contract: see specs/001-perception-core/contracts/module-interfaces.md

Architecture notes:
- BLIP-2 INT8 model loaded from .models/ via HuggingFace transformers
- Model loaded lazily on first call; cached for subsequent calls
- Post-processing enforces <=20 word limit and removes technical terms
- Never raises — returns empty string on any error
- Emits structured JSON logs for every description event (Principle II)
- Does NOT import other capability modules (Principle III)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Maximum word count for child-friendly output
_MAX_WORDS = 20

# Technical/model terms that must never appear in a toddler-facing description
_JARGON_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b" + term + r"\b", re.IGNORECASE)
    for term in [
        "tensor", "logit", "embedding", "token", "pixel", "rgb", "bgr",
        "classification", "inference", "softmax", "batch", "cuda", "cpu",
        "model", "neural", "network", "layer", "weight", "gradient",
        "dtype", "numpy", "ndarray", "array", "matrix",
    ]
]

# Minimum frame dimensions (below this, skip processing entirely)
_MIN_DIMENSION = 8

# Minimum mean pixel value — frames below this are considered too dark
_MIN_MEAN_BRIGHTNESS = 5.0

# Prompt used when querying BLIP-2
_BLIP_PROMPT = "Describe this picture in simple words a toddler can understand:"


def _is_degenerate_frame(frame: np.ndarray) -> bool:
    """Return True if the frame is too small or too dark to describe."""
    if frame is None:
        return True
    if frame.ndim < 2:
        return True
    h, w = frame.shape[:2]
    if h < _MIN_DIMENSION or w < _MIN_DIMENSION:
        return True
    # Check brightness — mostly black frames are not illustrations
    mean_brightness = float(np.mean(frame))
    return mean_brightness < _MIN_MEAN_BRIGHTNESS


def _truncate_to_n_words(text: str, n: int = _MAX_WORDS) -> str:
    """Truncate *text* to at most *n* words.

    Tries to end on a complete sentence if possible; falls back to hard truncation.
    """
    words = text.split()
    if len(words) <= n:
        return text

    # Try to end at a sentence boundary within the word limit
    partial = " ".join(words[:n])
    # Find last sentence-ending punctuation
    for end_char in (".", "!", "?"):
        idx = partial.rfind(end_char)
        if idx >= 0:
            return partial[: idx + 1].strip()

    # Hard truncation
    return partial.strip()


def _strip_jargon(text: str) -> str:
    """Remove known technical/jargon terms from *text*."""
    for pattern in _JARGON_PATTERNS:
        text = pattern.sub("", text)
    # Clean up extra whitespace left by removals
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _clean_description(raw: str) -> str:
    """Apply all post-processing steps to a raw model output string."""
    if not raw:
        return ""
    # Normalize whitespace
    text = " ".join(raw.split())
    # Strip jargon terms
    text = _strip_jargon(text)
    # Enforce word limit
    text = _truncate_to_n_words(text, _MAX_WORDS)
    # Remove any remaining leading/trailing whitespace
    return text.strip()


class IllustrationDescriber:
    """Generate child-friendly descriptions of book-page illustrations.

    Uses BLIP-2 (INT8 quantized) for image-to-text captioning.
    Loaded lazily on first ``describe()`` call.

    Usage::

        describer = IllustrationDescriber()
        description = describer.describe(frame)  # returns str, never raises
    """

    def __init__(self) -> None:
        self._processor: Optional[object] = None
        self._model: Optional[object] = None
        self._load_error: Optional[str] = None

    def _load_model(self) -> bool:
        """Attempt to load BLIP-2 model. Returns True on success."""
        if self._load_error is not None:
            return False
        if self._model is not None:
            return True

        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoProcessor,
                Blip2ForConditionalGeneration,
            )
            import torch  # type: ignore[import-untyped]

            model_path = ".models/blip2"

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )

            # Load model with INT8 quantization for memory efficiency on ARM64
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map="auto",
                local_files_only=True,
            )

            logger.info(
                json.dumps({
                    "event": "illustration_describer_loaded",
                    "module": "IllustrationDescriber",
                    "status": "ok",
                })
            )
            return True

        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.error(
                json.dumps({
                    "event": "illustration_describer_load_failed",
                    "module": "IllustrationDescriber",
                    "error": str(exc),
                })
            )
            return False

    def _run_inference(self, frame: np.ndarray) -> str:
        """Run BLIP-2 inference on *frame* and return raw caption text."""
        import torch  # type: ignore[import-untyped]
        from PIL import Image  # type: ignore[import-untyped]

        # Convert BGR numpy array → RGB PIL image
        rgb = frame[:, :, ::-1].copy()
        pil_image = Image.fromarray(rgb.astype("uint8"))

        # Tokenise
        inputs = self._processor(  # type: ignore[operator]
            images=pil_image,
            text=_BLIP_PROMPT,
            return_tensors="pt",
        )

        with torch.no_grad():
            generated_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=40,
                num_beams=4,
                early_stopping=True,
            )

        caption: str = self._processor.batch_decode(  # type: ignore[union-attr]
            generated_ids, skip_special_tokens=True
        )[0]

        # Strip the prompt prefix if the model echoed it back
        caption = caption.replace(_BLIP_PROMPT, "").strip()
        return caption

    def describe(self, image: np.ndarray) -> str:
        """Return a simple, child-friendly description of *image*.

        Returns empty string if image cannot be described (no model, blank frame,
        or any error). Never raises.

        Args:
            image: BGR numpy array (H x W x 3).

        Returns:
            A description of <=20 words, e.g. "a little yellow duck sitting in water",
            or ``""`` if the image cannot be described.
        """
        if _is_degenerate_frame(image):
            logger.debug(
                json.dumps({
                    "event": "illustration_skipped",
                    "module": "IllustrationDescriber",
                    "reason": "degenerate_frame",
                })
            )
            return ""

        if not self._load_model():
            return ""

        try:
            raw_caption = self._run_inference(image)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                json.dumps({
                    "event": "illustration_inference_error",
                    "module": "IllustrationDescriber",
                    "error": str(exc),
                })
            )
            return ""

        description = _clean_description(raw_caption)

        if description:
            logger.info(
                json.dumps({
                    "event": "illustration_described",
                    "module": "IllustrationDescriber",
                    "word_count": len(description.split()),
                    "description": description,
                })
            )
        else:
            logger.debug(
                json.dumps({
                    "event": "illustration_empty_result",
                    "module": "IllustrationDescriber",
                    "raw_caption": raw_caption,
                })
            )

        return description
