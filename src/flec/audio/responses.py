"""Audio response templates and narration helpers for Flec.

Generates child-friendly narration strings for detection events.
Provides multiple voice character variants to keep the experience
fresh and engaging across repeated detections.

Design:
- All output is audio-complete: no content relies on visual display.
- Templates use simple, encouraging language appropriate for ages 2-5.
- random_variant() selects a different phrasing each call to reduce repetition.
- narrate_detection() is the primary entry point for the ResponseEngine.

Privacy: no persistent state — all functions are pure (or use module-level
random state only).
"""

from __future__ import annotations

import json
import logging
import random
from typing import Optional

from flec.models import AudioPriority, AudioResponse, DetectionEvent, DetectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice character variants
# ---------------------------------------------------------------------------

# Templates receive format kwargs: label, color, shape, article
# All templates are ≤ ~10 words — appropriate for toddler attention spans.

_SHAPE_TEMPLATES: list[str] = [
    "I see a {label}!",
    "Oh, a {label}! Cool!",
    "Look, there's a {label}!",
    "Wow, a {label}!",
    "Hey, I found a {label}!",
]

_COLOR_TEMPLATES: list[str] = [
    "I see something {label}!",
    "Ooh, {label}! I love {label}!",
    "Look at that {label} thing!",
    "Wow, {label}! So bright!",
    "I spy something {label}!",
]

_SHAPE_COLOR_TEMPLATES: list[str] = [
    "I see a {color} {shape}!",
    "Wow, a {color} {shape}!",
    "Oh look, a {color} {shape}!",
    "I found a {color} {shape}!",
    "Hey! A {color} {shape}!",
]

# ---------------------------------------------------------------------------
# Article helpers
# ---------------------------------------------------------------------------

_VOWEL_SOUNDS = frozenset("aeiouAEIOU")


def _article(word: str) -> str:
    """Return 'an' if word starts with a vowel sound, else 'a'."""
    if word and word[0] in _VOWEL_SOUNDS:
        return "an"
    return "a"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def random_variant(templates: list[str], **kwargs) -> str:  # type: ignore[no-untyped-def]
    """Select a random template and format it with the given kwargs.

    Args:
        templates: List of format strings.
        **kwargs: Format arguments (label, color, shape, article, etc.)

    Returns:
        A formatted narration string.
    """
    template = random.choice(templates)
    return template.format(**kwargs)


def narrate_detection(event: DetectionEvent, paired_color: Optional[str] = None) -> str:
    """Generate a child-friendly narration string for a detection event.

    Args:
        event: The detection event to narrate.
        paired_color: If provided and event is a SHAPE, use combined "color + shape" template.

    Returns:
        A narration string ready for TTS synthesis. Never empty — falls back
        to a generic encouraging phrase if the event is unrecognised.
    """
    label = event.label.lower().strip()

    if event.type == DetectionType.SHAPE:
        if paired_color:
            text = random_variant(
                _SHAPE_COLOR_TEMPLATES,
                color=paired_color,
                shape=label,
                article=_article(paired_color),
            )
        else:
            text = random_variant(_SHAPE_TEMPLATES, label=label, article=_article(label))

    elif event.type == DetectionType.COLOR:
        text = random_variant(_COLOR_TEMPLATES, label=label, article=_article(label))

    else:
        # Fallback for unknown event types — audio-complete, never silent
        text = f"I see something interesting!"

    logger.debug(json.dumps({
        "event": "narration_generated",
        "detection_label": label,
        "detection_type": event.type.name,
        "text": text,
    }))

    return text


def build_exploration_response(
    event: DetectionEvent, paired_color: Optional[str] = None
) -> AudioResponse:
    """Build an AudioResponse for an exploration-mode detection event.

    Args:
        event: Detection event (SHAPE or COLOR).
        paired_color: Optional color label to pair with a SHAPE event.

    Returns:
        An AudioResponse at NORMAL priority with the narration text.
    """
    text = narrate_detection(event, paired_color=paired_color)
    return AudioResponse(
        text=text,
        priority=AudioPriority.NORMAL,
        pre_cached=False,
    )
