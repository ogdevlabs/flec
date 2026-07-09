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
# Voice character variants — Exploration Mode
# ---------------------------------------------------------------------------

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
    if word and word[0] in _VOWEL_SOUNDS:
        return "an"
    return "a"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def random_variant(templates: list[str], **kwargs) -> str:  # type: ignore[no-untyped-def]
    template = random.choice(templates)
    return template.format(**kwargs)


def narrate_detection(event: DetectionEvent, paired_color: Optional[str] = None) -> str:
    """Generate a child-friendly narration string for a detection event."""
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
        text = "I see something interesting!"

    logger.debug(json.dumps({
        "event": "narration_generated",
        "detection_label": label,
        "detection_type": event.type.name,
        "text": text,
    }))
    return text


def exploration_narration(shape_or_color: str) -> str:
    """Simple narration for a detected shape or color label string."""
    templates = [
        f"I see a {shape_or_color}!",
        f"Ooh, I found a {shape_or_color}!",
        f"Look, a {shape_or_color}!",
    ]
    return random.choice(templates)


def build_exploration_response(
    event: DetectionEvent, paired_color: Optional[str] = None
) -> AudioResponse:
    """Build an AudioResponse for an exploration-mode detection event."""
    text = narrate_detection(event, paired_color=paired_color)
    return AudioResponse(
        text=text,
        priority=AudioPriority.NORMAL,
        pre_cached=False,
    )


# ---------------------------------------------------------------------------
# Challenge Mode
# ---------------------------------------------------------------------------


def challenge_acknowledgment(target: str) -> str:
    templates = [
        f"Ok! Let's find a {target}!",
        f"Ooh fun! Can you find a {target}?",
        f"Let's go! Find something {target}!",
    ]
    return random.choice(templates)


def challenge_celebration(target: str) -> str:
    templates = [
        f"You found it! That's a {target}!",
        f"Amazing! You found a {target}! You're a superstar!",
        f"Yes! Great job! That is a {target}!",
        f"Woohoo! You found the {target}! You're so smart!",
    ]
    return random.choice(templates)


def challenge_encouraging() -> str:
    templates = [
        "Keep looking, hero!",
        "You can do it! Keep searching!",
        "Almost there! Keep going!",
        "Super job looking! Keep it up!",
        "You're doing great! Keep searching!",
    ]
    return random.choice(templates)


def challenge_hint(target: str) -> str:
    templates = [
        f"Remember, we're looking for a {target}!",
        f"Keep searching! We want to find a {target}!",
        f"Hey hero, look around for a {target}!",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Wear / session lifecycle
# ---------------------------------------------------------------------------


def wear_welcome() -> str:
    return "Hero mask activated! Let's explore!"


def wear_off_prompt() -> str:
    return "Put your mask back on, hero!"


def session_farewell() -> str:
    return "See you next time, hero!"


# ---------------------------------------------------------------------------
# Mode switching (voice-commanded)
# ---------------------------------------------------------------------------


def mode_switch_confirmation(mode) -> str:
    """Child-friendly confirmation spoken when a mode is entered by voice.

    ``mode`` is a flec.models.Mode member; falls back to a neutral "Okay!"
    for any mode without a dedicated line.
    """
    from flec.models import Mode

    phrases = {
        Mode.EXPLORATION: ["Let's explore!", "Exploration time!", "Let's look around!"],
        Mode.READING: ["Reading time!", "Let's read together!", "Point at the words!"],
        Mode.STORY: ["Story time!", "Let's read a story!", "Snuggle up for a story!"],
        Mode.CHALLENGE: ["Challenge time!", "Let's play a game!", "Ready to find things?"],
    }
    return random.choice(phrases.get(mode, ["Okay!"]))
