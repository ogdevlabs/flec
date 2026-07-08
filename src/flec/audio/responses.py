"""Audio response text templates for Flec.

All template functions return plain strings — the TTSEngine decides whether to
synthesize or play a pre-cached WAV.

Toddler-First UX contract:
- All messages are positive and encouraging.
- No negative words (wrong, bad, error, fail).
- Short sentences (children's attention span).
- Exclamation marks for energy and warmth.
"""

from __future__ import annotations

import random


# ---------------------------------------------------------------------------
# Exploration mode
# ---------------------------------------------------------------------------


def exploration_narration(shape_or_color: str) -> str:
    """Narrate a detected shape or color during Exploration Mode."""
    templates = [
        f"I see a {shape_or_color}!",
        f"Ooh, I found a {shape_or_color}!",
        f"Look, a {shape_or_color}!",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Challenge mode — acknowledgment
# ---------------------------------------------------------------------------


def challenge_acknowledgment(target: str) -> str:
    """Acknowledge a caregiver voice challenge and set expectation.

    e.g. "Ok! Let's find a triangle!"
    """
    templates = [
        f"Ok! Let's find a {target}!",
        f"Ooh fun! Can you find a {target}?",
        f"Let's go! Find something {target}!",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Challenge mode — celebration (target found)
# ---------------------------------------------------------------------------


def challenge_celebration(target: str) -> str:
    """Celebrate the toddler finding the target.

    e.g. "You found it! That's a triangle!"
    """
    templates = [
        f"You found it! That's a {target}!",
        f"Amazing! You found a {target}! You're a superstar!",
        f"Yes! Great job! That is a {target}!",
        f"Woohoo! You found the {target}! You're so smart!",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Challenge mode — encouraging (no match yet)
# ---------------------------------------------------------------------------


def challenge_encouraging() -> str:
    """Encourage the toddler to keep searching.

    Must be positive — no negative words.
    """
    templates = [
        "Keep looking, hero!",
        "You can do it! Keep searching!",
        "Almost there! Keep going!",
        "Super job looking! Keep it up!",
        "You're doing great! Keep searching!",
    ]
    return random.choice(templates)


# ---------------------------------------------------------------------------
# Challenge mode — hint (30s elapsed without match)
# ---------------------------------------------------------------------------


def challenge_hint(target: str) -> str:
    """Gently remind the toddler what they are looking for.

    e.g. "Remember, we're looking for a triangle!"
    """
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
