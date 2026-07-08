"""Pre-cached audio response constants for Flec.

Defines cache key constants and the corresponding text for all responses
that are pre-rendered at boot time (Constitution Rule 5: Toddler-First UX
— critical responses must have zero synthesis delay).

Usage:
    from flec.audio.responses import CACHE_MANIFEST, CacheKey
    tts_engine.preload_cache(CACHE_MANIFEST)

All text is kid-friendly, encouraging, and audio-complete.
No error messages or technical language ever reaches the child.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Cache key constants
# ---------------------------------------------------------------------------


class CacheKey:
    """String constants for TTSEngine WAV cache keys.

    Use these constants everywhere — never hardcode the strings directly.
    """

    BOOT_READY: str = "BOOT_READY"
    """Played when the boot sequence completes and the mask is ready."""

    MASK_OFF: str = "MASK_OFF"
    """Played when wear detection transitions to OFF_HEAD during an active session."""

    SHUTDOWN: str = "SHUTDOWN"
    """Played when the caregiver issues 'Hey Flec, off' while the mask is on."""

    CELEBRATION: str = "CELEBRATION"
    """Played when the toddler finds the challenge target."""

    THINKING: str = "THINKING"
    """Brief placeholder played while the system is processing (e.g. post-wake-word)."""

    CHALLENGE_HINT: str = "CHALLENGE_HINT"
    """Played after 30 seconds of no match in Challenge Mode."""

    ENCOURAGE: str = "ENCOURAGE"
    """Played for non-matching detections in Challenge Mode (encouraging, not negative)."""

    LOW_LIGHT: str = "LOW_LIGHT"
    """Played when the camera feed is too dark to analyse."""


# ---------------------------------------------------------------------------
# Cache manifest: key → text to pre-render at startup
# ---------------------------------------------------------------------------

CACHE_MANIFEST: dict[str, str] = {
    CacheKey.BOOT_READY: "Hero mask activated! Ready to explore!",
    CacheKey.MASK_OFF: "Put your mask back on, hero!",
    CacheKey.SHUTDOWN: "See you next time, hero!",
    CacheKey.CELEBRATION: "You found it! Amazing job, hero!",
    CacheKey.THINKING: "Hmm...",
    CacheKey.CHALLENGE_HINT: "Keep looking, hero! You are getting closer!",
    CacheKey.ENCOURAGE: "Good try! Keep looking!",
    CacheKey.LOW_LIGHT: "I cannot see very well. Can we find more light?",
}
