"""Shared data models for Flec.

All models are immutable-by-convention dataclasses or enums.
No model class imports other capability modules — they are shared primitives only.
All session data is ephemeral (never persisted).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WearState(Enum):
    """Whether the mask is currently on a head."""

    ON_HEAD = auto()
    OFF_HEAD = auto()
    STANDBY = auto()


class Mode(Enum):
    """Active session mode."""

    EXPLORATION = auto()    # Passive shape/color narration
    CHALLENGE = auto()      # Active target-finding game
    READING = auto()        # Finger-guided reading
    STORY = auto()          # Autonomous picture-book read-aloud
    STANDBY = auto()        # Mask off or system idle


class DetectionType(Enum):
    """Category of a detection event."""

    SHAPE = auto()
    COLOR = auto()
    OBJECT = auto()         # Real-world object recognised by YOLO (e.g. "cup", "dog")
    WEAR = auto()           # Wear state transition
    FINGER = auto()         # Finger tracking update
    TEXT = auto()           # OCR text found
    ILLUSTRATION = auto()   # Illustration described
    VOICE_CMD = auto()      # Wake word + parsed command


class AudioPriority(Enum):
    """Playback priority for audio responses.

    Higher-priority responses pre-empt lower-priority ones in the TTS queue.
    """

    LOW = 1       # Background / ambient narration
    NORMAL = 2    # Standard detection narration
    HIGH = 3      # Challenge match / celebration
    CRITICAL = 4  # Safety / wear / shutdown (cannot be interrupted)


class CommandIntent(Enum):
    """Parsed intent from a caregiver voice command."""

    UNKNOWN = auto()
    START_CHALLENGE = auto()    # "find something <color/shape>"
    CANCEL_CHALLENGE = auto()   # "stop" / "cancel"
    REPEAT_CHALLENGE = auto()   # "say it again" / "repeat"
    SHUTDOWN = auto()           # "Hey Flec, off"
    SWITCH_EXPLORATION = auto() # "exploration" / "explore" / "look around"
    SWITCH_READING = auto()     # "reading" / "read"
    SWITCH_STORY = auto()       # "story" / "story time"
    SWITCH_CHALLENGE = auto()   # bare "challenge" / "game" (no target yet)


class ChallengeTargetType(Enum):
    """Type of target in an active Challenge."""

    COLOR = auto()
    SHAPE = auto()


class ChallengeStatus(Enum):
    """Lifecycle status of a Challenge."""

    ACTIVE = auto()
    COMPLETED = auto()
    EXPIRED = auto()
    CANCELLED = auto()


class ReadingIntent(Enum):
    """Inferred reading intent from finger velocity analysis."""

    IDLE = auto()       # No finger detected
    SCANNING = auto()   # Finger moving fast (browsing / repositioning)
    READING = auto()    # Finger slow — near readable text, reading in progress


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    """Normalised bounding box in [0.0, 1.0] coordinate space.

    Origin is top-left. All values are fractions of frame width/height.
    """

    x: float        # Left edge (0.0 = left of frame)
    y: float        # Top edge (0.0 = top of frame)
    width: float    # Box width as fraction of frame width
    height: float   # Box height as fraction of frame height

    def __post_init__(self) -> None:
        for attr in ("x", "y", "width", "height"):
            val = getattr(self, attr)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"BoundingBox.{attr} must be in [0.0, 1.0], got {val!r}"
                )


@dataclass(frozen=True)
class DetectionEvent:
    """A single detection result emitted by a perception module.

    Ephemeral — never persisted. All consumers should process and discard.
    """

    type: DetectionType
    label: str                              # Human-readable label, e.g. "red", "circle"
    confidence: float                       # Detection confidence in [0.0, 1.0]
    timestamp: float = field(default_factory=time.monotonic)
    bounding_box: Optional[BoundingBox] = None   # Present for spatial detections
    metadata: dict = field(default_factory=dict) # Extensible; not persisted

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"DetectionEvent.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )


@dataclass
class FingerTrackingState:
    """Current state of finger tracking for reading-intent inference.

    Mutable — updated in-place each frame by FingerTracker.
    Ephemeral — resets on mode transitions.
    """

    detected: bool = False              # True if a finger is visible in frame
    position_x: float = 0.0            # Normalised x position [0.0, 1.0]
    position_y: float = 0.0            # Normalised y position [0.0, 1.0]
    velocity: float = 0.0              # Rolling average speed (pixels/frame, non-negative)
    intent: ReadingIntent = ReadingIntent.IDLE
    nearest_text: Optional[str] = None # Closest readable text region, if any


@dataclass(frozen=True)
class Challenge:
    """An active challenge presented to the toddler.

    Issued by caregiver voice command. Ephemeral — not persisted.
    """

    target_type: ChallengeTargetType
    target_label: str                           # e.g. "red", "triangle"
    issued_at: float = field(default_factory=time.monotonic)
    status: ChallengeStatus = ChallengeStatus.ACTIVE

    EXPIRY_SECONDS: float = field(default=30.0, init=False, repr=False, compare=False)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Return True if the challenge has been active longer than EXPIRY_SECONDS."""
        t = now if now is not None else time.monotonic()
        return (t - self.issued_at) >= self.EXPIRY_SECONDS


@dataclass
class StoryContext:
    """Detected content of the current picture-book page.

    Resets on page turn. Ephemeral — not persisted.
    """

    page_text: str = ""                          # OCR-extracted page text
    illustrations: list[str] = field(default_factory=list)  # Descriptions of detected illustrations
    narrative_position: int = 0                  # Character index through page_text read so far
    page_stable: bool = False                    # True when camera has been still long enough


@dataclass(frozen=True)
class IllustrationDescription:
    """A child-friendly description of an image region."""

    description: str    # e.g. "a little yellow duck sitting in the water"
    confidence: float   # Model confidence in [0.0, 1.0]

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"IllustrationDescription.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        words = self.description.split()
        if len(words) > 20:
            raise ValueError(
                f"IllustrationDescription must be <= 20 words for child-friendliness; "
                f"got {len(words)}"
            )


@dataclass(frozen=True)
class AudioResponse:
    """An audio output queued for playback via TTSEngine.

    Non-blocking — enqueued and played by the TTS background thread.
    """

    text: str
    priority: AudioPriority = AudioPriority.NORMAL
    pre_cached: bool = False   # If True, play from pre-rendered WAV (no synthesis delay)
    cache_key: Optional[str] = None  # Key into the TTSEngine WAV cache when pre_cached=True


@dataclass(frozen=True)
class VoiceCommand:
    """A parsed caregiver voice command from CommandSTT.

    Never raises — returns UNKNOWN intent on transcription failure.
    """

    intent: CommandIntent
    target_label: Optional[str] = None         # Set for START_CHALLENGE (e.g. "red", "triangle")
    target_type: Optional[ChallengeTargetType] = None  # Resolved type for START_CHALLENGE
    raw_text: str = ""                         # Original transcribed text, for logging
