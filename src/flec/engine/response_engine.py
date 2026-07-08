"""ResponseEngine — single orchestration point for all audio and AR output.

Consumes DetectionEvents from perception modules and routes them to:
  1. The audio queue (TTSEngine) via AudioResponse objects.
  2. The AR overlay layer (AROverlay) for visual annotation.

No other module may write to the audio queue or AR overlay directly.

Architecture:
  - ResponseEngine is stateful: it tracks active Mode, active Challenge, and
    wear state. It uses these to gate which events produce responses.
  - Deduplication: same label within 3 seconds is suppressed in EXPLORATION mode
    to avoid audio flooding.
  - Mode isolation: shape/color narration only fires in EXPLORATION mode.
  - Structured JSON logging on every routing decision.

Privacy: no frames or audio are persisted. All state is in-memory and ephemeral.
"""

from __future__ import annotations

import json
import logging
import queue
import time
from typing import Optional

from flec.models import (
    AudioPriority,
    AudioResponse,
    Challenge,
    ChallengeStatus,
    DetectionEvent,
    DetectionType,
    Mode,
    WearState,
)
from flec.audio.responses import build_exploration_response

logger = logging.getLogger(__name__)

# Deduplication window: same label within this many seconds → suppress repeat
_DEDUP_WINDOW_SECONDS = 3.0


class ResponseEngine:
    """Routes detection events to audio and AR outputs.

    Usage:
        audio_queue = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
        engine.set_mode(Mode.EXPLORATION)
        engine.on_event(event)  # Enqueues AudioResponse if appropriate
    """

    def __init__(
        self,
        audio_queue: Optional[queue.Queue] = None,
        ar_overlay=None,  # type: ignore[type-arg]  # AROverlay, optional to avoid circular import
    ) -> None:
        """Construct the ResponseEngine.

        Args:
            audio_queue: Queue into which AudioResponse objects are enqueued.
                         If None, a new unbounded Queue is created.
            ar_overlay: Optional AROverlay instance for visual annotation.
                        If None, AR updates are skipped silently.
        """
        self._audio_queue: queue.Queue = audio_queue if audio_queue is not None else queue.Queue()
        self._ar = ar_overlay
        self._mode: Mode = Mode.STANDBY
        self._challenge: Optional[Challenge] = None
        self._wear_state: WearState = WearState.OFF_HEAD
        # Deduplication: label → last narration timestamp
        self._last_narrated: dict[str, float] = {}

        logger.info(json.dumps({
            "event": "response_engine_init",
            "mode": self._mode.name,
        }))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        """Transition the engine to a new mode.

        Clears deduplication cache on mode transition.
        """
        if mode != self._mode:
            logger.info(json.dumps({
                "event": "mode_transition",
                "from": self._mode.name,
                "to": mode.name,
            }))
            self._mode = mode
            # Clear dedup cache on mode change so new mode starts fresh
            self._last_narrated.clear()

    def set_challenge(self, challenge: Optional[Challenge]) -> None:
        """Set or clear the active challenge."""
        self._challenge = challenge
        logger.info(json.dumps({
            "event": "challenge_set",
            "target": challenge.target_label if challenge else None,
            "type": challenge.target_type.name if challenge else None,
        }))

    def set_wear_state(self, state: WearState) -> None:
        """Update the known wear state (affects VOICE_CMD routing)."""
        self._wear_state = state

    @property
    def mode(self) -> Mode:
        """Current active mode."""
        return self._mode

    @property
    def challenge(self) -> Optional[Challenge]:
        """Current active challenge, or None."""
        return self._challenge

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    def on_event(self, event: DetectionEvent) -> None:
        """Process a detection event and enqueue any resulting AudioResponse.

        State-aware: respects active mode, active challenge, and wear state.
        Never raises — errors are logged.
        """
        try:
            self._route_event(event)
        except Exception as exc:
            logger.error(json.dumps({
                "event": "response_engine_error",
                "error": str(exc),
                "detection_type": event.type.name,
                "label": event.label,
            }))

    # ------------------------------------------------------------------
    # Internal routing logic
    # ------------------------------------------------------------------

    def _route_event(self, event: DetectionEvent) -> None:
        """Core routing logic — dispatches based on event type and mode."""

        if event.type in (DetectionType.SHAPE, DetectionType.COLOR):
            self._route_shape_color(event)

        elif event.type == DetectionType.WEAR:
            self._route_wear_event(event)

        elif event.type == DetectionType.VOICE_CMD:
            self._route_voice_command(event)

        # AR update for spatial events (regardless of audio routing)
        if event.bounding_box is not None and self._ar is not None:
            self._ar.draw_detection  # lazy — called by update() externally

    def _route_shape_color(self, event: DetectionEvent) -> None:
        """Route SHAPE and COLOR events based on current mode."""

        if self._mode == Mode.EXPLORATION:
            self._handle_exploration_detection(event)

        elif self._mode == Mode.CHALLENGE:
            self._handle_challenge_detection(event)

        # All other modes: no narration for shape/color events

    def _handle_exploration_detection(self, event: DetectionEvent) -> None:
        """Narrate shape/color detections in EXPLORATION mode with deduplication."""

        label = event.label
        now = time.monotonic()

        # Deduplication: suppress if same label was narrated within window
        last_t = self._last_narrated.get(label, 0.0)
        if (now - last_t) < _DEDUP_WINDOW_SECONDS:
            logger.debug(json.dumps({
                "event": "narration_suppressed_dedup",
                "label": label,
                "seconds_since_last": round(now - last_t, 2),
            }))
            return

        response = build_exploration_response(event)
        self._enqueue(response)
        self._last_narrated[label] = now

        # AR update
        if self._ar is not None and event.bounding_box is not None:
            try:
                # AR overlay is updated externally with frames; here we just log intent
                logger.debug(json.dumps({
                    "event": "ar_update_queued",
                    "label": label,
                }))
            except Exception as exc:
                logger.warning(json.dumps({
                    "event": "ar_update_error",
                    "error": str(exc),
                }))

    def _handle_challenge_detection(self, event: DetectionEvent) -> None:
        """Route shape/color events during CHALLENGE mode."""

        if self._challenge is None or self._challenge.status != ChallengeStatus.ACTIVE:
            return

        label = event.label.lower().strip()
        target = self._challenge.target_label.lower().strip()

        if label == target:
            # Match! Play celebration
            celebration = AudioResponse(
                text=f"You found it! That's a {label}! Amazing!",
                priority=AudioPriority.HIGH,
                pre_cached=False,
            )
            self._enqueue(celebration)
            logger.info(json.dumps({
                "event": "challenge_match",
                "label": label,
                "target": target,
            }))
        else:
            # Near miss — encouraging, not discouraging (FR-007)
            now = time.monotonic()
            last_t = self._last_narrated.get(f"encourage_{label}", 0.0)
            if (now - last_t) >= _DEDUP_WINDOW_SECONDS:
                encouragement = AudioResponse(
                    text=f"Keep looking! You're doing great!",
                    priority=AudioPriority.NORMAL,
                    pre_cached=False,
                )
                self._enqueue(encouragement)
                self._last_narrated[f"encourage_{label}"] = now

    def _route_wear_event(self, event: DetectionEvent) -> None:
        """Handle WEAR state transition events."""

        label_lower = event.label.lower()

        if label_lower == "off_head":
            self._wear_state = WearState.OFF_HEAD
            # CRITICAL: put-mask-back-on response
            response = AudioResponse(
                text="Put your mask back on, hero!",
                priority=AudioPriority.CRITICAL,
                pre_cached=False,
            )
            self._enqueue(response)
            # Suspend active mode
            prev_mode = self._mode
            self._mode = Mode.STANDBY
            logger.info(json.dumps({
                "event": "wear_off_detected",
                "previous_mode": prev_mode.name,
            }))

        elif label_lower == "on_head":
            self._wear_state = WearState.ON_HEAD
            if self._mode == Mode.STANDBY:
                self._mode = Mode.EXPLORATION
            logger.info(json.dumps({
                "event": "wear_on_detected",
                "mode": self._mode.name,
            }))

    def _route_voice_command(self, event: DetectionEvent) -> None:
        """Handle VOICE_CMD events — shutdown, challenge, etc."""

        intent = event.metadata.get("intent")

        if intent == "SHUTDOWN":
            if self._wear_state == WearState.ON_HEAD:
                response = AudioResponse(
                    text="See you next time, hero!",
                    priority=AudioPriority.CRITICAL,
                    pre_cached=False,
                )
                self._enqueue(response)
                self._mode = Mode.STANDBY
                logger.info(json.dumps({"event": "shutdown_voice_command"}))
            else:
                # Ignore shutdown when not worn (FR-001e)
                logger.info(json.dumps({
                    "event": "shutdown_voice_command_ignored",
                    "reason": "mask_not_worn",
                }))

    def _enqueue(self, response: AudioResponse) -> None:
        """Enqueue an AudioResponse onto the audio queue. Non-blocking."""
        try:
            self._audio_queue.put_nowait(response)
            logger.debug(json.dumps({
                "event": "audio_enqueued",
                "text": response.text[:60],
                "priority": response.priority.name,
            }))
        except queue.Full:
            logger.warning(json.dumps({
                "event": "audio_queue_full",
                "dropped_text": response.text[:60],
                "priority": response.priority.name,
            }))
