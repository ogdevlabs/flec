"""ResponseEngine — single orchestration point for all audio and AR output.

Consumes DetectionEvents from perception modules and routes them to the
TTSEngine (audio) and AROverlay (visual). No other module writes audio
or AR directly.

Architecture:
  - Stateful: tracks active Mode, active Challenge, WearState, and StoryContext.
  - Deduplication: same (type, label) key within 3s suppressed.
  - Mode isolation: each event type only fires audio in its relevant modes.
  - Story mode: TEXT events → cursor-gated narration; ILLUSTRATION → insert-point description.
  - StoryContext=None (book removed) → silent pause, no error audio.
  - Structured JSON logging on every routing decision.

Privacy: no frames or audio are persisted. All state in-memory and ephemeral.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Optional, Protocol

from flec.models import (
    AudioPriority,
    AudioResponse,
    Challenge,
    ChallengeStatus,
    ChallengeTargetType,
    CommandIntent,
    DetectionEvent,
    DetectionType,
    Mode,
    ReadingIntent,
    StoryContext,
    WearState,
)

logger = logging.getLogger(__name__)

_DEDUP_WINDOW_SECONDS: float = 3.0
_ENCOURAGE_THROTTLE_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# TTS Protocol
# ---------------------------------------------------------------------------


class _TTSProtocol(Protocol):
    def speak(self, response: AudioResponse) -> None: ...
    def stop_current(self) -> None: ...
    def clear_pending(self) -> None: ...


class _QueueTTS:
    """Wraps a queue.Queue to satisfy _TTSProtocol.

    Allows the Phase 4 test pattern ``ResponseEngine(audio_queue=q)`` to
    continue working unchanged.
    """

    def __init__(self, q: queue.Queue) -> None:
        self._q = q

    def speak(self, response: AudioResponse) -> None:
        self._q.put(response)

    def stop_current(self) -> None:
        pass

    def clear_pending(self) -> None:
        # No-op: tests inspect the queue directly, so it is never auto-drained.
        pass


class _NarrationQueueTTS:
    """Bounded narration queue + dispatcher thread wrapping a real TTS backend.

    Protects against unbounded queue growth (T-005). The dispatcher dequeues
    AudioResponse objects in FIFO order and calls the real backend's speak().
    When the queue is full, the incoming response is dropped with a warning log.
    """

    def __init__(self, tts_backend: _TTSProtocol, maxsize: int = 50) -> None:
        self._backend = tts_backend
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(
            target=self._dispatch_loop,
            name="flec.narration_dispatcher",
            daemon=True,
        )
        self._thread.start()

    def speak(self, response: AudioResponse) -> None:
        try:
            self._q.put_nowait(response)
        except queue.Full:
            logger.warning(json.dumps({
                "event": "narration_queue_full",
                "dropped": response.text[:40],
            }))

    def stop_current(self) -> None:
        self._backend.stop_current()

    def clear_pending(self) -> None:
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._backend.clear_pending()

    def _dispatch_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                self._backend.speak(item)
            except Exception as exc:
                logger.error(json.dumps({
                    "event": "tts_dispatch_error",
                    "error": str(exc),
                }))

    def shutdown(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# ResponseEngine
# ---------------------------------------------------------------------------


class ResponseEngine:
    """Routes detection events to audio and AR outputs.

    Accepts either a Protocol-compliant ``tts`` object or a legacy
    ``audio_queue`` (queue.Queue) for backward compatibility.

    Usage (preferred)::

        engine = ResponseEngine(tts=tts_engine)

    Usage (legacy / tests)::

        audio_queue = queue.Queue()
        engine = ResponseEngine(audio_queue=audio_queue)
    """

    def __init__(
        self,
        tts: Optional[_TTSProtocol] = None,
        audio_queue: Optional[queue.Queue] = None,
        ar_overlay=None,
    ) -> None:
        self._narration_tts: Optional[_NarrationQueueTTS] = None

        if tts is not None:
            self._narration_tts = _NarrationQueueTTS(tts)
            self._tts: _TTSProtocol = self._narration_tts
        elif audio_queue is not None:
            self._tts = _QueueTTS(audio_queue)
        else:
            self._tts = _QueueTTS(queue.Queue())

        self._ar = ar_overlay
        self._mode: Mode = Mode.STANDBY
        self._wear_state: WearState = WearState.OFF_HEAD
        self._challenge: Optional[Challenge] = None
        self._story_context: Optional[StoryContext] = None

        # Deduplication: (DetectionType, label) → last narration timestamp
        self._last_spoken: dict[tuple, float] = {}
        self._last_encourage_at: float = 0.0

        # Reading mode: pending illustration description
        self._pending_illustration: Optional[str] = None

        # Multi-thread reader infrastructure
        self._reader_threads: list[tuple[threading.Thread, queue.Queue]] = []
        self._shutdown_event: threading.Event = threading.Event()

        logger.info(json.dumps({"event": "response_engine.init", "mode": self._mode.name}))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def wear_state(self) -> WearState:
        return self._wear_state

    @property
    def active_challenge(self) -> Optional[Challenge]:
        return self._challenge

    @property
    def challenge(self) -> Optional[Challenge]:
        return self._challenge

    @property
    def story_context(self) -> Optional[StoryContext]:
        return self._story_context

    # ------------------------------------------------------------------
    # State setters
    # ------------------------------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        previous = self._mode
        self._mode = mode
        self._last_spoken.clear()
        # Drop queued narration from the previous mode so the mask stops talking
        # about things that are no longer relevant once the mode changes.
        if mode != previous:
            self._tts.clear_pending()
        if mode == Mode.STORY and self._story_context is None:
            self._story_context = StoryContext()
        elif previous == Mode.STORY and mode != Mode.STORY:
            self._story_context = None
        logger.info(json.dumps({
            "event": "response_engine.mode_changed",
            "from": previous.name,
            "to": mode.name,
        }))

    def set_wear_state(self, state: WearState) -> None:
        self._wear_state = state

    def set_challenge(
        self,
        target_label: Optional[str] = None,
        target_type: Optional[ChallengeTargetType] = None,
        challenge: Optional[Challenge] = None,
        issued_at_override: Optional[float] = None,
    ) -> None:
        """Set active challenge. Accepts a pre-built Challenge or label+type args."""
        if challenge is not None:
            self._challenge = challenge
        else:
            issued_at = issued_at_override if issued_at_override is not None else time.monotonic()
            self._challenge = Challenge(
                target_type=target_type or ChallengeTargetType.COLOR,
                target_label=target_label or "",
                issued_at=issued_at,
                status=ChallengeStatus.ACTIVE,
            )
        self._last_encourage_at = 0.0
        logger.info(json.dumps({
            "event": "response_engine.challenge_set",
            "target_label": self._challenge.target_label,
        }))

    def set_story_context(self, ctx: Optional[StoryContext]) -> None:
        """Set or clear the active StoryContext (e.g. on page turn or book removal)."""
        self._story_context = ctx

    def set_pending_illustration(self, description: str) -> None:
        """Inject a pending illustration description for the next READING event."""
        self._pending_illustration = description

    @property
    def has_pending_illustration(self) -> bool:
        return self._pending_illustration is not None

    # ------------------------------------------------------------------
    # Multi-thread reader API
    # ------------------------------------------------------------------

    def add_capability_queue(self, q: queue.Queue) -> None:
        """Spawn a daemon reader thread that drains DetectionEvents from ``q``.

        The thread applies mode-tag validation: events whose ``origin_mode``
        doesn't match the current engine mode are discarded (AC-10). Events
        with ``origin_mode=None`` always pass through (legacy/backward compat).

        The thread exits when it receives a ``None`` sentinel OR when
        ``_shutdown_event`` is set.
        """
        t = threading.Thread(
            target=self._reader_loop,
            args=(q,),
            name="flec.capability_reader",
            daemon=True,
        )
        self._reader_threads.append((t, q))
        t.start()

    def _reader_loop(self, q: queue.Queue) -> None:
        """Internal loop run by each capability reader thread."""
        while not self._shutdown_event.is_set():
            try:
                event = q.get(timeout=0.05)
            except queue.Empty:
                continue
            if event is None:
                break
            if event.origin_mode is not None and event.origin_mode != self._mode:
                logger.debug(json.dumps({
                    "event": "response_engine.stale_discarded",
                    "origin_mode": event.origin_mode.name,
                    "current_mode": self._mode.name,
                }))
                continue
            self.on_event(event)

    def shutdown(self) -> None:
        """Stop all reader threads and the narration dispatcher.

        Sends a ``None`` sentinel to each capability queue to unblock reader
        threads, then joins them with a 1-second timeout each. Also shuts down
        the ``_NarrationQueueTTS`` dispatcher if the ``tts=`` path is active.
        """
        self._shutdown_event.set()
        # Unblock all reader threads via sentinel
        for _, q in self._reader_threads:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t, _ in self._reader_threads:
            t.join(timeout=1.0)
        # Shutdown narration dispatcher if active
        if self._narration_tts is not None:
            self._narration_tts.shutdown()

    # ------------------------------------------------------------------
    # Main event router
    # ------------------------------------------------------------------

    def on_event(self, event: DetectionEvent) -> None:
        """Route a DetectionEvent to audio/AR. Never raises."""
        try:
            self._route(event)
        except Exception as exc:
            logger.error(json.dumps({
                "event": "response_engine.routing_error",
                "detection_type": event.type.name,
                "error": str(exc),
            }))

    def _route(self, event: DetectionEvent) -> None:
        etype = event.type
        if etype == DetectionType.VOICE_CMD:
            self._handle_voice_cmd(event)
        elif etype in (DetectionType.SHAPE, DetectionType.COLOR, DetectionType.OBJECT):
            self._handle_perception(event)
        elif etype == DetectionType.WEAR:
            self._handle_wear(event)
        elif etype == DetectionType.FINGER:
            self._handle_finger(event)
        elif etype == DetectionType.TEXT:
            self._handle_text(event)
        elif etype == DetectionType.ILLUSTRATION:
            self._handle_illustration(event)
        else:
            logger.debug(json.dumps({
                "event": "response_engine.event_unhandled",
                "type": etype.name,
            }))

    # ------------------------------------------------------------------
    # Voice command handling
    # ------------------------------------------------------------------

    def _handle_voice_cmd(self, event: DetectionEvent) -> None:
        cmd = event.metadata.get("command") if event.metadata else None
        if cmd is None:
            return
        intent = cmd.intent
        if intent == CommandIntent.START_CHALLENGE:
            self._start_challenge_flow(cmd)
        elif intent == CommandIntent.CANCEL_CHALLENGE:
            self._cancel_challenge_flow()
        elif intent == CommandIntent.SHUTDOWN:
            self._shutdown_flow()
        elif intent == CommandIntent.REPEAT_CHALLENGE:
            self._repeat_challenge_flow()
        elif intent == CommandIntent.SWITCH_EXPLORATION:
            self._switch_mode_flow(Mode.EXPLORATION)
        elif intent == CommandIntent.SWITCH_READING:
            self._switch_mode_flow(Mode.READING)
        elif intent == CommandIntent.SWITCH_STORY:
            self._switch_mode_flow(Mode.STORY)
        elif intent == CommandIntent.SWITCH_CHALLENGE:
            self._switch_mode_flow(Mode.CHALLENGE)

    def _switch_mode_flow(self, mode: Mode) -> None:
        """Enter ``mode`` on a spoken mode-switch command and confirm audibly."""
        from flec.audio.responses import mode_switch_confirmation
        self.set_mode(mode)
        self._tts.speak(AudioResponse(
            text=mode_switch_confirmation(mode),
            priority=AudioPriority.HIGH,
        ))

    def _start_challenge_flow(self, cmd) -> None:
        from flec.audio.responses import challenge_acknowledgment
        target = cmd.target_label or "something"
        target_type = cmd.target_type or ChallengeTargetType.COLOR
        self.set_challenge(target_label=target, target_type=target_type)
        self.set_mode(Mode.CHALLENGE)
        self._tts.speak(AudioResponse(
            text=challenge_acknowledgment(target),
            priority=AudioPriority.HIGH,
        ))

    def _cancel_challenge_flow(self) -> None:
        if self._challenge is not None:
            self._challenge = Challenge(
                target_type=self._challenge.target_type,
                target_label=self._challenge.target_label,
                issued_at=self._challenge.issued_at,
                status=ChallengeStatus.CANCELLED,
            )
        self.set_mode(Mode.EXPLORATION)

    def _shutdown_flow(self) -> None:
        from flec.audio.responses import session_farewell
        if self._wear_state != WearState.ON_HEAD:
            logger.info(json.dumps({
                "event": "response_engine.shutdown_ignored",
                "reason": "mask_not_worn",
            }))
            return
        self._tts.speak(AudioResponse(text=session_farewell(), priority=AudioPriority.CRITICAL))

    def _repeat_challenge_flow(self) -> None:
        if self._challenge and self._challenge.status == ChallengeStatus.ACTIVE:
            from flec.audio.responses import challenge_hint
            self._tts.speak(AudioResponse(
                text=challenge_hint(self._challenge.target_label),
                priority=AudioPriority.HIGH,
            ))

    # ------------------------------------------------------------------
    # Perception handling (SHAPE / COLOR)
    # ------------------------------------------------------------------

    def _handle_perception(self, event: DetectionEvent) -> None:
        if self._mode == Mode.CHALLENGE:
            self._handle_challenge_detection(event)
        elif self._mode == Mode.EXPLORATION:
            self._handle_exploration_detection(event)
        else:
            logger.debug(json.dumps({
                "event": "response_engine.perception_ignored",
                "mode": self._mode.name,
                "label": event.label,
            }))

    def _handle_challenge_detection(self, event: DetectionEvent) -> None:
        from flec.audio.responses import challenge_celebration, challenge_encouraging, challenge_hint
        challenge = self._challenge
        if challenge is None or challenge.status != ChallengeStatus.ACTIVE:
            return
        now = time.monotonic()
        if now - challenge.issued_at >= 30.0 and self._should_encourage(now):
            self._tts.speak(AudioResponse(
                text=challenge_hint(challenge.target_label),
                priority=AudioPriority.HIGH,
            ))
            self._last_encourage_at = now
            return
        if self._is_match(event, challenge):
            self._tts.speak(AudioResponse(
                text=challenge_celebration(challenge.target_label),
                priority=AudioPriority.CRITICAL,
            ))
            self._challenge = Challenge(
                target_type=challenge.target_type,
                target_label=challenge.target_label,
                issued_at=challenge.issued_at,
                status=ChallengeStatus.COMPLETED,
            )
        else:
            if self._should_encourage(now):
                self._tts.speak(AudioResponse(
                    text=challenge_encouraging(),
                    priority=AudioPriority.NORMAL,
                ))
                self._last_encourage_at = now

    def _handle_exploration_detection(self, event: DetectionEvent) -> None:
        from flec.audio.responses import build_exploration_response, exploration_narration
        dedup_key = (event.type, event.label.lower())
        if self._is_recently_spoken(dedup_key):
            return
        try:
            response = build_exploration_response(event)
        except Exception:
            response = AudioResponse(
                text=exploration_narration(event.label),
                priority=AudioPriority.NORMAL,
                pre_cached=False,
            )
        self._tts.speak(response)
        self._mark_spoken(dedup_key)
        if self._ar is not None and event.bounding_box is not None:
            try:
                self._ar.draw_detection(None, event)
            except Exception:
                pass
        logger.info(json.dumps({
            "event": "response_engine.exploration_narrated",
            "label": event.label,
        }))

    # ------------------------------------------------------------------
    # Wear event handling
    # ------------------------------------------------------------------

    def _handle_wear(self, event: DetectionEvent) -> None:
        from flec.audio.responses import wear_off_prompt, wear_welcome
        if event.label == WearState.ON_HEAD.name:
            self._wear_state = WearState.ON_HEAD
            if self._mode == Mode.STANDBY:
                self.set_mode(Mode.EXPLORATION)
                self._tts.speak(AudioResponse(text=wear_welcome(), priority=AudioPriority.HIGH))
        elif event.label == WearState.OFF_HEAD.name:
            self._wear_state = WearState.OFF_HEAD
            self._tts.speak(AudioResponse(text=wear_off_prompt(), priority=AudioPriority.CRITICAL))
            self.set_mode(Mode.STANDBY)

    # ------------------------------------------------------------------
    # Finger tracking (Reading Mode)
    # ------------------------------------------------------------------

    def _handle_finger(self, event: DetectionEvent) -> None:
        """Route FINGER_TIP events in READING mode.

        - READING intent + nearest_text → NORMAL narration
        - READING intent + no text + is_illustration → pending illustration description
        - SCANNING/IDLE intent → no audio (AR trail handled externally)
        """
        if self._mode != Mode.READING:
            return
        meta = event.metadata or {}
        intent = meta.get("intent", ReadingIntent.IDLE)
        nearest_text: Optional[str] = meta.get("nearest_text")
        is_illustration: bool = meta.get("is_illustration", False)

        if intent != ReadingIntent.READING:
            logger.debug(json.dumps({
                "event": "response_engine.finger_scanning",
                "intent": intent.name if hasattr(intent, "name") else str(intent),
            }))
            return

        if nearest_text:
            dedup_key = (DetectionType.FINGER, nearest_text)
            if not self._is_recently_spoken(dedup_key):
                self._tts.speak(AudioResponse(text=nearest_text, priority=AudioPriority.NORMAL))
                self._mark_spoken(dedup_key)
                logger.info(json.dumps({"event": "response_engine.reading_narrate", "word": nearest_text}))
            return

        if is_illustration and self._pending_illustration:
            description = self._pending_illustration
            self._pending_illustration = None
            self._tts.speak(AudioResponse(text=description, priority=AudioPriority.NORMAL))
            logger.info(json.dumps({"event": "response_engine.illustration_narrate"}))

    # ------------------------------------------------------------------
    # Text / Illustration (Story Mode and Reading Mode)
    # ------------------------------------------------------------------

    def _handle_text(self, event: DetectionEvent) -> None:
        """Handle TEXT detection in STORY or READING mode."""
        if self._mode == Mode.STORY:
            self._handle_story_text(event)
        elif self._mode == Mode.READING:
            self._tts.speak(AudioResponse(text=event.label, priority=AudioPriority.NORMAL))

    def _handle_story_text(self, event: DetectionEvent) -> None:
        """Cursor-gated narration for STORY mode TEXT events."""
        if self._story_context is None:
            logger.debug(json.dumps({"event": "response_engine.story_text_dropped_no_context"}))
            return

        text = event.label.strip()
        if not text:
            return

        ctx = self._story_context
        words = text.split()
        start_pos = ctx.narrative_position

        if start_pos < len(words):
            remaining = " ".join(words[start_pos:])
            if remaining.strip():
                self._tts.speak(AudioResponse(text=remaining, priority=AudioPriority.NORMAL))
                logger.info(json.dumps({
                    "event": "response_engine.story_text_narrated",
                    "words_remaining": len(remaining.split()),
                    "confidence": round(event.confidence, 3),
                }))

    def _handle_illustration(self, event: DetectionEvent) -> None:
        """Handle ILLUSTRATION events."""
        if self._mode == Mode.STORY:
            if self._story_context is not None:
                description = event.label.strip()
                if description:
                    self._tts.speak(AudioResponse(text=description, priority=AudioPriority.NORMAL))
                    logger.info(json.dumps({
                        "event": "response_engine.story_illustration_described",
                        "word_count": len(description.split()),
                        "confidence": round(event.confidence, 3),
                    }))
            else:
                logger.debug(json.dumps({"event": "response_engine.illustration_dropped_no_context"}))
        else:
            # In READING mode: store as pending for next FINGER event
            self._pending_illustration = event.label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_match(self, event: DetectionEvent, challenge: Challenge) -> bool:
        target = challenge.target_label.lower().strip()
        label = event.label.lower().strip()

        # Color challenges: satisfied by a COLOR detection of that color, or by
        # any OBJECT whose dominant color matches (so "find something red" works
        # in the live YOLO-only pipeline, where a red cup counts).
        if challenge.target_type == ChallengeTargetType.COLOR:
            if event.type == DetectionType.COLOR:
                return label == target
            if event.type == DetectionType.OBJECT:
                obj_color = (event.metadata or {}).get("color")
                return bool(obj_color) and obj_color.lower() == target
            return False

        # Shape / object challenges: match the label, tolerating singular/plural.
        return label == target or label == target + "s" or target == label + "s"

    def _should_encourage(self, now: float) -> bool:
        return (now - self._last_encourage_at) >= _ENCOURAGE_THROTTLE_SECONDS

    def _is_recently_spoken(self, key: tuple) -> bool:
        last = self._last_spoken.get(key)
        if last is None:
            return False
        return (time.monotonic() - last) < _DEDUP_WINDOW_SECONDS

    def _mark_spoken(self, key: tuple) -> None:
        self._last_spoken[key] = time.monotonic()
