from __future__ import annotations

import json
import logging
import threading

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.1   # 100ms
_MONITOR_JOIN_TIMEOUT = 1.0


class ThreadSupervisor:
    """Monitors CapabilityThread instances and restarts them on crash within 500ms."""

    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    @property
    def _threads(self) -> list[dict]:
        """Alias for _entries — used by tests to inspect supervised thread state."""
        return self._entries

    def add_thread(self, thread, factory_fn) -> None:
        """Register a thread for supervision. factory_fn() must return a fresh instance."""
        self._entries.append({
            "thread": thread,
            "factory": factory_fn,
            "intentional_stop": False,
        })

    def start(self) -> None:
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="ThreadSupervisor"
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        for entry in self._entries:
            entry["intentional_stop"] = True
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=_MONITOR_JOIN_TIMEOUT)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(timeout=_POLL_INTERVAL):
            for entry in self._entries:
                if entry["intentional_stop"]:
                    continue
                if not entry["thread"].is_alive():
                    self._restart(entry)

    def _restart(self, entry: dict) -> None:
        old_thread = entry["thread"]
        output_queue = old_thread._output_queue
        new_thread = entry["factory"]()
        new_thread._output_queue = output_queue
        new_thread.start()
        entry["thread"] = new_thread
        logger.info(json.dumps({
            "event": "vision_thread_restarted",
            "thread": old_thread.name,
        }))
