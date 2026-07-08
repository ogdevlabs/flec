"""Integration test fixtures — session-level mocks and simulated event sequences."""

import queue

import pytest


@pytest.fixture
def session_event_queue() -> queue.Queue:
    """Return an event queue configured for integration test session simulation."""
    return queue.Queue(maxsize=500)
