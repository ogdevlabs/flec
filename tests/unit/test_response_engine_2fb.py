"""Tests for ResponseEngine.has_pending_illustration property (flec-2fb)."""

from flec.engine.response_engine import ResponseEngine


def test_has_pending_illustration_false_by_default():
    engine = ResponseEngine()
    assert engine.has_pending_illustration is False


def test_has_pending_illustration_true_when_set():
    engine = ResponseEngine()
    engine.set_pending_illustration("a little duck")
    assert engine.has_pending_illustration is True


def test_has_pending_illustration_false_after_clear():
    engine = ResponseEngine()
    engine.set_pending_illustration("a little duck")
    engine._pending_illustration = None  # simulate consumption
    assert engine.has_pending_illustration is False
