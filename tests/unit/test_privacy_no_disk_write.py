"""AC-9 privacy test: perception pipeline must never write files to disk.

No frames, crops, text snippets, or debug dumps may be written to disk
during process_frame(). This is a hard requirement for a child-facing device.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest.mock as mock
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_finger_state() -> MagicMock:
    """Return a MagicMock that looks like a FingerTrackingState with IDLE intent."""
    state = MagicMock()
    state.detected = False
    state.intent = MagicMock()
    state.intent.name = "IDLE"
    state.nearest_text = None
    state.position_x = 0.0
    state.position_y = 0.0
    state.velocity = 0.0
    return state


def _make_session_with_mocks():
    """Instantiate FlecSession with all heavy deps stubbed out.

    Patches at the module level where FlecSession's __init__ lazily imports from.
    TTSEngine is patched via flec.audio.tts so TTS never starts a real audio thread.
    voice=False skips _start_mic entirely.
    """
    stub_state = _make_stub_finger_state()

    mock_finger_tracker = MagicMock()
    mock_finger_tracker.return_value.update.return_value = stub_state
    mock_finger_tracker.return_value.current_state = stub_state

    mock_shape_detector = MagicMock()
    mock_shape_detector.return_value.detect.return_value = []

    mock_tts_engine = MagicMock()
    mock_response_engine = MagicMock()
    mock_response_engine.return_value.mode = MagicMock()

    with (
        mock.patch("flec.perception.finger_tracker.FingerTracker", mock_finger_tracker),
        mock.patch(
            "flec.perception.shape_color_detector.ShapeColorDetector",
            mock_shape_detector,
        ),
        mock.patch("flec.audio.tts.TTSEngine", mock_tts_engine),
        mock.patch("flec.engine.response_engine.ResponseEngine", mock_response_engine),
    ):
        # Re-import FlecSession inside the patch context so the lazy imports inside
        # __init__ resolve to the mocked classes.
        import importlib
        import flec.main as _main_mod

        importlib.reload(_main_mod)
        FlecSession = _main_mod.FlecSession
        session = FlecSession(voice=False)

    return session


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_tmpdir() -> Generator[str, None, None]:
    """Create a temp directory, yield its path, and clean up afterwards.

    Also changes the working directory to the temp dir for the duration of
    the test so any accidental relative-path writes land there.
    """
    original_cwd = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="flec_privacy_test_")
    os.chdir(tmpdir)
    try:
        yield tmpdir
    finally:
        os.chdir(original_cwd)
        # Clean up (best-effort; don't mask test failures).
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoDiskWrite:
    """AC-9: process_frame() must not create any files on disk."""

    def test_process_frame_writes_no_files_to_tmpdir(
        self, isolated_tmpdir: str
    ) -> None:
        """Calling process_frame() must not create files in the current directory."""
        session = _make_session_with_mocks()

        before = set(os.listdir(isolated_tmpdir))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame)

        after = set(os.listdir(isolated_tmpdir))
        new_files = after - before

        assert not new_files, (
            f"process_frame() wrote unexpected files to disk: {new_files}"
        )

    def test_process_frame_writes_no_files_to_project_root(self) -> None:
        """Calling process_frame() must not modify the project root directory."""
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        root_before = set(os.listdir(project_root))

        session = _make_session_with_mocks()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame)

        root_after = set(os.listdir(project_root))
        new_entries = root_after - root_before

        assert not new_entries, (
            f"process_frame() created unexpected entries in project root: {new_entries}"
        )

    def test_process_frame_writes_no_files_to_tmp(
        self, isolated_tmpdir: str
    ) -> None:
        """Calling process_frame() must not create new files in /tmp."""
        tmp_dir = tempfile.gettempdir()
        # Snapshot /tmp *before* creating the session (session init could also write).
        before_mtime_threshold = time.time()

        session = _make_session_with_mocks()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame)

        # Check for any files in /tmp created strictly during process_frame.
        new_in_tmp = []
        try:
            for entry in os.scandir(tmp_dir):
                if entry.stat().st_mtime >= before_mtime_threshold:
                    new_in_tmp.append(entry.name)
        except PermissionError:
            # Some /tmp entries may be unreadable — skip those.
            pass

        # Filter out the temp dir we created for this test run.
        new_in_tmp = [
            n for n in new_in_tmp
            if not n.startswith("flec_privacy_test_")
        ]

        assert not new_in_tmp, (
            f"process_frame() caused new files to appear in {tmp_dir}: {new_in_tmp}"
        )

    def test_multiple_frames_write_no_files(self, isolated_tmpdir: str) -> None:
        """Repeated calls to process_frame() across multiple frames write nothing."""
        session = _make_session_with_mocks()

        before = set(os.listdir(isolated_tmpdir))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            session.process_frame(frame)

        after = set(os.listdir(isolated_tmpdir))
        new_files = after - before

        assert not new_files, (
            f"10 consecutive process_frame() calls wrote files to disk: {new_files}"
        )

    def test_process_frame_with_ocr_result_writes_no_files(
        self, isolated_tmpdir: str
    ) -> None:
        """process_frame() with an ocr_result argument must also write nothing."""
        session = _make_session_with_mocks()

        before = set(os.listdir(isolated_tmpdir))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        session.process_frame(frame, ocr_result=["hello", "world"])

        after = set(os.listdir(isolated_tmpdir))
        new_files = after - before

        assert not new_files, (
            f"process_frame(ocr_result=...) wrote unexpected files: {new_files}"
        )
