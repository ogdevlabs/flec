"""Integration test: capability threads must not open network sockets (T-006)."""
from __future__ import annotations

import socket
import pytest


def test_no_network_egress_from_ocr_reader():
    """OCRReader must not open network connections."""
    import numpy as np
    from flec.reading.ocr_reader import OCRReader

    original_connect = socket.socket.connect
    connections_attempted = []

    def patched_connect(self, address):
        connections_attempted.append(address)
        return original_connect(self, address)

    socket.socket.connect = patched_connect
    try:
        reader = OCRReader()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        reader.read_page(frame)
        if hasattr(reader, "shutdown"):
            reader.shutdown()
        elif hasattr(reader, "_executor"):
            reader._executor.shutdown(wait=False)
    finally:
        socket.socket.connect = original_connect

    assert not connections_attempted, (
        f"OCRReader opened network connections: {connections_attempted}"
    )


def test_no_network_egress_from_shape_color_detector():
    """ShapeColorDetector must not open network connections during detect()."""
    import numpy as np
    from flec.perception.shape_color_detector import ShapeColorDetector

    original_connect = socket.socket.connect
    connections_attempted = []

    def patched_connect(self, address):
        connections_attempted.append(address)
        return original_connect(self, address)

    socket.socket.connect = patched_connect
    try:
        detector = ShapeColorDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector.detect(frame)
    finally:
        socket.socket.connect = original_connect

    assert not connections_attempted, (
        f"ShapeColorDetector opened network connections: {connections_attempted}"
    )
