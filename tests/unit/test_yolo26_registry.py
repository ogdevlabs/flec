"""Smoke test for YOLO26_MODELS registry structure."""
from __future__ import annotations


def test_yolo26_registry_has_four_variants():
    from scripts.download_models import YOLO26_MODELS
    assert set(YOLO26_MODELS.keys()) == {"yolo26n", "yolo26n-pose", "yolo26n-seg", "yolo26n-obb"}


def test_yolo26_registry_entries_have_required_fields():
    from scripts.download_models import YOLO26_MODELS
    for name, spec in YOLO26_MODELS.items():
        assert "path" in spec, f"{name} missing path"
        assert "url" in spec, f"{name} missing url"
        assert "sha256" in spec, f"{name} missing sha256"
        assert "description" in spec, f"{name} missing description"
        assert spec["path"].endswith(".pt"), f"{name} path must end in .pt"
