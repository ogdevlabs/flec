"""Smoke test for YOLO26_MODELS registry structure."""
from __future__ import annotations

import importlib.util
import pathlib


def _load_registry():
    spec = importlib.util.spec_from_file_location(
        "download_models",
        pathlib.Path(__file__).parent.parent.parent / "scripts" / "download_models.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.YOLO26_MODELS


def test_yolo26_registry_has_four_variants():
    models = _load_registry()
    assert set(models.keys()) == {"yolo26n", "yolo26n-pose", "yolo26n-seg", "yolo26n-obb"}


def test_yolo26_registry_entries_have_required_fields():
    models = _load_registry()
    for name, spec in models.items():
        assert "path" in spec, f"{name} missing path"
        assert "url" in spec, f"{name} missing url"
        assert "sha256" in spec, f"{name} missing sha256"
        assert "description" in spec, f"{name} missing description"
        assert spec["path"].endswith(".pt"), f"{name} path must end in .pt"
