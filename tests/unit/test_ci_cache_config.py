"""Verify the CI workflow includes the model cache step (AC-18)."""
from __future__ import annotations

import pathlib


def test_ci_workflow_has_model_cache_step():
    """ci.yml must contain an actions/cache step for .models/ directory."""
    ci_path = pathlib.Path(__file__).parent.parent.parent / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), "ci.yml not found"
    content = ci_path.read_text()
    assert "actions/cache" in content, "ci.yml must use actions/cache"
    assert ".models/" in content, "cache path must include .models/"
    assert "model_checksums.json" in content or "download_models.py" in content, (
        "cache key must reference model files"
    )
