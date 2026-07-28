"""Smoke tests: fingertip reference fixture structure (flec-atl)."""
from pathlib import Path
import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "fingertip_reference"
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def test_fixture_dir_exists():
    assert FIXTURE_DIR.exists(), f"Fixture dir missing: {FIXTURE_DIR}"


def test_data_yaml_exists():
    assert (FIXTURE_DIR / "data.yaml").exists(), "data.yaml not found"


def test_labels_dir_exists():
    labels_dir = FIXTURE_DIR / "labels"
    assert labels_dir.is_dir(), f"labels/ dir missing: {labels_dir}"


def test_download_script_exists():
    script = SCRIPTS_DIR / "download_fingertip_fixtures.py"
    assert script.exists(), f"Download script missing: {script}"


def test_annotate_script_exists():
    script = SCRIPTS_DIR / "annotate_fingertip_fixtures.py"
    assert script.exists(), f"Annotation script missing: {script}"


def test_has_at_least_some_label_files_or_empty_is_ok():
    """Labels dir exists; images may not be present in CI (gitignored)."""
    labels_dir = FIXTURE_DIR / "labels"
    # Either empty (images not downloaded) or has .txt files — both are valid
    assert labels_dir.is_dir(), f"labels/ dir missing: {labels_dir}"


def test_label_files_have_yolo_pose_format():
    """Each .txt label must follow YOLO-pose format: 8 numeric tokens per line."""
    labels_dir = FIXTURE_DIR / "labels"
    txt_files = list(labels_dir.glob("*.txt"))
    if not txt_files:
        pytest.skip("No label files present (images not yet downloaded)")

    for lbl_path in txt_files:
        lines = [
            ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()
        ]
        for line in lines:
            tokens = line.split()
            assert len(tokens) == 8, (
                f"{lbl_path.name}: expected 8 tokens per line "
                f"(class cx cy w h kp_x kp_y kp_vis), got {len(tokens)}: {line!r}"
            )
            # class must be integer
            int(tokens[0])
            # all other values must be float
            for t in tokens[1:]:
                float(t)


def test_data_yaml_has_required_keys():
    """data.yaml must contain path, train, val, nc, names, kpt_shape."""
    import re

    content = (FIXTURE_DIR / "data.yaml").read_text()
    for key in ("path", "train", "val", "nc", "names", "kpt_shape"):
        assert key in content, f"data.yaml missing key: {key!r}"


def test_gitignore_excludes_images():
    """The .gitignore file must exclude the images/ directory."""
    gi = FIXTURE_DIR / ".gitignore"
    assert gi.exists(), ".gitignore not found"
    content = gi.read_text()
    assert "images/" in content, ".gitignore does not exclude images/"
