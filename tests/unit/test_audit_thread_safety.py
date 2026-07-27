"""Unit tests for flec-vhm: scripts/audit_thread_safety.py."""
from __future__ import annotations
import ast, json, subprocess, sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
AUDIT_SCRIPT = SCRIPTS_DIR / "audit_thread_safety.py"
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _run_audit(*args):
    """Run the audit script and return (returncode, stdout)."""
    result = subprocess.run(
        [sys.executable, str(AUDIT_SCRIPT), *args],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    return result.returncode, result.stdout, result.stderr


class TestAuditOnCurrentCodebase:
    def test_audit_exits_zero_on_clean_codebase(self):
        """Current codebase must pass the thread safety audit (AC-8)."""
        returncode, stdout, stderr = _run_audit()
        assert returncode == 0, \
            f"Audit found violations:\n{stdout}\nstderr: {stderr}"

    def test_audit_json_output_has_clean_true(self):
        """--json flag must output valid JSON with clean=true."""
        returncode, stdout, _ = _run_audit("--json")
        data = json.loads(stdout)
        assert data["clean"] is True, f"Expected clean=true, got: {data}"
        assert data["count"] == 0

    def test_audit_reports_pass_in_text_mode(self):
        """Default output must include [PASS] when no violations."""
        _, stdout, _ = _run_audit()
        assert "[PASS]" in stdout


class TestAuditRuleDetection:
    """Test that the audit rules fire on synthetic AST inputs."""

    def _parse(self, source: str, filename: str = "<test>") -> ast.AST:
        return ast.parse(source, filename=filename)

    def test_cross_module_import_detected(self):
        """Importing a capability module from another capability module must be flagged."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("audit", str(AUDIT_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Simulate detection_thread.py importing tracking_thread
        source = "from flec.perception import tracking_thread\n"
        tree = self._parse(source)
        fake_path = PROJECT_ROOT / "src" / "flec" / "perception" / "detection_thread.py"
        violations = mod.check_cross_module_imports(fake_path, tree)
        assert any(v.rule == "cross_module_import" for v in violations)

    def test_module_level_mutable_list_detected(self):
        """Module-level mutable list in a thread module must be flagged."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("audit", str(AUDIT_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        source = "_pending_events = []\n"
        tree = self._parse(source)
        fake_path = PROJECT_ROOT / "src" / "flec" / "perception" / "detection_thread.py"
        violations = mod.check_module_level_mutable(fake_path, tree)
        assert any(v.rule == "module_level_mutable" for v in violations)

    def test_constant_list_not_flagged(self):
        """ALL_CAPS module-level list must NOT be flagged."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("audit", str(AUDIT_SCRIPT))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        source = "VALID_CLASSES = ['cat', 'dog']\n"
        tree = self._parse(source)
        fake_path = PROJECT_ROOT / "src" / "flec" / "perception" / "detection_thread.py"
        violations = mod.check_module_level_mutable(fake_path, tree)
        assert not violations, f"Unexpected violations: {violations}"
