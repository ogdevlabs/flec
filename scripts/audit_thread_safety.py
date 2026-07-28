#!/usr/bin/env python3
"""Thread safety audit for Flec capability modules.

AST-based static analysis that detects:
  - Cross-module imports between capability thread modules (queue-only contract violation)
  - Module-level mutable state in thread modules (shared-state risk)
  - Direct underscore-attribute access across threads (encapsulation violation)

Usage:
    python scripts/audit_thread_safety.py
    python scripts/audit_thread_safety.py --json   # machine-readable JSON output
    python scripts/audit_thread_safety.py --fix    # (future: auto-fix suggestions)

Exit code: 0 = clean, 1 = violations found
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import NamedTuple

# Root of the project (two levels up from scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
SRC_FLEC = PROJECT_ROOT / "src" / "flec"

# Capability modules subject to queue-only contract
CAPABILITY_MODULES = {
    "shape_color_detector",
    "finger_tracker",
    "detection_thread",
    "segmentation_thread",
    "tracking_thread",
    "obb_thread",
    "depth_thread",
    "semantic_thread",
    "ocr_reader",
    "ocr_worker",
    "illustration_describer",
}


class Violation(NamedTuple):
    file: str
    line: int
    rule: str
    detail: str


def _module_name(path: Path) -> str:
    return path.stem


def _is_capability(path: Path) -> bool:
    return _module_name(path) in CAPABILITY_MODULES


def _collect_sources() -> list[Path]:
    return [p for p in SRC_FLEC.rglob("*.py") if p.is_file()]


def _top_level_imports(tree: ast.AST):
    """Yield only module-level import statements (not inside functions or classes).

    Lazy imports inside methods are the approved pattern for heavy ML deps and
    are NOT a queue-only contract violation — only top-level imports are checked.
    """
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def check_cross_module_imports(path: Path, tree: ast.AST) -> list[Violation]:
    """Detect capability module importing another capability module at module level."""
    violations = []
    mod_name = _module_name(path)
    if mod_name not in CAPABILITY_MODULES:
        return violations

    for node in _top_level_imports(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Gather all candidate module name tokens to check.
            # For `import a.b.c` → check "a.b.c"
            # For `from a.b import c, d` → check "a.b" (module) AND "c", "d" (names)
            # because `from flec.perception import tracking_thread` puts the
            # capability module name in node.names, not node.module.
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = []
                if node.module:
                    names.append(node.module)
                # Also include the imported symbol names (handles `from pkg import cap_mod`)
                names.extend(alias.name for alias in node.names)

            for name in names:
                # Check if any segment of the dotted module name is a capability module
                parts = name.split(".")
                for part in parts:
                    if part in CAPABILITY_MODULES and part != mod_name:
                        violations.append(Violation(
                            file=str(path.relative_to(PROJECT_ROOT)),
                            line=node.lineno,
                            rule="cross_module_import",
                            detail=f"{mod_name} imports capability module '{part}' "
                                   f"(queue-only contract violation)",
                        ))
                        break

    return violations


def check_module_level_mutable(path: Path, tree: ast.AST) -> list[Violation]:
    """Detect module-level mutable assignments (non-constant names) in thread modules."""
    violations = []
    mod_name = _module_name(path)
    if not mod_name.endswith("_thread") and mod_name not in CAPABILITY_MODULES:
        return violations

    # Only look at top-level Assign nodes (not inside class/function)
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        # Get target names
        targets = []
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets.append(node.target.id)

        for name in targets:
            # Constants are ALL_CAPS or start with _ (module-private constants)
            if name.isupper() or name.startswith("__"):
                continue
            # logger is a well-known module-level singleton — not mutable state
            if name in ("logger",):
                continue
            # Check if value is a mutable literal (list, dict, set) — those are risky
            value = node.value if isinstance(node, ast.Assign) else node.value
            if value is None:
                continue
            if isinstance(value, (ast.List, ast.Dict, ast.Set)):
                violations.append(Violation(
                    file=str(path.relative_to(PROJECT_ROOT)),
                    line=node.lineno,
                    rule="module_level_mutable",
                    detail=f"Module-level mutable {type(value).__name__} '{name}' "
                           f"in thread module (shared-state risk)",
                ))

    return violations


def check_private_attr_cross_access(path: Path, tree: ast.AST) -> list[Violation]:
    """Detect direct access to _private attributes on objects named *thread*."""
    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        attr = node.attr
        # Only care about _private (single underscore) accesses
        if not (attr.startswith("_") and not attr.startswith("__")):
            continue
        # Check if the object being accessed looks like a thread reference
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if "thread" in var_name.lower() and var_name != "self":
                # Accessing _private attr on another thread — potential encapsulation break
                # Exception: _input_queue and _output_queue are the documented contract
                if attr not in ("_input_queue", "_output_queue", "_stop_event"):
                    violations.append(Violation(
                        file=str(path.relative_to(PROJECT_ROOT)),
                        line=node.lineno,
                        rule="cross_thread_private_access",
                        detail=f"Direct access to '{attr}' on '{var_name}' "
                               f"(use get_output_queue() or documented API instead)",
                    ))

    return violations


def audit() -> list[Violation]:
    sources = _collect_sources()
    all_violations: list[Violation] = []

    for path in sorted(sources):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            all_violations.append(Violation(
                file=str(path.relative_to(PROJECT_ROOT)),
                line=e.lineno or 0,
                rule="syntax_error",
                detail=str(e),
            ))
            continue

        all_violations.extend(check_cross_module_imports(path, tree))
        all_violations.extend(check_module_level_mutable(path, tree))
        all_violations.extend(check_private_attr_cross_access(path, tree))

    return all_violations


def main() -> int:
    use_json = "--json" in sys.argv

    violations = audit()

    report = {
        "violations": [v._asdict() for v in violations],
        "count": len(violations),
        "clean": len(violations) == 0,
    }

    if use_json:
        print(json.dumps(report, indent=2))
    else:
        if violations:
            print(f"[FAIL] Thread safety audit: {len(violations)} violation(s) found\n")
            for v in violations:
                print(f"  {v.file}:{v.line}  [{v.rule}]  {v.detail}")
            print()
        else:
            print("[PASS] Thread safety audit: 0 violations found")

    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
