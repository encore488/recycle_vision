"""Guards against APIs newer than the declared minimum Python.

Twice now a 3.10+ API has shipped and failed on the maintainer's 3.9:
`zip(strict=True)`, and `Path.hardlink_to`. Both passed review, passed tests,
and passed CI — because CI ran only on 3.11.

CI now runs the declared minimum too, which is the real fix. This file is the
cheap second layer: it reads the source for names that do not exist on 3.9,
so the failure is a named test rather than a traceback in someone's terminal.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Attribute calls introduced after 3.9, with what to use instead.
TOO_NEW = {
    "hardlink_to": "os.link(src, dst) — Path.hardlink_to is 3.10+",
    "is_relative_to": "compare resolved parents — Path.is_relative_to is 3.9+, fine, but "
    "check the minimum before relying on it",
    "readlink": "os.readlink — Path.readlink is 3.9+",
    "removeprefix": "slicing — str.removeprefix is 3.9+",
    "pairwise": "zip(x, x[1:]) — itertools.pairwise is 3.10+",
}

#: Only these are actually forbidden at 3.9; the rest of TOO_NEW is 3.9-safe
#: and listed for the next person raising or lowering the floor.
FORBIDDEN_AT_39 = {"hardlink_to", "pairwise"}


def project_sources() -> list[Path]:
    roots = [PROJECT_ROOT / "recyclevision", PROJECT_ROOT / "scripts"]
    files = [PROJECT_ROOT / "train.py", PROJECT_ROOT / "app.py"]
    for root in roots:
        files += sorted(root.rglob("*.py"))
    return [f for f in files if f.is_file()]


@pytest.mark.parametrize("path", project_sources(), ids=lambda p: p.name)
def test_no_api_newer_than_the_minimum_python(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_AT_39
    ]
    assert not offenders, (
        f"{path.name} uses {sorted(set(offenders))}, newer than the declared minimum.\n"
        + "\n".join(f"  {name}: {TOO_NEW[name]}" for name in sorted(set(offenders)))
    )


def test_zip_is_never_called_with_strict():
    """`zip(strict=True)` is 3.10+, and it shipped once already."""
    for path in project_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "zip"
            ):
                assert not any(kw.arg == "strict" for kw in node.keywords), (
                    f"{path.name} calls zip(strict=...), which is 3.10+"
                )
