"""Guards on the documentation a fresh session relies on.

CLAUDE.md is read automatically at the start of every Claude Code session in
this repo, so it is the handoff between one session and the next. A dead link
or a renamed script in it is worse than no document: it sends someone
confidently to a file that is not there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS = (
    [PROJECT_ROOT / "README.md", PROJECT_ROOT / "ROADMAP.md", PROJECT_ROOT / "CLAUDE.md"]
    + sorted((PROJECT_ROOT / "docs").rglob("*.md"))
    + sorted((PROJECT_ROOT / "models").glob("*.md"))
)

LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(PROJECT_ROOT)))
def test_internal_links_resolve(doc):
    broken = []
    for target in LINK.findall(doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (doc.parent / target.split("#")[0]).resolve().exists():
            broken.append(target)
    assert not broken, f"{doc.name} links to missing files: {broken}"


def test_claude_md_exists_and_orients():
    """The one document a new session is guaranteed to read."""
    claude = PROJECT_ROOT / "CLAUDE.md"
    assert claude.is_file()
    text = claude.read_text(encoding="utf-8")
    # The two things a session must not have to rediscover.
    assert "routing accuracy" in text.lower(), "the headline metric is not named"
    assert "3.9" in text, "the Python floor is not stated"


@pytest.mark.parametrize(
    "script",
    [
        "inspect_dataset.py",
        "import_dataset.py",
        "build_pool.py",
        "build_eval_set.py",
        "evaluate.py",
        "zeroshot_eval.py",
        "prelabel.py",
        "extract_frames.py",
        "diagnose_eval.py",
        "fetch_samples.py",
        "build_vocab_embeddings.py",
    ],
)
def test_every_script_claude_md_names_exists(script):
    """CLAUDE.md lists the scripts by name; renaming one must break a test."""
    assert (PROJECT_ROOT / "scripts" / script).is_file()
    assert script[:-3] in (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
