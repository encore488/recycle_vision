"""Guards on the files the deployment host parses.

These are not exercised by anything local, so a mistake in them is invisible
until a deploy fails minutes later. That is exactly the kind of thing worth a
test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = PROJECT_ROOT / "packages.txt"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

#: Debian package names: lowercase alphanumerics plus . + - only.
APT_NAME = re.compile(r"^[a-z0-9][a-z0-9.+-]*$")


class TestPackagesTxt:
    """Streamlit Cloud passes every line of packages.txt to `apt-get install`.

    It does not strip comments, so a `#` line is not a comment -- it is an
    argument. A comment reading "(Manage app -> Reboot)" once broke a deploy
    with:

        E: Command line option '>' [from ->] is not understood

    Nothing but bare package names belongs in this file.
    """

    def test_exists(self):
        assert PACKAGES.is_file()

    @pytest.mark.parametrize("line_no,line", list(enumerate(PACKAGES.read_text().splitlines(), 1)))
    def test_every_line_is_a_bare_package_name(self, line_no, line):
        assert line == line.strip(), f"line {line_no}: leading/trailing whitespace"
        assert line, f"line {line_no}: blank line"
        assert not line.startswith("#"), (
            f"line {line_no}: packages.txt has no comment syntax -- apt receives this verbatim"
        )
        assert APT_NAME.match(line), f"line {line_no}: {line!r} is not a valid apt package name"

    def test_covers_the_opencv_shared_libraries(self):
        """Without these, ultralytics dies at import on a headless host."""
        names = set(PACKAGES.read_text().split())
        assert {"libgl1", "libglib2.0-0"} <= names


class TestRequirementsTxt:
    """pip *does* support comments here, unlike packages.txt."""

    def test_is_utf8_without_a_bom(self):
        # v0.2 shipped this file as UTF-16, which rendered as mojibake
        # everywhere except pip.
        raw = REQUIREMENTS.read_bytes()
        assert not raw.startswith(b"\xff\xfe") and not raw.startswith(b"\xef\xbb\xbf")
        raw.decode("utf-8")

    def test_pins_the_direct_dependencies(self):
        text = REQUIREMENTS.read_text()
        for package in ("streamlit", "ultralytics", "pillow", "pyyaml"):
            assert package in text, package


class TestVocabularyEmbeddings:
    """Editing a vocabulary without rebuilding its cache breaks the live app.

    `Vocabulary.load_embeddings` refuses a cache built for a different class
    list, which is right -- a silently misaligned cache would mislabel every
    detection. But the refusal happens at first inference, on the host, after
    a deploy that looked fine. Adding `beverage carton` to waste_v2 is exactly
    the edit that would have done it.

    Needs torch to read the cache, which CI does not install, so it skips
    rather than fails there. It still catches the mistake wherever the
    embeddings were actually built -- which is where the mistake gets made.
    """

    @pytest.mark.parametrize(
        "vocab_path",
        sorted((PROJECT_ROOT / "vocab").glob("*.yaml")),
        ids=lambda p: p.stem,
    )
    def test_cached_embeddings_match_the_class_list(self, vocab_path):
        torch = pytest.importorskip("torch", reason="cache is a torch pickle")
        import yaml

        cache = vocab_path.with_suffix(".pt")
        if not cache.is_file():
            pytest.skip(f"{vocab_path.name} ships no embeddings cache")

        declared = yaml.safe_load(vocab_path.read_text(encoding="utf-8"))["classes"]
        cached = torch.load(cache, weights_only=False)["classes"]
        assert cached == declared, (
            f"{cache.name} is stale. Rebuild it:\n"
            f"    python scripts/build_vocab_embeddings.py {vocab_path}"
        )
