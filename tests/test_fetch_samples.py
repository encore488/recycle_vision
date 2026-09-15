"""Licence gating and credit extraction for the demo-image fetcher.

The fetch itself needs Commons and is not exercised here. The parts that
decide what lands in a public repository are, because "a demo image nobody
can account for" is exactly the kind of thing that gets committed once and
noticed years later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_samples  # noqa: E402


def download_all_quiet(wanted, metadata, out, **kwargs):
    """download_all with the courtesy delay skipped, so tests do not sleep."""
    return fetch_samples.download_all(wanted, metadata, out, sleep=lambda _s: None, **kwargs)


class TestLicenceGate:
    @pytest.mark.parametrize(
        "licence",
        ["CC0", "Public domain", "CC BY 4.0", "CC BY-SA 3.0", "cc-by-sa-2.0", "PD-USGov"],
    )
    def test_redistributable_licences_pass(self, licence):
        assert fetch_samples.is_allowed(licence)

    @pytest.mark.parametrize(
        "licence",
        ["CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-NC-SA 3.0", "All rights reserved", "Fair use", ""],
    )
    def test_restricted_licences_are_refused(self, licence):
        assert not fetch_samples.is_allowed(licence)

    def test_noncommercial_beats_the_cc_by_prefix(self):
        # "CC BY-NC" contains "cc by"; matching that first would let every
        # NonCommercial file through, which is the whole failure mode.
        assert not fetch_samples.is_allowed("CC BY-NC-SA 4.0")


class TestDescribe:
    @staticmethod
    def _entry(licence="CC BY-SA 4.0", artist='<a href="/wiki/User:X">Jane Doe</a>'):
        return {
            "title": "File:Example.jpg",
            "imageinfo": [
                {
                    "url": "https://upload.wikimedia.org/example.jpg",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Example.jpg",
                    "extmetadata": {
                        "LicenseShortName": {"value": licence},
                        "Artist": {"value": artist},
                    },
                }
            ],
        }

    def test_html_is_stripped_from_the_credit(self):
        assert fetch_samples.describe(self._entry())["author"] == "Jane Doe"

    def test_licence_and_urls_come_through(self):
        described = fetch_samples.describe(self._entry())
        assert described["licence"] == "CC BY-SA 4.0"
        assert described["url"].endswith("example.jpg")
        assert "commons.wikimedia.org" in described["descriptionurl"]

    def test_a_missing_author_is_named_not_blank(self):
        assert fetch_samples.describe(self._entry(artist=""))["author"] == "unknown"

    def test_an_entry_with_no_imageinfo_does_not_raise(self):
        described = fetch_samples.describe({"title": "File:Missing.jpg"})
        assert described["url"] == ""

    def test_nested_markup_is_flattened(self):
        entry = self._entry(artist="<span><b>Ann</b> &amp; <i>Bo</i></span>")
        assert fetch_samples.describe(entry)["author"] == "Ann &amp; Bo"


def test_every_wanted_file_has_a_distinct_local_name():
    names = [name for _title, name in fetch_samples.WANTED]
    assert len(names) == len(set(names))


def test_wanted_titles_are_commons_file_pages():
    for title, _name in fetch_samples.WANTED:
        assert title.startswith("File:")


def _record(name="Example", licence="CC BY-SA 4.0"):
    return {
        "title": f"File:{name}.jpg",
        "url": f"https://upload.wikimedia.org/{name}.jpg",
        "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{name}.jpg",
        "licence": licence,
        "author": "Jane Doe",
    }


class TestPartialFailure:
    """The observed failure: two images fetched, the third 429'd, the whole
    run raised, and SOURCES.md was never written — leaving images in the
    repository with no licence or attribution recorded anywhere.
    """

    def test_one_failure_does_not_abandon_the_rest(self, tmp_path, monkeypatch):
        wanted = [("File:A.jpg", "a.jpg"), ("File:B.jpg", "b.jpg"), ("File:C.jpg", "c.jpg")]
        metadata = {t: _record(t[5:-4]) for t, _n in wanted}

        def fake_open(url, timeout, sleep=None):
            if url.endswith("B.jpg"):
                raise OSError("429 Too many requests")
            return b"\xff\xd8image"

        monkeypatch.setattr(fetch_samples, "_open", fake_open)
        credits, problems = download_all_quiet(wanted, metadata, tmp_path)

        assert [name for name, _r in credits] == ["a.jpg", "c.jpg"]
        assert len(problems) == 1 and "B.jpg" in problems[0]
        assert (tmp_path / "c.jpg").exists(), "a later file must still be attempted"

    def test_every_fetched_file_gets_a_credit_line(self, tmp_path, monkeypatch):
        wanted = [("File:A.jpg", "a.jpg"), ("File:B.jpg", "b.jpg")]
        metadata = {t: _record(t[5:-4]) for t, _n in wanted}
        monkeypatch.setattr(fetch_samples, "_open", lambda url, timeout, sleep=None: b"x")

        credits, _problems = download_all_quiet(wanted, metadata, tmp_path)
        text = fetch_samples.write_sources(tmp_path, credits).read_text(encoding="utf-8")

        images = {p.name for p in tmp_path.glob("*.jpg")}
        for name in images:
            assert f"`{name}`" in text, f"{name} is on disk with no attribution"

    def test_sources_says_these_are_not_test_data(self, tmp_path):
        text = fetch_samples.write_sources(tmp_path, [("a.jpg", _record())]).read_text()
        assert "Nothing is measured against these" in text

    def test_a_failed_download_leaves_no_partial_file(self, tmp_path, monkeypatch):
        # A truncated .jpg in images/ would be picked up by the app's glob.
        monkeypatch.setattr(
            fetch_samples,
            "_open",
            lambda url, timeout, sleep=None: (_ for _ in ()).throw(OSError()),
        )
        download_all_quiet([("File:A.jpg", "a.jpg")], {"File:A.jpg": _record("A")}, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_existing_files_are_kept_and_still_credited(self, tmp_path, monkeypatch):
        (tmp_path / "a.jpg").write_bytes(b"already here")
        monkeypatch.setattr(
            fetch_samples, "_open", lambda url, timeout, sleep=None: b"should not be used"
        )
        credits, _p = download_all_quiet(
            [("File:A.jpg", "a.jpg")], {"File:A.jpg": _record("A")}, tmp_path
        )
        assert (tmp_path / "a.jpg").read_bytes() == b"already here"
        assert credits, "a kept file still needs its credit line"


class TestRateLimitRetry:
    def test_a_429_is_waited_out_then_succeeds(self, monkeypatch):
        import urllib.error

        slept, attempts = [], {"n": 0}

        def flaky(request, timeout):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise urllib.error.HTTPError("u", 429, "Too many requests", None, None)

            class R:
                def read(self):
                    return b"ok"

            return R()

        monkeypatch.setattr(fetch_samples.urllib.request, "urlopen", flaky)
        body = fetch_samples._open("https://example.invalid/x", timeout=1, sleep=slept.append)
        assert body == b"ok"
        assert slept, "a 429 must be waited out, not retried immediately"

    def test_a_404_is_not_retried(self, monkeypatch):
        import urllib.error

        calls = {"n": 0}

        def missing(request, timeout):
            calls["n"] += 1
            raise urllib.error.HTTPError("u", 404, "Not Found", None, None)

        monkeypatch.setattr(fetch_samples.urllib.request, "urlopen", missing)
        with pytest.raises(urllib.error.HTTPError):
            fetch_samples._open("https://example.invalid/x", timeout=1, sleep=lambda _s: None)
        assert calls["n"] == 1, "only rate limits are worth retrying"
