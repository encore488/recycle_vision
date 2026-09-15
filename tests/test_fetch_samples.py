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
