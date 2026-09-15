"""Download demo images for the front end from Wikimedia Commons.

    python scripts/fetch_samples.py

These images exist to make a demo quick to run. They are **not** test data and
nothing should be measured against them: there is no ground truth for them and
none is coming. The project's measurements come from a held-out split of a
labelled dataset (see docs/DATA_CARD.md), which is the only place a number
should ever be quoted from.

Why a script instead of committed files: licence provenance. Every image here
is fetched by title from Commons, and its licence and author are read from the
API at download time and written to images/SOURCES.md. A file whose licence is
not on the allow-list below is skipped rather than quietly included, so the
repository cannot accumulate imagery nobody can account for.

Re-runnable: existing files are left alone unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API = "https://commons.wikimedia.org/w/api.php"

#: Sorting-line and MRF imagery: belts, mixed streams, the domain the model is
#: for. Titles rather than URLs, so the licence is resolved live rather than
#: trusted from whenever this list was written.
WANTED = [
    ("File:Material recovery facility 2004-03-24.jpg", "mrf_line.jpg"),
    ("File:Materials recovery facility 2.jpg", "mrf_sorting.jpg"),
    ("File:Greenville Public Works, ECVC Recycling Sorting facility - 10.jpg", "mrf_handsort.jpg"),
    ("File:Single stream recycling.jpg", "mrf_singlestream.jpg"),
    ("File:Recycling plant conveyor belt.jpg", "mrf_conveyor.jpg"),
]

#: Licences that permit redistribution with attribution. Anything carrying
#: NonCommercial or NoDerivatives is refused: this repository is public, and a
#: demo image is never worth a licence question.
ALLOWED = ("cc0", "public domain", "cc by", "cc-by", "pd-")
REFUSED = ("nc", "nd", "noncommercial", "noderiv")


def is_allowed(licence: str) -> bool:
    """Whether a licence string permits redistribution with attribution."""
    lowered = licence.lower()
    if any(
        bad in lowered.replace(" ", "") for bad in ("by-nc", "by-nd", "noncommercial", "noderiv")
    ):
        return False
    return any(good in lowered for good in ALLOWED)


def describe(entry: dict) -> dict[str, str]:
    """Pull the fields worth recording out of one imageinfo payload."""
    info = (entry.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata") or {}

    def field(name: str) -> str:
        raw = str(meta.get(name, {}).get("value", "")).strip()
        # extmetadata returns HTML; a demo credit line does not need markup.
        out, depth = [], 0
        for char in raw:
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(0, depth - 1)
            elif depth == 0:
                out.append(char)
        return " ".join("".join(out).split())

    return {
        "title": entry.get("title", ""),
        "url": info.get("url", ""),
        "descriptionurl": info.get("descriptionurl", ""),
        "licence": field("LicenseShortName"),
        "author": field("Artist") or "unknown",
    }


def fetch_metadata(titles: list[str]) -> dict[str, dict[str, str]]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": "|".join(titles),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": "recyclevision-sample-fetcher/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed host
        payload = json.load(response)

    found = {}
    for entry in payload.get("query", {}).get("pages", {}).values():
        described = describe(entry)
        if described["url"]:
            found[described["title"]] = described
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("images"))
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata([title for title, _name in WANTED])

    credits, skipped = [], []
    for title, filename in WANTED:
        record = metadata.get(title)
        if record is None:
            skipped.append(f"{title}: not found on Commons")
            continue
        if not is_allowed(record["licence"]):
            skipped.append(f"{title}: licence {record['licence']!r} is not redistributable")
            continue

        destination = args.out / filename
        if destination.exists() and not args.force:
            print(f"  kept     {filename}")
        else:
            request = urllib.request.Request(
                record["url"], headers={"User-Agent": "recyclevision-sample-fetcher/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                destination.write_bytes(response.read())
            print(f"  fetched  {filename}  ({record['licence']})")
        credits.append((filename, record))

    if skipped:
        print("\nskipped:")
        for line in skipped:
            print(f"  {line}")

    if not credits:
        print("\nnothing fetched — leaving images/SOURCES.md alone")
        return 1

    lines = [
        "# Sample image sources",
        "",
        "Demo images only. **Nothing is measured against these** — they have no",
        "ground truth and none is planned. Project numbers come from a held-out",
        "split of a labelled dataset; see [docs/DATA_CARD.md](../docs/DATA_CARD.md).",
        "",
        "Fetched and licence-checked by `scripts/fetch_samples.py`.",
        "",
        "| file | source | licence | author |",
        "| --- | --- | --- | --- |",
    ]
    for filename, record in sorted(credits):
        page = record["descriptionurl"] or record["url"]
        lines.append(
            f"| `{filename}` | [Commons]({page}) | {record['licence']} | {record['author']} |"
        )
    (args.out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out / 'SOURCES.md'} with {len(credits)} credit(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
