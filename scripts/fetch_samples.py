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
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

API = "https://commons.wikimedia.org/w/api.php"

#: Wikimedia asks automated clients to identify themselves and say where to
#: complain. A generic urllib agent is rate-limited on sight.
USER_AGENT = "recyclevision-sample-fetcher/1.0 (https://github.com/encore488/recycle_vision)"

#: Pause between downloads. Courtesy, and it is also what keeps a five-file
#: fetch under the rate limit that stopped the first version mid-run.
COURTESY_DELAY = 1.0

#: A 429 is a request to wait, not a failure. Doubling from here.
RETRY_DELAYS = (2.0, 5.0, 15.0)

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


def _open(url: str, timeout: int, sleep=time.sleep):
    """GET a URL, treating 429 and 503 as "wait", not as failure.

    Wikimedia rate-limits by asking politely. Honouring Retry-After costs a
    few seconds; ignoring it cost a half-finished fetch with two images on
    disk and no attribution file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            return urllib.request.urlopen(request, timeout=timeout).read()  # noqa: S310
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or delay is None:
                raise
            # The server's own number wins over ours when it gives one.
            wait = delay
            header = exc.headers.get("Retry-After") if exc.headers else None
            if header and header.strip().isdigit():
                wait = max(wait, float(header.strip()))
            print(
                f"    rate-limited ({exc.code}), waiting {wait:.0f}s "
                f"[attempt {attempt + 1}/{len(RETRY_DELAYS) + 1}]"
            )
            sleep(wait)
    raise RuntimeError("unreachable")


def write_sources(out: Path, credits: list[tuple[str, dict[str, str]]]) -> Path:
    """Record where every fetched image came from and under what licence.

    Called even when the run fails partway. An image on disk without its
    credit line is the exact situation this script exists to prevent, and a
    crash is no excuse for producing one.
    """
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
    path = out / "SOURCES.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    payload = json.loads(_open(f"{API}?{query}", timeout=60))

    found = {}
    for entry in payload.get("query", {}).get("pages", {}).values():
        described = describe(entry)
        if described["url"]:
            found[described["title"]] = described
    return found


def download_all(
    wanted: list[tuple[str, str]],
    metadata: dict[str, dict[str, str]],
    out: Path,
    force: bool = False,
    sleep=time.sleep,
) -> tuple[list[tuple[str, dict[str, str]]], list[str]]:
    """Fetch each wanted file, returning what succeeded and what did not.

    One file failing does not abandon the rest, and never discards the credits
    for files already on disk -- the caller writes SOURCES.md from what comes
    back either way.
    """
    credits: list[tuple[str, dict[str, str]]] = []
    problems: list[str] = []

    for index, (title, filename) in enumerate(wanted):
        record = metadata.get(title)
        if record is None:
            problems.append(f"{title}: not found on Commons")
            continue
        if not is_allowed(record["licence"]):
            problems.append(f"{title}: licence {record['licence']!r} is not redistributable")
            continue

        destination = out / filename
        if destination.exists() and not force:
            print(f"  kept     {filename}")
            credits.append((filename, record))
            continue

        if index:
            sleep(COURTESY_DELAY)
        try:
            body = _open(record["url"], timeout=120, sleep=sleep)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            problems.append(f"{title}: download failed ({exc})")
            continue

        # Written via a temporary name: a half-downloaded .jpg left in images/
        # would be picked up by the app's glob and fail to open.
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(body)
        partial.replace(destination)
        print(f"  fetched  {filename}  ({record['licence']})")
        credits.append((filename, record))

    return credits, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("images"))
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = fetch_metadata([title for title, _name in WANTED])
    credits, problems = download_all(WANTED, metadata, args.out, force=args.force)

    if credits:
        # Before reporting problems: an image on disk without its credit line
        # is worse than a failed fetch, so this happens even on a partial run.
        print(f"\nwrote {write_sources(args.out, credits)} with {len(credits)} credit(s)")

    if problems:
        print("\nnot fetched:")
        for line in problems:
            print(f"  {line}")
        print("\nRe-run to retry — existing files are kept.")

    if not credits:
        print("\nnothing fetched")
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
