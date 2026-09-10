"""Turning results into rows, and rows into files.

The app shows a `SortResult`; a person auditing a waste stream needs it in a
spreadsheet. This module is the one place that decides what a row looks like,
so the CSV, the JSON export and the on-screen table cannot drift apart.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable

from .models import SortResult

#: Column order for tabular exports. Explicit rather than derived from a dict,
#: so a spreadsheet someone has built formulas against does not reshuffle when
#: a field is added.
COLUMNS = [
    "source",
    "item",
    "detected_as",
    "bin",
    "bin_name",
    "diverted",
    "material",
    "confidence",
    "certainty",
    "needs_review",
    "handling",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
    "area_px",
]


def detection_rows(result: SortResult, source: str = "") -> list[dict]:
    """One row per routed item, geometry included."""
    rows = []
    for item in result.items:
        box = item.detection.box
        rows.append(
            {
                "source": source,
                "item": item.label,
                "detected_as": item.detection.label,
                "bin": item.bin.key,
                "bin_name": item.bin.name,
                "diverted": item.bin.diverted,
                "material": item.material,
                "confidence": round(item.confidence, 4),
                "certainty": item.certainty.value,
                "needs_review": item.needs_review,
                "handling": item.handling,
                "x1": round(box.x1, 1),
                "y1": round(box.y1, 1),
                "x2": round(box.x2, 1),
                "y2": round(box.y2, 1),
                "width": round(box.width, 1),
                "height": round(box.height, 1),
                "area_px": round(box.area, 1),
            }
        )
    return rows


def to_csv(rows: Iterable[dict]) -> str:
    """Rows as CSV text.

    Always writes the header, even for no rows: a spreadsheet with column
    names and nothing under them is a readable "we found nothing", whereas an
    empty file looks like a failure.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def result_payload(result: SortResult, source: str = "") -> dict:
    """A full result as a JSON-ready mapping."""
    return {
        "source": source,
        "model": result.model_name,
        "policy": result.policy_name,
        "total_items": result.total_items,
        "diverted_count": result.diverted_count,
        "diversion_rate": round(result.diversion_rate, 4),
        "contamination_rate": round(result.contamination_rate, 4),
        "average_confidence": round(result.average_confidence, 4),
        "needs_review": len(result.items_for_review),
        "counts_by_bin": dict(result.counts_by_bin),
        "counts_by_material": dict(result.counts_by_material),
        "ignored_classes": sorted({d.label for d in result.ignored}),
        "items": detection_rows(result, source),
    }


def to_json(payloads: list[dict]) -> str:
    return json.dumps(payloads, indent=2) + "\n"
