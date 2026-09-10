"""Headless entry point: `python -m recyclevision IMAGE [IMAGE ...]`.

Exists so the pipeline can be exercised, scripted and debugged without
booting Streamlit -- and so that "does it actually work?" has an answer that
does not involve a browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .models import SortResult
from .pipeline import DEFAULT_POLICY, SortingPipeline
from .render import annotate
from .report import detection_rows, result_payload, to_csv, to_json
from .session import Session
from .vocabulary import DEFAULT_VOCAB


def _print_human(result: SortResult, source: str) -> None:
    print(f"\n{source}")
    print(f"  model  {result.model_name}")
    print(f"  policy {result.policy_name}")

    if not result.items:
        print("  no waste items detected")
    else:
        print(f"  {result.total_items} item(s), {result.diversion_rate:.0%} diverted from landfill")
        for bin_ in result.bins_used:
            members = result.items_in(bin_.key)
            print(f"\n  {bin_.name} ({len(members)})")
            for item in members:
                flag = "  [review]" if item.needs_review else ""
                print(f"    - {item.label}  {item.confidence:.0%}{flag}")
                if item.handling:
                    print(f"        {item.handling}")

    if result.ignored:
        seen = sorted({d.label for d in result.ignored})
        print(f"\n  ignored (not waste): {', '.join(seen)}")


def _print_impact(session: Session, indent: str = "  ") -> None:
    from .impact import ImpactModel

    estimate = ImpactModel.load().estimate(session.as_result())
    print(f"\n{indent}mass          {estimate.total_mass_kg:.2f} kg")
    print(f"{indent}diverted      {estimate.diverted_mass_kg:.2f} kg")
    print(f"{indent}CO2e avoided  {estimate.co2e_avoided_kg:.2f} kg")
    if estimate.coverage_note:
        print(f"{indent}{estimate.coverage_note}")
    if estimate.caveat:
        print(f"{indent}! {estimate.caveat}")


def _print_totals(session: Session, impact: bool) -> None:
    print(f"\n{'=' * 60}")
    print(
        f"{session.image_count} image(s), {session.total_items} item(s), "
        f"{session.diversion_rate:.0%} diverted from landfill"
    )

    for bin_ in session.bins_used:
        print(f"\n  {bin_.name} ({len(session.items_in(bin_.key))})")

    composition = session.counts_by_item.most_common()
    if composition:
        print("\n  stream composition:")
        for name, count in composition:
            print(f"    {count:4}  {name}")

    if session.items_for_review:
        print(f"\n  {len(session.items_for_review)} item(s) need review")

    if impact:
        _print_impact(session)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recyclevision",
        description="Sort waste in an image into bins.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="image files to sort")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="routing policy YAML")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="COCO model weights (.pt); implies the closed-set detector",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="use the stock COCO detector instead of the open vocabulary",
    )
    parser.add_argument("--confidence", type=float, default=0.15, help="detection threshold")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--csv", action="store_true", help="emit CSV rows instead of text")
    parser.add_argument(
        "--save-annotated", type=Path, default=None, help="directory to write annotated images to"
    )
    parser.add_argument(
        "--impact",
        action="store_true",
        help="estimate mass and avoided emissions (factors are placeholders)",
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.images if not p.is_file()]
    if missing:
        parser.error(f"no such image(s): {', '.join(str(p) for p in missing)}")

    pipeline = SortingPipeline.build(
        policy_path=args.policy,
        weights_path=args.weights,
        vocabulary_path=None if (args.coco or args.weights) else DEFAULT_VOCAB,
    )

    if args.save_annotated:
        args.save_annotated.mkdir(parents=True, exist_ok=True)

    quiet = args.json or args.csv
    session = Session()
    payload = []
    rows = []

    for path in args.images:
        image = Image.open(path).convert("RGB")
        result = pipeline.sort(image, confidence=args.confidence)
        session.add(str(path), result)

        if args.json:
            payload.append(result_payload(result, str(path)))
        elif args.csv:
            rows.extend(detection_rows(result, str(path)))
        else:
            _print_human(result, str(path))

        if args.save_annotated:
            out = args.save_annotated / f"{path.stem}_sorted.png"
            annotate(image, result).save(out)
            if not quiet:
                print(f"\n  annotated -> {out}")

    if args.json:
        sys.stdout.write(to_json(payload))
    elif args.csv:
        sys.stdout.write(to_csv(rows))
    elif len(args.images) > 1:
        _print_totals(session, args.impact)
    elif args.impact:
        _print_impact(session, indent="  ")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
