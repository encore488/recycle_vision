"""Headless entry point: `python -m recyclevision IMAGE [IMAGE ...]`.

Exists so the pipeline can be exercised, scripted and debugged without
booting Streamlit -- and so that "does it actually work?" has an answer that
does not involve a browser.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .models import SortResult
from .pipeline import DEFAULT_POLICY, SortingPipeline
from .render import annotate


def _as_dict(result: SortResult, source: str) -> dict:
    return {
        "source": source,
        "model": result.model_name,
        "policy": result.policy_name,
        "total_items": result.total_items,
        "diversion_rate": round(result.diversion_rate, 4),
        "contamination_rate": round(result.contamination_rate, 4),
        "counts_by_bin": dict(result.counts_by_bin),
        "ignored_classes": sorted({d.label for d in result.ignored}),
        "items": [
            {
                "item": item.label,
                "bin": item.bin.key,
                "bin_name": item.bin.name,
                "material": item.material,
                "confidence": round(item.confidence, 4),
                "certainty": item.certainty.value,
                "handling": item.handling,
                "rationale": item.rationale,
                "box": [
                    item.detection.box.x1,
                    item.detection.box.y1,
                    item.detection.box.x2,
                    item.detection.box.y2,
                ],
            }
            for item in result.items
        ],
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recyclevision",
        description="Sort waste in an image into bins.",
    )
    parser.add_argument("images", nargs="+", type=Path, help="image files to sort")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="routing policy YAML")
    parser.add_argument("--weights", type=Path, default=None, help="model weights (.pt)")
    parser.add_argument("--confidence", type=float, default=0.15, help="detection threshold")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--save-annotated", type=Path, default=None, help="directory to write annotated images to"
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.images if not p.is_file()]
    if missing:
        parser.error(f"no such image(s): {', '.join(str(p) for p in missing)}")

    pipeline = SortingPipeline.build(policy_path=args.policy, weights_path=args.weights)

    if args.save_annotated:
        args.save_annotated.mkdir(parents=True, exist_ok=True)

    payload = []
    for path in args.images:
        image = Image.open(path).convert("RGB")
        result = pipeline.sort(image, confidence=args.confidence)

        if args.json:
            payload.append(_as_dict(result, str(path)))
        else:
            _print_human(result, str(path))

        if args.save_annotated:
            out = args.save_annotated / f"{path.stem}_sorted.png"
            annotate(image, result).save(out)
            if not args.json:
                print(f"\n  annotated -> {out}")

    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
