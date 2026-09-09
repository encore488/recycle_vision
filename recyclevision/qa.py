"""Human-in-the-loop QA for detections.

Judging a model from one cluttered annotated overview does not work: boxes
overlap, labels get displaced, and small false positives hide in the noise.
This module cuts each detection out into its own tile so a person can grade
them quickly and honestly.

The verdict taxonomy separates the two failure modes this architecture cares
about, because they have completely different fixes:

    correct         right object, right bin
    wrong_bin       real object, well localised, routed to the wrong bin
    false_positive  box is not on an object at all
    unsure          cannot tell from the image

`wrong_bin` on a class the detector does not have (a steel can read as a
"cup") is a *detector* problem and argues for training. `wrong_bin` on a
class it does have is a *policy* problem and argues for editing a YAML file.

    python -m recyclevision.qa sheet images/*.jpg --out qa/
    # ...grade qa/verdicts.json by hand...
    python -m recyclevision.qa score qa/verdicts.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import SortResult
from .pipeline import DEFAULT_POLICY, SortingPipeline
from .render import _hex_to_rgb, _readable_text_color

VERDICTS = ("correct", "wrong_bin", "false_positive", "unsure")
UNGRADED = "TODO"

#: Context kept around each detection so the grader can see what it sits on.
CROP_PADDING = 50
TILE_WIDTH = 300
HEADER_HEIGHT = 26
GRID_COLUMNS = 3
GUTTER = 10
CAPTION_MARGIN = 12


def _caption_font(draw: ImageDraw.ImageDraw, text: str) -> ImageFont.ImageFont:
    """Largest font at which the caption still fits the tile width.

    A clipped caption ("...-> Mixed R") is worse than a small one: the grader
    cannot see which bin the detection was routed to, which is the single
    thing being graded.
    """
    for size in range(15, 8, -1):
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            break
        if draw.textlength(text, font=font) <= TILE_WIDTH - CAPTION_MARGIN:
            return font
    return ImageFont.load_default()


def _tile(image: Image.Image, result: SortResult, index: int) -> Image.Image:
    """One detection, cropped with context, boxed, and captioned."""
    item = result.items[index]
    box = item.detection.box

    left = max(0, int(box.x1) - CROP_PADDING)
    top = max(0, int(box.y1) - CROP_PADDING)
    right = min(image.width, int(box.x2) + CROP_PADDING)
    bottom = min(image.height, int(box.y2) + CROP_PADDING)
    crop = image.crop((left, top, right, bottom))

    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [box.x1 - left, box.y1 - top, box.x2 - left, box.y2 - top],
        outline=(255, 0, 0),
        width=3,
    )

    # Scale to a common width so the grid reads evenly, and so small
    # detections are actually inspectable rather than thumbnail-sized.
    height = max(1, round(crop.height * TILE_WIDTH / crop.width))
    crop = crop.resize((TILE_WIDTH, height), Image.LANCZOS)

    tile = Image.new("RGB", (TILE_WIDTH, height + HEADER_HEIGHT), _hex_to_rgb(item.bin.color))
    tile.paste(crop, (0, HEADER_HEIGHT))

    caption = f"#{index}  {item.detection.label} {item.confidence:.0%} -> {item.bin.name}"
    header = ImageDraw.Draw(tile)
    header.text(
        (6, 6),
        caption,
        fill=_readable_text_color(_hex_to_rgb(item.bin.color)),
        font=_caption_font(header, caption),
    )
    return tile


def contact_sheet(image: Image.Image, result: SortResult) -> Image.Image | None:
    """Every detection as a grid of captioned tiles, or None if there are none."""
    if not result.items:
        return None

    tiles = [_tile(image, result, i) for i in range(len(result.items))]
    columns = min(GRID_COLUMNS, len(tiles))
    rows = [tiles[i : i + columns] for i in range(0, len(tiles), columns)]

    width = columns * TILE_WIDTH + (columns + 1) * GUTTER
    height = sum(max(t.height for t in row) for row in rows) + (len(rows) + 1) * GUTTER

    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    y = GUTTER
    for row in rows:
        row_height = max(t.height for t in row)
        for column, tile in enumerate(row):
            # Top-align within the row; ragged bottoms are fine, ragged tops
            # make the captions hard to scan.
            sheet.paste(tile, (GUTTER + column * (TILE_WIDTH + GUTTER), y))
        y += row_height + GUTTER
    return sheet


def verdict_stub(result: SortResult, source: str) -> list[dict]:
    """A gradable record per detection, with the verdict left blank."""
    return [
        {
            "source": source,
            "index": i,
            "detected_as": item.detection.label,
            "confidence": round(item.confidence, 3),
            "routed_to": item.bin.key,
            "policy_certainty": item.certainty.value,
            "box": [
                round(v)
                for v in (
                    item.detection.box.x1,
                    item.detection.box.y1,
                    item.detection.box.x2,
                    item.detection.box.y2,
                )
            ],
            "verdict": UNGRADED,
            "actually_is": "",
            "should_be_bin": "",
        }
        for i, item in enumerate(result.items)
    ]


def _cmd_sheet(args: argparse.Namespace) -> int:
    pipeline = SortingPipeline.build(policy_path=args.policy)
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for path in args.images:
        image = Image.open(path).convert("RGB")
        result = pipeline.sort(image, confidence=args.confidence)

        sheet = contact_sheet(image, result)
        if sheet is None:
            print(f"{path}: no detections")
            continue

        out = args.out / f"{path.stem}_qa.png"
        sheet.save(out)
        records.extend(verdict_stub(result, str(path)))
        print(f"{path}: {len(result.items)} detection(s) -> {out}")

    if records:
        verdicts = args.out / "verdicts.json"
        verdicts.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f'\nGrade {verdicts} by hand: set each "verdict" to one of {list(VERDICTS)}')
        print(f"Then: python -m recyclevision.qa score {verdicts}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    records = json.loads(args.verdicts.read_text(encoding="utf-8"))
    graded = [r for r in records if r.get("verdict") in VERDICTS]
    ungraded = len(records) - len(graded)

    if not graded:
        print(f"Nothing graded yet ({len(records)} record(s) waiting).")
        return 1

    counts = Counter(r["verdict"] for r in graded)
    total = len(graded)

    real = total - counts["false_positive"]
    print(f"Graded {total} detection(s)" + (f" ({ungraded} still ungraded)" if ungraded else ""))
    print()
    for verdict in VERDICTS:
        if counts[verdict]:
            print(f"  {verdict:16} {counts[verdict]:3}  {counts[verdict] / total:5.0%}")
    print()
    # Detection precision and routing accuracy fail for different reasons and
    # are fixed in different places, so they are never combined into one score.
    print(f"  detection precision  {real / total:5.0%}   (boxes that are on a real object)")
    if real:
        routing = counts["correct"] / real
        print(f"  routing accuracy     {routing:5.0%}   (correct bin, of real objects)")
    print(f"  end-to-end correct   {counts['correct'] / total:5.0%}")

    confusions = Counter(
        (r["detected_as"], r.get("actually_is") or "?")
        for r in graded
        if r["verdict"] == "wrong_bin" and r.get("actually_is")
    )
    if confusions:
        print("\n  misreadings driving wrong bins:")
        for (detected, actual), n in confusions.most_common():
            print(f"    {detected!r} is really {actual!r}  x{n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recyclevision.qa",
        description="Grade detections by hand, and score the grades.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sheet = sub.add_parser("sheet", help="build contact sheets and a verdict stub")
    sheet.add_argument("images", nargs="+", type=Path)
    sheet.add_argument("--out", type=Path, default=Path("qa"))
    sheet.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    sheet.add_argument("--confidence", type=float, default=0.15)
    sheet.set_defaults(func=_cmd_sheet)

    score = sub.add_parser("score", help="summarise a graded verdicts file")
    score.add_argument("verdicts", type=Path)
    score.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    if args.command == "sheet":
        missing = [p for p in args.images if not p.is_file()]
        if missing:
            parser.error(f"no such image(s): {', '.join(str(p) for p in missing)}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
