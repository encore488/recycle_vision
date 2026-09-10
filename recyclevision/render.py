"""Drawing detections onto an image.

Deliberately not ultralytics' `Results.plot()`. Two reasons:

1. `plot()` returns a BGR array, which Streamlit renders as RGB -- the v0.2
   app displayed its annotations with red and blue swapped. Drawing here in
   PIL's native RGB removes the whole class of bug.
2. `plot()` colours boxes by *class*, but the interesting thing about an item
   is its *destination*. Colouring by bin means the picture answers the
   product's actual question at a glance.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .models import SortResult

BOX_WIDTH = 3
LABEL_PADDING = 4
#: Below this, a fitted label is wider than the box it belongs to and reads
#: better tucked just outside the corner instead.
MIN_BOX_WIDTH_FOR_LABEL = 40
#: Vertical gap left between two label chips that would otherwise collide.
LABEL_GAP = 2
#: How many times to nudge a colliding chip before giving up and letting it
#: overlap. Cluttered images hit this, and an unbounded search would not
#: terminate usefully anyway.
MAX_LABEL_NUDGES = 14
#: A chip displaced further than this from its box gets a leader line back to
#: it. Without one, a nudged chip reads as belonging to whatever it landed on
#: -- which makes the annotation actively misleading during review.
LEADER_LINE_THRESHOLD = 6
LEADER_WIDTH = 2
#: Boxes are drawn over photographs, so a bin colour can land on a background
#: of almost the same tone -- grey "Landfill" boxes on a dark conveyor belt
#: were effectively invisible, which reads as the box being drawn on nothing.
#: A dark halo behind every stroke guarantees an edge whatever is underneath.
HALO_COLOR = (12, 12, 12)
HALO_SPREAD = 2

#: (x0, y0, x1, y1) in pixel coordinates.
Rect = tuple[float, float, float, float]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _readable_text_color(background: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white, whichever stays legible on the given background.

    Uses the standard sRGB luminance weighting -- bin colours are chosen for
    distinguishability, not contrast, so this cannot be assumed either way.
    """
    r, g, b = background
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return (0, 0, 0) if luminance > 0.6 else (255, 255, 255)


def _load_font(image_width: int) -> ImageFont.ImageFont:
    """Scale the label to the image so annotations stay readable at any size."""
    size = max(13, min(30, image_width // 45))
    for candidate in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _overlaps(a: Rect, b: Rect) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _find_label_slot(
    x: float,
    preferred_y: float,
    width: float,
    height: float,
    taken: list[Rect],
    canvas_height: int,
) -> Rect:
    """Find somewhere for a label chip that nothing else already occupies.

    A conveyor belt is a cluttered image: boxes sit close together, and chips
    drawn at their preferred position overprint each other into unreadable
    mush. Nudging downwards keeps every label legible and still adjacent to
    the box it belongs to.
    """
    y = preferred_y
    for _ in range(MAX_LABEL_NUDGES):
        candidate = (x, y, x + width, y + height)
        if not any(_overlaps(candidate, t) for t in taken):
            return candidate
        y += height + LABEL_GAP
        if y + height > canvas_height:
            break
    return (x, preferred_y, x + width, preferred_y + height)


def _draw_legend(draw: ImageDraw.ImageDraw, result: SortResult, font, canvas_size) -> None:
    """A key for the bin colours, so a saved image explains itself.

    In the app the bin cards below the image serve this purpose, but an
    annotated PNG on its own has no other way to say what blue means.
    """
    bins = result.bins_used
    if not bins:
        return

    width, height = canvas_size
    swatch = max(10, font.size if hasattr(font, "size") else 12)
    line_h = swatch + 6
    entries = [(b, b.name) for b in bins]
    text_w = max(draw.textlength(name, font=font) for _, name in entries)
    box_w = int(swatch + 8 + text_w + 16)
    box_h = line_h * len(entries) + 10

    x0, y0 = 8, height - box_h - 8
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(12, 12, 12))

    for row, (bin_, name) in enumerate(entries):
        y = y0 + 6 + row * line_h
        draw.rectangle([x0 + 8, y, x0 + 8 + swatch, y + swatch], fill=_hex_to_rgb(bin_.color))
        draw.text((x0 + 8 + swatch + 8, y - 1), name, fill=(255, 255, 255), font=font)


def annotate(
    image: Image.Image,
    result: SortResult,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_legend: bool = True,
) -> Image.Image:
    """Return a copy of `image` with each item boxed in its bin's colour."""
    canvas = image.convert("RGB").copy()
    if not show_boxes and not show_labels:
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = _load_font(canvas.width)

    # Boxes first, labels second: a box drawn later would otherwise cut
    # through a chip already placed for a neighbouring item.
    if show_boxes:
        for item in result.items:
            box = item.detection.box
            draw.rectangle(
                [
                    box.x1 - HALO_SPREAD,
                    box.y1 - HALO_SPREAD,
                    box.x2 + HALO_SPREAD,
                    box.y2 + HALO_SPREAD,
                ],
                outline=HALO_COLOR,
                width=BOX_WIDTH + HALO_SPREAD * 2,
            )
            draw.rectangle(
                [box.x1, box.y1, box.x2, box.y2],
                outline=_hex_to_rgb(item.bin.color),
                width=BOX_WIDTH,
            )

    if not show_labels:
        return canvas

    taken: list[Rect] = []
    for item in result.items:
        color = _hex_to_rgb(item.bin.color)
        box = item.detection.box

        # The bin is already carried by the colour, and repeating its name on
        # every box says nothing when a whole image routes the same way -- nine
        # chips reading "Mixed Recycling" is noise. The item name is the thing
        # the picture cannot otherwise tell you.
        text = item.label
        if show_confidence:
            text = f"{text}  {item.confidence:.0%}"
        if item.needs_review:
            text = f"{text}  ?"

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        chip_w = (right - left) + 2 * LABEL_PADDING
        chip_h = (bottom - top) + 2 * LABEL_PADDING

        chip_x = box.x1 if box.width >= MIN_BOX_WIDTH_FOR_LABEL else box.x1 - chip_w / 2
        chip_x = max(0.0, min(chip_x, canvas.width - chip_w))

        # Prefer sitting the chip above the box; drop it inside when the box
        # is already at the top edge, so labels never render off-canvas.
        preferred_y = box.y1 - chip_h if box.y1 - chip_h >= 0 else box.y1

        x0, y0, x1, y1 = _find_label_slot(chip_x, preferred_y, chip_w, chip_h, taken, canvas.height)
        taken.append((x0, y0, x1, y1))

        # A chip that had to move is no longer visibly attached to its box.
        # Tie it back, or the picture claims things it does not mean.
        if (
            abs(y0 - preferred_y) > LEADER_LINE_THRESHOLD
            or abs(x0 - box.x1) > LEADER_LINE_THRESHOLD
        ):
            leader = [(x0 + LEADER_WIDTH, y1), (box.x1, box.y1)]
            draw.line(leader, fill=HALO_COLOR, width=LEADER_WIDTH + HALO_SPREAD * 2)
            draw.line(leader, fill=color, width=LEADER_WIDTH)

        draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text(
            (x0 + LABEL_PADDING - left, y0 + LABEL_PADDING - top),
            text,
            fill=_readable_text_color(color),
            font=font,
        )

    if show_legend:
        _draw_legend(draw, result, font, canvas.size)

    return canvas
