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


def annotate(
    image: Image.Image,
    result: SortResult,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
) -> Image.Image:
    """Return a copy of `image` with each item boxed in its bin's colour."""
    canvas = image.convert("RGB").copy()
    if not show_boxes and not show_labels:
        return canvas

    draw = ImageDraw.Draw(canvas)
    font = _load_font(canvas.width)

    for item in result.items:
        color = _hex_to_rgb(item.bin.color)
        box = item.detection.box

        if show_boxes:
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=BOX_WIDTH)

        if not show_labels:
            continue

        text = item.bin.name
        if show_confidence:
            text = f"{text}  {item.confidence:.0%}"
        if item.needs_review:
            text = f"{text}  ?"

        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        chip_w = text_w + 2 * LABEL_PADDING
        chip_h = text_h + 2 * LABEL_PADDING

        # Prefer sitting the chip above the box; drop it inside when the box
        # is already at the top edge, so labels never render off-canvas.
        chip_x = box.x1 if box.width >= MIN_BOX_WIDTH_FOR_LABEL else max(0, box.x1 - chip_w / 2)
        chip_x = min(chip_x, canvas.width - chip_w)
        chip_x = max(0, chip_x)
        chip_y = box.y1 - chip_h if box.y1 - chip_h >= 0 else box.y1

        draw.rectangle([chip_x, chip_y, chip_x + chip_w, chip_y + chip_h], fill=color)
        draw.text(
            (chip_x + LABEL_PADDING - left, chip_y + LABEL_PADDING - top),
            text,
            fill=_readable_text_color(color),
            font=font,
        )

    return canvas
