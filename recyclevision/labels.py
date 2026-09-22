"""Label defects that make a training run produce NaN instead of a model.

A detector's loss divides by box area. A box with zero width or zero height
therefore divides by zero, and the NaN propagates through the whole batch --
so one bad label in tens of thousands poisons every epoch that touches it.

That failure is invisible from the outside, and it imitates a different one
convincingly. Ultralytics catches the NaN, restores `last.pt` and re-runs the
epoch; the re-run hits the same label and NaNs again. Its "attempt 1/3"
counter resets on each *successful* recovery, so a run never exhausts it and
never stops -- it just reports byte-identical metrics for hours. Two runs of
this project were read as fp16 overflow and then as a plateau before the
labels were looked at.

Rounding is enough to cause it on its own. A 1-pixel-tall annotation in a
1080-pixel frame normalises to 0.000926, and a sliver polygon imported as a
box can land under the 0.0000005 that six decimal places round to zero.

Nothing here needs torch, the images, or a GPU: it reads the label files.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: Normalised side length below which a box cannot survive the loss. Six
#: decimal places is what `box_line` writes, so anything under half of the
#: last place has already rounded to zero on disk.
MIN_SIDE = 5e-7


@dataclass
class LabelDefect:
    """One bad line, located precisely enough to go and look at it."""

    path: Path
    line_number: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line_number}  {self.kind} — {self.detail}"


@dataclass
class LabelScan:
    """What a split's label files contain, and what is wrong with them."""

    files: int = 0
    instances: int = 0
    defects: list[LabelDefect] = field(default_factory=list)

    @property
    def fatal(self) -> list[LabelDefect]:
        """Defects that produce NaN rather than merely a worse model."""
        return [d for d in self.defects if d.kind in {"zero-area", "not-a-number"}]

    def counts(self) -> Counter:
        return Counter(defect.kind for defect in self.defects)

    def report(self, limit: int = 10) -> str:
        if not self.defects:
            return f"{self.files} label file(s), {self.instances} instance(s) — clean."

        lines = [
            f"{self.files} label file(s), {self.instances} instance(s), "
            f"{len(self.defects)} defect(s):"
        ]
        for kind, count in self.counts().most_common():
            lines.append(f"  {kind:<14} {count}")

        if self.fatal:
            lines.append("")
            lines.append(
                f"  ⛔ {len(self.fatal)} of these make the loss NaN. A run over this data\n"
                "     cannot converge, and will not fail either: ultralytics recovers\n"
                "     from last.pt, hits the same label, and repeats forever."
            )

        lines.append("")
        lines.append("first few:")
        for defect in self.defects[:limit]:
            lines.append(f"  {defect}")
        if len(self.defects) > limit:
            lines.append(f"  … and {len(self.defects) - limit} more")
        return "\n".join(lines)


def _scan_line(path: Path, number: int, line: str, classes: int | None) -> list[LabelDefect]:
    fields = line.split()
    if len(fields) < 5:
        return [LabelDefect(path, number, "malformed", f"{len(fields)} field(s), expected 5")]

    def defect(kind: str, detail: str) -> list[LabelDefect]:
        return [LabelDefect(path, number, kind, detail)]

    try:
        index = int(fields[0])
    except ValueError:
        return defect("malformed", f"class {fields[0]!r} is not an integer")

    try:
        values = [float(v) for v in fields[1:5]]
    except ValueError:
        return defect("malformed", f"non-numeric geometry {' '.join(fields[1:5])}")

    if any(math.isnan(v) or math.isinf(v) for v in values):
        return defect("not-a-number", " ".join(fields[1:5]))

    found: list[LabelDefect] = []
    if classes is not None and not 0 <= index < classes:
        found += defect("bad-class", f"index {index}, dataset declares {classes}")

    cx, cy, w, h = values
    if w <= MIN_SIDE or h <= MIN_SIDE:
        found += defect("zero-area", f"w={w:.6f} h={h:.6f} — divides by zero in the loss")
    elif not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        found += defect("out-of-frame", f"centre ({cx:.6f}, {cy:.6f})")
    return found


def scan_labels(labels_dir: Path, classes: int | None = None) -> LabelScan:
    """Read every YOLO label file under a directory and report what is wrong."""
    scan = LabelScan()
    if not labels_dir.is_dir():
        return scan

    for path in sorted(labels_dir.rglob("*.txt")):
        scan.files += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            scan.defects.append(LabelDefect(path, 0, "unreadable", str(error)))
            continue

        seen: set[str] = set()
        for number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            scan.instances += 1
            if line in seen:
                scan.defects.append(LabelDefect(path, number, "duplicate", line))
                continue
            seen.add(line)
            scan.defects.extend(_scan_line(path, number, line, classes))

    return scan
