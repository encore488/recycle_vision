"""Everything checkable about a training run before it costs a night.

Three runs of this project have been spent discovering, after the fact, that
something was wrong with the setup rather than with the idea. Each finding was
available before the run started; nobody had written the check.

The rule here: **a check earns its place only if it would have caught a real
failure, or would catch one that is cheap to cause and expensive to notice.**
Every check below names the failure it guards.

Severity matters as much as detection. A blocking finding means the run cannot
produce a usable model; a warning means it will run but something will be
worth knowing afterwards. Blocking on a warning trains nobody and teaches
people to pass `--force`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .labels import scan_labels

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: A box thinner than this many pixels at the training resolution has almost
#: no gradient signal and is a plausible source of numerical trouble. It is not
#: proven to cause NaN here -- the pool's labels scanned clean -- so it warns.
MIN_PIXELS = 2.0

#: Share of instances in one class above which the pool is badly skewed.
DOMINANT_SHARE = 0.5


@dataclass
class Finding:
    level: str  # "block" or "warn"
    check: str
    detail: str

    def __str__(self) -> str:
        mark = "⛔" if self.level == "block" else "⚠️ "
        return f"{mark} {self.check}: {self.detail}"


@dataclass
class Preflight:
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def block(self, check: str, detail: str) -> None:
        self.findings.append(Finding("block", check, detail))

    def warn(self, check: str, detail: str) -> None:
        self.findings.append(Finding("warn", check, detail))

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "block"]

    def report(self) -> str:
        lines = list(self.notes)
        if self.findings:
            lines.append("")
            lines.extend(str(f) for f in self.findings)
        lines.append("")
        if self.blocking:
            lines.append(
                f"{len(self.blocking)} blocking finding(s). This run cannot produce a "
                "usable model; fix them first."
            )
        elif self.findings:
            lines.append(
                f"{len(self.findings)} warning(s), nothing blocking. Safe to train — "
                "read them so the result is not a surprise."
            )
        else:
            lines.append("Clean. Nothing checkable stands between this data and a run.")
        return "\n".join(lines)


def _stems(directory: Path, suffixes: set[str] | None = None) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.stem
        for path in directory.rglob("*")
        if suffixes is None or path.suffix.lower() in suffixes
    }


def check_split(
    result: Preflight,
    root: Path,
    split: str,
    names: list[str],
    imgsz: int,
) -> Counter:
    """One split: files present, paired, labelled sanely. Returns class counts."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    counts: Counter = Counter()

    if not images_dir.is_dir():
        result.block(split, f"no image directory at {images_dir}")
        return counts

    images = _stems(images_dir, IMAGE_SUFFIXES)
    labels = _stems(labels_dir, {".txt"})
    if not images:
        result.block(split, f"no images under {images_dir}")
        return counts

    # An image with no label file is not an empty scene to ultralytics -- it is
    # an image of nothing, and it teaches the model that whatever is in it is
    # background. This is the WaRP density trap in its purest form.
    unlabelled = images - labels
    if unlabelled:
        result.block(
            split,
            f"{len(unlabelled)} image(s) have no label file, e.g. "
            f"{sorted(unlabelled)[0]} — each teaches the model its contents are background",
        )
    orphans = labels - images
    if orphans:
        result.warn(split, f"{len(orphans)} label file(s) have no image, e.g. {sorted(orphans)[0]}")

    scan = scan_labels(labels_dir, classes=len(names))
    if scan.fatal:
        result.block(
            split,
            f"{len(scan.fatal)} label(s) make the loss NaN — run scripts/check_labels.py",
        )
    other = sum(v for k, v in scan.counts().items() if k not in {"zero-area", "not-a-number"})
    if other:
        result.warn(split, f"{other} non-fatal label defect(s) — run scripts/check_labels.py")

    # Box sizes at the resolution actually trained at, not normalised.
    tiny = 0
    for path in sorted(labels_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                index = int(fields[0])
                w, h = float(fields[3]), float(fields[4])
            except ValueError:
                continue
            if 0 <= index < len(names):
                counts[names[index]] += 1
            if min(w, h) * imgsz < MIN_PIXELS:
                tiny += 1
    if tiny:
        result.warn(
            split,
            f"{tiny} box(es) are under {MIN_PIXELS:g}px at imgsz={imgsz} — almost no "
            "gradient signal, and the first thing to suspect if this run goes NaN",
        )

    result.note(f"{split}: {len(images)} images, {sum(counts.values())} instances")
    return counts


def check_dataset(root: Path, names: list[str], imgsz: int = 960) -> Preflight:
    """Everything that can be known about a dataset without training on it."""
    result = Preflight()

    train = check_split(result, root, "train", names, imgsz)
    val = check_split(result, root, "val", names, imgsz)

    # The same frame on both sides makes validation measure memorisation. For
    # video-derived data this is the difference between a real number and a
    # meaningless one.
    leaked = _stems(root / "images" / "train", IMAGE_SUFFIXES) & _stems(
        root / "images" / "val", IMAGE_SUFFIXES
    )
    if leaked:
        result.block(
            "leakage",
            f"{len(leaked)} image(s) are in both splits, e.g. {sorted(leaked)[0]} — "
            "validation would be measuring memorisation",
        )

    # A class present in val and absent from train is scored as a failure to
    # generalise when it is really a failure to have the data.
    unreachable = sorted(c for c in val if train.get(c, 0) == 0)
    if unreachable:
        result.block(
            "coverage",
            f"class(es) {unreachable} appear in val and never in train — "
            "they can only ever be scored wrong",
        )

    declared_empty = sorted(set(names) - set(train) - set(val))
    if declared_empty:
        result.warn(
            "coverage",
            f"{len(declared_empty)} declared class(es) have no instances anywhere: "
            f"{declared_empty} — head channels and loss terms for nothing",
        )

    total = sum(train.values())
    if total:
        top, count = train.most_common(1)[0]
        if count / total > DOMINANT_SHARE:
            result.warn(
                "balance",
                f"{top!r} is {count / total:.0%} of training instances — a model can "
                "score well by predicting it everywhere",
            )

    # Trap 2 in CLAUDE.md: the cache is keyed on total byte size and paths,
    # never contents, so an edited label can be silently ignored.
    for split in ("train", "val"):
        cache = root / "labels" / f"{split}.cache"
        labels_dir = root / "labels" / split
        if not (cache.is_file() and labels_dir.is_dir()):
            continue
        newest = max((p.stat().st_mtime for p in labels_dir.rglob("*.txt")), default=0.0)
        if newest > cache.stat().st_mtime:
            result.block(
                "cache",
                f"{cache} is older than the labels beside it. Ultralytics keys this on "
                "total byte size and paths, never contents, so it will train on the OLD "
                "labels and report nothing wrong. Delete it.",
            )

    return result
