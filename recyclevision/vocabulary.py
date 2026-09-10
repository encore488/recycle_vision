"""Open-vocabulary class lists and their cached text embeddings.

An open-vocabulary detector is told what to look for in words, so the class
list becomes configuration rather than something frozen at training time.
That is what lets this project ask for "aluminum drink can" -- a thing COCO
simply cannot express.

Turning those words into embeddings needs a text encoder that is far larger
than the detector itself (~570MB). Doing that at request time would make the
model heavier than the whole rest of the app, so it happens once, offline,
and the result -- a few tens of kilobytes -- is committed beside the
vocabulary. See `scripts/build_vocab_embeddings.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab"
DEFAULT_VOCAB = VOCAB_DIR / "waste_v1.yaml"

#: Embeddings live beside their vocabulary, sharing its stem.
EMBEDDING_SUFFIX = ".pt"


class VocabularyError(ValueError):
    """Raised when a vocabulary file or its embeddings are unusable."""


@dataclass(frozen=True)
class Vocabulary:
    """A named list of detection prompts."""

    name: str
    classes: list[str]
    model: str
    description: str = ""
    path: Path | None = None

    @property
    def embeddings_path(self) -> Path:
        if self.path is None:
            raise VocabularyError(f"vocabulary {self.name!r} was not loaded from disk")
        return self.path.with_suffix(EMBEDDING_SUFFIX)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_VOCAB) -> Vocabulary:
        path = Path(path)
        if not path.is_file():
            raise VocabularyError(f"no vocabulary file at {path}")

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise VocabularyError(f"{path} is not valid YAML: {exc}") from exc

        if not isinstance(raw, dict):
            raise VocabularyError(f"{path} should contain a mapping at the top level")

        classes = raw.get("classes")
        if not classes or not all(isinstance(c, str) and c.strip() for c in classes):
            raise VocabularyError(f"{path} needs a non-empty 'classes' list of strings")
        if len(set(classes)) != len(classes):
            raise VocabularyError(f"{path} has duplicate class names")

        return cls(
            name=raw.get("name", path.stem),
            description=raw.get("description", ""),
            model=raw.get("model", "yoloe-11s-seg.pt"),
            classes=list(classes),
            path=path,
        )

    def load_embeddings(self):
        """Read the cached text embeddings for this vocabulary.

        Deliberately strict about drift: an embedding file that no longer
        matches the class list would silently mislabel every detection,
        which is far worse than refusing to start.
        """
        import torch  # deferred: heavy

        path = self.embeddings_path
        if not path.is_file():
            raise VocabularyError(
                f"no cached embeddings at {path}. Build them with:\n"
                f"    python scripts/build_vocab_embeddings.py {self.path}"
            )

        payload = torch.load(path, weights_only=False)
        cached = payload.get("classes")
        if cached != self.classes:
            raise VocabularyError(
                f"{path} is stale: it was built for a different class list. "
                f"Rebuild it with:\n"
                f"    python scripts/build_vocab_embeddings.py {self.path}"
            )
        return payload["embeddings"]


def discover() -> list[Path]:
    """Every vocabulary file that has embeddings built for it."""
    return sorted(p for p in VOCAB_DIR.glob("*.yaml") if p.with_suffix(EMBEDDING_SUFFIX).is_file())
