"""Precompute text embeddings for an open-vocabulary class list.

Run once, offline, whenever a vocabulary file changes:

    python scripts/build_vocab_embeddings.py vocab/waste_v1.yaml

This downloads the MobileCLIP text encoder (~570MB) on first use. The output
is a few tens of kilobytes and is committed to the repository, so serving a
request needs neither the encoder nor a network round-trip -- which is what
keeps the app deployable on a small host.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vocabulary", nargs="?", type=Path, default=DEFAULT_VOCAB)
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLOE

    vocab = Vocabulary.load(args.vocabulary)
    print(f"{vocab.name}: {len(vocab.classes)} classes, model {vocab.model}")

    model = YOLOE(vocab.model)
    embeddings = model.get_text_pe(vocab.classes)

    out = vocab.embeddings_path
    torch.save({"classes": vocab.classes, "embeddings": embeddings, "model": vocab.model}, out)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, shape {tuple(embeddings.shape)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
