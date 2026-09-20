# Building a robust model from public data

The premise changed: no access to a specific facility for a long while, so
the model has to generalise across facilities rather than fit one. That is a
different goal from the WaRP run, and it needs a different measurement.

## What the WaRP model actually proved

**mAP50 0.671, routing accuracy 96.1%** — on WaRP's own held-out split. Same
plant, same cameras, same lighting, same conveyor. It is an honest number for
"can this work at all", and it says yes.

It says nothing about a different facility. A model can reach 96% on one
plant by learning that plant's belt colour and still collapse elsewhere. The
only way to know is to hold out a **facility**, not a set of frames.

## Measure before you train

The cheapest cross-facility number needs no training at all: take the model
you already have and run it against a facility it has never seen. Every
imported dataset carries the same 17-class descriptor, so this works
directly.

```bash
python scripts/evaluate.py --weights models/best_model.pt \
    --data datasets/zerowaste/data.yaml --policy policies/mrf_conveyor.yaml
```

Twenty minutes, and it answers the question the long run was going to answer.
A model that holds up reasonably on an unseen plant says the approach
transfers; one that collapses says multi-facility data is not optional —
either way the next run is better chosen.

**Then probe before committing.** `--fraction 0.2` trains on a fifth of the
data. A direction that fails on a fifth rarely succeeds on all of it, and
finding that out costs hours rather than days.

## Choosing the holdout: check class coverage first

A facility can only be a fair holdout if the training pool can *reach* every
class in it. Otherwise the missing class is scored as a failure to
generalise, when it is really a failure to have the data.

```
hold out WaRP       -> pool cannot reach: ['glass bottle']
hold out SortWaste  -> pool cannot reach: ['plastic tub']
hold out ZeroWaste  -> pool cannot reach: nothing
```

**WaRP is the worst choice**, despite being the obvious one. Neither other
dataset has a single glass instance, so all 534 of WaRP's glass objects would
be missed — and `glass bottle` routes to Container Glass, a bin nothing else
reaches, so it damages routing accuracy too, not just class accuracy.

**ZeroWaste is the clean holdout.** Every class it contains is reachable from
WaRP + SortWaste. Hold that out, and the number means what it says.

## The measurement that matters now

Import each dataset separately, train on some, evaluate on one that was never
trained on:

```bash
# Train on two facilities
python scripts/import_dataset.py <warp>/data.yaml    --mapping mappings/warp.yaml    --out datasets/train_pool
python scripts/import_dataset.py <other>/data.yaml   --mapping mappings/<other>.yaml --out datasets/train_pool

# Hold a third out entirely
python scripts/import_dataset.py <third>/data.yaml   --mapping mappings/<third>.yaml --out datasets/holdout

python train.py --data datasets/train_pool/data.yaml --epochs 100
python scripts/evaluate.py --weights runs/detect/<run>/weights/best.pt \
    --data datasets/holdout/data.yaml --policy policies/mrf_conveyor.yaml
```

The gap between the two numbers is the thing being built. Expect it to be
large at first — that is the finding, not a failure.

Filenames are prefixed per source, so imports merge into one directory
without overwriting each other, and `sources.json` records which source
contributed what. That file is what lets a held-out facility be chosen
deliberately rather than by accident.

## The three datasets disagree, and it does not matter

Merging sources merges their annotation schemes, and these three do not
agree about two things:

| the object | WaRP says | SortWaste says | ZeroWaste says |
| --- | --- | --- | --- |
| a beverage carton | `beverage carton` | `beverage carton` | `cardboard box` — no separate class |
| a plastic tub | — | `plastic tub` | `plastic bottle` — no separate class |

ZeroWaste has four classes and cannot express either distinction, so its
17,751 cardboard instances include cartons and its 1,769 rigid-plastic
instances include bottles, tubs and trays alike. Train on all three and the
model is shown the same object under two names.

**Expect class accuracy and mAP to suffer, and routing accuracy not to.**
Both pairs share a destination in both shipped policies:

```
beverage carton vs cardboard box : Paper & Cardboard / Paper & Cardboard
plastic bottle  vs plastic tub   : Rigid Plastic     / Rigid Plastic
```

This is the clearest case the project has produced for why routing accuracy
is the headline number. A confusion that changes nothing about where an item
goes is not an error worth paying to remove, and a metric that counts it as
one will push toward the wrong model.

Read a combined run accordingly: **compare routing accuracy across
facilities**, and treat a class-accuracy drop against the WaRP-only model as
the expected cost of a coarser source rather than a regression.

## What to add, in priority order

The current model knows five classes, all rigid containers, from one plant.
The gaps are not evenly valuable:

1. **A second and third sorting facility.** Different belts, lighting and
   clutter. This buys generalisation and nothing else does. Highest value by
   a distance.
2. **Film and flexible plastic.** A MRF "tangler", its own destination, and
   absent from WaRP entirely. Also the hardest shape for a box detector.
3. **Fibre that is not a box** — loose paper, newspaper. Common, and the
   current model has never seen it.
4. **Organics.** Routes to a bin nothing else routes to.
5. **Tableware.** The drinking-glass-versus-jar case the README opens with.
   Real, but rare on an industrial line — worth less here than it looks.

Every one needs a `mappings/*.yaml` translating its classes into
`vocab/waste_v2.yaml`. Unmapped classes are refused rather than dropped, so
writing the mapping is where the thinking happens; the import is mechanical.

## Where things go

| what | where | in git? |
| --- | --- | --- |
| Demo images for the app | `images/` | yes — small, credited in `images/SOURCES.md` |
| Raw downloaded datasets | anywhere outside the repo (`~/datasets/…`) | no |
| Imported training sets | `datasets/` | no — gitignored |
| Class translations | `mappings/*.yaml` | yes — this is the real work |
| Trained weights | `models/best_model.pt` | no — gitignored |
| Frames from video | `datasets/raw/` via `scripts/extract_frames.py` | no |

Raw datasets do not belong in the repository: they are large, and their
licences usually permit use but not redistribution. The mapping file is the
part worth committing, and it is the part that took judgement.

## Video

`scripts/extract_frames.py` pulls frames and drops near-duplicates. Point it
anywhere; send the output to `datasets/raw`, then pre-label with the model
that exists:

```bash
python scripts/extract_frames.py <a video> --out datasets/raw
python scripts/prelabel.py datasets/raw --out datasets/round1 --weights models/best_model.pt
```

Correct the pre-labels, train, and use the better model to pre-label the next
batch. Each round is faster than the last — that loop is why the WaRP model
was worth 18 hours even though WaRP is not the target domain.

**Split video by block, never at random.** Consecutive frames are nearly
identical, so a random split puts near-copies of the same object on both
sides and reports a score that cannot be reproduced on anything new.
`prelabel.py` already splits by block; the trap is only there if frames are
shuffled by hand first.
