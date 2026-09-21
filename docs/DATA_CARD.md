# Data card — datasets in use

What each source actually contains, what it cannot express, and what that
costs. A model's failures are usually predictable from its data, but only if
somebody wrote the data down.

All three translate into `vocab/waste_v2.yaml` through a file in `mappings/`.
Every source class must have a rule — an unmapped class would be dropped
silently and teach the model those objects are background.

## Annotation density

This is the number that decides how a dataset can be used at all.

| dataset | images | instances | per image |
| --- | --- | --- | --- |
| WaRP | 2,974 | 10,374 | 3.5 |
| ZeroWaste-f | 4,503 | 26,766 | 5.9 |
| SortWaste | 5,261 | 87,525 | 16.6 |

WaRP's frames visibly hold dozens of objects. At 3.5 labelled, most of what
is in them is annotated as nothing, and a detector trained on that learns
that bottles on belts are background. **Sparse annotation harms training far
more than evaluation**, which is why WaRP is used only for the latter.

---

## WaRP (Warp-D)

- **Source:** Kaggle, `parohod/warp-waste-recycling-plant-dataset`
- **Setting:** industrial sorting plant conveyor, 1920×1080
- **Format:** YOLO boxes, 28 item-level classes
- **Splits:** 2,452 train / 522 val
- **Mapping:** `mappings/warp.yaml` → 5 classes
- **Status: evaluation only.** Retired from training for the density reason
  above.

| our class | instances |
| --- | --- |
| plastic bottle | 8,285 |
| beverage carton | 812 |
| metal can | 660 |
| glass bottle | 534 |
| cardboard box | 83 |

**Unusually good for this project** in one respect: its classes are
item-level, which is the granularity routing needs, where most public waste
datasets label materials. **It is the only source of glass** — 534 instances
that nothing else provides, and glass routes to a bin nothing else reaches.

Cannot express: glass jar vs bottle, tableware, tubs, cups, film, plain
paper, caps, organics.

---

## SortWaste

- **Setting:** industrial waste sorting line. The shipped descriptor contains
  absolute paths from the machine that built it (`/home/socialab/…`), which
  is how it was found to need path relocation on import.
- **Format:** COCO boxes *and* a YOLO export. **Import from COCO** — the YOLO
  export ships all 5,261 label files but only `test`'s 776 images.
- **Splits:** 3,705 train / 780 val / 776 test (test folded into val here)
- **Mapping:** `mappings/sortwaste.yaml` → 6 classes
- **Largest and densest source.** Precision measured here means considerably
  more than on WaRP.

| our class | instances |
| --- | --- |
| plastic bottle | 42,952 |
| beverage carton | 19,227 |
| plastic bag | 12,337 |
| plastic tub | 9,416 |
| cardboard box | 2,156 |
| metal can | 1,437 |

Two judgement calls recorded in the mapping:

- `mixed_plastic_rigid` → **`plastic tub`**, not bottle: the dataset already
  separates bottles into pet/pead/pet_oleo, so the mixed-rigid remainder is
  tubs and trays. Both route identically, so it costs no routing accuracy.
- `ecal` → **`beverage carton`**, read as *envases de cartón para alimentos y
  líquidos*. **Inferred from the abbreviation, not verified against images**,
  and it carries 19,227 instances. Worth a spot-check before it is trusted.

Its YOLO descriptor names classes `"0 pet"`, `"4 ecal"` — the original ids
baked into the names while the index beside them was renumbered, and the
originals skip 3. `inspect_dataset.py` cross-checks the YOLO labels against
the COCO annotations; they agree instance for instance.

Cannot express: glass of any kind, bag vs wrapper, plastic cup, non-board
fibre, caps, organics.

---

## ZeroWaste-f

- **Source:** [github.com/dbash/zerowaste](https://github.com/dbash/zerowaste),
  CVPR 2022. Zenodo `10.5281/zenodo.4899926`, `zerowaste-f-final.zip`
- **Licence: CC BY-NC 4.0 — NonCommercial.** Fine for research and a
  portfolio; **not fine for a commercial product**, and a model trained on it
  is at best a grey area. A live decision, not a footnote.
- **Setting:** US recycling facility, cluttered and deformable material
- **Format:** COCO with polygons. Imported as boxes, to stay uniform with the
  others — ultralytics will not train on a dataset mixing both.
- **Splits:** 3,002 train / 572 val / 929 test (test folded into val here)
- **Mapping:** `mappings/zerowaste.yaml` → 4 classes

| our class | instances |
| --- | --- |
| cardboard box | 17,751 |
| plastic bag | 6,864 |
| plastic bottle | 1,769 |
| metal can | 382 |

**The only real source of film** alongside SortWaste, and film routes to its
own destination. Severely imbalanced: cardboard is over 66% of annotations
and metal under 2% — read its metal numbers as noise.

Frames live in `data/`, segmentation masks in `sem_seg/`, **under identical
filenames**. A basename lookup can resolve to a mask; the importer prefers
photographs explicitly.

Cannot express: cartons (folded into `cardboard`), tubs and cups (folded into
`rigid_plastic`), bag vs wrapper, glass of any kind, non-board fibre,
organics.

---

## What none of them have

10 of 17 vocabulary classes have **no training data at all**: metal bottle
cap, plastic bottle cap, plastic cup, plastic wrapper, glass jar, empty
drinking glass, sheet of paper, newspaper, paper cup, food waste.

That is the ceiling. No amount of relabelling existing data creates them —
it needs new sources, or own footage. They include every case the README
opens with: the jar against the tumbler, the lined cup against plain paper,
anything that composts.

## Where the sources disagree

| the object | WaRP | SortWaste | ZeroWaste |
| --- | --- | --- | --- |
| beverage carton | `beverage carton` | `beverage carton` | `cardboard box` |
| plastic tub | — | `plastic tub` | `plastic bottle` |

Train on several and the model sees one object under two names. Both pairs
share a destination in both shipped policies, so the cost lands entirely on
class accuracy and mAP and not on routing. Expect that gap rather than
reading it as a regression.
