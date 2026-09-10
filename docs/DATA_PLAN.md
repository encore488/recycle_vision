# Data plan

What to train on, in what order, and why.

## The situation

Own footage from a local MRF is some way off. A survey turned up several public
conveyor datasets, one of which — **SortWaste** — is close to the target domain:
top-down, real conveyor, 5,261 images, 87,252 annotated objects, and the raw
`.mp4` available. The authors have confirmed we may use it.

**None of these datasets can be reached from the development sandbox.** Verified,
not assumed: `sortwaste.di.ubi.pt`, `huggingface.co`, `zenodo.org`, `kaggle.com`
and the ZeroWaste GitHub repo all fail. Only GitHub release assets and PyPI are
reachable. So every download and every step that touches the data happens on
your machine; this repo holds the machinery and the analysis.

## The one thing that shapes everything

**Public waste datasets label materials. This project routes bins, from items.**

That is a finer distinction, and translating one way loses information
permanently:

| Case the README opens with | Can SortWaste express it? |
| --- | --- |
| Container jar vs drinking glass | **No.** No glass classes at all. |
| Lined paper cup vs plain paper | **Partly.** `ECAL` (carton) is the nearest label. |
| Pizza slice → compost | **No.** No organics. |

So SortWaste's value is **domain**, not **taxonomy**. It teaches a model what a
cluttered, motion-blurred, top-down belt looks like — the gap our two clean
sample photos do not cover at all. It cannot teach the item-level distinctions
the routing layer is built around, and no mapping file can invent them.

That is fine, as long as it is deliberate:

- **Fine-tune on SortWaste** → a belt-aware model that handles the bulk of a real
  stream well.
- **Keep the open-vocabulary detector** for the long tail it can attempt and a
  material-trained model never will.
- **Fine-tune again on your own item-level labels** when the footage arrives.

`mappings/sortwaste.yaml` records exactly what is lost, in a `cannot_express`
section, so the limitation stays visible instead of resurfacing later as a
confusion matrix nobody can explain.

## The order to do things in

### 1. Import and measure — before training anything

```bash
python scripts/import_dataset.py <sortwaste>/data.yaml \
    --mapping mappings/sortwaste.yaml --out datasets/sortwaste --link
python scripts/zeroshot_eval.py --data datasets/sortwaste/data.yaml
```

The mapping file's class names were written from a second-hand description, so
the first run will almost certainly reject some. That is by design: it lists
every class it finds and refuses until each has a rule. Correct the keys and
re-run — an unmapped class silently dropped would teach the model those objects
are background.

`zeroshot_eval.py` then answers the question actually worth an hour: **does the
current pre-labeller survive real conveyor clutter?** Our 100% detection
precision was measured on two clean photos. This is the first honest number the
project will have.

That number decides the rest:

- **Good recall (say >60%)** → pre-labelling works on real belts, and the
  labelling estimates in `LABELLING.md` hold.
- **Poor recall** → pre-labelling saves less than hoped on cluttered frames, and
  fine-tuning on SortWaste comes first precisely so that pre-labelling your own
  footage is worth doing.

### 2. Tune prompts — but only now, not before

The survey's most valuable finding is that class-only prompts underperform, and
that LLM-optimised prompts nearly doubled zero-shot mAP in published work. That
matches our own result: rewriting the vocabulary took class accuracy from 62% to
85%, for an hour's work and no new data.

**But it must be measured on data the prompts have not seen.** Everything scored
so far rests on two images, and v2's prompts were written while looking at them
— stated in the README as overfitting by construction. Tuning further against
the same two pictures would be overfitting with extra steps.

SortWaste's val split is the first real measuring stick this project has had.
The loop is:

```bash
cp vocab/waste_v2.yaml vocab/waste_v3.yaml     # edit the prompts
python scripts/build_vocab_embeddings.py vocab/waste_v3.yaml
python scripts/zeroshot_eval.py --data datasets/sortwaste/data.yaml \
    --vocabulary vocab/waste_v3.yaml
```

Same data, same policy, so the difference is the prompts. Keep the older
vocabulary: comparing against it is how a regression gets caught.

This is still the cheapest lever available. An hour of prompt editing against a
real val split beats a day of labelling.

### 3. Fine-tune

```bash
python train.py --data datasets/sortwaste/data.yaml --epochs 100
python scripts/evaluate.py --weights runs/segment/<run>/weights/best.pt \
    --data datasets/sortwaste/data.yaml
```

Note what SortWaste gives and does not: **boxes, no masks.** For counting and
routing that is enough. For the grasping goal it is not, and the mask has to come
from SAM at inference time rather than from the labels. Worth knowing before the
robot work starts.

### 4. Add diversity, deliberately

A model trained only on one Portuguese MBT belt will not transfer to a US MRF —
the authors say so themselves. Once SortWaste is working, mix in a second source
so the model does not learn one facility's lighting as if it were physics. TACO
(MIT, item-level, in-the-wild) and Drinking Waste Classification (CC0, and
exactly the four distinctions the routing policy cares most about) are the two
with the friendliest licences and the most compatible taxonomies.

Each needs its own file in `mappings/`. That is the whole cost of adding one.

## First contact with real data: WaRP

SortWaste could not be obtained, so WaRP was used instead — 2,974 images from an
industrial sorting plant, 28 item-level classes translated down to 5 of ours.

Zero-shot, `waste_v2`, household policy, 200 val images:

```
  precision  4.4%        matched 11 of 348 labelled objects
  recall     3.2%
  class accuracy    9.1%
  routing accuracy 54.5%
```

Against 100% / 85% on two clean photos. The two-image numbers were worthless,
exactly as the README warned. This is the first honest measurement the project
has had, and it says the detector does not transfer to an unfamiliar belt.

### The failure has a shape

**All ~238 false positives came from prompts for things WaRP cannot contain:**
plastic wrapper (119), food waste (50), bottle caps (46), sheet of paper (15).
Not one came from a class actually present in the data.

That is the same mechanism as v1's `styrofoam` firing on a sheet of paper, now
visible at scale. An open-vocabulary detector is not asked "is this present?" —
it is asked "which of these names fits best". Given a name with no referent in
the scene, the best fit is background.

**The design rule that follows: ask only for what the stream can contain.** A
prompt for something that never appears on a line is not free; it is a permanent
false-positive source on that line. Vocabulary is therefore facility-scoped
configuration, like a routing policy. `vocab/containers.yaml` is the first
example.

### Two causes, separable by one command each

1. **Inference resolution.** `predict()` never passed `imgsz`, so ultralytics
   used its 640 default — on WaRP's 1920×1080 frames that shrinks every object
   threefold. Meanwhile `train.py` had argued for 960 *"because conveyor items
   are small in frame"*. The belief was right and simply never reached
   inference. Now `DEFAULT_IMGSZ = 960`, tunable everywhere.
2. **Unscoped vocabulary**, as above.

Neither is settled until looked at, which is what `scripts/diagnose_eval.py` is
for: it draws ground truth in green and predictions in red. A 4% precision has
two indistinguishable explanations — a failing model, or a harness comparing the
wrong things — and only a picture separates them.

### Also worth recording

WaRP is **sparse**: 1.74 labelled objects per image, not the 10–40 of a real
MRF belt. It is a good taxonomy match and a poor clutter match, so a good score
on it would not have proven much about the target domain either.

## Datasets considered

| Dataset | Domain | Labels | Taxonomy | Licence | Verdict |
| --- | --- | --- | --- | --- | --- |
| **SortWaste** | Real MBT conveyor, top-down | Boxes | Material | Authors approved | Could not be obtained. Still the best fit if it becomes available. |
| **WaRP** *(in use)* | Industrial sorting plant | Boxes | **Item-level** | Kaggle | Best taxonomy match, and imported. Sparse (1.7 objects/image), so a poor proxy for MRF clutter. |
| ZeroWaste-f | Real MRF conveyor | **Masks** | Material | CC BY-NC 4.0 | Pretraining only. It was captured on a *paper* line, so paper is labelled as background — actively wrong for a project whose thesis is routing paper correctly. |
| SpectralWaste | Conveyor | Masks | Jam-causers | CC BY 4.0 | Niche, but the only unambiguous commercial licence. |
| TACO | In the wild | Masks | Item-level | **MIT** | Best licence, good for diversity. |
| Drinking Waste | Clean background | Boxes | Item-level | **CC0** | Exactly our four key distinctions; no clutter, so pair it with belt data. |

Details in this table come from a second-hand survey and could not be verified
from here. Check counts, splits and licence terms against each dataset's own
page before relying on them.

## What is already built for this

- `recyclevision/external.py` — mapping model; refuses unmapped classes
- `mappings/sortwaste.yaml` — translation template, with `cannot_express`
- `scripts/import_dataset.py` — import and remap; writes nothing until the
  mapping is complete
- `scripts/zeroshot_eval.py` — measure a vocabulary against ground truth
- `recyclevision/evaluate.py` — IoU matching and routing-aware scoring

All tested against a synthetic stand-in with SortWaste-shaped class names, since
the real thing is unreachable here. The first contact with actual data will
still find something; the machinery is meant to fail loudly when it does.
