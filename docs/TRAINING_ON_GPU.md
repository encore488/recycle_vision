# Training on a GPU

Fine-tuning this project on CPU takes most of a day. On a free hosted T4 it
takes about an hour. This is how to get one.

## First: you may already have a GPU

`--device 0` means "the first NVIDIA GPU", which a Mac does not have — hence
the CUDA traceback. But **Apple Silicon has an accelerator that answers to
`mps`**, and it is worth roughly an order of magnitude over CPU.

```bash
python train.py --data datasets/warp/data.yaml --epochs 100     # auto-detects
```

`--device auto` is now the default and picks CUDA, then `mps`, then CPU. Asking
for `--device 0` on a machine without CUDA no longer produces a raw ultralytics
traceback; it tells you what to use instead.

**Is `mps` enough?** For a first run on ~3k images, probably — expect a few
hours rather than one. If it is slow or unstable (ultralytics' MPS support is
good but not flawless), move to Colab. Try local first: it costs one command
and no upload.

## The dataset picks the model, not you

WaRP ships **bounding boxes, not polygons**, and a `-seg` model has nothing to
learn masks from. Ultralytics does notice — but only after caching all 2,452
labels, several minutes into a run that was never going to start, and it
phrases the fix as "supply a segment dataset", which blames the data rather
than the model.

`train.py` now reads the label geometry first and picks `yolo11s.pt` for boxes,
`yolo11s-seg.pt` for polygons. Leave `--model` alone and it is right by
construction; pass a `-seg` model against boxes and it says so in a tenth of a
second.

Boxes cost you nothing here. Routing needs to know **what** an item is and
**where** it is, and a box says both. Masks earn their keep on two things —
pre-labelling, and picking a grasp point on an item that overlaps another — and
both of those are about your own footage, which `scripts/prelabel.py` outlines
for free.

One consequence: a detection run lands in `runs/detect/`, not `runs/segment/`.
`train.py` prints the real path when it finishes, so copy it from there rather
than from memory.

---

## Re-import WaRP before you train

The first version of `mappings/warp.yaml` sent WaRP's 812 beverage cartons to
`paper cup`, which routes to landfill with `certainty: high` — the wrong bin on
the second-largest class in the dataset. `waste_v2` now has a `beverage carton`
class and both policies route it, so the mapping is fixed, but **the class
indices changed**. A dataset imported before that is wrong on disk:

```bash
python scripts/import_dataset.py <warp>/data.yaml \
    --mapping mappings/warp.yaml --out datasets/warp
```

Expect `beverage carton 812` where it used to say `paper cup 812`. Minutes of
work, ahead of hours of training — and the notebook does it for you.

---

## Colab (recommended)

Open **[`notebooks/train_colab.ipynb`](../notebooks/train_colab.ipynb)** —
upload it to [colab.research.google.com](https://colab.research.google.com) or
open it straight from GitHub. It clones the repo, fetches WaRP, imports it,
measures zero-shot, trains, evaluates, and saves the weights to Drive.

Three things that actually bite, in order of likelihood:

**1. You forget to turn the GPU on.** Free Colab starts on CPU. *Runtime →
Change runtime type → T4 GPU*, before running anything. The notebook's first
cell is `nvidia-smi` precisely so you find out in five seconds rather than an
hour in.

**2. The dataset path is wrong.** Kaggle archives unpack differently than you
expect. The notebook runs `find` for `data.yaml` and asks you to paste the real
path rather than guessing one that works today and breaks tomorrow.

**3. The session dies and takes the weights with it.** Colab reclaims the
machine after idling, and free sessions are capped at a few hours. **Copy
`best.pt` to Drive before you close the tab.** The notebook's last step does
this; do not skip it.

If you hit CUDA out-of-memory, lower `--batch` to 4, then 2. A T4 has 16GB,
which is comfortable at 960px and batch 8, but not infinitely so.

---

## Kaggle Notebooks (the alternative)

Worth knowing about because of one real advantage: **the dataset is already
there.** Add WaRP to a notebook from the Data panel and it mounts at
`/kaggle/input/...` with no download at all.

- 30 GPU-hours per week, P100 or T4 ×2 — more generous than Colab's free tier
- Sessions run up to 12 hours and survive browser closure, which Colab's do not
- Output persists in `/kaggle/working` after the session ends
- You must verify your phone number to get GPU access

Same steps as the notebook, with two changes:

```python
# Dataset is already mounted — no download step
WARP_YAML = "/kaggle/input/warp-waste-recycling-plant-dataset/Warp-D-yolo/data.yaml"

# Internet is off by default: turn it on in the sidebar, or the git clone and
# the weights download both fail.
```

**Pick Kaggle if** the Colab session keeps dying, or you want to train
overnight. **Pick Colab if** you want to be running in two minutes.

---

## What success looks like

Zero-shot on WaRP, for comparison:

| | conf 0.01 | conf 0.25 |
| --- | --- | --- |
| Recall | 52.6% | 0.9% |
| Fair precision | 56.0% | 100% |

The model **already finds** these objects — it ranks them terribly, which is
why usable recall needs a threshold of 0.01. Training is meant to fix the
scores, not teach it to see.

So the thing to look for afterwards is not a higher peak recall. It is **recall
holding above 50% at a sane threshold like 0.25**, where the zero-shot model
collapses to 0.9%.

`scripts/evaluate.py` reports mAP alongside class accuracy and routing
accuracy. **Routing accuracy is the one that matters**, and it should be the
highest of the three: confusing two classes that share a bin costs nothing.

## What this will not fix

WaRP's 28 classes are all containers. It has no tableware, no organics, no
film. So it cannot teach:

- container jar vs drinking glass
- lined paper cup vs plain paper
- anything that composts

Those are the three cases the README opens with, and they still need your own
footage. What this run buys is a belt-aware model and a calibrated one — a much
better starting point for pre-labelling that footage when it arrives.

## Using the result

```bash
cp ~/Downloads/best_model.pt models/best_model.pt
```

The app prefers `models/best_model.pt` over stock weights automatically. No
code change, no config — `recyclevision/weights.py` has resolved it that way
since v0.3, and the "Demo mode" banner disappears on its own.
