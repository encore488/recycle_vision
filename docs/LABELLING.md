# Labelling conveyor footage

Everything here is arranged around one fact: **your attention is the scarce
resource, not compute.** Read the first two sections before annotating
anything.

---

## ⚠️ Read this first, when you come back with footage

You asked for a reminder if the labelling estimate exceeded five hours. It
does, for the naive approach — roughly **8–15 hours** to draw 300 crowded
conveyor frames from scratch. The workflow below cuts that to **2–4 hours**,
but before you start, spend twenty minutes checking whether the tooling has
moved on. It changes fast, and a better auto-labeller is worth far more than
an early start.

**Check for, in rough order of likely payoff:**

1. **A newer open-vocabulary detector.** This project uses YOLOE. Check whether
   a stronger checkpoint or successor exists in `ultralytics`, and whether a
   larger size (`yoloe-11m-seg`, `yoloe-11l-seg`) meaningfully improves
   pre-label quality — pre-labelling is offline, so slow-but-accurate is a
   good trade there even though it would be wrong for the live app.
2. **A newer SAM.** SAM 2 / successors turn a drawn box into a mask. Better
   masks mean less correction.
3. **A model already trained on waste.** Hugging Face, Roboflow Universe and
   Kaggle host waste datasets and trained models (search: ZeroWaste, TACO,
   TrashNet, "MRF", "waste sorting"). One trained on conveyor imagery would
   beat anything zero-shot for pre-labelling.
   **Note:** those hosts are unreachable from the sandbox this project is
   developed in — only GitHub and PyPI are. So this step has to happen on your
   machine, or in a session with wider network access.
4. **Whether a hosted labelling service now does this well enough** to be worth
   the money against your hours.

Then re-measure against `qa/openvocab_v2/verdicts.json` before switching:
better on paper is not better on your belt.

---

## Do I need to draw segmentation masks?

**No. You will never draw a polygon.**

You need masks *eventually* — but you will not be the one producing them.

**Why masks matter for your goal.** You want to reach out and grab the item. A
bounding box gives a centre point and an extent; in a frame with 10–40
overlapping objects, the centre of a box very often lands on a *different*
object underneath. A mask gives the actual silhouette, so a grasp point can be
chosen on a surface that genuinely belongs to the target. Boxes are enough to
count a stream; masks are what a gripper needs.

**Why you still do not draw them.** Two things do it for you:

- The shipped detector (`yoloe-11s-seg`) is a **segmentation** model. Its
  pre-labels already include outlines — verified, not assumed.
- For objects it misses entirely, **SAM turns a box you drew into a mask**
  (`mobile_sam.pt`, also verified working here).

So the human job is: **check the class, fix the box, delete what is not real,
draw the few that were missed.** Masks follow automatically.

**One thing to know about the run directory.** `train.py` reads your labels
before it starts and picks a model to match: `yolo11s-seg.pt` when they are
polygons, `yolo11s.pt` when they are boxes. Ultralytics then writes to
`runs/segment/` or `runs/detect/` accordingly. `prelabel.py` outputs polygons
unless you pass `--boxes-only`, so the paths below say `segment` — but read the
path `train.py` prints rather than assuming it. An outside dataset of boxes
(WaRP, say) trains a detection model and lands in `runs/detect/`.

---

## How many frames?

**Instances matter more than images.** With 10–40 objects per frame you
accumulate instances fast, so the binding constraint is not the frame count —
it is your *rarest class*.

| Target | Frames | Why |
| --- | --- | --- |
| First usable model | **150–200** | ~3,000–6,000 instances. Enough to beat zero-shot on your belt. |
| Solid model | **300–400** | Common classes get thousands of examples. |
| Diminishing returns | beyond ~600 | Unless you are adding *new* conditions — different lighting, a different shift, a fuller belt. |

**Rule of thumb per class:** under ~100 instances a class will not learn;
500–1,000+ is comfortable. `scripts/prelabel.py` prints the per-class counts
and warns about thin classes, so you will see this rather than guess.

Your rare classes will be the problem, not your common ones. Cans will hit
thousands while, say, foam hits twelve. The fix is not more frames at random —
it is **targeted frames containing the rare thing**. Scrub the footage for them
deliberately.

**On frame selection:** do not label consecutive frames. Two frames 1/30th of a
second apart are the same picture; labelling both costs twice as much, teaches
nothing, and — worse — inflates your validation score, because near-identical
frames end up on both sides of the split. `scripts/extract_frames.py` drops
near-duplicates automatically, and the train/val split is by frame *block*
rather than at random for the same reason.

---

## How long will it take?

**Do not trust these numbers. Measure yours.**

| Approach | Per frame | 300 frames |
| --- | --- | --- |
| Drawing from scratch | 1.5–3 min | **8–15 h** |
| Correcting good pre-labels | 20–40 s | **2–3.5 h** |
| Correcting mediocre pre-labels | 45–90 s | **4–7 h** |

Which row you land on depends entirely on how well the pre-labeller does on
*your* footage, and nobody can tell you that in advance — including me. The
current model scores 85% class accuracy on two clean sample photos; a crowded,
motion-blurred belt will be worse, and **missed** objects are the expensive
failure, because those you draw from scratch.

**So calibrate before committing.** Label 20 frames. Time it. Multiply.
If the answer is over five hours, stop and go back to the checklist at the top
of this file rather than grinding through it.

---

## The workflow

Your instinct — label a little, train on it, use that to label the rest — is
right, and it is what this pipeline is built for. It beats one-shot
pre-labelling because the model improves *while* you work.

### Round 0 — extract frames

```bash
python scripts/extract_frames.py belt.mp4 --out datasets/raw --every 10 --max 400
```

Prints how many near-duplicates it dropped. Each one is a frame you did not
have to label.

### Round 1 — pre-label with the stock model, correct ~50 frames

```bash
python scripts/prelabel.py datasets/raw --out datasets/round1
```

Open `datasets/round1` in any YOLO-format editor — **Label Studio** or **CVAT**
both import it directly, and both run locally — and correct the first ~50.

Time this round. It is your calibration.

### Round 2 — train on those, re-label with the result

```bash
python train.py --data datasets/round1/data.yaml --epochs 100
python scripts/prelabel.py datasets/raw --out datasets/round2 \
    --weights runs/segment/<run>/weights/best.pt
```

A model trained on your own belt pre-labels it far better than a zero-shot one.
Correct the next batch — it should be noticeably faster.

### Repeat

Each round the corrections shrink. Stop when the pre-labels are good enough
that correcting a frame is faster than your patience for it.

### Finally

```bash
python scripts/evaluate.py --weights runs/segment/<run>/weights/best.pt \
    --data datasets/roundN/data.yaml
cp runs/segment/<run>/weights/best.pt models/best_model.pt
```

The app prefers `models/best_model.pt` over stock weights automatically.

---

## What to actually correct

Spend your attention where it changes a bin, and nowhere else.

**Always fix:**
- A box on nothing, or on the belt itself → delete it.
- A missed object → draw it.
- A class error **that changes the bin** — a drinking glass read as a jar, a
  paper cup read as paper. These are the errors that contaminate a batch.

**Do not bother fixing:**
- A class error **that does not change the bin** — a steel can read as an
  aluminium one. Both go to the same place. `scripts/evaluate.py` reports these
  separately as "confusions that did not matter", precisely so you can ignore
  them with a clear conscience.
- Pixel-perfect box edges. Close is fine.

If a distinction never changes a destination, consider whether the vocabulary
should be asking for it at all — that question is what took class accuracy from
62% to 85% without any new data. See the note at the top of
`vocab/waste_v2.yaml`.

---

## Before you train: write the data card

`docs/DATA_CARD.md` is a template. Fill it in while the details are fresh —
what the footage is, what the lighting was, which classes are thin, what is
*not* represented. A model's failures are usually predictable from its data,
but only if someone wrote the data down.
