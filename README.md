# ♻️ RecycleVision AI

**Point it at waste. It tells you which bin each item goes in — and why.**

**[▶ Try the live demo](https://recyclevision-vfhgb8vencieb6ydhtzjcw.streamlit.app/)**

RecycleVision is a computer vision system for waste sorting. The long-term goal is
real-time perception for recycling conveyor belts, and eventually for automated
robotic sorting.

![RecycleVision AI](docs/screenshot.png)

> **Status:** v0.8, working and deployed. Ships two detectors and two
> routing policies, measured against held-out sorting-plant data. The road to
> v1.0 — and what is deliberately *not* in it — is [ROADMAP.md](ROADMAP.md).

> Working on this repo? **[CLAUDE.md](CLAUDE.md)** is the orientation:
> current measurements, settled decisions, and the traps that have
> already cost time.

## Bins, not materials

Most waste classifiers tell you what something is made of. That is the wrong
answer, because material does not determine destination:

| Item | Material | Correct bin | Why |
| --- | --- | --- | --- |
| Wine glass | Glass | **Landfill** | Drinking glass melts at a different temperature and ruins a batch of recycled container glass. |
| Disposable coffee cup | Paper | **Landfill** | Plastic-lined; rejected by most programs. |
| Pizza slice | Organic | **Compost** | Food waste, not recycling. |
| Plastic bottle | Plastic | **Recycling** | The easy case still has to work. |

A material-first design gets the first three wrong. RecycleVision routes to bins
directly, attaches handling instructions, and flags the items where it genuinely
cannot be sure.

## Two detectors

The project ships two, and measures both. The difference is the whole story:

| | Stock COCO | Open vocabulary |
| --- | --- | --- |
| Model | YOLOv8s | YOLOE + `vocab/waste_v2.yaml` |
| Class list | Fixed at training time, 80 everyday objects | Given in words, editable in a YAML file |
| A steel can | "bowl" or "cup" | "metal can" |
| A drink carton | no such class | "beverage carton" |

This table used to carry accuracy figures. They were measured on two
photographs that the vocabulary had been tuned against, so they described a
fit rather than a performance, and they are retired. **The honest numbers are
below, on held-out sorting-plant data.**

COCO has no class for a drink can — the most common item in a recycling stream —
so a COCO detector reports one as tableware and no downstream rule can undo
that. An open-vocabulary detector is told what to look for in words, so the
class list becomes configuration. `vocab/waste_v2.yaml` simply asks for
`metal can`.

Turning those words into embeddings needs a text encoder ten times the size of
the detector, so that happens once, offline, and the 40KB result is committed.
At runtime nothing but the cached embeddings is loaded — which is what keeps
the app deployable on a small host.

```bash
# Only when the vocabulary changes:
pip install -r requirements-vocab.txt
python scripts/build_vocab_embeddings.py vocab/waste_v2.yaml
```

## How it works

```
image ──▶ Detector ──▶ Detection(class, confidence, box)
             ▲
     vocab/*.yaml (open-vocabulary class prompts)
                            │
                            ▼
                     RoutingPolicy  ◀── policies/*.yaml
                            │
                            ▼
              RoutedItem(bin, handling, certainty)
                            │
                            ▼
                       SortResult ──▶ UI / CLI
```

Three swappable pieces:

- **Detector** — stock YOLOv8 today, a conveyor-trained model later. Nothing
  downstream knows the difference.
- **RoutingPolicy** — a YAML file, not code. Local recycling rules live in
  `policies/`, so a new municipality or facility is a new file rather than a
  deploy.
- **Presentation** — Streamlit and a CLI today; an HTTP API and robot control
  later, all reading the same `SortResult`.

## Quick start

```bash
git clone https://github.com/encore488/recycle_vision.git
cd recycle_vision

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

No model download step: if `models/best_model.pt` is absent, stock `yolov8s.pt`
is fetched automatically on first run and the UI says clearly that it is running
in demo mode. Sample images are included, so you can try it without a photo of
your own.

## Batch mode and export

Upload several photos at once and the app reports totals across the whole
stream — diversion rate, contamination, stream composition — rather than per
image. Every run can be downloaded as CSV (one row per item, geometry
included) or JSON.

### Impact estimates

Mass and avoided emissions are estimated from `impact/factors.yaml`.

**That file ships with unverified placeholder numbers**, and carries
`verified: false` to say so. The app propagates that flag to a visible warning
on every figure it derives. The arithmetic is sound; the constants are
order-of-magnitude guesses, deliberately rounded so they cannot be mistaken
for measurements.

To make them real, replace each value with a cited one and flip the flag:

- **Carbon** — US EPA WARM, "Recycling vs. Landfilling" factors, in MTCO2E per
  short ton. Convert: 1 MTCO2E/short ton = 1.102 kg CO2e per kg.
- **Mass** — weigh a sample of your own stream. Container masses vary hugely by
  brand and region.

Only diverted items count toward avoided emissions: recycling something is what
avoids the emission, so a fully contaminated stream correctly scores zero rather
than scoring for the material it happens to contain.

## Command line

For scripting, or just to check it works without a browser:

```bash
python -m recyclevision images/<a sample>.jpg
python -m recyclevision images/*.jpg --impact        # batch totals + estimates
python -m recyclevision images/*.jpg --csv > out.csv
python -m recyclevision images/*.jpg --json
python -m recyclevision images/<a sample>.jpg --save-annotated out/
```

```
images/recycl_test.jpg
  model  YOLOv8s (COCO)
  policy Household Single-Stream
  6 item(s), 33% diverted from landfill

  Mixed Recycling (2)
    - Beverage bottle  72%
        Empty and rinse. Leave the cap screwed on.
    - Beverage bottle  16%
        Empty and rinse. Leave the cap screwed on.

  Landfill (4)
    - Bowl  61%  [review]
        Scrape food residue into the organics bin first.
    - Cutlery  56%  [review]
        Metal cutlery should be kept, donated, or taken to scrap.
    - Cup  30%  [review]
        If it is a ceramic mug, keep or donate it instead.
    - Cutlery  18%  [review]
        Metal cutlery should be kept, donated, or taken to scrap.
```

Those `[review]` flags are the system working, not failing. The sample is a
photo of a real conveyor belt carrying steel cans — and **COCO has no class
for a can**, so the model reports them as "bowl", "cup" and "cutlery", which
the policy routes to landfill. Wrongly.

That is a limitation of the detector, not the routing, and it is the clearest
possible argument for training a purpose-built model — the next major
milestone. Until then the app says so on every screen rather than quietly
reporting a confident wrong answer.

## QA: grading the model

Judging a detector from one cluttered annotated image does not work — boxes
overlap, labels get displaced, and small false positives hide in the noise.
`recyclevision.qa` cuts each detection into its own captioned tile so a person
can grade them honestly:

```bash
python -m recyclevision.qa sheet images/*.jpg --out qa/
# ...grade qa/verdicts.json by hand...
python -m recyclevision.qa score qa/verdicts.json
```

Verdicts separate the two failure modes, because they have different fixes:

| Verdict | Means | Fix |
| --- | --- | --- |
| `correct` | Right object, right bin | — |
| `wrong_bin` | Real object, well localised, wrong destination | Detector or policy |
| `false_positive` | Box is not on an object at all | Detector |
| `unsure` | Cannot tell from the image | — |

### Current scores

Measured on held-out splits of real sorting-plant data. The two development
photographs these numbers used to come from are retired — the vocabulary was
tuned against them, so they measured a fit, not a performance.

**A detector fine-tuned on WaRP** (2,974 images from one plant):

| tested on | mAP50 | class acc | routing acc (of matched) | recall |
| --- | --- | --- | --- | --- |
| WaRP val — the same plant | 0.671 | 96.0% | 96.1% | 56.7% |
| ZeroWaste — a plant it has never seen | 0.033 | 31.5% | 56.5% | 6.3% |

**Read the second row.** 96% on the plant it trained on, 56% on a plant it did
not: it learned one conveyor belt rather than waste. Generalisation is the
open problem, and a single-facility number is not evidence about it.

**A detector trained on a SortWaste + ZeroWaste pool**, scored on WaRP — a
third facility it never saw:

| | mAP50 | class acc | routing acc (of matched) | recall |
| --- | --- | --- | --- | --- |
| WaRP — a plant it has never seen | 0.071 | 32.7% | 42.0% | 38.5% |

**Those two cross-facility rows cannot be compared as they stand**, and the
reason matters more than either number. Routing accuracy is computed over
*matched* instances, so it flatters a model that finds only easy objects. The
pool model matched 38.5% of labelled objects where the WaRP model matched
6.3%. Over the whole stream:

| model, on an unseen plant | routed correctly, of **all** labelled objects |
| --- | --- |
| trained on WaRP | ~3.6% |
| trained on the pool | **16.2%** |

Roughly 4.5× better, from the model the headline metric called worse.
`scripts/evaluate.py` now reports this end-to-end figure beside the
conditional one and says which denominator each uses. **Compare models on the
end-to-end number.**

**Routing accuracy is 1.8× class accuracy on the unseen plant** (56.5% vs
31.5%), and the single largest confusion there — `beverage carton` read as
`cardboard box`, 42 times — is a known disagreement between the two datasets'
annotation schemes. Both route to fibre, so it costs nothing. A report
quoting only mAP would have called that a failure.

**Zero-shot, before any training**, on the same WaRP split: 52.6% recall at
56% fair precision — but only at confidence 0.01, collapsing to 0.9% at 0.25.
The objects were always being found and ranked terribly. Training fixed the
ranking, not the eyes.

On precision: WaRP labels 3.5 objects per image in frames holding dozens, so
a correct detection of an unlabelled object scores as an error. True precision
there lies between the raw figure and a "fair" one counting false positives
only for classes the dataset actually labels. `scripts/zeroshot_eval.py`
reports both bounds rather than picking the flattering one.

## How the vocabulary is designed

The rule: **ask for a distinction only where it changes a bin decision.**

v1 asked the detector to separate aluminium cans from steel cans. Both go in
the same bin under every shipped policy, and a materials recovery facility
separates them mechanically downstream anyway — so the distinction bought
nothing and cost accuracy, producing the single largest source of mislabels.
v2 merges them into `metal can`.

The reverse also applies. Container glass and drinking glass *do* go to
different bins — a tumbler ruins a batch of recycled container glass — so that
split is kept and reinforced, even though both are "glass".

One prompt was deleted outright. v1's `styrofoam` fired on a sheet of paper,
sending recyclable fibre to landfill. Every replacement tested (`foam cup or
tray`, `white foam packaging`, `polystyrene foam block`) captured the same
sheet of paper, because neither validation image contains foam and the prompt
settled on the nearest pale flat object. A prompt that cannot be shown to work
is a liability rather than a gap, so it was removed rather than reworded again.

## Writing a policy

Two policies ship, and they disagree on purpose:

| | `household.yaml` | `mrf_conveyor.yaml` |
| --- | --- | --- |
| Context | A kitchen | A sorting line |
| Bins | Recycling / Organics / Reuse / Special / Landfill | Containers / Fibre / Organics / Residue |
| A detected `cup` | Landfill — it is a lined disposable cup | Containers — there is no tableware on a belt |

A policy maps detector class names onto bins. Only `bin` is required:

```yaml
name: Household Single-Stream
default: ignore          # unlisted classes are not waste, so don't count them

bins:
  - key: recycling
    name: Mixed Recycling
    color: "#1E6FD9"
    diverted: true       # false for landfill/residue; drives the diversion rate

rules:
  wine glass:
    bin: landfill
    item: Drinking glass
    material: glass
    handling: Wrap before binning if broken.
    certainty: high
    rationale: >-
      Not a container glass — it melts at a different temperature and will
      ruin a batch of recycled container glass.
```

`certainty: low` marks a route the system is not sure of. Those items are
surfaced in a review queue rather than counted silently, which is also the hook
for a future model-assisted fallback.

`default: ignore` matters more than it looks: COCO detects people, cars and
furniture constantly, and counting them as waste would corrupt every metric on
the dashboard.

## Deploying

Live at
[recyclevision.streamlit.app](https://recyclevision-vfhgb8vencieb6ydhtzjcw.streamlit.app/),
deployed from `main` on Streamlit Community Cloud.

`packages.txt` installs the system libraries `opencv-python` links against. It
contains **bare package names only** — Streamlit Cloud passes every line
straight to `apt-get install` and does not strip comments, so a `#` line is an
argument, not a comment. (A comment containing `->` once failed a deploy with
`E: Command line option '>' [from ->] is not understood`.) `tests/test_deployment.py`
enforces this.
**These are installed when the container is built, not on every redeploy** — so
if the app was first created before `packages.txt` existed, a plain redeploy
will keep failing with `libGL.so.1: cannot open shared object file`. Use
*Manage app → Reboot app* to force a container rebuild.

`requirements.txt` pulls CPU-only torch from PyTorch's own index; the default
PyPI wheel bundles CUDA and is roughly three times the size, which matters on a
constrained builder.
Model weights download on first request, so the first page load after a cold
start is slower than subsequent ones.

## Training on your own footage

The pipeline is built for one situation: crowded conveyor frames, 10–40 objects
each, and a person whose hours are the scarce resource.

```bash
python scripts/extract_frames.py belt.mp4 --out datasets/raw     # drops near-duplicates
python scripts/prelabel.py datasets/raw --out datasets/round1    # boxes AND masks, free
# ...correct in Label Studio or CVAT...
python train.py --data datasets/round1/data.yaml
python scripts/evaluate.py --weights runs/segment/<run>/weights/best.pt     --data datasets/round1/data.yaml
```

Then pre-label the next batch with the model you just trained:

```bash
python scripts/prelabel.py datasets/raw --out datasets/round2     --weights runs/segment/<run>/weights/best.pt
```

Each round the pre-labels get better and the correcting gets faster.

**You never draw a polygon.** The shipped detector is a segmentation model, so
pre-labels carry outlines already; for objects it misses, SAM turns a box you
drew into a mask. The human job is checking classes and fixing boxes.

**Evaluation reports routing accuracy, not just mAP.** Confusing an aluminium
can for a steel one is a class error worth nothing — both go in the same bin.
Confusing a drinking glass for a jar contaminates a batch. `scripts/evaluate.py`
separates "confusions that changed the bin" from "confusions that did not", so
effort goes where it changes an outcome.

### Training on a public dataset

Public waste datasets label *materials*; this project routes *bins* from
*items*. A mapping file translates one to the other and records what the
translation cannot express:

```bash
python scripts/import_dataset.py path/to/external/data.yaml     --mapping mappings/sortwaste.yaml --out datasets/sortwaste --link
python scripts/zeroshot_eval.py --data datasets/sortwaste/data.yaml
```

An unmapped source class is an error, not a silent drop — discarding one would
teach the model those objects are background. Nothing is written until the
mapping is complete.

`zeroshot_eval.py` scores a vocabulary against labelled ground truth, which is
what makes prompt tuning measurable rather than guesswork.

**No NVIDIA GPU?** `--device auto` is the default and picks CUDA, then Apple
Silicon's `mps`, then CPU. For a free hosted T4 see
[docs/TRAINING_ON_GPU.md](docs/TRAINING_ON_GPU.md) and the ready-to-run
[Colab notebook](notebooks/train_colab.ipynb).

### Docs

| file | what it is for |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | **Start here.** Current state, settled decisions, known traps |
| [ROADMAP.md](ROADMAP.md) | Milestones, what is done, what is measured |
| [docs/BUILDING_A_ROBUST_MODEL.md](docs/BUILDING_A_ROBUST_MODEL.md) | Combining datasets, choosing a holdout, what to add next |
| [docs/TRAINING_ON_GPU.md](docs/TRAINING_ON_GPU.md) | mps, Colab, Kaggle, and the failure modes of each |
| [docs/LABELLING.md](docs/LABELLING.md) | Annotating your own footage, and the bootstrap loop |
| [docs/DATA_CARD.md](docs/DATA_CARD.md) | What each dataset contains, and its licence |
| [docs/DATA_CARD_TEMPLATE.md](docs/DATA_CARD_TEMPLATE.md) | Blank card to fill in for your own footage |
| [docs/history/](docs/history/) | Superseded plans, kept for their findings |

## Development

```bash
pip install -r requirements-dev.txt
pytest              # 80 tests, no weights or network needed
ruff check .
ruff format .
```

The suite drives the whole pipeline through a `StubDetector`, so routing,
metrics and rendering are all testable without torch or a model download. CI
runs the same three commands.

## Project layout

```
app.py                  Streamlit UI (presentation only)
recyclevision/
  models.py             Domain types: Detection, Bin, RoutedItem, SortResult
  detector.py           Detector protocol, YOLO implementation, test stub
  policy.py             Policy loading, validation, and routing
  pipeline.py           Detect, then route
  render.py             Bin-coloured annotation
  weights.py            Custom weights if present, stock if not
  __main__.py           CLI entry point
  dataset.py            Frame sampling, splits, YOLO label formatting
  evaluate.py           Routing-aware scoring
  impact.py             Mass and CO2e estimates
  report.py             CSV / JSON export
  session.py            Totals across a batch
policies/*.yaml         Routing rules
vocab/*.yaml            Open-vocabulary prompts (+ cached embeddings)
impact/factors.yaml     Mass and carbon factors (placeholders)
train.py                Fine-tune on corrected labels
scripts/                Frame extraction, pre-labelling, evaluation
docs/LABELLING.md       How to label footage without wasting your time
tests/                  Weight-free test suite
```

## Tech stack

Python · Streamlit · Ultralytics YOLO · PyTorch · Pillow

## Roadmap

See [ROADMAP.md](ROADMAP.md), which is organised as phases with gates and an
explicit list of what v1.0 does *not* include.

Next up: video with object tracking and a virtual count line — the
conveyor-belt demo this is all building towards. It is deliberately **not**
gated on the detector improving, because the detector is a swappable component
and v1.0 should not be hostage to data we may never obtain.

## License

MIT
