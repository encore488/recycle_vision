# ♻️ RecycleVision AI

**Point it at waste. It tells you which bin each item goes in — and why.**

RecycleVision is a computer vision system for waste sorting. The long-term goal is
real-time perception for recycling conveyor belts, and eventually for automated
robotic sorting.

![RecycleVision AI](docs/screenshot.png)

> **Status:** v0.3, early but working. Runs on stock COCO weights while a
> waste-specific model is trained — see [ROADMAP.md](ROADMAP.md).

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

## How it works

```
image ──▶ Detector ──▶ Detection(class, confidence, box)
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

## Command line

For scripting, or just to check it works without a browser:

```bash
python -m recyclevision images/recycl_test.jpg
python -m recyclevision images/*.jpg --json
python -m recyclevision images/recycl_test.jpg --save-annotated out/
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

### Current scores (stock YOLOv8s, 9 detections over the 2 sample images)

```
  detection precision    78%   (boxes that are on a real object)
  routing accuracy       29%   (correct bin, of real objects)
  end-to-end correct     22%

  misreadings driving wrong bins:
    'bowl' is really 'metal can lid'      x1
    'cup'  is really 'steel food can'     x1
    'cup'  is really 'aluminium drink can' x1
```

Nine detections is a small sample, but the pattern is unambiguous: **every
wrong bin is a metal container misread as "cup" or "bowl"**, because COCO has
no class for a can. Two false positives sat on an empty seam in the conveyor
belt. This is the measured case for training a purpose-built model.

One result is worth calling out the other way. In the second sample a drinking
glass was detected as `cup` — the wrong class, but both route to landfill, so
the bin was still right. Bin-first routing absorbs misreads that never change
the destination, and only surfaces the ones that do.

## Writing a policy

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

The app is deployable to Streamlit Community Cloud or Hugging Face Spaces as-is.
`packages.txt` installs the system libraries `opencv-python` links against —
without them the app dies at import with `libGL.so.1: cannot open shared object
file`, which is invisible locally and immediate in deployment.

Stock weights download on first request, so the first page load after a cold
start is slower than subsequent ones.

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
policies/household.yaml Routing rules
tests/                  Weight-free test suite
```

## Tech stack

Python · Streamlit · Ultralytics YOLO · PyTorch · Pillow

## Roadmap

See [ROADMAP.md](ROADMAP.md). Next up: session metrics and impact accounting,
then video with object tracking and a virtual count line — the conveyor-belt
demo this is all building towards.

## License

MIT
