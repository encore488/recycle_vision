# RecycleVision AI — Roadmap

Working plan for taking RecycleVision from prototype to a demo-able, resume-ready,
and eventually genuinely useful tool.

**Status:** Prototype v0.2 — Streamlit UI + YOLO wrapper, no weights shipped.

## Goals, in priority order

1. **Runnable in 30 seconds by a stranger.** No weights hunt, no setup ritual, ideally a live URL.
2. **Answers a question a human actually has.** "4 objects at 83% confidence" is not useful.
   "That's recyclable — rinse it, lid goes separately" is.
3. **Demonstrates the conveyor-belt endgame** instead of promising it in a README.

## Standing decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Model for now | Stock `yolov8n.pt` (COCO) | No custom weights exist yet; stock keeps the repo clonable and the demo live today. |
| Model later | Custom model trained on real conveyor footage | The actual differentiator. Everything is built to swap the model without touching the UI. |
| Class handling | Detector classes → materials via a mapping layer | Decouples the app from whichever model is loaded. See "The material mapping layer" below. |
| Audience | Both resume/demo **and** eventual real tool | Milestones 1–3 serve both. Where they conflict, the demo wins until the video is shot. |

## Known defects in the current code

These block everything else and should be cleared first.

- [ ] **The app cannot start from a fresh clone.** `models/best_model.pt` is gitignored
      and there is no download or fallback path, so `WasteDetector(MODEL_PATH)` raises at
      import time. (`app.py:26`, `app.py:38`)
- [ ] **Annotated image has swapped color channels.** `result.plot()` returns BGR;
      `st.image()` assumes RGB. The "AI Detection" panel renders with red and blue inverted
      relative to the original beside it. Fix: `result.plot()[..., ::-1]`.
      (`vision/detector.py:50`)
- [ ] **Three sidebar controls are dead.** `show_labels`, `show_boxes`, `show_masks` are
      read but never passed anywhere — `get_annotated_image()` takes no arguments.
      Fix: forward them to `result.plot(labels=…, boxes=…, masks=…)`.
      (`app.py:57-71`, `vision/detector.py:50`)
- [ ] **Class names are hardcoded and unverified.** `recyclable_classes` and the `materials`
      list assume the model emits exactly `plastic`/`glass`/`metal`/`paper`/`waste`. Any other
      naming silently displays 0 for every count with no error. (`app.py:151`, `app.py:226`)
- [ ] **`st.metric("Model", "YOLOv8")` is hardcoded** rather than read from the loaded model.
      (`app.py:187`)
- [ ] **`requirements.txt` is UTF-16 and a full 62-line `pip freeze`.** pip reads it via the
      BOM, but it is unreadable in GitHub diffs and most editors, and it pins transitive deps.
      Rewrite as UTF-8 with ~8 direct dependencies, plus a `requirements-dev.txt`.
- [ ] **`README.txt` won't render on GitHub**, has an unclosed code fence that swallows the
      second half of the document, roadmap checkboxes that don't render, and instructs
      `cd recycle_sort` for a repo named `recycle_vision`.
- [ ] Unused `Path` import in `vision/detector.py:1`.

## The material mapping layer

The most important structural change, and the one that makes the stock-model-now /
custom-model-later plan work.

Stock YOLOv8 is COCO-trained: it has no `plastic`, `glass`, `metal`, or `paper` classes.
It has `bottle`, `wine glass`, `cup`, `bowl`, `fork`, `book`, `banana`, and so on. Rather
than fight that, introduce a mapping from **detector class → material**, loaded from config:

```
bottle      → plastic   (ambiguous: could be glass — flag low certainty)
wine glass  → glass
vase        → glass
cup         → paper
book        → paper
fork/knife/spoon/scissors → metal
banana/apple/orange/pizza/broccoli/carrot → organic  (compost — contamination if in the stream)
(everything else)         → waste
```

Two payoffs:

- The demo works **today** on real photos with stock weights.
- When the custom conveyor model lands, it ships an identity mapping and nothing in the UI
  changes. The abstraction is needed either way.

The UI should be honest about which model is loaded and how confident the mapping is — a
banner reading "Running on stock COCO weights; material inference is approximate" costs
nothing and reads as rigor rather than as a caveat.

---

## Milestone 1 — "It runs"

Nothing else matters until a stranger can click a link and see it work.

- [ ] Model bootstrap: try `models/best_model.pt`; if absent, auto-download stock
      `yolov8n.pt` on first launch and show a clear banner about which model is active.
- [ ] Fix the BGR swap.
- [ ] Wire up the three dead sidebar toggles.
- [ ] Build the material mapping layer; read class names from `model.names`.
- [ ] Rewrite `requirements.txt` as UTF-8 with direct dependencies only.
- [ ] `README.txt` → `README.md`: fixed fences, correct repo name, honest status, and an
      animated GIF of the app running at the top.
- [ ] Extract business logic (recyclable %, counts, material math) out of `app.py` into a
      `core/` module — one source of truth, testable without Streamlit.
- [ ] `pytest` suite with a stub detector so tests run without weights; golden-output
      regression on the two sample images in `images/`.
- [ ] GitHub Actions: ruff + pytest + an import smoke test.
- [ ] **Deploy to Streamlit Community Cloud or Hugging Face Spaces.**

> A live URL in the README is worth more than any single feature on this list.

**Done when:** `git clone && pip install -r requirements.txt && streamlit run app.py` works
on a clean machine, CI is green, and the README links to a working hosted demo.

## Milestone 2 — "It's useful"

The jump from demo to product. This is what makes non-engineers care.

- [ ] **Disposal guidance layer.** A pluggable JSON ruleset (`rules/`) mapping material →
      what to actually do: rinse, remove lid, which bin, and contamination warnings
      (greasy pizza box, plastic bags, black plastic). Structured so a second municipality
      is a data file, not a code change.
- [ ] **Contamination rate.** Percentage of non-recyclable items in the stream, with a
      red/amber/green verdict per frame. This is *the* metric materials recovery facilities
      actually track — it maps straight onto the conveyor goal and signals domain awareness.
- [ ] **Impact accounting.** Estimated mass (average mass per item class) and CO₂e avoided
      using published EPA WARM factors, accumulated across a session. Grounded in real
      figures, cited in the UI — never invented numbers.
- [ ] Per-detection table with bbox geometry; CSV/JSON export.
- [ ] Batch mode: upload N images, get aggregate statistics and a summary report.

**Done when:** the app tells you what to *do*, not just what it *sees*.

## Milestone 3 — "It's real"

The conveyor demo. This is what makes engineers care, and it is the centerpiece of the
resume video.

- [ ] **Video upload + object tracking.** `model.track(persist=True)` with ByteTrack, a
      virtual count line, each item counted exactly once as it crosses. Roughly a day of
      work and the highest impressiveness-per-hour item on this entire document.
- [ ] Throughput metrics: items/minute, FPS, per-frame latency distribution.
- [ ] Webcam / live stream input mode.
- [ ] **Record the demo video.** Conveyor footage in, live counts and contamination rate out.

**Done when:** there is a 60-second video showing material flowing past a count line with
live tallies, good enough to put on a resume.

## Milestone 4 — "It's credible ML"

Separates "used a model" from "understands ML". Largely gated on having real data.

- [ ] **Capture conveyor footage.** The blocking dependency for everything below.
- [ ] Dataset + data card: sourcing, label taxonomy, class balance, train/val/test split,
      known biases. Public options for bootstrapping before own footage exists: TACO,
      TrashNet, ZeroWaste.
- [ ] `train.py` with reproducible hyperparameters and logged runs.
- [ ] **Evaluation page:** mAP50-95, per-class precision/recall curves, confusion matrix,
      latency distribution on a held-out set.
- [ ] Publish trained weights as a GitHub Release asset; bootstrap prefers them over stock.
- [ ] **Active learning loop.** Let the user correct a wrong detection in the UI and write
      the corrected label to `data/feedback/` in YOLO format. Rarely built, feeds the
      dataset directly, and is a strong thing to be able to talk about in an interview.

**Done when:** the model is trained on real conveyor data and there are honest numbers
published for how well it performs.

## Milestone 5 — "It's a system"

Turns a Streamlit toy into something deployable.

- [ ] FastAPI `/detect` endpoint, with a `curl` example in the README. Also the natural seam
      for future robot/PLC integration.
- [ ] Dockerfile + compose.
- [ ] ONNX export and a CPU latency comparison table — delivers the "edge deployment"
      roadmap item in a demonstrable form.
- [ ] Config via `config.yaml` / typed settings rather than hardcoded paths and constants.

## Backlog — differentiators

Not scheduled. Pull forward whichever fits the moment.

- **Hybrid VLM fallback.** When YOLO is low-confidence or the object is outside the class
  vocabulary, send the crop to a vision-language model for a plain-language "what is this /
  is it recyclable" answer. Fast-detector-plus-VLM-for-hard-cases is a modern architecture
  and it fixes the closed-set limitation that will otherwise always cap this project.
- **Resin code OCR.** Detect and read the ♳–♹ triangle number on plastics. Very few projects
  do it, and it is exactly the detail that makes recycling decisions actually correct.
- **Instance segmentation** (`yolov8-seg`) → per-item area → better mass estimates. Also
  finally makes the `show_masks` toggle mean something.
- **Metric sizing** via an ArUco fiducial marker in frame.
- **Robot integration stub.** Publish detections over MQTT / ROS 2 with pick coordinates and
  a suggested gripper per item. Even as a stub, it proves the endgame was thought through.

## Working conventions

- Tests and CI are part of each milestone, not a cleanup pass afterward.
- The UI never claims more certainty than the model has — banner the stock-weights caveat,
  show confidence, cite sources for impact figures.
- Every milestone ends in something demonstrable, because the resume video may need to be
  shot at any point.
