# RecycleVision AI — Roadmap

Working plan for taking RecycleVision from prototype to a demo-able, resume-ready,
and eventually genuinely useful tool.

**Status:** v0.3 — rebuilt around bin routing. Runs on stock YOLO weights.

## The product, in one sentence

**Point it at waste; it tells you which bin each item goes in, and why.**

Not "what material is this" — *where does this go*. That distinction drives everything below.

## Goals, in priority order

1. **Runnable in 30 seconds by a stranger.** No weights hunt, no setup ritual, ideally a live URL.
2. **Answers a question a human actually has.** "4 objects at 83% confidence" is not useful.
   "Blue bin — rinse it first; the cap goes in too" is.
3. **Demonstrates the conveyor-belt endgame** instead of promising it in a README.

## Core architecture: detection is not the product, routing is

```
image ──▶ Detector ──▶ Detection(class, confidence, box)
                            │
                            ▼
                     RoutingPolicy  ◀── policies/*.yaml   (facility-specific)
                            │
                            ▼
              RoutedItem(bin, handling notes, certainty)
                            │
                            ▼
                   SortResult ──▶ UI / API / robot
```

Three swappable pieces, each isolated behind a small interface:

- **Detector** — stock COCO YOLO today, custom conveyor-trained model later. The rest of the
  system never learns which.
- **RoutingPolicy** — a YAML file, not code. Maps whatever classes the detector emits onto
  bins, with handling notes and a certainty flag. A new municipality or facility is a new
  file.
- **Presentation** — Streamlit today, FastAPI and robot control later, reading the same
  `SortResult`.

### Why this shape

- The eventual custom model ships as an *identity-ish* policy; no UI changes.
- A MRF's bins (PET / HDPE / OCC / aluminum / residue) and an office's bins
  (recycling / compost / landfill / special) are the same code, different config.
- Contamination rate falls out for free: it is the residue share of the stream.
- Low-certainty routes are an explicit first-class outcome, which is exactly the hook that
  later feeds the VLM fallback and the active-learning loop.

### Bins, not materials

Materials are still tracked as an item attribute, but they are not the output. This matters
because material does not determine destination:

- A **wine glass** is glass and belongs in **landfill** — drinking glass has a different
  melt point than container glass and contaminates the batch.
- A **disposable coffee cup** is paper and belongs in **landfill** — it is plastic-lined.
- A **pizza slice** is organic and belongs in **compost**, not recycling.

A material-first design gets all three of these wrong. A bin-first design gets them right and
can explain itself.

## Standing decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Output | Bin routing decision | The thing a user or a robot actually needs. |
| Model now | Stock `yolov8n.pt` (COCO) | No custom weights exist; keeps the repo clonable and the demo live today. |
| Model later | Custom model trained on real conveyor footage | The actual differentiator. Swapped without touching the UI. |
| Routing rules | YAML policy files, not code | Facility-specific; must be editable without a deploy. |
| Non-waste classes | Explicitly ignored, not counted | COCO detects people, cars, dogs. They are not waste. |
| Audience | Both resume/demo **and** eventual real tool | Milestones 1–3 serve both; where they conflict, the demo wins until the video is shot. |

## Restructure: what happened to v0.2

The v0.2 code was a Streamlit UI calling a YOLO wrapper, with material logic hardcoded across
`app.py`. Rather than a ground-up rewrite (wasteful — the UI shape was sound), v0.3:

- **Replaces** `vision/` with a `recyclevision/` package: `detector`, `policy`, `pipeline`,
  `render`, `models`, `weights`.
- **Rewrites** `app.py` as presentation only — no domain logic.
- **Deletes** `vision/test_model.py` (a scratch script), superseded by real tests and a CLI.
- **Keeps** the page layout, the sidebar-controls idea, and the sample images.

---

## Milestone 1 — "It runs, and it routes"  ← in progress

A stranger can clone it, run it, and get a correct, explained bin decision.

- [ ] `recyclevision/` package: `Detector` protocol + `YoloDetector`, `RoutingPolicy`,
      `SortingPipeline`, dataclass domain models.
- [ ] `policies/household.yaml` — COCO classes → bins, with handling notes, certainty flags,
      and explicit non-waste ignores.
- [ ] Model bootstrap: prefer `models/best_model.pt`; auto-download stock `yolov8n.pt` if
      absent; banner clearly stating which model is live.
- [ ] Bin-coloured annotation rendered in RGB via PIL — fixes the v0.2 BGR/RGB channel swap
      by not round-tripping through `result.plot()` at all, and colours each box by
      *destination* rather than by class.
- [ ] Sidebar controls that actually do something (the v0.2 toggles were wired to nothing).
- [ ] `app.py` rewritten: bin cards, diversion rate, per-item explanations, review queue.
- [ ] Headless CLI (`python -m recyclevision`) so the pipeline is testable and scriptable
      without Streamlit.
- [ ] `pytest` suite with a stub detector — full pipeline coverage with no weights, no
      network, no torch.
- [ ] `requirements.txt` rewritten as UTF-8 with direct dependencies only (v0.2's was UTF-16
      and a 62-line `pip freeze`); `requirements-dev.txt` split out.
- [ ] `README.txt` → `README.md` (v0.2's had an unclosed code fence swallowing half the doc
      and the wrong repo name).
- [ ] GitHub Actions: ruff + pytest.
- [ ] **Deploy to Streamlit Community Cloud or Hugging Face Spaces.**

> A live URL in the README is worth more than any single feature on this list.

**Done when:** clean-machine clone runs, CI is green, README links a working hosted demo.

## Milestone 2 — "It's quantified"

Make the routing decisions measurable and exportable.

- [ ] **Diversion and contamination rates** as headline metrics, tracked across a session.
- [ ] **Impact accounting.** Estimated mass (average mass per item class) and CO₂e avoided
      using published EPA WARM factors. Grounded in real, cited figures — never invented.
- [ ] Per-detection table with bbox geometry; CSV/JSON export; batch mode over N images with
      an aggregate report.
- [ ] A second policy file (a real municipality's rules) to prove the abstraction holds, plus
      a policy picker in the UI.
- [ ] Policy schema validation with helpful errors, so a hand-edited YAML fails loudly.

## Milestone 3 — "It's real"

The conveyor demo, and the centrepiece of the resume video.

- [ ] **Video upload + object tracking.** `model.track(persist=True)` with ByteTrack, a virtual
      count line, each item counted exactly once as it crosses. Roughly a day of work and the
      highest impressiveness-per-hour item in this document.
- [ ] Per-bin running tallies and throughput: items/minute, FPS, latency distribution.
- [ ] Webcam / live stream input mode.
- [ ] **Record the demo video.** Conveyor footage in, live bin tallies and contamination rate out.

**Done when:** there is a 60-second video of material flowing past a count line with live
per-bin tallies, good enough for a resume.

## Milestone 4 — "It's credible ML"

Separates "used a model" from "understands ML". Gated on real data.

- [ ] **Capture conveyor footage.** The blocking dependency for everything below.
- [ ] Dataset + data card: sourcing, label taxonomy, class balance, splits, known biases.
      Public bootstraps: TACO, TrashNet, ZeroWaste.
- [ ] Label taxonomy designed *backwards from the bins* — classes should be the distinctions
      that change a routing decision, not an arbitrary material ontology.
- [ ] `train.py` with reproducible hyperparameters and logged runs.
- [ ] **Evaluation page:** mAP50-95, per-class PR curves, confusion matrix, latency
      distribution on a held-out set. Report *routing* accuracy, not just detection mAP —
      a confusion between two classes that share a bin costs nothing.
- [ ] Publish weights as a GitHub Release asset; bootstrap prefers them over stock.
- [ ] **Active learning loop.** Correct a wrong route in the UI; write the corrected label to
      `data/feedback/` in YOLO format. Feeds the dataset and makes a great interview story.

## Milestone 5 — "It's a system"

- [ ] FastAPI `/sort` endpoint returning `SortResult` as JSON, with a `curl` example. The
      natural seam for robot/PLC integration.
- [ ] Dockerfile + compose.
- [ ] ONNX export and a CPU latency table — the "edge deployment" item, made demonstrable.
- [ ] Typed settings/config rather than constants.

## Backlog — differentiators

- **Hybrid VLM fallback.** Route low-certainty items to a vision-language model for a
  plain-language identification. Fixes the closed-set limitation that otherwise caps this
  project permanently. The certainty flag from Milestone 1 is the hook.
- **Resin code OCR.** Read the ♳–♹ triangle on plastics — often *the* fact that determines
  the correct bin.
- **Instance segmentation** → per-item area → better mass estimates, and makes a masks toggle
  meaningful.
- **Metric sizing** via an ArUco fiducial in frame.
- **Robot integration stub.** Publish `SortResult` over MQTT / ROS 2 with pick coordinates and
  a suggested gripper per bin.

## Working conventions

- Tests and CI are part of each milestone, not a cleanup pass afterward.
- Domain logic lives in `recyclevision/`; `app.py` is presentation only.
- Facility- and region-specific knowledge lives in `policies/`, never in code.
- The UI never claims more certainty than it has — banner the stock-weights caveat, surface
  low-certainty routes for review, cite sources for impact figures.
- Every milestone ends in something demonstrable, because the resume video may need to be shot
  at any point.
