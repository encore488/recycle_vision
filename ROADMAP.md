# RecycleVision AI — Roadmap

Working plan for taking RecycleVision from prototype to a demo-able, resume-ready,
and eventually genuinely useful tool.

**Status:** v0.7 — training pipeline ready. Batch mode, export and impact estimates. Open-vocabulary detection with vocabulary v2. 100% detection precision,
85% class accuracy, 100% routing accuracy on the (small, fitted) sample set.
[Live demo](https://recyclevision-vfhgb8vencieb6ydhtzjcw.streamlit.app/).

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
| Model now | **YOLOE open-vocabulary + `vocab/waste_v1.yaml`** | Zero-shot, and its class list is configuration — so it can be asked for "aluminum drink can", which COCO cannot express at all. |
| COCO detector | Kept as the measured baseline | Every claim of improvement is against a number, not a memory. |
| Text embeddings | Precomputed offline, committed (40KB) | The text encoder is ~570MB; needing it at request time would make the app undeployable. |
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

## Milestone 1 — "It runs, and it routes"  ✅ complete

A stranger can clone it, run it, and get an explained bin decision.

- [x] `recyclevision/` package: `Detector` protocol + `YoloDetector`, `RoutingPolicy`,
      `SortingPipeline`, dataclass domain models.
- [x] `policies/household.yaml` — 40 rules over 5 bins, with handling notes, certainty flags,
      and explicit non-waste ignores.
- [x] Model bootstrap: prefer `models/best_model.pt`; auto-download stock weights if absent;
      banner clearly stating which model is live and what it cannot do.
- [x] Bin-coloured annotation rendered in RGB via PIL — fixes the v0.2 BGR/RGB channel swap
      by not round-tripping through `result.plot()` at all, and colours each box by
      *destination* rather than by class. Label chips de-collide so crowded conveyor images
      stay readable.
- [x] Sidebar controls that actually do something (the v0.2 toggles were wired to nothing).
- [x] `app.py` rewritten: bin cards, diversion and contamination rates, per-item
      explanations, review queue, sample images so a visitor needs no photo of their own.
- [x] Headless CLI (`python -m recyclevision`) with text and JSON output.
- [x] 80-test suite driven through a stub detector — no weights, no network, no torch.
- [x] `requirements.txt` rewritten as UTF-8 with direct dependencies only; `requirements-dev.txt`
      split out; `packages.txt` for the Streamlit Cloud system libraries.
- [x] `README.txt` → `README.md`, with a screenshot.
- [x] GitHub Actions: ruff check, ruff format, pytest.
- [x] **Deployed to Streamlit Community Cloud**, from `main`:
      https://recyclevision-vfhgb8vencieb6ydhtzjcw.streamlit.app/

> A live URL in the README is worth more than any single feature on this list.

**Done when:** clean-machine clone runs, CI is green, README links a working hosted demo.

### What running it revealed

Verified end-to-end against the real sample images, which turned out to be the most
useful thing in this milestone:

- **Stock `yolov8n` is useless here** — zero detections at a sane threshold on the
  conveyor photo. Switched the stock default to `yolov8s`, which finds six items in
  ~70ms on CPU. `yolov8m` adds roughly one item for 2.3x the download.
- **COCO has no class for a drink can.** On a belt full of steel cans, the model reports
  "bowl", "cup" and "cutlery", and the policy dutifully routes them to landfill — wrongly.
  This is the single most concrete argument for Milestone 4, and the app now says so on
  every screen rather than reporting a confident wrong answer.
- Default confidence lowered from 0.25 to 0.15 on that evidence.
- **Grey "Landfill" boxes were invisible** against a dark conveyor belt, which read
  as boxes being drawn on nothing. Every stroke now carries a dark halo, and displaced
  label chips are tied back to their box with a leader line. An annotation that cannot
  be trusted visually cannot be QA'd at all.

### Measured baseline

`recyclevision.qa` was built to grade detections tile by tile, and the first graded run
(9 detections over the 2 sample images, `qa/verdicts.json`) gives:

| Metric | Household policy | MRF policy |
| --- | --- | --- |
| Detection precision | 75% | 75% |
| Routing accuracy (of real objects) | 33% | **83%** |
| End-to-end correct | 25% | **62%** |

Same model, same detections; only the policy file changed. Detection precision is
identical because the detector is untouched — the gain is entirely from context.

Remaining errors: two false positives on an empty belt seam (only a better detector
fixes those), and a drinking glass the MRF policy sends to containers. Nine detections
is a small sample and the MRF policy was written after seeing these images, so 83% is a
ceiling, not an expectation. **Detection precision — 75% — is the number Milestone 4 has
to beat, and no policy file can move it.**

## Milestone 2 — "It's quantified"  ✅ complete

- [x] **Diversion and contamination rates** as headline metrics, tracked across a session.
- [x] **Impact accounting.** Mass and CO₂e avoided, from `impact/factors.yaml`.
      The factors ship as **unverified placeholders** and the file's `verified: false`
      flag propagates to a visible warning on every derived figure. The mechanism is
      real; the constants are explicitly not. Replacing them with cited EPA WARM values
      is a data change, not a code change.
- [x] Per-detection table with bbox geometry; CSV and JSON export; batch mode over N
      images with aggregate totals and stream composition.
- [x] A second policy file (`mrf_conveyor.yaml`) to prove the abstraction holds, plus a
      policy picker in the UI. Done early: QA showed context, not code, was the biggest
      available accuracy win.
- [x] Policy schema validation with helpful errors, so a hand-edited YAML fails loudly.
- [ ] A real municipality's published rules as a third policy.

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

## Milestone 3.5 — "It's open-vocabulary"  ✅ complete

Unplanned, and it jumped the queue because QA said the detector was the bottleneck and
this fixes most of it without a single labelled image.

- [x] `vocab/waste_v1.yaml`: 19 waste-specific detection prompts, editable as config.
- [x] `OpenVocabularyDetector` (YOLOE) behind the existing `Detector` protocol — the
      pipeline, policies and UI needed no changes to accommodate it.
- [x] Offline embedding build (`scripts/build_vocab_embeddings.py`), cached to 40KB so the
      ~570MB text encoder is never needed at runtime. Verified: no clip/mobileclip module
      is imported when serving.
- [x] Open-vocabulary rules added to both policies. The MRF policy gains real
      metal/plastic/glass bins, which were impossible when every container looked
      like a "cup".
- [x] Detector picker in the UI, COCO kept as the baseline.
- [x] Recall added to the QA harness. It previously measured only precision, so a detector
      that found one easy object per image would have scored perfectly.
- [x] Annotations label the *item*, not the bin — nine chips reading "Mixed Recycling"
      carried no information — plus a colour legend so a saved image explains itself.

| Metric | COCO | Open vocab v1 | Open vocab v2 |
| --- | --- | --- | --- |
| Detection precision | 75% | 100% | **100%** |
| Class accuracy | — | 62% | **85%** |
| Routing accuracy | 33% | 92% | **100%** |

Class accuracy was added after the repo owner pointed out that the annotated
pictures were "badly mislabeled" — and they were. Routing accuracy forgives any
mislabel that lands in the right bin, so v1's 92% concealed a 38% mislabel rate.
Reporting the two separately is the only honest way to show it.

**These are fitted numbers, not predictions.** The v2 vocabulary was tuned against
these same two images. The first real measurement is the first image it has not seen.

Still zero-shot: it has never seen a labelled conveyor belt. Milestone 4 is unchanged,
but its baseline is now much higher and its argument is different — training has to beat
a good open-vocabulary model, not a bad closed-set one.

## Milestone 4 — "It's credible ML"

Separates "used a model" from "understands ML". Gated on real data.

- [ ] **Capture conveyor footage.** The blocking dependency for everything below.
- [ ] Use the open-vocabulary detector to *pre-label* that footage, then correct it by
      hand. Far cheaper than labelling from scratch, and the QA harness is already the
      correcting interface.
- [x] QA harness (`recyclevision.qa`): per-detection contact sheets, a hand-gradable
      verdict file, and scoring that separates detector failures from policy failures.
      Pulled forward from this milestone because it was needed to evaluate v0.3 at all.
- [ ] Dataset + data card: sourcing, label taxonomy, class balance, splits, known biases.
      Public bootstraps: TACO, TrashNet, ZeroWaste.
- [ ] Label taxonomy designed *backwards from the bins* — classes should be the distinctions
      that change a routing decision, not an arbitrary material ontology.
- [x] `train.py` with reproducible hyperparameters and logged runs.
- [x] Frame extraction with near-duplicate rejection, and a train/val split by frame block
      rather than at random — a random split of video frames leaks near-identical frames
      across both sides and makes validation meaningless.
- [x] Pre-labelling that emits **boxes and masks**, so no polygon is ever drawn by hand.
      Iterative by design: `--weights` pre-labels the next batch with the model trained on
      the last one.
- [x] `docs/LABELLING.md` — frame counts, honest time estimates, and what is worth
      correcting. **Estimated labelling time exceeds the 5-hour threshold for the naive
      approach, so it opens with a checklist to re-check auto-labelling tooling first.**
- [x] `docs/DATA_CARD.md` template.
- [x] **Routing-aware evaluation** (`scripts/evaluate.py`): mAP alongside class accuracy
      and routing accuracy, splitting confusions into those that changed a bin and those
      that did not. A model that trades the second for the first looks better on mAP and
      is worse in practice.
- [ ] Evaluation *page* in the app: per-class PR curves, confusion matrix, latency.
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
