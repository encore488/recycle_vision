# RecycleVision — the road to v1.0

This is a plan, not a log. The measurements live in
[CLAUDE.md](CLAUDE.md) and [docs/history/](docs/history/); what is here is
**what we are building, in what order, and what we are deliberately not
building.**

**Status:** v0.8. Library, CLI and Streamlit app shipped and deployed;
477 tests; two detectors and two policies measured against held-out
sorting-plant data.

---

## What v1.0 means

> **Point it at a video of a waste stream and get per-bin counts you can act
> on, with the system stating exactly how far to trust them.**

Read what that does *not* say. **It names no accuracy target.** Accuracy is
reported, never gated on. That is deliberate, and it is the single most
important line in this document.

### Why the model cannot be the gate

The architecture's whole claim is that the detector is a swappable component
behind a `Detector` protocol. An earlier version of this roadmap then made
shipping contingent on the detector being good — which quietly abandoned the
thesis and made v1.0 hostage to data we may never obtain.

10 of 17 vocabulary classes have **no training data at all** and no public
dataset supplies them. If v1.0 means "the model works", v1.0 never ships.

So: **the system ships at v1.0; the model is versioned separately.** A better
model is a release asset, not a release blocker. Phases 2, 3 and 4 below are
not data-gated and do not wait for Phase 1.

---

## The metric that nearly reversed a decision

Recorded here because it is the most expensive mistake available in this
project, and it has already been made once.

**Routing accuracy is computed over *matched* instances**, so it rises when a
detector finds only the easy objects and falls when it starts finding hard
ones. Two models were compared across a 6× difference in recall:

| | WaRP-trained → ZeroWaste | pool-trained → WaRP |
| --- | --- | --- |
| matched / labelled | 6.3% | 597 / 1551 = **38.5%** |
| routing accuracy (of matched) | **56.5%** | 42.0% |
| **routed correctly, of all labelled** | **~3.6%** | **16.2%** |

The headline metric said the second model was worse. End to end it is roughly
**4.5× better** — it finds six times as many objects, including every hard one
the first model never matched at all. Scoring the easy 6% will always flatter
a model against scoring a hard 38%.

`scripts/evaluate.py` now reports **`routed correctly`** over all labelled
instances alongside the conditional figure, and labels which denominator each
uses. **Compare models on the end-to-end number.** The conditional one answers
a different and still-real question — *when it finds something, does it route
it right* — and is only comparable at equal recall.

The two holdouts differ, so this is not a clean A/B. The metric artefact is
real regardless.

---

## Standing decisions

Settled. Re-open only with a reason, and write the reason down.

| Decision | Choice | Why |
| --- | --- | --- |
| Output | A bin, not a material | Material does not determine destination. The thesis. |
| Headline metric | **End-to-end routing rate** | Conditional routing accuracy is not comparable across models. See above. |
| Routing rules | YAML, never code | A new facility is a new file, not a deploy. |
| Vocabulary | Facility-scoped configuration | A class the stream cannot hold is a pure false-positive source. |
| **Training taxonomy** | **Merged where sources disagree** | *Approved 2026-09-21.* Costs zero routing accuracy; removes contradictory supervision. |
| **ZeroWaste (CC BY-NC)** | **Keep it; v1.0 weights ship non-commercial** | *Approved 2026-09-21.* Dropping it costs a third of the data for a commercial option v1.0 does not need. State the licence on the release. |
| Model later | Fine-tuned on own facility footage | The real differentiator, and explicitly **not** a v1.0 blocker. |

---

## Phase 0 — Truth pass · ~½ day · not data-gated

Every later decision is made against these numbers, and the documents
currently disagree with each other.

- [x] End-to-end routing rate in `RoutingScore`, reported beside recall.
- [x] `--classes` on `scripts/evaluate.py`, with `present` deriving the list
      from the holdout's own labels.
- [ ] `README.md` says "Status: v0.3"; fix it.
- [ ] `README.md`'s headline detector table quotes the two-image fitted
      numbers (75% / 33% / 100% / 92%) that this project marks *"superseded —
      do not cite as evidence."* They are cited, above the fold, in the
      most-read document.
- [ ] `scripts/evaluate.py` reports raw precision only. On a holdout labelling
      3.5 objects per image that number is uninterpretable, and
      `zeroshot_eval.py` already computes fair bounds. Reuse them.

**Gate:** no number appears in any document without its provenance and its
holdout.

## Phase 1 — Unblock the model · time-boxed · data-gated

### The stopping rule

> Stop when **either** the end-to-end routing rate on an unseen facility
> clearly beats the current 16.2%, **or** four training runs have been spent.
> Then ship what exists, with the number stated plainly.

The second outcome is a finding, not a failure — *"public data does not reach
item-level routing across facilities, and here is the measurement"* — and it
redirects scope instead of blocking it. The previous session had no exit
condition, so a plateau read as failure and the work drifted.

### Ordered by cost, cheapest first

1. **Scope predictions to what the holdout can contain.** No retraining.
   `plastic bag` predicted for `plastic bottle` **200 times** — a third of
   every matched instance — on a holdout containing no film whatsoever. The
   same mechanism measured on prompts earlier took precision 5.1% → 50%
   (`docs/history/2026-09-data-plan.md`). Class-balanced sampling deliberately
   up-weights film; WaRP is ~80% bottles and 0% film. We trained a prior that
   is wrong for the test distribution.

   ```
   python scripts/evaluate.py --data datasets/warp/data.yaml \
       --policy policies/mrf_conveyor.yaml --classes present
   ```

2. **Merge the classes the sources contradict each other on.** Approved.
   `beverage carton`+`cardboard box` and `plastic bottle`+`plastic tub` each
   share a bin in *both* shipped policies, so merging costs no routing
   accuracy and removes ~10k instances of irreconcilable supervision. Confirmed
   small at evaluation time (56 instances, all bin-neutral) but it puts a floor
   under classification loss during training. Check `train/cls_loss` against
   `train/box_loss` before paying for it.

3. **Density mismatch — the WaRP trap recurring.** WaRP was retired at 3.5
   labelled objects/image because unlabelled objects are background
   supervision. ZeroWaste is 5.9 against SortWaste's 16.6, in comparably
   cluttered frames, and is a third of the pool. If that logic was right for
   WaRP it is a real effect here. Test: SortWaste-only at the same budget.

4. **Data volume.** 3,000 train images of ~8,700 available, and accuracy
   scales with the log of dataset size. Cheapest to test, least likely to be
   the story.

**Gate:** the stopping rule fires. Publish weights as a GitHub Release asset,
`weights.py` prefers them over stock, licence stated.

## Phase 2 — The conveyor demo · ~2–3 days · not data-gated

The old Milestone 3, moved **off the critical path**: it depends on nothing
Phase 1 produces and can run in parallel. Verified greenfield — no `track`,
`video` or ByteTrack anywhere in the codebase.

- [ ] Video upload, `model.track(persist=True)` with ByteTrack.
- [ ] A virtual count line; each item counted exactly once as it crosses.
- [ ] Per-bin running tallies, throughput, FPS, latency distribution.
- [ ] Webcam / live stream input.
- [ ] **Record the demo video.**

Budget 2–3 days, not the "roughly a day" the old roadmap claimed: Streamlit
video decode/re-encode is fiddly, and this needs tests like everything else.

**Gate:** 60 seconds of material crossing a count line with live per-bin
tallies, good enough to put in front of someone.

## Phase 3 — Make it a system · ~2 days · not data-gated

- [ ] FastAPI `/sort` returning `SortResult` as JSON, with a `curl` example.
- [ ] Dockerfile and compose.
- [ ] ONNX export and a CPU latency table.
- [ ] Typed settings rather than module constants.

**Gate:** `curl` a photo at a container and get routed JSON back.

## Phase 4 — Ship it · ~1 day

- [ ] Weights published; `weights.py` prefers the release over stock.
- [ ] `docs/DATA_CARD.md` completed for every shipped source.
- [ ] `impact/factors.yaml` replaced with cited EPA WARM values, `verified:
      true` flipped — a data change, not a code change.
- [ ] Demo video in the README.
- [ ] A real municipality's published rules as a third policy.
- [ ] Tag `v1.0`.

---

## Not in v1.0

This list carries as much weight as the phases — unbounded scope is where the
drift comes from. **v1.0 waits for none of it.**

| Deferred | Why it can wait |
| --- | --- |
| Own conveyor footage | Unbounded on timing, and the one truly unbounded item. Non-blocking by design. |
| Hand-labelled evaluation set | The harness already reports raw *and* fair precision bounds. An honest interval beats a point estimate that costs weeks. |
| Active-learning loop | Great interview story, zero v1.0 value. |
| Hybrid VLM fallback | The `certainty: low` hook already exists and can stay unused. |
| Resin-code OCR | Real accuracy, narrow. |
| Instance segmentation, metric sizing | Needed for grasping, not for counting. |
| Robot / MQTT / ROS 2 stub | Phase 3's API is the seam it would hang off. Build the seam first. |
| The missing 10 vocabulary classes | No public data exists. Own footage or nothing. |

---

## Working conventions

- Work on a feature branch, then merge to `main` and push both. Never push
  `main` alone. (The branch in flight is named in CLAUDE.md.)
- Before every commit: `ruff check . && ruff format --check . && pytest`.
- **Every behaviour change gets a test, and the test is verified by
  reintroducing the bug and watching it fail.** A test that has never failed
  has not been tested.
- Domain logic in `recyclevision/`; `app.py` is presentation only; facility
  knowledge in YAML.
- The UI never claims more certainty than it has.
- Comments explain *why*, especially where something non-obvious was measured.
