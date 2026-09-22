# RecycleVision — orientation

Read this first. It is the current state of the project, the decisions that
are settled, and the traps that have already cost time. Everything here is
measured unless it says otherwise.

## What this is

A waste-sorting assistant: point it at an image of waste and it says **which
bin each item goes in**. Streamlit front end, YOLO detection, and a routing
layer that is deliberately separate from the detector.

**The thesis: material does not determine destination.** A wine glass and a
jam jar are both glass and go to different places. A lined paper cup and a
sheet of paper are both paper and go to different places. So the model
detects *items*, and a policy file maps items to bins per facility.

That is why **routing is the headline metric, not mAP**. Confusing two
classes that share a bin costs nothing; confusing two that do not contaminates
a batch.

**But use the end-to-end figure, not the conditional one.** Routing accuracy
over *matched* instances rises when a detector finds only easy objects and
falls when it starts finding hard ones, so it is not comparable across models
at different recall. `scripts/evaluate.py` reports `routed correctly` over all
labelled instances beside it. Trap 9 below is what happens when you forget.

## Where things are

| path | what |
| --- | --- |
| `recyclevision/` | the library — no Streamlit, no scripts, importable and tested |
| `app.py` | Streamlit UI. Presentation only; every decision is made in the library |
| `train.py` | fine-tuning entry point |
| `scripts/` | one job each: inspect, import, build_pool, build_eval_set, evaluate, zeroshot_eval, prelabel, extract_frames, diagnose_eval, fetch_samples, build_vocab_embeddings, check_labels |
| `policies/*.yaml` | item → bin, per facility. `household` and `mrf_conveyor` |
| `vocab/*.yaml` + `.pt` | open-vocabulary prompts and their precomputed embeddings |
| `mappings/*.yaml` | outside dataset classes → this project's vocabulary |
| `impact/factors.yaml` | mass and CO2e per class |
| `datasets/`, `runs/`, `models/*.pt` | gitignored; local only |

**Behaviour lives in YAML wherever it is facility knowledge**, not in code.
Policies, vocabulary, mappings and impact factors are all data.

## What is measured

Dates matter here; these are real runs, not estimates.

**Trained on WaRP alone** (63 epochs, best at 38, 18.6h on an M5 `mps`):

| tested on | mAP50 | routing acc | recall |
| --- | --- | --- | --- |
| WaRP val (same plant) | 0.671 | 96.1% | 56.7% |
| ZeroWaste (unseen plant) | **0.033** | **56.5%** | 6.3% |

**A 20× collapse. The model learned one conveyor belt, not waste.** That
single twenty-minute evaluation replaced a two-day training run premised on
the opposite, and it is the "before" number everything later is compared to.

**Zero-shot open-vocabulary on WaRP**: 52.6% recall at 56% fair precision,
but only at confidence 0.01 — 0.9% at 0.25. The objects were always found;
they were ranked terribly. Training fixed the ranking, not the eyes.

## The data

| dataset | images | instances | per image | status |
| --- | --- | --- | --- | --- |
| WaRP | 2,974 | 10,374 | 3.5 | **retired from training** — see below |
| SortWaste | 5,261 | 87,525 | 16.6 | training |
| ZeroWaste-f | 4,503 | 26,766 | 5.9 | training. CC BY-**NC** |

**WaRP is retired from training.** 3.5 labelled objects per image in frames
holding dozens means its unlabelled objects are *background supervision* —
teaching the model that bottles on belts are nothing. It stays as evaluation
data, where sparse annotation costs far less, and it is the only source of
glass.

**ZeroWaste is CC BY-NC.** Fine for research and a portfolio. Not fine for a
commercial product, and a model trained on it is at best a grey area. This is
a live decision, not a footnote.

**10 of 17 vocabulary classes have no training data at all**: caps, cups,
wrappers, jars, tableware, paper, newspaper, organics. That is the ceiling on
what the model can do, and no amount of relabelling existing data changes it.

**The sources disagree with each other**, and it does not matter: ZeroWaste
files cartons under `cardboard` while the others separate them, and folds
tubs into rigid plastic. Both pairs share a bin in both policies, so the cost
lands on class accuracy and mAP and not on routing. Expect that gap; it is
the thesis working.

## Traps that have already cost time

Each of these shipped, broke something, and is now guarded. Do not undo the
guards.

1. **Python 3.9 is the floor.** `zip(strict=True)` and `Path.hardlink_to` are
   both 3.10+ and both shipped. CI now runs a 3.9 + 3.11 matrix, and
   `tests/test_python_compat.py` parses sources for newer APIs.
2. **Ultralytics' label cache is keyed on total byte size and paths, never
   contents.** Change a class index from 14 to 15 and the cache silently
   wins. `prepare_tree` clears caches; anything writing labels must.
3. **NMS on MPS blows ultralytics' watchdog**, which `break`s out of a
   *per-image* loop and scores the rest of the batch as "predicted nothing".
   `train.py` raises the budget on mps/cpu only.
4. **Routing accuracy can be vacuous.** All five WaRP classes route to one
   bin under `household`, so it reads 100% before the model does anything.
   `evaluate.py` now says when that is the case. Score against
   `mrf_conveyor` for these datasets.
5. **Dataset descriptors carry other people's absolute paths.** SortWaste's
   points at `/home/socialab/...`. Resolution falls back to matching the
   path's tail.
6. **Filenames collide** across sources and across splits. Output names carry
   both a source prefix and the source split.
7. **COCO `file_name` can resolve to a segmentation mask** — ZeroWaste keeps
   frames in `data/` and masks in `sem_seg/` under identical names.
8. **An import that writes nothing, or only some splits, is a failure.**
   Both are refused now. A half-imported dataset trains fine and teaches
   the wrong thing.
9. **Routing accuracy conditioned on matched instances is not comparable
   across models.** The pool model scored 42.0% against the WaRP model's
   56.5% and looked like a regression; it had matched 38.5% of labelled
   objects against 6.3%, so end-to-end it routed 16.2% of the stream
   correctly against ~3.6% — about 4.5× better, reported as worse. Quote
   `routed correctly`.
10. **A zero-area box makes the loss NaN, and the run never stops.** The
   detector's loss divides by box area. `box_line` clamped coordinates but
   never rejected a degenerate box, while `polygon_line` three lines below
   always rejected a degenerate polygon. One such label NaNs its batch;
   ultralytics restores `last.pt`, re-runs, meets it again — and its
   "attempt 1/3" counter **resets on every successful recovery**, so the run
   alternates NaN and restore indefinitely, reporting identical metrics for
   hours. This was misread as fp16 overflow (`114870e`) and then as a
   plateau. Two runs, roughly 20 hours. `scripts/check_labels.py` finds them
   in seconds; `box_line` now drops them; `StallDetector` halts a run that
   repeats a score three times.
11. **A dead run looks exactly like a converged one.** Same `best.pt`, same
   `results.csv`, same timestamped directory. The pool run died of fp16
   overflow at epoch 22 of 40 and was scored a day later as the finished
   article; the "plateau" reasoned about afterwards was an artefact of the
   corpse. `recyclevision/runs.py` reads the run's own logs and
   `scripts/evaluate.py` warns before printing any number. Two signatures:
   NaN in any row, and validation metrics repeating byte-for-byte while
   training losses keep moving.
12. **A class the holdout cannot contain is a pure false-positive source.**
   The pool model predicted `plastic bag` for `plastic bottle` 200 times — a
   third of every matched instance — on WaRP, which contains no film at all.
   Same mechanism as the open-vocabulary phantom prompts, now in a trained
   model. `scripts/evaluate.py --classes present` scopes to whatever the
   holdout actually labels.

## Working conventions

- **Branch `claude/recycling-app-improvements-subgat`**, then merge to `main`
  and push both. Do not push to `main` alone.
- Before every commit: `ruff check . && ruff format --check . && pytest`.
- **Every behaviour change gets a test**, and the test is verified by
  reintroducing the bug and watching it fail. A test that has never failed
  has not been tested.
- Comments explain *why*, especially where something non-obvious was
  measured. Do not narrate what the code already says.
- **Never write shell placeholders** like `<run>` or `<...>` in instructions
  — a shell reads `<` as a redirect. This happened five times. Where a path
  is awkward, make the tool resolve it: `evaluate.py --weights` is optional
  for exactly this reason.

## The workflow

```bash
python scripts/inspect_dataset.py <a download>       # format, classes, counts
# write mappings/<name>.yaml translating its classes
python scripts/import_dataset.py <descriptor or COCO .json> \
    --mapping mappings/<name>.yaml --out datasets/<name> [--split train|val]
python scripts/build_pool.py --from datasets/a --from datasets/b \
    --out datasets/pool --images 3000 --val-images 500
python train.py --data datasets/pool/data.yaml --epochs 40 --cache disk
python scripts/evaluate.py --data datasets/warp/data.yaml --policy policies/mrf_conveyor.yaml
```

Validation is paid every epoch and dominates: 1,499 images took 17m43s, so an
uncapped val split can cost more hours than the training it measures.
Accuracy scales with the *log* of dataset size; validation cost scales
linearly.

## In flight

**Two training runs have died of NaN. Neither was fp16.**

`conveyor_20260921_023058` (AMP on) and `conveyor_20260921_161713` (AMP off,
after the `114870e` fix) both NaN'd and both looped: the second alternated
NaN and recovery from epoch 9 to at least 18, reporting
`0.555 0.514 0.53 0.403` six times in a row. Ultralytics' retry counter resets
on each successful recovery, so neither run would ever have stopped.

**The cause is in the labels, not the device.** `box_line` wrote boxes with
zero width or height; the loss divides by box area. Fixed at the source, and
`scripts/check_labels.py` reports how many exist in an already-imported
dataset without needing torch or a GPU.

So the second run's clean epochs are still worth something: **mAP50 0.53,
mAP50-95 0.403 on the pool's own val split** at epoch 5–8, which is the best
this project has measured. Its `best.pt` is real. Score it on WaRP before
re-importing anything.

Order of work ([ROADMAP.md](ROADMAP.md) Phase 1, four *completed* runs):

0. `python scripts/check_labels.py datasets/pool/data.yaml` — seconds, and it
   decides everything below.
1. Re-import whichever sources carry the bad labels, rebuild the pool, re-run.
   Consider Colab: `docs/TRAINING_ON_GPU.md`, `notebooks/train_colab.ipynb`.
   A T4 has working AMP and a fast NMS kernel, and mps has now produced zero
   completed runs in two attempts at ~10 hours each.
2. `evaluate.py --classes present` — free, no retraining. `plastic bag` for
   `plastic bottle` ×200 on a holdout with no film in it.
3. ZeroWaste's 5.9 objects/image against SortWaste's 16.6 — the WaRP density
   trap, possibly recurring at a third of the pool.
4. Data volume: 3,000 train images of ~8,700.

The taxonomy merge is approved but **demoted**: it predicted a floor under
`train/cls_loss`, which instead fell smoothly 1.90 → 0.54.

Settled 2026-09-21: **ZeroWaste stays** and v1.0 weights ship non-commercial;
the **training taxonomy merges** where sources disagree.

One confound to state whenever a WaRP number is quoted: WaRP has 534 glass
instances and neither training source has any glass at all, so ~5% of its
instances are unreachable by construction.

## What is not done

- A hand-labelled evaluation set. `scripts/build_eval_set.py` reserves and
  leak-guards it; nobody has labelled it. Until then no precision figure is
  fully trustworthy, because every source's annotations are incomplete in
  its own way.
- Video, tracking and a count line (Milestone 3).
- A FastAPI service, Docker, ONNX (Milestone 5).
- Own facility footage, which is the only route to the missing 10 classes.
