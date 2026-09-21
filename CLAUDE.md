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

That is why **routing accuracy is the headline metric, not mAP**. Confusing
two classes that share a bin costs nothing; confusing two that do not
contaminates a batch.

## Where things are

| path | what |
| --- | --- |
| `recyclevision/` | the library — no Streamlit, no scripts, importable and tested |
| `app.py` | Streamlit UI. Presentation only; every decision is made in the library |
| `train.py` | fine-tuning entry point |
| `scripts/` | one job each: inspect, import, build_pool, build_eval_set, evaluate, zeroshot_eval, prelabel, extract_frames, diagnose_eval, fetch_samples, build_vocab_embeddings |
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

A model is training on the balanced SortWaste + ZeroWaste pool (3,000 train,
500 val, 40 epochs, ~8.5h). **The number to beat is 56.5% routing accuracy on
WaRP**, which was never trained on and is a genuine third-facility holdout.

One confound to state whenever that number is quoted: WaRP has 534 glass
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
