# Trained weights

Put a model produced by `train.py` here as `best_model.pt`:

```bash
cp runs/detect/<run>/weights/best.pt models/best_model.pt
```

The app then offers it as a **Trained** detector alongside the open-vocabulary
and stock-COCO options, and `scripts/evaluate.py --weights models/best_model.pt`
works without a run-directory path.

Weights themselves are gitignored — this file exists so the directory does,
because `cp` cannot create a destination directory and the failure reads as
"No such file or directory" pointing at the weights rather than the folder.
