# Data card — <dataset name>

Fill this in while the details are fresh. A model's failures are usually
predictable from its dataset, but only if somebody wrote the dataset down.

## Source

- **Footage:** <what was filmed, where, when>
- **Facility / line:** <single stream? pre-sort? post-screen?>
- **Camera:** <model, resolution, frame rate, mounting height and angle>
- **Belt speed:** <m/s, or "static">
- **Lighting:** <overhead fluorescent? daylight? mixed? did it change?>
- **Sessions:** <how many separate recordings, over what period>

## Extraction

- **Frames extracted:** <n> from <m> total
- **Sampling:** every <N>th frame, near-duplicates below <diff> dropped
- **Split:** <train>/<val>, by frame block

## Labels

- **Classes:** <which vocabulary, and whether it was modified>
- **Task:** segmentation / detection
- **Pre-labelled by:** <stock open vocabulary | round-N model>
- **Corrected by:** <who>
- **Rounds:** <how many label-train-relabel cycles>
- **Approximate hours spent:** <n>

### Instances per class

| Class | Instances | Notes |
| --- | --- | --- |
| | | |

Flag anything under ~100 instances: it will not learn reliably.

## Known biases and gaps

Be specific. This section is the one that predicts failures.

- **Not represented:** <e.g. night shift, wet material, crushed cans, bags>
- **Over-represented:** <e.g. 70% of instances are steel cans>
- **Systematic oddities:** <e.g. belt seam produces false positives; one camera
  angle only; all footage from a single day>
- **Ambiguous by construction:** <classes a human grader could not separate
  either — record these, they cap achievable accuracy>

## Evaluation

- **Held-out set:** <is it genuinely independent? different day/shift?>
- **Class accuracy:** <n>%
- **Routing accuracy:** <n>%
- **Confusions that changed a bin:** <list them>

Routing accuracy is the number that matters. Class accuracy below it is
expected and fine — see `docs/LABELLING.md`.
