# Player-identification v2 production

## Current submitted version

User-reported official Public: **0.5236602** (2026-11-11).
Use the frozen-encoder runner below to reproduce the current version. It
preserves original-pool predictions and updates only the new pools; all
1,600 questions are then validated and assembled. No neural training occurs.

```bash
PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
  runtime/competition1_research/.venv/bin/python -m \
  competition1_player_identification.production_pipeline.frozen \
  --input production/player_v2_20261111 \
  --output production/player_v2_frozen_20261111 \
  --data-dir data/aicup2026_player_identification_v2_tests_20261108_fix_readme \
  --device cuda:0
```

The current upload file is
`production/player_v2_frozen_20261111/submission_player.csv`.
See [results and limitations](FROZEN_RESULTS.md).

## Historical candidate-retrained version

The following recipe produced the inferior Public **0.4650701** submission.
Its final upload CSV has been deleted at the user's request. Retained feature
caches and original-pool models are required by the current frozen version.
Running the historical final-assembly command regenerates that inferior CSV;
use the frozen runner above for the current submission.

Authorized input: `data/aicup2026_player_identification_v2_tests_20261108_fix_readme/`.
Output root: `production/player_v2_20261111/` (large artifacts remain local).
The final upload file will be `submission_player.csv`; it is ready only after
`submission_validation.json` reports PASS and `completed.txt` exists.

All four pools are required. Original public/private pools each contain 400
players with 100 reference games; new public/private pools each contain 600
players with 10 reference games. Each pool has 400 questions. New queries use
1/2/3/5 games and may mix black and white. Actor color is passed to the
existing player-decision and ownership-perspective feature extractor.

The method retains the best research pipeline's 64-decision NBT/HumanSL rich
profiles, three-seed 512D/1024D ArcFace branches, centroid cosine retrieval,
and Top-20 group-softmax reranker. No query-answer labels are available or
used. Each official candidate pool fits its own encoders and reranker.

For original 100-game pools, encoder epochs are frozen to the research best
settings: 3/3/3 for 512D and 3/4/3 for 1024D, seeds 20260927–20260929.
Candidate-held-out validation uses 20 games per player. For ten-game pools,
five games per player are held out; checkpoint selection (up to 60 epochs)
uses only held-out candidate games and query sizes 1/2/3/5. Final encoders
refit on all reference games for the selected epoch count. Both branches
retain all three seeds. This is an adaptation to a new few-game setting;
the historical 0.923040 proxy score is not a score for these official pools.

Dimension blends are selected on candidate-only held-out games. Synthetic
reranker tasks use only the development encoder's held-out candidate games;
reference games are disjoint. Reranker query owners are split 80/20 by player,
four samples per query count. The existing 64/32-unit group-softmax MLP uses
candidate-only early stopping and blend selection. Test pair labels are
explicitly -1 and never scored as known answers.

Extraction is resumable at immutable 128-game shards. A four-PID `finish.sh`
waits for existing extraction jobs, checks exact source record IDs during
profile loading, trains and predicts each pool, then combines predictions
in sample order. Incomplete extraction raises an error; no default answers
are generated. Final validation requires exactly 1,600 unique official IDs,
five distinct in-pool players per question, valid prefixes and exact headers.

The original two large pools use two disjoint shard workers per pool: worker
0 on physical GPU 0 and worker 1 on physical GPU 1. New pools run on GPU 1.
This keeps both GPUs useful after the smaller pools finish. The vLLM service
on GPU 0 is retained. `finish.sh` accepts comma-separated extraction PIDs for
each pool, in pub2/priv2/pub/priv order, and waits for every listed worker.
Only the worker's assigned shard numbers are written, preventing concurrent
writers for one shard.

Example pool inference after extraction:

```bash
PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
  runtime/competition1_research/.venv/bin/python -m \
  competition1_player_identification.production_pipeline.run \
  --pool pub2 \
  --data-dir data/aicup2026_player_identification_v2_tests_20261108_fix_readme \
  --output production/player_v2_20261111 --device cuda:0 --workers 4
```

Verification before production: pylint and mypy passed for changed Python
modules; synthetic 1/2/3/5-game retrieval and group-softmax fitting passed;
the official 1,600-row sample passed the strict format validator, and a
cross-pool mutation was rejected. Sample predictions were used only as a
format fixture and are never copied into the final submission.
