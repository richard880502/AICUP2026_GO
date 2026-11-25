# Fixed coverage pilot protocol, 2026-11-09

Written before any pilot model training or evaluation. The source manifests
are already fixed: 500 original training players, 50 per rank, nested two/eight
games per player. No official test predictions are used.

## Matched arms

- A uses two selected games per optimizer player; B uses eight.
- One shared cap8 extraction supplies both arms; select A by manifest game
  keys, not a separate engine run. Preserve 96 decisions and four profiles.
- Within each rank, deterministic SHA256 ordering of anonymous player IDs
  reserves ten of the fifty players for inner validation. The remaining forty
  train both arms. Thus 400 optimizer / 100 inner players, rank balanced.
- Use all eight cached games of inner players in both arms. Fit channel
  mean/std only on A's optimizer rows, then apply identical normalization to B
  and inner/evaluation rows. This isolates coverage, not normalization choice.
- Original d96 RankTransformer: width128, heads4, layers3, FFN384, dropout0.1,
  cumulative-link ordinal head; no pretraining or architecture search.
- Seeds0/1/2; forty epochs, sixty-four updates per epoch, batch128. Every
  update selects players uniformly with replacement and then one of their
  available games uniformly. Use matched player draws across arms, separate
  per-arm within-player game draws. Equal optimizer updates, not equal epochs
  through datasets of different sizes.
- AdamW lr0.0005, weight decay0.05, cosine schedule, gradient norm clip1;
  token dropout0.15 and input noise0.05. Select minimum inner per-game NLL,
  earliest epoch wins ties. Save histories, selected state and preprocessing.

## Evaluation and advancement

Predict the original 8,000 validation games for each arm/seed. These 400 players
must remain neural-unseen. Reuse the original fixed whole-question generator
and official expected-utility decode. This cohort is historically explored:
report it as exploratory, not fresh independent confirmation.

Primary comparison: seed-ensemble whole-question official utility B minus A.
Aggregate repeated questions within player and report a paired-player 95%
bootstrap interval, per-rank and per-query-size metrics, and all seed deltas.
Advance to full frozen-other-branch C1+C2 only if delta >=0.005, interval lower
bound >0, at least two seeds positive, and no rank decline >0.05. Otherwise
stop this pilot; do not try progressively different subset seeds against the
same evaluation cohort. A positive pilot does not itself replace production.

## Extraction state

Started one unchanged runtime extractor on GPU1, exec session88242, using
`production/game_coverage_manifest_20261109/cap8/` and writing only to
`production/game_coverage_pilot_20261109/features/`. Eight shards of500 games.
The session ID is historical context, not proof of ongoing execution; poll it
or inspect processes before deciding whether a job is live. GPU0 untouched.
No detached automatic training or submission job has been scheduled.

## Training implementation checks

`coverage_train.py` now implements the six fixed runs. It refuses partial
extraction and an existing training output directory. Both arms use the same
cap2 optimizer-only channel moments, same 800 inner games, same player draws,
and the same 2,560 optimizer updates. Full histories and selected checkpoint
metadata retain optimizer/inner IDs and preprocessing parameters. Evaluation
prediction files retain game keys, players and labels for later paired scoring.

Checks completed before launch: pylint 10/10, mypy passed; existing two data
guard tests passed. A CPU smoke update with the original encoder produced
finite changed weights, and matched sampling was verified independent of cap.
This validates basic execution, not the entire six-run experiment or its score.

`coverage_evaluate.py` requires six complete aligned prediction arrays and
forty finite history entries per run. It reports the probability-ensemble
primary contrast, separate seed contrasts, rank/query-size breakdowns and
player-cluster bootstrap, and saves inner-NLL learning curves. A synthetic
perfect-identical-arm check returned utility1.0, exactly zero delta/interval
and a failed advancement gate, as expected. Pylint10/10 and mypy passed for
both training and scoring modules. No real pilot score exists yet.

Launch only after extraction completes and GPU1 is available:

```bash
PYTHONPATH=competition2_rank_prediction OMP_NUM_THREADS=4 \
  runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.dense_data_scale.coverage_train
```
