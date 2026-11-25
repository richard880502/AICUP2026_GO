# Candidate-profile played-policy adaptation: feasible, unscored

2026-11-09. Four original training games, eight own decisions each. Enumerate
**all ten candidate profiles for every game**; never select a profile using
the unknown true rank. Synthetic labels0/1/2/3are used solely to test the
gradient path. Updated weights are discarded, not saved or evaluated.

This differs from the preceding hidden-feature adaptation: the classifier
inputs are the actual played moves' mean log probabilities under candidate
rank-conditioned policies, retaining the pretrained policy head. There is
no claim of new rank performance or of improved production information.

## Observed GPU1 costs

-32physical positions×10profiles=320conditioned positions/update.
-Score tensor shape4games×10candidate ranks.
-Three forward/backward updates:1.31731seconds total.
-Peak torch allocated memory5,612,493,824bytes, about5.23GiB; CUDA context
  is additional. This is a small-batch probe, not full-run throughput.
-Finite gradients reach blocks16/17and final trunk norm; a selected backbone
  parameter changes. Other parameters have requires_grad=False; model remains
  in eval mode to preserve normalization running statistics.

Inverse-rank metadata20/17/14/11/9/8/7/6/5/4represents
11k/8k/5k/2k/1d/2d/3d/4d/5d/6d. Mapping agrees with the pinned upstream
`cpp/neuralnet/sgfmetadata.h`:9d=1,1d=9,1k=10,etc. Both candidate player ranks
are set alike, matching single-rank HumanSL profiles, not true opponent ranks.
Other metadata retains the existing neutral settings.

## Limitations and next gate

The probe uses raw PyTorch policy normalization, not the engine's full legal
move mask. It preserves the legacy baseline board reconstruction. Policy
equivalence and exact position/move alignment must be checked before scored
training. Synthetic loss decrease is not evidence of rank improvement.

A controlled frozen-policy versus adapted-policy experiment must retain the
same head, profiles, sampling, game pooling and player exclusions. Candidate
metadata may be an inference-time hypothesis; it must never be chosen from
the target label. Cost is approximately ten profile forwards per position,
so cache/benchmark the frozen arm before scheduling a large run. No scored
training, official inference, submission change or upload has been started.

Source:probe.py. Pylint10/10 and mypy pass for the two new modules. Runtime
report:`production/policy_adaptation_probe_20261109/report.json`.
