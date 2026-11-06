# Recent rank-prediction experiments

These results are internal, player-disjoint evaluations, not Public scores.
The full original experiment source, preregistration and artifacts remain in
the local ignored `runtime/competition2_research/experiments/` workspace.

| Experiment | Cohort and result | Decision |
| --- | --- | --- |
| Joint player/opponent Transformer | 2,000-player cross-fit: 0.7170 to 0.7218, +0.0048; 95% CI [-0.0006, 0.0100] | Promising but does not meet the preregistered threshold. |
| Handicap information | 2,000-player cross-fit: primary shift -0.0016, CI [-0.0091, 0.0060] vs bias-only | Do not adopt. |
| Correcting handicap extraction at inference | 2,000-player cross-fit: 0.7170 to 0.6969 with validation also corrected; CI for difference [-0.031, -0.009] | Do not ship without retraining; existing models were trained on the original feature representation. |
| KataGo extraction averaging | 2,000-player cross-fit: +0.0004; CI [-0.0050, 0.0056] | No demonstrated gain. |
| Paired-rank HumanSL | 600 A/1D/2D questions: no improvement with confidence interval excluding zero | Rejected under preregistered stop rule. |
| Player identity representation | 2,000-player cross-fit: best residual -0.0001; CI [-0.0036, 0.0036] | Useful for identity but no demonstrated rank gain over C1+C2. |
| SGF structure families | 2,000-player cross-fit: pooled +0.001; CI [-0.010, 0.012] | No demonstrated gain. |

The fixed-player 5/8/12/16/20-game evaluation reported 0.647/0.678/0.705/
0.726/0.721; its extrapolated limit of 0.751 is uncertain and depends on the
model. It is not a proven upper bound on all possible SGF-based methods.
The joint two-sided model remains a possible follow-up on an independent
cohort; avoid repeated model selection on the official Public leaderboard.
