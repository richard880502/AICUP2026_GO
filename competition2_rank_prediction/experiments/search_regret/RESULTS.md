# Search regret: fixed screen failed

Completed 2026-11-09. This is exploratory calibration CV on the original
400 validation players, not Public and not a fresh confirmation cohort.

| Frozen pipeline plus calibration | Official utility |
| --- | ---: |
| Original C1+C2 | 0.7106057000 |
| Regularization-only control | 0.7078272178 |
| Selected-position raw features | 0.7086562329 |
| Raw plus searched features | 0.7069638862 |

Search minus original: -0.00364181, paired-player 97.5% bootstrap interval
[-0.01152863, +0.00391943]. Search minus matched raw control: -0.00169235,
interval [-0.00389181, +0.00044405]. Neither comparison supports adoption.
No submission was created; original production predictions remain unchanged.

## What was measured

All 8,000 validation games completed, with four selected own and four opponent
decisions per game: 64,000 decisions, each with pre/post analysis. Normal KataGo
used 32 visits, not 32 plies. Selection used cached policy surprisal without
rank labels. Extraction took approximately 16 minutes 13 seconds on GPU1.

The 26 aggregated features comprise 14 raw-control and 12 search features.
Finite-budget before-best-child versus after-root value changes are estimates,
not exact move regret. Roots all reached 32 visits; selected best children had
3–31 visits. No missing game features were observed.

Five-fold rank/player-stratified calibration used the original question tasks,
fitting-player-only preprocessing, frozen original branches and C1 corrections.
Two planned comparisons use 97.5% player-cluster bootstrap intervals. Repeated
queries are averaged within player before resampling.

## Verification and limitations

Independent checks recomputed all four signed losses from cached raw engine
outputs, all 26 game features, official utility and both bootstrap intervals.
Input/cache/model/config hashes were checked. The original score was replayed.
This retains the baseline's legacy handicap reconstruction and fixed komi;
it is not a claim of correct reconstruction for every handicap game.

Artifacts: `production/search_regret_20261109/`, including `report.json`,
`evaluation.npz`, eight feature shards and `manifest.json`.
After completion, source formatting was cleaned up without algorithm changes.
Exact run sources are preserved in that directory's `source_snapshot/` and
match the manifest's original hashes. Do not rewrite those hashes; resuming
with reformatted sources is intentionally rejected by the provenance guard.

Decision: stop this fixed recipe, rather than increase visits without a new
testable explanation. Negative evidence concerns this sampling/aggregation
recipe, not all possible search-based models.
