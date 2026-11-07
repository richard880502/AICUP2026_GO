# Diagnostic follow-up to issue #1, 2026-11-07

Goal: explain the existing negative pilot rather than search for a new model.
Repeat the exact four recipes on identical within-half folds, using seed
20261106 and 200 epochs. Save epoch 0/20/50/100/200 train and test NLL,
utility, exact rate, >=2-rank error, Brier, entropy and correction summaries.
Require final OOF probabilities to match the saved pilot within 1e-6.
Do not select epochs or alter expert hyperparameters based on these results.

OOF diagnostics cover all ranks, A/1D, 1D/2D, 4D/5D AUCs, upstream halves,
and 5-9/10-14/15-20 games. Training-fold summaries reuse each player four
times and are descriptive only; do not bootstrap them as independent players.
Top-label ECE uses 15 equal-width confidence bins and raw-probability argmax
correctness; utility and exact rate retain competition-utility decoding.

A single temperature scaler is a descriptive calibration control. Fit it
with training-fold CE only (200 Adam steps, lr 0.02, T constrained to [0.5, 2]);
do not deploy or select it on this already-studied cohort.

For redundancy, reconstruct 79 extra features from the 10 centered frozen
logits using fold-safe standardized ridge regression (lambda 10). Report
held-out R2 and residual-feature correlations with label minus expected rank.
This measures linear redundancy, not all information; maximum correlations
over many columns are exploratory and cannot establish an incremental signal.
No new OOF data, KataGo extraction, production inference or leaderboard tuning
is authorized by the diagnostic protocol. Larger OOF work needs a separate
cost/benefit assessment and a clearly defined hypothesis supported here.
