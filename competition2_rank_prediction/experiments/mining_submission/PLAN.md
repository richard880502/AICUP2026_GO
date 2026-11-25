# Experimental submission delivery, 2026-11-08

User objective: produce a different submission file ready for upload today.
Keep the original official submission and its known Public score intact.

Primary candidate: add the completed direct-mining three-seed own-player
probability ensemble as a fourth geometric-fusion member to C1+C2. Keep
C1 E1/E2/E3 inputs on the original own/opponent members, changing only
the fusion input. Refit weights and C1 on the original validation cohort.
Do not use official answers/Public feedback or force rank balance.

Before inference check that every original validation player is absent
from mining training; align games by keys, require complete usable test
features and preserve the official 400+400 question order. Reproduce the
original baseline from its frozen per-game predictions and same validation
fit, requiring exact predictions before producing the alternate.

Report five-fold player-stratified validation (seed20261108). Within each
fold fit per-game weights and C1 only on training validation players;
evaluate20 fixed5–20-game queries per held-out player, with shared queries
for baseline/candidate. Average repeated queries within player before
5,000 player-bootstrap intervals. This is exploratory validation already
used for model development, not untouched confirmation.

If the mining extension has nonzero contribution, produces different
decisions and does not drop validation utility by more than0.005, deliver
it as an explicitly experimental alternate. If it fails that screen,
investigate the previously evaluated joint two-sided Transformer primary
candidate rather than deliberately shipping an obviously inferior change.
This is a delivery screen, not a claim that the selected model improves
official Public or Private scores.

Required artifacts: new800-row UTF-8 CSV, strict official-format validation,
baseline diff, SHA-256, checkpoint/data provenance, per-question method
report and reproducible commands. All predictions/checkpoints remain in
ignored production/runtime directories. Record method and limitations in
tracked Markdown. No automatic upload or replacement of the stable model.
