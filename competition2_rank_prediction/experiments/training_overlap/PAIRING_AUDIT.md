# Opponent-label and cross-split repeated-game audit, 2026-11-09

## Question

If a source game appeared once for each actor, its second record could supply
the opponent's registered rank for auxiliary supervision. Such pairs might
also undermine player-only validation if the same board sequence appeared
across training and validation with different labeled actors.

## Read-only evidence

The authorized training index contains1,000,000 rows and **1,000,000 unique
move-sequence hashes**. Repeated-hash groups:0; opposite-color repeated-game
groups:0. Thus this index supplies no direct same-game opponent-label pairs.
Do not assign opponent rank from the labeled player's rank or fabricate a
dual-supervised training target.

Cached d96 game keys were independently aligned to the source index using
rank plus game_id, with exact counts:15,211training games and8,000validation
games. All keys resolved. Their move-hash set intersection is0. Therefore
no identical move sequence was found across these cached splits, even if
actor colors were ignored. This complements, but does not replace, player
exclusion checks.

Scope: the existing64-bit move-only hash ignores SGF root metadata. The audit
does not canonicalize board symmetries, identify games from shared prefixes,
or prove independence of all latent player/opponent relationships. No ranks
or predictions were inferred for official test questions. No other-task
candidate library was read, and no training or submission was changed.

## Research decision

Direct paired-opponent supervision is unavailable from duplicate source games,
so this proposed shortcut is not scheduled. GitHub Issue#1 was also re-read
through the connected repository tool: its latest main proposal remains
ordinal semi-hard mining, already covered by the local A/B/C experiments and
negative/uncertain results. No new untested issue proposal was found in that
issue. The remote issue was not modified.

This is a data-feasibility finding, not a new model result or a breakthrough.
No Python source changes were required for this read-only audit.
