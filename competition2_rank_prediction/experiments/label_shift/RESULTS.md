# Unlabeled uniform-prior adaptation: branch-specific result rejected

This file records an earlier diagnostic on the borrowed-opponent branch. It
is superseded for production conclusions by the current-best pipeline result
in [BEST_PIPELINE_RESULTS.md](BEST_PIPELINE_RESULTS.md). The historical
2,000-player holdout3 cohort has been used in prior research and is not a
pristine benchmark or a Public estimate.

| Method | Official utility |
| --- | ---: |
| Frozen NLL-C1 predictions | 0.70611025 |
| Per-half unlabeled uniform-marginal adaptation | 0.70921903 |

The borrowed-opponent diagnostic's observed gain was `+0.00310878`; its
paired-player95% interval was
`[-0.00118394, +0.00755696]`. Both cross-fit halves improve by about`+0.0031`.
The adaptation changes54of2,000 decoded decisions. It fails the fixed gate
because the interval included zero. On the current best three-member pipeline,
the same method instead declines by`-0.00208030` and both halves worsen.

Offsets are fitted independently for each test half from that half's
probability matrix only. The known label prior is set to uniform; no labels or
ranks are read by the fitting function. The mean probability vectors before
adaptation were already close to uniform; after adaptation they are uniform
to numerical tolerance. Labels were inspected only after fitting for audit,
and happen to be exactly100per class in each half.

The official task page describes the10 ranks and scoring rule but does not
state that hidden test questions are rank-balanced. Therefore equal class
prior remains an unverified assumption for the actual Public/Private test
sets. The old-branch result is not transferable and does not support changing
the submission. Do not tune priors or offsets on leaderboard feedback. The
current-best result rejects this adaptation for production.

Verification: pylint`10.00/10`; scoped mypy passes all3 Python files; one
unit test passes. Artifacts are under `production/label_shift_20261109/`.
The original production CSV remains unchanged.
