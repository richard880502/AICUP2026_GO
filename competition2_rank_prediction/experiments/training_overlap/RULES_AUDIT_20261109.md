# Official public Q&A audit and a distinct HD1 hypothesis

The official competition page links a public Word Q&A announced2026-10-29.
Downloaded read-only2026-11-09 from that exact linked document ID. The file
internally says updated2026-10-22. Local copy:
`production/rules_audit_20261109/official_qa.docx`, SHA256
`f3389c67d379b47e3dae294ee7ec3c2f60c2de075b50cd8f57e95660286952bf`.

Sources:

-[Official topic page](https://www.aidea-web.tw/topic/8c36af8c-2c70-497f-a862-6fb81a53a403)
-[Linked official Q&A](https://docs.google.com/document/d/1BQtAV7u_6dSDGSVxgzaXSglqAUMH33JJ/edit)

The read Q&A does not resolve train/test player overlap. Do not infer shared
identities or use unrelated candidate pools on its authority.

## New, specifically scoped finding

Q13 describes HD1 as a no-komi Black-first game, with ties going to White.
Our existing final-result extractor always uses6.5komi. The previous opt-in
handicap correction applies only to HD>=2 and explicitly leaves other games
unchanged. Thus the prior failed handicap-patch experiment did not test an
HD1-only final-result komi change.

**Caution:**Q13 itself retains a pending-confirmation label and an unresolved
alternative concerning the color column for HD>=2. This document is not a
clean final specification for every handicap detail. Treat HD1komi0.5(the
half-point representation of no komi/tie-to-White) as a controlled hypothesis,
not permission to silently rewrite production physics or colors.

The old cached validation CSVs have no HD headers. Matching their8,000keys
to the updated authorized training CSVs gives:

| HD | Games |
| --- | ---: |
| 0 | 6057 |
| 1 | 1542 |
| 2–9 | 401 |

All8,000keys resolved. HD1accounts for19.275% of validation games, materially
different from the4.7%-scale HD>=2issue. Use updated-source HD lookup by full
rank+game_id key; do not mistake absent cached HD headers for HD0.

No new predictions or submissions were changed by this audit, and no official
question/email/form was sent. The next action is a fixed matched re-extraction
test, not an inference-time production patch.
