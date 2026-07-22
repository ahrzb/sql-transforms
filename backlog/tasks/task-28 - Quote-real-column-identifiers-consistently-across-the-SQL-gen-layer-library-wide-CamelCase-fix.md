---
id: TASK-28
title: >-
  Quote real-column identifiers consistently across the SQL-gen layer
  (library-wide CamelCase fix)
status: In Progress
assignee:
  - Wren
created_date: '2026-07-18 19:48'
updated_date: '2026-07-21 19:24'
labels:
  - bug
  - sql-surface
  - usability
milestone: m-1
dependencies: []
priority: high
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SEVERITY: high. Wren found (while investigating _compose.py:214) that CamelCase is broken LIBRARY-WIDE, not just the transformer-ref path TASK-25 fixed. Evidence (table with column 'Age'): SELECT Age AS x FROM __THIS__ -> FAILS 'No field named __this__.age' (plain passthrough!); SELECT Age / AVG(Age) OVER () -> FAILS; quoting ("Age") works; {a}(MixedCol) composition fails. So today the library REQUIRES users to double-quote every non-lowercase column -- bites nearly everywhere on real datasets (House Prices 80-col CamelCase). Unquoted rebuilds fold to lowercase in: rewrite_sql (__THIS__.col), build_state_tables (AVG(col) + GROUP BY / PARTITION BY keys), _compose.py:104/140. TASK-25 fixed only _named_struct. State-key columns (engine-internal, lowercased by state_key()) stay UNQUOTED -- the _compose.py:214 invariant, Wren verified (quoting THAT would create a latent mismatch; leave it + document with a comment). DESIGN DECISION embedded (needs AmirHossein's conscious yes): the fix makes authored identifiers case-sensitive-exact (match the Arrow schema, = what quoting does) vs. consistently folding everything to lowercase (case-insensitive). Wren + PM recommend case-sensitive-exact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 authored non-lowercase columns work WITHOUT user double-quoting across: plain passthrough, window-agg OVER, PARTITION BY, composition (frozen + unfit); transformer-ref already fixed (regression-guard)
- [ ] #2 state-key columns stay unquoted (214 invariant preserved + documented with a comment)
- [ ] #3 CamelCase test matrix green; transform == infer parity (decision-1)
- [ ] #4 DESIGN (unresolved): auto-quote/case-sensitive-exact vs match-DataFusion-literally -- AmirHossein + Wren to settle DIRECTLY before AC #1's direction is locked (not PM-ratified)
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 00:25
---
Handed to Wren (2026-07-19). The embedded design fork (auto-quote 'just works' vs match-DataFusion-literally) is NOT PM-ratified -- AmirHossein wants to decide it with Wren directly. Wren: ping AmirHossein to settle the direction before implementing AC #1. Rest of the ticket stands as scoped.
---

author: Iris (PM)
created: 2026-07-21 19:24
---
Board-accuracy correction, flagged by Wren. AC #3 unchecked and status reopened — the _compose.py:214 documenting comment does not exist on master. I verified rather than take the report at face value this time: 7f21dbe has no _compose.py in its stat, and the comment is not present at HEAD. This one is on me: I checked AC #3 because the merge was reported complete, which is not evidence that a specific AC was satisfied. Going forward I verify file-level ACs against the diff before ticking.\n\nWren's fix (fadd87e, branch task-28-state-key-comment) documents WHY state-key columns are rebuilt unquoted: state_key() generates every state value-column lowercase (f\"{fn.lower()}_{col.lower()}\"), so post-TASK-28 unquoted-identifier folding is a no-op on those names — quoting them would instead pin a generated name case-exact and desync if state_key's casing ever changed. Includes a 'do not fix this by quoting' note. TASK-28 returns to Done only once fadd87e lands.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
CORRECTION (2026-07-19): this task was marked Done prematurely. AC #3's documenting comment was NEVER written — verified: `git show --stat 7f21dbe` touches 6 files (_codegen/plan.py, _transformer_ref.py, src/expr_build.rs, src/plan.rs, test_diff_identifier_folding.py, test_transformer_ref.py) and does NOT include sql_transform/_compose.py; the comment is absent from _compose.py on HEAD. The invariant held only by accident of the file not being touched. PM error: AC #3 was ticked off a merge report without verification. Reopened to In Progress. Fix is ready but UNLANDED: commit fadd87e on branch task-28-state-key-comment (comment only, no behaviour change; 490 passed / 15 skipped, ruff clean). Blocked on landing because the main checkout is currently on master, so a ref-push to master is refused (receive.denyCurrentBranch).
<!-- SECTION:FINAL_SUMMARY:END -->
