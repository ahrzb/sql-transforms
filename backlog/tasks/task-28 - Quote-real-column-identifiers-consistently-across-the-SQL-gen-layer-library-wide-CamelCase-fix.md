---
id: TASK-28
title: >-
  Fold unquoted identifiers to lowercase like DataFusion (library-wide CamelCase
  handling)
status: In Progress
assignee:
  - Wren
created_date: '2026-07-18 19:48'
updated_date: '2026-07-22 17:35'
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
- [x] #1 Native + codegen FOLD unquoted identifiers to lowercase like the DataFusion oracle (columns, compound col/field parts, SELECT aliases); quoted "Age" stays case-exact everywhere. Root cause was the engines DIDN'T fold — they accepted SQL the oracle rejects. Users quote CamelCase columns themselves (intentional, documented papercut).
- [x] #2 TASK-25 force-quote reverted: transformer refs carry the user's original quoting — {t}(LotArea) folds-and-fails, {t}("LotArea") works. Old test_camelcase_columns_compose (asserted unquoted 'just works') flipped to quoted-works / unquoted-fails.
- [ ] #3 state-key columns stay unquoted (_compose.py:214 invariant preserved + documented with a comment).
- [x] #4 transform == infer parity (decision-1) + CamelCase differential matrix green (tests/test_diff_identifier_folding.py, 14/14).
- [x] #5 Ceiling (ponytail note, expr_build.rs): a real CamelCase table/struct-column qualifier is NOT folded — unreachable today (tables are always __THIS__/generated); flagged for when qualified tables become reachable.
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 00:25
---
Handed to Wren (2026-07-19). The embedded design fork (auto-quote 'just works' vs match-DataFusion-literally) is NOT PM-ratified -- AmirHossein wants to decide it with Wren directly. Wren: ping AmirHossein to settle the direction before implementing AC #1. Rest of the ticket stands as scoped.
---

author: Iris (PM)
created: 2026-07-19 14:27
---
Design fork (AC #4) RESOLVED directly by AmirHossein + Wren → match-DataFusion, not auto-quote. This inverts the earlier Wren+PM recommendation, so I re-scoped AC #1 (fold-like-oracle, users quote CamelCase) and retitled the ticket (was 'Quote real-column identifiers consistently' — now describes folding, since we do the opposite of force-quoting). Marked Done per Wren's merge 7f21dbe. Branch/worktree left up for review before Wren cleans up.
---

author: Iris (PM)
created: 2026-07-22 17:35
---
Reopened: AC #3 (state-key comment at _compose.py:214) was ticked off the merge report but the comment is NOT in the landed diff of 7f21dbe. Wren has the fix ready as fadd87e — Done again once it lands on master.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved as MATCH-DATAFUSION (not auto-quote). Direction settled directly between AmirHossein + Wren — the reverse of the original Wren+PM 'case-sensitive-exact so it just works' recommendation. The real bug was that native + codegen did NOT fold unquoted identifiers, so they accepted SQL the DataFusion oracle rejects; the fix makes both engines fold unquoted identifiers to lowercase like the oracle while quoted identifiers stay case-exact. Users quote CamelCase columns themselves — intentional papercut. Also reverted the TASK-25 force-quote so transformer refs carry the user's original quoting. Merged to master (7f21dbe). Verified: folding matrix 14/14, full Python suite 473 passed, cargo test clean. Branch task-28-identifier-folding + worktree under .claude/worktrees/ still up for diff review before cleanup.
<!-- SECTION:FINAL_SUMMARY:END -->
