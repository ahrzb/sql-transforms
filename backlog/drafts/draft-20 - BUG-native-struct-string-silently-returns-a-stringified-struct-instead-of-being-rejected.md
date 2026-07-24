---
id: DRAFT-20
title: >-
  BUG native: struct || string silently returns a stringified struct instead of
  being rejected
status: Draft
assignee: []
created_date: '2026-07-24 02:19'
labels:
  - native
  - parity
  - bug
  - containers
  - no-coverage
dependencies: []
references:
  - 'PR #17'
  - src/expr.rs
  - sql_transform/_codegen/ (container-operand guard)
documentation:
  - doc-9 (Rich type system and UNNEST — status and deferred edges)
  - doc-1 (DataFusion function catalogue — parity oracle)
priority: medium
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WHAT A USER HITS

    SELECT s || 'x' FROM __THIS__      -- s is a struct column

DataFusion (the oracle) REJECTS this — concatenating a struct with a string is not a valid operation.

Native ACCEPTS it and silently returns the string `'{x: 1}x'` — it stringifies the struct's debug representation and glues the literal on. The user gets a plausible-looking string column built from an internal formatting of their data, with no indication anything went wrong.

Codegen refuses it correctly via the container-operand guard added in TASK-29 Phase C. So today the two engines disagree: codegen raises, native invents a value. Native is the DEFAULT serving engine, which is the wrong way round.

WHY THIS ONE IS WORTH FILING SEPARATELY
Found in passing by Ritchie during Phase C (2026-07-24), and notable for having ZERO test coverage anywhere in the suite — it was not a known-and-accepted gap, it was simply never asked about. It is a silent-wrong-answer class bug (same family as DRAFT-13), not a loud refusal.

It also retroactively justifies the container-operand guard test in Phase C: that guard is the only reason codegen does the right thing here, and the raise-test is what documents the intended behavior.

SCOPE NOTE
Native-only fix — codegen already behaves correctly. Unlike the UNNEST naming bugs (which must be fixed on both engines together to avoid splitting them), this one closes an existing split.

Worth checking as part of it: whether other container-typed operands reach binary operators on native without a type check, i.e. whether this is one hole or a class of them. A struct in a numeric expression, a list in a comparison, and so on.

DRAFT pending AmirHossein's review of scope/priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native rejects struct || string rather than returning a stringified struct, matching the DataFusion oracle
- [ ] #2 A differential test pins the behavior on both engines (native rejects, codegen already rejects, oracle rejects)
- [ ] #3 Root-cause sweep: determine whether other container-typed operands (struct/list) reach binary operators on native without a type check, and cover or fix what that sweep finds
<!-- AC:END -->
