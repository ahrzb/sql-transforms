---
id: TASK-123
title: >-
  A 10s subprocess timeout makes the arrow-drain test fail under load
status: To Do
assignee: []
created_date: '2026-08-18 00:00'
labels:
  - ci-hygiene
dependencies: []
type: chore
ordinal: 108000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`packages/sql-transform/sql_transform/_duckdb_arrow_test.py` spawns a CHILD
PROCESS per case with a hard budget:

```python
_TIMEOUT = 10.0   # "The healthy spellings finish in ~1s including interpreter start."
subprocess.run([sys.executable, "-c", _repro(fetch)], timeout=_TIMEOUT, check=True)
```

A 10x margin over the happy path is not enough on a busy machine. Measured
2026-08-17: `test_the_working_spellings[drain]` failed in two consecutive
full-suite runs (suite wall time 339s and 668s, against a normal ~100s) and
passed in three isolated runs and one later clean full run. Same tree
throughout — the only variable was machine load.

The failure is a `TimeoutExpired` on interpreter start plus `import duckdb`,
not the behaviour under test.

**Why the timeout cannot simply be deleted.** It is load-bearing: the whole
point of this file is that DuckDB *hangs* when fed its own undrained
`.arrow()` reader, and the timeout is what turns that hang into a test
failure instead of a wedged suite. The file's own docstring notes an earlier
version overstated the hang "from a run of three where it happened to hang
each time", so the fragility here has already bitten once.

Cost of leaving it: the gate lies about the tree on a loaded machine, which
is exactly when people are least able to tell a real regression from noise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the healthy spellings do not fail on a machine running a full suite
      concurrently — reproduce the load, do not assume
- [ ] #2 a genuine hang is still detected, and still as a FAILURE rather than
      a wedged run
- [ ] #3 whatever budget is chosen is justified against a MEASURED cold-start
      cost (interpreter + `import duckdb`) under load, not guessed
- [ ] #4 if a retry is used instead of a longer budget, it must not be able to
      mask an intermittent hang — say why in the code
<!-- AC:END -->
