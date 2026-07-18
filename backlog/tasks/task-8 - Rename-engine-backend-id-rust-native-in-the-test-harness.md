---
id: TASK-8
title: Rename engine backend id 'rust' -> 'native' in the test harness
status: Done
assignee: []
created_date: '2026-07-18 15:09'
updated_date: '2026-07-18 17:56'
labels:
  - refactor
  - tests
dependencies: []
references:
  - tests/differential.py
  - tests/conftest.py
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs now call the inference engine 'native' (docs rename 253c4ac) but the differential harness still uses the literal identifier 'rust'. Rename for consistency; keep the suite green on both backends (native + codegen). Sites: tests/differential.py:156 BACKENDS key 'rust'->'native', :157 _backend default, :8 docstring, optionally _run_infer (:148)->_run_native; tests/conftest.py:38 set_backend('rust') reset, :51 _backend=='rust' inside the xfail marker, the xfail_on_rust marker/fixture (~:43)->xfail_on_native, comments :6/:12. Update all @pytest.mark.xfail_on_rust / xfail_on_rust usages across tests/ (incl. test_diff_rust_bugs.py). Optional/bigger (flag, don't require in this task): rename file test_diff_rust_bugs.py->test_diff_native_bugs.py and module sql_transform._interpreter; leave the merged rust-parity-bugs branch name alone (history).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 differential.py backend id + BACKENDS key + set_backend + comparisons use 'native', not 'rust'
- [x] #2 xfail_on_rust marker/fixture renamed to xfail_on_native; all call sites updated
- [x] #3 full suite green on both backends (transform==native, transform==codegen); no 'rust' backend id left in tests/
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done & merged (task-8-native-backend-id -> 714429d). Suite green both backends: 439 passed, 16 skipped, 1 xfailed (the TASK-7 native float-display residual). Note: the fixture was actually named rust_bug (not xfail_on_rust); renamed to xfail_on_native.
<!-- SECTION:NOTES:END -->
