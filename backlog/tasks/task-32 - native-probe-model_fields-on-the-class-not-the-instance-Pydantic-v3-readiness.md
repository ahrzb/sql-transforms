---
id: TASK-32
title: >-
  native: probe model_fields on the class, not the instance (Pydantic v3
  readiness)
status: Done
assignee:
  - Wren
created_date: '2026-07-19 15:43'
updated_date: '2026-07-19 15:52'
labels:
  - rust
  - parity
  - pydantic
milestone: m-1
dependencies: []
references:
  - src/expr.rs
  - tests/test_diff_types.py
priority: medium
type: bug
ordinal: 32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At infer time the native engine classifies "is this value a struct/model?" via `obj.hasattr("model_fields")` on the INSTANCE (src/expr.rs:187). Pydantic 2.11 deprecated instance access to model_fields (PydanticDeprecatedSince211 — the root cause of all ~14 current suite warnings) and v3.0 REMOVES it: at that point the probe returns false and native misclassifies struct-typed values. So this is a latent CORRECTNESS bug on the v3 bump, not just log noise. Fix is small + local: probe the class instead (`obj.get_type().hasattr("model_fields")`, or an isinstance-vs-BaseModel check). The other model_fields reads (src/schema.rs:30, _codegen/plan.py:54) are already on the class and are fine. Only surfaces in tests/test_diff_types.py (struct-valued cases). Found by QA on current master (484 passed / 16 skipped / 1 xfailed, ~14 warnings all this one root cause). DataFusion is the oracle (decision-1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 native struct/model detection probes the CLASS, not the instance; the ~14 PydanticDeprecatedSince211 warnings clear
- [x] #2 transform == infer parity holds across the tests/test_diff_types.py struct cases after the change (guards the v3 misclassification regression)
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Iris (PM)
created: 2026-07-19 15:44
---
Dispatched to Wren (2026-07-19). Testing note that matters here: under Pydantic 2.x the instance probe STILL WORKS (just deprecated), so a plain transform==infer parity test passes both before AND after the fix — it can't catch the v3 regression. The test with teeth asserts the deprecation is gone / detection is class-level (e.g. no PydanticDeprecatedSince211 emitted for the struct cases). No test = not done (AmirHossein's standing bar).
---

author: Iris (PM)
created: 2026-07-19 15:52
---
Done per Wren's merge bae07d9. The test meets the bar I set: it asserts the deprecation is gone (RED→GREEN), not just transform==infer parity, so it actually catches the v3-removal regression. Warnings cleared verified under -W error::DeprecationWarning.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged to master (bae07d9, fix commit abbfe68). src/expr.rs from_pyobject_typed now probes obj.get_type().hasattr("model_fields") (the CLASS) instead of the instance — mirrors schema.rs's already class-level read; one logic line (the larger expr.rs diff is cargo-fmt whitespace normalization in the same file, tests unaffected). schema.rs:30 and _codegen/plan.py:54 left untouched. Test with teeth: tests/test_native_model_fields_probe.py runs a struct value (arrives at infer as a validated nested-model INSTANCE) through native infer and asserts NO PydanticDeprecatedSince211 warning — confirmed RED before ('Accessing the model_fields attribute on the instance is deprecated'), GREEN after; this is what would break on the v3 removal, so it guards the latent misclassification. Verified: full suite 485 passed / 16 skipped / 1 xfailed; clean under -W error::DeprecationWarning (proves the ~14 warnings gone); cargo test 2 passed.
<!-- SECTION:FINAL_SUMMARY:END -->
