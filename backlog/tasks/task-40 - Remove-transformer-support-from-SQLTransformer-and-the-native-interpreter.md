---
id: TASK-40
title: Remove transformer support from SQLTransformer and the native interpreter
status: In Progress
assignee:
  - '@Claude'
created_date: '2026-07-25 01:09'
updated_date: '2026-07-25 01:25'
labels:
  - epic
dependencies: []
references:
  - 'https://github.com/ahrzb/sql-transforms/pull/22'
priority: high
type: task
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Epic: rip out the transformer-ref / transformer-callout feature for now (v0, no backward compat — delete outright, no deprecation shims).

Scope, two removals:

1. **SQLTransformer surface (Python)** — remove the ability to reference fitted sklearn transformers inside a composition:
   - `sql_transform/_transformer_ref.py`, `sql_transform/_transformer_udf.py` (delete)
   - `is_transformer` branches in `sql_transform/_compose.py` (Ref.is_transformer, unfitted-transformer guard, callout build path)
   - transformer paths in `sql_transform/_batch.py`, `sql_transform/_codegen/engine.py`, `sql_transform/__init__.py` exports
2. **Native interpreter (Rust)** — remove the transformer callout / UDF machinery:
   - `Expr::Transform` and related plumbing in `src/expr.rs`, `src/plan.rs`, `src/types.rs`, `src/schema.rs`, `src/lib.rs`
   - regenerate `sql_transform/_interpreter.pyi`

Tests to delete: `tests/test_transformer_udf.py`, `tests/test_transformer_ref.py`, `tests/test_transformer_case.py`, `tests/test_transformer_callout_infer.py`, `tests/test_diff_transformer_callout.py`, plus transformer cases in `sql_transform/_compose_test.py` and `tests/test_diff_errors.py`.

Ticket interactions to resolve before/while dispatching:
- TASK-3 (In Progress, transformer-refs follow-ups) works on the exact surface being deleted — needs a stop/land decision.
- TASK-34 (To Do, codegen transformer support) becomes obsolete — archive when this epic lands.
- TASK-31's parity test gets deleted with the feature.

Docs: update SQL_SUPPORT.md / README where transformer refs are mentioned; the design specs under docs/superpowers/specs stay as history.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No transformer-ref / callout code paths remain in sql_transform (grep for is_transformer, transformer_ref, transformer_udf comes back empty)
- [x] #2 No Expr::Transform / callout machinery remains in src/*.rs; native extension rebuilds clean
- [x] #3 All transformer tests removed; full test suite passes on both engines (codegen + native)
- [x] #4 TASK-3 resolved (stopped or landed) and TASK-34 archived or re-scoped
- [x] #5 Docs no longer advertise transformer refs
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removal done on branch `claude/remove-transformer-support-deb837`; PR open, ticket stays In Progress until it merges.

PYTHON SURFACE
- Deleted `_transformer_ref.py` (239 lines: is_transformer, _bind_names, _probe, resolve_transformer_refs) and `_transformer_udf.py` (84 lines: the DataFusion ScalarUDF wrapper).
- `_compose.py`: dropped `Ref.is_transformer`, the `is_transformer(v)` branch, and the transformer-shaped-but-unfitted guard. A sklearn object now falls through to the existing TypeError, which names the type: "interpolation {sc} must be a SQLTransform or its .transform, got StandardScaler". SQLTransform-to-SQLTransform composition is untouched.
- `__init__.py`: dropped `_udf_specs`, the sqlt/tfm ref split, and the `transformers=` kwarg to InferFn; `inline_references` now takes `self._refs` whole.
- `_batch.py`: dropped the `transformers` param and UDF registration loop.
- `_codegen/engine.py`: dropped the `__tfm_` UnsupportedInCodegen branch (an unknown function is now uniformly a ValueError).

NATIVE INTERPRETER
- `expr.rs`: removed the `Expr::Transform` variant and its ~90-line eval arm (the GIL-attach / feature reorder / .transform() / .tolist() marshalling path).
- `lib.rs`: removed `ResolvedTransformer`, `read_feature_names_in`, the 70-line `resolve_transformers` tree rewrite, and the `transformers` constructor param.
- `types.rs`, `plan.rs`: removed the `Expr::Transform` arms from infer_type and validate_expr.
- `schema.rs`: removed `arrow_schema_to_ordered_fields` (its only caller was the transformer path).
- Unused imports cleaned: `Arc` + `FieldType` in expr.rs, `HashSet` in types.rs.

VERIFICATION
- `cargo check`: clean, zero warnings.
- `uv run pytest`: 523 passed, 12 xfailed (the xfails are pre-existing, unrelated to transformers).
- `ruff check` + `ruff format --check`: clean.
- Ran the rewritten README example end to end: transform() gives [1.0, 1.0, 1.0, 1.0], infer() gives m=1.0, a bare `{t}` re-fit ref gives [0.4, 0.8, 1.2, 1.6], and a fitted StandardScaler is rejected with the expected TypeError.
- `cargo fmt --check` reports 11 diffs, all pre-existing (master has 13 in the same files); cargo fmt is not in the project's `mise run fmt` pipeline, so the crate was never fmt-clean. Not touched -- reformatting would bury the deletion in noise.

TICKET FALLOUT
- TASK-3 closed Done: its five landed commits are on master, ACs #1-#6 are void (they describe deleted code). Its standing pre-authorization to dispatch to Wren is retired.
- TASK-34 archived: it exists to port this feature to codegen, so its premise is gone. Its pre-authorization to auto-dispatch to Ritchie when TASK-29 lands is retired -- that was the urgent part, otherwise a dev gets sent at deleted code.
- TASK-31's parity test (tests/test_transformer_case.py) was deleted with the feature; TASK-31 stays Done with a note.

DOCS
- README: replaced the "Referencing a fitted sklearn transformer" section with "Referencing another SQLTransform" (the surviving composition feature, with a verified runnable example), and noted the sklearn surface is gone for now. Design specs under docs/superpowers/specs and the backlog docs/decisions/drafts are left as history.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: Claude
created: 2026-07-25 01:25
---
PR open: https://github.com/ahrzb/sql-transforms/pull/22 (+152 -1616 across 24 files). All 5 ACs met and verified; ticket stays In Progress until the PR merges.
---
<!-- COMMENTS:END -->
