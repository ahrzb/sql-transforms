# The divergence record

| file | subject |
|---|---|
| `_helpers.py` | `probe` (fresh interpreter), `duck` (oracle leg), the tree-UDF fixture |
| `test_dropped_clauses.py` | clauses and modifiers parsed then dropped |
| `test_cast_semantics.py` | CAST rounding mode and its refusal text |
| `test_arrow_boundary.py` | infer_arrow output types and round-trips |
| `test_join_residual.py` | the join ON residual, three ways |
| `test_short_circuit.py` | WHERE short-circuit and three-valued logic |
| `test_model_tables.py` | model-table structure refusals |
| `test_trap_elision.py` | the constant folder and the proof that this class is syntactic |
| `test_string_budget.py` | the string-builder budget and pad counts |
| `test_literal_typing.py` | bare NULLs, INT32 overflow, signed zero |

Open divergences do NOT live here - see `../test_open_divergences.py`.

---

The engine's contract is: **either it matches DuckDB bit-for-bit, or it refuses
at build with a named error. There is no third mode.** Each entry here is a
place where that contract is easy to breach — SQL that DuckDB runs and an
engine could silently answer differently — together with the behaviour the
engine has and the reason for it.

**Everything here PASSES.** Divergences still OPEN live in
`test_open_divergences.py`, one xfail-strict pin each. The split is by INTENT:
this directory is behaviour we decided to KEEP and the ground for keeping it;
that file is behaviour we decided to CHANGE. A reader who cannot tell the two
apart at a glance either implements something we chose not to have, or walks
past a real bug because the paragraph above it sounded like a rationale.

So: an entry here owes a REASON, not just a description. Where a reason is a
claim about DuckDB it has to be measured, and it has to stay true.

Feature pins do NOT live here — they live with their feature's tests
(`test_decimals.py`, the width pin in `test_infer_arrow.py`). An integer
feature above 2**53 is pinned in `sql_transform/_trees_test.py` as a
packer-side question.

The model-table tests check every refusal the spec claims by construction
rather than assuming them.

Two tests here run in a SUBPROCESS because the failure is a process death
(stack overflow), not an exception: observed from inside the session it would
take the whole test run with it.
