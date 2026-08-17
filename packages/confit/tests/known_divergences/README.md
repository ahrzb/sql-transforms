# The divergence record

Split out of the single 1429-line `test_known_divergences.py` on 2026-08-16.
The doctrine below is unchanged; only the packaging is.

| file | subject |
|---|---|
| `_helpers.py` | `probe` (fresh interpreter), `duck` (oracle leg), the tree-UDF fixture |
| `test_dropped_clauses.py` | clauses and modifiers parsed then dropped (TASK-69, TASK-81) |
| `test_cast_semantics.py` | CAST rounding mode and its refusal text (TASK-70, TASK-113) |
| `test_arrow_boundary.py` | infer_arrow output types and round-trips (TASK-71, TASK-72) |
| `test_join_residual.py` | the join ON residual, three ways (TASK-73, TASK-74) |
| `test_short_circuit.py` | WHERE short-circuit and three-valued logic (TASK-75) |
| `test_model_tables.py` | model-table structure refusals (TASK-76) |
| `test_trap_elision.py` | the constant folder (TASK-85, TASK-87) and the proof that this class is syntactic |
| `test_string_budget.py` | the string-builder budget and pad counts (TASK-82, TASK-88) |
| `test_literal_typing.py` | bare NULLs, INT32 overflow, signed zero (TASK-86, TASK-84, TASK-80) |

Open divergences do NOT live here - see `../test_open_divergences.py`.

---

The 2026-08-08 adversarial sweep — the durable record of every finding.

The engine's contract is: **either it matches DuckDB bit-for-bit, or it refuses
at build with a named error. There is no third mode.** Most of what follows was
a breach of that contract rather than an exotic edge case — SQL that DuckDB
runs and the engine silently answered differently.

Started as nine xfail-strict pins. As each was fixed its marker came off and
the section above it became the account of what the fix was and why, so this
file reads as history rather than as a list of complaints. Tickets are
TASK-69..TASK-78 in `backlog/tasks/`.

**Everything here PASSES.** Divergences still OPEN moved to
`test_open_divergences.py` on 2026-08-16, one xfail-strict pin each, ticket
named. The split is by INTENT: this file is behaviour we decided to KEEP and
the ground for keeping it; that file is behaviour we decided to CHANGE. A
reader who cannot tell the two apart at a glance either implements something
we chose not to have, or walks past a real bug because the paragraph above it
sounded like a rationale — the census that prompted the split found both
mistakes already present in this file.

So: an entry here owes a REASON, not just a description. Where a reason is a
claim about DuckDB it has to be measured, and it has to stay true — one below
had gone false and was propagating into a user-facing message.

Feature pins do NOT live here — they live with their feature's tests
(`test_decimals.py`, the width pin in `test_infer_arrow.py`). TASK-77 (an
integer feature above 2**53) is pinned in `sql_transform/_trees_test.py`
as a packer-side question.

Two of the findings were adjudicated by hand after the sweep's own verifiers
split on them, and they went opposite ways: TASK-78 was real and is fixed;
TASK-76 was a defect in the SPEC, not the code, and the spec was corrected —
see the model-table section at the bottom, which now checks every OTHER
refusal the spec claims by construction rather than assuming them.

Provenance: 6 finder agents over distinct surfaces, then two independent
refuters per finding, each required to build its own construction and to
default to "refuted". 18 raw findings, 12 verified, 9 confirmed, 2 disputed,
1 refuted. Four of the nine were reproduced by hand before being written down.

Two tests here run in a SUBPROCESS because the failure is a process death
(stack overflow), not an exception: observed from inside the session it would
take the whole test run with it.
