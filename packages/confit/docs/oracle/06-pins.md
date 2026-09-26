# Pins

Pins are measured oracle evidence. They do not define confit's implementation or turn an
unmeasured generalization into oracle behavior.

## What a pin establishes

**claim: pin-as-data.** A pin records the exact SQL and inputs sent to DuckDB and the
exact answer returned, including reprs, float bits, or verbatim error heads when those
details matter. It specifies an oracle fact; it is not a test of confit. Measured pins
for one semantic family are summarized in a pins spec under `docs/specs/` with the raw
evidence beside it as JSON under `docs/specs/pins-*/`.

**claim: no-semantics-from-memory.** Implement semantics only after recording an
executed oracle query; the pins are the contract the code is written to. Documentation,
memory, intuition, and unsupported summaries are not measurements: a summary sentence
with no executed query behind it is a guess. **claim: over-generalized-summary** is the
concrete warning: integer `%`-by-zero returns NULL, but DOUBLE `%`-by-zero returns NaN,
and a summary that extended the integer result to DOUBLE without probing it was wrong.
`docs/specs/pins-wave3/math_tail.json` records both cells.

*Evidence:* the pin corpus under `docs/specs/pins-*/` and the measured pins specs
(`docs/specs/2026-07-2*-*-pins.md`, `2026-07-28-waveA-structural-tails.md`).

**claim: phase-separated-probes.** A claim about *when* DuckDB acts needs a
phase-separated probe. `con.execute` prepares and executes in one call, so a refusal
seen there does not say whether it came from the binder or from running the plan over
data. A claim that DuckDB refuses a construct at bind time is measured three ways, each
on its own connection:

1. `PREPARE p AS <sql>` — only the binder and planner run; an error here is bind-phase;
2. a plain execute over the fixture rows; and
3. a zero-row leg — the same query over an empty table, or with `... WHERE 1=0` — so no
   row reaches an expression that could trap.

All three agreeing on the refusal makes it bind-phase; a refusal that appears only in
the second leg is a runtime trap and must not be moved to construction. The pin records
the SQL, the source it rests on, and the measured phase. Value probes may use
`confit.oracle.Oracle.answer`; phase probes reach the underlying connection through the
same `Oracle` (it forwards unknown attributes), never a bare `duckdb.connect`.

*Evidence:* `tests/test_oracle.py::test_connection_passthrough`.

## Provenance required for replay

**claim: pin-provenance.** A replayable pin identifies the oracle version, settings
profile, capture date, and capture-harness commit. `Oracle.VERSION` records the
reference version, and opening the oracle asserts it (claim: oracle-version-constant).

**claim: dated-provenance.** Campaign results are dated and record the executed SQL,
inputs, generator revision where generated, engine revision, and reference
configuration. A seed is an aid, not a durable identity after a generator change. A
recorded run is never rewritten to appear current.

*Decision:* [oracle policy](../decisions/closed/oracle-policy.md#evidence-and-unresolved-observations).

*Enforced-by:* the campaign runner. `fuzz.runner.campaign` opens `findings.jsonl` with
one `{"provenance": ...}` line — UTC start date, engine revision (git `HEAD` plus a
dirty flag for `packages/confit`), generator revision (a hash of `fuzz/gen.py`), the
reference (installed DuckDB, `Oracle.VERSION`, optimizer-off baseline and optimizer-on
bracket), seed range, and platform. Every verdict line carries its SQL and its case's
`inputs` (row schema, rows, static tables, UDF and tree specs, shape), and a `TIMEOUT` or
`PANIC` is blamed with the same fields by regenerating its seed in the parent.
*Evidence:* `packages/confit/tests/test_fuzz_report.py::test_findings_open_with_a_dated_provenance_header`,
`::test_a_dead_worker_is_blamed_with_its_sql_and_inputs`, and
`packages/confit/tests/test_fuzz_smoke.py::test_a_verdict_line_carries_its_inputs_not_just_a_seed`.

**claim: generator-version-stamps.** `pin_ast_shapes.py`, `gen_casemap.py`, and
`gen_strip_accents.py` stamp the DuckDB version they read; the latter two call for
regeneration after a version change.

*Evidence:* `scripts/pin_ast_shapes.py`, `scripts/gen_casemap.py`, and
`scripts/gen_strip_accents.py`.

**claim: uniform-pin-header.** Every pin file opens with a `_pin` header of one shape:
`schema`, `subject`, `engine`, `engine_version`, `optimizer`, `captured`, `harness`, and
`committed`. `scripts/pin_corpus.py header` derives it from what the file and git
already say and writes `"unknown"` for anything the file does not state; `committed` is
the git date the file was added, never presented as a capture date, and `optimizer` is
`on` only for the three `pins-stageB/` files whose observations are optimizer plan choices.
The header leaves each file's own fields untouched and does not make a pin a fresh
observation.

*Enforced-by:* `packages/confit/tests/test_pin_corpus.py::test_every_pin_file_opens_with_a_current_header`,
which re-derives every header and fails on a stale or missing one.

**claim: pin-back-reference.** The header's `evidences` lists every oracle claim or
divergence slug citing the pin, and every other doc, test, or source file citing it by
path, file name, or directory. `scripts/pin_corpus.py` derives it; it is never
hand-kept, so editing a citation in these chapters requires re-running
`scripts/pin_corpus.py header`.

**claim: under-determined-token.** The header's `varies` lists `{at, mark, note}`
entries: `at` is a JSON pointer (`*` over list indices) and `mark` is either
`by:<discriminator>` or `unspecified`. The modulo-NaN-sign bits in
`pins-wave3/math_tail.json` are marked `by:platform`, and the join results of
`pins-stageB/order-contract.json` and `dup-key-equi.json` are marked `unspecified` in row
order; recorded values are unchanged. The token is an encoding, not the distinction
itself, which [status vocabulary](03-nondeterminism.md) defines.

*Enforced-by:* reviewed entries in `scripts/pin_corpus.py`;
`packages/confit/tests/test_pin_corpus.py::test_every_under_determined_token_names_real_fields`
and `::test_the_token_marks_the_platform_dependent_nan_sign`.
