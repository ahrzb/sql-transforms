# Pins

Pins are measured oracle evidence. They do not define confit's implementation or turn an
unmeasured generalization into oracle behavior.

## What a pin establishes

**claim: pin-as-data.** A pin records the exact SQL and inputs sent to DuckDB and the
exact answer returned, including reprs, float bits, or verbatim error heads when those
details matter. It specifies an oracle fact; it is not a test of confit.

**claim: no-semantics-from-memory.** Implement semantics only after recording an
executed oracle query. Documentation, memory, intuition, and unsupported summaries are
not measurements. **claim: over-generalized-summary** is the concrete warning: a wave-3
summary extended integer `%`-by-zero results to DOUBLE without probing it. The integers
return NULL, the missed DOUBLE case returns NaN, and
`docs/specs/pins-wave3/math_tail.json` now records both the correction and the original
coverage gap.

*Evidence:* `docs/reports/pins-first-methodology.md:20-28`; the pin corpus under
`docs/specs/pins-*/` (inventory measured 2026-08-25).

**claim: phase-separated-probes.** A claim about *when* DuckDB acts needs a
phase-separated probe. Because `con.execute` combines preparation and execution, a
bind-time claim requires PREPARE/EXECUTE, a zero-row leg, a pinned source, and the
measured phase. Value probes may use `confit.oracle.Oracle.answer`; phase probes may use
the underlying connection.

*Evidence:* `docs/rfcs/2026-08-19-keep-the-bind-time-refusals.md:29-58`;
`tests/test_oracle.py::test_connection_passthrough`. The methodology report states the
three-leg procedure in §2 (**ticket: phase-probing-in-methodology**, done).

## Provenance required for replay

**claim: pin-provenance.** A replayable pin identifies the oracle version, settings
profile, capture date, and capture-harness commit. `Oracle.VERSION` records an intended
version, and opening the oracle asserts it (claim: oracle-version-constant).

**claim: dated-provenance.** New campaign results must be dated and record the executed
SQL, inputs, generator revision where generated, engine revision, and reference
configuration. A seed is an aid, not a durable identity after a generator change.
Earlier runs remain frozen history, not rewritten to appear current.

*Decision:* [oracle policy](../decisions/oracle-policy.md#evidence-and-unresolved-observations).
It does not convert the existing pin corpus or adopt the metadata proposals below; the
existing pin-provenance rule still applies to pins.

*Enforced-by:* the campaign runner. `fuzz.runner.campaign` opens `findings.jsonl` with
one `{"provenance": ...}` line — UTC start date, engine revision (git `HEAD` plus a
dirty flag for `packages/confit`), generator revision (a hash of `fuzz/gen.py`), the
reference (installed DuckDB, `Oracle.VERSION`, optimizer-off baseline and optimizer-on
bracket), seed range, and platform. Every verdict line carries its SQL and its case's
`inputs` (row schema, rows, static tables, UDF and tree specs, shape), and a `TIMEOUT` or
`PANIC` is blamed with the same fields by regenerating its seed in the parent. The
2026-08-17 `findings.jsonl` predates this and stays frozen as it is.
*Evidence:* `packages/confit/tests/test_fuzz_report.py::test_findings_open_with_a_dated_provenance_header`,
`::test_a_dead_worker_is_blamed_with_its_sql_and_inputs`, and
`packages/confit/tests/test_fuzz_smoke.py::test_a_verdict_line_carries_its_inputs_not_just_a_seed`.

The 2026-08-25 inventory found 41 of 53 files with `duckdb_version`, 10 with a capture
date, and 3 with a harness or commit; version spelling and metadata shape varied.
`pins-dialect/joins.json` and `pins-waveB/fuzzer-task54.json` are concrete partial
examples. **claim: generator-version-stamps** records the narrower existing practice:
`pin_ast_shapes.py`, `gen_casemap.py`, and `gen_strip_accents.py` stamp a DuckDB version,
and the latter two call for regeneration after a bump, but that does not standardize the
pin corpus.

*Evidence:* `scripts/pin_ast_shapes.py:29, :36`; `scripts/gen_casemap.py:152, :159`;
`scripts/gen_strip_accents.py:135`; `docs/specs/pins-*/*.json` (measured 2026-08-25).

**claim: uniform-pin-header.** Every pin file opens with a `_pin` header of one shape:
`schema`, `subject`, `engine`, `engine_version`, `optimizer`, `captured`, `harness`, and
`committed`. `scripts/pin_corpus.py header` derives it from what the file and git
already say and writes `"unknown"` for anything the file does not state; `committed` is
the git date the file was added, never presented as a capture date, and `optimizer` is
`on` only for the three stage-B files whose observations are optimizer plan choices.
On 2026-09-26: 48 files name DuckDB 1.5.5 (41 by key, 7 in prose), 6 state a capture
date, and every other field left unknown stays unknown. The legacy fields are untouched
and no pin became a fresh observation (**ticket: uniform-pin-header**, done).

*Enforced-by:* `packages/confit/tests/test_pin_corpus.py`, which re-derives every
header and fails on a stale or missing one.

## Metadata proposals

The back-reference below is implemented as derived data, not as mandatory governance
metadata; claim: dated-provenance adopts neither proposal:
it fixes what new evidence must carry, not what shape a pin file takes.

| proposal | proposed addition | ticket |
|---|---|---|
| **claim: pin-back-reference** | record the stable slug of the decision the pin evidences | done: the header's `evidences` lists every oracle claim or divergence slug citing the pin, and every other doc, test or source file citing it by path, file name or directory; derived by `scripts/pin_corpus.py`, never hand-kept (**ticket: pin-decision-field**) |
| **claim: under-determined-token** | mark a field outside the contract or varying by a named discriminator such as platform | done: the header's `varies` lists `{at, mark, note}` with `at` a JSON pointer (`*` over list indices) and `mark` either `by:<discriminator>` or `unspecified`; reviewed entries in `scripts/pin_corpus.py`, pointers checked by `test_pin_corpus.py` (**ticket: pin-field-token**) |

No surveyed pin had either field on 2026-08-25. On 2026-09-26 the modulo-NaN-sign bits
in `pins-wave3/math_tail.json` are marked `by:platform`, and the join results of
`pins-stageB/order-contract.json` and `dup-key-equi.json` `unspecified` in row order; the
recorded values are unchanged. The token is an encoding, not the distinction itself,
which [status vocabulary](03-nondeterminism.md) defines.
