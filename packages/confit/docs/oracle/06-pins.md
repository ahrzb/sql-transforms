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
`tests/test_oracle.py::test_connection_passthrough`. The methodology report does not yet
state this rule; proposed **ticket: phase-probing-in-methodology** tracks that gap.

## Provenance required for replay

**claim: pin-provenance.** A replayable pin identifies the oracle version, settings
profile, capture date, and capture-harness commit. `Oracle.VERSION` records an intended
version but is not itself a runtime assertion (claim: oracle-version-constant).

**claim: dated-provenance.** New campaign results must be dated and record the executed
SQL, inputs, generator revision where generated, engine revision, and reference
configuration. A seed is an aid, not a durable identity after a generator change.
Earlier runs remain frozen history, not rewritten to appear current.

*Decision:* [oracle policy](../decisions/oracle-policy.md#evidence-and-unresolved-observations).
Adoption does not establish enforcement or choose a file format. It does not convert
the existing pin corpus or adopt the metadata proposals below; the existing
pin-provenance rule still applies to pins.

The 2026-08-25 inventory found 41 of 53 files with `duckdb_version`, 10 with a capture
date, and 3 with a harness or commit; version spelling and metadata shape varied.
`pins-dialect/joins.json` and `pins-waveB/fuzzer-task54.json` are concrete partial
examples. **claim: generator-version-stamps** records the narrower existing practice:
`pin_ast_shapes.py`, `gen_casemap.py`, and `gen_strip_accents.py` stamp a DuckDB version,
and the latter two call for regeneration after a bump, but that does not standardize the
pin corpus.

*Evidence:* `scripts/pin_ast_shapes.py:29, :36`; `scripts/gen_casemap.py:152, :159`;
`scripts/gen_strip_accents.py:135`; `docs/specs/pins-*/*.json` (measured 2026-08-25).
Proposed **ticket: uniform-pin-header** covers the missing common header.

## Metadata proposals

Neither proposal below is in force, and claim: dated-provenance does not adopt either:
it fixes what new evidence must carry, not what shape a pin file takes.

| proposal | proposed addition | ticket |
|---|---|---|
| **claim: pin-back-reference** | record the stable slug of the decision the pin evidences | **ticket: pin-decision-field** |
| **claim: under-determined-token** | mark a field outside the contract or varying by a named discriminator such as platform | **ticket: pin-field-token** |

No surveyed pin had either field on 2026-08-25. The current modulo-NaN-sign exception is
explained beside the data and therefore does not supply a general token format.
