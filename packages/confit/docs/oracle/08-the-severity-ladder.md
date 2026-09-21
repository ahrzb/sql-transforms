# Severity

## The four rungs

**claim: severity-ladder.** Use one directional definition:

| rung | confit behavior relative to DuckDB | meaning |
|---:|---|---|
| **1** | traps where DuckDB serves | contract violation |
| **2** | serves a wrong value | contract violation |
| **3** | serves where DuckDB refuses | unsafe direction; DuckDB cannot run the query |
| **4** | refuses where DuckDB serves | conservative direction; may be deliberate |

*Evidence:* compatible partial definitions in
`docs/specs/2026-08-25-task-114-design.md:140-142` and
`docs/specs/2026-08-25-task-127-remainders-design.md:154-156`; use in
`src/specializer/frontend.rs:64-73` and the two refusal RFCs. Proposed
**ticket: severity-definition-merge** replaces the partial copies with this definition.

## Direction and disposition

**claim: directional-rungs.** Rungs 3 and 4 are not symmetric errors. Rung 4 may be a
deliberate conservative refusal, as with divergence: regex-size-guard. Rung 3 is unsafe
and appears only where DuckDB cannot be run, including divergence: schema-qualifiers and
the row path of divergence: narrow-lane-overflow.

A rung-1 or rung-2 mismatch fails parity with the chosen reference. Scheduling a fix
under claim: feature-in-flight does not turn it into agreement. Separate references
must not be conflated: trap elision is severity 1 against optimizer-on behavior but
agrees with the optimizer-off oracle.

The stronger proposal that rungs 1 and 2 may never be retained remains unadopted
(ask: proposed-rules-adoption). Decimal literal typing and rounding remain severity-2
features in flight, not approved exceptions to the target.

*Evidence:* `docs/reports/pins-first-methodology.md:79` and the indexed divergences in
[the ledger](07-the-divergence-ledger.md).

**claim: countable-rung-four.** **[PROPOSED]** A deliberate rung-4 refusal should report
how many otherwise-serving cases it excludes. This is not in force because claim:
countable-cost is also proposed. Divergence: bind-time-constant-refusals is the live
unmeasured case; ask: refusal-cost-counting requests the ruling.
