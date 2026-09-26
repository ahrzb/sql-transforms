# Version changes and evidence mutability

## Adopted version rule

The [reference enforcement rule](01-what-the-oracle-is.md) retains DuckDB 1.5.5
and requires an exact oracle/test-environment pin plus a startup version assertion.
An upgrade remains a separate reviewed change. The current implementation gaps
below are not evidence that this policy is still open.

*Decision:* [oracle policy](../decisions/oracle-policy.md#reference-and-comparison).

## Current constraints

**claim: bump-object.** `confit.oracle.Oracle.VERSION` names the intended reference.
A deliberate reference upgrade must update that identity along with its environment.
Opening the oracle compares the constant with `duckdb.__version__`; pin provenance is
still partial, so pins recorded outside `Oracle` are not covered by that assertion.

*Evidence:* `confit/oracle.py:74-76`; claim: oracle-version-constant; claim:
pin-provenance.

**claim: pin-re-runnability.** **[FACT, measured 2026-08-25]** Re-recording first
requires mechanical replay. The 53 pin files used 21 top-level shapes; claims appeared
under `probes`, `pins`, `findings`, or domain keys; SQL fields used `query`, `sql`, `q`,
or `expr`; setup often lived in prose; and only `pins-dialect/probe_joins.py` supplied a
nearby replay harness. Proposed **ticket: convert-unrunnable-pins** tracks the inventory
and conversion.

**claim: capture-outside-the-oracle.** **[FACT, measured 2026-08-25]**
`gen_casemap.py`, `gen_pow10.py`, `gen_strip_accents.py`, and
`mine_duckdb_corpus.py` opened bare DuckDB connections without `disable_optimizer` and
did not import `confit.oracle`; only 4 of 53 pin files mentioned optimizer state, and
`pins-stageB/order-contract.json` explicitly describes optimizer-on capture. These are
**not observations of the defined optimizer-off oracle**. Whether a capture remains
useful for a narrower optimizer-on, historical, or provenance-only purpose must be
stated case by case; this fact does not establish general acceptability.

*Evidence:* the named scripts; `scripts/mine_duckdb_corpus.py:111`;
`docs/specs/pins-stageB/order-contract.json`. `Oracle` now asserts its version on
open (ticket: version-assert, done), which does not reach these bare connections; the
missing common header remains proposed ticket: uniform-pin-header.

**claim: mined-corpus-provenance.** **[FACT]** The miner ignores sqllogictest expected
blocks, obtains rows through fresh optimizer-on connections, and writes no DuckDB
version, date, or settings profile to `tests/corpus/duckdb_mined.jsonl`. Its expected
rows therefore are not observations of the defined optimizer-off oracle. They may still
support a narrower purpose only when that purpose and provenance are explicit.

*Evidence:* `scripts/mine_duckdb_corpus.py:1-12, :111`;
`tests/corpus/duckdb_mined.jsonl` (678 lines and no provenance field when measured
2026-08-25). Future mining runs write `duckdb_mined.provenance.json` beside the corpus —
date, DuckDB version, settings profile (optimizer on, fresh default connection per
file, threads), clone and miner revisions, mined directories and counts (**ticket:
mined-corpus-stamp**, done; `scripts/mine_duckdb_corpus.py::provenance`,
`tests/test_mined_corpus_stamp.py`). The current corpus predates the stamp and is not
backfilled, and a stamp saying optimizer on does not turn its rows into optimizer-off
evidence.

## Proposed re-recording discipline

New campaign evidence must be dated and provenance-bearing (claim: dated-provenance
in [pins](06-pins.md)). That rule does not adopt the workflow below: the diff-report
command, the triage classes, and the mutability classes with their memberships remain
proposals, and the provenance facts recorded above stay facts rather than approvals.

**claim: re-record-diff-report.** Capture new answers with one command and emit a
reviewable diff; do not silently replace the corpus. `scripts/pin_ast_shapes.py`
demonstrates this only for the AST-shape manifest. Proposed **ticket:
corpus-drift-report** would generalize it.

**claim: diff-triage-classes.** Classify each changed row exactly once:

| proposed class | proposed consequence |
|---|---|
| upstream DuckDB bug fixed | update the pin, retaining the old evidence and reason |
| DuckDB behavior changed | require an owner decision and ledger entry |
| confit bug newly exposed | open a ticket and add a strict-xfail pin |
| now abstaining | record it in the disposition table; prior success was accidental |

**claim: changed-pin-record.** Retain a decision record for an accepted changed value so
an upstream change and a regression remain distinguishable.

**claim: mutability-classes.** A decision may assign one of these version-change rules;
the classes and all listed memberships remain unratified:

| proposed class | meaning | proposed examples |
|---|---|---|
| `frozen` | movement means the oracle is wrong | pseudo-oracle, nondeterminism, float-bit equality, severity |
| `follows-oracle` | support may widen with the oracle | enumerated quirks, deduplication, error classes |
| `may-change-on-bump` | re-decide during the bump | oracle identity, optimizer scope, platform/libm, parser-sensitive comparison, named float tolerances |

The proposals require, in order, a binding version and complete provenance, replayable
pins, a reviewable diff, classification, and a retained decision. Their presence here
does not authorize a version bump or pin rewrite.

## Existing guard

**claim: remeasure-guard.** An in-force parity or divergence test may first assert the
live oracle answer, so oracle movement fails before confit's behavior can silently be
redefined as parity. This guards only tests using the pattern; it does not make the pin
corpus replayable or produce a corpus diff.

*Evidence:* `docs/specs/2026-08-19-cast-semantics-design.md:25`;
`docs/specs/2026-08-25-task-120-design.md:374`;
`docs/specs/2026-08-25-task-133-join-keys-design.md:625`; examples under
`tests/known_divergences/`.
