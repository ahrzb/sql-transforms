# Version changes and evidence mutability

## Version rule

The [reference enforcement rule](01-what-the-oracle-is.md) fixes DuckDB 1.5.5 and
requires an exact oracle/test-environment pin plus a startup version assertion. A
reference upgrade is a separate reviewed change.

*Decision:* [oracle policy](../decisions/oracle-policy.md#reference-and-comparison).

## Current constraints

**claim: bump-object.** `confit.oracle.Oracle.VERSION` names the reference. A
deliberate reference upgrade updates that identity along with its environment.
Opening the oracle compares the constant with `duckdb.__version__`; pins recorded
outside `Oracle` are not covered by that assertion.

*Evidence:* `confit.oracle.Oracle.VERSION`; claim: oracle-version-constant; claim:
pin-provenance.

**claim: pin-re-runnability.** Pin files do not share one shape. Claims appear under
`probes`, `pins`, `findings`, or domain keys; SQL fields use `query`, `sql`, `q`, or
`expr`; setup often lives in prose; and only `pins-dialect/probe_joins.py` supplies a
nearby replay harness. Re-recording therefore first requires mechanical replay.

**claim: pin-conversion.** Old pins are made replayable without being rewritten.
`scripts/pin_corpus.py convert` derives setup statements where a pin states its tables
mechanically — its own `setup`, the file's shared `setup`, or a typed `input_repr` such
as `t(a BIGINT); rows=[(7,)]` — and writes them to `docs/specs/pins-replay.json` keyed by
JSON pointer; pin files are untouched, and a replay under today's oracle is not a fresh
run of the original capture. Every other pin query is inventoried with a reason: an
untyped `input_repr` (a bare value names no column and no type), a table described only
in prose, or a file whose engine is not DuckDB or is unstated. Every converted pin that
recorded a `result_repr` reproduces it exactly.

*Enforced-by:* `packages/confit/tests/test_pin_corpus.py::test_a_converted_pin_reproduces_what_it_recorded`
and `::test_every_pin_is_replayed_or_inventoried_with_a_reason`.

**claim: capture-outside-the-oracle.** `gen_casemap.py`, `gen_pow10.py`,
`gen_strip_accents.py`, and `mine_duckdb_corpus.py` open bare DuckDB connections without
`disable_optimizer` and do not use `confit.oracle`. Few pin files state optimizer state,
and `pins-stageB/order-contract.json` explicitly describes optimizer-on capture. These
are **not observations of the defined optimizer-off oracle**. Whether a capture is
useful for a narrower optimizer-on or provenance-only purpose must be stated case by
case. `Oracle`'s version assertion does not reach these bare connections.

*Evidence:* the named scripts; `docs/specs/pins-stageB/order-contract.json`; the pin
header's `optimizer` field (claim: uniform-pin-header in [pins](06-pins.md)).

**claim: mined-corpus-provenance.** The miner ignores sqllogictest expected blocks and
obtains rows through fresh optimizer-on connections. Its expected rows therefore are not
observations of the defined optimizer-off oracle; they may support a narrower purpose
only when that purpose and provenance are explicit. A mining run writes
`duckdb_mined.provenance.json` beside the corpus — date, DuckDB version, settings
profile (optimizer on, fresh default connection per file, threads), clone and miner
revisions, mined directories and counts. The committed `tests/corpus/duckdb_mined.jsonl`
has no such stamp and no provenance field, and a stamp saying optimizer on does not turn
rows into optimizer-off evidence.

*Evidence:* `scripts/mine_duckdb_corpus.py` (module docstring and `provenance`);
`tests/test_mined_corpus_stamp.py`.

## Re-recording

New campaign evidence is dated and provenance-bearing (claim: dated-provenance in
[pins](06-pins.md)).

**claim: re-record-diff-report.** Capture new answers with one command and emit a
reviewable diff; do not silently replace the corpus. `scripts/pin_ast_shapes.py`
does this for the AST-shape manifest, and `scripts/pin_corpus.py drift` does it for the
pin corpus: every pin query that replays mechanically is re-run on a fresh `Oracle` and
its answer written to `docs/specs/pins-drift.json`, which carries the DuckDB version and
platform but no date, so an unchanged reference re-runs to an identical file and a
changed one shows up as the file's `git diff`. A query answering differently on two runs
is recorded `<unstable>` (for example `uuid()`). Only files whose header engine is
DuckDB are replayed, including the converted setups (claim: pin-conversion). The tool
is manual, not a gate: platform-marked fields legitimately differ across platforms, and
each changed answer needs review.

*Evidence:* `packages/confit/tests/test_pin_corpus.py::test_the_drift_manifest_names_real_pins_and_this_reference`.

## Existing guard

**claim: remeasure-guard.** An in-force parity or divergence test may first assert the
live oracle answer, so oracle movement fails before confit's behavior can silently be
redefined as parity. This guards only tests using the pattern; it does not make the pin
corpus replayable or produce a corpus diff.

*Evidence:* tests under `tests/known_divergences/` that take the `oracle` fixture, for
example `test_string_budget.py::test_a_bigint_pad_count_refuses_like_duckdb`, which
asserts DuckDB's live answer beside confit's refusal.
