# Divergence ledger

This chapter indexes recorded divergences and the evidence behind them. A passing test
or a directory name is not an owner ruling.

**claim: unresolved-observations.** A measured observation that is not yet understood
is reported in a visible, separately counted `unresolved` category. Unresolved is
neither agreement, nor an approved exception, nor by itself a confirmed defect, and an
unclassified difference stays unresolved rather than being relabelled
contract-unspecified. The unresolved/unspecified distinction itself is defined in
[ordering and status vocabulary](03-nondeterminism.md).

*Decision:* [oracle policy](../decisions/oracle-policy.md#evidence-and-unresolved-observations).

*Enforced-by:* `fuzz.runner.CATEGORY` and `fuzz.runner.report`. The campaign report's
outcome section counts, over a stated population, `agreement` (`AGREE`, `AGREE_TRAP`,
with `order-by-unevaluated` agreements named), `mismatch` (`DIVERGE_*`,
`OPT_EMULATED`, `BUILD_EXC`), `unresolved` (`SKIP`, `TIMEOUT`, `PANIC`: no verdict was
reached), `refused`, and `unshipped`; each `findings.jsonl` line carries its category.
A mismatch is never moved into `unresolved`. This covers the campaign's unanswered
cases; observations outside a campaign need a ledger entry.
*Evidence:* `packages/confit/tests/test_fuzz_report.py::test_unresolved_is_counted_apart_from_agreement_and_mismatch`
and `::test_every_kind_lands_in_exactly_one_outcome_category`.

## Executable ledgers

**claim: keep-vs-change.** The repository separates intent:

- `tests/known_divergences/` is the **KEEP** ledger: passing tests document behavior
  with a measured reason.
- `tests/test_open_divergences.py` is the **CHANGE** ledger: one `xfail(strict=True)`
  pin per entry, deleted rather than rewritten when fixed.

This separation does **not** approve every indexed behavior below. **claim:
strict-xfail** names the useful mechanism: an unexpected pass fails rather than
silently certifying an unreviewed change.

*Evidence:* `tests/known_divergences/README.md`; the module docstring of
`tests/test_open_divergences.py`.

**claim: keep-entry-reason.** A KEEP entry owes a measured reason, and a DuckDB claim
in that reason is subject to remeasurement. For the string-builder budget, DuckDB is
deterministic and the limit is confit's resource judgment; `known-limitations.md` §4
states it that way.

**claim: feature-in-flight.** Unfinished feature work is not an accepted permanent
divergence. Completing it replaces the strict xfail with a parity test, removes the
limitation and fuzzer marker, and is certified only by a campaign run after the
suppression is gone. The visible `UNSHIPPED` bucket is not independently gated; see
[the comparison contract](05-the-comparison-contract.md). Decimal literal typing and
decimal cast rounding are features in flight on these terms.

## Divergence index

**claim: ledger-adjudication.** The index below lists recorded divergences. A blank
ruling is **unruled**; neither code, tests, directory placement, nor the listed status
supplies approval.

Correctness is judged against the contract, never against membership in this index. A
confirmed in-contract mismatch is a defect unless an explicit approved exception permits
it; listing a mismatch here never approves it; and an invalid comparison is not by
itself an engine defect. An unlisted divergence is therefore neither excused nor a bug by
definition — the contract decides, and index completeness carries no service-level
promise.

*Decision:* [oracle policy](../decisions/oracle-policy.md#evidence-and-unresolved-observations).

Severity follows [the four-rung ladder](08-the-severity-ladder.md). Trap elision is a
contract entry rather than an engine-versus-oracle value divergence, but is indexed
where readers expect it.

| divergence | kind / severity | current status | ruling |
|---|---|---|---|
| **divergence: dedup-on-both-sides** | comparison contract / n/a | kept | unruled |
| **divergence: approximate-error-text** | comparison scope / n/a | kept | unruled |
| **divergence: ilike-nul** | source exclusion / n/a | kept | unruled |
| **divergence: trap-elision** | optimizer-on contract gap / 1 | kept | unruled |
| **divergence: nan-sign-per-platform** | platform-dependent answer / n/a | Linux bits are the contract | **ruled** by [oracle policy](../decisions/oracle-policy.md#reference-and-comparison) |
| **divergence: schema-qualifiers** | name resolution / 3 and 4 | kept | unruled |
| **divergence: decimal-literal-typing** | feature in flight / 2 | open severity-2 defect; not an approved exception | unruled |
| **divergence: decimal-cast-rounding** | same literal-typing mechanism / 2 | tied to parent | unruled |
| **divergence: bind-time-constant-refusals** | conservative refusal / 4 | kept | unruled |
| **divergence: regex-size-guard** | one-sided safety guard / 4 | kept | unruled |
| **divergence: string-builder-budget** | resource refusal / 4 | kept | unruled |
| **divergence: arrow-batch-ceiling** | resource refusal / 4 | kept | unruled |

The index covers `known-limitations.md` sections 3 and 5 and `tests/known_divergences/`.
Slugs are stable when multiple entries describe one mechanism.

## Evidence notes

- **divergence: dedup-on-both-sides.** DuckDB boundary deduplication renames duplicate
  output names on both sides before comparison. Evidence:
  `pins-wave5/dup-names-client-contract.json`; `confit.compare.dedup_names`;
  `test_known_limitations.py::test_duplicate_names_use_duckdbs_boundary_rename`.
- **divergence: approximate-error-text.** Corpus comparison checks successful answers
  and error class, not complete error-message identity. Evidence:
  `known-limitations.md` §5; `test_corpus_replay.py`; claim: error-texts.
- **divergence: ilike-nul.** `ILIKE` with embedded NUL varies with DuckDB column
  statistics; confit is NUL-transparent, so the source is excluded by name. Evidence:
  `test_corpus_replay.py` (`_KNOWN_DIVERGENT_SOURCES`); `pins-wave1/pins_like.json`;
  claim: statistics-dependent-exclusion.
- **divergence: trap-elision.** Optimizer-on DuckDB removes a trapping subexpression;
  optimizer-off DuckDB and confit evaluate it. Confit agrees with the oracle, while the
  user-visible optimizer-on surface is reported as `DIVERGE_OPT`. Evidence:
  `known-limitations.md` §5; `known_divergences/test_trap_elision.py`.
- **divergence: nan-sign-per-platform.** `%`-by-zero NaN sign follows platform libm.
  Under claim: reference-platform the contract is agreement with the Linux bits; other
  platforms' signs are not replicated. Evidence:
  `test_duckdb_wave3_mathtail.py::test_computed_nan_bits_match_oracle`;
  `pins-wave3/math_tail.json`.
- **divergence: schema-qualifiers.** `s1.t1` resolves by bare table name where DuckDB
  refuses (severity 3); `w.w.w` takes a longer schema-like parse and confit refuses where
  DuckDB serves (severity 4). A column qualified through a schema-qualified relation
  (`d.v` over `JOIN main.d`, or 3-part `s1.d.v`) also refuses where DuckDB serves the
  first (severity 4). Twins:
  `tests/test_known_limitations.py::test_a_relation_schema_qualifier_resolves_by_bare_name`,
  `::test_a_column_qualified_through_a_schema_qualified_relation_refuses`, and
  `::test_a_schema_like_struct_path_takes_the_longer_parse_and_refuses`.
- **divergence: bind-time-constant-refusals.** Confit refuses some trapping integer
  constants (for example `2147483647 + 1`) at construction. DuckDB defers them to
  execution: `PREPARE` succeeds, and `WHERE FALSE` or an empty input serves `[]`. This
  is not a blanket refusal of `WHERE FALSE`. Refusing where DuckDB serves is not by
  itself a correctness defect; the campaign report groups such refusals by DuckDB outcome
  (claim: refusal-absorb), and no twin measures this divergence's DuckDB-serves cost.
  Evidence: `eval_i32_literal` in `src/specializer/frontend.rs`; claim:
  phase-separated-probes.
- **divergence: regex-size-guard.** Confit's guard can fire before DuckDB's RE2 limit; it
  may over-refuse but cannot serve a query DuckDB rejects. Evidence:
  `pins-waveB/fuzzer-20260728.json`; `known-limitations.md` §4.
- **divergence: string-builder-budget.** Confit refuses a literal `pad`/`repeat` capable
  of exceeding 1 GiB. DuckDB is deterministic here (`repeat` through 4294967295;
  `lpad`/`rpad` binder-error above INT32), so the reason is resource policy, not
  spelling dependence. Evidence:
  `known_divergences/test_string_budget.py::test_a_budget_breaking_literal_count_refuses`.
- **divergence: arrow-batch-ceiling.** Matching DuckDB's `pa.string()` 32-bit offsets
  implies a 2 GiB-per-batch ceiling. `infer_arrow` raises
  `infer_arrow: string column exceeds 2 GiB in one ...` in `src/duckdb/arrow.rs`; no test
  exercises it. Evidence: the comment in `known_divergences/test_arrow_boundary.py`.
- **divergence: decimal-literal-typing / decimal-cast-rounding.** The row path types
  DECIMAL literals as f64, producing an `UNSHIPPED` schema difference and, for
  `CAST(-2.5 AS BIGINT)`, `-2` instead of DuckDB DECIMAL's `-3`. The casts agree when
  given the same input type. Evidence: `known-limitations.md` §3;
  `fuzz.oracle._type_delta`.

## Executable-twin coverage

**claim: doc-twin-totality.** The executable-twin mechanism is partial, not total.
Twins live in `test_known_limitations.py`, `test_arrow_schema_api.py`,
`test_corpus_replay.py`, `test_duckdb_wave3_mathtail.py`, and `known_divergences/`.
`known-limitations.md` does not claim that every limitation is asserted.

**claim: honest-coverage-claims.** State demonstrated coverage and important gaps;
do not claim totality without evidence. Fill behavioral gaps rather than require a
one-test-per-paragraph registry. Important gaps named in this chapter: no twin measures
the DuckDB-serves cost of divergence: bind-time-constant-refusals, and no test exercises
the Arrow batch ceiling.

*Decision:* [oracle policy](../decisions/oracle-policy.md#evidence-and-unresolved-observations).

## Where a decision is written down

**claim: divergence-placement.** An approved exception has one home, beside the
comparison rule that it qualifies, and is linked from this chapter. Open bugs and
unfinished investigations stay in the ledger. Indexing a divergence here never
approves it.

*Decision:* [oracle policy](../decisions/oracle-policy.md#limits-on-process-rules).
The [comparison contract](05-the-comparison-contract.md) lists the approved bounds
separately from independent references. The accepted policy itself is
[the oracle policy decision](../decisions/oracle-policy.md).
