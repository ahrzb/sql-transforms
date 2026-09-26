# Ordering and nondeterminism

Settled ordering and nondeterminism policy comes from the
[oracle policy record](../decisions/oracle-policy.md).

## Decision rule

**claim: nondeterminism-axiom.** An exact answer must be a function of the query and its
declared inputs. A result that also depends on hidden table history, an unselected
evaluation path, or an uncontrolled run is not an exact target.

Two ruled exceptions are narrower than a general nondeterminism allowance:

- confit defines serving-row order even where DuckDB's join-match sequence is
  unspecified; and
- per-row `sum` / `avg` over matched `DOUBLE` values is compared under the
  [float-reduction bound](05-the-comparison-contract.md#floating-point-comparisons)
  rather than exact bits.

*Evidence:* `packages/confit/docs/known-limitations.md` (introduction), claim:
optimizer-on-reading, and the [fold decision](../decisions/trustworthy-fold.md).
The rule is also law P21 in [properties](../properties.md), with its pinning tests.

**claim: disposition-table.** Apply the narrowest ruled action. Do not convert a source
of variation into a general tolerance.

| variation source | required action | authority |
|---|---|---|
| request-batch dependence, or no request table | outside the model ([goal](../goal.md#scope)) | serving-contract scope; fold decision |
| unspecified DuckDB row sequence | compare DuckDB as a multiset; enforce confit's order through self-legs | claim: compare-modes; claim: serving-row-order |
| hidden table statistics | remove the dependency where possible; otherwise exclude only the measured source | claim: optimizer-on-reading; claim: statistics-dependent-exclusion |
| two DuckDB evaluation paths disagree and the identity selects neither | refuse the construct by name | claim: evaluation-path-disagreement |
| scalar `cbrt` differs across DuckDB builds | allow at most 1 ulp for that family | claim: cbrt-ulp-tolerance |
| per-row `sum` / `avg` reduces matched `DOUBLE` rows | apply the addend-dependent bound | claim: float-reduction-bound |
| another order-sensitive value family | define a justified family contract before serving it, or refuse the family | claim: order-sensitive-family-contract |

The optimizer-on bracket is diagnostic, not an accepted alternative answer. A case that
only matches optimizer-off is still reported against the ordinary DuckDB surface as
`DIVERGE_OPT`.

**claim: target-status-vocabulary.** `PINNED` (stable contracted behavior),
`IMPL-DEFINED` (stable for a named build or configuration), and `UNSPECIFIED` (no unique
contracted answer) are available vocabulary, used where they clarify how a target may
vary. They are not a mandatory label on every claim, ledger row, or pin.

`UNSPECIFIED` and **unresolved** are not synonyms. `UNSPECIFIED` means the contract
deliberately leaves that aspect unconstrained; an unresolved observation is one not yet
understood. An unclassified difference stays unresolved and separately counted, as the
[ledger](07-the-divergence-ledger.md) records it; it is never relabelled unspecified.

## Row ordering

**claim: serving-row-order.** Serving preserves request-row order:

- `map`: output row `i` corresponds to input row `i`;
- `filter`: output is an input-order subsequence; and
- `many`: each input row's matches are contiguous, with input-row blocks in input order.

This is a confit contract, not a claim about DuckDB's output sequence.

*Enforced-by:* `fuzz.oracle._extra_legs`, using ordered batch-versus-single and input
reversal checks.
*Evidence:* `packages/confit/tests/test_fuzz_order_legs.py` and
`packages/confit/docs/known-limitations.md` (introduction, row order).

**claim: join-output-order.** DuckDB hash-join match order is not a stable sequence. It
varies with streamed-side selection, LIFO duplicate-key chains in 2048-row passes, and,
at roughly 500k rows or more, thread scheduling. DuckDB parity for `shape="many"` is
therefore a multiset. Confit still promises the deterministic serving order above: probe
rows follow input order and each probe row's matches follow build insertion order.

*Evidence:* `packages/confit/docs/specs/pins-stageB/order-contract.json` and
`packages/confit/docs/specs/2026-07-28-stageB-multiplicity-pins.md`.

**claim: compare-modes.** For a target row-path case, the campaign uses one value
comparison relation for both DuckDB readings and separate ordering checks:

1. compare confit with each DuckDB reading as a row multiset; then
2. use confit-only self-legs to enforce claim: serving-row-order.

Both legs use the [same value canonicalization](05-the-comparison-contract.md#rows-and-values).
DuckDB's unspecified sequence therefore cannot weaken confit's own ordering promise.

*Enforced-by:* `fuzz.oracle.compare_mode`, `confit.compare.multiset`, and
`fuzz.oracle._extra_legs`.
*Evidence:* `packages/confit/tests/test_compare.py::test_assert_rows_default_accepts_reordered_rows`,
`::test_assert_rows_ordered_rejects_reordered_rows`, and
`packages/confit/tests/test_fuzz_order_legs.py`.

## Value sources that cannot be exact targets

**claim: statistics-dependent-exclusion.** Exclude a statistics-dependent behavior by
source name only after measuring why a row-local engine cannot reproduce it. The known
case is DuckDB `ILIKE` over NUL-containing text: an all-ASCII column selects a NUL-safe
ASCII kernel, while a non-ASCII sibling row selects a generic kernel whose fold
NUL-truncates. The same row can therefore change value because of a sibling row. Confit
uses NUL-transparent row-local behavior. The corpus exclusion is one source file
covering two statements.

*Evidence:* `packages/confit/tests/test_corpus_replay.py` (`_KNOWN_DIVERGENT_SOURCES`),
`packages/confit/docs/specs/pins-wave1/pins_like.json`, and
`packages/confit/docs/known-limitations.md` §5.

**claim: corpus-exclusion-sets.** The corpus gate has three exclusion mechanisms, with
different reasons:

1. `_KNOWN_DIVERGENT_SOURCES`: the one statistics-dependent source above, covering two
   statements;
2. a blanket input-`FLOAT` rule, because widening f32 to f64 preserves a value but
   changes the grid observed by `nextafter`, shortest-round-trip text, and rounding
   (three sources, five cases); and
3. `_INEXPRESSIBLE_INPUTS`: one declared schema the test surface could not express.
   Its comment gives a width-less-pydantic explanation, which row schemas declared in
   Arrow may make stale.

*Evidence:* `packages/confit/tests/test_corpus_replay.py`.

**claim: evaluation-path-disagreement.** If two DuckDB evaluation paths in the same
build return different values and the reference identity selects neither path, refuse
the construct by name. Measured examples are anchor-only multi-anchor regexes and `$`
in non-final position: row evaluation may literal-optimize to PREFIX while constant
evaluation performs a normal regex match.

This differs from claim: native-tables, where the identity selects one input path, and
from claim: cbrt-ulp-tolerance, where separate builds differ within a named bound.

*Evidence:* `packages/confit/docs/specs/pins-waveB/fuzzer-20260728.json` and
`packages/confit/docs/known-limitations.md` §4.

**claim: native-tables.** Oracle inputs are native DuckDB tables, not registered Arrow
relations. DuckDB can push constant filters into a registered-Arrow scan using IEEE NaN
semantics that differ from native-table ordering. `Oracle.load` therefore registers the
Arrow data, creates a native table, and unregisters the relation. Column widths survive;
`NOT NULL` does not, so a constraint-sensitive fixture declares its table in SQL.

*Enforced-by:* `confit.oracle.Oracle.load` and `.table`.
*Evidence:* `packages/confit/tests/test_oracle.py::test_load_materializes_a_native_table_with_widths_intact`,
`::test_load_unregisters_its_alias`, and
`::test_table_keeps_the_declaration_including_not_null`. The semantic measurement is in
`packages/confit/docs/specs/2026-07-26-stretch4-builtin-pins.md`; no
registered-relation NaN regression gate exists.

## Thread count and order-sensitive values

**claim: threads-setting.** `Oracle.__init__` does not set `threads`; DuckDB's default
is machine-derived. With the optimizer disabled, `string_agg` element order over 400k
rows differs between thread settings. The independent fit/serving path uses
`threads = 1`. No global thread setting is pinned for the oracle; any such setting
would apply to the oracle as a whole, never as a caller, campaign, or per-case choice.

*Evidence:* P11 in `packages/confit/docs/properties.md`, `confit.oracle.Oracle.__init__`,
and the [fold decision](../decisions/trustworthy-fold.md).

**claim: order-sensitive-family-contract.** An order-sensitive value family is refused
until its comparison contract is clear. That contract names either the oracle setting
the family requires — pinned for every comparison, never per caller or case — or a
family-specific bound or comparison relation. Single-thread execution alone does not
make unordered SQL ordered.

The relational `DOUBLE` `sum` / `avg`
[reduction bound](05-the-comparison-contract.md#floating-point-comparisons) is one such
family contract and settles no other family. `string_agg`, `list`, variance, and
standard deviation have no contract, so each is refused.
