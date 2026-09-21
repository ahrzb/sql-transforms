# Ordering and nondeterminism

## Decision rule

**claim: nondeterminism-axiom.** An exact answer must be a function of the query and its
declared inputs. A result that also depends on hidden table history, an unselected
evaluation path, or an uncontrolled run is not an exact target.

Two ruled exceptions are narrower than a general nondeterminism allowance:

- confit defines serving-row order even where DuckDB's join-match sequence is
  unspecified; and
- future per-row `sum` / `avg` over matched `DOUBLE` values uses the adopted
  [float-reduction bound](05-the-comparison-contract.md#floating-point-comparisons)
  rather than exact bits.


*Evidence:* `packages/confit/docs/known-limitations.md:39`, claim:
optimizer-on-reading, and the fold-retirement decision.
*Gap:* `packages/confit/confit/oracle.py` still carries an older summary; proposed
**ticket: axiom-as-property** would also record this rule as P21.

**claim: disposition-table.** Apply the narrowest ruled action. Do not convert a source
of variation into a general tolerance.

| variation source | required action | authority |
|---|---|---|
| request-batch dependence, or no request table | refuse construction | serving-contract scope; fold-retirement decision |
| unspecified DuckDB row sequence | compare DuckDB as a multiset; enforce confit's order through self-legs | claim: compare-modes; claim: serving-row-order |
| hidden table statistics | remove the dependency where possible; otherwise exclude only the measured source | claim: optimizer-on-reading; claim: statistics-dependent-exclusion |
| two DuckDB evaluation paths disagree and the identity selects neither | refuse the construct by name | claim: evaluation-path-disagreement |
| scalar `cbrt` differs across DuckDB builds | allow at most 1 ulp for that family | claim: cbrt-ulp-tolerance, in force |
| future per-row `sum` / `avg` reduces matched `DOUBLE` rows | apply the addend-dependent bound after its open domain is resolved | claim: float-reduction-bound, adopted but unimplemented |
| another order-sensitive value family | no disposition yet | ask: threads-and-value-order |

The optimizer-on bracket is diagnostic, not an accepted alternative answer. A case that
only matches optimizer-off is still reported against the ordinary DuckDB surface as
`DIVERGE_OPT`.

**claim: target-status-vocabulary.** **[PROPOSED]** Require every target to be
`PINNED` (stable contracted behavior), `IMPL-DEFINED` (stable for a named build or
configuration), or `UNSPECIFIED` (no unique contracted answer). Neither the mandatory
classification nor the proposed ledger assignments is adopted. Decide whether to
classify all existing claims or limit the scheme to the ledger under
**ask: proposed-rules-adoption** in the [decision index](12-ask-index.md).

## Row ordering

**claim: serving-row-order.** Serving preserves request-row order:

- `map`: output row `i` corresponds to input row `i`;
- `filter`: output is an input-order subsequence; and
- `many`: each input row's matches are contiguous, with input-row blocks in input order.

This is a confit contract, not a claim about DuckDB's output sequence.

*Enforced-by:* `fuzz.oracle._extra_legs`, using ordered batch-versus-single and input
reversal checks.
*Evidence:* `packages/confit/tests/test_fuzz_order_legs.py` and
`packages/confit/docs/known-limitations.md:41-51`.

**claim: join-output-order.** DuckDB hash-join match order is not a stable sequence. It
varies with streamed-side selection, LIFO duplicate-key chains in 2048-row passes, and,
at roughly 500k rows or more, thread scheduling. DuckDB parity for `shape="many"` is
therefore a multiset. Confit still promises the deterministic serving order above: probe
rows follow input order and each probe row's matches follow build insertion order.

*Evidence:* `packages/confit/docs/specs/pins-stageB/order-contract.json`,
`packages/confit/docs/specs/2026-07-28-stageB-multiplicity-pins.md`, and
`packages/confit/docs/reports/pins-first-methodology.md:35`.

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
uses NUL-transparent row-local behavior.

The corpus set contains one source file and two statements, measured 2026-08-25. Older
text saying “two sources” is incorrect; see **ticket: exclusion-count-correction**.

*Evidence:* `packages/confit/tests/test_corpus_replay.py:38-49`,
`packages/confit/docs/specs/pins-wave1/pins_like.json`, and
`packages/confit/docs/known-limitations.md:225-230`.

**claim: corpus-exclusion-sets.** **[FACT]** The corpus gate currently has three
mechanisms, with different reasons:

1. `_KNOWN_DIVERGENT_SOURCES`: the one statistics-dependent source above, covering two
   statements;
2. a blanket input-`FLOAT` rule, because widening f32 to f64 preserves a value but
   changes the grid observed by `nextafter`, shortest-round-trip text, and rounding;
   Wave 3 measured three sources and five cases; and
3. `_INEXPRESSIBLE_INPUTS`: one declared schema the test surface could not express. Its
   original width-less-pydantic explanation may be stale and must be reviewed rather
   than silently removed.

*Evidence:* `packages/confit/tests/test_corpus_replay.py:38-65,103-110`.

**claim: evaluation-path-disagreement.** If two DuckDB evaluation paths in the same
build return different values and the reference identity selects neither path, refuse
the construct by name. Measured examples are anchor-only multi-anchor regexes and `$`
in non-final position: row evaluation may literal-optimize to PREFIX while constant
evaluation performs a normal regex match.

This differs from claim: native-tables, where the identity selects one input path, and
from claim: cbrt-ulp-tolerance, where separate builds differ within a named bound.

*Evidence:* `packages/confit/docs/reports/pins-first-methodology.md:78,87`,
`packages/confit/docs/specs/pins-waveB/fuzzer-20260728.json`, and
`packages/confit/docs/known-limitations.md:200`.

**claim: native-tables.** Oracle inputs are native DuckDB tables, not registered Arrow
relations. DuckDB can push constant filters into a registered-Arrow scan using IEEE NaN
semantics that differ from native-table ordering. `Oracle.load` therefore registers the
Arrow data, creates a native table, and unregisters the relation. Column widths survive;
`NOT NULL` does not, so a constraint-sensitive fixture declares its table in SQL.

*Enforced-by:* `confit.oracle.Oracle.load` and `.table`.
*Evidence:* `packages/confit/tests/test_oracle.py::test_load_materializes_a_native_table_with_widths_intact`,
`::test_load_unregisters_its_alias`, and
`::test_table_keeps_the_declaration_including_not_null`. The semantic measurement is in
`packages/confit/docs/specs/2026-07-26-stretch4-builtin-pins.md:114-117`; no
registered-relation NaN regression gate was identified.

## Thread count and order-sensitive values

**claim: threads-setting.** **[FACT]** `Oracle.__init__` does not set `threads`; DuckDB's
default is machine-derived. On 2026-08-25, DuckDB 1.5.5 with the optimizer disabled used
12 threads on the measured machine, and `string_agg` element order over 400k rows differed
between thread settings. The independent fit/serving path uses `threads = 1`.

The 2026-09-21 fold decision did **not** adopt `threads = 1` for the retired fold or as
part of the fixed oracle identity. That does not permanently reject a future identity
change motivated by a retained order-sensitive family. Any such setting would apply to
the oracle as a whole: it is not a caller, campaign, or per-case choice. The other
available directions are a named family-specific comparison or refusal. No direction is
adopted for the remaining families.

*Evidence:* the dated measurement above, P11 in
`packages/confit/docs/properties.md:118-121`, `confit.oracle.Oracle.__init__`, and the
fold-retirement decision.

> ### ask: threads-and-value-order — what governs a future order-sensitive family?
>
> The frozen-fold question is superseded, and `threads = 1` was not adopted as part of
> the fixed oracle identity. For a future retained family such as `string_agg`, `list`,
> variance, or another order-sensitive reduction, choose among:
>
> 1. amend the fixed oracle identity by pinning the necessary setting for every
>    comparison, never per caller or case;
> 2. adopt a named, family-specific bound or comparison relation; or
> 3. refuse the family.
>
> The adopted `DOUBLE` `sum` / `avg` bound is already a separate ruling and does not
> settle other families. See the [decision index](12-ask-index.md).
