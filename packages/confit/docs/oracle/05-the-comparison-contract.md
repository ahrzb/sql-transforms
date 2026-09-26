# Comparison contract

Exact agreement is the default. This chapter defines the row, schema, numeric, and error
comparisons that make that statement operational. A weaker comparison applies only to
the named family and surface that adopted it.

Settled comparison policy comes from the
[oracle policy record](../decisions/closed/oracle-policy.md).

For UDF-bearing SQL, the reference leg registers the same declared UDFs. That
parameterization is defined by the [oracle](README.md) and
[serving contract](../specs/serving-contract.md#udf-and-model-boundary); it is not a
numeric tolerance or a second oracle.

## Rows and values

**claim: repr-equality.** `confit.compare` is the shared row-comparison vocabulary for
tests and the differential campaign. Values are canonicalized with `repr`; a caller then
chooses sequence comparison or multiset comparison.

| values | Python `==` | canonical `repr` comparison |
|---|---|---|
| `NaN`, itself | false | equal |
| `-0.0`, `0.0` | equal | different |
| `1`, `1.0`, `True` | equal | different |
| `Decimal('0.50')`, `Decimal('0.5')` | equal | different |

This is exact canonical-value comparison, not universal bit comparison. Python renders
NaNs as `nan`, hiding payload and sign. Generic row equality therefore distinguishes
signed zero and type-sensitive spellings but cannot establish NaN payload/sign identity.
That limitation is not an accepted NaN tolerance; bit-sensitive claims require explicit
bit pins.

*Enforced-by:* `confit.compare.sequence`, `.multiset`, `.rows`, and
`.assert_rows(ordered=...)`.
*Evidence:* `packages/confit/tests/test_compare.py` covers NaN self-equality,
signed-zero and type distinction, order sensitivity, and canonical-form parity with the
fuzzer. The module uses only stdlib at runtime and raises plain `AssertionError`.

Ordering is selected by [claim: compare-modes](03-nondeterminism.md): the DuckDB row leg
is a multiset, while confit's serving sequence is checked by self-legs. The mined corpus
is a separate positional vocabulary: `test_corpus_replay.py` normalizes each row as
`tuple(repr(v) for v in row)` and sorts rows. It is positional rather than name-keyed.

## Output names and schemas

**claim: duplicate-name-dedup.** Duplicate output names are renamed on both sides,
left-to-right, to `<name>_N` using the smallest case-insensitively free `N`, including
names generated earlier in the same pass. DuckDB performs such renaming at
subquery/CTE/CTAS boundaries and in `.df()`, while top-level Arrow export preserves
duplicates. Renaming before `to_pylist()` prevents a dict row from retaining only the
last duplicate column.

*Enforced-by:* `confit.compare.dedup_names` and `.rows`, mirroring
`specializer/frontend.rs::dedup_output_names`; the fuzzer imports the same helper.
*Evidence:* duplicate-name tests in `packages/confit/tests/test_compare.py`,
`packages/confit/docs/specs/pins-wave5/dup-names-client-contract.json`, and
`packages/confit/tests/test_known_limitations.py::test_duplicate_names_use_duckdbs_boundary_rename`.

**claim: schema-comparison.** Output names, Arrow types, and field order are part of
the [serving output contract](../specs/serving-contract.md#api-and-output-shape).
Compare schemas, not just values:

- `confit.compare.assert_schema` compares field count, names, and types;
- campaign `_schema_delta` compares deduplicated names and recursively compares types;
- a campaign name or non-exempt type difference is `DIVERGE_VALUE` with class `schema`.

Output names, Arrow types, and field order must match exactly. Nullability
metadata must be truthful rather than identical to DuckDB's flags: a non-null promise
must be sound, while conservative nullable metadata need not reproduce another engine's
inference. Both checkers therefore compare types through `confit.compare.same_type`,
which ignores nullability at every depth, and soundness is checked against confit's own
rows: `confit.compare.non_null_violation` names the first row and field path (struct
child, list item, map value) holding a NULL under a non-null promise. The campaign
reports such a case as `DIVERGE_VALUE` with class `unsound-non-null`; tests assert it
with `assert_nullability_sound`.

*Evidence:* `confit.compare.assert_schema`, `same_type`, and `non_null_violation`;
`fuzz.oracle._schema_delta` and `_type_delta`;
`packages/confit/tests/test_compare.py::test_a_broken_non_null_promise_is_named`;
`packages/confit/tests/test_fuzz_smoke.py::test_a_broken_non_null_promise_is_a_divergence`;
`packages/confit/tests/test_fuzz_smoke.py::test_a_real_schema_difference_is_still_a_divergence`;
and `packages/confit/tests/test_compare.py::test_assert_schema_names_the_first_differing_field_and_attribute`.

**claim: unshipped-verdict.** An enumerated unshipped width is classified, never cast
into comparability. `UNSHIPPED` names the feature and lane, performs no value comparison,
does not enter `findings.jsonl`, and does not count as agreement. A real name or type
difference in another field outranks the exempt width. Confit-only boundary self-legs
still run because an unshipped DuckDB width cannot excuse internal inconsistency.

`UNSHIPPED` also outranks the optimizer bracket: without a value comparison, neither
DuckDB reading says anything about optimizer rewriting. `_type_delta` has one
unshipped arm: DuckDB decimal versus confit float64.

*Enforced-by:* `fuzz.oracle.run_case`, `_schema_delta`, and `_type_delta`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`
and `::test_a_real_schema_difference_is_still_a_divergence`.

**claim: unshipped-reach.** An empty `UNSHIPPED` bucket is not evidence of support, so
the campaign reports each unshipped feature's generator reach beside it. Every verdict
line is tagged `reaches:<feature>` when its SQL contains the construct that yields the
width (for `decimals`, a bare decimal literal such as `2.5`, string literals excluded),
and the report prints, per feature, how many cases reached it and their verdicts, or
says the feature was not reached. Reach is construct presence, not output width: a
decimal literal in a `WHERE` clause reaches the feature and may still `AGREE`.

*Enforced-by:* `fuzz.oracle.unshipped_reach`, `fuzz.oracle.run_case_json`, and
`fuzz.runner.report`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_unshipped_reach_is_read_off_the_construct_not_the_bucket`,
`packages/confit/tests/test_fuzz_report.py::test_unshipped_widths_are_reported_with_their_reach`,
and `::test_an_unreached_width_is_not_reported_as_support`.

Harness normalization never manufactures agreement, and any weaker comparison requires
its own named, reviewed bound. Coverage must show that an unshipped width is classified
rather than counted as agreement; the tests above do, and no duplicate strict-xfail test
is kept as bookkeeping. A strict xfail is the tool for a concrete defect whose repair
should expire the exception.

## Floating-point comparisons

**claim: float-bit-equality.** Where a float property is bit-sensitive, exact pins record
bit patterns. Do not substitute rounding, `%.3f`, arithmetic `==`, or generic `repr` when
NaN sign or payload matters. The generic comparator's NaN collapse is an enforcement
limit, not a tolerance.

Only explicitly approved exceptions may weaken exact comparison. Each must name its
operation, comparison rule, valid inputs, and rationale; tests do not choose new
tolerances independently. The inventory below distinguishes approved exceptions from
independent references.

| surface and family | comparison | status |
|---|---|---|
| DuckDB scalar `cbrt` | at most 1 ulp | gated; claim: cbrt-ulp-tolerance |
| campaign sklearn second-reference leg | absolute `1e-9` | that self-leg only; sklearn is not DuckDB |
| `sql_transform` transformer references | family-specific bounds; tree scoring has a bit-exact target with a numerical-equality test | independent C4 contract; see [success measures](../specs/success-measures.md#transformer-parity-c4) |
| per-row relational `sum` / `avg` over matched `DOUBLE` values | claim: float-reduction-bound below | adopted; no checker implements it, and confit refuses aggregation |

`confit.compare.assert_rows_close` is positional and ulp-based. It distinguishes signed
zero and treats NaN as self-equal, but at one ulp it can place `DBL_MAX` next to infinity.
Its contract use is the measured `cbrt` wobble, not a general float default.

**claim: cbrt-ulp-tolerance.** Scalar `cbrt` parity permits at most one ulp. The Windows
DuckDB wheel matched Rust/ucrt, while the Linux wheel returned, for example,
`3.0000000000000004` for `cbrt(27)`. This is a measured cross-build scalar allowance.
It is distinct from exact-default comparison, sklearn references, and relational
reduction order.

*Enforced-by:* `assert_rows_close` / `duck_check_ulp(max_ulp=1)` for `cbrt`.
*Evidence:* `packages/confit/tests/test_duckdb_interpreter.py::test_sqrt_cbrt_bigint` and
`packages/confit/docs/specs/2026-07-26-wave1-builtin-pins.md`.

**claim: float-reduction-bound.** The oracle policy adopts an exception for
order-dependent `sum` and `avg` over the relation of `DOUBLE` values matched by one
request row. For `n` non-NULL addends $v_i$ and $\mathrm{eps}=2^{-52}$, the declared sum
bound is

$$
\left|\mathrm{ours}-\mathrm{oracle}\right|
\le 2(n-1)\,\mathrm{eps}\sum_i |v_i|.
$$

Equivalently, in checker notation: `2*(n-1)*eps*sum(abs(v_i))`, with
`eps = 2^-52`.

For `avg`, divide that sum bound by `n`. The bound is data-dependent because cancellation
makes a fixed relative tolerance unsuitable.

It applies only to per-row relational `sum` / `avg` over matched `DOUBLE` rows. It does not cover scalar
arithmetic, casts, comparisons, integer or decimal aggregates, ordered list-value
reductions, variance, standard deviation, or another order-sensitive family. It does not
extend training-round-trip KPI C1, which is bit-exact under its own fixed-thread
comparison.

The bound does not define reduction algorithm and length, non-finite addends/results,
intermediate or `sum(abs(v_i))` overflow and underflow, empty/all-NULL reductions,
`n = 0` and `n = 1`, or the final `avg` division; the formula cannot be applied literally
at `n = 0`, and no general epsilon substitutes for defining these domains. No checker
implements the bound.

*Evidence:* the premise is a measurement: on one `DOUBLE` table, twenty multi-threaded
DuckDB runs produced twenty distinct `sum` and `avg` bit patterns, while a
single-threaded reading matched an in-order loop. That observation is not verification of the bound.

**claim: signed-zero.** `-0.0` and `+0.0` are distinct results. Unary float negation
subtracts from `-0.0`, preserving exact IEEE negation; integer negation retains `0 - x`
and its `i64::MIN` trap.

Regression pins use DOUBLE spellings such as `-0.0e0`. Bare `-0.0` is
`DECIMAL(2,1)` in DuckDB and has no sign; divergence involving that literal belongs to
divergence: decimal-literal-typing, not a relaxation of signed-zero equality.

*Evidence:* `packages/confit/tests/known_divergences/test_literal_typing.py::test_negative_zero_keeps_its_sign`
and `packages/confit/tests/test_compare.py::test_multiset_keeps_signed_zero_distinct`.

## Platform-discriminated exactness

**claim: modulo-nan-sign.** `%` by zero obtains a NaN sign from platform libm:
`7ff8...` on Windows ucrt and `fff8...` on Linux glibc. The pin compares engine bits with
oracle bits on the same platform. `fmod` uses hardware arithmetic and is pinned to
`fff8...` on x86. This is per-platform exact agreement, not a tolerance.

*Evidence:* `packages/confit/tests/test_duckdb_wave3_mathtail.py::test_computed_nan_bits_match_oracle`,
`packages/confit/docs/specs/pins-wave3/math_tail.json`, and
`packages/confit/docs/known-limitations.md` §5.

**claim: platform-libm.** Platform can discriminate every libm-backed function, not only
modulo-by-zero. Trig pins require remeasurement on the CI/serving platform. The `pow10`
table used by `floor` / `ceil` / `trunc` / `round` must be extracted from the DuckDB
binary because its `std::pow` result is neither guaranteed correctly rounded nor equal
to the host Python library's result. Existing pins record one platform.

*Evidence:* `packages/confit/docs/specs/pins-wave1/pins_trig-sin-x-cos-x-tan-x-p.json`
and `packages/confit/docs/specs/pins-wave1/pins_floor-ceil-trunc-round.json`.

**claim: oracle-extracted-tables.** Behavioral tables are extracted from DuckDB, never
recreated from a host library. This applies to `strip_accents`, case mapping, and
`pow10`. DuckDB's Unicode tables lag Unicode 16 by 57 codepoints, so host `unicodedata`
would silently select another reference.

*Evidence:* `packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md`,
`scripts/gen_strip_accents.py`, `scripts/gen_casemap.py`, and `scripts/gen_pow10.py`.

**claim: multi-answer-sets.** An expected answer may be selected by a justified
platform/build predicate known before comparison, within the fixed oracle identity.
Each selected answer must be correct for that case. Choosing the nearest answer after
seeing Confit's output is forbidden. This does not permit per-case changes of oracle
version or configuration. Claim: modulo-nan-sign is an example of platform-selected
exactness; claim: cbrt-ulp-tolerance remains a separate numerical bound, not a choice
among expected answers.

## Errors and internal backends

**claim: error-texts.** Error text is not oracle-compared output. Runtime traps may copy
DuckDB messages, while construction refusals may use confit's wording. The corpus
compares successful rows, not full message identity; DuckDB's own tests use substring
matching.

*Enforced-by:* `confit.oracle.Trap` stores exception class and text separately.
*Evidence:* `packages/confit/tests/test_corpus_replay.py` (`_replay`) and
`packages/confit/docs/known-limitations.md` §5.

**claim: backend-agreement.** Cranelift and interpreter agreement is settled before
either result is compared with DuckDB. `backend-split`, `backend-values`, and
`backend-trap-split` identify the failure. Their public verdict remains
`DIVERGE_BUILD` or `DIVERGE_VALUE`, so they enter the same finding totals.

`backend-split` returns before DuckDB executes. The value and trap splits occur after the
two DuckDB readings have executed but before value comparison. “Settled first” refers
to comparison precedence, not always execution order.

*Enforced-by:* `fuzz.oracle.run_case`.
*Evidence:* P19 in `packages/confit/docs/properties.md`.

**claim: interpreter-backend.** The interpreter is confit's coverage-oriented internal
oracle backend. A 500-seed random-IR differential requires Cranelift to agree with it
byte-for-byte. This internal invariant is additional to, not a replacement for, DuckDB
comparison.

*Evidence:* P19, [engine parity C2](../specs/success-measures.md#engine-parity-c2),
and `packages/confit/src/specializer/exec/tests.rs`.

## Rejected shortcuts

**claim: standing-rejections.** Two mechanisms are standing rejections, because each
replaces the contract instead of measuring against it: a nearest or shortest-diff
alternative conceals a mismatch, and a cross-engine majority vote substitutes another
authority for the configured reference. A growing expected-error allowlist is not a
third comparison rule either — an exception enters the approved list above with its
operation, comparison rule, valid inputs, and rationale, or the mismatch stays a
finding.

These rejections leave untouched the `cbrt` bound, the independent sklearn
reference checks, and the float-reduction bound.

**claim: unadopted-mechanisms.** Exact comparison rules out rounded `%.3f`
rendering as equality, because it discards exact float information. Hashing is not
banned: a digest alone is poor evidence, since it hides the values needed to diagnose a
failing pin, but it remains available as an incidental tool beside evidence that does
show those values.
