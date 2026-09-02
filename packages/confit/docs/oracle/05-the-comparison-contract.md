## 5. The comparison contract

This is what "same as the oracle" means, mechanically. It has been the strictest
contract in the project from the start. It now has exactly one implementation —
`confit.compare` — which the tests, the corpus gates and the campaign all import, so the
question "is this leg as strict as that one?" has stopped being answerable by reading
two call sites.

**ORC-90.** Two answers are equal when their canonical forms are equal, and a caller
declares exactly **one** axis: whether row order is part of the claim. What makes two
*values* the same is not an option — `repr` is the contract everywhere, because the
project's claim is bit-exactness and `repr` is stricter than `==` in precisely the places
bit-exactness differs from arithmetic agreement:

| the case | `==` says | `repr` says | why the strict reading is the right one |
|---|---|---|---|
| `NaN` against itself | not equal | equal | a bit-exact answer returned the same NaN |
| `-0.0` against `0.0` | equal | not equal | the bits differ, and so does `1/x` |
| `1`, `1.0`, `True` | all equal | all distinct | the output *type* is half of what is being checked |
| `Decimal('0.50')` against `Decimal('0.5')` | equal | not equal | the scale is exactly what a decimal contract owes |

A leg that wants a weaker check does not get one by omission — the strength of a leg is
declared, not inferred from a `sorted()` that somebody did or did not write.
*Enforced-by:* `confit.compare.sequence` (order-preserving) and `confit.compare.multiset`
(order-insensitive), through `confit.compare.assert_rows(ordered=...)`;
`confit.compare.rows` is the duplicate-name-safe read that feeds them.
*Verified-by:* `packages/confit/tests/test_compare.py::test_multiset_makes_nan_self_equal`,
`::test_multiset_keeps_signed_zero_distinct`,
`::test_multiset_keeps_equal_but_differently_typed_values_distinct`,
`::test_multiset_is_order_insensitive_where_sequence_is_not`, and
`::test_sequence_is_byte_identical_to_the_fuzzers_canonical_form` (the campaign and the
tests canonicalize identically, because they call the same function).
*Scope:* the module raises plain `AssertionError` and imports stdlib only at run time, so
a campaign running outside pytest uses the same definition of equal that the suite does —
`packages/confit/tests/test_compare.py::test_compare_imports_stdlib_and_pyarrow_only`.

**ORC-32.** Floats compare by **bit pattern**, and the pin records the bits, not a
rendering. No rounding, no `%.3f`. **Three exceptions are in force**, all named and all
narrow; there are no others, and a fourth would be a decision (ORC-92):

| exception | bound | why |
|---|---|---|
| `cbrt` (ORC-76) | `<= 1` ulp | the oracle's own wheels disagree by one ulp across platforms; repr-exact is unpinnable |
| the fuzzer's sklearn second-ground-truth leg | `1e-9` absolute | sklearn is a second reference, not the oracle; the leg exists where the oracle abstains (ORC-69) |
| matvec-tier parity (DRAFT-23, when native families land) | a declared per-family ulp bound | and the governance rule with it: "Loosening a control is a design decision, never a fix ... the new bound named — through review, not through a failing test" |

One mechanical limit is worth naming beside them, because it is *not* an exception — it
is a place where the contract says "bits" and the gate compares something else. The
canonical form is `repr`, which makes every NaN self-equal and its **sign and payload
invisible**; only the explicit bit pins see those.
*Enforced-by:* `confit.compare.multiset` / `.sequence` for the repr form;
`confit.oracle.Oracle.answer`, which normalizes nothing at all, so a comparison site sees
the oracle's own bits and decides its own equality.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-235` (explicit bit
pinning, since `repr` collapses every NaN to `nan`);
`packages/confit/tests/test_duckdb_interpreter.py`, `duck_check_ulp` (`max_ulp=1`, the
cbrt bound); `fuzz.oracle._extra_legs` (the sklearn `1e-9` leg);
`packages/confit/tests/test_oracle.py::test_answer_returns_arrow_unnormalized`;
`packages/confit/docs/kpis.md:17-20` (the loosening rule) and `:85-89` (DRAFT-23's
declared per-family ulp bound). Float bit patterns are a recorded field across the pins
corpus.

**ORC-33.** `-0.0` is distinguished from `+0.0`, and this is **fixed, not tolerated**.
Unary minus was lowered as `0 - x`; IEEE `0.0 - 0.0` is `+0.0`, so the sign vanished
everywhere it could arise — 113 of 963 findings in the 2026-08-11 campaign. The fix
subtracts from `-0.0` for FLOAT operands (exact IEEE negation for every double) and
keeps `0 - x` with its `i64::MIN` trap on the integer path, matching DuckDB.
*Enforced-by:* `confit.compare.multiset` is what makes the difference visible to a gate
at all — under `==` the class would be invisible.
*Verified-by:* `packages/confit/tests/known_divergences/test_literal_typing.py:133-165`
(a passing regression pin over both backends);
`packages/confit/tests/test_compare.py::test_multiset_keeps_signed_zero_distinct`;
`backlog/tasks/task-80 ...md:46` (the 113/963 measurement).
*Scope, and it is narrower than it reads.* All five parametrizations of the pin use the
**`e0` (DOUBLE) spelling** — `-0.0e0`, a DOUBLE column, `-1.5e0`. A bare `-0.0` is
`DECIMAL(2,1)` in DuckDB, and a decimal zero has no sign, so `SELECT (c * (- 0.0))`
answers `0.0` on DuckDB and `-0.0` here. That is not a regression of the fixed class —
it is the **opposite direction**: the fix now *keeps* a sign DuckDB's decimal path
discards, and it is the D7 literal-typing mechanism wearing a third face.

**ORC-34.** `%`-by-zero produces a NaN whose **sign bit is platform-libm** (`7ff8...`
on Windows ucrt, `fff8...` on Linux glibc), so the pin is *engine == oracle bit
agreement per platform*, not a constant. `fmod`'s NaN, by contrast, comes from hardware
arithmetic and is `fff8...` on every x86 platform, so it is pinned as a constant.
Status: `IMPL-DEFINED`, platform is the discriminator.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-235` (the
engine == oracle assertion is `assert bits(got["m"]) == bits(m)` at `:235`);
`packages/confit/docs/specs/pins-wave3/math_tail.json` (the wave-3 correction);
`packages/confit/docs/known-limitations.md:258-259`.

**ORC-81.** Platform is a discriminator for **every libm-backed function**, not only for
`%`-by-zero. The wave-1 trig pins say so in-file — "the oracle is platform-libm-dependent,
so cross-OS bit-identity of pins must be re-verified on the CI/serving platform" — and the
`pow10` modifier behind `floor`/`ceil`/`trunc`/`round` is the DuckDB binary's own
`std::pow`, "neither correctly-rounded nor ucrt pow", which "must be extracted from the
oracle binary" and is named in-file as a cross-platform divergence hazard for CI. Those
pins record a single value today while being per-platform in substance; if ORC-35 or
ORC-63 lands they are the next `IMPL-DEFINED` candidates.
*Verified-by:* `packages/confit/docs/specs/pins-wave1/pins_trig-sin-x-cos-x-tan-x-p.json`
(the platform-libm note); `pins-wave1/pins_floor-ceil-trunc-round.json` (the `pow10`
extraction rule).

**ORC-82.** Data tables that stand in for oracle behavior are **extracted from the
oracle**, never from a host library. `strip_accents`' per-codepoint map, the case map and
the `pow10` table are all generated by querying DuckDB; DuckDB's own Unicode tables lag
Unicode 16 by 57 codepoints, so a host `unicodedata` would be a different oracle wearing
the same name. This is ORC-10 (reproduce the quirk) at the level of data rather than
behavior, and it is the strongest form of pins-first in the repo.
*Verified-by:* `packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md:129-142`;
`scripts/gen_strip_accents.py`, `scripts/gen_casemap.py`, `scripts/gen_pow10.py`.

**ORC-35.** **[PROPOSED]** Not in force. The multi-answer rule: a set of accepted answers
is legitimate **only when every member is acceptable in every context**, and every
set-valued pin names the predicate that selects a member (platform, profile, oracle
version). Shortest-diff or best-match selection is never legitimate: it makes the suite
pick the answer that happens to be closest, which is indistinguishable from picking the
answer that hides the bug. ORC-34 is the nearest existing practice — platform is a real
discriminator, evaluable before the comparison runs — though ORC-34 is strictly a *pinned
agreement relation* (engine == oracle, per platform) rather than an enumerated set of
accepted answers, so no pin in force today is set-valued in the sense this rule
describes. ORC-76's bounded tolerance is a second shape the rule as written does not
cover.
*Verified-by:* Unverified — no decision outside this document states the rule. The
survey it comes from is external. Part of ASK-15.

**ORC-36.** **RETIRED.** It said "rows compare as a multiset except where a compare mode
says otherwise", which was the mode table (ORC-18) plus a default. With one comparison
vocabulary shipped, order is a declared axis rather than a property of each call site,
and the claim has no content of its own. Replaced by **ORC-90** (what equal means, and
the one axis) and **ORC-18** (which mode a query is owed).

**ORC-37.** Duplicate output names are normalized through the **same rule on both
sides** before comparison, which is what makes the rename a contract rather than a
divergence: duplicates rename left-to-right to `<name>_N`, smallest free N,
case-insensitive, generated candidates included. DuckDB itself applies exactly this at
every subquery / CTE / CTAS boundary and in `.df()`; only its top-level arrow export
keeps duplicates, so the oracle leg is renamed before comparing.
*Enforced-by:* `confit.compare.dedup_names`, which mirrors
`packages/confit/src/specializer/frontend.rs`'s `dedup_output_names` and is the single
Python home — the campaign imports it rather than carrying a copy.
`confit.compare.rows` applies it on the read, where a duplicate name would otherwise cost
half the answer with no error: `to_pylist()` builds one dict per row, so two columns
named `a` collapse to one key, last one wins.
*Verified-by:* `packages/confit/tests/test_compare.py::test_dedup_names_renames_left_to_right_case_insensitively`,
`::test_dedup_names_skips_a_generated_candidate_already_taken`,
`::test_rows_keeps_both_columns_that_to_pylist_collapses`;
`packages/confit/docs/specs/pins-wave5/dup-names-client-contract.json`;
`packages/confit/tests/test_known_limitations.py:255` (the twin).

**ORC-38.** Output **schemas** are compared, not just values: a name mismatch or a type
mismatch is a `DIVERGE_VALUE` in its own right, and a comparison that only looked at
values would pass a query whose answer has the wrong type. The single exception is an
enumerated unshipped-feature width, which is classified rather than compared — ORC-92.
*Enforced-by:* `fuzz.oracle._schema_delta` and `fuzz.oracle._type_delta` (which recurse
into structs, so a decimal lane inside `struct_pack` classifies too); a real difference
anywhere outranks an unshipped width sitting in another column.
`confit.compare.assert_schema` is the same check for a test, and it names the first
differing field and the attribute that differs on it rather than printing two schema
dumps.
*Verified-by:* `packages/confit/tests/test_fuzz_smoke.py::test_a_real_schema_difference_is_still_a_divergence`;
`packages/confit/tests/test_compare.py::test_assert_schema_names_the_first_differing_field_and_attribute`.

**ORC-92.** An unshipped feature **fails or is classified — never absorbed by weakening a
comparison.** Where the engine has not shipped a width the oracle emits, there is no
honest value comparison across the gap, so the case gets its own verdict (`UNSHIPPED`,
carrying the class and the lane that differs) and **no value comparison happens at all**.
Such a case is not agreement, is not a finding, and never enters `findings.jsonl`. The
general rule it instances: a deviation from raw equality happens only through a **named
bound in a reviewed draft** — the precedents are ORC-76's cbrt ulp, the sklearn leg's
`1e-9` and DRAFT-23's declared bounds (ORC-32's table) — never through a cast, a
normalization or a tolerance introduced inside a comparison harness.

Two consequences worth stating, because both were live questions before the ruling. An
`UNSHIPPED` verdict **outranks the optimizer bracket** (ORC-24): neither reading was
value-compared, so neither is evidence about a plan-rewrite pass. And it still earns the
boundary self-legs, which are ours-against-ours with no DuckDB in them — an unshipped
width cannot excuse a self-inconsistency, so a real `DIVERGE_VALUE` there outranks the
class.
*Enforced-by:* `fuzz.oracle.run_case` (the `UNSHIPPED` exit and its ranking) and
`fuzz.oracle._type_delta`, which carries one arm per unshipped feature — today exactly
one, decimal-against-float64, deleted when the feature lands.
*Verified-by:* `packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`
and `::test_a_real_schema_difference_is_still_a_divergence` (only the named classes take
the exit).

> ### ASK-12 — RULED. Is the comparison harness's own normalization part of the
> oracle's answer?
>
> **Ruling: no.** Normalization is out of the oracle's answer and out of the verdict. An
> unshipped feature FAILS or is CLASSIFIED (`UNSHIPPED`, or an xfail), never absorbed by
> weakening a comparison; deviations from raw equality happen only via a **named bound in
> a reviewed draft**, on the precedent of the cbrt 1-ulp, the sklearn `1e-9` and
> DRAFT-23's bounds. The 1-ulp decimal deltas reported in past campaigns were artifacts
> of the harness's own cast, **manufactured by neither engine**.
>
> *Implemented by:* the decimal-to-float64 cast is **deleted** from `fuzz/oracle.py`; the
> `UNSHIPPED` verdict kind replaces it, with its own report section in `fuzz/runner.py`.
> The rule is ORC-92; `confit.oracle.Oracle.answer` normalizes nothing on the oracle's
> side either.
>
> *Consequences already applied:* ORC-38 no longer authorizes a cast; ledger row D12's
> value deltas are resolved (section 7.3); ORC-68's normalization blind spot is replaced
> by the honest one, which is that an unshipped width leaves the values unchecked.

**ORC-39.** Error **texts** are not compared. Runtime traps reproduce DuckDB's message
bodies verbatim; some bind-time rejections use our own wording with the same error
class. The corpus compares successful results only, so texts never affect parity. Error
text is therefore *not oracle-decided output* — which is also a named blind spot
(ORC-57). Upstream does the same thing: DuckDB's own test infrastructure matches error
text by substring containment.
*Enforced-by:* `confit.oracle.Trap`, which carries the exception's class name and message
as two separate fields, so a caller can compare the class without touching the text.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:173-176` (only successful rows
compared); `packages/confit/docs/known-limitations.md:219-224`.

**ORC-40.** Backend agreement is settled **before either reading is compared against**:
cranelift vs interpreter is a question about us, not about the oracle, so it is checked
once and short-circuits. A split there carries **its own `klass`** — `backend-split`,
`backend-values`, `backend-trap-split` — so it is never confused with a DuckDB
disagreement when reading a finding. Its *kind* is still a divergence kind
(`DIVERGE_BUILD` or `DIVERGE_VALUE`), which is what `findings.jsonl` and the ledger
census key on, so a backend split does count into those totals.
*Enforced-by:* `fuzz.oracle.run_case` (the backend checks, upstream of `against`).
*Verified-by:* P19 in `packages/confit/docs/properties.md:240-245`.
*Precision:* settled before either reading is *compared*, not before either is *executed*
— `_duck_run` runs first, so both readings exist by the time the backend check fires.

**ORC-83.** The interpreter is the **internal oracle backend** for the engine's own
two-backend differential: correctness and coverage over speed, never optimized, and
cranelift is checked against it byte-for-byte over a 500-seed random-IR sweep. It is why
ORC-40 can settle backend agreement without DuckDB at all.
*Verified-by:* P19 in `packages/confit/docs/properties.md:240-245`;
`packages/confit/docs/kpis.md:62-64` (the 500-seed random-IR sub-invariant, in
`packages/confit/src/specializer/exec/tests.rs`);
`backlog/tasks/task-42 - Specializer-M-interp-closure-compiled-IR-interpreter-the-oracle-backend.md`.

**ORC-41.** Mechanisms other cross-engine suites use that **this project has never
adopted**, and the reason each one is a bad fit here. This is a survey with an argument,
not a list of past decisions: only the first row corresponds to a rule in force (ORC-32),
and none of the other four was ever proposed here, so none was ever rejected here.
Turning the four into standing rejections would itself be a decision — that is ORC-84.

| mechanism | where it comes from | why it is a bad fit here |
|---|---|---|
| float rendering at `%.3f` | sqllogictest's cross-engine rendering contract | directly destroys bit-for-bit, which **is** in force (ORC-32); it trades float fidelity for cross-engine agreement, and we have exactly one engine to agree with |
| MD5 hashing result streams above a threshold | sqllogictest | hashes make a failure undebuggable; pins-as-data (reprs, float bits, verbatim error heads, the exact SQL) is strictly better evidence, and DuckDB's own docs advise using the hash form sparingly |
| shortest-diff variant matching | Postgres's `resultmap` driver, which admits it "cannot tell which variant is actually correct" | picks the closest answer, which is the answer most likely to hide the defect; ORC-35 is the proposed replacement |
| cross-engine agreement or majority vote as the oracle | sqllogictest, Csmith | creates a second authority; the contract delegates to exactly one engine on purpose (ORC-01). Note the dialect gates (ORC-73) *do* run a second engine — as a target for a printed query, never as an authority over DuckDB |
| a growing expected-errors allowlist | SQLancer's `ExpectedErrors` | every entry added to silence a false positive is a place a real bug can hide; structurally the same shape as ORC-30. The nearest thing we have is `_CLEAN` (ORC-29), which is four entries and has not grown |

*Verified-by:* ORC-32 for the first row. The other four are `Unverified` as decisions —
no spec, ticket or review in this repo records adopting or rejecting them (searched
2026-08-25). They are here as an argument, and the argument is what ORC-84 asks about.

**ORC-84.** **[PROPOSED]** Not in force. The four unadopted mechanisms in ORC-41 become
**standing rejections**, so that proposing one later is a contradiction of the spec
rather than a fresh idea. The cost of adopting this is real: a standing rejection of
"tolerance" has to be written so that it does not contradict the three tolerances
already in force (ORC-32's exception table) or the dialect gate's designed epsilon tier
(ORC-73). ORC-92 is the half of it that is now decided — a deviation needs a named bound
in a reviewed draft — so what ORC-84 adds is the *rejection* of the four named shapes,
not the requirement that deviations be named.
*Verified-by:* Unverified. Part of ASK-15, and the substance is ASK-6.

> ### ASK-6 — is bit-for-bit float equality the contract, and what governs its exceptions?
>
> Every cross-engine suite surveyed quietly relaxes floats. The question is *not* whether
> this project has exceptions — it has three, all named in ORC-32's table — but whether
> the rule is "bit pattern, with a closed list of declared bounds" and what it takes to
> add a fourth entry to that list.
>
> **(a) The rule.** Is it "bit pattern, no exceptions" — in which case ORC-76's cbrt
> tolerance, the sklearn leg's `1e-9` and DRAFT-23's declared bound are three
> contradictions that need re-ruling — or is it "bit pattern, with a closed list of
> declared bounds, each naming its discriminator", in which case ORC-32's table *is* the
> list and adding to it is a decision like loosening any other control? ORC-92 has ruled
> the *procedure* for a fourth entry (a named bound, through review); what is still open
> is whether the closed list is the rule.
>
> **(b) Future float accumulation.** If parallel float accumulation ever lands, is its
> `UNSPECIFIED` region **refused** (the strict reading) or given a declared bound the way
> the dialect gate's epsilon tier already is (ORC-73)?
>
> **(c) The unenforced third home.** ORC-80's feature-in-flight rule wants a phase's
> markers deleted in three homes, and the fuzzer's was the unenforced one. It has since
> **changed shape rather than been enforced**: the marker is no longer a suppression tag
> that hides a comparison — it is the `UNSHIPPED` verdict, which classifies loudly and
> gets its own report section (ORC-92), so a tag outliving its phase now shows as a
> non-empty bucket instead of swallowing a regression silently. What is still not in place
> is a *gate*: nothing fails when `_type_delta`'s decimal arm outlives the feature. Does
> that arm get a strict-xfail twin now, or stay owned by the lattice phase's own
> definition of done, on the record?
>
> *Verified-by (the facts, not the ruling):*
> `packages/confit/docs/known-limitations.md:166-174`; `fuzz.oracle._type_delta` (the one
> arm, with its delete-when-it-ships note); `packages/confit/tests/test_decimals.py` (the
> shipped decimal *static* path, whose expectations are the live oracle compared on rows
> and on schema through `confit.compare`);
> `packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131` (the
> feature-in-flight rule and the three homes).
>
> *Binds:* ORC-32, ORC-76, ORC-80, ORC-84, ORC-92, ledger rows D7 and D8, and every future
> float-accumulation feature.

---
