## 5. The comparison contract

This is what "same as the oracle" means, mechanically. It has been the strictest
contract in the project from the start and has never been written down in one place.

**ORC-32.** Floats compare by **bit pattern**, and the pin records the bits, not a
rendering. No rounding, no `%.3f`. **Three exceptions are in force**, all named and all
narrow; there are no others, and a fourth would be a decision:

| exception | bound | why |
|---|---|---|
| `cbrt` (ORC-76) | `<= 1` ulp | the oracle's own wheels disagree by one ulp across platforms; repr-exact is unpinnable |
| the fuzzer's sklearn second-ground-truth leg | `1e-9` absolute | sklearn is a second reference, not the oracle; the leg exists where the oracle abstains (ORC-69) |
| matvec-tier parity (DRAFT-23, when native families land) | a declared per-family ulp bound | and the governance rule with it: "Loosening a control is a design decision, never a fix ... the new bound named — through review, not through a failing test" |

Two mechanical limits are worth naming beside them, because they are *not* exceptions —
they are places where the contract says "bits" and the gate compares something else.
Both differential gates canonicalize rows through `repr`, which makes every NaN
self-equal and its **sign and payload invisible**; only the explicit bit pins see those.
And the campaign's schema normalization can move a value before the comparison — ORC-38
and ASK-12.
*Verified-by:* `packages/confit/tests/test_duckdb_wave3_mathtail.py:205-235`
(explicit bit pinning, since `repr` collapses every NaN to `nan`); float bit patterns are
a recorded field across the pins corpus;
`packages/confit/tests/test_duckdb_interpreter.py:918-952` (the cbrt ulp bound);
`packages/confit/fuzz/oracle.py:851` (the sklearn `1e-9` leg), `:422-429` (`_key`,
repr-based) and `packages/confit/tests/test_corpus_replay.py:70-72` (`_norm_row`, whose
own comment says repr "makes NaN self-equal");
`packages/confit/docs/kpis.md:17-20` (the loosening rule) and `:85-89` (DRAFT-23's
declared per-family ulp bound).

**ORC-33.** `-0.0` is distinguished from `+0.0`, and this is **fixed, not tolerated**.
Unary minus was lowered as `0 - x`; IEEE `0.0 - 0.0` is `+0.0`, so the sign vanished
everywhere it could arise — 113 of 963 findings in the 2026-08-11 campaign. The fix
subtracts from `-0.0` for FLOAT operands (exact IEEE negation for every double) and
keeps `0 - x` with its `i64::MIN` trap on the integer path, matching DuckDB. It is now
a passing regression pin over both backends.
*Verified-by:* `packages/confit/tests/known_divergences/test_literal_typing.py:133-165`;
`backlog/tasks/task-80 ...md:46` (the 113/963 measurement), `:75` (the class measured
empty after).
*Scope, and it is narrower than it reads.* All five parametrizations of the pin use the
**`e0` (DOUBLE) spelling** — `-0.0e0`, a DOUBLE column, `-1.5e0`. A bare `-0.0` is
`DECIMAL(2,1)` in DuckDB, and a decimal zero has no sign, so `SELECT (c * (- 0.0))`
answers `0.0` on DuckDB and `-0.0` here. That is not a regression of the fixed class —
it is the **opposite direction**: the fix now *keeps* a sign DuckDB's decimal path
discards, and it is the D7 literal-typing mechanism wearing a third face. The
2026-08-17 campaign residual seed 998 is exactly this. Also: the "class measured empty
after" measurement is the **2026-08-13** campaign, one campaign earlier than the
committed baseline.

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

**ORC-36.** Rows compare as a **multiset** except where a compare mode says otherwise
(ORC-18). Row-order claims on the serving path are checked by DuckDB-free self-legs
(ORC-16), never by matching DuckDB's sequence.
*Verified-by:* `packages/confit/fuzz/oracle.py:422-429` (`_key` multiset form, `_seq`
sequence form), `:432-449`.

**ORC-37.** Duplicate output names are normalized through the **same rule on both
sides** before comparison, which is what makes the rename a contract rather than a
divergence: duplicates rename left-to-right to `<name>_N`, smallest free N,
case-insensitive, generated candidates included. DuckDB itself applies exactly this at
every subquery / CTE / CTAS boundary and in `.df()`; only its top-level arrow export
keeps duplicates, so the oracle leg is renamed before comparing.
*Verified-by:* `packages/confit/fuzz/oracle.py:467-488` (`_dedup_names`), `:646-651`
(applied to the DuckDB side);
`packages/confit/docs/specs/pins-wave5/dup-names-client-contract.json`;
`packages/confit/tests/test_known_limitations.py:255` (the twin).

**ORC-38.** Output **schemas** are compared, not just values. A name mismatch or a
type mismatch is a `DIVERGE_VALUE` in its own right; the only exception is the
enumerated unshipped-feature width class — today exactly one arm, decimal-vs-float64 —
which is *tagged* and cast before the value comparison so the tag stays visible in the
report rather than becoming an accepted equality. **The tag suppresses the schema delta
only. The value comparison still runs on the cast result**, and a tagged case that then
differs in value is still reported `DIVERGE_VALUE`; three such findings sit in the
committed baseline.
*Verified-by:* `packages/confit/fuzz/oracle.py:491-529` (`_schema_delta`, `_type_delta`
— the decimal arm at `:512-513` is the only unshipped-feature arm), `:700-710` (tag,
then `cast_to = sch_cl`), `:531-535` (`_norm` applies `table.cast(to)`), `:706-712`
(the value comparison after the cast); `packages/confit/fuzz/runner.py` contains no tag
filter of any kind.
*Measured, and it is the reason for ASK-12:* the cast is not value-preserving. Measured
2026-08-25 under the oracle, `SELECT -14.665` comes back `decimal128(5,3)`; the harness's
`table.cast(float64)` yields `-14.665000000000001`, while DuckDB's own `::DOUBLE` and
Python's `float()` both yield `-14.665` — which is also what confit returns. So for that
shape the normalization step, not either engine, is what produces the reported delta.

> ### ASK-12 — is the comparison harness's own normalization part of the oracle's answer?
>
> ORC-38 authorizes casting DuckDB's leg to our schema before comparing, and does not
> require the cast to preserve values. Measured, it does not: for a decimal literal the
> harness's `pyarrow` `decimal128 -> float64` cast lands one ulp away from the double that
> DuckDB's own `::DOUBLE` produces (measured above; the same shape reproduces for
> `94.579` and `-42.602`). Three of the four `DIVERGE_VALUE` residuals in the committed
> baseline — seeds 869, 1554, 3269, all `decimals`-tagged — are that shape, and they are
> currently reported as rung-2 contract violations.
>
> The **schema** divergence in those cases is real and is ledger row D7. What is at issue
> is only the value delta reported on top of it. Three ways:
>
> - **normalize through the oracle** — ask DuckDB for the cast (`::DOUBLE` in the emitted
>   SQL, or a second reading) rather than casting its output in the harness, so the
>   comparison never invents a value;
> - **compare on the decimal side** — cast *our* f64 up instead of casting DuckDB's
>   decimal down, and accept that this changes what "equal" means for the class;
> - **declare the normalization part of the harness contract** — the cast is what it is,
>   and a tagged case's value comparison is `UNSPECIFIED` until the feature lands.
>
> Not applied here — docs-only, and the choice is yours because it changes what the
> campaign reports. Proposed ticket T-18.
>
> *Binds:* ORC-32, ORC-38, ORC-26, ledger rows D7 and D12.

**ORC-39.** Error **texts** are not compared. Runtime traps reproduce DuckDB's message
bodies verbatim; some bind-time rejections use our own wording with the same error
class. The corpus compares successful results only, so texts never affect parity. Error
text is therefore *not oracle-decided output* — which is also a named blind spot
(ORC-57). Upstream does the same thing: DuckDB's own test infrastructure matches error
text by substring containment.
*Verified-by:* `packages/confit/docs/known-limitations.md:219-224`;
`packages/confit/tests/test_corpus_replay.py:173-176` (only successful rows compared).

**ORC-40.** Backend agreement is settled **before either reading is compared against**:
cranelift vs interpreter is a question about us, not about the oracle, so it is checked
once and short-circuits. A split there carries **its own `klass`** — `backend-split`,
`backend-values`, `backend-trap-split` — so it is never confused with a DuckDB
disagreement when reading a finding. Its *kind* is still a divergence kind
(`DIVERGE_BUILD` or `DIVERGE_VALUE`), which is what `findings.jsonl` and the ledger
census key on, so a backend split does count into those totals; ORC-23's table now says
so on both rows.
*Verified-by:* `packages/confit/fuzz/oracle.py:625-639` (the check and its in-code
comment), `:577-583` (the build-side split); P19 in
`packages/confit/docs/properties.md:240-245`.
*Precision:* the check is settled before either reading is *compared*, not before either
is *executed* — `_duck_run` runs at `:585`, upstream of the backend checks at `:625-639`.
The in-code comment says "before either reading" and means the comparison.

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
(ORC-73).
*Verified-by:* Unverified. Part of ASK-15, and the substance is ASK-6.

> ### ASK-6 — is bit-for-bit float equality the contract, and what governs its exceptions?
>
> Every cross-engine suite surveyed quietly relaxes floats. The question is *not* whether
> this project has exceptions — it has three, all named in ORC-32's table — but whether
> the rule is "bit pattern, with a closed list of declared bounds" and what it takes to
> add a fourth entry to that list.
>
> **What is already decided, so that the ruling is about the open part.** An earlier
> version of this block asked about D7/D8 as if their classification were open. Two of
> its premises were wrong and are corrected here:
>
> - "The fuzzer's `decimals` tag suppresses it" — **it does not.** The tag suppresses the
>   *schema* delta only; the value comparison runs on the cast result and still reports
>   `DIVERGE_VALUE` (ORC-38). Three such findings are in the committed baseline.
> - "D7/D8 have no home" — the **feature-in-flight rule (ORC-80), decided 2026-08-11**,
>   already governs them: anything with an m-8 phase is a feature in flight, not a known
>   divergence, its markers are scaffolding, and each phase's definition of done includes
>   deleting them in all three homes in the feature's own PR. Two of the three homes are
>   enforced. The third — the fuzzer's suppression tag — is explicitly recorded as
>   unenforced, which is what `oracle.py:126-131` means by "nothing rings".
>
> **So the live question narrows to three parts:**
>
> **(a) The rule.** Is it "bit pattern, no exceptions" — in which case ORC-76's cbrt
> tolerance, the sklearn leg's `1e-9` and DRAFT-23's declared bound are three
> contradictions that need re-ruling — or is it "bit pattern, with a closed list of
> declared bounds, each naming its discriminator", in which case ORC-32's table *is* the
> list and adding to it is a decision like loosening any other control?
>
> **(b) Future float accumulation.** If parallel float accumulation ever lands, is its
> `UNSPECIFIED` region **refused** (the strict reading) or given a declared bound the way
> the dialect gate's epsilon tier already is (ORC-73)?
>
> **(c) The unenforced third home.** Does the `decimals` tag get an enforcement now — the
> shape TASK-95 would give it, or a strict-xfail twin standing behind the
> known-limitations row — or does it stay unenforced until the lattice phase, on the
> record?
>
> *Verified-by (the facts, not the ruling):*
> `packages/confit/docs/known-limitations.md:166-174`;
> `packages/confit/fuzz/oracle.py:113-131` (the tag's own scope statement and its
> "no strict-xfail twin" note), `:512-513` (the one arm), `:700-712` (the cast, then the
> value comparison); `packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`
> (the feature-in-flight rule and the three homes).
>
> *Binds:* ORC-32, ORC-76, ORC-80, ORC-84, ledger rows D7 and D8, and every future
> float-accumulation feature.

---
