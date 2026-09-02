## 4. Verdicts: agreement, abstention, refusal

### 4.1 The taxonomy

**ORC-23.** One case in, one verdict out. Every outcome — refusal, trap, disagreement,
and the oracle's own failure — comes back *as* a verdict rather than as an exception, so
nothing is classified by a human reading a stack trace. The oracle module emits eleven
kinds; a **campaign** emits thirteen, because the runner synthesizes two more for a
worker that never answered.

| kind | meaning | emitted by |
|---|---|---|
| `AGREE` | ours == off == on | oracle |
| `AGREE_TRAP` | both sides error at run time | oracle |
| `DIVERGE_VALUE` | wrong value, wrong schema, a self-leg failure, **or a cranelift-vs-interpreter split** (klass `backend-values` / `backend-trap-split`, no DuckDB involved) | oracle |
| `DIVERGE_BUILD` | confit builds what DuckDB refuses, **or the two backends disagree about whether the build succeeds** (klass `backend-split`, no DuckDB involved) | oracle |
| `DIVERGE_TRAP` | one side traps where the other serves rows | oracle |
| `DIVERGE_OPT` | we match the optimizer-off baseline; an optimizer pass changes what the user sees | oracle |
| `OPT_EMULATED` | we match optimizer-ON against a baseline that disagrees: a plan-rewrite pass we are reproducing, which is a bug | oracle |
| `BUILD_EXC` | a build raised something other than the contract's `ValueError` | oracle |
| `REFUSED` | confit refused at build | oracle |
| `UNSHIPPED` | the answer has a width we have not shipped, so nothing was compared (ORC-92) | oracle |
| `SKIP` | the oracle harness itself raised | oracle |
| `TIMEOUT` | the worker did not answer inside the per-case budget; the detail is an 800-byte stderr tail | **runner** |
| `PANIC` | the worker died without answering; same detail shape | **runner** |

`TIMEOUT` and `PANIC` are in `INTERESTING` and reach `findings.jsonl` by the same path
as every other verdict, so any statement about "what a campaign reports" has to include
them — ORC-26's abstention story and ORC-68's blind-spot table both do now. `UNSHIPPED`
is the one kind that is neither a finding nor coverage; it has its own report section.
*Enforced-by:* `fuzz.oracle.KINDS` and `fuzz.oracle.run_case`; `fuzz.runner` synthesizes
`TIMEOUT` / `PANIC` and writes every `INTERESTING` verdict to `findings.jsonl`. The same
shape holds one level down, on the oracle's own side: `confit.oracle.Oracle.try_answer`
returns a frozen `Trap` (the exception's class name and message) instead of raising, so
a refusal by DuckDB is data a caller classifies rather than control flow it must catch.
*Verified-by:* `packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`;
`packages/confit/tests/test_oracle.py::test_try_answer_returns_a_trap_when_the_query_fails`
and `::test_trap_is_frozen`.

**ORC-24.** Every case is run against DuckDB twice on **one** connection, off then on.
Sharing the connection is not just a saving: `statistics_propagation` reads per-column
statistics, so two separate connections could differ for reasons that have nothing to
do with the optimizer. The pair therefore brackets the answer and a finding classifies
itself. An `UNSHIPPED` verdict outranks the bracket: neither reading was value-compared,
so neither can be evidence for or against a plan-rewrite pass (ORC-92).
*Enforced-by:* `fuzz.oracle._duck_run` (one connection, `Oracle` then
`Oracle.optimizer_on`) and the ranking at the end of `fuzz.oracle.run_case`.
*Verified-by:* `packages/confit/tests/test_oracle.py::test_optimizer_on_flips_the_same_connection`
(the flip is in place, on the same connection);
`packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`.

**ORC-25.** `OPT_EMULATED` is a bug, not an accepted class, and it is excluded from
coverage. Counting it as agreement would hide it twice: once as a finding and once as
coverage.
*Enforced-by:* `fuzz.runner.INTERESTING` (contains it) and `fuzz.runner.COVERED`
(`("AGREE",)`, which does not).
*Verified-by:* `packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`.

### 4.2 Abstention is a verdict

**ORC-26.** Abstention is reported, never silently downgraded to a pass. `SKIP` — the
oracle harness's own failure — is a finding and reaches `findings.jsonl`; it is not
allowed to look like agreement, because an error bucket that quietly grows is how a
suite hides real bugs behind a green bar. `TIMEOUT` and `PANIC` (ORC-23) are the same
species and are treated the same way.
*Enforced-by:* `fuzz.oracle.run_case_json` (an exception escaping `run_case` becomes
`SKIP`, blaming the oracle rather than the engine) and `fuzz.runner.INTERESTING`, which
holds all three.
*Verified-by:* `packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`.

**ORC-78.** A timeout is attributed before it is counted: **an oracle-side timeout and
an engine-side timeout mean opposite things.** Measured 2026-08-14 on seed 4395 —
`lpad(c1, 2147483647, 'NULL') LIKE '...'` — we refuse in 0.00s at bind ("lpad count
2147483647 exceeds the 1 GiB string-builder budget") while DuckDB takes 9.0s actually
building the 2 GiB pad and answering `false`. Under eight workers that exceeds the
per-case budget, which *is* the finding: no engine hang, no liveness bug. Three further
seeds are the same story. The recorded follow-ups are that the runner must record the
SQL *before* executing (a timeout currently loses it, so the case has to be recovered
from the generator by seed) and that oracle-side timeouts must classify apart from
engine-side ones.
*Verified-by:* `packages/confit/docs/2026-08-13-fuzz-triage.md:124-149`.

**ORC-27.** A comparison the checker cannot evaluate falls back to the weaker check
**with a logged tag**, never silently. The one instance today: an `ORDER BY` over an
expression that is not an output column cannot have its key evaluated, so the multiset
check stands and the case carries an `order-by-unevaluated` tag.
*Enforced-by:* `fuzz.oracle.run_case`'s static-only branch.
*Verified-by:* `backlog/tasks/task-129 ...md` AC #4.

**ORC-28.** `AGREE` is the only kind counted as coverage. A construct-coverage histogram
runs over agreeing cases only, so a grammar hole is visible rather than absorbed by
refusals — and `UNSHIPPED` is excluded for the stronger reason that nothing was compared
at all (ORC-92). It is not a finding either, so it is reported in a section of its own:
an empty one means either the feature shipped or the grammar stopped reaching it, and
both are worth seeing.
*Enforced-by:* `fuzz.runner.COVERED` and the unshipped-feature section of
`fuzz.runner.report`.
*Verified-by:* `packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`.

### 4.3 Refusals

**ORC-29.** A build-time refusal is the engine's second legal outcome and is always a
named `ValueError` at `DuckDBInferFn(...)` construction. Three documented message
prefixes classify it: `unsupported:` (real SQL, deliberately not served), `parse error:`
(the dialect surface ends here), `bind error:` (the query is wrong against your schema).
Refusal is cheap, named and testable by construction.
*Verified-by:* `packages/confit/docs/known-limitations.md:274-285`; P7 and P18 in
`packages/confit/docs/properties.md:63, :231-235`.
*Correction:* the prefix set is not exhaustive in code. The corpus gate's clean set is
`_CLEAN = ("unsupported:", "parse error:", "duplicate map key", "NULL in value column")`
(`test_corpus_replay.py:36`) — two real engine messages that carry none of the three
prefixes (`interp.rs` "@{i}: duplicate map key"; `duckdb/mod.rs` "static table '...' has
a NULL in value column '...'"), and `bind error:` is absent from that set entirely.
Either the two messages gain a prefix or the documented set gains them. See proposed
ticket T-17.

**ORC-79.** Refusal *grounds* are a decided three-way taxonomy, orthogonal to the
message prefix: **specialization-inherent** (the engine model cannot express it),
**scope-by-product-decision** (it could be served and we chose not to), and **resource**
(it would cost more than a serving engine may spend per row). The split is what makes
"we refuse" auditable — the first is permanent, the second is reversible by a decision,
the third is a judgement with a number attached.
*Verified-by:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:30-36`; mirrored by
`known-limitations.md` section 1 ("The specialization bargain (inherent to the engine
model)", `:65`) vs section 2 ("Out of scope for row-serving (by decision, not
difficulty)", `:80`), with the resource class at `known-limitations.md:205` and
`known_divergences/test_arrow_boundary.py:34-36` (ledger rows D14 and D15).

**ORC-30.** **[FACT]** Current behavior of the campaign's refusal path: both
DuckDB readings are computed and then **discarded unconditionally** when confit
refused. `REFUSED` carries only a class derived from the first six words of the
message, and it is not in `INTERESTING`, so it never reaches `findings.jsonl` — only a
"top refusal classes" histogram at the end of a run.

```python
# fuzz.oracle.run_case
    duck_off, duck_on = _duck_run(sql, case, udf_objs)

    if fn_cl is None:
        klass = _refusal_class(cl_err)
        return Verdict("REFUSED", klass, cl_err, tags)
```

*Verified-by:* `fuzz.oracle.run_case` (the block above) and `fuzz.oracle._refusal_class`;
`fuzz.runner.INTERESTING`, which does not contain `REFUSED`, and `fuzz.runner.report`'s
refusal-class histogram.

**ORC-31.** **[PROPOSED]** Not in force. The general rule this exposes, and the one this
document would like written down: **an accepted cost must be countable, and the counting
mechanism is named in the decision that accepts it.** Without that, "deliberate
strictness" and "unnoticed over-refusal" are the same observation. The rule is not in
force anywhere today; the live instance is ASK-3, and adopting the rule itself is part of
ASK-15.
*Verified-by:* Unverified — no decision outside this document states it. The nearest
existing practice is the m-8 phase rule that each phase's markers must be deleted in the
phase's own PR and certified by a campaign (ORC-80), which is a counting mechanism for a
different kind of cost.

> ### ASK-3 — the accepted severity-4 cost is currently uncountable. Which way?
>
> You accepted the bind-time constant refusals twice (2026-08-24, re-affirmed
> 2026-08-25 on corrected facts). The RFC justifies the accepted cost three times by
> asserting the campaign will measure it:
>
> - `rfcs/2026-08-19-keep-the-bind-time-refusals.md:95-96` — "Under the fuzzer's
>   refusal-absorb rule (a refusal is acceptable **where the oracle traps**)"
> - `:100-102` — "A campaign that generates those shapes will (correctly) log
>   refuse-where-oracle-serves findings"
> - `:146-149` — "a campaign that generates them will log severity-4 findings, which
>   are attributed to this RFC"
>
> Verified: it cannot. The absorb rule in code is **total, not conditional** (ORC-30) —
> a refusal is absorbed even when both oracle readings serve rows — and `REFUSED` never
> reaches the findings file. The decision itself is not in question; only whether its
> price is observable.
>
> **(a) Split the verdict** so the cost becomes measurable:
>
> ```python
> # fuzz.oracle.run_case, replacing the refusal return quoted in ORC-30
> if fn_cl is None:
>     klass = _refusal_class(cl_err)
>     oracle_serves = duck_off[0] is not None
>     kind = "REFUSED_ORACLE_SERVES" if oracle_serves else "REFUSED_ORACLE_TRAPS"
>     return Verdict(kind, klass, cl_err, tags)
> ```
>
> with `REFUSED_ORACLE_SERVES` added to `runner.INTERESTING`. Three lines. It may
> reveal the class is larger than assumed, which is the point.
>
> **(b) Amend the RFC** to say the cost is accepted unmeasured. Honest and free.
>
> Not applied here — docs-only. Proposed ticket T-5.
>
> *Binds:* ORC-30, ledger row D9, and the severity ladder's rung 4 in section 8.

> ### ASK-4 — `OPT_EMULATED` gets `AGREE` treatment at one branch
>
> Behaviorally the code is on the "bug" side everywhere (ORC-25) except one branch, where
> `OPT_EMULATED` is grouped with `AGREE` for the purpose of running the boundary legs:
>
> ```python
> # fuzz.oracle.run_case, just before the boundary legs
> if v.kind not in ("AGREE", "OPT_EMULATED", "UNSHIPPED"):
>     return v
> ```
>
> Deliberate — run the extra legs anyway, since the values matched *something* — or a
> survivor of the pre-2026-08-17 doctrine, when `OPT_EMULATED` meant expected? One fact
> has arrived since this was first asked: `UNSHIPPED` joined the same branch **on a
> stated ground** — the boundary legs are ours-against-ours with no DuckDB in them, so an
> unshipped width cannot excuse a self-inconsistency (ORC-92). That ground does not
> transfer to `OPT_EMULATED`, which is a DuckDB disagreement, so the branch now holds one
> member with a reason and one without.
>
> One recorded fact the ruling should have: the **only** observed `OPT_EMULATED` instance
> outside regex was a **mislabel, not an emulation**. Seed 1784's `FETCH FIRST 1 ROWS ONLY`
> without an `ORDER BY` had the two DuckDB reads lawfully pick different groups, and the
> triage records it as "mislabelled by this category because the readings disagree with
> each other" — filed as fuzzer QoL for TASK-94, not as a bug in the engine
> (`packages/confit/docs/2026-08-17-fuzz-triage.md:89-94`). ORC-25's "a bug, not an
> accepted class" is the right rule and has never yet had a true positive.
>
> *Binds:* ORC-25.

> ### ASK-5 — do abstention and refusal reason codes become user-visible?
>
> If a reason-code vocabulary (`unspecified-order`, `tie-break`, `fp-association`,
> `session-dependent`, `oracle-errored`) is adopted for campaign reporting and the
> ledger, it must not leak into build-error text without your explicit approval —
> refusal messages are product surface and fall under the API-change rule. Cleanest
> split, and my recommendation: internal codes for the ledger and reports, existing
> prose refusal messages unchanged.
>
> *Binds:* ORC-29.

---
