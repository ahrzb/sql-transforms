## 10. Campaign validity and blind spots

**claim: unspecified-residuals.** **[PROPOSED]** Not in force. A campaign residual that
lands in an `UNSPECIFIED` region is **not a parity defect**: classify before counting,
because a number that mixes determined-and-wrong with under-determined is not a defect
count and must not be treated as a backlog. The rule follows from
claim: nondeterminism-axiom, but nothing outside this document states it, and its own
basis claim: target-status-vocabulary is `[PROPOSED]` too. Its immediate application is
divergence: phase-two-width-residuals and ask: width-residual-classification.
*Verified-by:* Unverified. Part of ask: proposed-rules-adoption.

**claim: regexp-fuzz-gate.** **One** standing differential gate runs in the normal test
gate: the regexp fuzzer at N=250 with a fixed seed (`REGEXP_FUZZ_SEED` / `REGEXP_FUZZ_N`
for deep runs). Its first deep run found 122 divergences distilled to 12 reject classes,
then re-swept to **zero divergences over 40k cases across 8 seeds**. Its four-outcome
rule is decided and worth stating, because it is a precedent: duckdb-ok + engine-rejects
is declared "fine (conservative bind-time reject)" — a designed, unconditional rung-4
absorb — and a finding is dispositioned as a reject-list entry plus a pin note plus a
limitations row.
*Verified-by:* `packages/confit/tests/test_duckdb_regexp_fuzz.py:13-20` (the
four-outcome rule and the finding-disposition protocol), `:35-37` (the seed and N
defaults); `packages/confit/docs/specs/pins-waveB/fuzzer-task54.json`;
`packages/confit/docs/reports/pins-first-methodology.md:70-74`.
*Correction:* an earlier version of this claim, and `known-limitations.md:301-307`, call
the **campaign fuzzer** a second standing gate. It is not. `packages/confit/fuzz/` is a
manual CLI (`python -m fuzz.runner`) and is not collected — `packages/confit`'s
`testpaths = ["tests"]`. What runs in the gate is
`packages/confit/tests/test_fuzz_smoke.py`, whose docstring says the opposite of a
differential gate: "The fuzzer exists to find live bugs, so 'no findings over N seeds'
**cannot** be the CI invariant ... What CI pins instead: generation is deterministic,
the oracle produces verdicts across the seed range, verdicts are reproducible" —
machinery, not zero findings. That file is also where the verdict *rules* of chapters 4
and 5 are gated, `UNSHIPPED` included, so it carries more than smoke. See proposed
ticket: fuzzer-gate-correction.

**claim: campaign-as-acceptance.** A campaign is an **acceptance test, not a
formality**: each m-8 phase ends with a fuzz campaign certifying it before the next
starts, and for a feature whose scaffolding included a fuzzer suppression tag, the
certification campaign *after* the tag removal is what proves the class is gone rather
than hidden (claim: feature-in-flight).
*Verified-by:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:40-41`;
`packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`.

**claim: zero-fails-gate.** The corpus replay gates **zero FAILs, always**. Three
outcomes — match, clean-unsupported, FAIL — where a rejection is only clean if it is one
of the *documented* rejection classes, so an undocumented error is a FAIL like any wrong
answer. The match count is deliberately **ungated**: it is the growth ladder (53 -> 395
-> 505 -> 511 -> 529 -> 546 -> 550 of 678), and every construct learned flips cases from
clean-unsupported to match, never into FAIL.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:17-20, :179-198`;
`packages/confit/docs/reports/pins-first-methodology.md:41-62`.
*Correction:* the number 550/678 is quoted **without its hedge at six sites**, and tasks
have landed since that flip cases. `known-limitations.md:12` is the one place that
carries the "as of stage B" hedge; the unhedged sites are
`reports/confit-architecture.md:3`, `reports/performance-report.md:9`,
`reports/pins-first-methodology.md:3` (the abstract) and `:124` (a second quote in the
same document), `packages/confit/README.md:109`, and the repo-root `README.md:112`. A
remedy scoped to "the abstract and two reports", or to "exactly one place", would leave
three of them standing. See ask: match-count-ratchet and proposed
ticket: match-count-single-home.
*Second correction, structural:* the corpus's expected rows are themselves an
optimizer-on recording with no provenance (claim: mined-corpus-provenance), so this
ladder is measured against a different reading of DuckDB than the one
claim: oracle-identity names. That does not make the zero-FAILs gate wrong — a match is
still a match — but it does mean the ladder is not evidence about the oracle in
claim: oracle-identity's sense, and section 9 cannot re-record what has no provenance.

**claim: blind-spots.** Our full-result diff is **stronger** than any partial oracle
wherever it is total. Wherever it is partial we have re-created a partial oracle and we
own its blind spot. The blind spots, named:

| blind spot | what the oracle does not pin there | mitigation in force |
|---|---|---|
| row order **in every mode** except a total `ORDER BY` — including the row path, where DuckDB's order is not a function of the query either (claim: compare-modes) | the sequence, against DuckDB | multiset against DuckDB, plus **our-side-only** self-legs (claim: serving-row-order); the self-legs pin our contract, not agreement with the oracle |
| order **inside a value** (`string_agg`, `list`) on the frozen path | element order, which is a function of `threads` (claim: threads-setting) | **none today** — the assigned compare mode cannot see it. This is ask: threads-and-value-order |
| `order-by-unevaluated` fallback | sortedness on a non-output-column key | logged tag, not silent (claim: logged-fallback) |
| error texts (divergence: approximate-error-text) | message bodies for bind-time rejections | error *class* is compared; texts are not oracle-decided output (claim: error-texts) |
| the excluded ILIKE-NUL source (divergence: ilike-nul), the f32 blanket rule and `_INEXPRESSIBLE_INPUTS` (claim: corpus-exclusion-sets) | statistics-dependent kernel selection; every f32-grid-sensitive operation; declared-schema-inexpressible inputs | exclusion by name with a measured reason (claim: statistics-dependent-exclusion, claim: corpus-exclusion-sets) |
| refusals | whether the oracle would have served (claim: refusal-absorb) | **none today** — this is ask: refusal-cost-counting |
| NaN sign and payload, wherever the canonical form is used | which NaN | `repr` makes every NaN self-equal — in `confit.compare` and, independently, in `test_corpus_replay.py`'s own `_norm_row` (`:70-72`); only the explicit bit pins see the difference (claim: float-bit-equality, claim: repr-equality) |
| schema, on the campaign's **static-only** path | name, type, width and nullability, none of which is compared against DuckDB there | **none today** — `against()` returns before `_schema_delta` is reached, so a wrong-width or zero-row answer grades `AGREE`, and a bare-decimal static-only case value-compares across an unshipped width instead of classifying (claim: schema-comparison, claim: unshipped-verdict, ticket: static-only-schema-check) |
| an **unshipped width** (today: decimal literals) | whether the values would have agreed, since none were compared | `UNSHIPPED` classifies loudly, is neither a finding nor coverage, and gets its own report section — so the blind spot is *counted* rather than absorbed (claim: unshipped-verdict). The harness no longer casts, so it no longer invents a value either |
| a worker that never answered | everything about that case | `TIMEOUT` / `PANIC` are findings, not silence (claim: verdict-taxonomy, claim: abstention-reporting), and are attributed oracle-side vs engine-side by hand (claim: timeout-attribution) |

*Verified-by:* each row's cited claim.

**claim: metamorphic-self-legs.** Metamorphic self-legs are the oracle *substitute* in
the abstention region, and the set is capped rather than grown: batch-vs-single sequence
equality, input reversal reversing the output blocks, hostile-arrow invariance (sliced,
chunked, empty), `infer_rows` vs `infer_arrow` agreement, cranelift vs interpreter
agreement, and sklearn as a second ground truth on plain tree cases. They involve no
DuckDB, which is what makes them usable exactly where the oracle abstains. Five of the
six compare exactly; the sklearn leg compares within `1e-9` (claim: float-bit-equality),
because sklearn is a second reference and not the oracle. They run for an `UNSHIPPED`
case too, precisely because they contain no DuckDB (claim: unshipped-verdict).
*Enforced-by:* **two** homes, and the split matters when reading a finding.
`fuzz.oracle._extra_legs` (`fuzz/oracle.py:740-852`) holds five of the six — infer_rows
vs infer_arrow (`:747-762`), hostile arrow (`:764-789`), batch-vs-single (`:796-811`),
reversal (`:812-830`) and the sklearn `1e-9` leg (`:833-851`) — with its exact
comparisons being `confit.compare.sequence` and `.multiset`. Cranelift vs interpreter is
**not** there: it is settled early in `fuzz.oracle.run_case` (`:611-627`), which is what
claim: backend-agreement describes.
*Verified-by:* `packages/confit/tests/test_fuzz_order_legs.py` for the order legs
(`test_a_correct_engine_passes_the_order_legs`,
`test_a_scrambled_batch_is_caught_as_an_order_bug`); P19 in
`packages/confit/docs/properties.md:240-245` for the backend leg. The hostile-arrow,
infer_rows-vs-arrow and sklearn legs have **no test of their own** — `Unverified`
(measured 2026-09-02).

**claim: coverage-denominator.** **[PROPOSED]** Not in force. A campaign declares a
coverage signal, because "20k queries, N residuals" is unanchored without a denominator
that means something. We have no query plans to diversify over, so the analogue is
distinct **(operator, argument-type, edge-class)** triples reached per campaign — which
extends the existing `AGREE`-only construct histogram's axis rather than replacing it.
That same triple is the right unit for decision coverage: one operator x one type x one
edge class, not one feature.
*Verified-by:* the histogram exists in `fuzz.runner.report`, over `fuzz.runner.COVERED`;
the triple axis does not. Proposed ticket: coverage-triples.

**claim: abstention-rate.** **[PROPOSED]** Not in force. A campaign reports an
**abstention rate per kind** alongside the coverage histogram, and a rising rate is read
as generator drift rather than as good news. A generator that has drifted out of the
answerable region measures nothing while still printing a green bar. The **kinds** it
would report over all exist and are all findings today — `SKIP`
(claim: abstention-reporting), `TIMEOUT` and `PANIC` (claim: verdict-taxonomy), the
`order-by-unevaluated` tag (claim: logged-fallback) — and the rule that oracle-side and
engine-side timeouts are opposite things is **already decided**
(claim: timeout-attribution). What does not exist is the rate, and the machine-readable
separation claim: timeout-attribution asks for by hand. `UNSHIPPED` is the one
abstention kind that already reports separately (claim: coverage-accounting), which is
the shape this claim wants for the rest.
*Verified-by:* the kinds exist per claim: verdict-taxonomy, claim: abstention-reporting,
claim: logged-fallback; claim: timeout-attribution for the attribution rule; the rate
does not exist — `fuzz.runner.report` prints raw verdict counts only. Proposed
ticket: per-kind-abstention-report, and see ask: reason-code-visibility for the
user-visibility boundary.

> ### ask: width-residual-classification — the width residuals: defect count or mixed bag?
>
> claim: unspecified-residuals says residuals in the unspecified region are not parity
> defects, so before the width-residual number is treated as a backlog, each residual
> needs classifying as determined-and-wrong vs under-determined. That classification is
> work, and authorizing it is yours; the answer changes what the number means.
>
> One honesty note first: **the 79-of-84 figure is not reconstructible from this tree.**
> Searched 2026-08-25 across `packages/confit/docs/`, `backlog/`, and the committed
> `findings.jsonl` — the numbers appear in none of them. Whatever is authorized should
> begin by re-running the campaign that produced them, because a count nobody can
> re-derive is not evidence.
>
> A second, practical one: the committed baseline **cannot** serve as the starting point
> for that re-run. Its seeds no longer address the cases they addressed
> (ask: baseline-as-evidence / ledger divergence: snapshot-baseline), so any
> classification pass has to start from a fresh campaign, not from the file.
>
> *Binds:* divergence: phase-two-width-residuals and divergence: snapshot-baseline.

> ### ask: match-count-ratchet — does the corpus match count become a ratchet?
>
> Zero FAILs is the gate and should stay the gate (claim: zero-fails-gate). The match
> count is deliberately ungated as a growth ladder, and it has now gone stale in three
> documents because nothing watches it. Two options:
>
> - **leave ungated**, and generate the number with a date stamp in exactly one place
>   so drift cannot recur silently — note the drift is at **six** sites, not three
>   (claim: zero-fails-gate's correction);
> - **add a never-decreases ratchet.** Real cost: any deliberate scope reduction
>   becomes a gate failure, and a scope reduction is sometimes the right call.
>
> The precedent for the second option already exists inside this package and is worth
> reading before ruling: the dialect L3 gate carries exactly that ratchet — "The match
> floor is the measured count at introduction — raise it when the surface grows, never
> lower it" (claim: dialect-gate-oracle) — so the question is whether it generalizes
> from a printed-surface gate to a mined-corpus one, not whether it is workable.
>
> *Binds:* claim: zero-fails-gate, claim: dialect-gate-oracle.

> ### ask: proposed-rules-adoption — do this document's own rules become normative?
>
> Eight rules in this document (across nine claims) were originally written as
> decisions in force with a `Verified-by` of "derived" or "normative here", or with no
> source at all. They are not decisions in force: nothing outside this document states
> them, and re-checking found no spec, ticket or review that adopted them. They are
> marked `[PROPOSED]` now rather than deleted, because each one is a real answer to a
> real question — but adopting a rule is your call, not a document's.
>
> **Scope, so the marker's population and this ASK are not confused.** Fifteen claims
> carry `[PROPOSED]`; this ASK covers nine of them. The other **eight** —
> claim: pin-back-reference, claim: under-determined-token,
> claim: re-record-diff-report, claim: diff-triage-classes, claim: mutability-classes,
> claim: changed-pin-record, claim: coverage-denominator, claim: abstention-rate — are
> pins-and-campaign machinery rather than comparison doctrine, and they are routed
> through proposed tickets ticket: pin-decision-field, ticket: pin-field-token,
> ticket: corpus-drift-report, ticket: coverage-triples and
> ticket: per-kind-abstention-report, each blocked on your go rather than on this table.
> Two claims this table lists, **claim: unadopted-mechanisms and
> claim: directional-rungs, carry no marker**: claim: unadopted-mechanisms is a survey
> whose four unadopted mechanisms claim: standing-rejections would turn into rejections,
> and claim: directional-rungs is in force except for the absolute the row names.
>
> | claim | the rule | the cost of adopting it |
> |---|---|---|
> | claim: target-status-vocabulary | every comparison target carries a status | seven claims have one today; the rest must be statused or the rule narrowed to the ledger |
> | claim: countable-cost | an accepted cost must be countable, with the mechanism named in the decision | makes ask: refusal-cost-counting's split mandatory rather than optional, and applies retroactively to divergence: regex-size-guard |
> | claim: multi-answer-sets | multi-answer sets are legitimate only with a selecting predicate | no pin in force is set-valued in that sense; claim: cbrt-ulp-tolerance's bounded tolerance is a shape the rule does not cover |
> | the four mechanisms in claim: unadopted-mechanisms / claim: standing-rejections | they become standing rejections | must be written so as not to contradict claim: float-bit-equality's three in-force tolerances or claim: dialect-gate-oracle's designed epsilon tier |
> | claim: divergence-placement | a divergence note sits at the requirement it violates | there is no engine spec to put them in |
> | claim: directional-rungs' absolute | "rungs 1 and 2 are never accepted" | contradicted on its face by divergence: trap-elision, divergence: decimal-literal-typing and divergence: decimal-cast-rounding; adopting it means re-ruling those |
> | claim: countable-rung-four | a deliberate rung-4 choice must be countable | follows claim: countable-cost and stands or falls with it |
> | claim: unspecified-residuals | an `UNSPECIFIED` residual is not a parity defect | needs claim: target-status-vocabulary first, since "unspecified" has to be a recorded status to be checkable |
>
> The cheap option is to adopt none of them and let the document record only what was
> decided elsewhere. The expensive-but-useful option is to adopt them individually. The
> option to avoid is leaving them marked forever, because a `[PROPOSED]` rule that
> everyone cites is a decision that was never made.
>
> *Binds:* claim: target-status-vocabulary, claim: countable-cost,
> claim: multi-answer-sets, claim: unadopted-mechanisms, claim: divergence-placement,
> claim: directional-rungs, claim: countable-rung-four, claim: unspecified-residuals,
> claim: standing-rejections.

---
