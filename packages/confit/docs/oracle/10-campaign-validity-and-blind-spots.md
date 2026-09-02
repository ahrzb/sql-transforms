## 10. Campaign validity and blind spots

**ORC-65.** **[PROPOSED]** Not in force. A campaign residual that lands in an
`UNSPECIFIED` region is **not a parity defect**: classify before counting, because a
number that mixes determined-and-wrong with under-determined is not a defect count and
must not be treated as a backlog. The rule follows from ORC-13, but nothing outside this
document states it, and its own basis ORC-15 is `[PROPOSED]` too. Its immediate
application is ledger row D13 and ASK-10.
*Verified-by:* Unverified. Part of ASK-15.

**ORC-66.** **One** standing differential gate runs in the normal test gate: the regexp
fuzzer at N=250 with a fixed seed (`REGEXP_FUZZ_SEED` / `REGEXP_FUZZ_N` for deep runs).
Its first deep run found 122 divergences distilled to 12 reject classes, then re-swept to
**zero divergences over 40k cases across 8 seeds**. Its four-outcome rule is decided and
worth stating, because it is a precedent: duckdb-ok + engine-rejects is declared "fine
(conservative bind-time reject)" — a designed, unconditional rung-4 absorb — and a finding
is dispositioned as a reject-list entry plus a pin note plus a limitations row.
*Verified-by:* `packages/confit/tests/test_duckdb_regexp_fuzz.py:13-20` (the four-outcome
rule and the finding-disposition protocol), `:35-37` (the seed and N defaults);
`packages/confit/docs/specs/pins-waveB/fuzzer-task54.json`;
`packages/confit/docs/reports/pins-first-methodology.md:70-74`.
*Correction:* an earlier version of this claim, and `known-limitations.md:301-307`, call
the **campaign fuzzer** a second standing gate. It is not. `packages/confit/fuzz/` is a
manual CLI (`python -m fuzz.runner`) and is not collected — `packages/confit`'s
`testpaths = ["tests"]`. What runs in the gate is `packages/confit/tests/test_fuzz_smoke.py`,
whose docstring says the opposite of a differential gate: "The fuzzer exists to find live
bugs, so 'no findings over N seeds' **cannot** be the CI invariant ... What CI pins
instead: generation is deterministic, the oracle produces verdicts across the seed range,
verdicts are reproducible" — machinery, not zero findings. That file is also where the
verdict *rules* of chapters 4 and 5 are gated, `UNSHIPPED` included, so it carries more
than smoke. See proposed ticket T-22.

**ORC-89.** A campaign is an **acceptance test, not a formality**: each m-8 phase ends
with a fuzz campaign certifying it before the next starts, and for a feature whose
scaffolding included a fuzzer suppression tag, the certification campaign *after* the tag
removal is what proves the class is gone rather than hidden (ORC-80).
*Verified-by:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:40-41`;
`packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`.

**ORC-67.** The corpus replay gates **zero FAILs, always**. Three outcomes — match,
clean-unsupported, FAIL — where a rejection is only clean if it is one of the
*documented* rejection classes, so an undocumented error is a FAIL like any wrong
answer. The match count is deliberately **ungated**: it is the growth ladder
(53 -> 395 -> 505 -> 511 -> 529 -> 546 -> 550 of 678), and every construct learned flips
cases from clean-unsupported to match, never into FAIL.
*Verified-by:* `packages/confit/tests/test_corpus_replay.py:17-20, :179-198`;
`packages/confit/docs/reports/pins-first-methodology.md:41-62`.
*Correction:* the number 550/678 is quoted **without its hedge at six sites**, and tasks
have landed since that flip cases. `known-limitations.md:12` is the one place that
carries the "as of stage B" hedge; the unhedged sites are
`reports/confit-architecture.md:3`, `reports/performance-report.md:9`,
`reports/pins-first-methodology.md:3` (the abstract) and `:124` (a second quote in the
same document), `packages/confit/README.md:109`, and the repo-root `README.md:112`. A
remedy scoped to "the abstract and two reports", or to "exactly one place", would leave
three of them standing. See ASK-11 and proposed ticket T-13.
*Second correction, structural:* the corpus's expected rows are themselves an
optimizer-on recording with no provenance (ORC-87), so this ladder is measured against a
different reading of DuckDB than the one ORC-02 names. That does not make the zero-FAILs
gate wrong — a match is still a match — but it does mean the ladder is not evidence about
the oracle in ORC-02's sense, and section 9 cannot re-record what has no provenance.

**ORC-68.** Our full-result diff is **stronger** than any partial oracle wherever it is
total. Wherever it is partial we have re-created a partial oracle and we own its blind
spot. The blind spots, named:

| blind spot | what the oracle does not pin there | mitigation in force |
|---|---|---|
| row order **in every mode** except a total `ORDER BY` — including the row path, where DuckDB's order is not a function of the query either (ORC-18) | the sequence, against DuckDB | multiset against DuckDB, plus **our-side-only** self-legs (ORC-16); the self-legs pin our contract, not agreement with the oracle |
| order **inside a value** (`string_agg`, `list`) on the frozen path | element order, which is a function of `threads` (ORC-75) | **none today** — the assigned compare mode cannot see it. This is ASK-13 |
| `order-by-unevaluated` fallback | sortedness on a non-output-column key | logged tag, not silent (ORC-27) |
| error texts (D2) | message bodies for bind-time rejections | error *class* is compared; texts are not oracle-decided output (ORC-39) |
| the excluded ILIKE-NUL source (D3), the f32 blanket rule and `_INEXPRESSIBLE_INPUTS` (ORC-77) | statistics-dependent kernel selection; every f32-grid-sensitive operation; declared-schema-inexpressible inputs | exclusion by name with a measured reason (ORC-20, ORC-77) |
| refusals | whether the oracle would have served (ORC-30) | **none today** — this is ASK-3 |
| NaN sign and payload, wherever the canonical form is used | which NaN | `repr` makes every NaN self-equal in `confit.compare`; only the explicit bit pins see the difference (ORC-32, ORC-90) |
| an **unshipped width** (today: decimal literals) | whether the values would have agreed, since none were compared | `UNSHIPPED` classifies loudly, is neither a finding nor coverage, and gets its own report section — so the blind spot is *counted* rather than absorbed (ORC-92). The harness no longer casts, so it no longer invents a value either |
| a worker that never answered | everything about that case | `TIMEOUT` / `PANIC` are findings, not silence (ORC-23, ORC-26), and are attributed oracle-side vs engine-side by hand (ORC-78) |

*Verified-by:* each row's cited claim.

**ORC-69.** Metamorphic self-legs are the oracle *substitute* in the abstention region,
and the set is capped rather than grown: batch-vs-single sequence equality, input
reversal reversing the output blocks, hostile-arrow invariance (sliced, chunked,
empty), `infer_rows` vs `infer_arrow` agreement, cranelift vs interpreter agreement,
and sklearn as a second ground truth on plain tree cases. They involve no DuckDB, which
is what makes them usable exactly where the oracle abstains. Five of the six compare
exactly; the sklearn leg compares within `1e-9` (ORC-32), because sklearn is a second
reference and not the oracle. They run for an `UNSHIPPED` case too, precisely because
they contain no DuckDB (ORC-92).
*Enforced-by:* `fuzz.oracle._extra_legs`, whose exact comparisons are
`confit.compare.sequence` and `.multiset`.
*Verified-by:* `packages/confit/tests/test_fuzz_order_legs.py`; P1-P20 in
`packages/confit/docs/properties.md`.

**ORC-70.** **[PROPOSED]** Not in force. A campaign declares a coverage signal, because
"20k queries, N residuals" is unanchored without a denominator that means something. We have no query
plans to diversify over, so the analogue is distinct **(operator, argument-type,
edge-class)** triples reached per campaign — which extends the existing `AGREE`-only
construct histogram's axis rather than replacing it. That same triple is the right unit
for decision coverage: one operator x one type x one edge class, not one feature.
*Verified-by:* the histogram exists in `fuzz.runner.report`, over `fuzz.runner.COVERED`;
the triple axis does not. Proposed ticket T-14.

**ORC-71.** **[PROPOSED]** Not in force. A campaign reports an **abstention rate per
kind** alongside the coverage histogram, and a rising rate is read as generator drift
rather than as good news. A generator that has drifted out of the answerable region
measures nothing while still printing a green bar. The **kinds** it would report over all
exist and are all findings today — `SKIP` (ORC-26), `TIMEOUT` and `PANIC` (ORC-23), the
`order-by-unevaluated` tag (ORC-27) — and the rule that oracle-side and engine-side
timeouts are opposite things is **already decided** (ORC-78). What does not exist is the
rate, and the machine-readable separation ORC-78 asks for by hand. `UNSHIPPED` is the one
abstention kind that already reports separately (ORC-28), which is the shape this claim
wants for the rest.
*Verified-by:* the kinds exist per ORC-23, ORC-26, ORC-27; ORC-78 for the attribution
rule; the rate does not exist — `fuzz.runner.report` prints raw verdict counts only.
Proposed ticket T-15, and see ASK-5 for the user-visibility boundary.

> ### ASK-10 — the width residuals: defect count or mixed bag?
>
> ORC-65 says residuals in the unspecified region are not parity defects, so before the
> width-residual number is treated as a backlog, each residual needs classifying as
> determined-and-wrong vs under-determined. That classification is work, and
> authorizing it is yours; the answer changes what the number means.
>
> One honesty note first: **the 79-of-84 figure is not reconstructible from this tree.**
> Searched 2026-08-25 across `packages/confit/docs/`, `backlog/`, and the committed
> `findings.jsonl` — the numbers appear in none of them. Whatever is authorized should
> begin by re-running the campaign that produced them, because a count nobody can
> re-derive is not evidence.
>
> A second, practical one: the committed baseline **cannot** serve as the starting point
> for that re-run. Its seeds no longer address the cases they addressed (ASK-14 / ledger
> D16), so any classification pass has to start from a fresh campaign, not from the file.
>
> *Binds:* ledger rows D13 and D16.

> ### ASK-11 — does the corpus match count become a ratchet?
>
> Zero FAILs is the gate and should stay the gate (ORC-67). The match count is
> deliberately ungated as a growth ladder, and it has now gone stale in three documents
> because nothing watches it. Two options:
>
> - **leave ungated**, and generate the number with a date stamp in exactly one place
>   so drift cannot recur silently — note the drift is at **six** sites, not three
>   (ORC-67's correction);
> - **add a never-decreases ratchet.** Real cost: any deliberate scope reduction
>   becomes a gate failure, and a scope reduction is sometimes the right call.
>
> The precedent for the second option already exists inside this package and is worth
> reading before ruling: the dialect L3 gate carries exactly that ratchet — "The match
> floor is the measured count at introduction — raise it when the surface grows, never
> lower it" (ORC-73) — so the question is whether it generalizes from a printed-surface
> gate to a mined-corpus one, not whether it is workable.
>
> *Binds:* ORC-67, ORC-73.

> ### ASK-15 — do this document's own rules become normative?
>
> Eight rules in this document (across nine claim ids) were originally written as
> decisions in force with a `Verified-by` of "derived" or "normative here", or with no
> source at all. They are not decisions in force: nothing
> outside this document states them, and re-checking found no spec, ticket or review that
> adopted them. They are marked `[PROPOSED]` now rather than deleted, because each one is
> a real answer to a real question — but adopting a rule is your call, not a document's.
>
> | claim | the rule | the cost of adopting it |
> |---|---|---|
> | ORC-15 | every comparison target carries a status | eight claims have one today; the rest must be statused or the rule narrowed to the ledger |
> | ORC-31 | an accepted cost must be countable, with the mechanism named in the decision | makes ASK-3's split mandatory rather than optional, and applies retroactively to D10 |
> | ORC-35 | multi-answer sets are legitimate only with a selecting predicate | no pin in force is set-valued in that sense; ORC-76's bounded tolerance is a shape the rule does not cover |
> | ORC-41's four unadopted mechanisms / ORC-84 | they become standing rejections | must be written so as not to contradict ORC-32's three in-force tolerances or ORC-73's designed epsilon tier |
> | ORC-56 | a divergence note sits at the requirement it violates | there is no engine spec to put them in |
> | ORC-58's absolute | "rungs 1 and 2 are never accepted" | contradicted on its face by D4, D7 and D8; adopting it means re-ruling those |
> | ORC-59 | a deliberate rung-4 choice must be countable | follows ORC-31 and stands or falls with it |
> | ORC-65 | an `UNSPECIFIED` residual is not a parity defect | needs ORC-15 first, since "unspecified" has to be a recorded status to be checkable |
>
> The cheap option is to adopt none of them and let the document record only what was
> decided elsewhere. The expensive-but-useful option is to adopt them individually. The
> option to avoid is leaving them marked forever, because a `[PROPOSED]` rule that
> everyone cites is a decision that was never made.
>
> *Binds:* ORC-15, ORC-31, ORC-35, ORC-41, ORC-56, ORC-58, ORC-59, ORC-65, ORC-84.

---
