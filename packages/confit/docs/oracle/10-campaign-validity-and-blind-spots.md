# Campaign validity and blind spots

## Existing gates

**claim: regexp-fuzz-gate.** The regexp fuzzer is the standing differential gate. Its
normal run uses `N=250` and a fixed seed; `REGEXP_FUZZ_SEED` and `REGEXP_FUZZ_N` select
deeper runs. The first deep run found 122 divergences distilled to 12 rejection classes;
a later dated sweep reported zero divergences across 40,000 cases and 8 seeds. Its
four-outcome rule permits DuckDB-success/confit-reject as conservative bind-time
rejection, and findings feed the reject list, pin note, and limitations row.

*Evidence:* `tests/test_duckdb_regexp_fuzz.py:13-20, :35-37`;
`docs/specs/pins-waveB/fuzzer-task54.json`;
`docs/reports/pins-first-methodology.md:70-74`.

`packages/confit/fuzz/` is instead a manual `python -m fuzz.runner` campaign.
`tests/test_fuzz_smoke.py` gates deterministic generation, reproducible verdicts, and
verdict rules; it does not demand zero campaign findings. `known-limitations.md` §7 now
says so instead of listing the campaign among the gated mechanisms (**ticket:
fuzzer-gate-correction**, done).

**claim: campaign-as-acceptance.** An m-8 phase completes only after a campaign certifies
it. When a feature used an xfail or fuzzer marker, certification occurs after that marker
is removed or emptied; before then the run proves only that suppression still works.

*Evidence:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:40-41`;
`docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`; claim:
feature-in-flight.

**claim: zero-fails-gate.** Corpus replay classifies each case as match,
clean-unsupported, or FAIL; it requires zero FAILs, and unsupported is clean only with a
documented rejection class. The shipped test also enforces `MATCH_FLOOR = 547`.

The code records why the former 550 headline became 547: declared unsigned widths
exposed three type divergences that value-only comparison had counted as matches. That
explains the current constant.

**claim: stable-corpus-ratchet.** A stable corpus allows no
unexplained decrease in support: floors rise as support grows, and a reduction requires
a reviewed reason recorded beside the adjusted floor together with the affected cases.
The shipped `MATCH_FLOOR = 547` is an existing instance of that mechanism, not a new
one, and its total is not universal SQL compatibility: a regression offset by a new
match leaves the total unchanged, so the floor never replaces case-level regression
checks. Adopting the rule measures nothing by itself.

*Decision:* [oracle policy](../decisions/oracle-policy.md#reporting-and-measurement);
population and reporting detail in
[success measures](../specs/success-measures.md#measurement-policy).

*Enforced by:* `tests/test_corpus_replay.py:39-47, :185-208`. Proposed **ticket:
match-count-single-home** separates dated headline counts from the shipped constant.

The expected rows were recorded optimizer-on without capture metadata (claim:
mined-corpus-provenance). A replay match remains an observation about that corpus, but
it is not evidence of agreement with the defined optimizer-off oracle. Any narrower use
must say so explicitly.

## Comparison blind spots

**claim: blind-spots.** Full-result comparison is strong only where it is total.

| blind spot | what is not compared | current mitigation or limit |
|---|---|---|
| no total `ORDER BY` | DuckDB row sequence | multiset comparison; our-side self-legs check serving order, not oracle order |
| future order-sensitive aggregate values | element order that may vary with `threads` | each such family needs a justified contract before it is supported, and is refused until then (claim: order-sensitive-family-contract in [ordering](03-nondeterminism.md)); the global `threads` setting is unchanged |
| `order-by-unevaluated` fallback | sortedness on a non-output key | visible logged tag, never silent |
| approximate bind errors | message body | compare error class; bodies are outside claim: error-texts |
| named exclusions | statistics-dependent kernels, f32-grid operations, or inexpressible schemas | measured source and input exclusions, classified under [scope classification](../specs/serving-contract.md#scope-classification); classification does not ratify each individual exclusion |
| absorbed refusal | whether confit could have served what DuckDB serves | the refusal keeps the oracle outcome and the report groups refusals by it; reporting, not a finding |
| canonicalized NaN | sign and payload | repr equality self-equalizes NaNs; explicit bit pins remain exact |
| `UNSHIPPED` width | whether values would agree | separately reported; neither coverage nor finding; no value normalization |
| output nullability | DuckDB's exact nullable flags | not required: `same_type`/`assert_schema` ignore flags at any depth; soundness of our non-null promises is checked on our own rows by `non_null_violation` (`DIVERGE_VALUE`, class `unsound-non-null`) |
| timeout or panic | all semantics for the unanswered case | finding with SQL, inputs and side (`oracle`/`confit`/`harness`) attributed from the worker's last phase marker; rated in the abstention section |


## Self-checks when oracle comparison abstains

**claim: metamorphic-self-legs.** Six checks remain useful without treating them as
DuckDB agreement:

1. batch versus single-row sequence equality;
2. input reversal reverses output blocks;
3. hostile Arrow slicing, chunking, and empty-batch invariance;
4. `infer_rows` versus `infer_arrow` agreement;
5. Cranelift versus interpreter agreement;
6. sklearn agreement for plain-tree cases.

The first five use exact canonical comparison. Sklearn uses an absolute `1e-9` bound as
a second reference, not as the oracle. These legs also run for `UNSHIPPED`; canonical
NaN equality retains the bit-level blind spot above.

*Evidence:* `fuzz.oracle._extra_legs`; `fuzz.oracle.run_case:611-627` for backend
agreement; `tests/test_fuzz_order_legs.py`; P19 in `docs/properties.md:240-245`.
Dedicated hostile-Arrow, rows-versus-Arrow, and sklearn tests were absent when measured
2026-09-02.

## Reporting rules and what is not built

**claim: unspecified-residuals.** Unknown observations are reported separately as
unresolved, not counted as confirmed parity defects or agreement. Differences in a
genuinely unconstrained aspect are not parity defects; unexplained differences must
not be relabelled unspecified. A confirmed in-contract mismatch remains a defect.
The retired “79 of 84” summary supplies no current cases or count for any category.
See [unresolved observations](07-the-divergence-ledger.md).

**claim: acceptance-reporting.** Report generated-campaign acceptance; no percentage
target is adopted. Each rate states its population and shows unknown outcomes and
invalid declarations separately. Do not improve it by silently changing the
generator or denominator. C1–C5 and D1–D2 remain unchanged.

*Decision:* [oracle policy](../decisions/oracle-policy.md#reporting-and-measurement);
dispositions in [success measures](../specs/success-measures.md#measurement-policy).

The adopted reporting intent does not adopt a schema for it. The runner reports raw
verdict counts, the same verdicts by outcome category over the case population, refusals
by oracle outcome, and an AGREE-only construct histogram; the table below says which
proposal is implemented.

| proposal | proposed effect | status |
|---|---|---|
| **claim: coverage-denominator** | report distinct `(operator, argument-type, edge-class)` triples rather than raw query count | proposed schema, unimplemented; ticket: coverage-triples |
| **claim: abstention-rate** | report rates for `SKIP`, `TIMEOUT`, `PANIC`, and `order-by-unevaluated`, keeping `UNSHIPPED` separate | implemented as claim: abstention-report (ticket: per-kind-abstention-report) |

Refusal-quality and unsupported-width reporting come before any new blocking KPI, and
none is adopted here. Where reason codes may appear is claim: reason-code-placement in
[verdicts](04-verdicts-agreement-abstention-refusal.md).

## Remaining campaign prerequisites

These are implementation gaps, not open decisions:

- replace the retired “79 of 84” phase-2 figure by replaying stored SQL or by a clearly
  labelled fresh campaign, then classify the residuals; seeds cannot recreate the
  2026-08-17 baseline after generator changes;
- keep dated displayed match counts apart from the shipped floor (ticket:
  match-count-single-home).

See [the decision index](12-ask-index.md) for the compact status of every decision and
[the oracle policy decision](../decisions/oracle-policy.md) for the accepted policy.
