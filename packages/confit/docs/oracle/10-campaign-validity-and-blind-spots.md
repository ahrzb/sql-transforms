# Campaign validity and blind spots

## Existing gates

**claim: regexp-fuzz-gate.** The regexp fuzzer is the standing differential gate. Its
normal run uses `N=250` and a fixed seed; `REGEXP_FUZZ_SEED` and `REGEXP_FUZZ_N` select
deeper runs. Per case it requires identical rows, a conservative confit rejection, or
both engines erroring; DuckDB-success/confit-reject is permitted as conservative
bind-time rejection. Findings feed the reject list, a pin note, and a limitations row.
`pins-waveB/fuzzer-task54.json` records the 12 rejection classes and a zero-divergence
sweep of 40,000 cases over 8 seeds.

*Evidence:* `tests/test_duckdb_regexp_fuzz.py` (module docstring);
`docs/specs/pins-waveB/fuzzer-task54.json`; `known-limitations.md` §4.

`packages/confit/fuzz/` is instead a manual `python -m fuzz.runner` campaign.
`tests/test_fuzz_smoke.py` gates deterministic generation, reproducible verdicts, and
verdict rules; it does not demand zero campaign findings. `known-limitations.md` §7 says
the same.

**claim: campaign-as-acceptance.** Feature work completes only after a campaign
certifies it. When a feature used an xfail or fuzzer marker, certification occurs after
that marker is removed or emptied; before then the run proves only that suppression
still works.

*Evidence:* claim: feature-in-flight in [the ledger](07-the-divergence-ledger.md).

**claim: zero-fails-gate.** Corpus replay classifies each case as match,
clean-unsupported, or FAIL; it requires zero FAILs, and unsupported is clean only with a
documented rejection class. The test also enforces `MATCH_FLOOR`, whose comment records
the reason for each adjustment — including that declared unsigned widths exposed three
type divergences that value-only comparison had counted as matches.

**claim: stable-corpus-ratchet.** A stable corpus allows no unexplained decrease in
support: floors rise as support grows, and a reduction requires a reviewed reason
recorded beside the adjusted floor together with the affected cases. `MATCH_FLOOR` is an
instance of that mechanism, and its total is not universal SQL compatibility: a
regression offset by a new match leaves the total unchanged, so the floor never replaces
case-level regression checks.

*Decision:* [oracle policy](../decisions/closed/oracle-policy.md#reporting-and-measurement);
population and reporting detail in
[success measures](../specs/success-measures.md#measurement-policy).

*Enforced by:* `tests/test_corpus_replay.py` (`MATCH_FLOOR`, `replay_counts`). The dated
headline count lives apart from that constant in `docs/reports/corpus-counts.json`,
written by `scripts/corpus_counts.py` from the same replay, and
`tests/test_corpus_counts.py` keeps every displayed count equal to it and dated.

The expected rows were recorded optimizer-on without capture metadata (claim:
mined-corpus-provenance). A replay match is an observation about that corpus, but it is
not evidence of agreement with the defined optimizer-off oracle. Any narrower use must
say so explicitly.

## Comparison blind spots

**claim: blind-spots.** Full-result comparison is strong only where it is total.

| blind spot | what is not compared | current mitigation or limit |
|---|---|---|
| no total `ORDER BY` | DuckDB row sequence | multiset comparison; our-side self-legs check serving order, not oracle order |
| order-sensitive aggregate values | element order that may vary with `threads` | each such family is refused until it has a justified contract (claim: order-sensitive-family-contract in [ordering](03-nondeterminism.md)); the global `threads` setting is unpinned |
| `order-by-unevaluated` fallback | sortedness on a non-output key | visible logged tag, never silent |
| approximate bind errors | message body | compare error class; bodies are outside claim: error-texts |
| named exclusions | statistics-dependent kernels, f32-grid operations, or inexpressible schemas | measured source and input exclusions, classified under [scope classification](../specs/serving-contract.md#scope-classification); classification does not ratify each individual exclusion |
| absorbed refusal | whether confit could have served what DuckDB serves | the refusal keeps the oracle outcome and the report groups refusals by it; reporting, not a finding |
| canonicalized NaN | sign and payload | repr equality self-equalizes NaNs; explicit bit pins remain exact |
| `UNSHIPPED` width | whether values would agree | separately reported; neither coverage nor finding; no value normalization |
| output nullability | DuckDB's exact nullable flags | not required: `same_type`/`assert_schema` ignore flags at any depth; soundness of our non-null promises is checked on our own rows by `non_null_violation` (`DIVERGE_VALUE`, class `unsound-non-null`) |
| timeout or panic | all semantics for the unanswered case | finding with SQL, inputs and side (`oracle`/`confit`/`harness`) attributed from the worker's last phase marker; rated in the abstention section |

## Self-checks when oracle comparison abstains

**claim: metamorphic-self-legs.** Six checks are useful without treating them as
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

*Evidence:* `fuzz.oracle._extra_legs`; `fuzz.oracle.run_case` for backend agreement;
`tests/test_fuzz_order_legs.py`; P19 in `docs/properties.md`. No dedicated test covers
the hostile-Arrow, rows-versus-Arrow, or sklearn legs.

## Reporting rules

**claim: unspecified-residuals.** Unknown observations are reported separately as
unresolved, not counted as confirmed parity defects or agreement. Differences in a
genuinely unconstrained aspect are not parity defects; unexplained differences must
not be relabelled unspecified. A confirmed in-contract mismatch is a defect.
See [unresolved observations](07-the-divergence-ledger.md).

**claim: acceptance-reporting.** Report generated-campaign acceptance; no percentage
target is adopted. Each rate states its population and shows unknown outcomes and
invalid declarations separately. Do not improve it by silently changing the
generator or denominator. C1–C5 and D1–D2 are the success measures.

*Decision:* [oracle policy](../decisions/closed/oracle-policy.md#reporting-and-measurement);
dispositions in [success measures](../specs/success-measures.md#measurement-policy).

The runner reports raw verdict counts, the same verdicts by outcome category over the
case population, refusals by oracle outcome, and an AGREE-only construct histogram.

**claim: coverage-denominator.** The report counts distinct
`(operator, argument-type, edge-class)` triples, reached versus agreed per operator,
rather than raw query count. This is reporting only, not a universal coverage-metadata
scheme.

*Enforced-by:* `fuzz.coverage` and the report's triples section.

**claim: abstention-rate.** The report gives rates for `SKIP`, `TIMEOUT`, `PANIC`, and
`order-by-unevaluated`, keeping `UNSHIPPED` separate; see claim: abstention-report in
[verdicts](04-verdicts-agreement-abstention-refusal.md).

No blocking KPI is adopted from refusal-quality or unsupported-width reporting. Where
reason codes may appear is claim: reason-code-placement in
[verdicts](04-verdicts-agreement-abstention-refusal.md). The accepted policy is
[the oracle policy decision](../decisions/closed/oracle-policy.md).
