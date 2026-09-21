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
verdict rules; it does not demand zero campaign findings. Proposed **ticket:
fuzzer-gate-correction** corrects contrary text in `known-limitations.md:301-307`.

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
explains the current constant. The test's comment permits a deliberate tightening
with a reason recorded beside the adjusted floor. A general owner-ratified policy
remains open under ask: match-count-ratchet.

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
| future order-sensitive aggregate values | element order that may vary with `threads` | ask: threads-and-value-order remains open for future retained families; `threads=1` is not adopted |
| `order-by-unevaluated` fallback | sortedness on a non-output key | visible logged tag, never silent |
| approximate bind errors | message body | compare error class; bodies are outside claim: error-texts |
| named exclusions | statistics-dependent kernels, f32-grid operations, or inexpressible schemas | measured source and input exclusions; see ask: exclusion-ratification |
| absorbed refusal | whether DuckDB would have served | no current measure; ask: refusal-cost-counting |
| canonicalized NaN | sign and payload | repr equality self-equalizes NaNs; explicit bit pins remain exact |
| `UNSHIPPED` width | whether values would agree | separately reported; neither coverage nor finding; no value normalization |
| campaign field nullability | top-level output-field nullability | `_schema_delta` compares names and types, while `confit.compare.assert_schema` checks nullability in tests; this observed comparator difference does **not** prove a breach of any normative nullability guarantee |
| timeout or panic | all semantics for the unanswered case | finding with manual oracle-side versus engine-side attribution |


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

## Unadopted campaign proposals

None of these proposals is in force.

| proposal | proposed effect | path to decision |
|---|---|---|
| **claim: unspecified-residuals** | do not count an `UNSPECIFIED` residual as a parity defect until classified as determined-and-wrong or under-determined | ask: proposed-rules-adoption; currently affects phase-two width residuals |
| **claim: coverage-denominator** | report distinct `(operator, argument-type, edge-class)` triples rather than raw query count | ticket: coverage-triples |
| **claim: abstention-rate** | report rates for `SKIP`, `TIMEOUT`, `PANIC`, and `order-by-unevaluated`; keep `UNSHIPPED` separate | ticket: per-kind-abstention-report; ask: reason-code-visibility |

The runner already reports raw verdict counts and an AGREE-only construct histogram; it
does not implement the proposed triples or rates. Proposed KPIs and measurement
acceptance are centralized in [success measures](../specs/success-measures.md).

## Open campaign decisions

- **ask: width-residual-classification** — run a fresh campaign and classify stored SQL
  before using the unreconstructible “79 of 84” phase-2 figure as a defect count. Seeds
  cannot recreate the baseline after generator changes.
- **ask: match-count-ratchet** — decide whether the shipped `MATCH_FLOOR = 547` becomes
  policy or remains an implementation observation, and define how approved adjustments
  are made.
- **ask: proposed-rules-adoption** — rule on each listed proposal independently. A
  citation does not adopt target-status vocabulary, countable cost, multi-answer sets,
  standing rejections, divergence placement, absolute severity rungs 1/2,
  countable-rung-four, or unspecified residuals.

See [the decision index](12-ask-index.md) for bindings and the other open decisions.
