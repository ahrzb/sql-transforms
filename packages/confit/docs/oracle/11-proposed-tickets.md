## 11. Proposed tickets

Everything this document wants changed in code or in another document. **This document
applies none of it** — the deliverable is this chapter directory and nothing else. Two
rows are nevertheless struck through: ticket: value-preserving-normalization was applied
in code by the ask: unshipped-never-compared ruling, which shipped separately, and
ticket: static-only-schema-check was refuted by measurement. Both are kept struck rather
than deleted so the slugs stay resolvable.

| ticket | change | claim | blocked on |
|---|---|---|---|
| **ticket: oracle-docstring-corrections** | correct two statements in `confit/oracle.py`'s module docstring and their twins in `known-limitations.md`: (a) `disable_optimizer` also changes behavior at **twelve** sites outside the 33-pass list, two of them in `src/planner/binder/`; (b) "constant folding still happens" is an observation, not a mechanism — the folder is an `EXPRESSION_REWRITER` rule, so the pragma removes it | claim: disable-optimizer-scope | editorial, no ruling needed |
| **ticket: version-assert** | assert `duckdb.__version__ == Oracle.VERSION` in `Oracle.__init__`, on the reserved line beside the pragma | claim: oracle-version-constant | ask: version-pin |
| **ticket: axiom-as-property** | number the axiom as P21 in `properties.md`; make its three existing sites cite it | claim: nondeterminism-axiom | owner's go |
| **ticket: exclusion-count-correction** | correct the source count: `pins-first-methodology.md:89` says two sources, the set holds one (two statements) | claim: statistics-dependent-exclusion | editorial |
| **ticket: split-refused-verdict** | split `REFUSED` into `REFUSED_ORACLE_TRAPS` / `REFUSED_ORACLE_SERVES`, the latter in `INTERESTING` | claim: refusal-absorb | ask: refusal-cost-counting |
| **ticket: phase-probing-in-methodology** | put phase-separated probing into the methodology report, where "how we measure DuckDB" lives | claim: phase-separated-probes | owner's go |
| **ticket: uniform-pin-header** | uniform provenance header on every pin file | claim: pin-provenance | owner's go |
| **ticket: pin-decision-field** | decision back-reference (a claim slug) field on every pin | claim: pin-back-reference | owner's go |
| **ticket: pin-field-token** | an inline under-determined-field token in the pin format | claim: under-determined-token | owner's go |
| **ticket: ambiguity-class-closed** | correct `2026-08-17-fuzz-triage.md:56-58, :70-72`: the ambiguous-reference class is TASK-121, Done | section 7.3 | editorial |
| **ticket: severity-definition-merge** | delete the two parenthetical severity-ladder definitions; cite claim: severity-ladder | claim: severity-ladder | editorial |
| **ticket: corpus-drift-report** | generalize `pin_ast_shapes.py`'s drift-report pattern to the pins corpus | claim: re-record-diff-report | owner's go — **not** ask: version-pin(b), which an earlier version listed. Which version the corpus targets has no bearing on whether a re-record tool exists, and that spurious dependency is what made section 9 read as having no first step |
| **ticket: match-count-single-home** | generate the corpus match count with a date stamp in exactly one place, and correct the **six** unhedged sites claim: zero-fails-gate enumerates | claim: zero-fails-gate | ask: match-count-ratchet |
| **ticket: coverage-triples** | extend the coverage histogram to (operator, arg-type, edge-class) triples | claim: coverage-denominator | owner's go |
| **ticket: per-kind-abstention-report** | report an abstention rate per kind per campaign, and classify oracle-side vs engine-side timeouts in the runner rather than by hand (claim: timeout-attribution's recorded follow-up, plus recording the SQL before executing) | claim: abstention-rate, claim: timeout-attribution | ask: reason-code-visibility |
| **ticket: threads-one-setting** | `SET threads = 1` in `Oracle.__init__`, beside the pragma, if ask: threads-and-value-order(a) takes that option | claim: threads-setting | ask: threads-and-value-order |
| **ticket: clean-prefix-reconcile** | reconcile `_CLEAN`'s two unprefixed messages with the documented three-prefix rule, either way | claim: refusal-message-prefixes | editorial |
| ~~**ticket: value-preserving-normalization**~~ | ~~make the campaign's schema normalization value-preserving~~ — **done**. ask: unshipped-never-compared ruled that normalization leaves the answer and the verdict; the cast is deleted and `UNSHIPPED` replaces it (claim: unshipped-verdict) | claim: schema-comparison, claim: unshipped-verdict | closed |
| **ticket: string-budget-ground-fix** | correct `known-limitations.md:205`: DuckDB is deterministic on pad/repeat budgets; the "spelling-dependent" ground was measured false and restated 2026-08-16 | claim: keep-entry-reason, divergence: string-builder-budget | editorial |
| **ticket: convert-unrunnable-pins** | enumerate the pins that cannot be re-run mechanically and convert them; this is the bump's actual first task | claim: pin-re-runnability | owner's go |
| **ticket: mined-corpus-stamp** | stamp provenance onto `duckdb_mined.jsonl` at mining time (version, date, settings profile) | claim: mined-corpus-provenance | owner's go |
| **ticket: fuzzer-gate-correction** | correct `known-limitations.md:301-307`: the campaign fuzzer is a manual CLI, not a gate mechanism; what runs is `test_fuzz_smoke.py`, and its invariant is machinery | claim: regexp-fuzz-gate | editorial |
| **ticket: fold-reading-decision** | decide whether the engine's build-time fold moves to the oracle's reading, and write the answer into claim: oracle-identity — today the constant reads as though it had no exceptions. First step is the unmeasured half: run the suite under both readings | claim: one-door-bypass | ask: engine-fold-reading |
| **ticket: verdict-tuple-test** | give `fuzz.runner`'s `INTERESTING` / `COVERED` tuples a test. Nothing in `packages/confit/tests/` imports `fuzz.runner`, so which verdict kinds become findings and which count as coverage — the substance of claim: contract-surface-gap, claim: optimizer-bracket, claim: opt-emulated-classification, claim: abstention-reporting and claim: coverage-accounting — is enforced by an untested tuple | claim: contract-surface-gap, claim: optimizer-bracket, claim: opt-emulated-classification, claim: abstention-reporting, claim: coverage-accounting | owner's go |
| ~~**ticket: static-only-schema-check**~~ | ~~compare schemas on the campaign's static-only path too, where a bare-decimal literal was reported to value-compare across an unshipped width~~ — **refuted by measurement**: a static-tables-only query never prepares, so `Engine::Constant` hands back DuckDB's own rows *and schema* verbatim and no our-side width exists there to classify. Pinned by `packages/confit/tests/test_fuzz_smoke.py::test_the_static_only_leg_has_no_unshipped_width_to_classify`, merged on master | claim: schema-comparison, claim: unshipped-verdict | closed |

Two further editorial corrections found while writing, no ticket needed if fixed in
place: `known-limitations.md:248-249` says a 4000-seed campaign puts trap elision at
"8 seeds in 28 findings; all eight are labelled `DIVERGE_OPT`" — the committed snapshot
holds **7** `DIVERGE_OPT` seeds (312, 812, 1196, 1563, 1564, 2174, 2805) of 28 findings
(counted 2026-08-25 over `packages/confit/findings.jsonl`), and
`2026-08-17-fuzz-triage.md:102` names only five of them in prose, while its table at
`:87` says `OPT_EMULATED = 0` and its prose at `:89-94` says one remains.

And one comment that is *not* merely editorial, because it states the opposite of a
claim in force: the `OPT_EMULATED` block in `fuzz.runner.report` still opens "The passes
we reproduce **on purpose** ... Empty here means no emulation was exercised at all,
which is **a coverage hole rather than good news**", sitting directly above the comment
that corrects it ("Since the oracle became optimizer-off DuckDB these are **BUGS**, not
notes") and above a printed header that says "(each one is a bug)". A reader who stops
at the first comment gets claim: opt-emulated-classification backwards. Fold into
ticket: oracle-docstring-corrections' editorial pass.

---
