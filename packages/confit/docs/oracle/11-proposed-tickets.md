## 11. Proposed tickets

Everything this document wants changed in code or in another document. **None of it is
applied here** — this deliverable is one markdown file.

| id | change | claim | blocked on |
|---|---|---|---|
| **T-1** | correct two statements in `confit/oracle.py`'s module docstring and their twins in `known-limitations.md`: (a) `disable_optimizer` also changes behavior at **twelve** sites outside the 33-pass list, two of them in `src/planner/binder/`; (b) "constant folding still happens" is an observation, not a mechanism — the folder is an `EXPRESSION_REWRITER` rule, so the pragma removes it | ORC-05 | editorial, no ruling needed |
| **T-2** | assert `duckdb.__version__ == Oracle.VERSION` in `Oracle.__init__`, on the reserved line beside the pragma | ORC-09 | ASK-1 |
| **T-3** | number the axiom as P21 in `properties.md`; make its three existing sites cite it | ORC-13 | owner's go |
| **T-4** | correct the source count: `pins-first-methodology.md:89` says two sources, the set holds one (two statements) | ORC-20 | editorial |
| **T-5** | split `REFUSED` into `REFUSED_ORACLE_TRAPS` / `REFUSED_ORACLE_SERVES`, the latter in `INTERESTING` | ORC-30 | ASK-3 |
| **T-6** | put phase-separated probing into the methodology report, where "how we measure DuckDB" lives | ORC-45 | owner's go |
| **T-7** | uniform provenance header on every pin file | ORC-46 | owner's go |
| **T-8** | decision back-reference (`ORC-NN`) field on every pin | ORC-48 | owner's go |
| **T-9** | an inline under-determined-field token in the pin format | ORC-49 | owner's go |
| **T-10** | correct `2026-08-17-fuzz-triage.md:56-58, :70-72`: the ambiguous-reference class is TASK-121, Done | section 7.3 | editorial |
| **T-11** | delete the two parenthetical severity-ladder definitions; cite ORC-57 | ORC-57 | editorial |
| **T-12** | generalize `pin_ast_shapes.py`'s drift-report pattern to the pins corpus | ORC-61 | owner's go — **not** ASK-1(b), which an earlier version listed. Which version the corpus targets has no bearing on whether a re-record tool exists, and that spurious dependency is what made section 9 read as having no first step |
| **T-13** | generate the corpus match count with a date stamp in exactly one place, and correct the **six** unhedged sites ORC-67 enumerates | ORC-67 | ASK-11 |
| **T-14** | extend the coverage histogram to (operator, arg-type, edge-class) triples | ORC-70 | owner's go |
| **T-15** | report an abstention rate per kind per campaign, and classify oracle-side vs engine-side timeouts in the runner rather than by hand (ORC-78's recorded follow-up, plus recording the SQL before executing) | ORC-71, ORC-78 | ASK-5 |
| **T-16** | `SET threads = 1` in `Oracle.__init__`, beside the pragma, if ASK-13(a) takes that option | ORC-75 | ASK-13 |
| **T-17** | reconcile `_CLEAN`'s two unprefixed messages with the documented three-prefix rule, either way | ORC-29 | editorial |
| ~~**T-18**~~ | ~~make the campaign's schema normalization value-preserving~~ — **done**. ASK-12 ruled that normalization leaves the answer and the verdict; the cast is deleted and `UNSHIPPED` replaces it (ORC-92) | ORC-38, ORC-92 | closed |
| **T-23** | decide whether the engine's build-time fold moves to the oracle's reading, and write the answer into ORC-02 — today the constant reads as though it had no exceptions | ORC-91 | ASK-16 |
| **T-19** | correct `known-limitations.md:205`: DuckDB is deterministic on pad/repeat budgets; the "spelling-dependent" ground was measured false and restated 2026-08-16 | ORC-53, D14 | editorial |
| **T-20** | enumerate the pins that cannot be re-run mechanically and convert them; this is the bump's actual first task | ORC-85 | owner's go |
| **T-21** | stamp provenance onto `duckdb_mined.jsonl` at mining time (version, date, settings profile) | ORC-87 | owner's go |
| **T-22** | correct `known-limitations.md:301-307`: the campaign fuzzer is a manual CLI, not a gate mechanism; what runs is `test_fuzz_smoke.py`, and its invariant is machinery | ORC-66 | editorial |

Two further editorial corrections found while writing, no ticket needed if fixed in
place: `known-limitations.md:248-249` says a 4000-seed campaign puts trap elision at
"8 seeds in 28 findings; all eight are labelled `DIVERGE_OPT`" — the committed snapshot
holds **7** `DIVERGE_OPT` seeds (312, 812, 1196, 1563, 1564, 2174, 2805) of 28 findings
(counted 2026-08-25 over `packages/confit/findings.jsonl`), and
`2026-08-17-fuzz-triage.md:102` names only five of them in prose, while its table at
`:87` says `OPT_EMULATED = 0` and its prose at `:89-94` says one remains.

And one comment that is *not* merely editorial, because it states the opposite of a claim
in force: the `OPT_EMULATED` block in `fuzz.runner.report` still opens "The passes we
reproduce **on purpose** ... Empty here means no emulation was exercised at all, which is
**a coverage hole rather than good news**", sitting directly above the comment that
corrects it ("Since the oracle became optimizer-off DuckDB these are **BUGS**, not
notes") and above a printed header that says "(each one is a bug)". A reader who stops at
the first comment gets ORC-25 backwards. Fold into T-1's editorial pass.

---
