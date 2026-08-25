## 7. The divergence ledger

### 7.1 The split is by intent, and it stays

**ORC-50.** The record is split by INTENT, not by severity:
`packages/confit/tests/known_divergences/` holds behavior we decided to **KEEP** — all
passing, each entry owing a measured REASON — and
`packages/confit/tests/test_open_divergences.py` holds behavior we decided to
**CHANGE** — one `xfail(strict=True)` pin each, ticket named, deleted rather than
edited when it closes. The ground for the split is a measured census, not a
preference: it found readers both implementing something we chose not to have, and
walking past a real bug because the paragraph above it sounded like a rationale.
*Verified-by:* `packages/confit/tests/known_divergences/README.md:19, :35-46`;
`packages/confit/tests/test_open_divergences.py:9-25`.

**ORC-51.** `strict=True` is the load-bearing part of the CHANGE ledger: a pin that
silently starts passing is worse than no pin, because it certifies work nobody did.
The marker self-expires — closing a divergence makes its pin fail loudly.
*Verified-by:* `packages/confit/tests/test_open_divergences.py:27-28`.

**ORC-52.** The CHANGE ledger is **empty as of 2026-08-25**, deliberately, with named
successor tickets. An empty CHANGE ledger with a named successor is the mechanism
working, not an absence of divergences. It has emptied and refilled inside a single day
before; that rhythm is intended.
*Verified-by:* `packages/confit/tests/test_open_divergences.py:30-35`;
`backlog/tasks/task-134 ...md` (To Do).

**ORC-53.** A KEEP entry owes a REASON, not just a description, and where the reason is
a claim about DuckDB it must be measured and must stay true. One had already gone false
and was propagating into a user-facing message when the census caught it.
*Verified-by:* `packages/confit/tests/known_divergences/README.md:44-46`.
*Live instance, uncaught:* the string-budget entry's ground was restated 2026-08-16
because the old one was false — measured, DuckDB is entirely deterministic there ("no
coin flip, no spelling-dependence"), so the honest ground is ours and is a judgement.
`known-limitations.md:205` still carries the **corrected-false** claim that "DuckDB's
own behaviour is spelling-dependent". See proposed ticket T-19 and ledger row D14.

**ORC-80.** "Known divergence" is reserved for **decided-and-unscheduled** differences —
the rows in `known-limitations.md` and in section 7.3 below. Anything carrying an m-8
phase or a ticket is a **feature in flight**, its tests live with the feature, and its
markers (xfail pins, fuzzer suppression tags) are **scaffolding**. Each phase's
definition of done includes deleting that phase's markers in all three homes, in the
same PR as the feature: (1) the xfail-strict pin flips to a real parity test — enforced,
because strict xfail turns XPASS-loud the moment the feature lands; (2) the
known-limitations row goes — enforced by its executable twin; (3) the fuzzer's
suppression tag is removed — **unenforced**, stated in the design precisely because a tag
that outlives its phase would silently swallow regressions in the code the phase just
changed. The certification campaign *after* the tag removal is what proves the class is
gone rather than hidden. Decided with the owner 2026-08-11.
*Verified-by:* `packages/confit/docs/specs/2026-08-11-duckdb-type-lattice-design.md:110-131`.
This rule is what classifies D7, D8 and D11; the unenforced third home is ASK-6(c).

### 7.2 Doc-twin totality is partial

**ORC-54.** **Five sites across four documents** assert that every limitation has an
executable twin. The code denies it in-file, TASK-95 is open with both acceptance
criteria unchecked, and the measured shape is: `test_known_limitations.py` holds 14 test
functions (two parameterized) against roughly three dozen enumerated limitations, with
several sections' twins living elsewhere rather than in the named twin file. Where they
actually live, per the ledger's own evidence column: `test_arrow_schema_api.py` for the
row-limit rule; `test_corpus_replay.py` for D2 and D3; `test_duckdb_wave3_mathtail.py`
for D5; **no test at all** for D6; and of D2-D6 only D4 is in `known_divergences/`
(D8 and D9, outside that range, are).
*Verified-by:* the claims at `packages/confit/docs/known-limitations.md:5-7` and
`:294-296` (two sites, one document), `packages/confit/docs/reports/pins-first-methodology.md:66`,
`packages/confit/docs/kpis.md:58-61`, and
`packages/confit/docs/reports/confit-architecture.md:150` ("each is a named build-time
rejection with an executable twin" — the fifth site, missed by an earlier version of this
claim); the denial at
`packages/confit/fuzz/oracle.py:126-131` ("doc-twin accounting — a strict-xfail pin
standing behind every known-limitations row — does not exist yet");
`backlog/tasks/task-95 ...md` (To Do, ACs `[ ]`); counts measured 2026-08-25.
*Status:* the mechanism works for the rows it covers. The overstatement is the problem,
because it stops readers checking. See ASK-7.

> ### ASK-7 — close TASK-95, or downgrade the four claims to what is true?
>
> An overstated guarantee is worse than a stated partial one. Two ways:
>
> - **close TASK-95** — a known-limitations row without a named twin fails a unit test,
>   and missing twins get written. This is the sig.rs totality-test pattern applied to
>   prose; it was approved 2026-08-13 and never built.
> - **downgrade the claim** at all five sites to "every limitation with a twin is
>   asserted; the twins are enumerated here", and drop the totality word. Note the fifth
>   site (`confit-architecture.md:150`) was missed by the original version of this ASK,
>   so a remedy scoped to "four documents" would have left it standing.
>
> Recommendation, if you want the cheap one: downgrade now, keep TASK-95 open. The
> claim costs nothing to correct and currently buys false confidence.
>
> *Binds:* ORC-54, and the four documents named in its Verified-by.

### 7.3 The ledger

**ORC-55.** The table below is the enumeration of currently-tolerated divergences —
engine-vs-oracle, plus two rows (D4, D16) that are kept here because readers look for
them and the row says plainly what each one actually is. Each row is a decision, so each
carries a proposed status from the section 3 vocabulary plus a permanence, and each
awaits an owner ruling. **This table is the adjudication surface: `accept` writes the
proposed status into force, `reject` sends the row back as a defect owing a ticket.**
Nothing in the `proposed status` column is in force until the ruling column is filled —
which means that today no row's status is in force, so under ORC-15's rule as written
none of them is yet a comparison target. That tension is real and is part of ASK-15;
until it is resolved, read the `proposed status` column as this document's
recommendation, not as the vocabulary's application.

*Severity* is the ladder of section 8: 1 = trap where DuckDB serves, 2 = wrong value,
3 = serve where DuckDB refuses, 4 = refuse where DuckDB serves.

| id | behavior | measured evidence | sev | proposed status | OWNER RULING |
|---|---|---|---|---|---|
| **D1** | duplicate output names renamed by DuckDB's own boundary algorithm; applied to *both* sides before comparison | `pins-wave5/dup-names-client-contract.json`; `oracle.py:467-488, :646-651`; twin `test_known_limitations.py:255` | n/a (contract) | `PINNED` / permanent | |
| **D2** | error texts approximate where noted; the corpus compares successes only | `known-limitations.md:219-224`; `test_corpus_replay.py:173-176` | n/a | `UNSPECIFIED` / permanent | |
| **D3** | `ILIKE` with embedded NUL is statistics-dependent; engine is NUL-transparent; source excluded by name | `test_corpus_replay.py:40-49`; `pins-wave1/pins_like.json` | n/a | `UNSPECIFIED` / permanent | |
| **D4** | a trapping subexpression the optimizer deletes, we still evaluate — the standing price of optimizer-off. **Not an engine-vs-oracle divergence**: against the oracle we agree exactly, so the gap is engine-vs-the-user's-optimizer-on-DuckDB, and ORC-06 reports it (`DIVERGE_OPT`) rather than accepting it. Listed here because it is the tolerated cost readers look for | `known-limitations.md:231-257`; `known_divergences/test_trap_elision.py`; campaign snapshot `packages/confit/findings.jsonl` (see D16) | 1 **against the contract surface**, n/a against the oracle | `PINNED` / permanent | |
| **D5** | `%`-by-zero NaN sign bit is platform-libm; the pin is per-platform bit agreement, not a constant | `test_duckdb_wave3_mathtail.py:205-235`; `pins-wave3/math_tail.json` | n/a | `IMPL-DEFINED` / permanent, discriminator = platform | |
| **D6** | schema qualifiers are registry-noise: `s1.t1` resolves on the bare table name; DuckDB's schema-existence errors are not reproduced; `w.w.w` binds the longer schema-ish parse | `known-limitations.md:260-272` | **3 for the first two clauses** (`s1.t1` on a non-existent schema serves here and gets `schema "x" does not exist` on DuckDB), **4 for `w.w.w`**. The source's "always a loud build-time rejection, never a different served value" closes the `w.w.w` sub-case only and was mis-lifted to the whole row | `PINNED` / permanent | |
| **D7** | DECIMAL **literals** are f64; exact-decimal accumulation is not reproduced. Three visible faces: a schema delta (decimal128 vs float64), a rounding-mode delta (D8), and a signed-zero delta (`- 0.0` has no sign as DECIMAL) | `known-limitations.md:166-174`; `oracle.py:113-131, :512-513`; faces measured 2026-08-25 over `findings.jsonl` seeds 869/998/1554/3269 | 2 | *unruled* — see ASK-6; the reported value delta also depends on ASK-12 | |
| **D8** | **D7's named consequence, not an independent divergence.** `CAST(-2.5 AS BIGINT)`: DuckDB types the bare `-2.5` as `DECIMAL(2,1)` and casts DECIMAL->BIGINT half away from zero (`-3`); we type it f64 and cast DOUBLE->BIGINT half to even (`-2`). Both engines agree on both casts *given the type* — the divergence is entirely the literal's type | `known-limitations.md:166-174` treats it as one limitation with "one visible consequence". **Not** `known_divergences/test_cast_semantics.py`: that file records the DOUBLE cast as FIXED 2026-08-08, contains no bare-literal test, and warns "Measure a DOUBLE cast with a DOUBLE column or an explicit `::DOUBLE`, never a literal", assigning the mechanism to D7 | 2, **the same instance as D7** | *unruled with D7* — see ASK-6 | |
| **D9** | bind-time constant refusals: `WHERE FALSE` and empty-input shapes refuse here and serve on DuckDB | RFC `2026-08-19-keep-the-bind-time-refusals.md:29-58` (the phase-separated measurement) and the decision ACCEPTED twice. **No executable twin exists**: `known_divergences/test_literal_typing.py:77-131` pins refusal where DuckDB *traps* (its DuckDB leg asserts `pytest.raises(..., "[Oo]verflow")`), which is the absorbed case, and `WHERE FALSE` appears in the suite only as a DuckDB probe in `test_trap_elision.py`. The missing twin is the same absence ASK-3 is about | 4 | `PINNED` / permanent — **but its cost is uncountable and unpinned, see ASK-3** | |
| **D10** | one-sided regex program-size guard: always fires before DuckDB's real RE2 budget, so it may over-refuse and can never serve where DuckDB errors | `pins-waveB/fuzzer-20260728.json`; `pins-first-methodology.md:79` ("the asymmetry is the contract") | 4 by construction | `PINNED` / permanent | |
| **D11** | narrow-lane overflow trap threshold not yet shipped: an overflowing narrow lane serves the i64 value on the row path and refuses by name at the `infer_arrow` boundary | `known-limitations.md:177-188`; catalogue pinned in `test_integer_widths.py` | 3 (row path) | `PINNED` / **until-fixed** (m-8 phase 3) — owes a strict-xfail twin if until-fixed is accepted | |
| **D12** | ~~unattributed~~ **attributed** campaign residuals in the committed 2026-08-17 baseline. Re-measured 2026-08-25, four of the five are attributed and the row's original premise was wrong: seeds **869, 1554, 3269** carry the `decimals` tag and are bare-DECIMAL-literal cases, i.e. **D7**, and their reported 1-ulp delta is produced by the harness's own cast (ASK-12); seed **998** is `(c3.f1 * (- 0.0))`, the same D7 mechanism in its signed-zero face (ORC-33); seed **2668** (`-2147483648 / -1`) is **TASK-122, Done**, whose AC #5 records "the campaign's seed 2668 class is gone at 4000 seeds". What survives as genuinely unruled is not a residual set but a question about the file: see D16 | `packages/confit/findings.jsonl`, counted 2026-08-25: 28 findings = 16 `DIVERGE_BUILD` + 7 `DIVERGE_OPT` + 4 `DIVERGE_VALUE` + 1 `DIVERGE_TRAP`, tags and SQL read per seed; `2026-08-17-fuzz-triage.md:62-63`; `backlog/tasks/task-122 ...md:63, :86-88` | 2, **as D7** | *ruled by attribution* — D7's ruling covers 869/998/1554/3269; 2668 is closed. See ASK-6 and ASK-12, not ASK-9 | |
| **D13** | the phase-2 width residuals, quoted from memory as 79 of 84, treated as a defect count | **Unverified** — the numbers appear nowhere in the tree (searched `packages/confit/docs/`, `backlog/`, `findings.jsonl`, 2026-08-25) | unknown | *unruled* — see ASK-10 | |
| **D14** | the **1 GiB string-builder budget**: a literal pad/repeat count that can exceed it refuses at build by name. Measured 2026-08-16, DuckDB is entirely deterministic here (`repeat` serves to n <= 4294967295 and errors above; `lpad`/`rpad` binder-error above INT32), so the ground is ours and is a judgement: "a serving engine does not allocate a gigabyte per row. **We refuse where DuckDB would serve**" | `known_divergences/test_string_budget.py:108-132` (a passing KEEP entry with a restated measured REASON); `known-limitations.md:205`, which **still carries the corrected-false ground** (ORC-53) | 4 | `PINNED` / permanent, resource class (ORC-79) | |
| **D15** | the **2 GiB-per-arrow-batch ceiling** that comes with matching DuckDB's `pa.string()` 32-bit offsets: refused by name rather than wrapped | `known_divergences/test_arrow_boundary.py:34-36` | 4 | `PINNED` / permanent, resource class (ORC-79) | |
| **D16** | **the baseline file's standing as evidence.** `packages/confit/findings.jsonl` is cited as evidence by D4 and by section 11's corrections, and it is a **snapshot of the 2026-08-17 campaign, not a live artifact**: TASK-129's notes record "the 8% draw shifts rng for static-bearing seeds, so residue seed IDs moved ... The committed findings.jsonl baseline is a historical artifact and was not regenerated here", and at least two of its classes are since closed (TASK-121's 16 `DIVERGE_BUILD`, TASK-122's seed 2668) while still sitting in the file. Any instruction phrased as "re-run seed N" against it is not executable as written | `backlog/tasks/task-129 ...md:148-150`; `backlog/tasks/task-121 ...md:84` (78/78 re-classify `REFUSED`); `backlog/tasks/task-122 ...md:86-88` | n/a (evidence hygiene) | *unruled* — see ASK-14 | |

*Verified-by:* each row cites its own evidence. The enumeration is complete against
`packages/confit/docs/known-limitations.md` section 5 plus section 3's value-family
rows, `packages/confit/tests/known_divergences/` (all ten modules, re-swept 2026-08-25 —
which is how D14 and D15 were added), and the committed campaign snapshot, all as of
master `85b4739`. Note D8 is D7's consequence and D4 is not an engine-vs-oracle
divergence at all; counting *distinct tolerated engine-vs-oracle divergences* the table
holds fewer rows than it has ids, and the ids are kept stable rather than renumbered.

**Closed, deliberately not a row:** the 16-seed `DIVERGE_BUILD` ambiguous-reference
class (the largest single class the 2026-08-17 campaign saw, 57% of findings) is
**TASK-121, status Done**. Its closure evidence is the implementation note "All 78
ambiguous findings of the 20k campaign now classify `REFUSED` (78/78 re-run
individually)" at `backlog/tasks/task-121 ...md:84` — **not** its acceptance criteria,
which are all five unticked, including "#5 the campaign's `DIVERGE_BUILD` ambiguity class
is gone at 4000 seeds". Stated plainly because ORC-54 uses unticked ACs to convict
TASK-95, and the same standard has to apply here: the class is closed on a different
campaign than AC #5 names, and the snapshot file still holds all 16 seeds (D16).
`packages/confit/docs/2026-08-17-fuzz-triage.md:56-58` and `:70-72` still say it is "not
yet ticketed" and "has no ticket". See proposed ticket T-10.

### 7.4 Placement

**ORC-56.** **[PROPOSED]** Not in force. Where confit deliberately does not match DuckDB,
the note belongs **at the requirement it violates**, with this ledger as the index; a
divergence filed only in an appendix stops being read. D9 (bind-time refusals) and D10
(one-sided guards) are the two rows this would apply to today. The honest obstacle: there
is no engine spec to place them in, so adopting this means either naming the destination
document or accepting that "the requirement it violates" is `known-limitations.md`'s own
section, which is where they already are.
*Verified-by:* Unverified — the anti-pattern is real (it is the reason ORC-53's census
found readers walking past entries), but no decision outside this document states the
placement rule. Part of ASK-15.

> ### ASK-8 — does "an unlisted divergence is a bug by definition" go in the spec?
>
> This one sentence converts "bit-for-bit or refuse" from an aspiration into a
> falsifiable promise. It also binds us: it makes the section 7.3 table's completeness
> a contract, and any divergence found in the wild becomes automatically a defect
> rather than a discussion.
>
> Recommendation: adopt the sentence, without an SLA. The SLA form exists to serve
> external implementors on a clock; this project has no external claimants.
>
> *Binds:* ORC-55 and every future divergence.

> ### ASK-9 — do we admit a "measured but not yet ruled" bucket?
>
> KEEP and CHANGE both presuppose a ruling (ORC-50). Things that are measured facts
> awaiting your call today live only in prose: build-vs-build repeatability (ORC-22),
> intra-value order under `threads` (ORC-75), the width residuals (ledger D13), and the
> baseline's standing as evidence (ledger D16). A `tentative` **tag** — not a third
> directory — would give them a home without promoting them to contract.
>
> The stricter alternative is that everything measured gets ruled at measurement time.
> That is slower, and it is a real option: it means a campaign cannot end until its
> residuals are classified. The repo already leans this way — the 2026-08-13 triage
> closes with "Every family is now mapped to a ticket", and the 2026-08-17 triage
> attributes every family it found. D12's original "no decision, no ticket" framing was
> wrong precisely because that practice is already in force.
>
> *Correction to this block's original premise:* it named D12's residuals as the second
> of three unruled things. They are attributed (see D12), so the open set is the four
> named above.
>
> *Verified-by (the facts):* `packages/confit/docs/2026-08-13-fuzz-triage.md:150-165`
> ("Every family is now mapped to a ticket").
>
> *Binds:* ORC-22, ORC-75, ledger rows D13 and D16.

> ### ASK-14 — is a campaign baseline evidence, or a snapshot?
>
> `packages/confit/findings.jsonl` is cited across this document as standing evidence,
> and it is a snapshot of one campaign that the tree itself records as superseded (ledger
> D16). Two of its classes are closed by Done tickets and still sit in the file, and its
> seeds are not re-addressable: TASK-129 records that the 8% static draw shifted the rng
> for static-bearing seeds, so a seed id no longer names the case it named in the file.
>
> That makes any instruction of the form "re-run seed N and classify it" — which is what
> ASK-10's remedy would ask for — unexecutable as written; the cases are recoverable only
> from the `sql` field stored in the file.
>
> Three options:
>
> - **regenerate on a cadence and commit the result**, so the file is always the current
>   campaign and "re-run seed N" means something;
> - **freeze it as a dated artifact** and rename it accordingly, so it is read as history
>   and never cited as current state;
> - **stop committing it** and cite triage documents (which carry dates and attributions)
>   as the evidence instead.
>
> Whichever way it goes, the operative sentence this document owes is: *a finding is
> addressed by its stored SQL, not by its seed, unless the generator is unchanged.*
>
> *Verified-by (the facts):* `backlog/tasks/task-129 ...md:148-150`;
> `backlog/tasks/task-121 ...md:84`; `backlog/tasks/task-122 ...md:86-88`.
>
> *Binds:* ORC-66, ledger rows D4, D12, D16, and ASK-10.

---
