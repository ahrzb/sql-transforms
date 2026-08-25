# Pins-first: building a bit-exact engine twin without trusting yourself

**Abstract.** Confit (`DuckDBInferFn`) is a partial evaluator that turns a SQL query plus frozen static tables into a specialized native serving function, bit-exact with DuckDB 1.5.5. Its central engineering problem is not performance — it is that a second implementation of somebody else's SQL dialect is, by default, a machine for producing plausible wrong answers. This report describes the discipline that kept it honest: a two-outcome contract (serve bit-for-bit or refuse loudly at build time), *pins* — measuring DuckDB with recorded, verbatim queries before writing any code — a three-outcome corpus replay mined from DuckDB's own test suite (550 of 678 statements bit-exact, zero failures throughout), an executable limitations document, and a standing differential fuzzer that has kept auditing the translation layer long after the initial battery passed. Every mechanism assumes the same thing: the author's intuition about SQL semantics is wrong until an executed query says otherwise. The costs of the discipline are real and are itemised at the end.

## 1. The two-outcome contract

The engine's user-facing contract (packages/confit/docs/known-limitations.md) is deliberately binary. For any SQL you hand it, exactly one of two things happens:

1. It serves the query **bit-for-bit identical to DuckDB**, or
2. It **refuses at build time** — `DuckDBInferFn(...)` raises a `ValueError` naming the construct.

There is no third mode. Nothing is silently dropped, approximated, or "close enough" at inference time.

The reason is the workload. This is a serving engine: the function it emits runs millions of times against production feature rows, downstream of nobody who is checking. An engine that is 99% compatible does not fail on 1% of queries — it silently corrupts some fraction of rows on queries it *appears* to support, and the corruption surfaces as model skew weeks later, unattributable. A build-time refusal, by contrast, costs one engineer one minute at deploy time and tells them exactly which construct to rewrite. "Mostly right" is therefore worth less than "narrowly right and loud about the edges": the value of the engine is precisely that its yes means yes.

This contract shapes everything below. Refusal has to be cheap, named, and testable; serving has to be verified against the real DuckDB continuously, not once; and any place where the two engines *cannot* agree has to be a named, documented divergence rather than a patch.

## 2. Pins: measure DuckDB before writing any code

The founding design (packages/confit/docs/specs/2026-07-25-sql-specializer-design.md) made DuckDB the frontend check, the prepare-time evaluator of static subtrees, and the differential oracle. The waves that followed added the harder rule: **no semantics are implemented from memory, documentation, or intuition — only from executed queries against DuckDB 1.5.5, recorded verbatim.**

A "pin" is one measured behavioural claim, backed by the exact SQL that was run and the exact result that came back. Each support wave begins with a fleet of parallel measurement agents (eight for the wave-5 structural sweep, six for the regexp wave, five for stage-B join multiplicity) probing one semantic family each. Their findings land as a pins spec — packages/confit/docs/specs/2026-07-26-wave5-structural-pins.md, 2026-07-27-waveB-regexp-pins.md, 2026-07-28-waveA-structural-tails.md, 2026-07-28-stageB-multiplicity-pins.md — with the raw evidence committed alongside in `pins-wave5/`, `pins-waveB/`, `pins-waveA/`, `pins-stageB/` as JSON: query text, input reprs, result reprs, float bit patterns, verbatim error heads. Implementation starts only after the pins exist; the pins are the contract the code is written to.

### The incident that created the verbatim rule

The discipline was not designed in the abstract; it was bought. During wave 3, the fleet's summary claimed that `%`-by-zero returns NULL — generalising from the integer probes, which do return NULL. The DOUBLE case was never actually run. It returns NaN (`7ff8...`), not NULL. The correction is appended to packages/confit/docs/specs/pins-wave3/math_tail.json with the honest note: *"raw probes never covered this cell (only int rows + fmod(1.0,0.0)); the summary over-generalized."* (The follow-up was itself instructive: the NaN's sign bit turned out to be platform-libm — `7ff8` on Windows ucrt, `fff8` on Linux glibc — so the pin is *bit agreement with the oracle per platform*, not a constant.)

Every wave dispatched since carries the rule explicitly — the task briefs for waves 5 and B (backlog/tasks/task-52\*, task-53\*) both read: *"wave-3 over-generalization precedent applies — every pin claim needs an executed query recorded."* A summary sentence with no query behind it is treated as a guess, because once, it was.

### Four pins that would have been silently-wrong guesses

The point of pins-first is best made by the pins no reasonable engineer would have predicted:

- **`reverse()` has two code paths, and the "correct" one is wrong on ASCII.** DuckDB byte-reverses all-ASCII strings — which *splits CRLF*: `'a\r\nb'` → `'b\n\ra'`, violating UAX-29. Only non-ASCII input takes the extended-grapheme-cluster path. A clean, pure UAX-29 implementation — the obvious thing to write — diverges from DuckDB on plain ASCII input. The engine reproduces both paths (packages/confit/docs/specs/2026-07-28-waveA-structural-tails.md §4, pins-waveA/reverse-graphemes.json).
- **DuckDB's join output order is an accident, not a contract.** The stage-B fleet found it is a hash-join artifact on three independent axes: the optimizer picks the streamed side by cost, duplicate-key matches emit in *reverse* build-insertion order (LIFO chains) in per-2048-row lockstep passes, and at multiple threads with ~500k+ rows the order differs run-to-run on the same connection. A row-at-a-time engine cannot reproduce this and must not try. The decision: parity for `shape='many'` is **multiset**, and the engine defines its own documented deterministic order — probe rows in input order, matches contiguous in build-insertion order (packages/confit/docs/specs/2026-07-28-stageB-multiplicity-pins.md, pins-stageB/order-contract.json). Chasing byte-order parity here would have meant chasing a nondeterministic target.
- **Paren-less `* REPLACE e AS c` consumes exactly one item.** A following comma starts a *new select item*: `SELECT * REPLACE i+100 AS i, j+1 AS j` yields three columns, `i, j, j` — measured, duplicate name and all (pins-waveA/columns-replace.json). Any parser written from the grammar one imagines would have absorbed the second item into the REPLACE list.
- **Double-quoted identifiers are still case-insensitive in struct EXCLUDE.** `a.* EXCLUDE("J")` removes field `j`, quoting notwithstanding (pins-waveA/struct-star.json). The intuition that quoting forces case-sensitivity is simply false here.

The pins also cut the other way — where DuckDB's behaviour is a quirk (`^` is power, not xor; `~` is full-match, not search; `SIMILAR TO` does no wildcard translation), the quirk is reproduced, not "fixed". As the wave-5 spec puts it: pins are engine==oracle contracts.

## 3. The corpus replay: three outcomes, zero FAILs

Pins prove behaviours in isolation; the corpus proves the assembled engine. `scripts/mine_duckdb_corpus.py` extracts 678 statements from DuckDB's own test suite into packages/confit/tests/corpus/duckdb_mined.jsonl, and packages/confit/tests/test_corpus_replay.py reconstructs each case's tables in a fresh DuckDB, feeds them through `DuckDBInferFn`, and classifies:

- **match** — engine output equals the mined rows;
- **clean-unsupported** — a build-time rejection naming the limit;
- **FAIL** — mismatch, wrong error, or crash.

The gate requires **zero FAILs**, always. The match count is deliberately *not* gated — it is the growth ladder. Every construct the engine learns flips cases from clean-unsupported to match; nothing is ever allowed to flip into FAIL:

| bit-exact matches (of 678) | after |
|---|---|
| 53 | first replay of the mined corpus |
| 395 | the builtin and join waves (census baseline recorded in the wave-5 spec) |
| 505 | the structural and regexp waves |
| 511 | small-tails sweep (before wave A, per its spec) |
| 529 | wave A structural tails |
| 546 → 550 | stage-B join multiplicity, across its two PRs |

Zero FAILs at every rung. The 128 remaining non-matches are all clean, named rejections — aggregation, ORDER BY, CTEs, and the other whole-relation constructs that are out of scope for a row-serving engine by decision (packages/confit/docs/known-limitations.md §2).

The three-outcome shape matters more than the ladder. A conventional pass/fail suite would have forced a choice between skipping unsupported cases (hiding regressions in the rejection surface) and marking them expected-fail (letting wrong errors hide among right ones). Here, a rejection is only clean if it is one of the *documented* rejection classes; an undocumented error is a FAIL like any wrong answer.

## 4. The executable limitations twin

packages/confit/docs/known-limitations.md would rot like any other document if it were only a document. Its executable twin is packages/confit/tests/test_known_limitations.py, whose module docstring states the mechanism plainly: every deliberate limitation in the document is asserted as a test — the SQL that hits it and the named build-time rejection. **If an engine change lifts a limitation, a test fails, and the document must change in the same commit.**

This inverts the usual failure mode. Normally, docs describe capabilities and silently lag behind; here the doc describes *incapabilities* and mechanically cannot lag, in either direction. The `reverse()` story ran through this machinery end to end: descoped in wave 3 (grapheme segmentation for three corpus cases failed the cost test — a named rejection citing grapheme semantics), pinned anyway, then lifted in wave A with the instruction "remove the limitations row + flip its twin test in the same commit" (packages/confit/docs/specs/2026-07-28-waveA-structural-tails.md §4).

## 5. Differential fuzzing as a standing gate

The regexp wave translated DuckDB's RE2 patterns to rust-regex behind a measured reject list, validated by a 98-entry differential battery. A one-time battery, however, only guards the constructs its author thought of — the residual risk is exactly the pass-through path. So TASK-54 made the differential a *standing* gate: packages/confit/tests/test_duckdb_regexp_fuzz.py generates patterns from a grammar biased toward the divergence-prone axes and asserts, per case: identical rows, or a conservative engine reject, or both engines error. DuckDB-serves-while-we-mismatch and DuckDB-errors-while-we-serve both fail, with seed, case index, and SQL in the message. It runs at N=250 in the normal gate (~4s, fixed seed) and takes `REGEXP_FUZZ_SEED`/`REGEXP_FUZZ_N` for deep runs.

The first deep run vindicated the design: 122 divergences, distilled to 12 reject classes — including three silent wrong-answer shapes (rust set-notation `--`/`&&`/`~~` in character classes, whitespace inside `{1, 3}` bounds, non-POSIX `[` inside classes) — then a re-sweep to **zero divergences over 40k cases across 8 seeds** (pins-waveB/fuzzer-task54.json).

Then seed 20260728 earned the "standing" in the name, catching two more:

- **An oracle self-inconsistency.** One catch was a pattern family on which DuckDB's two evaluation paths return different answers for the same input — there is no single behaviour to be bit-exact *with*, so the family is rejected by name. (The specifics are recorded in the pins evidence and held for a separate upstream-issues report.)
- **RE2's post-simplification program-size budget.** `(\p{L}){1,500}` is "pattern too large" in DuckDB but serves fine in rust-regex. The exact budget point is an RE2 internal, so the guard is a deliberately **one-sided** weight estimate: it always fires before DuckDB's real budget. It may over-refuse; it can never serve where DuckDB errors. The asymmetry is the contract.

Both are pinned in pins-waveB/fuzzer-20260728.json and rowed into packages/confit/docs/known-limitations.md §4.

## 6. When the oracle is not reproducible row-locally

Two kinds of case break the clean oracle picture, and both are handled by naming rather than patching.

The first is the self-inconsistency family above: where the oracle's evaluation paths disagree with each other, there is no behaviour to be bit-exact *with*; the constructs are rejected, with the measurement recorded.

The second is subtler: DuckDB behaviours that depend on **column statistics**. The measured exemplar involves `ILIKE` on strings with embedded NUL bytes: the result for a given row can change depending on which *other* rows are present in the column, because the engine selects its comparison kernel from column-level statistics. The result for one row depends on its siblings. A row-at-a-time engine cannot reproduce this even in principle. The response is a **named divergence**: the source file is listed in `packages/confit/tests/test_corpus_replay.py::_KNOWN_DIVERGENT_SOURCES`, each entry required to cite a measured reason the divergence is irreproducible row-locally, and the engine's own behaviour (NUL-transparent) is documented in packages/confit/docs/known-limitations.md §5. Two such sources exist. They are excluded from the corpus *by name* — not silently skipped, not fudged to pass.

## 7. What the discipline costs

Honesty requires the bill:

- **Measurement time.** Every wave front-loads a fleet before a line of engine code: 300+ probes for wave 3's builtins, 57 pins across 5 agents for stage B, an exhaustive 1,112,063-codepoint sweep to establish that `ord == unicode`. Much of this measured things that turned out to be exactly what one would have guessed — the pins only pay for themselves on the surprises, and you cannot know in advance which probes those are.
- **Over-rejection.** The one-sided regex budget refuses patterns DuckDB would serve. Mixed BETWEEN/IN with non-numeric string literals is conservatively unsupported because DuckDB converts at execution time (an empty input succeeds). Narrow-integer CASTs are rejected rather than modelling narrow-width overflow. Every conservative choice is a query someone must rewrite — the deliberate price of never serving a wrong answer.
- **Reject-list maintenance.** The regex reject list is now a small grammar of its own — a quantifier-state machine, a POSIX-class tracker, a rewrite-template pre-scan in `retrans.rs` — and each fuzzer catch grows it. The verbatim error-text reproduction (overflow traps with operand values, DuckDB's exact shift-error ladder) is upkeep against a moving upstream.
- **Named divergences are still divergences.** DECIMAL literals compute as f64; the schema-less registry accepts qualifiers DuckDB would reject. Each is measured, loud, and documented — but the list must be curated forever.

## 8. The same discipline, applied to performance claims

"Validate, don't assume" governed the performance work too, and twice killed attractive code. The TASK-57 boundary decomposition (titanic scenario: 10 input columns, 31 output columns, release builds, p50) established where serving time actually goes:

| component | measured cost |
|---|---|
| whole boundary floor (trivial 1-col query, incl. 10-col ingest) | ~262 ns/row |
| output emission | ~37 ns per output column |
| compute (the compiled program) | ~1.7 µs/row |
| handcrafted Python twin — everything | ~2.2 µs/row total |

A `__dict__` ingest fast path was built, benched **neutral**, and deleted. Against DuckDB itself, given a pre-built Arrow table per call (packages/confit/docs/proposals/2026-07-28-columnar-path.md):

| rows/call | DuckDB | us (row path) | ratio |
|---|---|---|---|
| 1 | 6.58 ms | 3.3 µs | 2055× faster |
| 64 | 6.75 ms | 206 µs | 33× |
| 1024 | 7.75 ms | 3.42 ms | 2.3× |
| 16k–262k | — | — | DuckDB wins 3–5× |

Crossover ~2–3k rows/call. The Arrow boundary (`infer_arrow`, row core) beats the row-object path everywhere at n ≥ 1024 and beats the handcrafted twin on most scenarios (house_prices 1.8× faster than the twin). The v1 columnar *core*, however, measured at compute parity with the row core — its kernels call the same scalar helpers per row — and was closed **unmerged**, by decision: large-batch columnar is ceded to DuckDB, and the sub-5k serving regime is the product. The work is preserved on branches `task-61-columnar-core` / `task-61-columnar-exec` (after `git fetch origin`, see `git show origin/task-61-columnar-core:benchmarks/scaling_results.json`). Merging code that measures at parity would have been the performance edition of serving a plausible wrong answer.

## Closing

None of the individual mechanisms is novel — differential testing, mined corpora, executable documentation, and fuzzing are all standard tools. What transfers is the arrangement: every layer is built on the assumption that the layer's author is wrong. Semantics are measured before they are implemented (and re-measured after one summary lied); the corpus permits growth but never regression; the limitations document cannot drift because it breaks tests; the fuzzer keeps auditing the translation layer after everyone has stopped worrying about it; and where the oracle itself is incoherent, the incoherence is named rather than smoothed over. The result — 550 of 678 statements bit-exact, zero wrong answers across the entire arc, and a refusal message for everything else — is modest by design. That modesty is the product.
