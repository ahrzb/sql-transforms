## 9. Version bumps and mutability

The urgency is external: DuckDB ships minor versions roughly every four months,
semantics have already moved inside a patch release, and v2.0 brings a new SQL parser.
This protocol is cheap to write now and expensive to write during a migration.

**ORC-60.** **[FACT]** The bump protocol's object is `confit.oracle.Oracle.VERSION`: the
version a bump changes is that constant, and everything below is written against it.
Its two preconditions — that the constant is **binding** and that it appears in **every
pin file** (ORC-46) — are today unmet and partial respectively. `Oracle.VERSION` is
recorded and never compared to `duckdb.__version__` (ORC-09), so a bump can happen
without the constant moving at all, which is the failure mode the protocol exists to
prevent.
*Verified-by:* `packages/confit/confit/oracle.py:74-76` (the constant and the comment
reserving the assert); ORC-02, ORC-09, ORC-46.

**ORC-85.** **[FACT]** Step one of a bump is not re-recording; it is making the corpus
re-runnable, and that work is not ticketed anywhere. Measured 2026-08-25 over the 53 pin
files: **21 distinct top-level key sets**; the claim body is spelled `probes` / `pins` /
`findings` / bare domain keys; the query field is spelled four ways (`query` in 20 files,
`sql` in 12, `q` in 13, `expr` in 7); a header key exists in **2** files (`_meta` in one,
`meta` in one); inputs are frequently prose rather than data
(`"input_repr": "t(a BIGINT, b BIGINT); rows=[(7, 2)]"`, or an `expr` string holding
three queries at once, or setup buried in a free-text `note`); and exactly **one** pin
ships a re-runnable capture harness beside it (`pins-dialect/probe_joins.py`). The
operative sentence this document owes the next migration: **a pin that cannot be re-run
mechanically is not re-recordable, and enumerating and converting those is the bump's
first task** — before ORC-61's diff report has anything to run against.
*Verified-by:* measured 2026-08-25 over `packages/confit/docs/specs/pins-*/*.json` and
`pins-*/`. Proposed ticket T-20.

**ORC-86.** **[FACT]** Pin *capture* runs outside the oracle, and at least one pin file
was demonstrably captured with the optimizer **on**. The ban on raw connections is read
off `tests/` and `fuzz/` (ORC-07); `scripts/` is outside it, and capture scripts are not
tests. Measured: no capture script applies `PRAGMA disable_optimizer` —
`scripts/gen_casemap.py`, `scripts/gen_pow10.py`, `scripts/gen_strip_accents.py` and
`scripts/mine_duckdb_corpus.py` all use a bare `duckdb.connect()`, and none of them
imports `confit.oracle` — and `pins-stageB/order-contract.json`'s own notes
("The optimizer picks stream/build sides by cost", "Chained-join nesting is
optimizer-chosen") describe an optimizer-on capture. Only 4 of 53 pin files mention the
optimizer at all. By ORC-02's own sentence — a comparison run against anything else is
not a comparison against the oracle — a real part of the corpus is not a recording of
*this* oracle, and this document does not currently say whether that matters. It is
recorded here as state, not ruled: see ASK-1 and proposed ticket T-7.
*Verified-by:* measured 2026-08-25; `scripts/mine_duckdb_corpus.py:111`;
`packages/confit/docs/specs/pins-stageB/order-contract.json` (its own notes).

**ORC-87.** **[FACT]** The mined corpus has no provenance and its expectations are
optimizer-on. `scripts/mine_duckdb_corpus.py` recomputes every case's expected rows by
running it in a fresh `duckdb.connect()` at mining time — deliberately ignoring the
sqllogictest file's own expected block, "and DuckDB itself is our oracle anyway" — at an
unrecorded version, on an unrecorded date, with the optimizer on, and
`tests/corpus/duckdb_mined.jsonl` carries no provenance field. The 550-of-678 headline
(ORC-67) rests on that artifact, and ORC-46's provenance measurement scopes to `pins-*/`
only, so the largest single recorded artifact in the project sits outside it.
*Verified-by:* `scripts/mine_duckdb_corpus.py:1-12` (the recompute rule) and `:111` (the
bare connect); `packages/confit/tests/corpus/duckdb_mined.jsonl` (678 lines, no
provenance field). Proposed ticket T-21.

**ORC-61.** **[PROPOSED]** Partly in force; the generalization is not. Re-record means one
command that re-runs a pinned artifact against a new build and emits a **diff report,
never a silent rewrite** — a silent re-record is indistinguishable from adopting
whatever the new build does, which is the opposite of pinning. The pattern already
exists, correctly, for exactly one artifact: `scripts/pin_ast_shapes.py` re-pins the
AST shape manifest against the installed DuckDB "so that a version bump surfaces as a
reviewable diff instead of as a wrong answer three layers down", with the usage note
`git diff  # this diff IS the drift report`. What does not exist is the same command
over the 53-file pins corpus.
*Verified-by:* `scripts/pin_ast_shapes.py:1-12` (the prototype, for
`sql_transform/model/_shapes.json`); the pins corpus has no equivalent (measured
2026-08-25). Proposed ticket T-12.

**ORC-62.** **[PROPOSED]** Not in force. Every diff row from a re-record is triaged into
exactly one of four classes, and each class has a different consequence:

| class | consequence |
|---|---|
| DuckDB bug fixed upstream | update the pin, note the change, keep the evidence |
| DuckDB behavior changed | owner decision; becomes a ledger row |
| confit bug newly exposed | a ticket, and a strict-xfail pin |
| now-abstaining | we were passing on luck; move it into the section 3.2 disposition table |

*Verified-by:* Unverified — proposed with ORC-61.

**ORC-63.** **[PROPOSED]** Not in force. Each decision carries a **mutability class**, so
a bump's diff can be triaged mechanically rather than argued case by case. Three
classes, and the assignment for the decisions in this document:

| class | meaning | which decisions |
|---|---|---|
| `frozen` | a change means the oracle is wrong; we file upstream and keep the pin | ORC-01 delegation, ORC-13 axiom, ORC-32 bit-for-bit floats, ORC-35 multi-answer rule, ORC-41 rejected tolerances, ORC-50 ledger split, ORC-57 severity ladder |
| `follows-oracle` | support widens monotonically as the oracle's does; a bump may legitimately move it | ORC-10/ORC-11 inherited quirks, ORC-37 name dedup, ORC-39 error texts, D1, D6 |
| `may-change-on-bump` | expected to move; the bump protocol is where it gets re-decided | ORC-02 the version itself, ORC-05 what the pragma removes, ORC-34 / D5 platform NaN, D7/D8 decimal family, D11 narrow widths, ORC-18 compare modes if the parser changes |

*Verified-by:* Unverified — the classification is proposed here for ratification. Note
ORC-81's libm-backed pins and ORC-76's cbrt tolerance are `may-change-on-bump` if this
lands, and no claim covers them today.

**ORC-88.** The **live-oracle remeasure guard** is the in-force partial form of what
ORC-61 proposes: a divergence or parity test asserts the *oracle's* answer first, in the
same test, so an oracle that moves under us fails loudly at that assertion rather than
silently changing what parity means. It covers the tests that use it; it does not cover
the pins corpus, which is exactly the gap ORC-61 names.
*Verified-by:* `packages/confit/docs/specs/2026-08-19-cast-semantics-design.md:25`;
`packages/confit/docs/specs/2026-08-25-task-120-design.md:374`;
`packages/confit/docs/specs/2026-08-25-task-133-join-keys-design.md:625`; the pattern is
visible in every `known_divergences/` module that opens a connection.

**ORC-64.** **[PROPOSED]** Not in force. A pin whose value changed across a bump is itself
a decision and earns a line in this document, not just a new JSON blob. A changed pin
that leaves no trace here makes the next reader unable to tell a fix from a regression.
*Verified-by:* Unverified — proposed with ORC-61.

---
