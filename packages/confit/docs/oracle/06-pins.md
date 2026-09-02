## 6. Pins

### 6.1 What a pin is

**ORC-42.** A pin is **one measured oracle fact**: a behavioral claim backed by the
exact SQL that was run and the exact result that came back — query text, input reprs,
result reprs, float bit patterns, verbatim error heads. A pin is spec-as-data. It is
not a test of our code; it is a recording of the oracle's answer, which our code is
then written to.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:20-22`;
the corpus at `packages/confit/docs/specs/pins-*/` (53 files measured 2026-08-25).

**ORC-43.** Pins-first: **no semantics are implemented from memory, documentation, or
intuition — only from executed queries against the oracle, recorded verbatim.**
Implementation starts only after the pins exist. A summary sentence with no query
behind it is treated as a guess.
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:20-22, :28`.

**ORC-44.** The rule was bought, not designed. During wave 3 a fleet summary claimed
`%`-by-zero returns NULL, generalizing from integer probes; the DOUBLE case was never
run and returns NaN. The correction is appended to the pin file with the honest note
that "raw probes never covered this cell ... the summary over-generalized", and every
wave dispatched since carries the rule explicitly.
*Verified-by:* `packages/confit/docs/specs/pins-wave3/math_tail.json` (the
`corrections` key); `packages/confit/docs/reports/pins-first-methodology.md:24-28`.

**ORC-45.** Phase separation is required for any claim about *when* DuckDB does
something. `con.execute` conflates prepare and execute, so a bind-time claim needs a
PREPARE/EXECUTE split, a zero-row leg, and the pinned source. This is the same genus as
the wave-3 incident with a larger blast radius: it killed the stated premise of an
already-accepted RFC.
*Scope, precisely:* `confit.oracle.Oracle.answer` is one `con.execute`, so it inherits
the conflation by construction — that is the right shape for a value claim and the wrong
tool for a phase claim. A phase claim goes through the connection directly (the
`__getattr__` passthrough exists so no wrapper has to be invented for it) and says in the
pin which phase it measured.
*Verified-by:* `packages/confit/docs/rfcs/2026-08-19-keep-the-bind-time-refusals.md:29-58`
(the corrected facts, and the phase-confusion admission at `:31-35`);
`packages/confit/tests/test_oracle.py::test_connection_passthrough`.
*Note:* this rule lives in memory and in one RFC's body. The methodology report owns
"how we measure DuckDB" and does not carry it. Proposed ticket T-6.

### 6.2 Provenance

**ORC-46.** A pin's provenance is what makes a disagreement re-verifiable: without the
oracle version that produced a recorded answer, a future disagreement cannot be
re-run, only argued about. The version a pin should carry now has a name in code —
`confit.oracle.Oracle.VERSION` — so "which oracle recorded this" and "which oracle is
installed" are at least the same string in two places, even though nothing yet compares
them (ORC-09). Measured state of the corpus today: of 53 pin files, **41 carry a
`duckdb_version` field, 10 mention a capture date anywhere, and 3 mention a harness or
commit**; the version field itself is free text with at least four spellings in use
(`1.5.5`, `v1.5.5`, `v1.5.5 (python pkg 1.5.5)`, and a sentence).
*Verified-by:* measured 2026-08-25 over `packages/confit/docs/specs/pins-*/*.json`.
Best existing examples: `pins-dialect/joins.json` `_meta` (date, engine, task, spec,
how) and `pins-waveB/fuzzer-task54.json` `meta` (task, measured, method, contract).
*Proposed:* a uniform header — oracle version, settings profile, capture date, capture
harness commit — on every pin file. Not applied here. Proposed ticket T-7.

**ORC-47.** Generation scripts already stamp the oracle version into their outputs,
and two already say to regenerate after a duckdb bump — the right instinct, without a
uniform shape and without covering the pins corpus.
*Verified-by:* `scripts/pin_ast_shapes.py:29, :36`; `scripts/gen_casemap.py:152, :159`
("regenerate after a duckdb bump"); `scripts/gen_strip_accents.py:135` (same).

**ORC-48.** **[PROPOSED]** Not in force. Every pin carries a **decision back-reference** —
the `ORC-NN` id of the claim it evidences. This is the mechanical instrument for this
project's own definition of completeness: with back-references, "decisions with zero
pins" is the uncovered set and is computable in one query; without them, decision
coverage is a promise nobody can audit. Ids in this document are hand-assigned and
stable precisely so a pin can point at one.
*Verified-by:* Unverified — no pin carries such a field today (measured 2026-08-25).
Proposed ticket T-8.

**ORC-49.** **[PROPOSED]** Not in force. The pin format gains an inline token for an
under-determined field, so "this field is not part of the contract" or "this field is
contract per-platform" is written *in the pin* rather than in prose beside it. ORC-34 is the
existing instance and is currently a special case explained in a comment. Without a
token, every re-measurement pass must re-derive which fields were deliberate.
*Verified-by:* Unverified — no such token exists. Proposed ticket T-9.

---
