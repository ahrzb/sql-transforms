# The oracle spec

**What this document is.** The definition of *correct* for the confit engine: what
the oracle is, what it decides, what it declines to decide, and how to compare
against it. It is a consolidation, not a proposal — where a decision is already in
force it is written down here so the next reader stops re-deriving it.

**Non-circularity, and it is the load-bearing rule of this document.** The claims are
the authority; `confit.oracle`, `confit.compare` and `packages/confit/fuzz/` are their
**enforcement**. *Correct* is never defined as "whatever `Oracle` and `compare` do" — if
it were, a bug in either class would become normative the moment it shipped, and the
spec would certify it. The direction of authority runs one way: a claim says what must
hold, `Enforced-by:` names the code that makes it hold, and `Verified-by:` names the
test that would fail if the code stopped. Those modules' docstrings mirror this
document, and **a module's own prose is never a claim's evidence** — a docstring in the
module a claim governs is that module certifying itself, so `Verified-by:` names a test,
a source line whose *content is the fact* (a constant, an enum member, an upstream
`.cpp` line), a pin file, or a dated measurement. Where a measurement is recorded only
in a docstring, the claim says so and reads `Unverified`. A disagreement between a claim
and a module is a bug in the module until the owner rules otherwise.

**Governance.** The oracle spec states what is considered correct. Every
contradiction goes through the owner. This document therefore keeps three kinds of
content strictly apart:

- **Normative claims** — settled decisions in force, each pointing at the decision
  that made it. Each carries a stable slug and a `Verified-by` pointer.
  Nothing unmarked here is new.
- **ASK blocks** — questions only the owner can answer, placed at the point in the
  document where the answer would bind. An ASK is never phrased as decided, and no
  normative claim depends on one.
- **Editorial** — framing, indexes, corrections to other documents. No slug, so nothing
  cites it.

**How an item is named.** Nothing here is named by a number. Every claim, ASK block,
divergence-ledger row and proposed ticket carries a **slug**: a short kebab-case noun
phrase naming what the item is *about* — its subject, never its verdict, so the name
still reads true when the ruling changes. The family is carried by the citation rather
than by the slug:

| kind | written as | example |
|---|---|---|
| normative claim | `claim: <slug>` | claim: one-door-bypass |
| ASK block | `ask: <slug>` | ask: version-pin |
| divergence-ledger row | `divergence: <slug>` | divergence: trap-elision |
| proposed ticket | `ticket: <slug>` | ticket: version-assert |

Slugs are unique across all four kinds, and the form above is used everywhere — at the
definition and at every reference — so `grep -rn "<slug>"` finds an item and everything
that cites it, and `grep -rn "claim:"` finds every claim.

**A slug is assigned once.** Renaming one is the same ceremony as retiring a claim — the
old name stays, in a tombstone line naming both. The numeric codes this document used
before — `ORC-NN`, `ASK-NN`, `D-N`, `T-N` — are cited from merged PRs, backlog tickets
and code comments that nothing here can update, so they are kept as pointers rather than
deleted: [13-old-ids.md](13-old-ids.md) maps every one of them to its slug.

**Two markers keep the first category honest.** A claim carrying either marker is
*not* a decision in force:

- **`[PROPOSED]`** — a rule this document would like, which nobody has ruled on. It
  holds a slug only so a pin or a ticket can reference it. Several of these are rules
  this document originally stated as though they were already in force; they are marked
  now. **15 claims** carry this marker, and they adopt through two doors: the nine
  comparison-doctrine ones go through ask: proposed-rules-adoption, and the remaining
  eight (claim: pin-back-reference, claim: under-determined-token,
  claim: re-record-diff-report, claim: diff-triage-classes, claim: mutability-classes,
  claim: changed-pin-record, claim: coverage-denominator, claim: abstention-rate — pins
  and campaign machinery) through their own proposed tickets. Note
  ask: proposed-rules-adoption's table also names claim: unadopted-mechanisms and
  claim: directional-rungs, which carry no marker; ask: proposed-rules-adoption says
  why.
- **`[FACT]`** — a measured statement of current state with no decision attached.
  It is here because the state is load-bearing, not because anyone ruled on it.
  **9 claims** carry this marker.

Of 92 live claims, 68 are decisions in force. One claim is **retired**: a claim the
shipped code dissolved keeps its slug and a one-line tombstone naming its old code and
what replaced it, because slugs are cited from outside this document and are never
reassigned or reused.

Every other claim is a decision in force, made somewhere outside this document,
and its `Verified-by` names where.

**How to read a claim.** One normative sentence per claim, then its pointers:
`Enforced-by:` names the module function that makes the claim hold, and `Verified-by:`
names the test, pin file, source line or measurement that would catch it not holding.
Where nothing verifies a claim it says `Unverified` and says so plainly. "Derived" and
"normative here" are not verification and no longer appear: a claim that only this
document asserts is `[PROPOSED]`. Prose about where a behavior lives, or how two copies
of it are kept in sync, is not a claim and is not here — the pointers are that.
Behaviors carry a status:

| status | meaning |
|---|---|
| `PINNED` | the oracle's answer is stable and it is the contract |
| `IMPL-DEFINED` | stable for the pinned build and configuration, fragile across versions or platforms; the pin names the discriminator |
| `UNSPECIFIED` | not stable; we refuse, normalize, or exclude, and never claim bit-for-bit |

**Completeness** here means *decision coverage*: every oracle fact that was decided
appears in this document. Emergent behavior owes nothing. A gap is a decision that
was made somewhere and is not written here — not a behavior nobody has ruled on.

**Scope, so that "complete" has edges.** This document covers **confit's DuckDB
oracle**: the identity in claim: oracle-identity, everything compared against it, and
every gate that does the comparing (`packages/confit/tests/`, `packages/confit/fuzz/`,
the pins corpus). Three neighbouring oracles exist in this repo and are **explicitly
excluded**, each with a pointer rather than a silent omission —
claim: fit-serving-oracle, claim: dialect-gate-oracle and claim: duckdb-three-roles name
them. Their decisions are real decisions; they are simply not this document's. Anything
else that is an oracle fact and is not here is a gap.

**Doc homes.** confit's docs live under `packages/confit/docs/`. A spec, ticket or
comment citing a bare `docs/known-limitations.md` is a stale path (the move merged as
master `85b4739`).

**The three enforcement modules**, so a claim's `Enforced-by:` line needs no
introduction:

| module | what it enforces | its tests |
|---|---|---|
| `packages/confit/confit/oracle.py` | the oracle's identity and setup verbs — `Oracle`, `Trap` | `packages/confit/tests/test_oracle.py` |
| `packages/confit/confit/compare.py` | what "equal" means, and the one axis a caller declares | `packages/confit/tests/test_compare.py` |
| `packages/confit/fuzz/oracle.py`, `fuzz/runner.py` | the verdict taxonomy and what a campaign reports | `packages/confit/tests/test_fuzz_smoke.py` |

Both `confit` modules ship in the wheel rather than in `tests/`, because the campaign
runner must not import from `tests/` and the tests must not import from `fuzz/`; a
comparison vocabulary both of them use is therefore package surface.

---

## Contents

- [1. What the oracle is](01-what-the-oracle-is.md)
- [2. Inherited quirks](02-inherited-quirks.md)
- [3. Nondeterminism](03-nondeterminism.md)
- [4. Verdicts: agreement, abstention,
  refusal](04-verdicts-agreement-abstention-refusal.md)
- [5. The comparison contract](05-the-comparison-contract.md)
- [6. Pins](06-pins.md)
- [7. The divergence ledger](07-the-divergence-ledger.md)
- [8. The severity ladder](08-the-severity-ladder.md)
- [9. Version bumps and mutability](09-version-bumps-and-mutability.md)
- [10. Campaign validity and blind spots](10-campaign-validity-and-blind-spots.md)
- [11. Proposed tickets](11-proposed-tickets.md)
- [12. ASK index](12-ask-index.md)
- [13. Old ids](13-old-ids.md)

Claims are cited by slug (`claim: <slug>`), never by chapter or line - `grep -rn
"<slug>" packages/confit/docs/oracle/` finds the definition and every reference to it.
ASK blocks are indexed in [12-ask-index.md](12-ask-index.md); the old `ORC-NN` /
`ASK-NN` / `D-N` / `T-N` codes are mapped to slugs in [13-old-ids.md](13-old-ids.md).
