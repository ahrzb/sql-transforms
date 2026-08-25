# The oracle spec

**What this document is.** The definition of *correct* for the confit engine: what
the oracle is, what it decides, what it declines to decide, and how to compare
against it. It is a consolidation, not a proposal — where a decision is already in
force it is written down here so the next reader stops re-deriving it.

**Governance.** The oracle spec states what is considered correct. Every
contradiction goes through the owner. This document therefore keeps three kinds of
content strictly apart:

- **Normative claims** — settled decisions in force, each pointing at the decision
  that made it. Each carries a stable id `ORC-NN` and a `Verified-by` pointer.
  Nothing unmarked here is new.
- **ASK blocks** — questions only the owner can answer, placed at the point in the
  document where the answer would bind. An ASK is never phrased as decided, and no
  normative claim depends on one.
- **Editorial** — framing, indexes, corrections to other documents. Unnumbered.

**Two markers keep the first category honest.** A claim carrying either marker is
*not* a decision in force:

- **`[PROPOSED]`** — a rule this document would like, which nobody has ruled on. It
  holds an id only so a pin or a ticket can reference it. Several of these are rules
  this document originally stated as though they were already in force; they are
  marked now, and adopting them is ASK-15. **15 claims** carry this marker.
- **`[FACT]`** — a measured statement of current state with no decision attached.
  It is here because the state is load-bearing, not because anyone ruled on it.
  **8 claims** carry this marker.

Of 89 claims, 66 are decisions in force.

Every other `ORC-NN` is a decision in force, made somewhere outside this document,
and its `Verified-by` names where.

**How to read a claim.** One claim per paragraph block, so a content-hash field can
be added later by tooling without re-cutting the text. `Verified-by` names a test
path, a pin file, a source line, or a measurement; where nothing verifies a claim it
says `Unverified` and says so plainly. "Derived" and "normative here" are not
verification and no longer appear: a claim that only this document asserts is
`[PROPOSED]`. Behaviors carry a status:

| status | meaning |
|---|---|
| `PINNED` | the oracle's answer is stable and it is the contract |
| `IMPL-DEFINED` | stable for the pinned build and configuration, fragile across versions or platforms; the pin names the discriminator |
| `UNSPECIFIED` | not stable; we refuse, normalize, or exclude, and never claim bit-for-bit |

**Completeness** here means *decision coverage*: every oracle fact that was decided
appears in this document. Emergent behavior owes nothing. A gap is a decision that
was made somewhere and is not written here — not a behavior nobody has ruled on.

**Scope, so that "complete" has edges.** This document covers **confit's DuckDB
oracle**: the identity in ORC-02, everything compared against it, and every gate that
does the comparing (`packages/confit/tests/`, `packages/confit/fuzz/`, the pins
corpus). Three neighbouring oracles exist in this repo and are **explicitly excluded**,
each with a pointer rather than a silent omission — ORC-72, ORC-73 and ORC-74 name
them. Their decisions are real decisions; they are simply not this document's.
Anything else that is an oracle fact and is not here is a gap.

**Doc homes.** confit's docs live under `packages/confit/docs/`. A spec, ticket or
comment citing a bare `docs/known-limitations.md` is a stale path (the move merged as
master `85b4739`).

---

## Contents

- [1. What the oracle is](01-what-the-oracle-is.md)
- [2. Inherited quirks](02-inherited-quirks.md)
- [3. Nondeterminism](03-nondeterminism.md)
- [4. Verdicts: agreement, abstention, refusal](04-verdicts-agreement-abstention-refusal.md)
- [5. The comparison contract](05-the-comparison-contract.md)
- [6. Pins](06-pins.md)
- [7. The divergence ledger](07-the-divergence-ledger.md)
- [8. The severity ladder](08-the-severity-ladder.md)
- [9. Version bumps and mutability](09-version-bumps-and-mutability.md)
- [10. Campaign validity and blind spots](10-campaign-validity-and-blind-spots.md)
- [11. Proposed tickets](11-proposed-tickets.md)
- [12. ASK index](12-ask-index.md)

Claims are cited by id (`ORC-NN`), never by chapter or line - `grep -rn <id> packages/confit/docs/oracle/`. ASK blocks are indexed in [12-ask-index.md](12-ask-index.md).
