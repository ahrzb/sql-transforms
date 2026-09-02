## 8. The severity ladder

**claim: severity-ladder.** One definition, cited everywhere, restated nowhere:

| rung | meaning | direction |
|---|---|---|
| **1** | we trap where DuckDB serves | contract violation |
| **2** | we serve a wrong value | contract violation |
| **3** | we serve where DuckDB refuses | directional — the query cannot be run against DuckDB at all |
| **4** | we refuse where DuckDB serves | directional — the safe side, sometimes chosen on purpose |

*Verified-by:* the two existing parenthetical definitions, which are **compatible but
not identical** — `packages/confit/docs/specs/2026-08-25-task-114-design.md:140-142`
defines rungs 1-4,
`packages/confit/docs/specs/2026-08-25-task-127-remainders-design.md:154-156` defines
only 2-4 — plus use by name in the Rust source at
`packages/confit/src/specializer/frontend.rs:64-73` ("refusing is the severity ladder's
own preference") and in both RFCs.
*Note:* two partial definitions that agree today is one definition away from drift,
which is the case for consolidating them here. Proposed
ticket: severity-definition-merge replaces the parentheses with a citation of this
claim.

**claim: directional-rungs.** The ladder is also the scope tiering, and rungs 3 and 4
are **directional**: rung 4 is the safe direction, and choosing it deliberately is
legitimate — divergence: regex-size-guard's one-sided guard states it outright, "it may
over-refuse; it can never serve where DuckDB errors. The asymmetry is the contract."
Rung 3 is the unsafe direction and is chosen only where DuckDB cannot be run at all
(divergence: schema-qualifiers, and the row path of divergence: narrow-lane-overflow).
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:79`; ledger rows
divergence: schema-qualifiers, divergence: bind-time-constant-refusals,
divergence: regex-size-guard, divergence: narrow-lane-overflow.
*Correction:* an earlier version added "Rungs 1 and 2 are contract violations and are
never accepted." No source in the tree states that absolute, and the ledger contradicts
it on its face — divergence: decimal-literal-typing / divergence: decimal-cast-rounding
are sev 2 and tolerated (as a feature in flight, claim: feature-in-flight), and
divergence: trap-elision is sev 1 against the contract surface and `PINNED` / permanent.
The true sentence is weaker and is what the tree supports: *a rung-1 or rung-2
divergence against the oracle is a defect unless a named decision places it under
claim: feature-in-flight or claim: fit-serving-oracle.* Whether the stronger absolute is
adopted is part of ask: proposed-rules-adoption.

**claim: countable-rung-four.** **[PROPOSED]** Not in force, because
claim: countable-cost is not. A deliberate rung-4 choice must be countable: rung 4's
whole defense is that it is the safe direction, and that defense is only inspectable if
the class size is observable. divergence: bind-time-constant-refusals is the live
instance and the mechanism it needs is ask: refusal-cost-counting.
*Verified-by:* Unverified — it follows from claim: countable-cost, which is itself
`[PROPOSED]`.

---
