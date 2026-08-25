## 8. The severity ladder

**ORC-57.** One definition, cited everywhere, restated nowhere:

| rung | meaning | direction |
|---|---|---|
| **1** | we trap where DuckDB serves | contract violation |
| **2** | we serve a wrong value | contract violation |
| **3** | we serve where DuckDB refuses | directional — the query cannot be run against DuckDB at all |
| **4** | we refuse where DuckDB serves | directional — the safe side, sometimes chosen on purpose |

*Verified-by:* the two existing parenthetical definitions, which are **compatible but not
identical** — `packages/confit/docs/specs/2026-08-25-task-114-design.md:140-142` defines
rungs 1-4, `packages/confit/docs/specs/2026-08-25-task-127-remainders-design.md:154-156`
defines only 2-4 — plus use by name in the Rust source at
`packages/confit/src/specializer/frontend.rs:64-73` ("refusing is the severity ladder's
own preference") and in both RFCs.
*Note:* two partial definitions that agree today is one definition away from drift, which
is the case for consolidating them here. Proposed ticket T-11 replaces the parentheses
with a citation of this claim.

**ORC-58.** The ladder is also the scope tiering, and rungs 3 and 4 are **directional**:
rung 4 is the safe direction, and choosing it deliberately is legitimate — D10's
one-sided guard states it outright, "it may over-refuse; it can never serve where DuckDB
errors. The asymmetry is the contract." Rung 3 is the unsafe direction and is chosen
only where DuckDB cannot be run at all (D6's schema qualifiers, D11's row path).
*Verified-by:* `packages/confit/docs/reports/pins-first-methodology.md:79`; ledger rows
D6, D9, D10, D11.
*Correction:* an earlier version added "Rungs 1 and 2 are contract violations and are
never accepted." No source in the tree states that absolute, and the ledger contradicts
it on its face — D7/D8 are sev 2 and tolerated (as a feature in flight, ORC-80), and
D4 is sev 1 against the contract surface and `PINNED` / permanent. The true sentence is
weaker and is what the tree supports: *a rung-1 or rung-2 divergence against the oracle
is a defect unless a named decision places it under ORC-80 or ORC-72.* Whether the
stronger absolute is adopted is part of ASK-15.

**ORC-59.** **[PROPOSED]** Not in force, because ORC-31 is not. A deliberate rung-4
choice must be countable: rung 4's whole defense is that it is the safe direction, and
that defense is only inspectable if the class size is observable. D9 is the live instance
and the mechanism it needs is ASK-3.
*Verified-by:* Unverified — it follows from ORC-31, which is itself `[PROPOSED]`.

---
