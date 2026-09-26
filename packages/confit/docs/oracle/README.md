# The oracle

The oracle is the fixed reference for Confit's SQL behavior: **DuckDB 1.5.5 with
`PRAGMA disable_optimizer`, all other settings at their defaults, and the same
declared UDFs registered for comparison.** Tests, campaigns, and callers do not
choose a different reference. Comparisons against another configuration are not
comparisons against this oracle.

It defines compatibility with that configured implementation, not correctness
according to an abstract SQL standard. Its quirks are part of the reference.

**Exact agreement is the default.** The [comparison contract](05-the-comparison-contract.md)
defines schema and value equality, ordering, runtime-error comparison, and
explicitly approved numerical bounds. A known difference is not an exception
merely because it appears in a ledger.

The optimizer is disabled because some rewrites depend on stored table statistics
and can change answers or suppress errors. Confit cannot reproduce a table's
history from a schema and the declared inputs. This choice does not promise that
every reference operation is deterministic; [ordering and nondeterminism](03-nondeterminism.md)
define how variation is handled.

## Details

| Need | Read |
|---|---|
| Exact comparison rules | [Comparison contract](05-the-comparison-contract.md) · [Ordering and nondeterminism](03-nondeterminism.md) |
| Reference rationale and implementation | [Reference notes](01-what-the-oracle-is.md) · [Inherited quirks](02-inherited-quirks.md) |
| Interpreting a comparison run | [Verdicts and refusal](04-verdicts-agreement-abstention-refusal.md) · [Campaign validity](10-campaign-validity-and-blind-spots.md) |
| Recording and maintaining evidence | [Pins](06-pins.md) · [Reference version changes](09-version-bumps-and-mutability.md) |
| Investigating a difference | [Divergence ledger](07-the-divergence-ledger.md) · [Severity](08-the-severity-ladder.md) |
| Settled policy | [Oracle policy](../decisions/oracle-policy.md) |

The [goal](../goal.md) defines product scope. The
[success measures](../specs/success-measures.md) also cover fit/serving and sklearn
references; those are separate checks, not alternative definitions of this oracle.
Future work lives in [PLANS.md](../../PLANS.md), not in these chapters.
