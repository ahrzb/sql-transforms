# Which query classes next

**Question.** Which query class does confit widen to next, and in what order?

**Options.** Each candidate with what it unlocks and what it costs.

| candidate | unlocks | cost / blocker |
|---|---|---|
| decimal arithmetic | exact DECIMAL expressions and literals; empties `UNSHIPPED` | per-operator scale rules through the whole expression tree |
| HUGEINT / unsigned (i128 lane) | the remaining integer widths, and exact wide aggregates | i128 arithmetic and traps on both backends |
| struct-valued outputs | whole structs and struct literals as outputs | nested output schema at the Arrow boundary |
| multiple joins under `shape='many'` | multi-join serving | multiplicity composition across joins |
| native transform families | the sklearn call is ~107 µs of a ~109 µs transformer row (about 80x the SQL) | per-family parity bounds adopted by review first |
| derived tables and row-local CTEs / subqueries | the two largest refusal classes the generator reaches | binder support for nested scopes, and a named row-locality test |
| per-row aggregation over statics | aggregates over matched static rows | a `many`-walk accumulator, and resolving the float-reduction bound's domain |

**Evidence.** Refusals of queries DuckDB answers, on the generated grammar
([reading N=3](../../reports/2026-09-26-goal-reading-n3.md), section 10.1):
derived tables 144, `WITH` 87, the literal 9223372036854775808 (i128 lane) 35,
non-scalar columns 13, of 525. The grammar measures the generator, not demand.

**Row-locality of the two largest classes** (seeds 0–1999, master `84ea988`,
each refused case's AST checked for `DISTINCT`, `ORDER BY`, a row limit,
`GROUP BY`, `QUALIFY`, an aggregate or a window over request rows, in the
outer query or a subquery over the request table):

| class | refused, DuckDB answers | of them row-local |
|---|---|---|
| derived table, `FROM (SELECT …)` | 157 | 150 |
| `WITH` | 94 | 88 |

A row-local derived table or CTE is inside the model (the
[serving contract](../../../../packages/confit/docs/specs/serving-contract.md#restriction-inventory-by-class)
class 2); these 238 are the recoverable part.

**Ruling (owner, 2026-09-27), partial.** Derived tables and row-local CTEs
first, per the approved
[design](../../../../packages/confit/docs/specs/2026-09-26-row-local-subqueries-design.md): phase 1 over
the request table, phase 2 over static tables (projection only).

**Ruling (owner, 2026-10-05).** The owner left the order to the maintainer:
pick one, easiest first, and stick to it. The order:

1. DECIMAL as a row-column lane (the i128 lane exists for expressions; this
   finishes decimals).
2. HUGEINT and the unsigned family on the same i128 lane.
3. Struct-valued outputs (also what multi-output SQL transforms need).
4. More than one join under `shape='many'`.
5. Per-row aggregation over matched static rows.

Native transform families leave this list: the owner proposed modelling most
transformers as SQL-defined transforms instead (a design under discussion,
2026-10-05), with Rust kernels for the rest.
