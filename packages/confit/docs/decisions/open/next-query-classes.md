# Which query classes next

**Question.** Which query class does confit widen to next, and in what order?

**Options.** Each candidate with what it unlocks and what it costs.

| candidate | unlocks | cost / blocker |
|---|---|---|
| decimal arithmetic | exact DECIMAL expressions and literals; empties `UNSHIPPED` | per-operator scale rules through the whole expression tree |
| HUGEINT / unsigned (i128 lane) | the remaining integer widths, and exact wide aggregates | i128 arithmetic and traps on both backends |
| struct-valued outputs | whole structs, struct literals, bracket access | nested output schema at the Arrow boundary |
| multiple joins under `shape='many'` | multi-join serving | multiplicity composition across joins |
| native transform families | ~100x on transformer queries | per-family parity bounds adopted by review first |
| row-local CTEs / subqueries | the largest refusal class the generator reaches | binder support for nested scopes, and a named row-locality test |
| per-row aggregation over statics | aggregates over matched static rows | a `many`-walk accumulator, and resolving the float-reduction bound's domain |

**Evidence.** Refusals of queries DuckDB answers, on the generated grammar
([2026-09-26 reading](../../reports/2026-09-26-goal-reading.md), section 10.1):
`WITH` 86, the literal 9223372036854775808 (i128 lane) 34, non-scalar columns 12
of 526. The grammar measures the generator, not demand.

**Ruling.** None yet; the owner's choice.
