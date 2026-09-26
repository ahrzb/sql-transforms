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

**Ruling.** None yet; the owner's choice.
