# Plan

Work that is not settled: draft RFCs, questions for the owner, and work that
a change or a measurement must do. These files change freely. A file that is
used up is deleted, or moved to [records/](../records/README.md) when it is
worth keeping.

Implementation plans for one change are not committed. They stay in the
session or in the PR description.

## RFCs

An RFC is a complete proposed design, with the options weighed. It waits
for the owner's review before it changes the [spec](../spec/README.md).
There are no draft RFCs.

An RFC is the file `rfcs/<topic-name>.md`:

- The title is `# RFC (draft): <title>`. The next line is
  `**Status:** draft, waiting for the owner's review.`
- Its sections are `## Summary`, `## Motivation`, `## Design`,
  `## Options weighed` and `## Changes to the spec`. The last section lists
  each claim that the RFC adds, changes or removes.
- When the owner approves it, apply it to the spec in the same PR. Change
  the title to `# RFC: <title>`, and the status to
  `**Status:** approved by the owner on <date>, applied in <PR link>.` Then
  move the file to `records/rfcs/`, with the same name.
- When the owner rejects it, add a `## Decision` section with the date and
  the reason, and move it to `records/rfcs/`.

## Open decisions

None. The [decision guide](decisions/README.md) says when to write a
decision record and how to close it.

## Postponed decisions

None.

## Open work

Work that a change or a measurement must do. Each row links to the spec
section that it would change. The full text of the decorrelation items is
in the old refusal page, before this layout moved it:
[docs/decorrelation-unsupported.md at commit 1168933](https://github.com/ahrzb/sql-transforms/blob/1168933/docs/decorrelation-unsupported.md).
The [decorrelation research](../research/decorrelation/algorithms-and-traps.md)
explains the terms that the rows use, such as the count bug and a join miss.

| Area | Work | Blocked on |
|---|---|---|
| decorrelation | Lift inequality correlation, such as `f.ts <= t.ts`, as a second kind of params table: a prefix aggregate with one row for each fitted timestamp (`GROUP BY ts`), served with `ASOF LEFT JOIN`. It needs its own empty-group rule, because the count bug can occur where no join miss occurs. The backlog draft [Step semantics for order-keyed windows off the training support](../../../backlog/drafts/draft-21%20-%20Step-semantics-for-order-keyed-windows-off-the-training-support.md) holds this design ([not-an-equality](../spec/fit/decorrelation-refusals.md#not-an-equality)). | |
| decorrelation | Lift `OR` and `NOT` correlation as a union of keyed tables, one for each disjunct. The params size multiplies ([not-an-equality](../spec/fit/decorrelation-refusals.md#not-an-equality)). | a real query that needs it |
| decorrelation | Lift the `__THIS__` part out of an aggregate that distributes over it, such as `sum(f.x) + n * t.y`, `avg(f.x) - t.y` and `count(*)`, from a small table of cases with a sign condition for `min` and `max` ([outside-where](../spec/fit/decorrelation-refusals.md#outside-where)). | real queries to choose the cases |
| decorrelation | Rewrite `ORDER BY … LIMIT 1` for each key as `arg_max` with a carrier that keeps NULL values, `arg_max(struct_pack(v := price), ts)`, because plain `arg_max` skips NULL values ([modifier](../spec/fit/decorrelation-refusals.md#modifier)). | an owner decision to build it |
| decorrelation | Append the correlation key to the subquery's own `GROUP BY` and keep its grouping as a nested aggregate; `HAVING` moves in unchanged. First prove one row for each key across the two levels ([grouping](../spec/fit/decorrelation-refusals.md#grouping)). | |
| decorrelation | Add the correlation key to the `PARTITION BY` of a window inside a correlated subquery. First make this rule agree with the window params rule ([window](../spec/fit/decorrelation-refusals.md#window)). | |
| decorrelation | Lift correlated `EXISTS` as `count(*) > 0` for each key. Before `IN`, `NOT IN` and `> ALL`, write their three-valued truth table and measure each case ([not-a-scalar-subquery](../spec/fit/decorrelation-refusals.md#not-a-scalar-subquery)). | |
| decorrelation | Move the definition of a CTE that holds the correlated subtree to where the lookup goes: inline a CTE used once, and read one params table twice for a CTE used twice ([not-in-a-subquery](../spec/fit/decorrelation-refusals.md#not-in-a-subquery)). | a real query that needs it |
| decorrelation | With a declared `__FIT__` schema at construction, refuse `shadowed-by-a-nested-column` at construction instead of at fit ([shadowed-by-a-nested-column](../spec/fit/decorrelation-refusals.md#shadowed-by-a-nested-column)). | a declared fit schema |
| decorrelation | With a declared `__THIS__` catalog, group by the key at its comparison type (`GROUP BY cat::INTEGER`, or `COLLATE NOCASE` for a collation), so that the equivalence of the params key equals the join predicate's ([cross-type correlation keys](../spec/fit/decorrelation-refusals.md#cross-type-correlation-keys)). | a declared request catalog |
| fit | Replace the refusal of a bare `__FIT__` beside `__THIS__` with a table of rewrites where the cross join reduces: `sum` over an affine expression, `min` and `max` over a monotone one with a sign condition, and `count` ([freezing](../spec/fit/freezing.md#a-bare-fit-relation-beside-the-request-table)). | |
| fit | Freeze the `__FIT__`-only subtrees of a recursive CTE in place, inside a rebuilt `WITH RECURSIVE`, with a check that nothing lifted reads the CTE's own name ([freezing](../spec/fit/freezing.md#a-fit-relation-inside-a-recursive-cte)). | |
| fit | A declared byte budget for the fitted artifact. The backlog draft [Serving pipelines in SQL: marginalized aggregates, fitted artifacts and the leakage question](../../../backlog/drafts/draft-20%20-%20Serving-pipelines-in-SQL-marginalized-aggregates-fitted-artifacts-and-the-leakage-question.md) holds the question ([artifact size](../spec/fit/freezing.md#artifact-size)). | |
| spec | Pin these claims with tests: `request-schema-widths`, `refuses-not-a-select`, `refuses-not-in-a-subquery`, and the seeded case of `refuses-sample`. | |
| spec | Rewrite the moved text of `spec/fit/decorrelation-refusals.md` and `spec/python/tree-models.md` in the `simple-english` style. Change no rule. | |
