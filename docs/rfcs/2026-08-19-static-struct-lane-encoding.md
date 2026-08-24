# RFC: the static-struct lane encoding (TASK-127 AC #4)

**Status: needs AmirHossein's pick. No code written -- TASK-127's remaining
ACs all hang on this, so the ticket stays parked rather than half-built.**

## The question

TASK-116 serves a static table's struct leaves by flattening them into
lanes NAMED `parent.leaf` inside `StaticTable::cols`. Is the dotted NAME
the right encoding, or should the lane carry a structured path with the
dotted spelling as a display detail?

## Fresh measurements (2026-08-19, re-measured per AC #5 -- the ticket's
#2/#3 were relayed from a review and unconfirmed until now)

Static `s(id BIGINT, w STRUCT(mean DOUBLE), "w.mean" DOUBLE)` -- a struct
AND a literal column whose name is the flattened spelling:

```
SELECT s.w.mean    duck: 1.5   (the struct leaf)      ours: refuses, "Could not find key"
SELECT s."w.mean"  duck: 99.0  (the literal column)   ours: refuses, "ambiguous column"
```

DuckDB SERVES BOTH, distinguishing them -- the two spellings are different
references and its catalog keeps them apart. Our dotted encoding collapses
them into colliding lane names, so we cannot even in principle serve both:
the information "leaf vs literal" is destroyed at catalog build.

Also confirmed: unqualified `SELECT w.mean` (no collision) serves 1.5 on
DuckDB; we refuse with the wrong reason ("unknown table 'w'").

## Options

**(a) Structured path on the lane** (`name: Vec<String>` or
`name + path: Option<Vec<String>>`), dotted string becomes display-only.
Collisions stop existing structurally; `s."w.mean"` and `s.w.mean` resolve
to different lanes; TASK-127's #2 and #3 fall out naturally. Cost: touches
every consumer of `StaticTable::cols[..].name` (star, aliases TASK-126,
EXCLUDE, the row-side flatten uses the same convention -- audit needed
there too), so it is a day-plus refactor across frontend + duckdb/mod.

**(b) Keep dotted names, refuse collisions loudly at REFERENCE time** with
a message naming both origins ("'w.mean' is both a struct leaf and a
column; this engine cannot serve the pair -- rename one"). Cheap (the
duplicate already trips the existing ambiguity error, just with the wrong
words), honest, severity-4. Unqualified resolution (#2) can still be built
on top. Cost: permanently refuses a query DuckDB serves, and the refusal
depends on name-mangling trivia a user shouldn't need to know.

**(c) Refuse the COLLIDING TABLE at build.** Simplest of all, but violates
the unreferenced-columns-cost-nothing doctrine: a user who never touches
`w` still loses the build.

## Recommendation

(a), scheduled as its own ticket rather than squeezed into TASK-127 -- the
collision is pathological but the encoding ALSO blocks #2's legitimate
unqualified resolution and keeps producing wrong-reason messages (three
found so far). (b) is the acceptable interim if (a) doesn't earn a slot;
(c) is listed only for completeness.

If you pick (a) I will split TASK-127 into: encoding refactor (new ticket,
specced first), then #1-#3 land on top of it mostly for free.
