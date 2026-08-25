# A typed model of DuckDB's serialized AST

Date: 2026-08-08. Status: designed with AmirHossein (this session).
Every DuckDB behavior cited was measured on 2026-08-08 against **1.5.5**.

## What breaks

`sql_transform/model/_ast.py` reads the oracle's AST through untyped dicts:

```python
type Node = dict[str, Any]

if v.get("type") == "BASE_TABLE" and v.get("table_name") == FIT: ...
```

`json_serialize_sql`'s shape is an internal DuckDB detail with no stability
promise, so every one of those `.get()` calls is a bet on a schema that is
version-scoped and undocumented. Measured, the bet loses quietly.

**The deserializer requires exactly one field.** Dropping each field of a
`BASE_TABLE` in turn and asking DuckDB to print the result:

```
  -table_name           accepted  ** DIFFERENT TEXT **
  -alias                accepted  ** DIFFERENT TEXT **
  -schema_name          accepted
  -at_clause            accepted
  -query_location       accepted
  -type                 REJECTED: Expected but did not find property 'type'
```

Only `type` is required. Everything else silently defaults. So a model that
forgets a field does not fail — **it emits a different query**. That is C5's
one unrecoverable state: not a refusal, not the right answer, no error.

The same leniency runs through the reader. `table_name` is carried by exactly
one node type out of 53. If 1.6 renames it, every `v.get("table_name")`
returns `None`, the walk stops finding base tables, freezing quietly freezes
the wrong subtree, and nothing raises.

P9 already says the answer: *interpreted node shapes are validated, and drift
fails as one named error*. The old `_marginalize.py` still does this with eight
pydantic `_View` classes. The redesigned model dropped it.

Version-dependence is the argument **for** typing, not against it. An
undocumented schema that moves is precisely the one you want pinned, so the
move surfaces as a diff instead of as a wrong number.

## The shape is regular enough to pin

Swept 1,039 statements — the mined window corpus, the curated corpus, and
`packages/confit/tests/corpus/duckdb_mined.jsonl` — of which 719 are queries
(the other 320 are `INSERT`/`CREATE`/`DROP`/`UPDATE`; `json_serialize_sql`
handles queries only, which is all this model ever sees).

```
distinct node types:                    53
  with a fixed field set:               52 / 53
(type, field) pairs:                   498
  polymorphic (>1 value type):           0
nullable fields:                         6
```

Zero polymorphic fields across 498. One ragged tag. That is regular enough to
generate from and cheap enough to check.

The ragged one is `SUBQUERY`, and it is ragged for a real reason: DuckDB uses
the tag for two different things.

```
SUBQUERY as a table ref:    sample, column_name_alias
SUBQUERY as an expression:  class, subquery_type, comparison_type, child
```

The dict form let us paper over that. Typing forces the split.

Two more measured facts that shape the design: JSON key order is free (a
reversed node round-trips identically), and unknown extra keys are ignored.

## The model

**Typed where we interpret; carried verbatim everywhere else.** A class exists
for a tag when the walk branches on its contents — about a dozen of the 53.
The other forty are data we move around without reading, and inventing a class
per tag for them buys nothing that carrying the dict does not already give.

Where a class does exist it carries **every** field DuckDB emits for that tag,
including the ones nothing reads, because a field that is not carried is a
field that is dropped, and a dropped field is different SQL.

```python
class BaseTable(AstNode):
    """duckdb 1.5.5"""
    type: Literal["BASE_TABLE"] = "BASE_TABLE"
    table_name: str
    alias: str
    schema_name: str
    catalog_name: str
    column_name_alias: list[str]
    sample: Sample | None
    at_clause: AtClause | None
    query_location: int

type Node = Select | SetOperation | RecursiveCte     # query nodes
          | BaseTable | SubqueryRef | Join | TableFunction | EmptyTable
          | ColumnRef | Function                     # the two expressions we read
          | Opaque                                   # everything else
```

The set is not a taste judgement — it is read off the walk. `_analysis.py`,
`_plan.py` and `_transform.py` branch on exactly these, and the gate below
pins that: a class that nothing branches on is a class we should not have
written.

Pydantic, not dataclasses: the repo already depends on it, P9 already names it,
and `extra="forbid"` turns an unexpected field into a named error at parse time
— a second drift gate for free, on top of the manifest below. Construction is
not a hot path (`_ast.py` already says so about catalog lookups).

### The tail descends

A tag the manifest does not know becomes `Opaque` — but its children are still
parsed:

```python
class Opaque(AstNode):
    """An unrecognised tag. Carried verbatim, never interpreted.

    Its children are typed all the same, so a `__FIT__` reference nested
    under a node DuckDB added last week is still found by the walk.
    """
    tag: str
    fields: dict[str, Node | list[Node] | str | int | bool | None]
```

If `Opaque` held a raw dict, a new DuckDB node type would hide a `__FIT__`
reference from freezing — silently. That is the bug class TASK-68..76 was.

`Opaque` is deliberately un-matchable on anything but its presence: the walk
can carry it and descend through it, and cannot branch on what it means.

### Losslessness is a property, not a hope

```python
@pytest.mark.parametrize("sql", CORPUS)          # 719 query statements
def test_the_tree_is_lossless(sql):
    raw = _serialize(sql)
    assert to_json(from_json(raw)) == raw        # structural, not merely printable
```

Structural equality rather than "it still prints the same", because the
measurement above is exactly that a dropped field still prints — just
differently, and only for some inputs.

## No generator: a test instead

An earlier draft of this spec generated the classes from a corpus sweep. That
was the right answer for 53 tags and 498 declarations, where the transcription
itself is the risk. For a dozen tags and about a hundred fields it is not:
the classes are hand-written, and the property the generator would have
guaranteed is *checked* instead.

```python
@pytest.mark.parametrize("model", INTERPRETED)
def test_every_field_duckdb_emits_is_carried(model):
    """A field we do not carry is a field we drop, and a dropped field is
    accepted by the deserializer as different SQL. So this is not style."""
    assert set(model.model_fields) == FIELDS_IN_CORPUS[model.tag]
```

Six lines, the same guarantee, and no generated artifact to keep honest. It
fails on a DuckDB bump exactly where the generator's diff would have.

The corpus sweep still happens — it is what `FIELDS_IN_CORPUS` is — and it
still reports coverage (`53 tags from 719 statements`) rather than implying
completeness. There is no way to enumerate DuckDB's serialization vocabulary
from Python; everything unobserved lands in `Opaque`, and the number is printed
so nobody has to guess how wide the net was.

### The bump story

`model/_shapes.json` pins `{duckdb: "1.5.5", shapes: {tag: {field: type}}}` for
**all 53 tags**, not just the dozen with classes. A test re-derives the shapes
from the corpus and diffs, so upgrading DuckDB fails as

```
AstDrift: duckdb 1.6.0 moved 2 shapes
  BASE_TABLE.table_name: str -> missing
  BASE_TABLE.name:       missing -> str
```

rather than as a wrong answer three layers down. Re-pinning is the reviewable
fix:

```bash
uv run python scripts/pin_ast_shapes.py
git diff        # this diff IS the drift report
```

That is a fifteen-line dump of the sweep, not a code generator — the classes
stay hand-written.

Covering all 53 is the point of the manifest. `Opaque` makes drift in a tag we
do not read *harmless*, which also makes it *invisible*; the manifest is what
turns it into an early warning instead of a surprise on the day we start
reading that tag. It also catches a wholly new tag appearing, which neither
parse-time validation nor the lossless gate can see — both are satisfied by
carrying it verbatim.

## Staging

1. **The file.** `_nodes.py`, `_shapes.json`, `scripts/pin_ast_shapes.py`, the
   lossless gate and the manifest gate. Lands alongside; `_ast.py` and the walk
   are untouched. Nothing in the model's behaviour changes.
2. **The walk.** `_analysis.py`, `_plan.py` and `_transform.py` move onto the
   typed tree. Frozen models mean freezing stops mutating in place —
   `parent[key] = freeze(sub)` becomes a rebuild — and the dict patterns become
   real ones:

   ```python
   case {"type": "BASE_TABLE", "table_name": name} if ...   # today
   case BaseTable(table_name=FIT):                          # after
   ```

3. **Then** the correlated-clause and CTE work, which is what motivated this:
   `_plan.py`'s three `whole_fit()` sites are where the training set is shipped
   whole, and they are much easier to reason about against a typed tree.

Step 1 is this spec. Steps 2 and 3 get their own.

## What this does not do

- **It does not put a class on every tag.** Only the ones the walk branches on.
  The other forty round-trip through `Opaque` and are never interpreted, which
  is exactly what they are today — the difference is that now they cannot be
  interpreted *by accident*.
- **It does not make the model exhaustive over DuckDB.** The manifest is
  exhaustive over what 719 statements exercise, and honest about the rest via
  `Opaque` plus a printed coverage number.
- **It does not change any refusal or any answer.** Step 1 is additive; the
  gate is that the corpus replay and the full suite are unchanged.
- **It does not pin the DuckDB version.** A bump is a re-pin-and-review, not a
  blocked upgrade — the manifest reports what moved and the suite says whether
  it mattered.
- **It does not type non-query statements.** `json_serialize_sql` does not
  serialize `INSERT`/`CREATE`/`DROP`/`UPDATE`, and the model refuses anything
  that is not a query well before this layer.
