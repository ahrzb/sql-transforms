# sql-transform authoring contract

This is the current authoring specification for `sql-transform`.
Maintain this specification by topic as behavior changes.
It describes the implemented system and its refusals, not successive implementation plans.
The [glossary](../../../GLOSSARY.md) defines the project terms.
Confit and the native catalog keep their separate contracts.

The examples assume Python and SQL knowledge.
DuckDB executes batch SQL.
PyArrow (`pyarrow`) supplies columnar Arrow tables and their field schemas.
`pa` is the examples' import alias for PyArrow.

The estimator examples use scikit-learn, imported as `sklearn`.
NumPy supplies arrays, and pandas supplies DataFrames for the corresponding output modes.
Confit's linked serving contract defines engine details outside this authoring specification.

## 1. Start with authored fit and request SQL

A transform computes `(F, T) -> R` over relations.
`F` is the fit data, `T` is the request data, and `R` is the result.
Authored SQL names these arguments as `__FIT__` and `__THIS__`.
The author states the fit relationship in the SQL.

Fit is partial application: it binds `F` and writes params tables.
The fitted transform then computes `T -> R` without reading the original fit relation.
Window marginalization is a bounded convenience for writing this SQL.
It is not the definition of fit or the general model.

`SQLTransform` is the general, compositional model.
Its SQL can change the number of rows and can depend on several request rows.
`SQLProjection` adds a row-local rule: each request produces one answer independently of other requests.
Only `SQLProjection` offers Confit compilation.

These are separate checks:

1. The authoring model resolves and plans the SQL.
2. `SQLProjection` checks row-local SQL and the number of matches from params joins.
3. Confit either builds the fitted artifact or names a construct it does not serve.

A successful batch transform does not imply Confit admission.
The SQL check cannot establish a relation callback's semantics.
Relation callbacks can depend on their Arrow invocation and remain batch-only, including when referenced by a projection.
Compilation never substitutes another executor for a Confit refusal.

### Authored correlation and an honest serving refusal

Decorrelation rewrites supported correlated fit subqueries into fit queries and queries over params tables.
It runs before normal freezing of independent fit subqueries.
The fit-only predicates stay in the fit query.
Cross-relation equalities become keys.
Request-only predicates stay in the params-reading query.

The pass preserves each authored `=` or `IS NOT DISTINCT FROM` operator.
It distinguishes a missing group from a present group whose result is NULL.
It preserves the authored aggregate's empty-input value and false request guards.
These cases do not share one universal NULL or `COALESCE` rule.

This example uses guarded means and counts.
Batch execution preserves unseen keys, NULL keys, and false request guards.
Confit separately refuses this emitted correlated expression.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0], "enabled": [True, True, True]})
>>> requests = pa.table({"g": ["NEW", None, "a", "a"], "v": [2.0, 14.0, 12.0, 12.0], "enabled": [True, True, True, False]})
>>> authored = SQLProjection('''
... SELECT t.g,
...        t.v - (SELECT avg(f.v) FROM __FIT__ f
...               WHERE f.g IS NOT DISTINCT FROM t.g AND t.enabled) AS d,
...        (SELECT count(*) FROM __FIT__ f
...         WHERE f.g IS NOT DISTINCT FROM t.g AND t.enabled) AS n,
...        (SELECT count(*) FROM __FIT__ f
...         WHERE f.g = t.g AND t.enabled) AS n_eq
... FROM __THIS__ t
... ''')
>>> fitted = authored.fit(fit_data)
>>> answer = fitted.transform(requests).to_pydict()
>>> assert answer == {"g": ["NEW", None, "a", "a"], "d": [None, 7.0, -3.0, None], "n": [0, 1, 2, 0], "n_eq": [0, 0, 2, 0]}
>>> try:
...     fitted.compile()
... except ValueError as refusal:
...     assert str(refusal).startswith("unsupported:")
... else:
...     raise AssertionError("this correlated expression requires a Confit refusal")

```

Supported lifting includes aggregate scalar subqueries and correlated derived tables produced by member composition.
The pass preserves column aliases when flattening those tables.
It checks column names shadowed by nested relations and names unsupported correlations.
The [detailed decorrelation reference](../../../docs/decorrelation-unsupported.md) lists the unsupported forms.
This contract does not add inequality, prefix, or ASOF decorrelation.

Fit does not guarantee a constant artifact size or privacy.
Near-unique keys and list-valued aggregates can retain much of the fit data.
`WholeTrainingSet` names a refusal when an unreduced bare fit relation would remain in the artifact.
To retain fit data, write a subquery selecting the required rows and columns.
Use `(SELECT ... FROM __FIT__) f`.
Inspect the artifact rather than infer a size or disclosure limit from decorrelation.

## 2. General composition, execution, and outputs

Use `SQLTransform(sql, output='default', connection=None, captured=None)` for general SQL.
Members accept two relation arguments in SQL: `member(fit_relation, request_relation)`.
Members can contain other members, up to the public `MAX_DEPTH` limit.
`MAX_DEPTH` is the exported limit on nested member composition.
Composition resolves names within each member's captured scope.
An outer SQL name does not rebind a member's captured object.

Chaining states both fit and request arguments explicitly.
For example, `b(a(__FIT__, __FIT__), a(__FIT__, __THIS__))` fits `b` on `a`'s transformed fit data.
Pure SQL projection members also support scalar fit/transform composition with a struct of fitted values.
Explicit keyed projection members preserve the author's join equality.
Marginalized window projections are not scalar or keyed projection members.
Use an explicit grouped SQL projection member instead.

`run(transform, data)` binds both arguments to `data` and executes the resolved program directly.
It does not fit or freeze first.
This provides the reference for comparing a general transform's fit-data execution with its fitted execution.
Raw projection estimators do not support this direct `run` path.
Compare them with independent estimator execution instead.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLTransform, run
>>> fit_data = pa.table({"v": [10.0, 20.0]})
>>> requests = pa.table({"v": [12.0, 22.0]})
>>> shift = SQLTransform("SELECT t.v - p.lo AS v FROM __THIS__ t, (SELECT min(v) AS lo FROM __FIT__) p")
>>> scale = SQLTransform("SELECT t.v / p.m AS z FROM __THIS__ t, (SELECT avg(v) AS m FROM __FIT__) p")
>>> composed = SQLTransform("SELECT * FROM scale(shift(__FIT__, __FIT__), shift(__FIT__, __THIS__)) s", captured={"shift": shift, "scale": scale})
>>> fitted = composed.fit(fit_data)
>>> assert fitted.transform(requests).to_pydict() == {"z": [0.4, 2.4]}
>>> assert run(composed, fit_data).to_pydict() == {"z": [0.0, 2.0]}
>>> assert fitted.transform(fit_data).equals(run(composed, fit_data))

```

`fit(data)` returns `Fitted`, not the estimator itself.
The estimator also remembers that artifact for later `transform(data)` calls.
`fit_transform`, `get_params`, `set_params`, sklearn cloning, and `set_output` use the same program.
The optional sklearn `y` argument is ignored.
Put targets in columns that the authored SQL can read.

| Estimator output mode | Result of `SQLTransform.transform` |
| --- | --- |
| `default` or `arrow` | Materialized `pyarrow.Table` |
| `duckdb` | Lazy DuckDB relation |
| `pandas` | pandas DataFrame |
| `numpy` | NumPy array |

`Fitted.transform` always returns Arrow.
`Fitted.relation` returns a lazy relation, and `Fitted` is callable as `transform`.
For pandas output, the estimator preserves the source index only when row counts agree and no top-level ordering or limit breaks alignment.
General SQL has no implicit row-order promise.

A lazy relation belongs to the connection that built it.
Use the same caller connection for lazy chaining.
Keep the fitted artifact alive while consuming its lazy relations, including derived relations.
The artifact retains their table and function leases until `release()` or finalization.
`release()` is idempotent and invalidates outstanding relations that still need those registrations.
Use eager transforms for repeated serving without retained lazy registrations.

### Freezing order and the static DISTINCT pick

Fit freezes maximal independent fit subqueries into params tables.
It does not recursively freeze smaller queries inside an already frozen subtree.
Supported correlation lifting precedes the nested-query visit.
The planner records a common table expression's dependencies after rewriting its body.
Later reads therefore see the rewritten dependencies, not the original fit dependency.

One narrow exception also freezes a query with no remaining `__FIT__` read.
It must satisfy all these conditions:

- It is a flat ordinary `SELECT DISTINCT` of direct columns.
- Every selected column uses the same local source qualifier.
- It has one source and no other clauses, expressions, functions, or nested queries.
- Its source is an enclosing common table expression already rewritten to `SELECT * FROM` a prior fit-step params table.
- It has no outward references or duplicate output names.

A live source or a source that reads `__THIS__` does not qualify.
This rule lets window marginalization retain distinct keys and values instead of the full fit-time result table.
The planner executes each pick after its source.
It releases intermediates only after their last downstream fit use.
It removes dead common table expressions and ships only learned params reachable from the serving SQL.
If authored serving SQL still needs an explicitly retained whole-fit relation, that relation remains.

## 3. Explicit Python callbacks for general transforms

`Transform(fit, transform, takes, returns)` supplies two relation callbacks.
The fit callback receives a complete SQL aggregate group as an Arrow table.
It returns an instance that the fitted artifact retains.
The transform callback receives that instance and its rows within one Arrow invocation.
It does not receive the whole request relation or a fixed number of rows.

`takes` and `returns` declare ordered field names with authoritative DOUBLE types.
The callback must return those fields and one row per supplied row.
Callback results can depend on the invocation's rows.
Such callbacks are batch-only and cannot compile through Confit.
There is no `Transform.from_estimator` adapter.

```pycon
>>> import pyarrow as pa
>>> import pyarrow.compute as pc
>>> from sql_transform import SQLTransform, Transform, run
>>> def fit_mean(group):
...     return pc.mean(group["v"]).as_py()
>>> def subtract_mean(mean, rows):
...     return pa.table({"v": pc.subtract(rows["v"], mean)})
>>> center = Transform(fit_mean, subtract_mean, takes=("v",), returns=("v",))
>>> program = SQLTransform('''
... WITH p AS (SELECT center_fit(struct_pack(v := v)) AS theta FROM __FIT__)
... SELECT center_transform(p.theta, struct_pack(v := t.v)).v AS z
... FROM __THIS__ t, p
... ''', captured={"center": center})
>>> fit_data = pa.table({"v": [10.0, 20.0]})
>>> requests = pa.table({"v": [12.0, 22.0]})
>>> fitted = program.fit(fit_data)
>>> assert fitted.transform(requests).to_pydict() == {"z": [-3.0, 7.0]}
>>> assert run(program, fit_data).to_pydict() == {"z": [-5.0, 5.0]}

```

The generic callback protocol uses an opaque `STRUCT(type, id)` value.
`type` identifies the callback, and `id` selects its fitted instance.
Pure SQL members use a struct of fitted SQL values.
Raw projection estimators use BIGINT IDs.
These representations are separate protocols, not authenticated handles.

## 4. Projection-only estimators and scalar UDFs

Only `SQLProjection` admits captured raw row-local estimators and declared scalar user-defined functions (UDFs).
General `SQLTransform` admits explicit relation callbacks instead.
An estimator capture must be an instance with callable `fit` and `transform` methods, not an estimator class.
Here, “raw estimator” means that captured instance, rather than an explicit `Transform` callback.

`UDF` declares a scalar serving function's input and output types.
`PythonUDF` wraps a Python scalar callable with those declarations.
Declared names, arity, input types, and return types remain authoritative.
Case-insensitive builtin collisions and conflicting fit/transform names refuse.

### Canonical raw fit and application

A raw params query selects GROUP BY keys and exactly one `x_fit(bundle)` item.
Its source is direct `__FIT__` or one private inline SELECT over direct `__FIT__`.
The params query can have GROUP BY, but no other selected computations or query clauses.

The inline source can project ordinary windows and estimator-free scalar subqueries.
It cannot add joins, WHERE, GROUP BY, collapsing aggregates, or another source level.
Both levels exclude HAVING, QUALIFY, DISTINCT, set operations, ORDER BY, and LIMIT.
These source limits do not remove clauses inside an admitted estimator-free scalar subquery.

The params query must be a common table expression or derived table read directly by the request SELECT.
It cannot hide beneath another fit-only query.
A request application consumes that source's direct ID column in the same lexical scope.
The other legal application form is an inline canonical scalar fit subquery.
Arbitrary ID expressions and intermediate ID passthrough do not select a learned schema.

A bundle is a named scalar `struct_pack(...)`, or a bare column named by its final path component.
Fit normalizes a bare column to a named struct.
Application fields must match the fit fields in order.
Nested estimator bundles, whole rows, nested feature structures, and feature field access refuse.
Select learned output fields explicitly, or keep the ordinary output struct.

Each separately authored explicit fit source fits independently, even when its SQL text repeats.
One fit source can serve several applications.
Window marginalization alone coalesces equivalent inline fits by their printed source, keys, bundle fields, FILTER, and order.
Raw params stay separate from ordinary SQL-window params.

Fit creates per-scope IDs and publishes `PythonTransform` UDF objects.
Each such object wraps its scope's fitted estimator instances.
Fit publishes each learned UDF before a later fit step or the serving SQL binds.
Each UDF contains only its scope's ID-to-estimator mapping.
The artifact's `instances` mapping references those same estimator objects.

IDs are schema selectors, not authentication.
Unknown runtime IDs retain their named error behavior.

A display struct such as `struct_pack(type := 'sc', id := p.iid)` is ordinary SQL, not a consumable raw handle.
The examples name a params ID column `iid`.
That column name is an authored alias, not a special handle.

### Schema, conversion, ordering, and empty groups

BIGINT and DOUBLE are DuckDB's 64-bit integer and floating types.
Arrow names the corresponding declarations `int64` and `float64`.

Raw fit reads the actual Arrow bundle schema.
Integer declarations widen to int64, floating declarations widen to float64, and boolean declarations remain boolean.
String and large-string declarations become string.
Other feature types refuse by name.
Generated application calls cast features to these learned declarations.

At the estimator boundary, numeric values become Python floats and numeric NULL becomes NaN.
Numeric-only matrices use float64.
`dtype` is a NumPy array's element-type declaration.
Matrices with string or boolean fields use object dtype and preserve those values.

Numeric-looking strings stay strings.
Use an explicit SQL CAST when numeric conversion is intended.
Integer values above `2**53` can lose precision at the numeric estimator boundary.
A declared integer scalar UDF does not inherit this estimator conversion.

Learned outputs are a DOUBLE struct.
Output width is the number of fields in that struct.
Fit validates one-row output shape and requires equal output schemas across groups within one scope.
`Named` wraps an estimator with authoritative output labels, including width-one labels.
Estimator feature names supply labels when usable; otherwise labels are `f0`, `f1`, and so on.

Missing fields, case-colliding fields, and unsupported output shapes refuse.
Different scopes can learn different widths from the same prototype.

FILTER and in-call ORDER BY belong on the fit item.
Fit preserves declared directions, NULL placement, and collation, then uses materialized input position to order ties.
Unordered raw fits also preserve materialized input position.
`OrderSensitive`, including nested `Named` wrappers, requires in-call ORDER BY.
`OrderSensitive` is the wrapper that declares this ordering requirement.

An ORDER BY in the fit window clause is a running fit and refuses.
Literal orders remain literal and follow normal DuckDB binding policy.
A DuckDB relation's fit order means only its materialized result order.

Raw fit materializes its source once and collects complete groups through Arrow list aggregates.
It uses omitted Arrow parameter declarations and returns BIGINT IDs.
Filtered-empty groups create no estimator instance, and fit removes their NULL-ID rows without fitting again.
Empty fit data and entirely filtered scopes raise `TransformError`.
A missing params match supplies a NULL ID and produces a NULL result.

Raw fit refuses `__cf_fit_row` and `__cf_row` fit columns, case-insensitively, before adding private ordinals.
Batch projections also reserve `__cf_row` in request data.
Ordinary single-underscore names remain ordinary columns.
Private ordinals never become public output columns.

### Explicit scaler, public reconstruction, and strict native selection

This example fits one scaler per key.
An unseen key returns NULL, and the NULL key matches its fitted group.
The four public serving fields reconstruct Confit without private state.
Strict native conversion must produce native entries or raise `NotNative`.
It cannot silently retain Python execution.

```pycon
>>> import pyarrow as pa
>>> from sklearn.preprocessing import StandardScaler
>>> from confit import DuckDBInferFn
>>> from sql_transform import PythonTransform, SQLProjection
>>> from sql_transform.native import to_native
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["a", "NEW", None], "v": [20.0, 4.0, 14.0]})
>>> projection = SQLProjection('''
... WITH p AS (SELECT g, sc_fit(v) AS iid FROM __FIT__ GROUP BY g)
... SELECT t.g, sc_transform(p.iid, t.v).v AS z
... FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
... ''', captured={"sc": StandardScaler()})
>>> fitted = projection.fit(fit_data)
>>> expected = [{"g": "a", "z": 1.0}, {"g": "NEW", "z": None}, {"g": None, "z": 7.0}]
>>> assert fitted.transform(requests).to_pylist() == expected
>>> assert all(isinstance(udf, PythonTransform) for udf in fitted.udfs.values())
>>> assert all(fitted.instances[iid] is instance for udf in fitted.udfs.values() for iid, instance in udf.instances.items())
>>> public = DuckDBInferFn(sql=fitted.sql, row_tables={"__THIS__": fitted.schema}, static_tables=fitted.params, udfs=list(fitted.udfs.values()), shape="map")
>>> assert public.infer_rows(requests.to_pylist()) == expected
>>> assert public.infer_arrow(requests).to_pylist() == expected
>>> native_udfs = [to_native(udf, strict=True) for udf in fitted.udfs.values()]
>>> native = DuckDBInferFn(sql=fitted.sql, row_tables={"__THIS__": fitted.schema}, static_tables=fitted.params, udfs=native_udfs, shape="map")
>>> assert native.infer_rows(requests.to_pylist()) == expected
>>> assert native.infer_arrow(requests).to_pylist() == expected
>>> replay = SQLProjection(projection.source, captured=projection.captured).fit(fit_data)
>>> assert replay.transform(requests).to_pylist() == expected

```

Native selection is explicit and has no automatic cache.
The [native catalog documentation](native/README.md) defines supported entries and parity bounds.
The native contract and parity bounds apply independently of the authoring model.

### A lawful window-derived fit feature

A raw inline source can compute a window-derived feature over the original fit data.
The request expression must use the same fitted mean.
Here the mean comes from a DISTINCT pick of the original window values, not a replacement GROUP BY reduction.

```pycon
>>> import pyarrow as pa
>>> from sklearn.preprocessing import StandardScaler
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["a", "NEW", None], "v": [20.0, 4.0, 14.0]})
>>> fitted = SQLProjection('''
... WITH c AS (SELECT g, avg(v) OVER (PARTITION BY g) AS m FROM __FIT__),
... means AS (SELECT DISTINCT c.g, c.m FROM c),
... p AS (SELECT g, sc_fit(cv) AS iid
...       FROM (SELECT g, v - avg(v) OVER (PARTITION BY g) AS cv FROM __FIT__) f
...       GROUP BY g)
... SELECT t.g, sc_transform(p.iid, struct_pack(cv := t.v - means.m)).cv AS z
... FROM __THIS__ t
... LEFT JOIN means ON t.g IS NOT DISTINCT FROM means.g
... LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
... ''', captured={"sc": StandardScaler()}).fit(fit_data)
>>> expected = [{"g": "a", "z": 1.0}, {"g": "NEW", "z": None}, {"g": None, "z": 7.0}]
>>> assert fitted.transform(requests).to_pylist() == expected
>>> assert fitted.compile().infer_rows(requests.to_pylist()) == expected
>>> assert fitted.compile().infer_arrow(requests).to_pylist() == expected

```

## 5. Source replay and the public fitted artifact

Constructor `source` is author-level explicit SQL.
Replay or clone with `source` and `captured`.
The constructor's `sql` is resolved diagnostic SQL, not replay input.
In particular, lowered list-fit calls must not pass through parsing as authored fit calls again.
For marginalization, `source` contains the explicit SQL produced by the derivation.

The constructor captures referenced caller names without retaining the caller frame.
Explicit `captured` entries win over caller names.
The constructor adopts the captured mapping so sklearn cloning preserves its identity.
Private per-scope estimator bindings do not add author capture keys.
Replay regenerates them.

`SQLProjection.fit(data)` returns `FittedProjection`.
Its public fields have these meanings:

| Field or method | Meaning |
| --- | --- |
| `sql` | Standalone unordered serving SQL; no public batch ordinal |
| `schema` | Nullable request schema containing the fit columns the serving SQL reads |
| `params` | One stored mapping of normalized captured statics and serving-live learned tables |
| `udfs` | UDF objects under the names used by serving SQL |
| `instances` | Fitted estimator or callback instances keyed by ID |
| `transform(data)` | Arrow batch result in request order |
| `compile()` | A fresh Confit `DuckDBInferFn` with shape `map`, or an explicit refusal |

For Confit-admitted projections, `sql`, `schema`, `params`, and `udfs` are the complete public serving artifact.
Read Confit metadata and call `infer_rows` or `infer_arrow` on the returned function.
The projection does not forward those interfaces.
Direct execution of public serving SQL has no row-order promise.
Batch `transform` uses a private ordinal query to preserve request order.

The request schema keeps real input widths.
Raw feature casts do not widen every request column.
Confit can therefore refuse a referenced float32 request column even when its learned estimator declaration is float64.
No declared-schema selection, missing-column policy, or extra-column dropping overrides normal DuckDB and Confit binding.

Projection fit normalizes captured non-Arrow relations to Arrow once.
DuckDB relations use `to_arrow_table()`, not a reusable Arrow reader.
Fit, probes, batch, compilation, and public-field reconstruction all read the same stored `params` mapping.
Callers may replace mapping entries, but must treat underlying Arrow buffers as read-only.
Refit when a captured source intentionally changes.
General `Fitted.params` remains learned-only, with separate live captured bindings.

Caller-catalog tables remain batch-only unless the author captures an Arrow snapshot and refits.
Fit, join probes, and batch execute on the caller connection when supplied.
Compilation refuses catalog dependencies because Confit has no caller catalog.
The package never silently captures catalog tables.

## 6. Connections, leases, and errors

Each execution leases private table and function names on a shared connection.
This includes scalar UDFs, fit functions, learned transform functions, and numbered raw-fit sources.
Function adapters change names without duplicating registration or requiring dataclass UDFs.
Overlapping artifacts therefore keep independent registrations, including outstanding general lazy relations.

Cleanup releases only registrations that succeeded.
A partial registration failure releases earlier registrations.
The first execution or callback error remains the reported error.
Cleanup or thread-restoration errors cannot replace it or skip later cleanup.

Projection fit, probes, and batch use `threads=1`.
They save the observed thread setting and restore that value with SET after releasing registrations.
They do not RESET the database setting.
The setting is database-wide.
Callers must not change it concurrently on the same database.
General `SQLTransform` retains its own thread policy.

## 7. Bounded window marginalization

Call `SQLProjection.marginalize(sql, connection=None, captured=None)` for the kept window convenience.
The ordinary constructor expects explicit fit/request SQL and does not marginalize automatically.
The convenience accepts one SELECT over `__THIS__`, not a common-table-expression or derived-table projection chain.
It refuses source joins, WHERE, GROUP BY, HAVING, QUALIFY, SAMPLE, and set operations.
Top-level DISTINCT, ORDER BY, and LIMIT also refuse.
Use general SQLTransform for computations that change request rows.

Admitted window values must depend on row-visible keys, not physical row position.
Partition keys determine whole-partition values.
For moving RANGE or GROUPS frames, order values also become lookup keys.
An order key absent from fit data returns a lookup miss, not a newly evaluated window over requests.
COLLATE remains in the window expression but is removed from lookup keys.

| Window feature | Admission boundary |
| --- | --- |
| SQL aggregates | FILTER, DISTINCT, ordered arguments, and serialized named windows remain supported |
| Frames | Whole-partition frames and admitted RANGE/GROUPS frames with valid constant bounds |
| Rank family | `rank`, `dense_rank`, `percent_rank`, and `cume_dist` |
| Value windows | Admitted `first_value`, `last_value`, and `nth_value`; nth offset must be a positive integer constant |
| IGNORE NULLS | Admitted value windows only, not general aggregates or raw fit windows |
| Physical-position functions | `row_number`, `ntile`, `lag`, and `lead` refuse |
| Other frame features | Bounded ROWS, EXCLUDE, invalid bounds, and unsupported offsets/defaults refuse |

Window fit arguments cannot contain nested windows, aggregates, subqueries, estimator bundles, or projection calls.
Positional column references and whole-row fit arguments also refuse.
Use explicit stages and named scalar features instead.

The convenience preserves uncorrelated scalar and EXISTS subqueries over fit data.
It rebinds their `__THIS__` reads to `__FIT__` and retains inner clauses.
Correlated subqueries, subqueries over other tables, and IN subqueries refuse in this convenience.
Use explicit authored SQL for supported core correlations.

A bare captured estimator call, `x(bundle)`, means global fit plus row-local application.
A partitioned estimator uses inline `x_transform(x_fit(bundle) OVER (...), bundle)`.
Standalone raw fit-window output, including a parked `_th` alias, refuses.

Raw fit windows require whole partitions without window-clause running order.
Their FILTER and in-call ORDER BY remain supported.
Pure SQL projection fit windows exclude FILTER, DISTINCT, argument ORDER BY, IGNORE NULLS, and running order.
Use explicit SQL when a projection member needs those computations.

Simple `*` and qualified request stars remain supported.
The derivation qualifies request stars so params fields cannot enter request output.
It preserves DuckDB's unaliased output names.
Ordinary scalar lateral aliases remain legal when they require no expansion into a fit expression.
A fit expression cannot read a sibling output alias.
Repeat the expression or state an explicit stage instead.

### Full-window numerical fidelity

The convenience builds one fit-time result table from the original executable SELECT items.
It retains window expressions, surrounding arithmetic, and their order.
It omits top-level star passthrough only from that private table, not from request output.
It appends ordinary window values and lookup keys, then selects distinct key/value columns for each scope.

This preserves the original window computation instead of replacing it with a GROUP BY aggregate.
DuckDB can change floating reduction order when windows execute separately or as grouped aggregates.
The numerical reference is the original admitted window SELECT over the fit data at `threads=1`.
Exact schemas and values include signed zero and NaN behavior.
Grouped SQL written by an author is lawful, but is not a universal bit-exact replacement for a window.

Raw estimator fits remain independent of this private SQL-window table.
Mixed SQL-window and estimator queries keep the original ordinary windows, including windows inside raw feature expressions.
Raw feature, FILTER, and order columns do not become reads of the SQL-window table.

Window lookups match keys with `IS NOT DISTINCT FROM`.
NULL partition keys form one fitted partition.
A missing LEFT params match preserves the request and returns NULL values.
An empty SQL-window LEFT params table also returns NULL values.

A CROSS params side requires exactly one match.
Empty CROSS tables and duplicate-key matches refuse.
A request-preserving outer params side allows zero or one match.
Supported authored correlations keep their separate empty-input and guard behavior.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["NEW", None, "a"], "v": [2.0, 14.0, 12.0]})
>>> fitted = SQLProjection.marginalize("SELECT g, v - avg(v) OVER (PARTITION BY g) AS d FROM __THIS__").fit(fit_data)
>>> expected = [{"g": "NEW", "d": None}, {"g": None, "d": 7.0}, {"g": "a", "d": -3.0}]
>>> assert fitted.transform(requests).to_pylist() == expected
>>> fn = fitted.compile()
>>> assert fn.infer_rows(requests.to_pylist()) == expected
>>> assert fn.infer_arrow(requests).to_pylist() == expected

```

## 8. Unsupported forms and explicit alternatives

The forms below are not admitted.
Explicit alternatives preserve computations without making unsupported syntax legal.
There is one authoring model and no compatibility layer or alternative marginalizer.

| Unsupported interface or syntax | Supported alternative |
| --- | --- |
| Automatic constructor marginalization | Call `SQLProjection.marginalize`, or author `__FIT__`/`__THIS__` SQL |
| Window projection chains | State fit and request stages explicitly, or compose general SQLTransform members |
| `this_schema`, COLUMNS, and star EXCLUDE/REPLACE/RENAME expansion | Select explicit columns and aliases; retain normal binding checks |
| Hidden `_` aliases and alias expansion into fit expressions | Repeat the expression or state an explicit stage; `_` input names remain ordinary |
| Standalone fit-window outputs and parked fit aliases | Use inline fit/application or a canonical explicit params source |
| Automatic `unnest(transformer(...))` field expansion | Select named fields or return the ordinary struct |
| `Transform.from_estimator` | Capture a row-local estimator in SQLProjection, or supply explicit relation callbacks in SQLTransform |
| Marginalized scalar/keyed projection members | Use an explicit plain or grouped SQLProjection member |
| Value-dependent numeric-string coercion | Keep strings, or author a SQL CAST |
| Shared-CTE, deep, or joined raw fit sources; intermediate ID passthrough | Use direct FIT or one inline FIT projection, then a direct request-level ID |
| Live non-Arrow projection captures | Normalize once to Arrow artifact state and refit for source changes |
| Fit returning the estimator itself | Use `Fitted` or `FittedProjection`; SQLTransform also remembers its fitted state |
| `serving_sql`, `infer`, and `infer_batch` forwarding | Use fitted `sql`, then `compile().infer_rows` or `compile().infer_arrow` |
| Estimator backend, boundary, or output-schema metadata | Read Confit metadata and the fitted projection's `schema` |
| `from_file` | Read SQL files explicitly before calling the constructor |
| Public ordinal SQL and learned-only projection params | Use unordered serving `sql` and the stored complete `params` mapping |
| Standalone `marginalize`, `Marginalized`, `MarginalizeError`, `FitStep`, `ParamsSpec`, and `UDFSpec` | Use SQLProjection methods, fitted artifacts, and `TransformError` subclasses |
| `sql_transform.model` imports | Import the public API from `sql_transform` |

Raw fit also refuses request-data refitting, applications inside fit-only subtrees, nested estimators, and learned-output-to-later-fit composition.
The positive canonical ID grammar does not trace forged structs, altered IDs, or cross-scope expressions.
No exporter, runtime authentication, or intermediate-handle tracing extends that grammar.

An explicit projected fit source needs no marginal chain walker.
This grouped example states its fit computation and request computation independently.
It does not claim the grouped reduction reproduces every original window's reduction order.

```pycon
>>> import pyarrow as pa
>>> from sql_transform import SQLProjection
>>> fit_data = pa.table({"g": ["a", "a", None], "v": [10.0, 20.0, 7.0]})
>>> requests = pa.table({"g": ["NEW", None, "a"], "v": [2.0, 14.0, 12.0]})
>>> fitted = SQLProjection('''
... WITH p AS (
...   SELECT g, avg(x) AS m
...   FROM (SELECT g, 2 * v AS x FROM __FIT__) f GROUP BY g
... )
... SELECT t.g, 2 * t.v - p.m AS d
... FROM __THIS__ t LEFT JOIN p ON t.g IS NOT DISTINCT FROM p.g
... ''').fit(fit_data)
>>> assert fitted.transform(requests).to_pydict() == {"g": ["NEW", None, "a"], "d": [None, 14.0, -6.0]}

```

General relation callbacks remain batch-only.
Caller catalogs require explicit snapshots for Confit.
Raw estimators retain per-scope learned schemas and IDs.
Authoring admission does not extend the native catalog or Confit's SQL support.
