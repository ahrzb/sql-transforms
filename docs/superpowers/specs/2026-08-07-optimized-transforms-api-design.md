# Optimized transforms: the family protocol and typed UDF entries

Date: 2026-08-07
Status: design, approved in dialogue with AmirHossein (this session)
Supersedes the open question in DRAFT-23's 2026-08-04 addendum; refines
DRAFT-22's addendum (typed udf entries) and DRAFT-24's named-struct framing.

**Scope note (2026-08-07, later the same session):** the confit half of this
document — artifacts carried as `instances` on a typed UDF entry — assumes
DRAFT-22's `udfs=` surface, which is not the current confit direction. Native
scoring in confit is specified separately in
`2026-08-07-confit-tree-ensemble-design.md`, where a model is a prepare-time
declaration and scoring is one IR instruction. Read that document for the
engine side; this one stands for the family protocol (tiers, full-SELECT
halves, symbolic θ, the shape tier) and the schema-typed UDF protocol.

## The problem

A fitted sklearn transformer serves today as an opaque Python callout: one
`PythonTransform` UDF per fit step, per-group estimators held in an
`instances` dict, `est.transform([row])` per row. That is correct, and it is
the permanent fallback and the oracle — but it is the slow path, and it is
the only path.

This spec defines how a transform gets *fast*: how an author writes a
transform family the engine can compile into SQL or into a native kernel,
and how those two optimized tiers reach the serving boundary without a
second SQL surface, a second store, or a second correctness story.

## The three tiers

A **family** is a plain Python class supplied in `fits=`. Which methods it
defines decides its tier — no registration, no base class, no flags.

| tier | provides | fit runs | serving cost |
|---|---|---|---|
| SQL | `fit_sql` + `transform_sql` | one aggregate SQL pass | inlined SQL; `udfs=[]` |
| artifact | `fit` + `transform_sql`, with an artifact field in `Theta` | Python per group | one native kernel call |
| legacy | `fit` + `transform` (sklearn) | Python per group | `PythonTransform` trampoline |

Defining both `fit_sql` and `fit` refuses by name at construction.

Legacy is not an engine mode. It is one family shipped in the box
(`SklearnTransform`, below) whose artifact is the live estimator and whose
kernel is the existing trampoline. The protocol is total; the clone-per-group
loop in `_projection.py` becomes that family's implementation.

## Halves are full SELECT statements

A family's two halves are `string.templatelib.Template` values whose slots
are **relations**, not expression fragments:

```python
class StandardScaler:                                    # SQL tier
    def fit_sql(self, data):
        return t"SELECT avg(x) AS mean, stddev_pop(x) AS scale FROM {data}"
    def transform_sql(self, data, th):
        return t"SELECT (d.x - th.mean) / th.scale AS v FROM {data} d, {th} th"
```

Expression fragments were rejected: they are a second SQL surface with no
joins, no CTEs, no subqueries, no window functions, and they would need a
fragment grammar maintained alongside marginalize forever. Full statements
splice through the same marginalize/nesting path as user SQL, so every engine
feature is inherited permanently. The immediate payoff: a fit half may use
GROUP BY, a join, or a subquery — `TargetEncoder.fit_sql` is expressible, and
was not under fragments.

```python
class TargetEncoder:                                     # SQL tier, joined fit
    def fit_sql(self, data):
        return t"""SELECT list(x ORDER BY x) AS cats, list(m ORDER BY x) AS means
                   FROM (SELECT x, avg(y) AS m FROM {data} GROUP BY x)"""
    def transform_sql(self, data, th):
        return t"SELECT th.means[list_position(th.cats, d.x)] AS enc FROM {data} d, {th} th"
```

Input columns bind by canonical name (`x`, `y`) through the existing input
renaming; output columns are SELECT aliases, and the existing alias lints
(identifier, `_`-leading, case collision) apply.

### Injected parameters

A half's parameters are supplied by the engine, matched by name — the
signature declares what the family needs and nothing else:

| name | meaning | effect |
|---|---|---|
| `data` | the input relation | required |
| `th` | the θ relation, symbolic | required in `transform_sql` |
| `inputs` | resolved input schema (`.names`, Arrow types) | available at construction |
| `shape` | θ shape descriptor (lengths, key names) | opts into the shape tier |

`fit`'s own Python signature names the fit bundle (`X`, `y`, or a variadic
column list); a keyword-only parameter present in `fit` but absent from
`transform_sql` — `y` — is a fit-only input and never reaches serving.

## θ is declared, never inspected

```python
class DTreeEncoder:                                      # artifact tier
    def __init__(self, max_depth=None):
        self.max_depth = max_depth                       # hyperparams on self: fine
    class Theta:
        artifact: TreeEnsembleArtifact                   # -> instances of a typed entry
    def fit(self, X, y):                                 # stateless: returns theta
        return self.Theta(artifact=TreeEnsembleArtifact.from_sklearn(
            DecisionTreeRegressor(max_depth=self.max_depth).fit(X, y)))
    def transform_sql(self, data, th):
        return t"""SELECT dt({th.artifact}, struct_pack(price := d.price, sqft := d.sqft)).pred
                          AS pred
                   FROM {data} d, {th} th"""
```

Two field kinds, decided by the declared **type**, never by sniffing a value:

- plain (scalar, list, list of list) — becomes a column of the params table,
  and `th.mean` renders as `p0.mean`.
- artifact type — becomes an entry in a typed UDF's `instances`, and
  `th.artifact` renders as the instance-id column `p0.__cf_est`, exactly the
  mechanism `PythonTransform` uses today.

Fitted state never lives on `self`; `fit` returns `Theta`. Models are never
serialized into structs or lists. An artifact is the model *lowered to plain
data* (below), and the live estimator survives only inside the legacy family.

## `transform_sql` never sees fitted values

The θ argument passed to `transform_sql` is symbolic: it yields column
references, never data. Consequences, all of them load-bearing:

- The statement text depends only on hyperparams. **Per-group text divergence
  is unrepresentable**, not checked — no template hashing, no normal form.
- Serving text stays final at construction (`_projection.py`'s "nothing left
  to rewrite at fit" invariant is preserved), so the composed SQL is DuckDB
  bind-probed before any data — P7, refusal by name.
- `udfs=` is derivable at construction: the artifact *type* on `Theta` names
  the entry kind. Vocabulary resolution never waits for fit.
- Rendering happens once, not per group.

### The shape tier

Learned-shape families (a OneHot lane per category) may read θ **shape** —
lengths and key names — but never θ values:

```python
class OneHotEncoder:
    class Theta:
        cats: list
    def fit(self, x):
        return self.Theta(cats=sorted({v for v in x if v is not None}))
    def transform_sql(self, data, th, shape):
        lanes = tjoin(", ", (t"CAST(d.x = th.cats[{i+1}] AS TINYINT) AS cat_{i}"
                             for i in range(shape.cats)))
        return t"SELECT {lanes} FROM {data} d, {th} th"
```

A shape-reading family renders **once per fit**, after the shape descriptors
of all groups are compared. Divergence (3 categories in one group, 5 in
another) is the shape-uniformity refusal already at `_projection.py:364`, now
able to say which groups differ and by what. Shape-free families keep
construction-time finality and bind-probing; only shape-reading families move
text finality to fit.

Rejected alternative: emit one LIST column and let shape live in the value.
It needs no new tier, but per-group divergence then produces ragged lists
instead of a refusal — converting a construction-time law into bad data — and
the output stops being sklearn-shaped lanes.

## Artifacts are data; the kernel is Rust

An artifact is a struct-of-arrays the Rust kernel executes directly. The
sklearn object dies at extraction.

```python
@dataclass(frozen=True)
class TreeEnsembleArtifact:
    kind, version = "tree_ensemble", 1
    feature: list[int]        # node -> feature index; -1 = leaf
    threshold: list[float]    # x <= t goes left
    left: list[int]
    right: list[int]
    value: list[float]
    tree_root: list[int]      # tree -> root node; iteration order FIXED here
    agg: Literal["sum", "mean"]
    base: float
    link: Literal["identity", "sigmoid"]
    names: tuple[str, ...]    # feature names, checked against the entry's schema
    @classmethod
    def from_sklearn(cls, est): ...   # DecisionTree*, RandomForest*, GradientBoosting*
```

```rust
// confit-core: THE implementation. Both engine paths bind this one function.
pub struct TreeEnsemble { feature: Vec<i32>, threshold: Vec<f64>, /* ... */ }
impl TreeEnsemble {
    pub fn from_payload(bytes) -> Result<Self, Refusal>  // validates here, refuses by name
    pub fn predict(&self, feats: &[f64]) -> f64          // row path binds this
    pub fn predict_batch(&self, cols, out)               // batch path: same traversal
}
```

There is **no Python reference twin**. A twin would be a second
implementation posing as an oracle; the oracle is sklearn, one level up. The
DuckDB batch registration is an Arrow UDF whose body is one pyo3 call into
`predict_batch`, so batch and row execute the same code and C2 holds by
construction rather than by test.

Globals: the **kind registry** (kind name -> validator + kernel) is global,
immutable code. Fitted artifacts are never global — they live in the entry's
`instances`, and the entry's lifetime is their scope.

### Which algorithms need an artifact

Expressibility — which algorithms *can* be written in the SQL tier at all,
independent of whether that is the fastest way to serve them:

| algorithms | expressible in SQL over θ |
|---|---|
| scalers, ordinal/one-hot/target encoding | yes |
| linear, logistic, PCA | yes — dot products over θ lists |
| decision tree, random forest, GBM, XGBoost, LightGBM | no: traversal is a loop |
| kNN, kernel SVM | no: the training set is the state |

Scope, decided 2026-08-07: the first kernel is `tree_ensemble` only.

Whether the SQL-tier-expressible matvec cases (linear, logistic, PCA) would
still serve faster through a kernel is **open and unmeasured** — see the open
question in `2026-08-07-confit-tree-ensemble-design.md`. Nothing in this spec
depends on the answer: those families are written in the SQL tier either way,
and a kernel would change only what their template calls.

## The UDF protocol, schema-typed

`takes`/`returns` become Arrow schemas; `take_names`/`return_names` and the
`"i1"|"i64"|"f64"|"str"` string vocabulary are deleted (v0, no compat shim).
Arrow is already the currency on every side and serializes for export.

```python
class UDF:
    name: str
    takes:   pa.Schema        # input struct  S
    returns: pa.Schema        # output struct T
    def __call__(self, *args) -> tuple | None: ...              # positional, in field order
    def apply_batch(self, b: pa.RecordBatch) -> pa.RecordBatch: ...
```

- A UDF's SQL signature is `(STRUCT(S)) -> STRUCT(T)`, plus the engine-owned
  leading `BIGINT` instance id for instance-carrying entries. The id is never
  a schema field.
- Measured on DuckDB 1.5.5: a struct argument binds **name-keyed** —
  `struct_pack(sqft := ..., price := ...)` is reordered into declared order at
  bind, and an unknown field set raises a BinderException before any row
  moves. Names cost nothing at runtime; the row path stays positional.
- Measured: DuckDB's Python API refuses two functions with one name, and
  rejects `"ANY"` as a parameter type. Therefore **one registration per (kind,
  input schema)** — never per group, never per fit step. Steps sharing a
  schema share the entry and the name; a second schema gets a suffix
  (`dt`, `dt_1`).
- Deletes the `exec`-generated arity wrapper in `register()` (arity is now 1
  or 2) and `lane_of`'s hand-rolled search (`Schema.get_field_index`, with the
  existing case-collision refusal kept because Arrow lookup is
  case-sensitive while DuckDB's struct access is not).

```python
class TreeEnsemble(UDF):                     # confit.udfs.TreeEnsemble
    kind = "tree_ensemble"                   # names the Rust kernel
    def __init__(self, name, takes, returns, instances): ...
    # __post_init__: every artifact's `names` must equal `takes` field names -> refuse by name
    #   (else a tree trained on (price, sqft) serves (sqft, price) and returns plausible garbage);
    #   lowers all instances into ONE Rust kernel object held on the entry; drop -> freed.
    # __call__(iid, *feats): NULL id -> None; unknown id -> raise; else the Rust row fn.
    # apply_batch: one pyo3 call per chunk into the same kernel.
```

## The serving boundary

```python
p1 = pa.table({"region": ["SF", "NY"], "__cf_est": [0, 1]})
p2 = pa.table({"region": ["SF", "NY"], "__cf_est": [2, 3]})

fn = DuckDBInferFn(
    """
    SELECT dt  (p1.__cf_est, struct_pack(price := r.price, sqft := r.sqft)).pred AS pred,
           dt_1(p2.__cf_est, struct_pack(dom := r.dom)).pred                     AS pred_dom
    FROM __THIS__ r
    LEFT JOIN p1 ON (r.region IS NOT DISTINCT FROM p1.region)
    LEFT JOIN p2 ON (r.region IS NOT DISTINCT FROM p2.region)
    """,
    row_tables={"__THIS__": Row},
    static_tables={"p1": p1, "p2": p2},
    udfs=[TreeEnsemble("dt",   takes=S1, returns=T, instances={0: art_a_sf, 1: art_a_ny}),
          TreeEnsemble("dt_1", takes=S2, returns=T, instances={2: art_b_sf, 3: art_b_ny})],
    shape="map",
)
fn({"region": "SF", "price": 700_000, "sqft": 900, "dom": 12})   # -> {"pred": .74, "pred_dom": .31}
fn({"region": "TX", ...})                                        # -> both None
```

Two entries because two input schemas. Four fitted trees because four
groups; the id does all per-group work. Nothing is registered globally and
nothing is registered per fit.

P14 has two mechanisms and one behaviour: an instance-carrying call gets NULL
from the protocol itself (NULL id -> `None`), while an **inlined** SQL-tier
family needs an engine-inserted guard, because a family body could otherwise
compute a real value from NULL θ columns:

```sql
CASE WHEN p0.__cf_est IS NULL THEN NULL ELSE (r.sqft - p0.mean) / p0.scale END
```

Params tables therefore keep `__cf_est` as a presence witness even when θ is
entirely plain columns.

Confit's row path is unchanged in cost: the IR stays scalar (DRAFT-22), and
the frontend lowers `struct_pack(...)` to positional extern arguments at build
using the registered field order. No struct is materialized per row. The
extern slot holds the native kernel instead of the boxed trampoline — the last
row of DRAFT-22's binding table.

## The legacy family

```python
class SklearnTransform:
    def __init__(self, proto): self.proto = proto
    class Theta:
        artifact: SklearnArtifact                        # the live estimator, opaque
        names: Keys                                      # feature_names_out -> SHAPE
    def fit(self, X, y=None):
        est = clone(self.proto).fit(X, y)
        return self.Theta(artifact=SklearnArtifact(est), names=est.get_feature_names_out())
    def transform_sql(self, data, th, shape, inputs):
        outs = tjoin(", ", (t"o.out[{i+1}] AS {ident(n)}" for i, n in enumerate(shape.names)))
        feats = tjoin(", ", (t"{ident(c)} := d.{ident(c)}" for c in inputs.names))
        return t"""SELECT {outs}
                   FROM (SELECT sklearn_transform({th.artifact}, struct_pack({feats})) AS out
                         FROM {data} d, {th} th) o"""
```

`sklearn_transform` is today's `PythonTransform` behaviour wearing a
vocabulary name. Output names ride the shape tier (they must be uniform
across groups anyway) rather than a params column. Its θ is the only θ that
cannot export — a named limitation of one family, not a hole in the system.

## The gate

Acceptance is **swap the entry**, nothing else:

```python
udfs=[PythonTransform("dt", takes=S, returns=T, instances={0: sk_tree_sf, 1: sk_tree_ny})]  # oracle
udfs=[TreeEnsemble   ("dt", takes=S, returns=T, instances={0: art_sf,     1: art_ny})]      # fast
```

Same serving SQL, same static tables, same rows, run on both paths (DuckDB
batch and confit row). Because the oracle side holds the live sklearn object,
one gate covers extraction (`from_sklearn` lowering) and execution (the
kernel) together. Tree kinds compare exactly (comparisons plus leaf sums) and
gate bit-for-bit; matvec-class kinds carry a declared per-kind ulp tolerance,
written down per kind rather than discovered per bug (DRAFT-22).

SQL-tier families have no entry to swap — `udfs=` is empty and the optimized
form is inlined SQL. Their gate is one level up: same authoring SQL, swap the
family against the sklearn wrapper, compare outputs on both paths. This is the
one place where the two sides do not share a serving text.

## What this changes in the codebase

- `_udf.py`: schema-typed `takes`/`returns`; delete `take_names`,
  `return_names`, `_TYPES`, `_check_types`, the `exec` arity wrapper.
- `_marginalize.py`: one shared family-recognition predicate replaces the
  ~5 `hasattr(fit)/hasattr(transform)` gates (:1515, :1863, :2041, :2085).
  It must see through `Named`/`OrderSensitive` `__getattr__` forwarding, or a
  wrapped family silently degrades to the legacy tier.
- `_projection.py`: `_fit_step` grows two new paths (SQL tier: compose the
  aggregate into params materialization, no Python at all; artifact tier: call
  `fit` per group, distribute `Theta` fields into params columns and entry
  instances). The clone-per-group loop stays only as `SklearnTransform`'s
  implementation.
- Template hygiene refusals: non-empty conversion or format spec in a slot;
  ndarray θ (np scalars coerce, arrays refuse by name); `tjoin` separators are
  static template text.
- `confit.udfs`: `TreeEnsemble` entry + `tree_ensemble` Rust kernel; the kind
  registry.

## Export (slice 6) and other runtimes

Every optimized artifact is plain data, so the serving bundle — serving SQL,
params tables as Arrow, entry declarations with Arrow schemas, artifact
payloads — serializes with no pickle anywhere. The same Rust crate compiled to
WASM gives Go and Java the identical kernel (validated by the wazero/chicory
spike), so a fitted tree ensemble serves outside Python with no per-language
work.

## Sequencing

1. Schema-typed UDF protocol in `_udf.py` (mechanical, gated by the existing
   suite) — lands independently of everything below.
2. Family recognition + the SQL tier end to end (`StandardScaler`,
   `TargetEncoder`), with `SklearnTransform` still the legacy path.
3. The shape tier and its refusal (`OneHotEncoder`).
4. `tree_ensemble`: artifact extraction, the Rust kernel, the
   `confit.udfs.TreeEnsemble` entry, both engine bindings, the swap-the-entry
   gate.
5. Fold the legacy clone loop into `SklearnTransform` and delete the
   special-cased fit path.

## Rejected

- Expression fragments as the family surface (loses joins/CTEs/windows;
  second SQL grammar).
- A Python reference twin per kind (second implementation, smears extraction
  bugs into execution bugs).
- Serializing estimators into STRUCT/LIST θ.
- Fitted state on `self`.
- A process-global or session-global artifact store, and an `artifacts=`
  channel on `DuckDBInferFn`: the entry's `instances` is already the store.
- One function name per fit step or per group.
- `LinearArtifact` and friends: SQL expresses them.
