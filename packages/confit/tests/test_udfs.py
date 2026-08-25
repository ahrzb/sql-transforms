"""UDF externs (DRAFT-22 step 2): the `udfs=` surface of DuckDBInferFn.

The contract, restated with one parameter: serve bit-for-bit identical to
DuckDB *with the same udfs registered* (`con.create_function`), or refuse.
The udf objects here are duck-typed to the sql_transform protocol (name /
takes / returns / optional `instances` marking the implicit leading i64
id / scalar ``__call__`` returning a tuple or None) — deliberately defined
in-file, not imported: this package's contract is the protocol, not the
class.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from sql_transform._udf import UDF
from test_duckdb_interpreter import _row_schema, static


@pytest.mark.parametrize("name", ["least", "upper", "ROUND", "Coalesce", "abs"])
def test_a_udf_may_not_take_a_builtin_name(name):
    """`function()` matches the builtin catalogue before it ever consults the
    declared UDFs, so a UDF named after a builtin would be silently shadowed —
    while DuckDB, which lets a registered function shadow its own builtin,
    binds the UDF. Two engines, one SQL, different answers: refuse instead.

    Matching DuckDB by letting the UDF win is not the fix — DuckDB
    overload-resolves by arity and we do not, so `least(a, b, c)` against a
    two-argument UDF would fall back to its builtin and diverge the other way.
    """

    class Collide:
        takes = pa.schema([("x", pa.float64())])
        returns = pa.float64()

        def __call__(self, x):
            return (111.0,)

    u = Collide()
    u.name = name
    schema = _row_schema({"x": "float"})
    with pytest.raises(Exception, match=f"'{name}'.*builtin"):
        DuckDBInferFn(
            f"SELECT {name}(x) AS p FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[u],
        )


@pytest.mark.parametrize(
    ("lane", "vals"),
    [
        (pa.int64(), (2**53 + 1, 2**53 + 3)),  # two doubles cannot tell these apart
        (pa.int64(), (2**60 + 1, 2**60 + 2)),
        (pa.bool_(), (False, True)),
        (pa.string(), ("s0", "s1")),
        (pa.float64(), (1.5, -0.0)),
    ],
)
def test_an_unnamed_width_k_return_honours_its_lane_type(lane, vals):
    """`returns=pa.list_(t, k)` registered on DuckDB as `DOUBLE[]` whatever `t`
    said, so an int64 lane came back rounded through a double while the engine
    served the integer."""

    class Pack(UDF):
        name = "pk"
        takes = pa.schema([("x", pa.float64())])
        returns = pa.list_(lane, 2)

        def __call__(self, x):
            return vals

    got = udf_check(
        "SELECT pk(x) AS p FROM __THIS__",
        {"x": "float"},
        [{"x": 1.0}],
        {},
        [Pack()],
    )
    assert got == [{"p": list(vals)}], got


def test_a_width_one_list_return_refuses():
    """`pa.list_(t, 1)` is the one shape where the lane COUNT and the arrow
    SHAPE disagree: the engine serves width-1 unnamed as a plain scalar
    everywhere, and DuckDB was told a scalar too, so a value declared as a
    list crossed as its element. There is no 1-element list boundary to serve
    it on and no reason to build one — `pa.float64()` is what it means."""

    class One(UDF):
        name = "one"
        takes = pa.schema([("x", pa.float64())])
        returns = pa.list_(pa.float64(), 1)

        def __call__(self, x):
            return (x,)

    schema = _row_schema({"x": "float"})
    with pytest.raises(Exception, match="width-1 list.*scalar"):
        DuckDBInferFn(
            "SELECT one(x) AS p FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[One()],
        )


def test_the_schema_is_arrow():
    """`takes` is a `pa.Schema` and `returns` is the SQL return TYPE — one
    declaration each, names included. The three return shapes are three arrow
    types rather than a width plus an optional names tuple."""
    schema = _row_schema({"id": "int", "x": "float"})

    class Halve:
        name = "halve"
        takes = pa.schema([("x", pa.float64())])
        returns = pa.float64()

        def __call__(self, x):
            return (None if x is None else x / 2.0,)

    fn = DuckDBInferFn(
        "SELECT halve(x) AS p FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[Halve()],
    )
    assert fn.backend == "cranelift"
    got = [r["p"] for r in fn.infer_rows([{"id": 0, "x": 5.0}])]
    assert got == [2.5]


class Scale:
    """A fitted-transformer stand-in: tf(id, x) = x * factor[id] (width 1)."""

    name = "tf0"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.float64()
    instances = {0: 10.0, 1: 100.0}

    def __call__(self, iid, x):
        if iid is None or x is None:
            return None
        return (x * self.instances[iid],)


class Embed2:
    """A width-2 transformer stand-in: emb(id, x) = (x + id, x - id)."""

    name = "emb"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.list_(pa.float64(), 2)
    instances = {0: 0.0, 1: 1.0}

    def __call__(self, iid, x):
        if iid is None or x is None:
            return None
        d = self.instances[iid]
        return (x + d, x - d)


class NamedEmbed2:
    """Embed2 with declared output field names + a call counter."""

    name = "emb"
    takes = pa.schema([("x", pa.float64())])
    returns = pa.struct([("a", pa.float64()), ("b", pa.float64())])
    instances = {0: 0.0, 1: 1.0}

    def __init__(self):
        self.calls = 0

    def __call__(self, iid, x):
        self.calls += 1
        if iid is None or x is None:
            return None
        d = self.instances[iid]
        return (x + d, x - d)


class Shout:
    """A plain author UDF (no instances -> no implicit id): str -> str."""

    name = "shout"
    takes = pa.schema([("s", pa.string())])
    returns = pa.string()

    def __call__(self, s):
        return (None if s is None else s.upper(),)


def _lanes(obj):
    """`(field names, lane types)` off a declared `returns` — the same reading
    the engine does, spelled here so this package's tests exercise the arrow
    protocol rather than importing sql_transform's implementation of it."""
    r = obj.returns
    if pa.types.is_struct(r):
        f = [r.field(i) for i in range(r.num_fields)]
        return tuple(x.name for x in f), tuple(x.type for x in f)
    if pa.types.is_fixed_size_list(r):
        return (), (r.value_type,) * r.list_size
    return (), (r,)


def _scalar_form(obj):
    """The DuckDB registration of the same object: unwrap width-1 tuples;
    width-k becomes a dict (STRUCT) when field names are declared, else a
    list (DOUBLE[])."""

    names, types = _lanes(obj)
    rn = names if len(types) > 1 else None
    listy = pa.types.is_fixed_size_list(obj.returns)

    def unwrap(out):
        if out is None:
            return None
        if rn:
            return dict(zip(rn, out, strict=True))
        return list(out) if listy else out[0]

    n = len(obj.takes) + (1 if hasattr(obj, "instances") else 0)
    args = ", ".join(f"a{i}" for i in range(n))
    ns = {"call": obj, "unwrap": unwrap}
    exec(f"def w({args}): return unwrap(call({args}))", ns)  # noqa: S102
    return ns["w"]


_DUCK_T = {
    pa.bool_(): "BOOLEAN",
    pa.int64(): "BIGINT",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
}


def udf_check(sql, row_schema, row_rows, statics, udfs, after_engine=None):
    """Differential: engine with udfs= vs DuckDB with the same objects
    registered. Returns the engine rows (dicts) for extra assertions.
    ``after_engine`` runs between the two legs (call-count attribution)."""
    schema = _row_schema(row_schema)
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": schema},
        static_tables=statics,
        udfs=udfs,
    )
    got = fn.infer_rows(row_rows)
    if after_engine is not None:
        after_engine()

    con = duckdb.connect()
    for u in udfs:
        params = [_DUCK_T[t] for t in u.takes.types]
        if hasattr(u, "instances"):
            params = ["BIGINT", *params]
        rn, rt = _lanes(u)
        if rn:
            ret = duckdb.struct_type(
                {n: _DUCK_T[t] for n, t in zip(rn, rt, strict=True)}
            )
        elif pa.types.is_fixed_size_list(u.returns):
            # A LIST at every width including 1, with the DECLARED lane type
            # — `DOUBLE[]` regardless would round an int64 lane through a
            # double on this side only.
            ret = f"{_DUCK_T[rt[0]]}[]"
        else:
            ret = _DUCK_T[rt[0]]
        con.create_function(
            u.name, _scalar_form(u), params, ret, null_handling="special"
        )
    for name, table in statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
    con.register("__arrow_this", static(row_schema, row_rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    want = con.execute(sql).to_arrow_table().to_pylist()
    con.close()

    key = lambda r: sorted((k, repr(v)) for k, v in r.items())  # noqa: E731
    assert sorted(map(key, got)) == sorted(map(key, want)), f"{got} != {want}"
    return got


PARAMS = static(
    {"g": "str?", "est": "int?"},
    [{"g": "de", "est": 0}, {"g": None, "est": 1}],
)


def test_marginalizer_shape_differential():
    # The DRAFT-22 serving_sql shape end to end: INDF params join, the call
    # mid-expression, unseen group -> NULL id -> NULL.
    udf_check(
        "SELECT (tf0(p.est, t.age) + 1.0) AS z, t.g AS g FROM __THIS__ AS t "
        "LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        {"g": "str?", "age": "float?"},
        [
            {"g": "de", "age": 1.5},
            {"g": None, "age": 2.0},
            {"g": "fr", "age": 3.0},
            {"g": "de", "age": None},
        ],
        {"p0": PARAMS},
        [Scale()],
    )


def test_keyless_literal_id():
    # DRAFT-23 addendum: a keyless (global) transformer inlines id 0 — no
    # params table, no join.
    got = udf_check(
        "SELECT tf0(0, x) AS z FROM __THIS__",
        {"x": "float?"},
        [{"x": 1.0}, {"x": None}],
        {},
        [Scale()],
    )
    assert got[0]["z"] == 10.0 and got[1]["z"] is None


def test_author_udf_no_instances():
    udf_check(
        "SELECT shout(name) AS n FROM __THIS__",
        {"name": "str?"},
        [{"name": "ab"}, {"name": None}],
        {},
        [Shout()],
    )


def test_wide_udf_bare_item_is_list_field():
    got = udf_check(
        "SELECT emb(p.est, t.x) AS e, t.g AS g FROM __THIS__ AS t "
        "LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        {"g": "str?", "x": "float?"},
        [{"g": None, "x": 5.0}, {"g": "fr", "x": 5.0}, {"g": "de", "x": None}],
        {"p0": PARAMS},
        [Embed2()],
    )
    by_g = {r["g"]: r["e"] for r in got}
    assert by_g[None] == [6.0, 4.0]  # NULL g joins the NULL-key params row
    assert by_g["fr"] is None  # unseen group: NULL id -> NULL list
    assert by_g["de"] is None  # NULL feature: the callable's convention


def test_case_colliding_return_names_refuse_at_build():
    # Field binding is ASCII-case-insensitive on both sides — declared
    # names that collide under it would bind silently wrong.
    class Bad2:
        name = "bad2"
        takes = pa.schema([("x", pa.float64())])
        returns = pa.struct([("x", pa.float64()), ("X", pa.float64())])

        def __call__(self, x):
            return (x, x)

    schema = _row_schema({"x": "float?"})
    with pytest.raises(Exception, match="collide case-insensitively"):
        DuckDBInferFn(
            "SELECT (bad2(x)).x AS a FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[Bad2()],
        )


def test_field_access_shares_one_call_per_row():
    # Two field reads of one width-2 call — ONE callable invocation per row
    # on the engine, counted. (DuckDB's leg makes two; see below.)
    u = NamedEmbed2()
    engine_calls = []
    got = udf_check(
        "SELECT (emb(p.est, t.x)).a AS ea, (emb(p.est, t.x)).b AS eb, t.g AS g "
        "FROM __THIS__ AS t LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        {"g": "str?", "x": "float?"},
        [{"g": None, "x": 5.0}, {"g": "fr", "x": 5.0}, {"g": "de", "x": 2.0}],
        {"p0": PARAMS},
        [u],
        after_engine=lambda: engine_calls.append(u.calls),
    )
    by_g = {r["g"]: (r["ea"], r["eb"]) for r in got}
    assert by_g[None] == (6.0, 4.0)  # NULL g joins the NULL-key row: id 1
    assert by_g["de"] == (2.0, 2.0)  # id 0, d = 0
    assert by_g["fr"] == (None, None)  # unseen group: NULL id -> NULL fields
    # One call per ROW on OUR path — the single-evaluation guarantee, and the
    # assertion that matters here.
    assert engine_calls == [3], f"engine leg: {engine_calls} calls for 3 rows"
    # DuckDB's leg makes TWO per row, one per field read. Sharing them is its
    # `common_subexpressions` pass, and the oracle runs with the optimizer off
    # (see conftest), so there is nothing to share. Still pinned rather than
    # dropped, because the number is what attributes calls to a leg: if our
    # side ever leaked a call into DuckDB's, this moves.
    assert u.calls == 3 + 6, f"DuckDB leg: {u.calls - 3} calls for 3 rows"


def test_wide_udf_dict_output():
    got = udf_check(
        "SELECT emb(0, x) AS e FROM __THIS__",
        {"x": "float?"},
        [{"x": 2.0}],
        {},
        [Embed2()],
    )
    assert got == [{"e": [2.0, 2.0]}]


def test_infer_arrow_wide_and_scalar():
    import pyarrow as pa

    schema = _row_schema({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT emb(0, x) AS e, x + 1.0 AS y FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[Embed2()],
    )
    batch = pa.table({"x": pa.array([2.0, None], type=pa.float64())})
    out = fn.infer_arrow(batch)
    assert out.column("e").to_pylist() == [[2.0, 2.0], None]
    assert out.column("y").to_pylist() == [3.0, None]


def test_infer_arrow_named_struct():
    # Slice-5 review round: the named branch (pa.struct_ keyed by the
    # declared field names; whole-NULL row stays a NULL struct) was
    # unpinned on the columnar boundary.
    import pyarrow as pa

    schema = _row_schema({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT emb(0, x) AS e FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[NamedEmbed2()],
    )
    batch = pa.table({"x": pa.array([2.0, None], type=pa.float64())})
    out = fn.infer_arrow(batch)
    assert out.schema.field("e").type == pa.struct(
        [pa.field("a", pa.float64()), pa.field("b", pa.float64())]
    )
    assert out.column("e").to_pylist() == [{"a": 2.0, "b": 2.0}, None]


def test_both_backends_agree(monkeypatch):
    def run():
        schema = _row_schema({"x": "float?"})
        fn = DuckDBInferFn(
            "SELECT tf0(0, x) AS z, emb(0, x) AS e FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[Scale(), Embed2()],
        )
        rows = [{"x": 1.5}, {"x": None}]
        return fn.backend, fn.infer_rows(rows)

    b_default, got_default = run()
    monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    b_interp, got_interp = run()
    assert b_interp == "interpreter"
    assert got_default == got_interp
    assert b_default == "cranelift"  # udf programs must not fall back


def test_udf_exception_traps():
    class Boom:
        name = "boom"
        takes = pa.schema([("x", pa.float64())])
        returns = pa.float64()

        def __call__(self, x):
            raise RuntimeError("kapow")

    schema = _row_schema({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT boom(x) AS z FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[Boom()],
    )
    with pytest.raises(Exception, match="boom.*kapow"):
        fn.infer_rows([{"x": 1.0}])


def test_wrong_return_shape_traps():
    class Liar:
        name = "liar"
        takes = pa.schema([("x", pa.float64())])
        returns = pa.list_(pa.float64(), 2)

        def __call__(self, x):
            return (x,)  # declared 2, returns 1

    schema = _row_schema({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT liar(x) AS z FROM __THIS__",
        row_tables={"__THIS__": schema},
        static_tables={},
        udfs=[Liar()],
    )
    with pytest.raises(Exception, match="liar.*returned 1 values, declared 2"):
        fn.infer_rows([{"x": 1.0}])


def test_unknown_function_still_refuses():
    schema = _row_schema({"x": "float?"})
    with pytest.raises(Exception, match="mystery"):
        DuckDBInferFn(
            "SELECT mystery(x) AS z FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[Scale()],
        )


def test_wide_mid_expression_refuses():
    schema = _row_schema({"x": "float?"})
    with pytest.raises(Exception, match="width-2.*bare SELECT items"):
        DuckDBInferFn(
            "SELECT emb(0, x) + 1.0 AS z FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[Embed2()],
        )


def test_declaration_validation():
    """The vocabulary is exactly bool/int64/double/string. A narrower arrow
    type refuses rather than widening silently, which would make the declared
    schema a lie about what is served."""

    class BadTy:
        name = "bad"
        takes = pa.schema([("x", pa.float32())])
        returns = pa.float64()

        def __call__(self, x):
            return (x,)

    class BadRet(BadTy):
        takes = pa.schema([("x", pa.float64())])
        returns = pa.int32()

    class UnsizedList(BadTy):
        takes = pa.schema([("x", pa.float64())])
        returns = pa.list_(pa.float64())

    schema = _row_schema({"x": "float?"})
    for cls, needle in (
        (BadTy, "float.*bool/int64/double/string"),
        (BadRet, "int32.*bool/int64/double/string"),
        (UnsizedList, "must declare its width"),
    ):
        with pytest.raises(Exception, match=needle):
            DuckDBInferFn(
                "SELECT bad(x) AS z FROM __THIS__",
                row_tables={"__THIS__": schema},
                static_tables={},
                udfs=[cls()],
            )
    with pytest.raises(Exception, match="duplicate udf name"):
        DuckDBInferFn(
            "SELECT tf0(0, x) AS z FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[Scale(), Scale()],
        )


def test_arity_mismatch_refuses():
    schema = _row_schema({"x": "float?"})
    with pytest.raises(Exception, match="tf0"):
        DuckDBInferFn(
            "SELECT tf0(0, x, x) AS z FROM __THIS__",
            row_tables={"__THIS__": schema},
            static_tables={},
            udfs=[Scale()],
        )
