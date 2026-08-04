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
import pytest
from confit import DuckDBInferFn
from test_duckdb_interpreter import _row_model, static


class Scale:
    """A fitted-transformer stand-in: tf(id, x) = x * factor[id] (width 1)."""

    name = "tf0"
    takes = ("f64",)
    returns = ("f64",)
    instances = {0: 10.0, 1: 100.0}

    def __call__(self, iid, x):
        if iid is None or x is None:
            return None
        return (x * self.instances[iid],)


class Embed2:
    """A width-2 transformer stand-in: emb(id, x) = (x + id, x - id)."""

    name = "emb"
    takes = ("f64",)
    returns = ("f64", "f64")
    instances = {0: 0.0, 1: 1.0}

    def __call__(self, iid, x):
        if iid is None or x is None:
            return None
        d = self.instances[iid]
        return (x + d, x - d)


class NamedEmbed2:
    """Embed2 with declared output field names (TASK-63) + a call counter."""

    name = "emb"
    takes = ("f64",)
    returns = ("f64", "f64")
    return_names = ("a", "b")
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
    takes = ("str",)
    returns = ("str",)

    def __call__(self, s):
        return (None if s is None else s.upper(),)


def _scalar_form(obj):
    """The DuckDB registration of the same object: unwrap width-1 tuples;
    width-k becomes a dict (STRUCT) when field names are declared, else a
    list (DOUBLE[])."""

    rn = getattr(obj, "return_names", None) if len(obj.returns) > 1 else None

    def unwrap(out):
        if out is None:
            return None
        if len(out) == 1:
            return out[0]
        return dict(zip(rn, out, strict=True)) if rn else list(out)

    n = len(obj.takes) + (1 if hasattr(obj, "instances") else 0)
    args = ", ".join(f"a{i}" for i in range(n))
    ns = {"call": obj, "unwrap": unwrap}
    exec(f"def w({args}): return unwrap(call({args}))", ns)  # noqa: S102
    return ns["w"]


_DUCK_T = {"i1": "BOOLEAN", "i64": "BIGINT", "f64": "DOUBLE", "str": "VARCHAR"}


def udf_check(sql, row_schema, row_rows, statics, udfs, output=None):
    """Differential: engine with udfs= vs DuckDB with the same objects
    registered. Returns the engine rows (dicts) for extra assertions."""
    model = _row_model(row_schema)
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": model},
        static_tables=statics,
        udfs=udfs,
        output=output,
    )
    inputs = [model(**r) for r in row_rows]
    res = fn.infer({"__THIS__": inputs})
    got = res if output == "dict" else [r.model_dump() for r in res]

    con = duckdb.connect()
    for u in udfs:
        params = [_DUCK_T[t] for t in u.takes]
        if hasattr(u, "instances"):
            params = ["BIGINT", *params]
        rn = getattr(u, "return_names", None)
        if len(u.returns) == 1:
            ret = _DUCK_T[u.returns[0]]
        elif rn:
            ret = duckdb.struct_type(
                {n: _DUCK_T[t] for n, t in zip(rn, u.returns, strict=True)}
            )
        else:
            ret = "DOUBLE[]"
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


def test_field_access_shares_one_call_per_row():
    # TASK-63: two field reads of one width-2 call — ONE callable
    # invocation per row on the engine AND on DuckDB (its CSE), counted.
    u = NamedEmbed2()
    got = udf_check(
        "SELECT (emb(p.est, t.x)).a AS ea, (emb(p.est, t.x)).b AS eb, t.g AS g "
        "FROM __THIS__ AS t LEFT JOIN p0 AS p ON ((t.g IS NOT DISTINCT FROM p.g))",
        {"g": "str?", "x": "float?"},
        [{"g": None, "x": 5.0}, {"g": "fr", "x": 5.0}, {"g": "de", "x": 2.0}],
        {"p0": PARAMS},
        [u],
    )
    by_g = {r["g"]: (r["ea"], r["eb"]) for r in got}
    assert by_g[None] == (6.0, 4.0)  # NULL g joins the NULL-key row: id 1
    assert by_g["de"] == (2.0, 2.0)  # id 0, d = 0
    assert by_g["fr"] == (None, None)  # unseen group: NULL id -> NULL fields
    # 3 rows served by the engine + 3 by DuckDB — one call per ROW each.
    assert u.calls == 6, f"expected 6 calls (3 rows x 2 paths), got {u.calls}"


def test_wide_udf_dict_output():
    got = udf_check(
        "SELECT emb(0, x) AS e FROM __THIS__",
        {"x": "float?"},
        [{"x": 2.0}],
        {},
        [Embed2()],
        output="dict",
    )
    assert got == [{"e": [2.0, 2.0]}]


def test_infer_arrow_wide_and_scalar():
    import pyarrow as pa

    model = _row_model({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT emb(0, x) AS e, x + 1.0 AS y FROM __THIS__",
        row_tables={"__THIS__": model},
        static_tables={},
        udfs=[Embed2()],
    )
    batch = pa.table({"x": pa.array([2.0, None], type=pa.float64())})
    out = fn.infer_arrow(batch)
    assert out.column("e").to_pylist() == [[2.0, 2.0], None]
    assert out.column("y").to_pylist() == [3.0, None]


def test_both_backends_agree(monkeypatch):
    def run():
        model = _row_model({"x": "float?"})
        fn = DuckDBInferFn(
            "SELECT tf0(0, x) AS z, emb(0, x) AS e FROM __THIS__",
            row_tables={"__THIS__": model},
            static_tables={},
            udfs=[Scale(), Embed2()],
        )
        rows = [model(x=1.5), model(x=None)]
        return fn.backend, [r.model_dump() for r in fn.infer_rows(rows)]

    b_default, got_default = run()
    monkeypatch.setenv("SPECIALIZER_FORCE_INTERP", "1")
    b_interp, got_interp = run()
    assert b_interp == "interpreter"
    assert got_default == got_interp
    assert b_default == "cranelift"  # udf programs must not fall back


def test_udf_exception_traps():
    class Boom:
        name = "boom"
        takes = ("f64",)
        returns = ("f64",)

        def __call__(self, x):
            raise RuntimeError("kapow")

    model = _row_model({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT boom(x) AS z FROM __THIS__",
        row_tables={"__THIS__": model},
        static_tables={},
        udfs=[Boom()],
    )
    with pytest.raises(Exception, match="boom.*kapow"):
        fn.infer_rows([model(x=1.0)])


def test_wrong_return_shape_traps():
    class Liar:
        name = "liar"
        takes = ("f64",)
        returns = ("f64", "f64")

        def __call__(self, x):
            return (x,)  # declared 2, returns 1

    model = _row_model({"x": "float?"})
    fn = DuckDBInferFn(
        "SELECT liar(x) AS z FROM __THIS__",
        row_tables={"__THIS__": model},
        static_tables={},
        udfs=[Liar()],
    )
    with pytest.raises(Exception, match="liar.*returned 1 values, declared 2"):
        fn.infer_rows([model(x=1.0)])


def test_unknown_function_still_refuses():
    model = _row_model({"x": "float?"})
    with pytest.raises(Exception, match="mystery"):
        DuckDBInferFn(
            "SELECT mystery(x) AS z FROM __THIS__",
            row_tables={"__THIS__": model},
            static_tables={},
            udfs=[Scale()],
        )


def test_wide_mid_expression_refuses():
    model = _row_model({"x": "float?"})
    with pytest.raises(Exception, match="width-2.*bare SELECT items"):
        DuckDBInferFn(
            "SELECT emb(0, x) + 1.0 AS z FROM __THIS__",
            row_tables={"__THIS__": model},
            static_tables={},
            udfs=[Embed2()],
        )


def test_declaration_validation():
    class BadTy:
        name = "bad"
        takes = ("f32",)
        returns = ("f64",)

        def __call__(self, x):
            return (x,)

    model = _row_model({"x": "float?"})
    with pytest.raises(Exception, match="f32.*i1/i64/f64/str"):
        DuckDBInferFn(
            "SELECT bad(x) AS z FROM __THIS__",
            row_tables={"__THIS__": model},
            static_tables={},
            udfs=[BadTy()],
        )
    with pytest.raises(Exception, match="duplicate udf name"):
        DuckDBInferFn(
            "SELECT tf0(0, x) AS z FROM __THIS__",
            row_tables={"__THIS__": model},
            static_tables={},
            udfs=[Scale(), Scale()],
        )


def test_arity_mismatch_refuses():
    model = _row_model({"x": "float?"})
    with pytest.raises(Exception, match="tf0"):
        DuckDBInferFn(
            "SELECT tf0(0, x, x) AS z FROM __THIS__",
            row_tables={"__THIS__": model},
            static_tables={},
            udfs=[Scale()],
        )
