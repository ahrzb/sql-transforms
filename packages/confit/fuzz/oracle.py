"""One Case in, one Verdict out.

The contract under test: confit either matches DuckDB bit-for-bit — with the
same UDFs registered — or refuses at build with a named ValueError. Anything
else is a finding. Extra legs beyond the three-way comparison: infer_rows vs
infer_arrow, hostile Arrow input (sliced / chunked / empty), single-row vs
batch concatenation, rebuild determinism, and sklearn as a second ground
truth on tree cases.

DuckDB setup mirrors tests/test_udfs.py `udf_check` (native tables, repr-keyed
multiset compare) — duplicated here on purpose: fuzz/ must not import from
tests/, and the duplication is itself a check that the registration recipe is
writable from the documented protocol alone.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from dataclasses import field as dfield

import duckdb
import pyarrow as pa

from . import gen as G

KINDS = (
    "DIVERGE_VALUE",
    "DIVERGE_BUILD",
    "DIVERGE_TRAP",
    "BUILD_EXC",
    "AGREE",
    "AGREE_TRAP",
    "REFUSED",
    "SKIP",
)

_ARROW = {
    # semantic names, for UDF signatures and tree features
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
    # storage names, for table columns — the width IS the declaration
    # ("bool" is spelled the same on both axes)
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "double": pa.float64(),
    "string": pa.string(),
    # out of vocabulary on purpose: unreferenced, these must not block a build
    "float32": pa.float32(),
    "timestamp": pa.timestamp("us"),
}
_DUCK_T = {
    pa.bool_(): "BOOLEAN",
    pa.int64(): "BIGINT",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
}

# Campaign-report filter for the one feature not yet shipped: decimals
# (lattice-spec phase 5, Dec(p,s) arithmetic). One unshipped feature =
# hundreds of random spellings per run; delete the tag in the feature's own
# PR or it hides regressions. (int-widths deleted with TASK-79/phase 2.)
#
# This tag has NO strict-xfail twin, contrary to what this comment claimed
# until 2026-08-15. The bare-literal 1-ulp class is documented instead, in
# docs/known-limitations.md — and TASK-95 (doc-twin totality) is still To
# Do, so nothing rings when phase 5 lands. test_decimals.py's only pin is
# TASK-91, which is phase 1 (decimal STATIC ingest), a different member.
# Until one of those closes, this tag is a report filter and nothing more.


@dataclass
class Verdict:
    kind: str
    klass: str = ""  # dedup key within the kind
    detail: str = ""
    tags: list[str] = dfield(default_factory=list)

    def to_json(self):
        return {
            "kind": self.kind,
            "klass": self.klass,
            "detail": self.detail[:500],
            "tags": self.tags,
        }


# ------------------------------------------------------------- materialize


def _arrow_field(name: str, spec) -> pa.Field:
    if isinstance(spec, G.Struct):
        return pa.field(
            name,
            pa.struct([_arrow_field(n, s) for n, s in spec.fields]),
            nullable=spec.nullable,
        )
    return pa.field(name, _ARROW[spec.rstrip("?")], nullable=spec.endswith("?"))


def _arrow_schema(schema: dict) -> pa.Schema:
    return pa.schema([_arrow_field(n, s) for n, s in schema.items()])


def _arrow_table(schema: dict, rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=_arrow_schema(schema))


def _mix(vals, tys):
    """The deterministic pure body every generated UDF shares: fold each
    argument into a float accumulator, None-propagating per SQL."""
    acc = 0.0
    for v, t in zip(vals, tys, strict=True):
        if v is None:
            return None
        if t == "str":
            acc += float(len(v)) + (float(ord(v[0])) if v else 0.0)
        elif t == "bool":
            acc += 1.0 if v else -1.0
        else:
            acc += float(v) if abs(float(v)) < 1e12 else math.copysign(1e12, float(v))
        acc = acc * 0.5 + 3.0
    return acc


def _lane_val(acc: float, ty: str, salt: int):
    if ty == "float":
        return acc + salt
    if ty == "int":
        return int(acc) % 1000 + salt
    if ty == "bool":
        return (int(acc) + salt) % 2 == 0
    return f"v{(int(acc) + salt) % 97}"


def make_udf(spec: G.UdfSpec):
    lanes = (
        [(None, spec.ret[1])]
        if spec.ret[0] == "scalar"
        else spec.ret[1]
        if spec.ret[0] == "struct"
        else [(None, spec.ret[1])] * spec.ret[2]
    )
    takes_tys = [t for _, t in spec.takes]

    class U:
        name = spec.name
        takes = pa.schema([(n, _ARROW[t]) for n, t in spec.takes])
        returns = (
            _ARROW[spec.ret[1]]
            if spec.ret[0] == "scalar"
            else pa.struct([(n, _ARROW[t]) for n, t in spec.ret[1]])
            if spec.ret[0] == "struct"
            else pa.list_(_ARROW[spec.ret[1]], spec.ret[2])
        )
        if spec.instances:
            instances = dict.fromkeys(range(spec.instances))

        def __call__(self, *args):
            if spec.instances:
                iid, *args = args
                if iid is None or iid not in range(spec.instances):
                    return None
                bias = float(iid)
            else:
                bias = 0.0
            acc = _mix(args, takes_tys)
            if acc is None:
                return None
            return tuple(_lane_val(acc + bias, t, i) for i, (_, t) in enumerate(lanes))

    return U()


def make_tree(spec: G.TreeSpec, seed: int):
    """A fitted sklearn ensemble wrapped as a TreeBasedTransform — the same
    object confit routes to the native kernel and DuckDB calls as a python
    UDF, with `estimators[iid].predict` as the second ground truth."""
    import numpy as np
    from sql_transform import TreeBasedTransform

    rng = np.random.RandomState(seed % (2**31))
    n = 40
    x = np.round(rng.uniform(-4, 4, size=(n, spec.n_features)), 2)
    # Integer features draw from a boundary pool (the 2**53 grid lesson) —
    # randint cannot span the full i64 range portably.
    pool = np.array(
        [0, 1, -1, 7, -13, 100, 2**31 - 1, 2**53 - 1, 2**53, 2**53 + 1, -(2**53) - 1],
        dtype=np.float64,
    )
    for i in spec.int_features:
        x[:, i] = rng.choice(pool, size=n)
    y = rng.uniform(-10, 10, size=n)

    def fit(i: int):
        from sklearn.ensemble import (
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.tree import DecisionTreeRegressor

        cls = {
            "dtr": DecisionTreeRegressor,
            "rf": RandomForestRegressor,
            "gbr": GradientBoostingRegressor,
        }[spec.kind]
        kw = {"max_depth": spec.depth, "random_state": i}
        if spec.kind != "dtr":
            kw["n_estimators"] = 3
        est = cls(**kw)
        est.fit(x, y + i)  # each instance is a genuinely different model
        return est

    ests = {i: fit(i) for i in range(spec.instances)}
    takes = pa.schema(
        [
            (f"f{i}", pa.int64() if i in spec.int_features else pa.float64())
            for i in range(spec.n_features)
        ]
    )
    return TreeBasedTransform(name="trees", instances=ests, takes=takes), ests


def _scalar_form(obj):
    """DuckDB registration of a protocol object (see tests/test_udfs.py)."""
    r = obj.returns
    if pa.types.is_struct(r):
        names = tuple(r.field(i).name for i in range(r.num_fields))
        listy = False
    elif pa.types.is_fixed_size_list(r):
        names, listy = (), True
    else:
        names, listy = (), False

    def unwrap(out):
        if out is None:
            return None
        if names:
            return dict(zip(names, out, strict=True))
        return list(out) if listy else out[0]

    # duckdb reads the python signature for arity, so *args won't do
    n = len(obj.takes) + (1 if hasattr(obj, "instances") else 0)
    argl = ", ".join(f"a{i}" for i in range(n))
    ns = {"call": obj, "unwrap": unwrap}
    exec(f"def w({argl}): return unwrap(call({argl}))", ns)  # noqa: S102
    return ns["w"]


def _duck_ret(obj):
    r = obj.returns
    if pa.types.is_struct(r):
        return duckdb.struct_type(
            {r.field(i).name: _DUCK_T[r.field(i).type] for i in range(r.num_fields)}
        )
    if pa.types.is_fixed_size_list(r):
        return f"{_DUCK_T[r.value_type]}[]"
    return _DUCK_T[r]


# ------------------------------------------------------------------ oracle


def _build(sql, schema, statics, udf_objs, shape, force_interp):
    from confit import DuckDBInferFn

    prev = os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
    try:
        if force_interp:
            os.environ["SPECIALIZER_FORCE_INTERP"] = "1"
        return DuckDBInferFn(
            sql,
            row_tables={"__THIS__": schema},
            static_tables=statics,
            udfs=udf_objs or None,
            shape=shape,
        )
    finally:
        if force_interp:
            os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
        if prev is not None:
            os.environ["SPECIALIZER_FORCE_INTERP"] = prev


_DUCK_BUILD_ERRS = (
    "ParserException",
    "BinderException",
    "CatalogException",
    "NotImplementedException",
    "SyntaxException",
)


def _duck_con(case: G.Case, udf_objs):
    con = duckdb.connect()
    for u in udf_objs:
        params = [_DUCK_T[t] for t in u.takes.types]
        if hasattr(u, "instances"):
            params = ["BIGINT", *params]
        con.create_function(
            u.name, _scalar_form(u), params, _duck_ret(u), null_handling="special"
        )
    for name, (sch, rows) in case.statics.items():
        con.register(f"__arrow_{name}", _arrow_table(sch, rows))
        # our own generated table names, not user input
        ddl = f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"'  # noqa: S608
        con.execute(ddl)
    con.register("__arrow_this", _arrow_table(case.row_schema, case.rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    return con


def _duck_run(sql, case: G.Case, udf_objs):
    con = _duck_con(case, udf_objs)
    try:
        try:
            return con.execute(sql).to_arrow_table(), None, None
        except Exception as e:  # noqa: BLE001 — classify, don't die
            phase = "build" if type(e).__name__ in _DUCK_BUILD_ERRS else "run"
            return None, phase, f"{type(e).__name__}: {e}"
    finally:
        con.close()


def _key(rows: list[dict]):
    return sorted(sorted((k, repr(v)) for k, v in r.items()) for r in rows)


def _schema_delta(duck: pa.Schema, ours: pa.Schema):
    """None if schemas agree; ("known", tag) for open-ticket width classes;
    ("diff", detail) otherwise."""
    if duck.names != ours.names:
        return ("diff", f"names {duck.names} != {ours.names}")
    known = []
    for d, o in zip(duck, ours, strict=True):
        r = _type_delta(d.type, o.type)
        if r == "diff":
            return ("diff", f"{d.name}: duck {d.type} != ours {o.type}")
        if r is not None:
            known.append(r)
    return ("known", known[0]) if known else None


def _type_delta(duck: pa.DataType, ours: pa.DataType) -> str | None:
    """None = equal; a tag = an unshipped feature's width class (recursing
    into structs — a decimal lane inside struct_pack still tags); "diff"."""
    if duck == ours:
        return None
    # One arm per unshipped feature; delete when it ships (see note above).
    if pa.types.is_decimal(duck) and ours == pa.float64():
        return "decimals"
    if (
        pa.types.is_struct(duck)
        and pa.types.is_struct(ours)
        and duck.num_fields == ours.num_fields
    ):
        tag = None
        for i in range(duck.num_fields):
            df, of = duck.field(i), ours.field(i)
            if df.name != of.name:
                return "diff"
            r = _type_delta(df.type, of.type)
            if r == "diff":
                return "diff"
            tag = tag or r
        return tag
    return "diff"


def _norm(table: pa.Table, to: pa.Schema | None = None) -> list[dict]:
    if to is not None:
        table = table.cast(to)
    return table.to_pylist()


def run_case(case: G.Case) -> Verdict:
    sql = G.render(case.query)
    tags = list(case.tags)
    schema = _arrow_schema(case.row_schema)
    statics = {n: _arrow_table(sch, rows) for n, (sch, rows) in case.statics.items()}
    udf_objs = [make_udf(u) for u in case.udfs]
    ests = {}
    if case.tree is not None:
        tree_obj, ests = make_tree(case.tree, case.seed)
        udf_objs.append(tree_obj)

    # case.output (gen.py still generates None/"dict"/"model") exercised the
    # deleted output= kwarg; dict-out is now the only mode, so it is not
    # forwarded to _build.
    # --- build both backends -------------------------------------------
    def build(force):
        try:
            return (
                _build(sql, schema, statics, udf_objs, case.shape, force),
                None,
                None,
            )
        except ValueError as e:
            return None, "refused", str(e)
        except Exception as e:  # noqa: BLE001
            return None, "exc", f"{type(e).__name__}: {e}"

    fn_cl, cl_err_kind, cl_err = build(force=False)
    fn_in, in_err_kind, in_err = build(force=True)

    if cl_err_kind == "exc" or in_err_kind == "exc":
        detail = cl_err if cl_err_kind == "exc" else in_err
        return Verdict("BUILD_EXC", detail.split(":")[0], detail, tags)
    if (fn_cl is None) != (fn_in is None):
        return Verdict(
            "DIVERGE_BUILD",
            "backend-split",
            f"cranelift: {cl_err or 'builds'} / interp: {in_err or 'builds'}",
            tags,
        )

    duck_out, duck_phase, duck_err = _duck_run(sql, case, udf_objs)

    if fn_cl is None:
        klass = _refusal_class(cl_err)
        return Verdict("REFUSED", klass, cl_err, tags)
    # "constant" is the third legitimate backend: the whole query folded at
    # build time, so there is nothing left to compile OR interpret.
    if fn_cl.backend not in ("cranelift", "constant"):
        tags.append("fallback")
    if fn_in.backend not in ("interpreter", "constant"):
        return Verdict(
            "DIVERGE_VALUE",
            "force-interp-ignored",
            f"forced interpreter, got {fn_in.backend}",
            tags,
        )

    if duck_out is None and duck_phase == "build":
        return Verdict(
            "DIVERGE_BUILD",
            _first_words(duck_err),
            f"confit builds what DuckDB refuses: {duck_err}",
            tags,
        )

    # --- execute --------------------------------------------------------
    table = _arrow_table(case.row_schema, case.rows)
    static_only = (
        case.query.body.frm not in (None, "__THIS__") and not case.query.body.joins
    )

    # infer_arrow cannot take a struct row schema yet (TASK-114), so a
    # struct-bearing case runs through infer_rows instead of being scored as
    # a divergence against our own boundary. Values are still fully compared;
    # only the ARROW leg is unavailable, and the tag says so rather than the
    # case quietly counting as covered.
    rows_only = not static_only and any(
        isinstance(s, G.Struct) for s in case.row_schema.values()
    )
    if rows_only:
        tags.append("rows_only")

    def run_fn(fn):
        """(rows, output schema, error) — one shape whichever entry point."""
        try:
            if static_only:
                return fn.infer_rows([]), fn.output_schema, None
            if rows_only:
                return fn.infer_rows(case.rows), fn.output_schema, None
            out = fn.infer_arrow(table)
            return out.to_pylist(), out.schema, None
        except Exception as e:  # noqa: BLE001
            return None, None, f"{type(e).__name__}: {e}"

    got_cl, sch_cl, trap_cl = run_fn(fn_cl)
    got_in, sch_in, trap_in = run_fn(fn_in)

    if (trap_cl is None) != (trap_in is None):
        return Verdict(
            "DIVERGE_VALUE",
            "backend-trap-split",
            f"cranelift: {trap_cl or 'rows'} / interp: {trap_in or 'rows'}",
            tags,
        )
    if trap_cl is not None:
        if duck_out is None:  # both sides error at run time
            return Verdict("AGREE_TRAP", _first_words(duck_err), trap_cl, tags)
        return Verdict(
            "DIVERGE_TRAP",
            _first_words(trap_cl),
            f"confit traps, DuckDB returns rows: {trap_cl}",
            tags,
        )
    if duck_out is None:
        return Verdict(
            "DIVERGE_TRAP",
            _first_words(duck_err),
            f"DuckDB errors, confit returns rows: {duck_err}",
            tags,
        )

    # --- compare --------------------------------------------------------
    if static_only:
        want = duck_out.to_pylist()
        if _key(got_cl) != _key(want) or _key(got_in) != _key(want):
            return Verdict(
                "DIVERGE_VALUE", "static-only-values", f"{got_cl} != {want}", tags
            )
        return Verdict("AGREE", "", "", tags)

    delta = _schema_delta(duck_out.schema, sch_cl)
    cast_to = None
    if delta is not None:
        if delta[0] == "diff":
            return Verdict("DIVERGE_VALUE", "schema", delta[1], tags)
        tags.append(delta[1])
        cast_to = sch_cl
    try:
        want = _norm(duck_out, cast_to)
    except Exception as e:  # noqa: BLE001 — cast refuses: widths lied
        return Verdict("DIVERGE_VALUE", "schema-cast", str(e), tags)

    if sch_cl != sch_in or _key(got_cl) != _key(got_in):
        return Verdict(
            "DIVERGE_VALUE", "backend-values", "cranelift != interpreter", tags
        )
    if _key(got_cl) != _key(want):
        return Verdict("DIVERGE_VALUE", "values", f"{got_cl[:4]} != {want[:4]}", tags)

    v = _extra_legs(fn_cl, case, table, got_cl, want, ests, tags, rows_only)
    if v is not None:
        return v
    return Verdict("AGREE", "", "", tags)


def _extra_legs(
    fn, case, table, got, want, ests, tags, rows_only=False
) -> Verdict | None:
    """The boundary checks a plain differential run misses.

    `got` is the primary run's rows. Under `rows_only` the primary run WAS
    infer_rows, so every leg here needs infer_arrow and none of them can
    run; the case keeps its value comparison and loses the boundary legs.
    That is the honest shape of TASK-114 — not silence, and not a finding.
    """
    if rows_only:
        return None

    # infer_rows (dict rows) agrees with infer_arrow -- case.rows is already
    # a list of TOTAL dicts against case.row_schema (gen.py's own contract).
    try:
        rows = fn.infer_rows(case.rows)
    except Exception as e:  # noqa: BLE001
        return Verdict(
            "DIVERGE_VALUE",
            "infer-vs-arrow",
            f"infer_rows() raised where infer_arrow ran: {e}",
            tags,
        )
    if _key(rows) != _key(got):
        return Verdict(
            "DIVERGE_VALUE",
            "infer-vs-arrow",
            f"{rows[:4]} != {got[:4]}",
            tags,
        )

    # hostile Arrow: sliced offset, chunked, empty (TASK-67 class)
    if len(table) >= 2:
        for name, hostile, sub in (
            ("sliced", table.slice(1), table.slice(1)),
            ("chunked", pa.concat_tables([table.slice(0, 1), table.slice(1)]), table),
            ("empty", table.slice(0, 0), table.slice(0, 0)),
        ):
            try:
                h = fn.infer_arrow(hostile)
                ref = fn.infer_arrow(sub.combine_chunks())
            except ValueError as e:
                # A refusal that NAMES the hostile condition is the contract
                # working ("column has 2 chunks — call combine_chunks()").
                # A wrong answer would be the bug; a named no is not.
                if "chunk" in str(e) or "align" in str(e):
                    continue
                return Verdict("DIVERGE_VALUE", f"hostile-{name}", f"raised: {e}", tags)
            except Exception as e:  # noqa: BLE001
                return Verdict("DIVERGE_VALUE", f"hostile-{name}", f"raised: {e}", tags)
            if _key(h.to_pylist()) != _key(ref.to_pylist()):
                return Verdict(
                    "DIVERGE_VALUE",
                    f"hostile-{name}",
                    "hostile arrow input changed the answer",
                    tags,
                )

    # single-row concatenation == batch (cross-row state leak)
    if 2 <= len(table) <= 6:
        singles = [fn.infer_arrow(table.slice(i, 1)) for i in range(len(table))]
        cat = pa.concat_tables(singles).to_pylist()
        if _key(cat) != _key(got):
            return Verdict(
                "DIVERGE_VALUE",
                "batch-vs-single",
                "row-at-a-time disagrees with the batch",
                tags,
            )

    # sklearn is a second ground truth on the plain tree template
    if ests and _plain_tree_shape(case):
        import numpy as np

        cols = list(case.row_schema)
        out = got
        for i, r in enumerate(case.rows):
            iid = r[cols[0]]
            feats = [r[c] for c in cols[1 : 1 + case.tree.n_features]]
            if iid in ests and all(f is not None for f in feats):
                x = np.asarray([feats], dtype=np.float64)
                p = float(ests[iid].predict(x)[0])
                o = out[i][case.query.body.items[0][1]]
                if o is None or abs(o - p) > 1e-9:
                    return Verdict(
                        "DIVERGE_VALUE",
                        "sklearn",
                        f"row {i}: kernel {o} != sklearn {p}",
                        tags,
                    )
    return None


def _plain_tree_shape(case: G.Case) -> bool:
    b = case.query.body
    return (
        case.tree is not None
        and not case.query.ctes
        and not b.joins
        and b.where is None
        and len(b.items) == 1
        and isinstance(b.items[0][0], G.Call)
        and b.items[0][0].name == "trees"
        and b.items[0][1] is not None
        and all(isinstance(a, G.Col) for a in b.items[0][0].args)
    )


def _refusal_class(msg: str) -> str:
    return _first_words(msg.replace("'", "").replace('"', ""))


def _first_words(s: str, n: int = 6) -> str:
    return " ".join(s.split()[:n])[:80]


def run_case_json(seed: int) -> dict:
    case = G.gen(seed)
    try:
        v = run_case(case)
    except Exception as e:  # noqa: BLE001 — oracle's own bug, not the engine's
        v = Verdict("SKIP", f"oracle:{type(e).__name__}", str(e), case.tags)
    return {"seed": seed, "sql": G.render(case.query), **v.to_json()}
