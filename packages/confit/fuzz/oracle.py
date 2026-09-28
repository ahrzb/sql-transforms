"""One Case in, one Verdict out.

The contract under test: confit either matches DuckDB bit-for-bit — with the
same UDFs registered — or refuses at build with a named ValueError. Anything
else is a finding. Extra legs beyond the three-way comparison: infer_rows vs
infer_arrow, hostile Arrow input (sliced / chunked / empty), single-row vs
batch concatenation, and rebuild determinism. Tree cases use fuzz.trees:
random models in the engine's tree protocol, with a reference walk as the
DuckDB-side UDF (sklearn parity is sql-transform's gate).

# TWO DuckDB readings, and the BASELINE is the optimizer-off one

Every case is run against DuckDB twice, on one connection:

    PRAGMA disable_optimizer   the BASELINE — eager evaluation, and a
                               function of the QUERY alone
    PRAGMA enable_optimizer    what a user actually sees

Measured: every trap-elision divergence in the record collapses to
a TRAP with the optimizer off, which is what this engine does natively. The
two readings therefore bracket the answer, and a finding classifies itself
instead of needing a human to reason about fold visibility:

  ours == off == on        AGREE.
  ours != off, ours != on  a real bug, and the baseline says so with the
                           optimizer out of the picture.
  ours == off, off != on   DIVERGE_OPT. We match eager semantics; an optimizer
                           pass makes the user's DuckDB answer differently, so
                           this is user-visible and needs a decision.
  ours == on, off != on    OPT_EMULATED. We answer like the optimizer and
                           unlike the oracle, which means we are reproducing
                           a plan-rewrite pass. That is a BUG, not a note:
                           the class is empty, and if it refills something
                           reintroduced an emulation.

Why the optimizer-off run is the BASELINE and not merely a second opinion: it
is the only one of the two whose answer is a function of the query. The
optimizer's is not — `statistics_propagation` reads the column's null
statistic, so the same query over the same rows answers differently depending
on the table's insert history (see
tests/known_divergences/test_trap_elision.py). A baseline you cannot compute
from the query is not a baseline.

Note this does NOT restate the user-facing contract, which still names what a
user's DuckDB returns — optimizer on. `DIVERGE_OPT` is exactly the gap
between the two, which is why it stays a finding.

The connection is `confit.oracle.Oracle` and the canonical forms every
comparison below is written in -- `multiset`, `sequence`, `dedup_names` --
are `confit.compare`'s. Both are the ones the tests compare against, so
neither the baseline nor the meaning of "equal" can drift from theirs, and
both come from the PACKAGE, which is what keeps the standing rule intact:
fuzz/ must not import from tests/. What is NOT shared is the UDF
`create_function` recipe: it mirrors tests/test_udfs.py `udf_check` and stays
duplicated on purpose, under that same rule and because writing it a second
time from the documented protocol alone is itself the check that the protocol
doc suffices.
"""

from __future__ import annotations

import dataclasses
import decimal
import math
import os
import re
import sys
from dataclasses import dataclass
from dataclasses import field as dfield

import duckdb
import pyarrow as pa
from confit.compare import (
    dedup_names,
    multiset,
    non_null_violation,
    same_type,
    sequence,
)
from confit.oracle import Oracle

from . import coverage, trees
from . import gen as G

KINDS = (
    "DIVERGE_VALUE",
    "DIVERGE_BUILD",
    "DIVERGE_TRAP",
    # we match the eager baseline, an optimizer pass makes the user's DuckDB
    # answer differently — user-visible, so a finding
    "DIVERGE_OPT",
    "BUILD_EXC",
    "AGREE",
    "AGREE_TRAP",
    # we match optimizer-ON against an eager baseline that disagrees: a pass
    # we are reproducing. Since the oracle moved, that is a FINDING.
    "OPT_EMULATED",
    "REFUSED",
    # the case's answer has a width we have not shipped, so there is nothing
    # to compare — see the unshipped-feature note below
    "UNSHIPPED",
    # confit hit a resource ceiling the serving contract excludes from parity
    # (exclusion: resource-ceilings): nothing to compare, see RESOURCE_CEILINGS
    "EXCLUDED",
    "SKIP",
)

# The runtime traps of the serving contract's resource-ceilings exclusion
# (docs/specs/serving-contract.md), verbatim. A case whose run hits one on BOTH
# backends is EXCLUDED and DuckDB is not run: the exclusion is defined by the
# ceiling, not by DuckDB's answer, and past it DuckDB builds the gigabyte
# value the ceiling exists to refuse -- minutes per case, and not
# interruptible inside one vector (nightly seed 1003321, `lpad` to 2^31
# characters). Anything else, a ceiling on one backend included, compares.
RESOURCE_CEILINGS = (
    "string builder result exceeds 1 GiB",
    "string column exceeds 2 GiB in one",
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
    # STATIC-ONLY: served exactly, emitted as decimal128(p,s) whatever the
    # internal storage tier.
    "decimal(4,2)": pa.decimal128(4, 2),
    "decimal(9,4)": pa.decimal128(9, 4),
    "decimal(18,6)": pa.decimal128(18, 6),
    "decimal(38,0)": pa.decimal128(38, 0),
}
_DUCK_T = {
    pa.bool_(): "BOOLEAN",
    pa.int64(): "BIGINT",
    pa.float64(): "DOUBLE",
    pa.string(): "VARCHAR",
}

# THE UNSHIPPED-FEATURE VERDICT, and why it is not a comparison
#
# A feature whose output WIDTH is not shipped gets its own verdict,
# UNSHIPPED, carrying the class and the lane that differs, and no value
# comparison: casting DuckDB's answer down to our width so the rows could
# still be checked manufactures artifacts and grades the gap as agreement.
# `_type_delta` holds one arm per such feature, and `_UNSHIPPED_REACH` one
# reach pattern.
#
# No feature is unshipped today. The last one, decimals (DuckDB types `1.5`
# as DECIMAL(2,1), which served as f64), shipped with decimal expressions
# (docs/specs/decimal-expressions.md); its arm was deleted, so a
# decimal-vs-double delta is now a divergence like any other.


@dataclass
class Verdict:
    kind: str  # one of KINDS
    klass: str = ""  # dedup key within the kind
    detail: str = ""
    # the case's own construct tags, plus oracle-side notes (a known width
    # class, `fallback`)
    tags: list[str] = dfield(default_factory=list)
    # REFUSED only: what the baseline reading did with the same query, one of
    # ORACLE_OUTCOMES. Reporting, not adjudication — a query DuckDB serves and
    # we refuse is not by itself a defect.
    oracle: str = ""
    # AGREE_TRAP only: DuckDB's whole error (`klass` keeps its first words).
    # In memory for `fuzz.parity`; not part of the campaign's JSON.
    oracle_detail: str = ""

    def to_json(self):
        out = {
            "kind": self.kind,
            "klass": self.klass,
            "detail": self.detail[:500],
            "tags": self.tags,
        }
        if self.oracle:
            out["oracle"] = self.oracle
        return out


# What the baseline reading did: returned rows, refused at bind/build, or
# trapped at run time. The same split `_exec` phases on.
ORACLE_OUTCOMES = ("serves", "rejects", "traps")


def _oracle_outcome(duck) -> str:
    out, phase, _ = duck
    if out is not None:
        return "serves"
    return "rejects" if phase == "build" else "traps"


# ------------------------------------------------------------- materialize


def _arrow_field(name: str, spec) -> pa.Field:
    """One field from a gen.py storage spec: a trailing `?` is the nullable
    flag, and a `G.Struct` nests."""
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
    """One output lane off the shared accumulator. `salt` is the lane's own
    index, so sibling lanes never carry the same value — a lane read off the
    wrong position cannot pass by coincidence."""
    if ty == "float":
        return acc + salt
    if ty == "int":
        return int(acc) % 1000 + salt
    if ty == "bool":
        return (int(acc) + salt) % 2 == 0
    return f"v{(int(acc) + salt) % 97}"


def make_udf(spec: G.UdfSpec):
    """The protocol object for `spec`, deterministic by construction.

    The SAME instance is handed to confit as a `udfs` entry and to DuckDB via
    `create_function`, so a value difference is the engine's and never two
    different functions.
    """
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


# One oracle connection per process, handed from case to case: opening one is
# about half a case's cost. A case hands it on only in the state a fresh
# `Oracle()` starts in -- its tables dropped, its UDFs removed, the optimizer
# off again -- and only when no statement on it raised and the query was
# SELECTs alone. Otherwise it is closed and the next case opens a fresh one:
# an INTERNAL error invalidates the whole database (DuckDB's
# client_context.cpp), an interrupted or half-fetched query leaves its result
# behind, and a statement with effects of its own would outlive the case. A
# worker the runner's watchdog kills takes its connection with it. What a
# SELECT alone can still change, such as `setseed()` reseeding `random()`, no
# generated query does, and a nondeterministic answer compares nothing anyway.
_idle: Oracle | None = None


def _duck_con(case: G.Case, udf_objs) -> Oracle:
    """The oracle, with the case's UDFs registered and its tables loaded as
    NATIVE tables. Both readings share it, and `_duck_run` hands it back."""
    global _idle
    con, _idle = _idle or Oracle(), None
    try:
        for u in udf_objs:
            params = [_DUCK_T[t] for t in u.takes.types]
            if hasattr(u, "instances"):
                params = ["BIGINT", *params]
            con.create_function(
                u.name, _scalar_form(u), params, _duck_ret(u), null_handling="special"
            )
        for name, (sch, rows) in case.statics.items():
            con.load(name, _arrow_table(sch, rows))
        con.load("__THIS__", _arrow_table(case.row_schema, case.rows))
    except BaseException:
        con.close()
        raise
    return con


def _hand_back(con: Oracle, case: G.Case, udf_objs, reuse: bool) -> None:
    """Undo what `_duck_con` and the readings did to `con` and keep it for the
    next case, or close it when `reuse` is False."""
    global _idle
    if not reuse:
        con.close()
        return
    for name in (*case.statics, "__THIS__"):
        con.execute(f'DROP TABLE "{name}"')
    for u in udf_objs:
        con.remove_function(u.name)
    con.execute("PRAGMA disable_optimizer")
    _idle = con


def _exec(con, sql):
    """`(table, phase, detail)`, phase being "build" or "run" — the split
    that separates "DuckDB refuses this query" from "DuckDB traps on this
    data". On success the two error slots are None.

    Deliberately wider than the oracle's own `try_answer`, which names
    `duckdb.Error` and `UnicodeDecodeError` and lets everything else
    through: a campaign classifies
    whatever comes back rather than dying on it, so anything unnamed is
    phased as a run-time trap and reported with its class.
    """
    try:
        return con.answer(sql), None, None
    except Exception as e:  # noqa: BLE001 — classify, don't die
        phase = "build" if type(e).__name__ in _DUCK_BUILD_ERRS else "run"
        return None, phase, f"{type(e).__name__}: {e}"


def _duck_run(sql, case: G.Case, udf_objs):
    """Both readings, on ONE connection: `(optimizer_off, optimizer_on)`.

    Sharing the connection is not just a saving (the tables materialise once,
    so the second execute is nearly free) — it is also what makes the pair
    comparable. `statistics_propagation` reads per-column statistics, so two
    separate connections could differ for reasons that have nothing to do
    with the optimizer.

    The baseline reading needs no pragma of its own: an oracle is
    optimizer-off by construction, and the flip below is the exception.
    """
    con = _duck_con(case, udf_objs)
    reuse = False
    try:
        off = _exec(con, sql)
        con.optimizer_on()
        on = _exec(con, sql)
        reuse = (
            off[0] is not None
            and on[0] is not None
            and all(
                s.type == duckdb.StatementType.SELECT
                for s in con.extract_statements(sql)
            )
        )
        return off, on
    finally:
        _hand_back(con, case, udf_objs, reuse)


def _schema_delta(duck: pa.Schema, ours: pa.Schema):
    """`(kind, klass, detail)`, or None when the schemas agree.

    `kind` is "unshipped" for a feature whose width we have not shipped —
    classified, never compared — and "diff" for every other mismatch, which
    is a divergence. The first unshipped lane names the case; a "diff"
    anywhere outranks it, because a real difference is not excused by a
    known gap sitting in another column.
    """
    if dedup_names(list(duck.names)) != list(ours.names):
        return ("diff", "", f"names {duck.names} != {ours.names}")
    unshipped = None
    for d, o in zip(duck, ours, strict=True):
        r = _type_delta(d.type, o.type)
        detail = f"{d.name}: duck {d.type} != ours {o.type}"
        if r == "diff":
            return ("diff", "", detail)
        if r is not None and unshipped is None:
            unshipped = ("unshipped", r, detail)
    return unshipped


def _type_delta(duck: pa.DataType, ours: pa.DataType) -> str | None:
    """None = equal; a class name = an unshipped feature's width (recursing
    into structs — a decimal lane inside struct_pack still classifies);
    "diff". Nullability is never a difference: it must be truthful, not
    DuckDB's, and `run_case` checks it against our own rows instead."""
    if same_type(duck, ours):
        return None
    # One arm per unshipped feature; none today (see note above).
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


def run_case(case: G.Case) -> Verdict:
    """One case's verdict: build both backends, run both DuckDB readings,
    classify, then the boundary legs.

    Refusals, traps and disagreements all come back AS a Verdict. An
    exception escaping here is the oracle's own bug, and `run_case_json`
    turns that into SKIP rather than blaming the engine.
    """
    sql = case.sql if case.sql is not None else G.render(case.query)
    tags = list(case.tags)
    schema = _arrow_schema(case.row_schema)
    statics = {n: _arrow_table(sch, rows) for n, (sch, rows) in case.statics.items()}
    udf_objs = [make_udf(u) for u in case.udfs]
    if case.tree is not None:
        tree_obj = trees.make_tree(case.tree, case.seed)
        udf_objs.append(tree_obj)

    # `case.output` is not forwarded: dict rows are the only output mode, so
    # the field gen.py still fills has nothing to select.
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

    _phase("confit:build")
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

    if fn_cl is None:
        # Run both readings anyway and keep the baseline's outcome, so the
        # report can say what each refusal costs.
        _phase("oracle")
        duck_off, _ = _duck_run(sql, case, udf_objs)
        klass = _refusal_class(cl_err)
        return Verdict("REFUSED", klass, cl_err, tags, _oracle_outcome(duck_off))
    if fn_cl.backend != "cranelift":
        tags.append("fallback")
    if fn_in.backend != "interpreter":
        return Verdict(
            "DIVERGE_VALUE",
            "force-interp-ignored",
            f"forced interpreter, got {fn_in.backend}",
            tags,
        )

    # --- execute --------------------------------------------------------
    table = _arrow_table(case.row_schema, case.rows)

    # No escape hatch for struct row columns: infer_arrow takes a struct row
    # schema, so a struct-bearing case runs through the SAME boundary as
    # every other case and gets the whole battery below with it.
    def run_fn(fn):
        """(rows, output schema, error) — one shape whichever entry point."""
        try:
            out = fn.infer_arrow(table)
            return out.to_pylist(), out.schema, None
        except Exception as e:  # noqa: BLE001
            return None, None, f"{type(e).__name__}: {e}"

    _phase("confit:run")
    got_cl, sch_cl, trap_cl = run_fn(fn_cl)
    got_in, sch_in, trap_in = run_fn(fn_in)

    if (
        trap_cl is not None
        and trap_in is not None
        and any(c in trap_cl for c in RESOURCE_CEILINGS)
        and any(c in trap_in for c in RESOURCE_CEILINGS)
    ):
        return Verdict("EXCLUDED", "resource-ceiling", trap_cl, tags)

    if (trap_cl is None) != (trap_in is None):
        return Verdict(
            "DIVERGE_VALUE",
            "backend-trap-split",
            f"cranelift: {trap_cl or 'rows'} / interp: {trap_in or 'rows'}",
            tags,
        )
    # Backend agreement is a question about US and does not involve DuckDB, so
    # it is settled once, before either reading.
    if sch_cl != sch_in or (
        got_cl is not None
        and got_in is not None
        and multiset(got_cl) != multiset(got_in)
    ):
        return Verdict(
            "DIVERGE_VALUE", "backend-values", "cranelift != interpreter", tags
        )
    # A non-null promise is ours to keep whatever DuckDB declares, so it is
    # checked against our own rows, and like backend agreement it needs no
    # DuckDB reading at all.
    unsound = got_cl is not None and non_null_violation(sch_cl, got_cl)
    if unsound:
        return Verdict("DIVERGE_VALUE", "unsound-non-null", f"NULL at {unsound}", tags)

    # DuckDB runs after confit, so an excluded case never waits on it.
    _phase("oracle")
    duck_off, duck_on = _duck_run(sql, case, udf_objs)

    # --- compare, against each reading -----------------------------------
    def against(duck) -> Verdict:
        """Our one result versus ONE DuckDB reading. Pure comparison — every
        side has already been executed, so calling it twice costs nothing."""
        duck_out, duck_phase, duck_err = duck
        if duck_out is not None and len(set(duck_out.schema.names)) != len(
            duck_out.schema.names
        ):
            duck_out = duck_out.rename_columns(dedup_names(list(duck_out.schema.names)))
        t = list(tags)
        if duck_out is None and duck_phase == "build":
            return Verdict(
                "DIVERGE_BUILD",
                _first_words(duck_err),
                f"confit builds what DuckDB refuses: {duck_err}",
                t,
            )
        if trap_cl is not None:
            if duck_out is None:  # both sides error at run time
                return Verdict(
                    "AGREE_TRAP",
                    _first_words(duck_err),
                    trap_cl,
                    t,
                    oracle_detail=duck_err,
                )
            return Verdict(
                "DIVERGE_TRAP",
                _first_words(trap_cl),
                f"confit traps, DuckDB returns rows: {trap_cl}",
                t,
            )
        if duck_out is None:
            return Verdict(
                "DIVERGE_TRAP",
                _first_words(duck_err),
                f"DuckDB errors, confit returns rows: {duck_err}",
                t,
            )
        delta = _schema_delta(duck_out.schema, sch_cl)
        if delta is not None:
            dkind, klass, detail = delta
            if dkind == "diff":
                return Verdict("DIVERGE_VALUE", "schema", detail, t)
            # An unshipped width is classified, never compared: casting the
            # oracle's answer down to ours would absorb the gap as agreement.
            return Verdict("UNSHIPPED", klass, detail, t)
        want = duck_out.to_pylist()
        if multiset(got_cl) != multiset(want):
            return Verdict("DIVERGE_VALUE", "values", f"{got_cl[:4]} != {want[:4]}", t)
        return Verdict("AGREE", "", "", t)

    v_off, v_on = against(duck_off), against(duck_on)
    agreed = ("AGREE", "AGREE_TRAP")
    if "UNSHIPPED" in (v_off.kind, v_on.kind):
        # An unshipped width outranks the optimizer bracket: neither reading
        # was value-compared, so neither can be evidence for or against a
        # plan-rewrite pass.
        v = v_off if v_off.kind == "UNSHIPPED" else v_on
    elif v_off.kind in agreed and v_on.kind not in agreed:
        # Eager semantics agree with us; a pass changes what the USER sees.
        v = Verdict(
            "DIVERGE_OPT",
            v_on.klass,
            f"agrees with optimizer-off DuckDB, not with optimizer-on: {v_on.detail}",
            v_on.tags,
        )
    elif v_on.kind in agreed and v_off.kind not in agreed:
        # We answer like a plan-rewrite pass and unlike the oracle: a finding,
        # never agreement or coverage.
        v = Verdict(
            "OPT_EMULATED",
            v_off.klass,
            f"agrees with the user-visible optimizer-on answer; the eager "
            f"baseline differs: {v_off.detail}",
            v_off.tags,
        )
    else:
        # Either both agree, or neither does — and when neither does, the
        # baseline is the one to report, because it names the bug without an
        # optimizer in the way.
        v = v_off

    # UNSHIPPED still earns the boundary legs: they are OUR side against
    # itself, with no DuckDB in them, so an unshipped width cannot excuse a
    # self-inconsistency and a real DIVERGE_VALUE there outranks the class.
    # OPT_EMULATED does not: it is a mismatch, and like every other mismatch
    # it is the primary finding, which no later leg may replace.
    if v.kind not in ("AGREE", "UNSHIPPED"):
        return v
    if trap_cl is not None:
        return v  # the boundary legs all need a non-trapping row run
    _phase("confit:legs")
    extra = _extra_legs(fn_cl, case, table, got_cl, v.tags)
    return extra if extra is not None else v


def _extra_legs(fn, case, table, got, tags) -> Verdict | None:
    """The boundary checks a plain differential run misses.

    `got` is the primary run's rows, which always came from infer_arrow.
    """
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
    if multiset(rows) != multiset(got):
        return Verdict(
            "DIVERGE_VALUE",
            "infer-vs-arrow",
            f"{rows[:4]} != {got[:4]}",
            tags,
        )

    # hostile Arrow: sliced offset, chunked, empty
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
            if multiset(h.to_pylist()) != multiset(ref.to_pylist()):
                return Verdict(
                    "DIVERGE_VALUE",
                    f"hostile-{name}",
                    "hostile arrow input changed the answer",
                    tags,
                )

    # single-row concatenation == batch, AS A SEQUENCE: the serving contract
    # is that output rows follow input rows -- map exactly, filter as a
    # subsequence, many as per-input-row blocks in input order. `multiset`
    # here would accept any permutation and leave an order bug on the row path
    # invisible. (The leg also catches cross-row state leaks.)
    if 2 <= len(table) <= 6:
        singles = [fn.infer_arrow(table.slice(i, 1)) for i in range(len(table))]
        cat = pa.concat_tables(singles).to_pylist()
        if sequence(cat) != sequence(got):
            kind = (
                "batch-vs-single-order"
                if multiset(cat) == multiset(got)
                else "batch-vs-single"
            )
            return Verdict(
                "DIVERGE_VALUE",
                kind,
                "row-at-a-time disagrees with the batch"
                + (" (order only)" if kind.endswith("order") else ""),
                tags,
            )
        # reversal: feeding the rows backwards must reverse the BLOCKS.
        # Self-contained -- no DuckDB involved, because DuckDB's own row
        # order is not a function of the query even on the row path.
        try:
            rev = fn.infer_arrow(
                table.take(list(range(len(table) - 1, -1, -1)))
            ).to_pylist()
        except Exception as e:  # noqa: BLE001
            return Verdict(
                "DIVERGE_VALUE", "reversal", f"reversed input raised: {e}", tags
            )
        want_rev = [r for s in reversed(singles) for r in s.to_pylist()]
        if sequence(rev) != sequence(want_rev):
            return Verdict(
                "DIVERGE_VALUE",
                "reversal",
                "reversed input did not reverse the output blocks",
                tags,
            )

    return None


def _refusal_class(msg: str) -> str:
    return _first_words(msg.replace("'", "").replace('"', ""))


def _first_words(s: str, n: int = 6) -> str:
    return " ".join(s.split()[:n])[:80]


# Generator reach, per unshipped feature: does the SQL contain the construct
# that yields the width? Read off the rendered text, string literals removed,
# so a verdict of any kind still says whether the width was in play.
# None today: decimals, the last one, shipped.
_STRING_LIT = re.compile(r"'(?:[^']|'')*'")
_UNSHIPPED_REACH: dict[str, re.Pattern] = {}
UNSHIPPED_FEATURES = tuple(_UNSHIPPED_REACH)


def unshipped_reach(sql: str) -> set[str]:
    code = _STRING_LIT.sub("''", sql)
    return {f for f, rx in _UNSHIPPED_REACH.items() if rx.search(code)}


# Phase markers. A worker the runner kills mid-case returns nothing, so the
# case announces each stage on stderr BEFORE running it; the last marker in
# a dead worker's stderr names the side that hung or crashed. Internal audit
# vocabulary, like the verdict classes: not a public API.
PHASE_MARK = "@@phase"


def _phase(name: str) -> None:
    print(f"{PHASE_MARK} {name}", file=sys.stderr, flush=True)


def plain(x):
    """`x` as JSON-safe data: dataclasses become dicts, tuples lists, and the
    values JSON has no type for (Decimal, bytes) their exact text."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, decimal.Decimal):
        return str(x)
    if isinstance(x, bytes):
        return x.hex()
    return x


def case_inputs(case: G.Case) -> dict:
    """Everything besides the SQL that a replay needs. A seed only names a
    case under one generator revision; these name it under any."""
    return {
        "row_schema": plain(case.row_schema),
        "rows": plain(case.rows),
        "statics": plain(case.statics),
        "udfs": plain(case.udfs),
        "tree": plain(case.tree),
        "shape": case.shape,
    }


def _revive_spec(spec):
    """A `plain`ed storage spec back to itself: a Struct was a dict."""
    if isinstance(spec, dict):
        return G.Struct(
            tuple((n, _revive_spec(s)) for n, s in spec["fields"]), spec["nullable"]
        )
    return spec


def _revive_cell(spec, v):
    """A `plain`ed cell back to itself under its spec: a Decimal was its
    text, a struct value a dict of cells. Every other value JSON keeps."""
    if v is None:
        return None
    if isinstance(spec, G.Struct):
        return {n: _revive_cell(s, v[n]) for n, s in spec.fields}
    if spec.rstrip("?").startswith("decimal("):
        return decimal.Decimal(v)
    return v


def _revive_table(schema: dict, rows: list[dict]) -> tuple[dict, list[dict]]:
    schema = {c: _revive_spec(s) for c, s in schema.items()}
    return schema, [{c: _revive_cell(schema[c], r[c]) for c in schema} for r in rows]


def case_from_inputs(seed: int, sql: str, inputs: dict) -> G.Case:
    """The case a stored `(sql, case_inputs(case))` names, under any generator
    revision: `run_case` answers it exactly as it answered the original. It
    carries no AST (`query` is None), so only `run_case` can take it."""
    row_schema, rows = _revive_table(inputs["row_schema"], inputs["rows"])
    statics = {
        name: _revive_table(schema, rows)
        for name, (schema, rows) in inputs["statics"].items()
    }
    udfs = [
        G.UdfSpec(
            u["name"],
            [tuple(t) for t in u["takes"]],
            tuple(u["ret"]),
            u["instances"],
        )
        for u in inputs["udfs"]
    ]
    tree = inputs["tree"]
    if tree is not None:
        tree = G.TreeSpec(**{**tree, "int_features": tuple(tree["int_features"])})
    return G.Case(
        seed=seed,
        row_schema=row_schema,
        rows=rows,
        statics=statics,
        udfs=udfs,
        tree=tree,
        query=None,
        shape=inputs["shape"],
        output=None,
        sql=sql,
    )


def run_case_json(seed: int) -> dict:
    """`gen(seed)` through `run_case`, as the JSON line the worker prints:
    the verdict, the SQL, and the case's inputs."""
    case = G.gen(seed)
    try:
        v = run_case(case)
    except Exception as e:  # noqa: BLE001 — oracle's own bug, not the engine's
        v = Verdict("SKIP", f"oracle:{type(e).__name__}", str(e), case.tags)
    sql = G.render(case.query)
    out = {"seed": seed, "sql": sql, **v.to_json(), "inputs": case_inputs(case)}
    out["tags"] = out["tags"] + [f"reaches:{f}" for f in sorted(unshipped_reach(sql))]
    out["triples"] = sorted(coverage.key(t) for t in coverage.triples(case.query))
    return out
