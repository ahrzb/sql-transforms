"""NATURAL / USING join keys over STRUCT shared columns (TASK-133).

Design: docs/superpowers/specs/2026-08-25-task-133-join-keys-design.md.

DuckDB's join binder intersects column NAME SETS with no type inspection
(bind_joinref.cpp:185-208), so a shared STRUCT is an ordinary join key there
and `=` on a struct is `row_matcher.cpp:379-382`: top-level Equals, every
child matched with NOT_DISTINCT_FROM. A struct key expands here into
composite ordinary keys -- one PLAIN presence key for the whole struct, one
NOT-DISTINCT presence key per nested node, one NOT-DISTINCT key per leaf --
which is that recursion written out.

Every test asserts THE ORACLE's answer first, so an oracle move is a loud
failure rather than a silent rebaseline. The conftest fixture already puts
`PRAGMA disable_optimizer` on every connection.
"""

from __future__ import annotations

import datetime

import duckdb
import pyarrow as pa
import pytest
from confit import DuckDBInferFn

_S1 = pa.struct([("mean", pa.float64())])
_S2 = pa.struct([("inner", pa.struct([("val", pa.float64())]))])
_S3 = pa.struct([("j", pa.struct([("v", pa.float64())]))])

_ROW_W = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _S1)])
_ROW_W2 = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _S2)])
_ROW_W3 = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _S3)])

# The THREE forms that must be told apart: NATURAL keys on ALL shared
# columns, USING only on the named ones (matrix (h) row 2 proves they answer
# differently on the same data).
_FORMS = [
    "SELECT z AS o FROM __THIS__ NATURAL JOIN s",
    "SELECT z AS o FROM __THIS__ JOIN s USING (id, w)",
    "SELECT z AS o FROM __THIS__ JOIN s USING (w)",
]
_LEFT_FORMS = [
    "SELECT z AS o FROM __THIS__ NATURAL LEFT JOIN s",
    "SELECT z AS o FROM __THIS__ LEFT JOIN s USING (id, w)",
    "SELECT z AS o FROM __THIS__ LEFT JOIN s USING (w)",
]


def _row_table(schema, rows):
    return pa.Table.from_pylist(rows, schema=schema)


def oracle(sql, row_table, static):
    """DuckDB's own answer, over exactly the arrow data we hand the engine."""
    con = duckdb.connect()
    con.register("ra", row_table)
    con.register("sa", static)
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM ra")
    con.execute("CREATE TABLE s AS SELECT * FROM sa")
    res = con.execute(sql)
    names = [d[0] for d in res.description]
    return [dict(zip(names, r, strict=True)) for r in res.fetchall()]


def ours(sql, row_table, static, **kw):
    """The engine's answer through BOTH boundaries -- the arrow ingest and
    the row-object marshaller. A presence lane is filled by each of them
    separately, so a test that exercises one proves nothing about the other.
    """
    fn = DuckDBInferFn(
        sql,
        row_tables={"__THIS__": row_table.schema},
        static_tables={"s": static},
        **kw,
    )
    arrow = fn.infer_arrow(row_table).to_pylist()
    rows = fn.infer_rows(row_table.to_pylist())
    assert rows == arrow, f"row boundary {rows} != arrow boundary {arrow}"
    return arrow


def check(sql, row_table, static, want):
    got_oracle = oracle(sql, row_table, static)
    assert got_oracle == want, f"oracle moved: {got_oracle} != {want}"
    assert ours(sql, row_table, static) == want


def _static_w(wtype, wval, sid=5):
    return pa.table(
        {
            "id": pa.array([sid], pa.int64()),
            "w": pa.array([wval], wtype),
            "z": pa.array([7], pa.int64()),
        }
    )


# --- matrix (a): struct STRUCT(mean DOUBLE) ---------------------------------

_A_CELLS = [
    ({"mean": 1.0}, {"mean": 1.0}, [{"o": 7}]),
    ({"mean": 1.0}, {"mean": 2.0}, []),
    ({"mean": None}, {"mean": 2.0}, []),
    ({"mean": None}, {"mean": None}, [{"o": 7}]),
    (None, {"mean": 2.0}, []),
    (None, None, []),
    ({"mean": None}, None, []),
]


@pytest.mark.parametrize("sql", _FORMS)
@pytest.mark.parametrize(("rw", "sw", "want"), _A_CELLS)
def test_a_struct_join_key_matches_duckdb(sql, rw, sw, want):
    check(sql, _row_table(_ROW_W, [{"id": 5, "w": rw}]), _static_w(_S1, sw), want)


# --- matrix (b): nested structs, including THE DISCRIMINATORS ---------------

_B_CELLS = [
    ({"inner": {"val": 9.0}}, {"inner": {"val": 9.0}}, [{"o": 7}]),
    ({"inner": {"val": 9.0}}, {"inner": {"val": 8.0}}, []),
    ({"inner": {"val": None}}, {"inner": {"val": None}}, [{"o": 7}]),
    ({"inner": None}, {"inner": None}, [{"o": 7}]),
    ({"inner": None}, {"inner": {"val": 9.0}}, []),
    # THE DISCRIMINATOR: both sides flatten to the identical leaf tuple
    # (val = NULL) and DuckDB still says MISS. A leaf-lane-only encoding
    # cannot tell these apart; presence keys are why they exist.
    ({"inner": None}, {"inner": {"val": None}}, []),
]


@pytest.mark.parametrize("sql", _FORMS)
@pytest.mark.parametrize(("rw", "sw", "want"), _B_CELLS)
def test_a_nested_struct_join_key_matches_duckdb(sql, rw, sw, want):
    check(sql, _row_table(_ROW_W2, [{"id": 5, "w": rw}]), _static_w(_S2, sw), want)


@pytest.mark.parametrize("sql", _FORMS)
@pytest.mark.parametrize(
    ("rw", "sw", "want"),
    [
        # the THREE-deep discriminator: {j: NULL} vs {j: {v: NULL}}
        ({"j": None}, {"j": {"v": None}}, []),
        ({"j": None}, {"j": None}, [{"o": 7}]),
        ({"j": {"v": None}}, {"j": {"v": None}}, [{"o": 7}]),
        ({"j": {"v": 1.0}}, {"j": {"v": 1.0}}, [{"o": 7}]),
    ],
)
def test_a_three_deep_struct_join_key_matches_duckdb(sql, rw, sw, want):
    check(sql, _row_table(_ROW_W3, [{"id": 5, "w": rw}]), _static_w(_S3, sw), want)


# --- matrix (c) + (d): float edges inside a field vs the scalar control -----

_FLOAT_CELLS = [
    (float("nan"), float("nan"), [{"o": 7}]),
    (-0.0, 0.0, [{"o": 7}]),
    (float("inf"), float("inf"), [{"o": 7}]),
    (float("nan"), None, []),
    (1.0, 1.0, [{"o": 7}]),
    (None, 1.0, []),
]

_ROW_D = pa.schema(
    [pa.field("id", pa.int64(), nullable=False), pa.field("d", pa.float64())]
)


@pytest.mark.parametrize(("rv", "sv", "want"), _FLOAT_CELLS)
def test_a_struct_join_key_on_float_edges_agrees_with_the_scalar_control(rv, sv, want):
    """NaN keys NaN and -0.0 keys 0.0 INSIDE a struct field exactly as they
    already do for a bare DOUBLE column (canon_f64_bits, matrix (c) vs (d))."""
    check(
        _FORMS[0],
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": rv}}]),
        _static_w(_S1, {"mean": sv}),
        want,
    )
    scalar = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "d": pa.array([sv], pa.float64()),
            "z": pa.array([7], pa.int64()),
        }
    )
    check(
        "SELECT z AS o FROM __THIS__ NATURAL JOIN s",
        _row_table(_ROW_D, [{"id": 5, "d": rv}]),
        scalar,
        want,
    )


# --- matrix (g): the LEFT legs keep the left-miss NULL shape ----------------


@pytest.mark.parametrize("sql", _LEFT_FORMS)
@pytest.mark.parametrize(
    ("rw", "sw", "want"),
    [
        ({"mean": 1.0}, {"mean": 1.0}, [{"o": 7}]),
        ({"mean": 1.0}, {"mean": 2.0}, [{"o": None}]),
        ({"mean": None}, {"mean": None}, [{"o": 7}]),
        (None, None, [{"o": None}]),
        (None, {"mean": 2.0}, [{"o": None}]),
    ],
)
def test_a_left_join_on_a_struct_key_keeps_the_left_miss_shape(sql, rw, sw, want):
    check(sql, _row_table(_ROW_W, [{"id": 5, "w": rw}]), _static_w(_S1, sw), want)


# --- matrix (h): NATURAL and USING key on DIFFERENT column sets -------------


def test_natural_and_using_key_on_different_column_sets():
    """`w` matches while `id` does not: NATURAL keys on both and misses,
    `USING (w)` keys on `w` alone and hits. Same fixture, different answers."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    static = _static_w(_S1, {"mean": 1.0}, sid=6)
    check(_FORMS[0], row, static, [])
    check(_FORMS[1], row, static, [])
    check(_FORMS[2], row, static, [{"o": 7}])


def test_natural_misses_when_only_the_struct_differs():
    """The ticket's own severity-2: ids equal, `w` unequal, and we used to
    key on `id` alone and emit a row DuckDB never produces."""
    check(
        _FORMS[0],
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}]),
        _static_w(_S1, {"mean": 2.0}),
        [],
    )


# --- matrix (j) rows 4, 7-8: field order and name case ----------------------


@pytest.mark.parametrize(
    "sfields", [("b", "a"), ("B", "A")], ids=["reordered", "reordered-and-recased"]
)
def test_struct_fields_pair_by_name_not_position(sfields):
    """A reordered field set is a lossless cast on DuckDB, so it still keys;
    field names pair case-insensitively like every other identifier."""
    ab = pa.struct([("a", pa.float64()), ("b", pa.float64())])
    st = pa.struct([(n, pa.float64()) for n in sfields])
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("w", ab)]
    )
    for a, want in ((1.0, [{"o": 7}]), (9.0, [])):
        check(
            _FORMS[0],
            _row_table(row_schema, [{"id": 5, "w": {"a": 1.0, "b": 2.0}}]),
            _static_w(st, {sfields[0]: 2.0, sfields[1]: a}),
            want,
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT w.mean AS o FROM __THIS__ NATURAL LEFT JOIN s",
        "SELECT w.mean AS o FROM __THIS__ LEFT JOIN s USING (w)",
    ],
)
def test_a_merged_struct_key_resolves_to_the_left_occurrence(sql):
    """The merged USING/NATURAL column is the LEFT one, struct heads
    included: `w.mean` is the row's 1.0 even though the join misses."""
    check(
        sql,
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}]),
        _static_w(_S1, {"mean": 2.0}),
        [{"o": 1.0}],
    )


@pytest.mark.parametrize(
    ("sw", "want"), [({"mean": 2.0}, []), ({"mean": 1.0}, [{"o": 1.0}])]
)
def test_a_struct_key_leaf_stays_addressable_on_the_static_side(sw, want):
    """`s.w.mean` after a struct-keyed NATURAL JOIN: its leaves are KEY
    lanes now, and a key column reconstructs from the dynamic side."""
    check(
        "SELECT s.w.mean AS o FROM __THIS__ NATURAL JOIN s",
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}]),
        _static_w(_S1, sw),
        want,
    )


def test_an_explicit_on_over_a_struct_still_refuses():
    """Scope boundary, pinned: this ticket serves the NATURAL and USING
    arms. `ON t.w = s.w` keeps the existing named refusal."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    sql = "SELECT z AS o FROM __THIS__ JOIN s ON __THIS__.w = s.w"
    assert oracle(sql, row, _static_w(_S1, {"mean": 2.0})) == []
    with pytest.raises(ValueError, match="is a struct"):
        DuckDBInferFn(
            sql,
            row_tables={"__THIS__": row.schema},
            static_tables={"s": _static_w(_S1, {"mean": 2.0})},
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT z AS o FROM __THIS__ NATURAL JOIN s",
        "SELECT z AS o FROM __THIS__ JOIN s USING (w)",
        "SELECT z AS o FROM __THIS__ JOIN s USING (W)",
        'SELECT z AS o FROM __THIS__ JOIN s USING ("W")',
    ],
)
def test_a_shared_struct_name_matches_case_insensitively(sql):
    """Row `W` against static `w`: one shared column, keyed, and UNEQUAL
    values miss (they used to serve -- matrix (j) rows 7-8)."""
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("W", _S1)]
    )
    check(
        sql,
        _row_table(row_schema, [{"id": 5, "W": {"mean": 1.0}}]),
        _static_w(_S1, {"mean": 2.0}),
        [],
    )


# --- matrix (i-scalar): the merge rule the struct head has to join ----------


def test_the_using_merge_takes_the_left_value_on_a_miss():
    """Merged USING output: the LEFT spelling wins and the LEFT value
    survives a miss -- no COALESCE. Pinned on the ALL-SCALAR fixture, which
    is what our engine serves as a VALUE (a struct stays key-only)."""
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("v", pa.int64())]
    )
    row = _row_table(row_schema, [{"id": 5, "v": 1}])
    static = pa.table(
        {
            "id": pa.array([6], pa.int64()),
            "v": pa.array([9], pa.int64()),
            "z": pa.array([7], pa.int64()),
        }
    )
    # USING (id) leaves `v` on both sides -- bare `v` is ambiguous there.
    check(
        "SELECT __THIS__.v AS o FROM __THIS__ LEFT JOIN s USING (id)",
        row,
        static,
        [{"o": 1}],
    )
    check(
        "SELECT v AS o FROM __THIS__ LEFT JOIN s USING (id, v)",
        row,
        static,
        [{"o": 1}],
    )


def test_a_struct_key_is_not_a_servable_value():
    """Non-goal, pinned: `w` becomes a KEY, it does not become servable."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    static = _static_w(_S1, {"mean": 1.0})
    for sql in (
        "SELECT w AS o FROM __THIS__ NATURAL JOIN s",
        "SELECT * FROM __THIS__ NATURAL JOIN s",
    ):
        with pytest.raises(ValueError, match="w"):
            DuckDBInferFn(
                sql,
                row_tables={"__THIS__": row.schema},
                static_tables={"s": static},
            )


# --- the refuse-by-name backstop (AC #4) -----------------------------------

_T0 = datetime.datetime(2020, 1, 1)


def _refuses(sql, row_schema, static, needle, **kw):
    with pytest.raises(ValueError) as e:
        DuckDBInferFn(
            sql,
            row_tables={"__THIS__": row_schema},
            static_tables={"s": static},
            **kw,
        )
    msg = str(e.value)
    assert needle in msg, msg
    # The false claim this ticket removes: the column plainly DOES exist.
    assert "does not exist" not in msg, msg
    return msg


# `(arrow type, [row value, static value], column)` -- two values means the
# sides DISAGREE, which is the cell that used to answer wrongly.
_OPAQUE_SHARED = [
    (pa.timestamp("us"), [_T0, datetime.datetime(2021, 6, 30)], "t"),
    (pa.date32(), [datetime.date(2020, 1, 1), datetime.date(2021, 6, 30)], "t"),
    (pa.time64("us"), [datetime.time(1, 2, 3), datetime.time(4, 5, 6)], "t"),
    (pa.float32(), [1.5, 2.5], "t"),
    (pa.uint64(), [3, 4], "t"),
    (pa.list_(pa.int64()), [[1, 2], [3]], "t"),
]


@pytest.mark.parametrize(
    "sql", _FORMS[:1] + ["SELECT z AS o FROM __THIS__ JOIN s USING (t)"]
)
@pytest.mark.parametrize(("aty", "vals", "col"), _OPAQUE_SHARED)
def test_an_opaque_shared_column_refuses_by_name(sql, aty, vals, col):
    """A shared column with no lane on either side REFUSES naming the column
    (severity-2 -> severity-4). A follow-up ticket adds key-only lanes so
    they serve; what dies here is the wrong ANSWER.

    The ticket's TIMESTAMP leg is the first row: DuckDB keys on `t` and
    returns nothing, and we used to key on `id` alone and return the row.
    """
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("t", aty)]
    )
    row = _row_table(row_schema, [{"id": 5, "t": vals[0]}])
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "t": pa.array(vals[1:] or vals, aty),
            "z": pa.array([7], pa.int64()),
        }
    )
    # The oracle keys on the shared column, so an UNEQUAL one misses.
    assert oracle(sql, row, static) == ([] if len(vals) > 1 else [{"o": 7}])
    _refuses(sql, row_schema, static, f"'{col}'")


def test_a_row_side_decimal_shared_column_refuses_by_name():
    """decimal128 is the asymmetric one: servable on the STATIC side, opaque
    on the ROW side, so the static loop found it and the row side dropped
    it. The message must say which side."""
    dec = pa.decimal128(10, 2)
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("t", dec)]
    )
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "t": pa.array([1], dec),
            "z": pa.array([7], pa.int64()),
        }
    )
    _refuses(_FORMS[0], row_schema, static, "'t'")


def test_a_struct_key_with_an_unlaneable_field_refuses_by_name():
    """No lane exists for the TIMESTAMP field, so the struct cannot be
    keyed -- matrix (j) row 2, which used to serve the wrong rows."""
    sty = pa.struct([("mean", pa.float64()), ("t", pa.timestamp("us"))])
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("w", sty)]
    )
    static = _static_w(sty, {"mean": 1.0, "t": _T0})
    msg = _refuses(_FORMS[0], row_schema, static, "'w'")
    assert "t" in msg, msg


def test_a_struct_key_with_a_dotted_field_name_refuses_by_name():
    """`flatten_static` / `build_fields` skip a dotted field name (it would
    break the path encoding), so no lane exists -- matrix (j) row 3."""
    sty = pa.struct([("a.b", pa.float64()), ("c", pa.float64())])
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("w", sty)]
    )
    static = _static_w(sty, {"a.b": 1.0, "c": 2.0})
    _refuses(_FORMS[0], row_schema, static, "'w'")


def test_mismatched_struct_field_name_sets_refuse_by_name():
    """DuckDB answers a constant-empty join; we have no leaf to pair, and a
    deliberate severity-4 refusal is the decision (2026-08-25)."""
    ab = pa.struct([("a", pa.float64()), ("b", pa.float64())])
    xy = pa.struct([("x", pa.float64()), ("y", pa.float64())])
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("w", ab)]
    )
    _refuses(_FORMS[0], row_schema, _static_w(xy, {"x": 1.0, "y": 2.0}), "'w'")
    # a SUBSET is a mismatch too
    a = pa.struct([("a", pa.float64())])
    _refuses(_FORMS[0], row_schema, _static_w(a, {"a": 1.0}), "'w'")


def test_a_scalar_against_a_struct_refuses_by_name():
    """DuckDB refuses too (`Unimplemented type for cast`); we used to serve
    the join keyed on nothing at all -- matrix (j) row 6."""
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("v", pa.int64())]
    )
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "v": pa.array([{"mean": 1.0}], _S1),
            "z": pa.array([7], pa.int64()),
        }
    )
    _refuses("SELECT z AS o FROM __THIS__ NATURAL JOIN s", row_schema, static, "'v'")


def test_a_struct_key_serves_under_shape_map():
    """The decision of 2026-08-25: struct keys SERVE under the map and
    filter shapes (unique static keys, LEFT or inner)."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    static = _static_w(_S1, {"mean": 2.0})
    fn = DuckDBInferFn(
        _LEFT_FORMS[0],
        row_tables={"__THIS__": row.schema},
        static_tables={"s": static},
        shape="map",
    )
    assert fn.shape == "map"
    assert fn.infer_arrow(row).to_pylist() == [{"o": None}]


def test_a_struct_key_under_shape_many_refuses_naming_the_column():
    """The fan-out loop implements plain equality only (the pre-existing
    NOT-DISTINCT gap at lower.rs), so a struct key refuses there BY NAME
    rather than by the generic IS-NOT-DISTINCT-FROM message."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    msg = _refuses(
        _FORMS[0], row.schema, _static_w(_S1, {"mean": 1.0}), "'w'", shape="many"
    )
    assert "many" in msg, msg


def test_a_struct_on_one_side_only_is_not_a_key():
    """No shared NAME, so nothing to key on -- these already matched and
    must not move."""
    static = pa.table(
        {
            "id": pa.array([5], pa.int64()),
            "z": pa.array([7], pa.int64()),
        }
    )
    check(
        "SELECT z AS o FROM __THIS__ NATURAL JOIN s",
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}]),
        static,
        [{"o": 7}],
    )


# The lazy-minting contract -- a presence lane costs ~25 ns/row at the
# boundary, so an unjoined query over a struct-carrying row model must
# marshal exactly the lanes it did before -- is pinned on `program.in_cols`
# itself, in specializer/tests.rs (`presence_lanes_are_minted_lazily`).
