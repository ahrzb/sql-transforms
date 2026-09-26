"""NATURAL / USING join keys over STRUCT shared columns.

DuckDB's join binder intersects column NAME SETS with no type inspection
(bind_joinref.cpp:185-208), so a shared STRUCT is an ordinary join key there
and `=` on a struct is `row_matcher.cpp:379-382`: top-level Equals, every
child matched with NOT_DISTINCT_FROM. A struct key expands here into
composite ordinary keys -- one PLAIN presence key for the whole struct, one
NOT-DISTINCT presence key per nested node, one NOT-DISTINCT key per leaf --
which is that recursion written out.

Every test asserts THE ORACLE's answer first, so an oracle move is a loud
failure rather than a silent rebaseline. `confit.oracle.Oracle` puts
`PRAGMA disable_optimizer` on every connection it opens.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest
from confit import DuckDBInferFn
from confit.oracle import Oracle

_S1 = pa.struct([("mean", pa.float64())])
_S2 = pa.struct([("inner", pa.struct([("val", pa.float64())]))])
_S3 = pa.struct([("j", pa.struct([("v", pa.float64())]))])

_ROW_W = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _S1)])
_ROW_W2 = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _S2)])
_ROW_W3 = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _S3)])

# The THREE forms that must be told apart: NATURAL keys on ALL shared
# columns, USING only on the named ones
# (test_natural_and_using_key_on_different_column_sets shows they answer
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


def _oracle(sql, row_table, static):
    """DuckDB's own answer, over exactly the arrow data we hand the engine."""
    o = Oracle()
    o.load("__THIS__", row_table)
    o.load("s", static)
    res = o.execute(sql)
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
    got_oracle = _oracle(sql, row_table, static)
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


# --- struct STRUCT(mean DOUBLE) ---------------------------------------------

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


# --- nested structs, including THE DISCRIMINATORS ---------------------------

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


# --- float edges inside a field vs the scalar control -----------------------

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
    do for a bare DOUBLE column (canon_f64_bits)."""
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


# --- the LEFT legs keep the left-miss NULL shape ----------------------------


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


# --- NATURAL and USING key on DIFFERENT column sets -------------------------


def test_natural_and_using_key_on_different_column_sets():
    """`w` matches while `id` does not: NATURAL keys on both and misses,
    `USING (w)` keys on `w` alone and hits. Same fixture, different answers."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    static = _static_w(_S1, {"mean": 1.0}, sid=6)
    check(_FORMS[0], row, static, [])
    check(_FORMS[1], row, static, [])
    check(_FORMS[2], row, static, [{"o": 7}])


def test_natural_misses_when_only_the_struct_differs():
    """The wrong answer this file exists for: ids equal, `w` unequal, and
    keying on `id` alone would emit a row DuckDB never produces."""
    check(
        _FORMS[0],
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}]),
        _static_w(_S1, {"mean": 2.0}),
        [],
    )


# --- field order and name case ----------------------------------------------


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
    lanes, and a key column reconstructs from the dynamic side."""
    check(
        "SELECT s.w.mean AS o FROM __THIS__ NATURAL JOIN s",
        _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}]),
        _static_w(_S1, sw),
        want,
    )


def test_an_explicit_on_over_a_struct_still_refuses():
    """Scope boundary, pinned: only the NATURAL and USING arms key on a
    struct. `ON t.w = s.w` keeps the existing named refusal."""
    row = _row_table(_ROW_W, [{"id": 5, "w": {"mean": 1.0}}])
    sql = "SELECT z AS o FROM __THIS__ JOIN s ON __THIS__.w = s.w"
    assert _oracle(sql, row, _static_w(_S1, {"mean": 2.0})) == []
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
    values miss."""
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("W", _S1)]
    )
    check(
        sql,
        _row_table(row_schema, [{"id": 5, "W": {"mean": 1.0}}]),
        _static_w(_S1, {"mean": 2.0}),
        [],
    )


# --- the merge rule the struct head has to join ----------------------------


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


# --- the refuse-by-name backstop -------------------------------------------

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
    # The false claim the refusal must not make: the column plainly DOES exist.
    assert "does not exist" not in msg, msg
    return msg


# `(arrow type, [row value, static value], column)` -- two values means the
# sides DISAGREE, which is the cell that would answer wrongly if served.
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
    (a severity-4 refusal instead of a severity-2 wrong ANSWER).

    The TIMESTAMP leg is the first row: DuckDB keys on `t` and returns
    nothing, and keying on `id` alone would return the row.
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
    assert _oracle(sql, row, static) == ([] if len(vals) > 1 else [{"o": 7}])
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
    keyed; serving it would return the wrong rows."""
    sty = pa.struct([("mean", pa.float64()), ("t", pa.timestamp("us"))])
    row_schema = pa.schema(
        [pa.field("id", pa.int64(), nullable=False), pa.field("w", sty)]
    )
    static = _static_w(sty, {"mean": 1.0, "t": _T0})
    msg = _refuses(_FORMS[0], row_schema, static, "'w'")
    assert "t" in msg, msg


_DOT = pa.struct([("a.b", pa.float64()), ("c", pa.struct([("d.e", pa.float64())]))])


@pytest.mark.parametrize("sql", _FORMS)
@pytest.mark.parametrize(
    ("rw", "sw", "want"),
    [
        ({"a.b": 1.0, "c": {"d.e": 2.0}}, {"a.b": 1.0, "c": {"d.e": 2.0}}, [{"o": 7}]),
        ({"a.b": 1.0, "c": {"d.e": 2.0}}, {"a.b": 1.0, "c": {"d.e": 3.0}}, []),
        ({"a.b": None, "c": None}, {"a.b": None, "c": None}, [{"o": 7}]),
        ({"a.b": None, "c": None}, {"a.b": None, "c": {"d.e": None}}, []),
    ],
)
def test_a_struct_key_with_dotted_field_names_matches_duckdb(sql, rw, sw, want):
    """A field name is one path SEGMENT, dots and all."""
    row = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _DOT)])
    check(sql, _row_table(row, [{"id": 5, "w": rw}]), _static_w(_DOT, sw), want)


def test_mismatched_struct_field_name_sets_refuse_by_name():
    """DuckDB answers a constant-empty join; we have no leaf to pair, and a
    deliberate severity-4 refusal is the answer."""
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
    """DuckDB refuses too (`Unimplemented type for cast`); serving would key
    the join on nothing at all."""
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
    """Struct keys SERVE under the map and
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


# --- the minted lane's own NAME is user-reachable ---------------------------
#
# A minted presence lane is named `"<dotted path> (present)"`, and that
# synthetic name reaches users verbatim through the boundaries' "missing
# attribute" / "no field" refusals. Every ordinary struct shape SHADOWS it:
# the struct's leaf lanes come first in lane order and refuse under their own
# names. A struct NODE with no scalar leaf beneath it is the shape where the
# presence lane is the only lane that walks there, so it is what pins the
# name -- on all three ingest paths, which fill presence lanes separately.

_EMPTY = pa.struct([])
_ROW_WE = pa.schema([pa.field("id", pa.int64(), nullable=False), pa.field("w", _EMPTY)])
# a real leaf beside a leafless node, so the dotted JOIN of the path is pinned
# too and not just a one-segment name
_NESTED_E = pa.struct([("m", pa.float64()), ("a", _EMPTY)])
_ROW_WN = pa.schema(
    [pa.field("id", pa.int64(), nullable=False), pa.field("w", _NESTED_E)]
)


def _present_fn(row_schema, wtype, wval):
    return DuckDBInferFn(
        "SELECT z AS o FROM __THIS__ NATURAL JOIN s",
        row_tables={"__THIS__": row_schema},
        static_tables={"s": _static_w(wtype, wval)},
    )


# True pins the generic row boundary (the pre-marshaller baseline), which
# fills presence lanes in its own loop -- the env var is read at construction.
@pytest.mark.parametrize("generic", [False, True])
def test_a_minted_presence_lane_names_itself_at_the_row_boundary(monkeypatch, generic):
    if generic:
        monkeypatch.setenv("SPECIALIZER_GENERIC_BOUNDARY", "1")
    fn = _present_fn(_ROW_WE, _EMPTY, {})
    with pytest.raises(ValueError) as e:
        fn.infer_rows([{"id": 5}])
    assert str(e.value) == "Row for table '__THIS__' is missing attribute 'w (present)'"


@pytest.mark.parametrize("generic", [False, True])
def test_a_nested_presence_lane_dots_its_path_into_its_name(monkeypatch, generic):
    if generic:
        monkeypatch.setenv("SPECIALIZER_GENERIC_BOUNDARY", "1")
    fn = _present_fn(_ROW_WN, _NESTED_E, {"m": 1.0, "a": {}})
    with pytest.raises(ValueError) as e:
        fn.infer_rows([{"id": 5, "w": {"m": 1.0}}])
    assert (
        str(e.value) == "Row for table '__THIS__' is missing attribute 'w.a (present)'"
    )


def test_a_minted_presence_lane_names_itself_at_the_arrow_boundary():
    fn = _present_fn(_ROW_WN, _NESTED_E, {"m": 1.0, "a": {}})
    only_m = pa.struct([("m", pa.float64())])
    batch = pa.table(
        {"id": pa.array([5], pa.int64()), "w": pa.array([{"m": 1.0}], only_m)}
    )
    with pytest.raises(ValueError) as e:
        fn.infer_arrow(batch)
    assert str(e.value) == (
        "infer_arrow: column 'w.a (present)': the batch's struct 'w' has no field 'a'"
    )


# The lazy-minting contract -- a presence lane costs ~25 ns/row at the
# boundary, so an unjoined query over a struct-carrying row model must
# marshal exactly the lanes it did before -- is pinned on `program.in_cols`
# itself, in specializer/tests.rs (`presence_lanes_are_minted_lazily`).


def test_a_leafless_struct_key_refuses_a_non_struct_input_on_both_paths():
    # A struct with no leaf lanes is read only through its presence lane, so
    # nothing else would notice a scalar or a list where the schema declares
    # the struct: it used to count as a present node and join.
    wt = pa.struct([])
    row = pa.schema([pa.field("w", wt), pa.field("z", pa.int64())])
    s = pa.table({"w": pa.array([{}], wt), "v": pa.array([7], pa.int64())})
    fn = DuckDBInferFn(
        "SELECT z, v FROM __THIS__ NATURAL JOIN s",
        row_tables={"__THIS__": row},
        static_tables={"s": s},
    )
    good = pa.table({"w": pa.array([{}, None], wt), "z": [1, 2]})
    assert fn.infer_arrow(good).to_pylist() == [{"z": 1, "v": 7}]
    assert fn.infer_rows([{"w": {}, "z": 1}, {"w": None, "z": 2}]) == [{"z": 1, "v": 7}]
    bad = pa.table({"w": pa.array([5, None], pa.int64()), "z": [1, 2]})
    with pytest.raises(ValueError, match="the schema declares a struct"):
        fn.infer_arrow(bad)
    for v in (5, "x", [1]):
        with pytest.raises(ValueError, match="the schema declares a struct"):
            fn.infer_rows([{"w": v, "z": 1}])


# A number joined with a VARCHAR key: DuckDB casts the VARCHAR side to the
# numeric side's exact type (CAST's parse, so ' 3' matches 3). A static key
# that cannot convert errors on every query, zero request rows included; a
# request key that cannot convert errors on that row.
_MIX_R = pa.schema(
    [
        ("a", pa.int64()),
        ("k", pa.int64()),
        ("i", pa.int32()),
        ("x", pa.float64()),
        ("sk", pa.string()),
    ]
)
_MIX_ROWS = [
    {"a": 1, "k": 1, "i": 1, "x": 1.0, "sk": "1"},
    {"a": 2, "k": 2, "i": 2, "x": 2.5, "sk": " 2"},
    {"a": 3, "k": 3, "i": 3, "x": 3.0, "sk": "3"},
    {"a": 4, "k": None, "i": None, "x": None, "sk": None},
]
_MIX_D = pa.table(
    {
        "k": pa.array(["1", "2", " 3", None], pa.string()),
        "v": pa.array([10, 20, 30, 40], pa.int64()),
        "n": pa.array([1, 2, 3, 4], pa.int64()),
    }
)


def _mix_outcome(sql, statics, rows, **kw):
    o = Oracle()
    for n, t in statics.items():
        o.load(n, t)
    o.load("__THIS__", pa.Table.from_pylist(rows, schema=_MIX_R))
    try:
        w = o.answer(sql)
        want = (w.schema.types, sorted(map(str, w.to_pylist())))
    except Exception:  # noqa: BLE001 -- any DuckDB error is the outcome
        want = "ERR"
    try:
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": _MIX_R}, static_tables=statics, **kw
        )
        got = (fn.output_schema.types, sorted(map(str, fn.infer_rows(rows))))
    except ValueError:
        got = "ERR"
    return got, want


@pytest.mark.parametrize(
    ("sql", "shape"),
    [
        ("SELECT a, v FROM __THIS__ JOIN d ON __THIS__.k = d.k", None),
        ("SELECT a, v FROM __THIS__ LEFT JOIN d ON __THIS__.k = d.k", None),
        ("SELECT a, v FROM __THIS__ JOIN d USING (k)", None),
        ("SELECT * FROM __THIS__ JOIN d USING (k)", None),
        ("SELECT a, v FROM __THIS__ JOIN d ON __THIS__.i = d.k", None),
        ("SELECT a, v FROM __THIS__ JOIN d ON __THIS__.x = d.k", None),
        ("SELECT a, v FROM __THIS__ JOIN d ON __THIS__.sk = d.n", None),
        ("SELECT a, v FROM __THIS__ LEFT JOIN d ON __THIS__.sk = d.n", None),
        ("SELECT a, v FROM __THIS__ JOIN d ON __THIS__.k = d.k", "many"),
        ("SELECT a, v FROM __THIS__ JOIN d ON __THIS__.sk = d.n", "many"),
    ],
)
def test_a_number_joins_a_varchar_key_as_duckdb_casts_it(sql, shape):
    kw = {"shape": shape} if shape else {}
    got, want = _mix_outcome(sql, {"d": _MIX_D}, _MIX_ROWS, **kw)
    assert got == want


def test_a_request_varchar_key_that_cannot_convert_errors_like_duckdb():
    rows = [*_MIX_ROWS, {"a": 5, "k": 5, "i": 5, "x": 5.0, "sk": "zz"}]
    got, want = _mix_outcome(
        "SELECT a, v FROM __THIS__ JOIN d ON __THIS__.sk = d.n", {"d": _MIX_D}, rows
    )
    assert got == want == "ERR"


@pytest.mark.parametrize(
    ("probe", "bad"),
    [("k", "zz"), ("x", "zz"), ("i", "3000000000")],
)
def test_a_static_varchar_key_that_cannot_convert_refuses_at_build(probe, bad):
    # DuckDB errors on every query, with no request rows at all; the probe's
    # declared width decides (INTEGER: '3000000000' does not fit).
    e = pa.table(
        {"k": pa.array(["1", bad], pa.string()), "v": pa.array([10, 99], pa.int64())}
    )
    sql = f"SELECT a, v FROM __THIS__ JOIN e ON __THIS__.{probe} = e.k"
    for rows in ([], _MIX_ROWS):
        got, want = _mix_outcome(sql, {"e": e}, rows)
        assert got == want == "ERR"


# --- a static struct LEAF as an ON key ---------------------------------------
#
# `k = v.x` keys the join on the leaf lane, exactly like `k = id`: through
# the struct head, the relation, its alias, or its schema. Before, a leaf was
# never a key candidate, the condition went residual, and any static table of
# two or more rows refused as a "duplicate map key".

_LEAF_ROW = pa.schema(
    [
        pa.field("k", pa.int64()),
        pa.field("w", pa.struct([("x", pa.int64())])),
        pa.field("x", pa.int64()),
    ]
)
_LEAF_ROWS = [
    {"k": 1, "w": {"x": 1}, "x": 7},
    {"k": 2, "w": {"x": 9}, "x": 8},
    {"k": None, "w": None, "x": None},
    {"k": 4, "w": None, "x": 1},
]
_LEAF_D = pa.table(
    {
        "id": pa.array([1, 2, 3, 4], pa.int64()),
        "v": pa.array(
            [{"x": 1, "y": 5}, {"x": 2, "y": 6}, None, {"x": None, "y": 1}],
            pa.struct([("x", pa.int64()), ("y", pa.int64())]),
        ),
        "w": pa.array([{"x": 1}, {"x": 2}, None, None], pa.struct([("x", pa.int64())])),
        "z": pa.array([10, 20, 30, 40], pa.int64()),
    }
)


def _leaf_parity(sql, **kw):
    o = Oracle()
    o.load("__THIS__", pa.Table.from_pylist(_LEAF_ROWS, schema=_LEAF_ROW))
    o.load("d", _LEAF_D)
    res = o.execute(sql)
    names = [c[0] for c in res.description]
    want = [dict(zip(names, r, strict=True)) for r in res.fetchall()]
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": _LEAF_ROW}, static_tables={"d": _LEAF_D}, **kw
    )
    got = fn.infer_rows(_LEAF_ROWS)
    key = lambda r: repr(sorted(r.items()))  # noqa: E731
    assert sorted(got, key=key) == sorted(want, key=key), sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT z FROM __THIS__ JOIN d ON k = v.x",
        "SELECT z FROM __THIS__ JOIN d ON v.x = k",
        "SELECT z FROM __THIS__ JOIN d ON k = d.v.x",
        "SELECT z FROM __THIS__ JOIN d ON k = main.d.v.x",
        "SELECT z FROM __THIS__ JOIN d ON k = memory.main.d.v.x",
        "SELECT z FROM __THIS__ JOIN d AS q ON k = q.v.x",
        # `v` is the relation's alias with no column `v.x`: the struct head
        "SELECT z FROM __THIS__ JOIN d AS v ON k = v.x",
        "SELECT z, v.x, v.y FROM __THIS__ LEFT JOIN d ON k = v.x",
        "SELECT z FROM __THIS__ JOIN d ON k = v.x AND v.y > 0",
        "SELECT z FROM __THIS__ JOIN d ON k + 1 = v.y",
        "SELECT z FROM __THIS__, d WHERE k = v.x",
        # the scalar spelling through the schema keys too
        "SELECT z FROM __THIS__ JOIN d ON k = main.d.id",
    ],
)
def test_a_static_struct_leaf_keys_the_join(sql):
    _leaf_parity(sql)


@pytest.mark.parametrize(
    "sql",
    [
        # `v` is the ROW table here, so `v.x` is its column: no static key
        "SELECT z FROM __THIS__ AS v JOIN d ON k = v.x",
        "SELECT z FROM __THIS__ JOIN d ON v.x = v.y",
        "SELECT z FROM __THIS__ JOIN d ON v.x = d.id",
    ],
)
def test_a_join_with_no_equality_key_names_that_cause(sql):
    with pytest.raises(ValueError, match="has no equality key") as e:
        DuckDBInferFn(
            sql, row_tables={"__THIS__": _LEAF_ROW}, static_tables={"d": _LEAF_D}
        )
    assert "duplicate map key" not in str(e.value)
    _leaf_parity(sql, shape="many")


def test_a_struct_leaf_head_in_both_relations_is_ambiguous():
    # DuckDB: Ambiguous reference to column name "w"
    with pytest.raises(ValueError, match="ambiguous column 'w'"):
        DuckDBInferFn(
            "SELECT z FROM __THIS__ JOIN d ON k = w.x",
            row_tables={"__THIS__": _LEAF_ROW},
            static_tables={"d": _LEAF_D},
        )
