"""Slice 2 gates: calling — nesting, chaining, per group.

Members splice. The oracle for a nested call is a *registered table function*,
and since DuckDB's Python API cannot register one, its materialized output
stands in — computed in pyarrow, sharing no engine, no control flow and no
code with the SQL path, so it cannot agree by sharing a bug.
"""

import gc
import weakref

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from sql_transform.model import (
    NestingTooDeep,
    SQLTransform,
    UnknownName,
    run,
)
from sql_transform.model._freezing_test import approx, reads_fit, rows

F = pa.table(
    {
        "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
        "store": ["S1"] * 3 + ["S2"] * 3,
    }
)
# A serving batch that is *not* the training data. Without it, binding the two
# parameters the wrong way round is invisible: measured, a swapped splice
# passes every gate below when fit and transform are handed the same table.
S = pa.table(
    {
        "price": [1.0, 2.0, 7.0, 13.0, 17.0, 19.0],
        "store": ["S1"] * 3 + ["S2"] * 3,
    }
)

# Members live at module scope so the caller's frame finds them in globals.
z = SQLTransform(
    "SELECT t.store, t.price / s.m AS z "
    "FROM __THIS__ t, (SELECT avg(price) m FROM __FIT__) s"
)
a = SQLTransform(
    "SELECT t.store, t.price - s.lo AS price "
    "FROM __THIS__ t, (SELECT min(price) lo FROM __FIT__) s"
)


def z_ref(fit_t: pa.Table, this_t: pa.Table) -> pa.Table:
    m = pc.mean(fit_t["price"]).as_py()
    return pa.table({"store": this_t["store"], "z": pc.divide(this_t["price"], m)})


def a_ref(fit_t: pa.Table, this_t: pa.Table) -> pa.Table:
    lo = pc.min(fit_t["price"]).as_py()
    return pa.table(
        {"store": this_t["store"], "price": pc.subtract(this_t["price"], lo)}
    )


# ------------------------------------------------------------------- nesting


def test_a_member_splices_and_matches_the_reference():
    outer = SQLTransform("SELECT * FROM z(__FIT__, __THIS__) s")
    assert approx(outer.fit(F)(S), 4) == approx(z_ref(F, S), 4)
    assert approx(z_ref(F, S), 4) != approx(z_ref(S, F), 4)  # order is visible


def test_splicing_equals_materializing():
    """The nesting oracle: a call behaves as if the member were registered.

    A table function call *is* its materialized output, so compute the member
    in pyarrow, register the result, and run the outer query over that table.
    The outer deliberately reuses the member's own CTE name.
    """
    # The member defines a CTE named `cs`; so does the outer, meaning
    # something else. A parenthesised derived table already scopes its own
    # definitions, so this must not collide.
    member = SQLTransform(
        "WITH cs AS (SELECT avg(price) m FROM __FIT__) "
        "SELECT t.store, t.price / cs.m AS z FROM __THIS__ t, cs"
    )
    assert member is not None
    outer_sql = (
        "WITH cs AS (SELECT 0.05 AS threshold) "
        "SELECT s.store, round(s.z, 4) AS z FROM {member} s, cs "
        "WHERE s.z > cs.threshold ORDER BY s.store, z"
    )
    spliced = SQLTransform(outer_sql.format(member="member(__FIT__, __THIS__)"))

    con = duckdb.connect()
    con.register("z_out", z_ref(F, S))
    expected = con.execute(outer_sql.format(member="z_out")).fetchall()

    assert rows(spliced.fit(F)(S)) == expected
    assert len(expected) > 0  # the gate would be vacuous on an empty result


def test_splicing_is_capture_free():
    """An outer CTE cannot rebind a name the member resolved from its frame.

    Measured before the fix: the member saw the outer's ``scale`` and returned
    [10000, 20000, 30000]. Silent, no error, different numbers.
    """
    scale = pa.table({"factor": [2.0]})
    doubler = SQLTransform("SELECT t.price * scale.factor AS z FROM __THIS__ t, scale")
    assert scale is not None and doubler is not None  # bound in this frame

    hostile = SQLTransform(
        "WITH scale AS (SELECT 1000.0 AS factor) "
        "SELECT * FROM doubler(__FIT__, __THIS__) s ORDER BY z"
    )
    batch = pa.table({"price": [10.0, 20.0, 30.0]})
    assert approx(hostile.fit(batch)(batch)) == [(20.0,), (40.0,), (60.0,)]


def test_splicing_is_text_checkable():
    """Spliced SQL equals a hand-written equivalent — it says *where* it broke."""
    twice = SQLTransform("SELECT price * 2 AS z FROM __THIS__")
    assert twice is not None
    spliced = SQLTransform("SELECT * FROM twice(__FIT__, __THIS__) s")
    expected = SQLTransform(
        "SELECT * FROM (SELECT price * 2 AS z FROM __THIS__ AS __THIS__) s"
    )
    assert spliced.sql == expected.sql


def test_members_splice_never_macro():
    outer = SQLTransform("SELECT * FROM z(__FIT__, __THIS__) s")
    assert "MACRO" not in outer.sql.upper()


# ------------------------------------------------------------------ chaining


def test_chaining_fits_on_transformed_data():
    """``b``'s fit input is ``a(F, F)``, not ``F``. Each stage appears twice."""
    chained = SQLTransform(
        "SELECT * FROM z(a(__FIT__, __FIT__), a(__FIT__, __THIS__)) s"
    )
    # The spec's pinned numbers, training data served back to itself.
    right = approx(z_ref(a_ref(F, F), a_ref(F, F)), 4)
    wrong = approx(z_ref(F, a_ref(F, F)), 4)
    assert approx(chained.fit(F)(F), 4) == right
    assert right[:3] == [("S1", 0.0), ("S1", 0.0667), ("S1", 0.1333)]
    assert wrong[:3] == [("S1", 0.0), ("S1", 0.0625), ("S1", 0.125)]
    assert right != wrong  # the data can tell the two readings apart

    # And on a serving batch, where each stage's two appearances differ.
    assert approx(chained.fit(F)(S), 4) == approx(z_ref(a_ref(F, F), a_ref(F, S)), 4)


# ----------------------------------------------------------------- per group

PER_GROUP = """
SELECT x.* FROM (SELECT DISTINCT store FROM __FIT__) g,
  LATERAL z((FROM __FIT__  WHERE store = g.store),
            (FROM __THIS__ WHERE store = g.store)) x
"""


def _per_group_expected(this_t: pa.Table) -> list[tuple]:
    groups = sorted(set(F["store"].to_pylist()))
    parts = [
        z_ref(
            F.filter(pc.equal(F["store"], g)),
            this_t.filter(pc.equal(this_t["store"], g)),
        )
        for g in groups
    ]
    return sorted(approx(pa.concat_tables(parts), 4))


def test_per_group_matches_a_python_loop():
    """The member needs no notion of the group; slicing both parameters is all."""
    fitted = SQLTransform(PER_GROUP).fit(F)
    assert sorted(approx(fitted(S), 4)) == _per_group_expected(S)
    assert sorted(approx(fitted(F), 4)) == _per_group_expected(F)


def test_per_group_is_faithful_under_run():
    t = SQLTransform(PER_GROUP)
    assert sorted(approx(run(t, F), 4)) == _per_group_expected(F)


def test_per_group_costs_one_row_per_group_not_one_per_training_row():
    """This used to retain the whole training set, and it is the shape the
    guide teaches — so it was the loudest argument for marginalising.

    The correlating predicate arrives a level below the aggregate, inside the
    derived table the splice built out of the argument. Flattened, it is an
    ordinary type-JA and Kim's temporary relation applies: two stores, two
    means, one empty-input probe, and not a single training row.
    """
    fitted = SQLTransform(PER_GROUP).fit(F)
    assert not reads_fit(fitted.sql)
    groups = len(set(F["store"].to_pylist()))
    assert sum(len(p) for p in fitted.params.values()) == 2 * groups + 1
    assert 2 * groups + 1 < len(F)


def test_both_parameters_must_be_sliced():
    """Filtering only ``__FIT__`` leaves ``__THIS__`` whole: every group crosses
    every row. Measured: 4 rows in, 8 out, with S1's statistics on S2's rows."""
    small = pa.table(
        {"price": [10.0, 30.0, 100.0, 300.0], "store": ["S1"] * 2 + ["S2"] * 2}
    )
    half = SQLTransform("""
        SELECT x.* FROM (SELECT DISTINCT store FROM __FIT__) g,
          LATERAL z((FROM __FIT__ WHERE store = g.store), __THIS__) x
    """)
    both = SQLTransform(PER_GROUP)
    assert len(run(half, small)) == 8
    assert len(run(both, small)) == 4


# ----------------------------------------------------------- name resolution


def test_capture_is_by_value():
    inner = SQLTransform("SELECT price * 2 AS z FROM __THIS__")
    assert inner is not None
    outer = SQLTransform("SELECT * FROM inner(__FIT__, __THIS__) s")
    inner = SQLTransform("SELECT price * 100 AS z FROM __THIS__")  # noqa: F841
    batch = pa.table({"price": [10.0, 20.0, 30.0]})
    assert approx(outer.fit(batch)(batch)) == [(20.0,), (40.0,), (60.0,)]


def test_no_frame_is_retained():
    """Construction holds no reference to the caller's frame."""

    class Marker:
        pass

    def build():
        marker = Marker()  # only the frame could keep this alive
        tbl = pa.table({"x": [1]})
        assert tbl is not None
        return SQLTransform("SELECT * FROM tbl"), weakref.ref(marker)

    transform, ref = build()
    gc.collect()
    assert ref() is None
    assert transform is not None


def test_unknown_name_refuses():
    with pytest.raises(UnknownName, match="no_such_thing"):
        SQLTransform("SELECT * FROM no_such_thing(__FIT__, __THIS__) s")
    with pytest.raises(UnknownName, match="no_such_table"):
        SQLTransform("SELECT * FROM no_such_table")


def _chain(depth: int) -> SQLTransform:
    """``depth`` nested member calls, each name bound in globals."""
    globals()["lvl0"] = SQLTransform("SELECT price FROM __THIS__")
    for i in range(1, depth + 1):
        globals()[f"lvl{i}"] = SQLTransform(
            f"SELECT * FROM lvl{i - 1}(__FIT__, __THIS__) s"
        )
    return globals()[f"lvl{depth}"]


def test_depth_cap_holds():
    assert _chain(8) is not None
    with pytest.raises(NestingTooDeep, match="8"):
        _chain(9)
