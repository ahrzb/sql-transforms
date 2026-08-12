"""SQLProjection: one output row per ``__THIS__`` row, from that row and params.

The refusal cases are a table (`REFUSED`), and `test_every_reason_is_exercised`
walks `REASONS` looking for gaps, the way `_correlate_test` does for its own —
a reason nothing exercises is a refusal nobody has named.

Implements the gates of
`docs/superpowers/specs/2026-08-11-row-wise-projections-design.md`.
"""

import pyarrow as pa
import pytest

from sql_transform.model import NotRowWise, run
from sql_transform.model._projection import REASONS, SQLProjection

F = pa.table(
    {
        "store": ["S1", "S1", "S1", "S2", "S2", "S2"],
        "price": [10.0, 20.0, 30.0, 100.0, 300.0, 500.0],
    }
)
# Shuffled on purpose, with an unseen store in the middle: row order and the
# NULL for a key fit never saw are both part of what `transform` promises.
X = pa.table(
    {
        "store": ["S2", "NEW", "S1"],
        "price": [200.0, 7.0, 10.0],
    }
)

KEYED = """
    SELECT t.store, t.price / f.m AS z
    FROM __THIS__ t
    LEFT JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
      ON t.store = f.store
"""

GLOBAL = """
    SELECT t.price / f.m AS z
    FROM __THIS__ t, (SELECT avg(price) AS m FROM __FIT__) f
"""


# ---------------------------------------------------------------- the laws


def test_rows_and_order_with_an_unseen_key():
    out = SQLProjection(KEYED).fit(F).transform(X)
    assert out.num_rows == X.num_rows
    assert out.column_names == ["store", "z"]
    assert out.to_pylist() == [
        {"store": "S2", "z": 200.0 / 300.0},
        {"store": "NEW", "z": None},
        {"store": "S1", "z": 0.5},
    ]


def test_solo_batch_equals_concatenation_of_single_rows():
    fitted = SQLProjection(KEYED).fit(F)
    batch = fitted.transform(X).to_pylist()
    solo = [
        row
        for i in range(X.num_rows)
        for row in fitted.transform(X.slice(i, 1)).to_pylist()
    ]
    assert batch == solo


def test_faithful_fit_transform_equals_run():
    p = SQLProjection(GLOBAL)
    frozen = p.fit(F).transform(F).to_pylist()
    both = run(p, F).to_pylist()
    key = lambda r: tuple((v is None, v) for v in r.values())  # noqa: E731
    assert sorted(frozen, key=key) == sorted(both, key=key)


def test_cross_join_to_a_one_row_params_table():
    out = SQLProjection(GLOBAL).fit(F).transform(X)
    assert [r["z"] for r in out.to_pylist()] == [
        200.0 / 160.0,
        7.0 / 160.0,
        10.0 / 160.0,
    ]


def test_params_are_inspectable_and_fit_is_gone_from_the_text():
    fitted = SQLProjection(KEYED).fit(F)
    assert all(isinstance(t, pa.Table) for t in fitted.params.values())
    assert {len(t) for t in fitted.params.values()} == {2}  # two stores
    assert "__FIT__" not in fitted.sql


def test_a_lifted_correlated_subquery_still_serves():
    """The equality lifting is upstream of the gate; its LEFT JOIN passes."""
    p = SQLProjection("""
        SELECT t.store,
               (SELECT avg(f.price) FROM __FIT__ f WHERE f.store = t.store) AS m
        FROM __THIS__ t
    """)
    out = p.fit(F).transform(X)
    assert [r["m"] for r in out.to_pylist()] == [300.0, None, 20.0]


def test_the_input_order_is_the_output_order_not_the_join_order():
    """A LEFT JOIN emits unmatched rows last; the threaded ordinal wins."""
    fitted = SQLProjection(KEYED).fit(F)
    first = fitted.transform(
        pa.table({"store": ["NEW", "S1"], "price": [1.0, 10.0]})
    ).to_pylist()
    assert first[0]["store"] == "NEW"


# ------------------------------------------------------------- what refuses

INNER_KEYED = """
    SELECT t.price - f.m AS d
    FROM __THIS__ t
    JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
      ON t.store = f.store
"""

THIS_ON_THE_RIGHT = """
    SELECT t.price - f.m AS d
    FROM (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
    LEFT JOIN __THIS__ t ON t.store = f.store
"""

FULL_OUTER = """
    SELECT t.price - f.m AS d
    FROM __THIS__ t
    FULL JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
      ON t.store = f.store
"""

REFUSED = [
    ("aggregate", "SELECT sum(t.price) AS s FROM __THIS__ t"),
    (
        "window",
        "SELECT avg(t.price) OVER (PARTITION BY t.store) AS m FROM __THIS__ t",
    ),
    ("group-by", "SELECT t.store, count(*) AS c FROM __THIS__ t GROUP BY t.store"),
    ("group-by", "SELECT t.store, count(*) AS c FROM __THIS__ t GROUP BY ALL"),
    ("modifier", "SELECT DISTINCT t.store AS s FROM __THIS__ t"),
    ("modifier", "SELECT t.price AS p FROM __THIS__ t ORDER BY t.price"),
    ("modifier", "SELECT t.price AS p FROM __THIS__ t LIMIT 2"),
    ("filter", "SELECT t.price AS p FROM __THIS__ t WHERE t.price > 0"),
    (
        "filter",
        "SELECT t.price AS p FROM __THIS__ t QUALIFY row_number() OVER () = 1",
    ),
    ("this-twice", "SELECT a.price - b.price AS d FROM __THIS__ a, __THIS__ b"),
    ("set-operation", "SELECT t.price AS p FROM __THIS__ t UNION ALL SELECT 1"),
    (
        "recursive-cte",
        "WITH RECURSIVE r AS ("
        "  SELECT t.price AS p FROM __THIS__ t"
        "  UNION ALL SELECT p + 1 FROM r WHERE p < 100"
        ") SELECT p FROM r",
    ),
    ("join", INNER_KEYED),
    ("join", THIS_ON_THE_RIGHT),
    ("join", FULL_OUTER),
    # __THIS__ read from an expression: the output rows are the params rows,
    # so nothing tracks the batch.
    (
        "spine",
        "SELECT (SELECT max(t.price) FROM __THIS__ t) AS m"
        " FROM (SELECT avg(price) AS m0 FROM __FIT__) f",
    ),
    # __THIS__ never read at all: one row out whatever the batch is.
    ("spine", "SELECT f.m FROM (SELECT avg(price) AS m FROM __FIT__) f"),
    # a correlated subquery over __THIS__ reads the *other* rows of the batch.
    (
        "spine",
        "SELECT (SELECT sum(o.price) FROM __THIS__ o WHERE o.store = t.store) AS s"
        " FROM __THIS__ t",
    ),
]


@pytest.mark.parametrize(("reason", "sql"), REFUSED)
def test_refused_at_construction(reason, sql):
    with pytest.raises(NotRowWise) as e:
        SQLProjection(sql)
    assert e.value.reason == reason, e.value


def test_every_reason_is_exercised():
    assert {reason for reason, _ in REFUSED} == set(REASONS)


def test_the_refusal_names_the_offending_expression():
    with pytest.raises(NotRowWise, match=r"sum\(t\.price\)"):
        SQLProjection("SELECT sum(t.price) AS s FROM __THIS__ t")


# --------------------------------------------------- what deliberately builds


def test_params_only_levels_are_free():
    """Aggregates, ORDER BY and LIMIT over params are constant at serving —
    the gate only constrains the levels that carry the batch's rows."""
    p = SQLProjection("""
        WITH s AS (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store)
        SELECT t.price / (SELECT max(m) FROM (SELECT m FROM s ORDER BY m LIMIT 2)) AS z
        FROM __THIS__ t
    """)
    out = p.fit(F).transform(X)
    assert out.num_rows == X.num_rows


def test_a_where_inside_a_params_relation_is_free():
    p = SQLProjection("""
        SELECT t.price / f.m AS z
        FROM __THIS__ t,
             (SELECT avg(price) AS m FROM __FIT__ WHERE price > 15) f
    """)
    assert p.fit(F).transform(X).num_rows == X.num_rows


# ------------------------------------------------- what refuses at fit


def test_a_join_without_group_by_refuses_at_fit_naming_the_key():
    """The measured case: four training rows in the artifact, one serving row
    became three, no error — under SQLTransform. Here it refuses, with the
    repeating key and its count straight out of the probe."""
    from sql_transform.model import KeyNotUnique

    p = SQLProjection("""
        WITH s AS (SELECT store, price AS m FROM __FIT__)
        SELECT t.price / s.m AS z FROM __THIS__ t LEFT JOIN s ON t.store = s.store
    """)
    with pytest.raises(KeyNotUnique, match=r"S1.*3 rows.*become 3"):
        p.fit(F)


def test_a_multi_row_relation_beside_this_refuses_at_fit():
    from sql_transform.model import KeyNotUnique

    p = SQLProjection("""
        SELECT t.price - f.price AS d
        FROM __THIS__ t, (SELECT store, price FROM __FIT__ WHERE price > 15) f
    """)
    with pytest.raises(KeyNotUnique, match=r"5 rows"):
        p.fit(F)


def test_an_empty_relation_beside_this_refuses_at_fit():
    """Cross join to zero rows makes every serving row disappear — the other
    way to break the row count, and quieter."""
    from sql_transform.model import KeyNotUnique

    p = SQLProjection("""
        SELECT t.price - f.price AS d
        FROM __THIS__ t, (SELECT store, price FROM __FIT__ WHERE price > 999) f
    """)
    with pytest.raises(KeyNotUnique, match=r"0 rows"):
        p.fit(F)


def test_de_dup_idioms_pass_the_measurement():
    """DISTINCT and QUALIFY row_number are correct de-dup spellings; a syntax
    rule would refuse them, the measurement does not."""
    distinct = SQLProjection("""
        SELECT t.price / f.price AS r
        FROM __THIS__ t
        LEFT JOIN (SELECT DISTINCT store, first(price) OVER (
            PARTITION BY store ORDER BY price) AS price FROM __FIT__) f
          ON t.store = f.store
    """)
    assert distinct.fit(F).transform(X).num_rows == X.num_rows

    qualified = SQLProjection("""
        SELECT t.price / f.price AS r
        FROM __THIS__ t
        LEFT JOIN (SELECT store, price FROM __FIT__
                   QUALIFY row_number() OVER (PARTITION BY store ORDER BY price) = 1) f
          ON t.store = f.store
    """)
    assert qualified.fit(F).transform(X).num_rows == X.num_rows


def test_null_keyed_duplicates_refuse_conservatively():
    """GROUP BY folds NULL keys into one group, which is exact for the INDF
    joins the model emits and conservative for an author-written `=` join —
    the safe direction (spec: the KeyNotUnique section)."""
    from sql_transform.model import KeyNotUnique

    nulls = pa.table({"store": ["S1", None, None], "price": [10.0, 5.0, 7.0]})
    p = SQLProjection("""
        SELECT t.price / f.m AS z
        FROM __THIS__ t
        LEFT JOIN (SELECT store, price AS m FROM __FIT__) f ON t.store = f.store
    """)
    with pytest.raises(KeyNotUnique, match=r"2 rows"):
        p.fit(nulls)


def test_a_unique_key_with_extra_conjuncts_passes():
    """Extra non-equality conjuncts only filter matches further; uniqueness
    over the equality keys already bounds them at one."""
    p = SQLProjection("""
        SELECT t.price / f.m AS z
        FROM __THIS__ t
        LEFT JOIN (SELECT store, avg(price) AS m FROM __FIT__ GROUP BY store) f
          ON t.store = f.store AND f.m > 0
    """)
    assert p.fit(F).transform(X).num_rows == X.num_rows


def test_the_reserved_row_name_is_already_unwritable():
    """P8 owns the threading column: an author cannot collide with it."""
    from sql_transform.model import TransformError

    with pytest.raises(TransformError, match="reserved"):
        SQLProjection("SELECT t.price AS __cf_row FROM __THIS__ t")
