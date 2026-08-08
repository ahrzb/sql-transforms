"""TASK-68: a generated parameter name may never mean a user's relation.

``_plan`` mints parameter names from user-controlled strings — a CTE key
becomes ``__param_{key}``. Two ways that went wrong, both silent, both
breaking *freezing is faithful*:

* a CTE named ``fit`` claimed ``__param_fit``, and ``whole_fit()`` read the
  collision as *already emitted* rather than *name taken*, so a bare
  ``FROM __FIT__`` was aliased onto the CTE's table;
* a CTE named ``__FIT__`` shadowed the parameter for DuckDB but not for us,
  so the rewrite pointed it at the training set and the row count changed.

``whole_fit()`` is gone — a bare ``FROM __FIT__`` beside ``__THIS__`` refuses
rather than shipping the training set — so the first bug's *mechanism* cannot
recur. The property it was an instance of still can: several producers mint
names from one pool, and correlation lifting added two more per subquery. The
shapes below keep that pressure on.

``run`` is the reference throughout: it binds both parameters to the same
relation with no freezing at all, so any disagreement is the plan's fault.
"""

import pyarrow as pa
import pytest

from sql_transform.model import SQLTransform, TransformError, run

D = pa.table({"price": [1.0, 2.0]})


# ------------------------------------------------- a CTE named `fit` (F1)


def test_a_cte_named_fit_does_not_capture_the_training_sets_parameter():
    """The whole of F1 in one comparison.

    ``WITH fit AS (...)`` is an unremarkable name in a library whose two
    parameters are ``__FIT__`` and ``__THIS__``. Before the fix the frozen
    CTE's table was served in place of the other step's: same shape, same
    column names, hundred-fold different numbers, no error.
    """
    t = SQLTransform(
        "WITH fit AS (SELECT price * 100 AS price FROM __FIT__) "
        "SELECT t.price AS live, (SELECT sum(price) FROM fit) AS trained, "
        "(SELECT avg(f.price) FROM __FIT__ f WHERE f.price = t.price) AS m "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pydict() == run(t, D).to_pydict()


def test_every_producer_draws_from_one_pool_under_distinct_names():
    """The mechanism, pinned separately from the symptom.

    ``whole_fit()`` guarded with ``if "__param_fit" not in taken`` — one name
    pool, two producers, and only ``freeze`` treated a hit as a collision. A
    name being taken must never be read as *my step is already there*. Three
    producers now: the CTE freeze, and the keyed table and empty-input probe
    that correlation lifting emits.
    """
    t = SQLTransform(
        "WITH fit AS (SELECT price * 100 AS price FROM __FIT__) "
        "SELECT t.price AS live, (SELECT sum(price) FROM fit) AS trained, "
        "(SELECT avg(f.price) FROM __FIT__ f WHERE f.price = t.price) AS m "
        "FROM __THIS__ t"
    )
    names = [name for name, _ in t._steps]
    assert len(names) == 3
    assert len(set(names)) == 3
    assert len(t.fit(D).params) == 3


@pytest.mark.parametrize("cte", ["fit", "0", "fit_1", "THIS"])
def test_a_cte_may_be_named_anything_without_changing_the_answer(cte):
    """The general property, not the one name that bit us.

    ``__param_0`` is what an unnamed frozen subtree gets, so a CTE named ``0``
    aims at the same pool from the other direction.
    """
    t = SQLTransform(
        f'WITH "{cte}" AS (SELECT price * 100 AS price FROM __FIT__) '
        "SELECT t.price AS live, "
        f'(SELECT sum(price) FROM "{cte}") AS s, '
        "(SELECT avg(f.price) FROM __FIT__ f WHERE f.price = t.price) AS m "
        "FROM __THIS__ t"
    )
    assert t.fit(D).transform(D).to_pydict() == run(t, D).to_pydict()


def test_generated_names_stay_distinct_across_many_colliding_ctes():
    t = SQLTransform(
        "WITH fit AS (SELECT price FROM __FIT__), "
        '     "0" AS (SELECT price FROM fit), '
        '     "1" AS (SELECT price FROM "0") '
        'SELECT t.price AS live, (SELECT sum(price) FROM "1") AS s, '
        "(SELECT avg(f.price) FROM __FIT__ f WHERE f.price = t.price) AS m "
        "FROM __THIS__ t"
    )
    names = [name for name, _ in t._steps]
    assert len(names) == len(set(names))
    assert t.fit(D).transform(D).to_pydict() == run(t, D).to_pydict()


# ---------------------------------------------- a CTE named __FIT__ (F2)


@pytest.mark.parametrize("name", ["__FIT__", "__THIS__", "__fit__", "__This__"])
def test_a_cte_may_not_shadow_a_parameter(name):
    """Refused at construction, by name (P7).

    DuckDB would let the CTE win; we rewrote the reference to the training set
    instead, which changed the row count with no error. Rather than teach the
    walk to honour the shadowing — which would make ``__FIT__`` mean two
    things in one text — the name is refused where it is defined.
    """
    with pytest.raises(TransformError, match="may not be named"):
        SQLTransform(
            f'WITH "{name}" AS (SELECT 100.0 AS price) '
            f'SELECT t.price + f.price AS v FROM __THIS__ t, "{name}" f'
        )


def test_shadowing_is_refused_before_any_data_exists():
    """Construction, not fit — there is no relation in sight when it raises."""
    with pytest.raises(TransformError, match="may not be named"):
        SQLTransform("WITH __FIT__ AS (SELECT 1 AS x) SELECT * FROM __FIT__")


def test_an_ordinary_cte_name_is_still_fine():
    t = SQLTransform(
        "WITH fitted AS (SELECT avg(price) m FROM __FIT__) "
        "SELECT t.price / fitted.m AS z FROM __THIS__ t, fitted"
    )
    assert t.fit(D).transform(D).to_pydict() == run(t, D).to_pydict()
