"""Differential parity for SQL identifier folding (TASK-28).

DataFusion is the oracle and follows ANSI folding: an UNQUOTED identifier folds
to lowercase before resolution, a double-QUOTED one stays case-exact. The serving
engines must match bug-for-bug -- i.e. `SELECT Age` on a column named `Age` must
fold to `age` and MISS on every engine, just as it does on DataFusion, and only
`"Age"` resolves. Each case runs once per backend via the conftest fixture.
"""

from differential import check, check_both_raise, rows

_R = "__THIS__"
_AGE = {_R: rows({"Age": "float"}, [{"Age": 10.0}])}
_age = {_R: rows({"age": "float"}, [{"age": 10.0}])}


def test_unquoted_camelcase_column_folds_and_misses():
    # `Age` folds to `age`; the real column is `Age`, so it's missing -> error.
    check_both_raise("SELECT Age AS x FROM __THIS__", _AGE)


def test_quoted_camelcase_column_resolves():
    # `"Age"` is case-exact -> resolves.
    check('SELECT "Age" AS x FROM __THIS__', _AGE, expect=[{"x": 10.0}])


def test_unquoted_lowercase_column_resolves():
    # Folding an already-lowercase name is a no-op.
    check("SELECT age AS x FROM __THIS__", _age, expect=[{"x": 10.0}])


def test_qualified_unquoted_camelcase_folds_and_misses():
    # The qualifier (`__THIS__`) is left alone; only the column part folds:
    # `Age` -> `age`, missing.
    check_both_raise("SELECT __THIS__.Age AS x FROM __THIS__", _AGE)


def test_qualified_quoted_camelcase_resolves():
    check('SELECT __THIS__."Age" AS x FROM __THIS__', _AGE, expect=[{"x": 10.0}])


def test_unquoted_alias_folds_to_lowercase():
    # DataFusion folds an unquoted output alias too: `AS Foo` -> column `foo`.
    check("SELECT age AS Foo FROM __THIS__", _age, expect=[{"foo": 10.0}])


def test_quoted_alias_stays_case_exact():
    check('SELECT age AS "Foo" FROM __THIS__', _age, expect=[{"Foo": 10.0}])
