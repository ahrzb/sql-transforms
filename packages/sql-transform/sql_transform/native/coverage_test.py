from sql_transform.native import coverage


def test_the_coverage_table_is_current():
    assert coverage.block(coverage.DOC.read_text()) == coverage.table(), (
        "regenerate it: uv run python -m sql_transform.native.coverage --write"
    )


def test_every_listed_name_is_a_sklearn_transformer():
    listed = {n for n, _, _ in coverage.rows()}
    assert set(coverage.COMPOSITIONS) <= listed
    assert set(coverage.OUT_OF_SCOPE) <= listed
