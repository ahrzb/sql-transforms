from sql_transform.native import coverage


def test_the_coverage_table_is_current():
    assert coverage.block(coverage.DOC.read_text()) == coverage.table(), (
        "regenerate it: uv run python -m sql_transform.native.coverage --write"
    )


def test_every_named_class_has_a_row():
    # sklearn transformers, and Pipeline: a composition the catalog serves.
    listed = {n for n, _, _ in coverage.rows()}
    assert set(coverage.COMPOSITIONS) <= listed
    assert set(coverage.OUT_OF_SCOPE) <= listed


def test_a_bound_reads_by_its_shape():
    from sql_transform.native import Entry

    def t(est, x, types):
        return x

    assert coverage._exactness(Entry(t, 0)) == "bit-exact"
    assert coverage._exactness(Entry(t, 4)) == "within 4 ulps"
    assert (
        coverage._exactness(Entry(t, 3, lambda est: 0))
        == "bit-exact; within 3 ulps for some configurations"
    )


def test_nonzero_ulp_bounds_counts_the_classes_with_a_bound():
    rows = {n: note for n, s, note in coverage.rows() if s == "native"}
    assert coverage.nonzero_ulp_bounds() == sum(
        1 for note in rows.values() if note != "bit-exact"
    )
