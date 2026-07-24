"""Differential parity for fitted transformers referenced as {ref} in a t-string."""

import math

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sql_transform import SQLTransform, _transformer_ref

# The nameless-input warning is a known false positive (see test_transformer_udf).
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names:UserWarning"
)


def _assert_close(a, b, path=""):
    """Recursively compare two structures (dict/list/scalar), floats within a
    tolerance, everything else exactly.

    transform (DataFusion) and infer_batch (Rust) take different numerical
    paths -- transform stacks the whole batch into one matrix, infer_batch runs
    row-at-a-time -- so their floats can differ in the last bit or two (pure
    ULP noise, e.g. a non-collinear PCA fixture measured at ~2e-16). That is
    not the divergence class these tests exist to catch: the bugs this branch
    fixed were whole-feature swaps and hard planning failures, orders of
    magnitude above any float tolerance. Exact equality would flake on the
    next fixture that isn't lucky enough to land on exactly-representable
    arithmetic.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{path}: key mismatch {a.keys()} != {b.keys()}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: length mismatch {a!r} != {b!r}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (
            f"{path}: {a!r} != {b!r}"
        )
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _both_engines(t, test_df):
    """transform (DataFusion) and infer (Rust) as plain dicts; assert close."""
    batch = t.transform(pa.Table.from_pandas(test_df)).to_pylist()
    infer = [r.model_dump() for r in t.infer_batch(test_df.to_dict("records"))]
    _assert_close(infer, batch)
    return batch


def test_single_scaler_ref_parity():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = sc.transform(test)
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)


@pytest.mark.parametrize(
    "sql_of",
    [
        pytest.param(
            lambda sc: t"SELECT age FROM __THIS__ QUALIFY {sc}(age) > 1", id="qualify"
        ),
        pytest.param(
            lambda sc: t"SELECT DISTINCT ON ({sc}(age)) age FROM __THIS__",
            id="distinct_on",
        ),
        pytest.param(
            lambda sc: t"SELECT age FROM __THIS__ CLUSTER BY {sc}(age)", id="cluster_by"
        ),
        pytest.param(
            lambda sc: t"SELECT age FROM __THIS__ SORT BY {sc}(age)", id="sort_by"
        ),
        pytest.param(
            lambda sc: (
                t"SELECT AVG(age) OVER w AS a FROM __THIS__ "
                t"WINDOW w AS (PARTITION BY {sc}(age))"
            ),
            id="window_clause",
        ),
    ],
)
def test_transformer_call_outside_projection_raises(sql_of):
    # TASK-2 AC#1. The native engine resolves transformer calls only over the
    # projection (src/lib.rs); DataFusion plans the whole statement. These five
    # clauses survive parse_and_validate, so before this guard fit() ACCEPTED
    # them and the disagreement surfaced late and asymmetrically: transform()
    # raised a DataFusion planning error while infer() silently ignored the
    # clause and returned rows (WINDOW crashed at fit with a bare
    # AttributeError). Reject at build time with one clear message instead.
    # ORDER BY/WHERE/GROUP BY/HAVING/LIMIT/JOIN are already rejected upstream
    # by parse_and_validate, so they cannot reach this guard.
    train = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
    sc = StandardScaler().fit(train)
    t = SQLTransform(sql_of(sc))
    with pytest.raises(ValueError, match="projection"):
        t.fit(pa.Table.from_pandas(train))


def test_nested_threading_parity():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    # Wrap as a DataFrame (not the bare ndarray sc.transform returns) so PCA's
    # fit records feature_names_in_ == sc.get_feature_names_out() -- required
    # for is_transformer(pca) to hold and for the nested schema match.
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=1).fit(scaled)
    t = SQLTransform(t"SELECT {pca}({sc}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)

    expected = pca.transform(sc.transform(test))
    out_names = [str(n) for n in pca.get_feature_names_out()]
    got = np.array([[b["out"][n] for n in out_names] for b in batch])
    assert np.allclose(got, expected)


def test_ordinal_encoder_ref_string_in_int_out():
    train = pd.DataFrame(
        {"color": ["red", "green", "blue", "red"], "size": ["S", "M", "L", "M"]}
    )
    enc = OrdinalEncoder(dtype=np.int64).fit(train)
    t = SQLTransform(t"SELECT {enc}(color, size) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"color": ["blue", "red"], "size": ["L", "S"]})
    batch = _both_engines(t, test)

    expected = enc.transform(test)
    got = np.array([[b["out"]["color"], b["out"]["size"]] for b in batch])
    assert (got == expected).all()


def test_transformer_and_native_window_agg_coexist():
    # A native window agg over __THIS__ alongside a transformer call: proves the
    # two compose without either engine choking. No agg reads the transformer output.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train[["age", "income"]])
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS out, age / (MEAN(age) OVER ()) AS an FROM __THIS__"  # noqa: E501
    ).fit(pa.Table.from_pandas(train))

    test = pd.DataFrame({"age": [25.0, 35.0], "income": [2.5, 3.5]})
    batch = _both_engines(t, test)
    assert np.allclose(
        [b["an"] for b in batch], test["age"].to_numpy() / train["age"].mean()
    )


def test_camelcase_columns_quoted_compose():
    # A transformer ref on case-sensitive columns works when the user quotes them
    # (DataFusion keeps a quoted identifier case-exact). The ref carries the
    # quoting through verbatim, so both engines resolve `"LotArea"` identically.
    train = pd.DataFrame(
        {"LotArea": [1.0, 2.0, 3.0, 4.0], "YearBuilt": [1990.0, 2000.0, 2010.0, 2020.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t'SELECT {sc}("LotArea", "YearBuilt") AS out FROM __THIS__').fit(
        pa.Table.from_pandas(train)
    )

    test = pd.DataFrame({"LotArea": [2.5], "YearBuilt": [2005.0]})
    batch = _both_engines(t, test)

    expected = sc.transform(test)
    got = np.array([[b["out"]["LotArea"], b["out"]["YearBuilt"]] for b in batch])
    assert np.allclose(got, expected)


def test_camelcase_columns_unquoted_folds_and_fails():
    # Unquoted CamelCase folds to lowercase in DataFusion (the oracle) and misses,
    # so the transformer-ref path must NOT silently make it work (the reverted
    # TASK-25 force-quote did). The user quotes instead (see the quoted test).
    # TASK-28.
    train = pd.DataFrame(
        {"LotArea": [1.0, 2.0, 3.0, 4.0], "YearBuilt": [1990.0, 2000.0, 2010.0, 2020.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(LotArea, YearBuilt) AS out FROM __THIS__")
    with pytest.raises(Exception, match="(?i)lotarea|no field|schema"):
        fitted = t.fit(pa.Table.from_pandas(train))
        fitted.transform(pa.Table.from_pandas(train))


def test_ndarray_fit_transformer_binds_positionally():
    # sklearn records feature_names_in_ only for DataFrame fit. An ndarray-fit
    # transformer has no names, so arguments bind positionally in call order.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train.to_numpy())
    assert not hasattr(sc, "feature_names_in_")

    t = SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = sc.transform(train.to_numpy())
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, expected)

    # clone contract: the user's object must be left untouched
    assert not hasattr(sc, "feature_names_in_")


def test_ndarray_fit_arity_mismatch_raises():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train.to_numpy())  # n_features_in_ == 2
    t = SQLTransform(t"SELECT {sc}(age) AS out FROM __THIS__")
    with pytest.raises(ValueError, match="bind positionally"):
        t.fit(pa.Table.from_pandas(train))


def test_named_fit_column_mismatch_still_raises():
    # The named path is unchanged: names are validated as a set.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT {sc}(age) AS out FROM __THIS__")
    with pytest.raises(ValueError, match="must match feature_names_in_"):
        t.fit(pa.Table.from_pandas(train))


def test_ndarray_fit_outer_in_nested_ref_works():
    # The nested branch used to read feature_names_in_ unconditionally, so an
    # ndarray-fit OUTER died with a raw AttributeError. The inner's materialised
    # output order is the natural positional order, so binding it works.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    a = StandardScaler().fit(train)
    a_out = pd.DataFrame(a.transform(train), columns=a.get_feature_names_out())
    b = PCA(n_components=1).fit(a_out.to_numpy())  # ndarray fit: no names
    assert not hasattr(b, "feature_names_in_")

    t = SQLTransform(t"SELECT {b}({a}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = b.transform(a.transform(train))
    out_names = [str(n) for n in b.get_feature_names_out()]
    got = np.array([[r["out"][n] for n in out_names] for r in batch])
    assert np.allclose(got, expected)
    assert not hasattr(b, "feature_names_in_")  # clone contract


def test_unsettable_feature_names_gives_actionable_error():
    # Pipeline.feature_names_in_ is a read-only property delegating to steps[0].
    # Synthesising names onto it raises AttributeError; the user must see an
    # actionable message, not a raw attribute error from inside fit().
    from sklearn.pipeline import Pipeline

    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    p = Pipeline([("sc", StandardScaler())]).fit(train.to_numpy())
    t = SQLTransform(t"SELECT {p}(age, income) AS o FROM __THIS__")
    with pytest.raises(ValueError, match="Re-fit it on a pandas DataFrame"):
        t.fit(pa.Table.from_pandas(train))


class _SpyTransformer:
    """Wraps a fitted transformer and counts .transform() calls."""

    def __init__(self, obj):
        self._obj = obj
        self.calls = 0
        self.feature_names_in_ = obj.feature_names_in_
        self.n_features_in_ = obj.n_features_in_

    def transform(self, x):
        self.calls += 1
        return self._obj.transform(x)

    def get_feature_names_out(self):
        return self._obj.get_feature_names_out()


def test_leaf_ref_probes_transform_once():
    # A leaf ref's materialised output is only ever read by an OUTER ref's probe.
    # With no outer, materialising it is a discarded .transform() call.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    spy = _SpyTransformer(StandardScaler().fit(train))

    SQLTransform(t"SELECT {spy}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert spy.calls == 1, f"expected a single probe, got {spy.calls}"


def test_nested_refs_probe_once_each():
    # The inner IS consumed, so it must still be materialised -- but from the
    # probe's own output, not a second .transform() call.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    inner = _SpyTransformer(sc)
    outer = _SpyTransformer(PCA(n_components=1).fit(scaled))

    SQLTransform(t"SELECT {outer}({inner}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert (inner.calls, outer.calls) == (1, 1), (inner.calls, outer.calls)


def test_leaf_ref_skips_materialization(monkeypatch):
    # The `consumed` gate saves a _table_from_probe call (a table allocation),
    # not a .transform() call -- _table_from_probe reuses the probe's `y` and
    # never calls .transform. So the spy tests above cannot see this gate: they
    # count .transform(), which is identical whether or not the gate exists.
    # Count _table_from_probe calls directly instead.
    calls = []
    orig = _transformer_ref._table_from_probe

    def counting(y, out_schema):
        calls.append(1)
        return orig(y, out_schema)

    monkeypatch.setattr(_transformer_ref, "_table_from_probe", counting)

    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)

    # Leaf-only: no outer consumer, so the materialised table is never built.
    SQLTransform(t"SELECT {sc}(age, income) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert len(calls) == 0, f"expected 0 materializations for a leaf, got {len(calls)}"

    calls.clear()
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=1).fit(scaled)

    # Nested: the inner ref IS consumed by the outer, so it must be built once.
    SQLTransform(t"SELECT {pca}({sc}(age, income)) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    assert len(calls) == 1, (
        f"expected 1 materialization for the inner, got {len(calls)}"
    )


def test_aggregate_over_transformer_output_raises():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    t = SQLTransform(t"SELECT AVG({sc}(age, income)) OVER () AS m FROM __THIS__")
    with pytest.raises(ValueError, match="two-stage"):
        t.fit(pa.Table.from_pandas(train))


def test_aggregate_over_sqltransform_ref_still_works():
    # The guard must NOT overreach. A SQLTransform ref inlines to a scalar, so an
    # aggregate over it is ordinary flat SQL -- a shipped, documented capability.
    train = pa.table({"age": [10.0, 20.0, 30.0, 40.0]})
    inner = SQLTransform("SELECT age / MEAN(age) OVER () AS a FROM __THIS__").fit(
        pa.table({"age": [1.0, 2.0, 3.0]})
    )
    t = SQLTransform(
        t"SELECT AVG({inner.transform}(age)) OVER () AS m FROM __THIS__"
    ).fit(train)
    assert t.transform(train).column("m").to_pylist() == [12.5, 12.5, 12.5, 12.5]


def test_mixed_leaf_and_nested_args_raises():
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)
    scaled = pd.DataFrame(sc.transform(train), columns=sc.get_feature_names_out())
    pca = PCA(n_components=2).fit(scaled)
    t = SQLTransform(t"SELECT {pca}({sc}(age, income), age) AS o FROM __THIS__")
    with pytest.raises(ValueError, match="plain columns or another transformer ref"):
        t.fit(pa.Table.from_pandas(train))


def test_transformer_alongside_partitioned_window_agg():
    # Lock-in: a transformer callout and a PARTITION BY window agg coexist, and
    # both engines agree. Currently works; nothing must silently break it.
    train = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, 40.0],
            "income": [1.0, 2.0, 3.0, 4.0],
            "city": ["a", "a", "b", "b"],
        }
    )
    sc = StandardScaler().fit(train[["age", "income"]])
    t = SQLTransform(
        t"SELECT {sc}(age, income) AS o, AVG(age) OVER (PARTITION BY city) AS m "
        t"FROM __THIS__"
    ).fit(pa.Table.from_pandas(train))

    batch = t.transform(pa.Table.from_pandas(train)).to_pylist()
    infer = [r.model_dump() for r in t.infer_batch(train.to_dict("records"))]
    # `o` is StandardScaler output -- same batch-column_stack-vs-row-at-a-time
    # risk class as PCA -- so compare with the tolerance helper, not exact
    # equality, even though this particular fixture happens to land exact.
    _assert_close(infer, batch)
    assert [r["m"] for r in batch] == [15.0, 15.0, 35.0, 35.0]


def test_unfit_ref_is_fitted_once_globally_not_per_partition():
    # An unfit ref under an outer PARTITION BY is fitted ONCE over all rows; the
    # partitioning applies to the outer aggregate over its output. Matches
    # sklearn, where a Pipeline step is fitted once on all training data.
    # Per-group fitting is a separate feature (DRAFT-14), not this.
    train = pa.table({"age": [10.0, 20.0, 30.0, 50.0], "city": ["a", "a", "b", "b"]})
    norm = SQLTransform("SELECT age / MEAN(age) OVER () AS a FROM __THIS__")
    t = SQLTransform(
        t"SELECT AVG({norm}(age)) OVER (PARTITION BY city) AS m FROM __THIS__"
    ).fit(train)

    # global mean 27.5, NOT per-city 15.0 / 40.0
    assert t._state_tables["__STATE_R0__"].column("avg_age").to_pylist() == [27.5]
    got = t.transform(train).column("m").to_pylist()
    assert np.allclose(got, [0.5454545, 0.5454545, 1.4545455, 1.4545455])


def test_three_level_nesting_parity():
    # AC#4. Load-bearing after Task 2: the `consumed` set decides materialisation
    # by nesting position, and 3 levels is the only shape where a ref is
    # simultaneously consumed AND a consumer.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    a = StandardScaler().fit(train)
    a_out = pd.DataFrame(a.transform(train), columns=a.get_feature_names_out())
    b = StandardScaler().fit(a_out)
    b_out = pd.DataFrame(b.transform(a_out), columns=b.get_feature_names_out())
    c = PCA(n_components=1).fit(b_out)

    t = SQLTransform(t"SELECT {c}({b}({a}(age, income))) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = c.transform(b.transform(a.transform(train)))
    out_names = [str(n) for n in c.get_feature_names_out()]
    got = np.array([[r["out"][n] for n in out_names] for r in batch])
    assert np.allclose(got, expected)


def test_named_fit_call_order_is_free():
    # A DataFrame-fitted transformer binds BY NAME, so the SQL may list columns
    # in any order -- a capability the README documents. The regression this
    # guards: in_schema was probed in FITTED order while the named_struct was
    # built in CALL order, so the UDF's Exact struct signature stopped matching
    # what the SQL built. DataFusion then refused the call while the native
    # engine (which binds by name) still returned rows -- the two engines
    # disagreeing about one fitted object.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    sc = StandardScaler().fit(train)  # fitted order: [age, income]
    t = SQLTransform(t"SELECT {sc}(income, age) AS out FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )  # CALL order: [income, age]
    batch = _both_engines(t, train)
    got = np.array([[b["out"]["age"], b["out"]["income"]] for b in batch])
    assert np.allclose(got, sc.transform(train))


def test_call_order_free_for_a_validating_transformer():
    # The fit-time probe must feed .transform in feature_names_in_ order, the
    # way both engines feed it at runtime. Probing in call order ran the
    # transform on feature-swapped data, so a transformer that VALIDATES its
    # input (OrdinalEncoder with handle_unknown="error") rejected the probe and
    # fit() raised -- for a query both engines execute correctly.
    df = pd.DataFrame({"cat": ["a", "b", "a", "b"], "grp": ["x", "y", "x", "y"]})
    oe = OrdinalEncoder(handle_unknown="error").fit(df)  # names: [cat, grp]

    t = SQLTransform(t"SELECT {oe}(grp, cat) AS o FROM __THIS__").fit(
        pa.Table.from_pandas(df)
    )  # CALL order reversed
    batch = _both_engines(t, df)
    expected = oe.transform(df)
    got = np.array([[r["o"]["cat"], r["o"]["grp"]] for r in batch])
    assert np.allclose(got, expected)


def test_nested_outer_fitted_in_permuted_order_parity():
    # The outer's fitted order is a PERMUTATION of the inner's output names. The
    # struct the outer actually receives is the inner's output, so in_schema must
    # describe THAT, not the outer's fitted order. Declaring the fitted order
    # desynced the UDF's Exact struct signature from what the SQL builds:
    # fit() accepted, the DataFusion oracle refused to plan, and the native
    # engine still returned rows -- the two engines disagreeing about one query.
    # n_components=1 (not 2, like the fit is otherwise wide enough for): with
    # this fixture's exactly-collinear age/income, a 2nd component would be a
    # true zero-variance direction where the two engines' float noise disagrees
    # in the low digits -- a real but unrelated sensitivity this test must not
    # trip over. The permutation bug being tested lives in the INPUT struct
    # order, unaffected by output width.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    inner = StandardScaler().fit(train)  # emits struct<age, income>
    mid = pd.DataFrame(inner.transform(train), columns=inner.get_feature_names_out())
    outer = PCA(n_components=1).fit(mid[["income", "age"]])  # fitted [income, age]

    t = SQLTransform(t"SELECT {outer}({inner}(age, income)) AS o FROM __THIS__").fit(
        pa.Table.from_pandas(train)
    )
    batch = _both_engines(t, train)
    expected = outer.transform(mid[["income", "age"]])
    out_names = [str(n) for n in outer.get_feature_names_out()]
    got = np.array([[r["o"][n] for n in out_names] for r in batch])
    assert np.allclose(got, expected)


def test_nested_outer_column_name_mismatch_raises():
    # Outer fitted on DIFFERENT NAMES than the inner emits. Previously this died
    # with a raw KeyError from inside _probe, mid-fit(). The nested branch now
    # validates the name set like the leaf branch does.
    train = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0], "income": [1.0, 2.0, 3.0, 4.0]}
    )
    inner = StandardScaler().fit(train)  # emits age, income
    mid = pd.DataFrame(inner.transform(train), columns=inner.get_feature_names_out())
    renamed = mid.rename(columns={"age": "c1", "income": "c2"})
    outer = PCA(n_components=2).fit(renamed)  # feature_names_in_ = [c1, c2]

    t = SQLTransform(t"SELECT {outer}({inner}(age, income)) AS o FROM __THIS__")
    with pytest.raises(ValueError, match="must match feature_names_in_"):
        t.fit(pa.Table.from_pandas(train))
