VOCAB = ["a", "b", "c", "d", "é", "日本"]
UNSEEN = ["zz", "", "A"]
# Serving values beyond the fit's range: signed zeros, extremes.
EDGES = [0.0, -0.0, 1e-300, -1e300, 1e300, 5e-324]
# A row of only these has a norm under sklearn's zero-scale threshold.
SMALL = [0.0, -0.0, 1e-300, -5e-324, 1e-17, -2.5e-16]
# The widest step drawn, in output lanes: a wider fixture checks the same
# translation, only slower. 1,000 holds degree-2 PolynomialFeatures over
# all 32 features and degree 3 over 16; with 8 seeds the classes that draw
# past 300 lanes (PolynomialFeatures, OneHotEncoder, KBinsDiscretizer) run
# in 35 s, against 31 s at 300 and 44 s at 2,000 (master 49acad5).
MAX_LANES = 1000
# Seeds per configuration: 8 in the gate; a milestone report sweeps more
# (NATIVE_SEEDS=200, loops/native/report-format.md).
SEEDS = int(os.environ.get("NATIVE_SEEDS", "8"))


def _fit_matrix(
    rng: np.random.Generator,
    kinds: list[int],
    types: list[pa.DataType],
    holes: list[str | None],
    marker: float,
    positive: bool = False,
    regression: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One instance's fit data, a column per feature of its kind (an integer
    one rounded), and a binary target, or for a `regression` a continuous
    one: a random multiple of the first feature, standardized, plus noise.
    `kinds` and `holes` are the step's, the same for every instance, as a
    feature keeps its nature across fitted groups: `holes[j]` is None for
    nowhere missing (and never holding the marker), "some" for about a
    fifth of the rows and at least one, "all" for everywhere. `marker`
    spells missing."""
    n = int(rng.integers(5, 60))
    cols = []
    strings = [t == pa.string() for t in types]
    for kind, t, hole in zip(kinds, types, holes, strict=True):
        if t == pa.string():
            # The step's share of VOCAB for this feature: kind - 3 of them.
            c = rng.choice(np.array(VOCAB[: kind - 3], dtype=object), n)
            if hole == "all":
                c[:] = None
            elif hole == "some":
                c[rng.random(n) < 0.2] = None
                c[rng.integers(n)] = None
            cols.append(c)
            continue
        if kind == 0:
            c = rng.normal(rng.uniform(-100, 100), rng.uniform(0.01, 50), n)
        elif kind == 1:
            c = rng.integers(-1000, 1000, n).astype(float)
        elif kind == 2:
            c = np.full(n, rng.uniform(-5, 5))  # zero variance
        elif kind == 3:
            c = rng.exponential(rng.uniform(0.1, 1e6), n)
        else:
            c = rng.integers(-3, 4, n).astype(float)  # few distinct values
        if t == pa.int64():
            c = np.round(c)
        if positive:
            c = np.abs(c)
            c[c == 0] = 1.0
        if not np.isnan(marker):
            c[c == marker] = marker + 1.0
        if hole == "all":
            c[:] = marker
        elif hole == "some":
            c[rng.random(n) < 0.2] = marker
            c[rng.integers(n)] = marker
        cols.append(c)
    y = (rng.random(n) < 0.5).astype(int)
    y[:4] = (0, 1, 0, 1)  # two of each class, for a 2-fold split
    if regression and not strings[0]:
        c = np.nan_to_num(np.asarray(cols[0], dtype=float))
        t = (c - c.mean()) / (c.std() or 1.0)
        y = rng.uniform(-3, 3) * t + rng.normal(0, rng.uniform(0.1, 2), n)
    if any(strings):
        # A string beside numbers makes an object matrix, as the step's
        # rows reach `transform`.
        X = np.empty((n, len(cols)), dtype=object)
        for j, c in enumerate(cols):
            X[:, j] = c if strings[j] else [float(v) for v in c]
        return X, y
    return np.column_stack(cols), y


def _step(cls_factory, seed: int, variant: int = 0) -> PythonTransform:
    # Each of a class's configurations (`variant`) draws shapes of its own,
    # so a class's seeds are not the same few shapes for every one.
    rng = np.random.default_rng([seed, variant])
    # A shape the estimator cannot fit (a selector asked for more features
    # than the step has), that keeps no lane or that is wider than
    # MAX_LANES is drawn again.
    for _ in range(20):
        step = _draw(rng, cls_factory)
        if step is not None:
            return step
    raise AssertionError(f"no fixture of {cls_factory} fits in 20 draws")


def _runs(est: Any) -> list[Any]:
    """The estimators `transform` runs, in order: `est`, or a pipeline's
    steps that run, or a column transformer's or union's parts (remainder
    last), nested ones flattened. A composition's own tags do not say what
    it takes: sklearn 1.9 copies only `pairwise` (a pipeline's first step)
    and `sparse` (all steps or parts) from its estimators, so `allow_nan`
    and `categorical` read False. The generator reads theirs instead."""
    if isinstance(est, Pipeline):
        return [r for _, _, s in est._iter() for r in _runs(s)]
    if isinstance(est, ColumnTransformer):
        parts = [t for _, t, _ in est.transformers] + [est.remainder]
    elif isinstance(est, FeatureUnion):
        parts = [t for _, t in est.transformer_list]
    else:
        return [est]
    return [r for t in parts if not isinstance(t, str) for r in _runs(t)]


def _draw(rng: np.random.Generator, cls_factory) -> PythonTransform | None:
    # Mostly narrow; sometimes wide enough for a row reduction's blocks.
    wide = rng.random() < 0.3
    n_features = int(rng.integers(5, 33) if wide else rng.integers(1, 5))
    n_features = min(n_features, getattr(cls_factory, "max_features", n_features))
    types = [
        pa.float64() if rng.random() < 0.7 else pa.int64() for _ in range(n_features)
    ]
    kinds = [int(rng.integers(5)) for _ in range(n_features)]
    runs = _runs(cls_factory())
    proto = runs[0]  # what reads the row
    positive = getattr(cls_factory, "positive", False)
    if not proto.__sklearn_tags__().input_tags.two_d_array:
        # A one-dimensional input (IsotonicRegression): one feature.
        n_features, types, kinds = 1, types[:1], kinds[:1]
    regression = runs[-1].__sklearn_tags__().estimator_type == "regressor"
    if proto.__sklearn_tags__().input_tags.categorical:
        # Categories: few distinct values per feature, about half of them
        # strings (kind 5..8: two to five of VOCAB).
        for j in range(n_features):
            if rng.random() < 0.5:
                types[j], kinds[j] = pa.string(), int(rng.integers(5, 9))
            else:
                kinds[j] = 4
    takes = pa.schema([(f"x{i}", t) for i, t in enumerate(types)])
    # Missing values in the fit data, for an estimator that takes them: an
    # imputer's own `missing_values`, or NaN where sklearn says it allows it
    # (for a pipeline, every step: a scaler passes NaN on).
    marker = float(getattr(proto, "missing_values", np.nan))
    holes: list[str | None] = [None] * n_features
    if hasattr(proto, "missing_values") or all(
        r.__sklearn_tags__().input_tags.allow_nan for r in runs
    ):
        holes = [
            ("some", "all", None)[int(np.searchsorted([0.4, 0.48], rng.random()))]
            for _ in range(n_features)
        ]
        if hasattr(proto, "missing_values") and "some" not in holes:
            holes[int(rng.integers(n_features))] = "some"
    # Every fit gets a target (an unsupervised one ignores it). A step's
    # instances share one width, as a fitted step's do: an instance that
    # fits to another width ends the step before it.
    instances: dict[int, Any] = {}
    width = 0
    with warnings.catch_warnings():  # all-missing columns, by design
        warnings.simplefilter("ignore")
        try:
            for k in range(int(rng.integers(1, 4))):
                X, y = _fit_matrix(
                    rng, kinds, types, holes, marker, positive, regression
                )
                est = cls_factory().fit(X, y)
                w = np.asarray(est.transform(X[:1])).reshape(1, -1).shape[1]
                if k and w != width:
                    break
                instances[k], width = est, w
        except ValueError:
            return None
    if width == 0 or width > MAX_LANES:
        return None
    if width == 1:
        returns = pa.float64()
    elif rng.random() < 0.3:  # unnamed lanes
        returns = pa.list_(pa.float64(), width)
    else:
        returns = pa.struct([(f"f{j}", pa.float64()) for j in range(width)])
    return PythonTransform("tf", instances, takes, returns)


def _value(rng: random.Random, regime: str) -> float | None:
    r = rng.random()
    if regime == "nulls" and r < 0.3:
        return None
    if regime == "edges" and r < 0.3:
        return rng.choice(EDGES)
    if regime == "small":
        return rng.choice(SMALL)
    if regime == "ints":
        return float(rng.randint(-3, 3))
    return rng.uniform(-1e3, 1e3)


def _rows(step: PythonTransform, seed: int, positive: bool = False) -> pa.Table:
    rng = random.Random(seed)  # noqa: S311
    n = 40
    ids = [rng.choice([None, *step.instances]) for _ in range(n)]
    # A row's regime: plain values; some edges; some NULLs (where most
    # twins answer and a validating one raises); all small; or few distinct
    # integers (a category, or an imputer's numeric missing marker).
    regimes = rng.choices(
        ["plain", "edges", "nulls", "small", "ints"], [45, 15, 15, 10, 15], k=n
    )
    cols: dict[str, pa.Array] = {"__iid": pa.array(ids, pa.int64())}
    for f in step.takes:
        if f.type == pa.string():
            strs = [
                None
                if g == "nulls" and rng.random() < 0.3
                else rng.choice(VOCAB + UNSEEN)
                for g in regimes
            ]
            cols[f.name] = pa.array(strs, pa.string())
            continue
        vals = [_value(rng, g) for g in regimes]
        if positive:
            vals = [None if v is None else abs(v) for v in vals]
        if f.type == pa.int64():
            vals = [
                None if v is None else max(-(2**62), min(2**62, round(v))) for v in vals
            ]
        cols[f.name] = pa.array(vals, f.type)
    return pa.table(cols)


def test_every_entry_has_fixtures():
    assert set(catalog()) == set(FIXTURES), "add fixtures for every catalog entry"


@pytest.mark.parametrize(
    "cls, j, seed",
    [
