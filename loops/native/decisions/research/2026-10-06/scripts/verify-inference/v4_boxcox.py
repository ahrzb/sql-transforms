"""V4: the entry the repo SHIPS: PowerTransformer(box-cox, standardize=False),
spelled as native/power.py spells it, evaluated by DuckDB 1.5.5, then HGB/RF/
DT/LR fitted on fit_transform. NEW rows only (fresh draws), grid-valued data.
Compared: twin row (default), twin batch, fit_transform for training rows."""
import importlib.util, math, sys, warnings
import numpy as np, pyarrow as pa, duckdb
warnings.filterwarnings("ignore")
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
spec = importlib.util.spec_from_file_location("_udf", "/home/user/sql-transforms/packages/sql-transform/sql_transform/_udf.py")
u = importlib.util.module_from_spec(spec); sys.modules["_udf"] = u; spec.loader.exec_module(u)


def gen(n, rng):
    amount = np.round(rng.lognormal(3.5, 1.2, n), 2) + 0.01
    qty = rng.negative_binomial(3, 0.01, n).astype(float) + 1
    tenure = rng.integers(1, 3650, n).astype(float)
    rate = np.round(rng.beta(2, 5, n) * 100, 1) + 0.1
    score = np.clip(np.round(rng.normal(600, 80, n)), 300, 850)
    ratio = np.round(rng.lognormal(0, 0.5, n), 3) + 0.001
    X = np.column_stack([amount, qty, tenure, rate, score, ratio])
    z = (0.9 * np.log(amount) - 3.2 + 0.003 * (qty - 300) - 0.0003 * (tenure - 1800)
         + 0.02 * (rate - 28) + 0.01 * (score - 600) + 0.8 * np.log(ratio) + rng.logistic(0, 1, n))
    return X, (z > 0).astype(int)


def lit(v):
    return f"'{float(v)!r}'::DOUBLE"


def bc_sql(x, lam):
    lx = f"ln({x})"
    if abs(lam) < 1e-19:
        e = lx
    else:
        w = f"({lit(lam)} * {lx})"
        uu = f"exp({w})"
        expm1 = (f"(CASE WHEN {uu} = 1::DOUBLE THEN {w} WHEN {uu} - 1::DOUBLE = -1::DOUBLE THEN -1::DOUBLE "
                 f"WHEN {uu} <= 0::DOUBLE THEN -1::DOUBLE ELSE ({uu} - 1::DOUBLE) * ({w} / ln({uu})) END)")
        big = f"exp({w} - {lit(math.log(abs(lam)))})"
        big = (big if lam > 0 else f"(-{big})") + f" - {lit(1 / lam)}"
        e = f"(CASE WHEN {w} < {lit(709.78)} THEN {expm1} / {lit(lam)} ELSE {big} END)"
    return f"(CASE WHEN isnan({x}) OR {x} <= 0::DOUBLE THEN 'nan'::DOUBLE ELSE {e} END)"


def main(seed):
    rng = np.random.default_rng(seed)
    ntr, nnew = 30000, 200000
    X, y = gen(ntr + nnew, rng)
    Xtr, ytr, Xn = X[:ntr], y[:ntr], X[ntr:]
    est = PowerTransformer(method="box-cox", standardize=False)
    Ftr = est.fit_transform(Xtr)
    ms = dict(lr=LogisticRegression(max_iter=5000).fit(Ftr, ytr), dt=DecisionTreeClassifier(random_state=0).fit(Ftr, ytr),
              rf=RandomForestClassifier(100, random_state=0, n_jobs=4).fit(Ftr, ytr),
              hgb=HistGradientBoostingClassifier(random_state=0).fit(Ftr, ytr))
    Xall = np.vstack([Xn, Xtr])
    pt = u.PythonTransform(name="t", instances={0: est}, takes=pa.schema([(f"f{i}", pa.float64()) for i in range(6)]),
                           returns=pa.list_(pa.float64(), 6))
    T = np.array([pt(0, *r) for r in Xall.tolist()])
    B = est.transform(Xall)
    con = duckdb.connect()
    con.register("t", pa.table({f"f{i}": Xall[:, i] for i in range(6)}))
    sql = "SELECT " + ", ".join(f"{bc_sql(f'f{i}', float(l))} AS o{i}" for i, l in enumerate(est.lambdas_)) + " FROM t"
    r = con.execute(sql).fetchnumpy()
    N = np.column_stack([np.asarray(r[f"o{i}"], dtype=np.float64) for i in range(6)])
    nn = len(Xn)
    print(f"seed {seed}: lambdas {np.round(est.lambdas_, 3).tolist()}")
    print(f"  twin batch==row: {np.array_equal(B, T)}; fit_transform==row(train): {np.array_equal(Ftr, T[nn:])}")
    print(f"  N!=T entries: {np.mean(N != T):.4f} per lane {np.round((N != T).mean(0), 3).tolist()}; max ulps-ish rel {np.max(np.abs(N - T) / np.maximum(np.abs(T), 1e-300)):.1e}")
    def o(F):
        return dict(lr=ms["lr"].predict(F), dt=ms["dt"].apply(F.astype(np.float32)), rf=ms["rf"].apply(F.astype(np.float32)),
                    hgbr=ms["hgb"]._raw_predict(F).ravel(), hgb=ms["hgb"].predict(F))
    oT, oN = o(T), o(N)
    for nm, sl in (("NEW", slice(0, nn)), ("TRAIN", slice(nn, None))):
        print(f"  {nm:5s} rows={sl.stop - sl.start if sl.stop else ntr}: LR {int((oT['lr'][sl] != oN['lr'][sl]).sum())} "
              f"DT leaf {int((oT['dt'][sl] != oN['dt'][sl]).sum())} RF leaf-rows {int((oT['rf'][sl] != oN['rf'][sl]).any(1).sum())} "
              f"HGB branch-rows {int((oT['hgbr'][sl] != oN['hgbr'][sl]).sum())} HGB labels {int((oT['hgb'][sl] != oN['hgb'][sl]).sum())}")
    # which lanes: zero out N's differences lane by lane
    for j in range(6):
        C = T.copy(); C[:, j] = N[:, j]
        oc = o(C)
        print(f"    lane {j}: distinct train values {len(np.unique(Ftr[:, j]))}, NEW HGB branch-rows {int((oT['hgbr'][:nn] != oc['hgbr'][:nn]).sum())}, labels {int((oT['hgb'][:nn] != oc['hgb'][:nn]).sum())}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(int(s))
