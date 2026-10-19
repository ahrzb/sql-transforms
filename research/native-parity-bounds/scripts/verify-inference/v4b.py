"""V4b: Box-Cox (shipped entry) -> models; adds the cross-CPU twin (glibc
without FMA ifuncs) as T'.  usage: v4b.py make SEED | v4b.py eval SEED"""
import pickle, sys, warnings
import numpy as np
from pathlib import Path
V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
warnings.filterwarnings("ignore")


def make(seed):
    import v4_boxcox as m
    from sklearn.preprocessing import PowerTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    import duckdb, pyarrow as pa
    rng = np.random.default_rng(seed)
    X, y = m.gen(230000, rng)
    Xtr, ytr, Xn = X[:30000], y[:30000], X[30000:]
    est = PowerTransformer(method="box-cox", standardize=False)
    Ftr = est.fit_transform(Xtr)
    ms = dict(lr=LogisticRegression(max_iter=5000).fit(Ftr, ytr), dt=DecisionTreeClassifier(random_state=0).fit(Ftr, ytr),
              rf=RandomForestClassifier(100, random_state=0, n_jobs=4).fit(Ftr, ytr),
              hgb=HistGradientBoostingClassifier(random_state=0).fit(Ftr, ytr))
    Xall = np.vstack([Xn, Xtr])
    con = duckdb.connect(); con.register("t", pa.table({f"f{i}": Xall[:, i] for i in range(6)}))
    sql = "SELECT " + ", ".join(f"{m.bc_sql(f'f{i}', float(l))} AS o{i}" for i, l in enumerate(est.lambdas_)) + " FROM t"
    r = con.execute(sql).fetchnumpy()
    N = np.column_stack([np.asarray(r[f"o{i}"], dtype=np.float64) for i in range(6)])
    pickle.dump({"bc": dict(est=est, X=Xall)}, open(V / "data" / f"v4b_{seed}_cases.pkl", "wb"))
    pickle.dump(dict(ms=ms, N=N, nn=len(Xn), Ftr=Ftr), open(V / "data" / f"v4b_{seed}_fit.pkl", "wb"))


def ev(seed):
    f = pickle.load(open(V / "data" / f"v4b_{seed}_fit.pkl", "rb"))
    ms, N, nn = f["ms"], f["N"], f["nn"]
    T = np.load(V / "data" / f"v4b_{seed}_def.npz")["bc__row"]
    Tn = np.load(V / "data" / f"v4b_{seed}_nofma.npz")["bc__row"]
    def o(F):
        return dict(lr=ms["lr"].predict(F), dt=ms["dt"].apply(F.astype(np.float32)), rf=ms["rf"].apply(F.astype(np.float32)),
                    hgbr=ms["hgb"]._raw_predict(F).ravel(), hgb=ms["hgb"].predict(F))
    oT = o(T)
    for nm, C in (("N (DuckDB entry)", N), ("twin, glibc w/o FMA", Tn)):
        oc = o(C)
        s = f"seed {seed} {nm:20s} entries!= {np.mean(C != T):.4f}"
        for lab, sl in (("NEW", slice(0, nn)), ("TRAIN", slice(nn, None))):
            s += (f" | {lab}: LR {int((oT['lr'][sl] != oc['lr'][sl]).sum())} DT {int((oT['dt'][sl] != oc['dt'][sl]).sum())} "
                  f"RF {int((oT['rf'][sl] != oc['rf'][sl]).any(1).sum())} HGB rows {int((oT['hgbr'][sl] != oc['hgbr'][sl]).sum())} "
                  f"HGB lab {int((oT['hgb'][sl] != oc['hgb'][sl]).sum())}")
        print(s)


if __name__ == "__main__":
    {"make": make, "eval": ev}[sys.argv[1]](int(sys.argv[2]))
