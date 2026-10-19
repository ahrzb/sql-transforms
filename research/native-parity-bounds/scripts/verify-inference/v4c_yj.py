"""V4c: YJ(standardize) -> HGB etc. on grid data, more seeds; T' = numpy w/o AVX-512."""
import pickle, sys, warnings
import numpy as np
from pathlib import Path
V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
warnings.filterwarnings("ignore")


def make(seed):
    from v3_common import gen_txn, nat_yj
    from sklearn.preprocessing import PowerTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(seed)
    X, y = gen_txn(230000, rng)
    Xtr, ytr, Xn = X[:30000], y[:30000], X[30000:]
    est = PowerTransformer(method="yeo-johnson", standardize=True)
    Ftr = est.fit_transform(Xtr)
    ms = dict(lr=LogisticRegression(max_iter=5000).fit(Ftr, ytr), dt=DecisionTreeClassifier(random_state=0).fit(Ftr, ytr),
              rf=RandomForestClassifier(100, random_state=0, n_jobs=4).fit(Ftr, ytr),
              hgb=HistGradientBoostingClassifier(random_state=0).fit(Ftr, ytr))
    Xall = np.vstack([Xn, Xtr])
    pickle.dump({"yj": dict(est=est, X=Xall)}, open(V / "data" / f"v4c_{seed}_cases.pkl", "wb"))
    pickle.dump(dict(ms=ms, N=nat_yj(est, Xall), nn=len(Xn)), open(V / "data" / f"v4c_{seed}_fit.pkl", "wb"))


def ev(seed):
    f = pickle.load(open(V / "data" / f"v4c_{seed}_fit.pkl", "rb"))
    ms, N, nn = f["ms"], f["N"], f["nn"]
    T = np.load(V / "data" / f"v4c_{seed}_def.npz")["yj__row"]
    Tb = np.load(V / "data" / f"v4c_{seed}_def.npz")["yj__batch"]
    Tn = np.load(V / "data" / f"v4c_{seed}_noavx.npz")["yj__row"]
    def o(F):
        return dict(lr=ms["lr"].predict(F), dt=ms["dt"].apply(F.astype(np.float32)), rf=ms["rf"].apply(F.astype(np.float32)),
                    hgbr=ms["hgb"]._raw_predict(F).ravel(), hgb=ms["hgb"].predict(F))
    oT = o(T)
    for nm, C in (("N", N), ("twin batch", Tb), ("twin numpy w/o AVX-512", Tn)):
        oc = o(C); sl = slice(0, nn)
        print(f"seed {seed} {nm:24s} entries!= {np.mean(C != T):.3f} | NEW {nn}: LR {int((oT['lr'][sl] != oc['lr'][sl]).sum())} "
              f"DT {int((oT['dt'][sl] != oc['dt'][sl]).sum())} RF {int((oT['rf'][sl] != oc['rf'][sl]).any(1).sum())} "
              f"HGB rows {int((oT['hgbr'][sl] != oc['hgbr'][sl]).sum())} HGB lab {int((oT['hgb'][sl] != oc['hgb'][sl]).sum())}")


if __name__ == "__main__":
    {"make": make, "eval": ev}[sys.argv[1]](int(sys.argv[2]))
