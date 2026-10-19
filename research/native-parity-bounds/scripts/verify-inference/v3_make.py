"""V3: my own pipelines (new seeds, new data; test rows are NEW draws from the
generator, plus the training rows re-scored), fitted as Pipeline.fit does."""
import pickle, sys, warnings
import numpy as np
from pathlib import Path
V = Path(__file__).resolve().parent
sys.path.insert(0, str(V))
from v3_common import gen_txn, gen_cont  # noqa
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

N_TR, N_TE = 30000, 100000


def models():
    return dict(lr=LogisticRegression(max_iter=5000), dt=DecisionTreeClassifier(random_state=1),
                rf=RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=4),
                hgb=HistGradientBoostingClassifier(random_state=1))


PIPES = [  # name, data, prescale?, transformer
    ("yj_txn", "txn", False, lambda: PowerTransformer(method="yeo-johnson", standardize=True)),
    ("chi2_txn", "txn", False, lambda: AdditiveChi2Sampler(sample_steps=2)),
    ("kmeans_txn", "txn", True, lambda: KMeans(n_clusters=12, n_init=3, random_state=3)),
    ("pca_txn", "txn", True, lambda: PCA(n_components=0.99, whiten=True)),
    ("pcaraw_txn", "txn", False, lambda: PCA(whiten=True)),
    ("pca_cont", "cont", False, lambda: PCA(whiten=True)),
    ("kmeans_cont", "cont", False, lambda: KMeans(n_clusters=20, n_init=3, random_state=3)),
    ("yj_cont", "cont", False, lambda: PowerTransformer(method="yeo-johnson", standardize=True)),
]


def main():
    rng = np.random.default_rng(20261006)
    data = {}
    Xa, ya = gen_txn(N_TR + N_TE, rng)
    data["txn"] = (Xa[:N_TR], ya[:N_TR], Xa[N_TR:])
    Xb, yb = gen_cont(N_TR + N_TE, rng)
    data["cont"] = (Xb[:N_TR], yb[:N_TR], Xb[N_TR:])
    cases, fitted = {}, {}
    for name, dn, prescale, mk in PIPES:
        Xtr, ytr, Xnew = data[dn]
        if prescale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xnew = sc.transform(Xtr), sc.transform(Xnew)
        est = mk()
        Ftr = est.fit_transform(Xtr)
        ms = {k: m.fit(Ftr, ytr) for k, m in models().items()}
        X = np.vstack([Xnew, Xtr])  # new rows, then the training rows re-scored
        cases[name] = dict(est=est, X=X)
        fitted[name] = dict(Ftr=Ftr, models=ms, n_new=len(Xnew), n_tr=len(Xtr), ytr=ytr)
        print(name, Ftr.shape, {k: round(float(m.score(Ftr, ytr)), 3) for k, m in ms.items()}, flush=True)
    pickle.dump(cases, open(V / "data" / "v3_cases.pkl", "wb"))
    pickle.dump(fitted, open(V / "data" / "v3_fitted.pkl", "wb"))


if __name__ == "__main__":
    main()
