"""Q2: which sklearn 1.9 transformers return a sparse object from transform()
on a DENSE one-row input, at defaults and under sparse-related params.
Empirical sweep over all_estimators(type_filter='transformer')."""
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
X = np.abs(rng.normal(size=(60, 4))) + 0.1  # positive (chi2/NMF/LDA happy)
X[:, 0] = np.round(X[:, 0] * 2)  # a few repeated values
y = (X[:, 1] > np.median(X[:, 1])).astype(int)
Xnan = X.copy()
Xnan[::7, 2] = np.nan
row = X[:1]

OUT = {"DictVectorizer", "FeatureHasher", "HashingVectorizer", "TfidfTransformer",
       "PatchExtractor", "LabelEncoder", "LabelBinarizer", "MultiLabelBinarizer",
       "KernelCenterer", "TSNE", "CountVectorizer", "TfidfVectorizer"}

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

REQ = {
    "ColumnTransformer": lambda C: C([("a", StandardScaler(), [0, 1]), ("b", OneHotEncoder(), [0])]),
    "FeatureUnion": lambda C: C([("a", StandardScaler()), ("b", OneHotEncoder())]),
    "RFE": lambda C: C(LinearRegression(), n_features_to_select=2),
    "RFECV": lambda C: C(LinearRegression()),
    "SelectFromModel": lambda C: C(LinearRegression()),
    "SequentialFeatureSelector": lambda C: C(LinearRegression(), n_features_to_select=2),
    "SparseCoder": lambda C: C(dictionary=rng.normal(size=(3, 4))),
    "StackingClassifier": None, "StackingRegressor": None,
    "VotingClassifier": None, "VotingRegressor": None,
}


def probe(make, Xfit=X, r=row, sup=False):
    est = make()
    try:
        est.fit(Xfit, y) if sup else est.fit(Xfit)
    except TypeError:
        est.fit(Xfit, y)
    except ValueError:
        est.fit(Xfit, y)
    out = est.transform(r)
    return out


rows = []
for name, C in all_estimators(type_filter="transformer"):
    if name in OUT:
        continue
    mk = REQ.get(name, lambda C: C())
    if mk is None:
        continue
    try:
        out = probe(lambda: mk(C))
        rows.append((name, "defaults", type(out).__name__, sp.issparse(out),
                     getattr(out, "shape", None)))
    except Exception as e:
        rows.append((name, "defaults", f"ERR {type(e).__name__}: {str(e)[:60]}", None, None))

# Parameterized variants that can make a dense-input output sparse.
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion

VARIANTS = [
    ("OneHotEncoder", "sparse_output=True (default)", lambda: OneHotEncoder(), X),
    ("OneHotEncoder", "sparse_output=False", lambda: OneHotEncoder(sparse_output=False), X),
    ("KBinsDiscretizer", "encode='onehot' (default)", lambda: KBinsDiscretizer(), X),
    ("KBinsDiscretizer", "encode='onehot-dense'", lambda: KBinsDiscretizer(encode="onehot-dense"), X),
    ("KBinsDiscretizer", "encode='ordinal'", lambda: KBinsDiscretizer(encode="ordinal"), X),
    ("SplineTransformer", "sparse_output=True", lambda: SplineTransformer(sparse_output=True), X),
    ("PolynomialFeatures", "dense input (no sparse param)", lambda: PolynomialFeatures(), X),
    ("MissingIndicator", "sparse='auto' (default), dense in", lambda: MissingIndicator(), Xnan),
    ("MissingIndicator", "sparse=True", lambda: MissingIndicator(sparse=True), Xnan),
    ("SimpleImputer", "add_indicator=True, dense in", lambda: SimpleImputer(add_indicator=True), Xnan),
    ("RandomTreesEmbedding", "sparse_output=True (default)", lambda: RandomTreesEmbedding(n_estimators=5, random_state=0), X),
    ("RandomTreesEmbedding", "sparse_output=False", lambda: RandomTreesEmbedding(n_estimators=5, random_state=0, sparse_output=False), X),
    ("KNeighborsTransformer", "defaults (mode='distance')", lambda: KNeighborsTransformer(), X),
    ("KNeighborsTransformer", "mode='connectivity'", lambda: KNeighborsTransformer(mode="connectivity"), X),
    ("RadiusNeighborsTransformer", "defaults", lambda: RadiusNeighborsTransformer(), X),
    ("SparseRandomProjection", "dense_output=False (default), dense in", lambda: SparseRandomProjection(n_components=3, random_state=0), X),
    ("FunctionTransformer", "func=sp.csr_matrix", lambda: FunctionTransformer(sp.csr_matrix), X),
    ("ColumnTransformer", "OHE part, sparse_threshold=0.3 (default)", lambda: ColumnTransformer([("a", StandardScaler(), [1, 2, 3]), ("b", OneHotEncoder(), [0])]), X),
    ("ColumnTransformer", "OHE part only, sparse_threshold=0.3", lambda: ColumnTransformer([("b", OneHotEncoder(), [0])]), X),
    ("ColumnTransformer", "OHE part, sparse_threshold=0", lambda: ColumnTransformer([("b", OneHotEncoder(), [0])], sparse_threshold=0), X),
    ("FeatureUnion", "StandardScaler + OHE", lambda: FeatureUnion([("a", StandardScaler()), ("b", OneHotEncoder())]), X),
    ("Pipeline", "OHE -> MaxAbsScaler", lambda: make_pipeline(OneHotEncoder(), __import__("sklearn.preprocessing", fromlist=["MaxAbsScaler"]).MaxAbsScaler()), X),
    ("Pipeline", "OHE -> StandardScaler(with_mean=False)", lambda: make_pipeline(OneHotEncoder(), StandardScaler(with_mean=False)), X),
]
for name, desc, mk, Xf in VARIANTS:
    try:
        out = probe(mk, Xf, Xf[:1])
        extra = ""
        if name == "ColumnTransformer":
            est = mk().fit(Xf)
            extra = f" sparse_output_={est.sparse_output_}"
        rows.append((name, desc, type(out).__name__ + extra, sp.issparse(out), getattr(out, "shape", None)))
    except Exception as e:
        rows.append((name, desc, f"ERR {type(e).__name__}: {str(e)[:80]}", None, None))

print(f"{'transformer':30s} {'config':45s} {'type':28s} sparse shape")
for r in rows:
    if r[1] == "defaults" and r[3] is False:
        continue
    print(f"{r[0]:30s} {r[1]:45s} {r[2]:28s} {r[3]!s:6s} {r[4]}")
n_def = sum(1 for r in rows if r[1] == "defaults")
n_def_sparse = [r[0] for r in rows if r[1] == "defaults" and r[3]]
n_def_err = [r[0] for r in rows if r[1] == "defaults" and r[3] is None]
print(f"\ndefaults swept: {n_def}; sparse at defaults: {n_def_sparse}; errors: {n_def_err}")
