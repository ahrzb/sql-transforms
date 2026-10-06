import warnings; warnings.simplefilter("ignore")
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler, KBinsDiscretizer
rng = np.random.default_rng(3)
for n in (37, 203, 1000):
    X = rng.choice(list("abcdefg"), size=(n, 2)).astype(object)
    a = make_pipeline(OneHotEncoder(), StandardScaler(with_mean=False)).fit(X)
    b = make_pipeline(OneHotEncoder(sparse_output=False), StandardScaler(with_mean=False)).fit(X)
    sa, sb = a[-1].scale_, b[-1].scale_
    du = np.abs(sa.view(np.int64) - sb.view(np.int64))
    oa = a.transform(X).toarray(); ob = b.transform(X)
    d = np.abs(oa.view(np.int64) - ob.view(np.int64))
    print(f"n={n}: scale_ ulp diffs max {du.max()} ({(du>0).sum()}/{du.size} differ); output rows differing {(d>0).any(1).sum()}/{n}, max ulp {d.max()}")
    # var_ comparison
    print("   var_ max ulp", np.abs(a[-1].var_.view(np.int64) - b[-1].var_.view(np.int64)).max())
