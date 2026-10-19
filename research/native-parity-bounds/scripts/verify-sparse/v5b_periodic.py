import warnings; warnings.simplefilter("ignore")
import numpy as np, traceback
from sklearn.preprocessing import SplineTransformer
X = np.linspace(-3, 4, 50)[:, None]
for nk in (2, 3, 5):
  for sp_ in (True, False):
    try:
        s = SplineTransformer(degree=0, n_knots=nk, extrapolation="periodic", sparse_output=sp_).fit(X)
        r = s.transform([[0.3]]); r = r.toarray() if hasattr(r, "toarray") else r
        r2 = s.transform(X[:5]); 
        print(nk, sp_, "ok", r)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(nk, sp_, type(e).__name__, str(e)[:120], "@", tb.filename.split("site-packages/")[-1], tb.lineno)
