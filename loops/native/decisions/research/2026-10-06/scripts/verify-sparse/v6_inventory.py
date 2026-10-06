import warnings; warnings.simplefilter("ignore")
import numpy as np, scipy.sparse as sp, inspect, signal, time
from sklearn.utils import all_estimators
rng = np.random.default_rng(0)
X = np.abs(rng.normal(size=(40, 4))) + 0.1
Xi = rng.integers(0, 5, size=(40, 4)).astype(float)
y = rng.integers(0, 2, 40)
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
ests = all_estimators(type_filter="transformer")
print("n transformers:", len(ests), flush=True)
for name, cls in ests:
    params = inspect.signature(cls.__init__).parameters
    spp = {p: params[p].default for p in params if "sparse" in p or p == "encode"}
    res = "?"
    t0 = time.time()
    try:
        est = cls()
        signal.alarm(15)
        for XX in (X, Xi):
            try:
                est.fit(XX, y); out = est.transform(XX[:1])
                res = f"SPARSE {type(out).__name__} {out.shape}" if sp.issparse(out) else "dense"
                break
            except TO: raise
            except Exception as e:
                res = "err " + type(e).__name__
        signal.alarm(0)
    except TO:
        res = "timeout"
    except Exception as e:
        res = "ctor " + type(e).__name__
    if res.startswith("SPARSE") or spp or res == "timeout":
        print(f"{name:32s} {res:40s} {spp}", flush=True)
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer
print("KNT params:", sorted(KNeighborsTransformer().get_params()))
print("RNT params:", sorted(RadiusNeighborsTransformer().get_params()))
