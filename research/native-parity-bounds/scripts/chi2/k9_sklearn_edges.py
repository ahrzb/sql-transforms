"""Q1: sklearn 1.9 AdditiveChi2Sampler on edge inputs, called as the step
calls it (est.transform([row])), and per-row == vectorized on 5-feature rows."""
import math, numpy as np, warnings
from sklearn.kernel_approximation import AdditiveChi2Sampler
for steps, s in ((1, None), (2, None), (3, None), (2, 0.0)):
    est = AdditiveChi2Sampler(sample_steps=steps, sample_interval=s).fit([[1.0, 1.0]])
    print(f"steps={steps} s={s} width={est.transform([[1.0, 1.0]]).shape[1]}")
    for row in ([0.0, 0.5], [-0.0, 0.5], [5e-324, 0.5], [1e308, 0.5], [-1e-300, 0.5], [math.nan, 0.5], [math.inf, 0.5]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                y = est.transform([row])[0]
            print(f"  {row!r:22} -> {[repr(float(v)) for v in y]}")
        except Exception as e:
            print(f"  {row!r:22} -> {type(e).__name__}: {str(e)[:70]}")
rng = np.random.default_rng(1)
X = np.abs(rng.normal(0, 3, (20_000, 5))) ** 3
X[rng.random(X.shape) < 0.1] = 0.0
for steps in (2, 3):
    est = AdditiveChi2Sampler(sample_steps=steps).fit(X[:2])
    V = est.transform(X)
    P = np.array([est.transform([list(map(float, r))])[0] for r in X])
    print(f"steps={steps}: per-row == vectorized on 20,000 x 5 rows:", np.array_equal(V.view(np.int64), P.view(np.int64)))
