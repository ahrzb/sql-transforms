"""Does the Yeo-Johnson twin depend on the CPU? The same fitted steps and
rows (catalog generator, seeds 0..N-1), transformed with numpy's SVML
kernels (default on AVX-512) and with them disabled (NPY_DISABLE_CPU_FEATURES
=X86_V4: numpy's log1p/expm1 are glibc's, as on a CPU without AVX-512).

usage:
  python twin_cpu.py gen N FIX.pkl          # fit steps, draw rows (default dispatch)
  python twin_cpu.py eval FIX.pkl OUT.npy   # twin outputs under this process's dispatch
  python twin_cpu.py fit N OUT.npy          # fitted lambdas under this process's dispatch
"""

import math
import pickle
import sys
import typing
import warnings

_eval = typing._eval_type
typing._eval_type = lambda *a, prefer_fwd_module=None, **k: _eval(*a, **k)

import numpy as np  # noqa: E402
from sklearn.preprocessing import PowerTransformer  # noqa: E402

from sql_transform.native import catalog_test as ct  # noqa: E402

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
CONFIGS = {
    "yj_std0": lambda: PowerTransformer(standardize=False),
    "yj_std1": lambda: PowerTransformer(),
}
mode = sys.argv[1]
if mode == "gen":
    n, path = int(sys.argv[2]), sys.argv[3]
    fx = []
    for name, make in CONFIGS.items():
        for seed in range(n):
            step = ct._step(make, seed, 0)
            rows = ct._rows(step, seed, False).to_pylist()
            fx.append((name, seed, step.instances, list(step.takes.names), rows))
    pickle.dump(fx, open(path, "wb"))
elif mode == "eval":
    fx = pickle.load(open(sys.argv[2], "rb"))
    out = []
    for name, seed, inst, names, rows in fx:
        for r in rows:
            if r["__iid"] is None:
                continue
            est = inst[r["__iid"]]
            vals = [math.nan if r[f] is None else float(r[f]) for f in names]
            try:
                tw = np.asarray(est.transform([vals]), dtype=float)[0]
            except ValueError:
                continue
            for j, x in enumerate(vals):
                lam = float(est.lambdas_[j])
                if x >= 0:
                    w = lam * math.log1p(x) if abs(lam) >= 2**-52 else 0.0
                elif x < 0 and abs(lam - 2) > 2**-52:
                    w = (2 - lam) * math.log1p(-x)
                else:
                    w = 0.0
                m = float(est._scaler.mean_[j]) if est.standardize else 0.0
                s = float(est._scaler.scale_[j]) if est.standardize else 1.0
                out.append((0 if name == "yj_std0" else 1, tw[j], w, m, s, x, lam))
    np.save(sys.argv[3], np.array(out, dtype=float))
elif mode == "fit":
    n = int(sys.argv[2])
    lams = []
    for seed in range(n):
        rng = np.random.default_rng([seed, 99])
        for k in range(3):
            # fit-like columns: the generator's kinds 0, 1, 3, 4
            m = int(rng.integers(5, 60))
            X = np.column_stack([
                rng.normal(rng.uniform(-100, 100), rng.uniform(0.01, 50), m),
                rng.integers(-1000, 1000, m).astype(float),
                rng.exponential(rng.uniform(0.1, 1e6), m),
                rng.integers(-3, 4, m).astype(float),
            ])
            lams.extend(PowerTransformer().fit(X).lambdas_.tolist())
    np.save(sys.argv[3], np.array(lams))
