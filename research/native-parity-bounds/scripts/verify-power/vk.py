"""Independent kernel check: numpy (whatever dispatch) vs libm via math, vs MPFR."""
import sys, math, numpy as np, gmpy2
from gmpy2 import mpfr
gmpy2.get_context().precision = 200
N = int(sys.argv[1]); out = sys.argv[2]
rng = np.random.default_rng(12345)
def draws_log1p(n):
    k = n // 4
    a = rng.uniform(0, 1e3, k)
    b = 10.0 ** rng.uniform(-12, 300, k)
    c = rng.uniform(-0.999, 1.0, k)
    d = 10.0 ** rng.uniform(-20, -1, n - 3 * k)
    return np.concatenate([a, b, c, d])
def draws_expm1(n):
    k = n // 3
    a = rng.uniform(-40, 709, k)
    b = 10.0 ** rng.uniform(-18, 0.5, k) * rng.choice([-1, 1], k)
    c = rng.uniform(-5, 5, n - 2 * k)
    return np.concatenate([a, b, c])
def draws_log(n):
    return np.concatenate([10.0 ** rng.uniform(-300, 300, n//2), rng.uniform(0.5, 2, n - n//2)])
def draws_exp(n):
    return rng.uniform(-700, 709, n)
ref = {'log1p': gmpy2.log1p, 'expm1': gmpy2.expm1, 'log': gmpy2.log, 'exp': gmpy2.exp}
lm = {'log1p': math.log1p, 'expm1': math.expm1, 'log': math.log, 'exp': math.exp}
dr = {'log1p': draws_log1p, 'expm1': draws_expm1, 'log': draws_log, 'exp': draws_exp}
res = {}
for f in ['log1p', 'expm1', 'log', 'exp']:
    x = dr[f](N)
    y_np = getattr(np, f)(x)
    y_np1 = np.array([getattr(np, f)(np.array([v]))[0] for v in x[:20000]])
    y_lm = np.array([lm[f](float(v)) for v in x])
    def ulperr(y):
        e = np.empty(len(x))
        for i, (xi, yi) in enumerate(zip(x, y)):
            ex = ref[f](mpfr(float(xi)))
            cr = float(ex)
            if cr == 0 or not math.isfinite(cr):
                e[i] = 0 if yi == cr else np.inf; continue
            ulp = math.ulp(cr)
            # ulp of correctly rounded result (use the binade of cr)
            e[i] = float(abs(mpfr(float(yi)) - ex) / ulp)
        return e
    e_np = ulperr(y_np); e_lm = ulperr(y_lm)
    bits_np = y_np.view(np.int64); bits_lm = y_lm.view(np.int64)
    d = np.abs(bits_np - bits_lm)
    res[f] = dict(n=len(x), np_max=e_np.max(), np_ncr=(e_np > 0.5).mean(), lm_max=e_lm.max(), lm_ncr=(e_lm > 0.5).mean(),
                  differ=(d != 0).mean(), maxdist=int(d.max()), batch_vs_single=int((y_np1.view(np.int64) != y_np[:20000].view(np.int64)).sum()))
    np.save(f"{out}_{f}.npy", y_np)
    print(f, res[f], flush=True)
