import math, numpy as np, gmpy2
rng = np.random.default_rng(7)
x = rng.uniform(0.0, 1e3, 200_000)
L = np.array([math.log(v) for v in x.tolist()]); n = np.log(x); d = n != L
ctx = gmpy2.context(precision=300)
for v, a, b in list(zip(x[d], n[d], L[d]))[:29]:
    with gmpy2.context(ctx):
        e = gmpy2.log(gmpy2.mpfr(v))
        u = math.ulp(float(e))
        ea, eb = float((gmpy2.mpfr(a)-e)/u), float((gmpy2.mpfr(b)-e)/u)
    m, k = math.frexp(v)
    print(f"x={v!r:24} m={2*m:.6f} numpy_err={ea:+.4f} glibc_err={eb:+.4f} mant_bits_low={int(v.hex().split('p')[0][-4:],16) if 'p' in v.hex() else 0:04x}")
# rate in uniform(512,1000) vs exp(U(ln512, ln1000))
for name, xs in (("uniform(512,1000)", rng.uniform(512, 1000, 1_000_000)),
                 ("exp(U(ln512,ln1000))", np.exp(rng.uniform(math.log(512), math.log(1000), 1_000_000)))):
    Ls = np.array([math.log(v) for v in xs.tolist()])
    print(name, "differ rate", np.mean(np.log(xs) != Ls))
