"""numpy's float64 exp/log (SIMD kernels on this AVX-512 host) vs glibc (math)
vs correctly rounded (Decimal at 60 digits). 200,000 draws each."""
import math, struct
from decimal import Decimal, getcontext
import numpy as np
getcontext().prec = 60

def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)

rng = np.random.default_rng(1)
for name, args, npf, mf, df in (
    ("exp", -rng.uniform(0, 40, 200000), np.exp, math.exp, lambda d: d.exp()),
    ("log", np.exp(rng.uniform(-30, 30, 200000)), np.log, math.log, lambda d: d.ln()),
):
    a = npf(args)
    mis, mu, eu_np, eu_m = 0, 0, 0, 0
    for i, x in enumerate(args):
        x = float(x)
        n, m = float(a[i]), mf(x)
        cr = float(df(Decimal(x)))
        mis += n != m
        mu = max(mu, abs(ordered(n) - ordered(m)))
        eu_np = max(eu_np, abs(ordered(n) - ordered(cr)))
        eu_m = max(eu_m, abs(ordered(m) - ordered(cr)))
    print(f"{name}: numpy != glibc on {mis}/{len(args)}, max {mu} ulp apart; "
          f"max ulps from correctly rounded: numpy {eu_np}, glibc {eu_m}")
