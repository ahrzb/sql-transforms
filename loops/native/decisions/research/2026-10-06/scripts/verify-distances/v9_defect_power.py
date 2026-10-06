"""Detection power of the proposed distance check |dn^2-dt^2| <= K eps S2 with K = n+4 (derived) or 2.5 (measured):
which share of lanes flags a native that (a) rounds its output to float32, (b) computes ||x||^2 in float32,
(c) drops the last feature from the dot (and from ||x||^2)?"""
import math
from fractions import Fraction
import numpy as np
from sklearn.cluster import KMeans
EPS = 2.0 ** -52
rng = np.random.default_rng(11)
for n, off, label in ((8, 0.0, "centred"), (8, 1e3, "offset 1e3"), (32, 0.0, "centred"), (32, 50.0, "offset 50")):
    X = rng.normal(size=(1500, n)) + rng.integers(0, 5, size=(1500, 1)) * 4 + off
    km = KMeans(n_clusters=5, n_init=1, random_state=0).fit(X); C = km.cluster_centers_
    rows = np.vstack([X[rng.choice(1500, 200, replace=False)], C[rng.integers(0, 5, 100)] + 1e-3 * rng.normal(size=(100, n))])
    flags = {k: [0, 0] for k in ("f32out", "f32xx", "drop")}; tot = 0
    for r in rows:
        tw = km.transform([list(map(float, r))])[0]
        for k in range(5):
            c = C[k]; x = r
            S2 = float(x @ x + c @ c + 2 * np.abs(x * c).sum()); dt = float(tw[k])
            exact_q = float(((x - c) ** 2).sum())
            cands = {
                "f32out": float(np.float32(dt)),
                "f32xx": math.sqrt(max(-2 * float(x @ c) + float(np.float32(x @ x)) + float(c @ c), 0.0)),
                "drop": math.sqrt(max(-2 * float(x[:-1] @ c[:-1]) + float(x[:-1] @ x[:-1]) + float(c @ c), 0.0)),
            }
            for name, dn in cands.items():
                r2 = abs(float(Fraction(dn) ** 2 - Fraction(dt) ** 2)) / (EPS * S2)
                flags[name][0] += r2 > (n + 4); flags[name][1] += r2 > 2.5
            tot += 1
    print(f"n={n} {label:<10} lanes={tot}: " + " | ".join(f"{k}: flagged {100*v[0]/tot:5.1f}% at K=n+4, {100*v[1]/tot:5.1f}% at K=2.5" for k, v in flags.items()))
