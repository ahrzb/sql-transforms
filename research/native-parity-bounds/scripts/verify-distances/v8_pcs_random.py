import math, numpy as np
from fractions import Fraction
from sklearn.kernel_approximation import PolynomialCountSketch
EPS = 2.0 ** -52
rng = np.random.default_rng(99)
for n, D, deg in ((4, 100, 2), (5, 97, 2), (24, 100, 2), (10, 64, 3), (300, 100, 2)):
    X = rng.lognormal(size=(300, n))
    est = PolynomialCountSketch(degree=deg, n_components=D, random_state=5).fit(X)
    H, B = est.indexHash_, est.bitHash_
    Kf, Kn, zl, tz = [], [], 0, 0
    for r in X[:150]:
        x = [float(v) for v in r]; tw = est.transform([x])[0]
        a = [[Fraction(0)] * D for _ in range(deg)]
        for j in range(n):
            for d in range(deg): a[d][H[d, j]] += int(B[d, j]) * Fraction(x[j])
        y = a[0]
        for d in range(1, deg): y = [sum(y[mm] * a[d][(l - mm) % D] for mm in range(D)) for l in range(D)]
        A = np.fft.fft(np.array([[float(v) for v in ad] for ad in a]), axis=1)
        S = float(np.prod(np.abs(A), axis=0).sum() / D)
        Sn = math.prod(math.sqrt(sum(float(v) ** 2 for v in ad)) for ad in a)
        err = np.array([abs(float(Fraction(float(t)) - e)) for t, e in zip(tw, y)])
        Kf.append(err / (EPS * S)); Kn.append(err / (EPS * Sn))
        zl += sum(1 for v in y if v == 0); tz += sum(1 for v, t in zip(y, tw) if v == 0 and t != 0)
    Kf, Kn = np.concatenate(Kf), np.concatenate(Kn)
    print(f"n={n} D={D} deg={deg}: lanes={Kf.size} exact-zero lanes={zl} twin!=0 there={tz} | twin-vs-exact K(row-wide S) max={Kf.max():.3f} | K(prod ||a_d||_2) max={Kn.max():.3f}", flush=True)
