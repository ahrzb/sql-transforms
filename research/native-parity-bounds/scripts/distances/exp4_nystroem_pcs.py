"""Nystroem(kernel='rbf') and PolynomialCountSketch: twin (1-row transform)
vs native-like SQL-order evaluations.

Nystroem lane l:  q_j = max(((-2*dot_ltr(x,b_j)) + row_sumsq(x)) + BB_j, 0)
                  k_j = exp(-gamma*q_j) (glibc), y_l = sum_ltr_j k_j N_lj
   S_l = sum_j |N_lj| k_j (1 + gamma*S2_j),  S2_j = ||x||^2+||b_j||^2+2 sum|x_i b_ji|
PolynomialCountSketch: count sketches a_d in the twin's order, then a direct
   circular convolution, left to right (degree 3: (a1*a2)*a3).
   S_term_l = (|a1| * |a2| [* |a3|])_l    per-lane term scale of the direct conv
   S_fft    = (1/D) sum_m prod_d |A_dm|, A_d = FFT(a_d)  (row-wide, normwise)
"""
import math, struct, sys, time
from fractions import Fraction
import numpy as np
from sklearn.kernel_approximation import Nystroem, PolynomialCountSketch
from sklearn.utils.extmath import row_norms
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from rowsumsq import row_sumsq

EPS = 2.0 ** -52


def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def ulps(a, b):
    return abs(ordered(a) - ordered(b))


def dot_ltr(x, c):
    acc = x[0] * c[0]
    for i in range(1, len(x)):
        acc = acc + x[i] * c[i]
    return acc


def nystroem(rng):
    for nf, data, ncomp in ((4, "std", 50), (16, "std", 100), (16, "offset10", 100), (64, "std", 100)):
        X = rng.normal(size=(2000, nf)) + (10.0 if data == "offset10" else 0.0)
        est = Nystroem(n_components=ncomp, random_state=0).fit(X)
        B = est.components_; Bl = [list(map(float, b)) for b in B]
        BB = row_norms(B, squared=True)
        N = est.normalization_
        gamma = 1.0 / nf  # rbf_kernel's default, Nystroem passes no gamma
        rows = np.vstack([X[rng.choice(2000, 300, replace=False)],
                          B[rng.integers(0, len(B), 100)] + 1e-9 * rng.normal(size=(100, nf))])
        K, U, exp_mis = [], [], 0
        Kmag = []
        for r in rows:
            x = [float(v) for v in r]
            twin = est.transform([x])[0]
            xx = row_sumsq(x)
            q = np.array([max((-2.0 * dot_ltr(x, Bl[j]) + xx) + BB[j], 0.0) for j in range(len(B))])
            arg = q * -gamma  # twin: K *= -gamma
            k = np.array([math.exp(a) for a in arg])
            exp_mis += int((np.exp(arg) != k).sum())
            y = k[0] * N[:, 0]
            for j in range(1, len(B)):
                y = y + k[j] * N[:, j]
            S2 = np.array([sum(v * v for v in x) + BB[j] + 2 * sum(abs(a * c) for a, c in zip(x, Bl[j])) for j in range(len(B))])
            S = (np.abs(N) * (k * (1 + gamma * S2))[None, :]).sum(axis=1)
            Smv = (np.abs(N) * k[None, :]).sum(axis=1)  # a plain matvec term scale, ignoring the kernel's error
            d = np.abs(y - twin)
            K.append(d / (EPS * S)); Kmag.append(d / (EPS * Smv))
            U.extend(ulps(float(a), float(b)) for a, b in zip(y, twin))
        K = np.concatenate(K); Kmag = np.concatenate(Kmag); U = np.array(U, float)
        print(f"Nystroem rbf nf={nf} {data} n_comp={ncomp:<4} lanes={K.size:>6} ulp max={U.max():.3g} p99={np.percentile(U,99):.3g}"
              f" | K(S incl. kernel) max={K.max():.3f} p99={np.percentile(K,99):.3f} | K(S matvec only) max={Kmag.max():.3g}"
              f" | np.exp!=glibc {exp_mis} | max|N|={np.abs(N).max():.3g}", flush=True)


def conv_ltr(a, b):
    D = len(a)
    out = []
    for l in range(D):
        acc = a[0] * b[l % D]
        for m in range(1, D):
            acc = acc + a[m] * b[(l - m) % D]
        out.append(acc)
    return out


def conv_exact(a, b):
    D = len(a)
    A = [Fraction(v) for v in a]; Bf = [Fraction(v) for v in b]
    return [sum(A[m] * Bf[(l - m) % D] for m in range(D)) for l in range(D)]


def pcs(rng):
    for nf, D, deg, coef0, nrows in ((4, 100, 2, 0, 200), (32, 100, 2, 0, 200), (32, 128, 2, 1, 200),
                                     (32, 97, 2, 0, 200), (16, 64, 3, 0, 100), (200, 100, 2, 0, 100)):
        X = rng.normal(size=(500, nf))
        est = PolynomialCountSketch(degree=deg, n_components=D, coef0=coef0, random_state=0).fit(X)
        Kt, Kf, U, zero_lanes, twin_nonzero_on_zero = [], [], [], 0, 0
        Kte, Kfe = [], []
        for r in X[:nrows]:
            x = [float(v) for v in r]
            twin = est.transform([x])[0]
            xg = list(np.sqrt(est.gamma) * np.asarray([x])[0])
            if coef0 != 0:
                xg.append(float(np.sqrt(est.coef0) * 1.0))
            cs = [[0.0] * D for _ in range(deg)]
            for j in range(len(xg)):
                for d in range(deg):
                    h = est.indexHash_[d, j]; bit = est.bitHash_[d, j]
                    cs[d][h] = cs[d][h] + bit * xg[j]
            y = conv_ltr(cs[0], cs[1])
            st = conv_ltr([abs(v) for v in cs[0]], [abs(v) for v in cs[1]])
            if deg == 3:
                y = conv_ltr(y, cs[2]); st = conv_ltr(st, [abs(v) for v in cs[2]])
            y = np.array(y); st = np.array(st)
            A = np.fft.fft(np.array(cs), axis=1)
            Sf = (np.prod(np.abs(A), axis=0)).sum() / D
            d = np.abs(y - twin)
            with np.errstate(divide="ignore", invalid="ignore"):
                kt = np.where(d == 0, 0.0, d / (EPS * st))
            Kt.append(kt); Kf.append(d / (EPS * Sf))
            zero_lanes += int((st == 0).sum())
            twin_nonzero_on_zero += int(((st == 0) & (twin != 0)).sum())
            U.extend(ulps(float(a), float(b)) for a, b in zip(y, twin))
            if deg == 2 and nrows <= 200 and len(Kte) < 40:  # twin vs exact on a subsample
                ex = conv_exact(cs[0], cs[1])
                de = np.array([abs(float(Fraction(float(t)) - e)) for t, e in zip(twin, ex)])
                with np.errstate(divide="ignore", invalid="ignore"):
                    Kte.append(np.where(de == 0, 0.0, de / (EPS * st)))
                Kfe.append(de / (EPS * Sf))
        Kt = np.concatenate(Kt); Kf = np.concatenate(Kf); U = np.array(U, float)
        extra = ""
        if Kte:
            Kte = np.concatenate(Kte); Kfe = np.concatenate(Kfe)
            extra = f" | twin-vs-exact: K_term max={Kte.max():.3g}, K_fft max={Kfe.max():.3f}"
        print(f"PolyCountSketch nf={nf} D={D} deg={deg} coef0={coef0} lanes={Kt.size:>6} ulp max={U.max():.3g} p99={np.percentile(U,99):.3g}"
              f" | K_term max={Kt.max():.3g} | K_fft max={Kf.max():.3f} p99={np.percentile(Kf,99):.3f}"
              f" | lanes with S_term=0: {zero_lanes}, twin!=0 there: {twin_nonzero_on_zero}{extra}", flush=True)


t0 = time.time()
rng = np.random.default_rng(3)
nystroem(rng)
pcs(rng)
print(f"elapsed {time.time()-t0:.1f}s")
