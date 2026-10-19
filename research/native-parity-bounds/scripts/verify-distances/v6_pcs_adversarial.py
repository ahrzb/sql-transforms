"""Can the report's row-wide PCS scale S = (1/D) sum_m prod_d |FFT(a_d)_m| be zero while the twin is not?
Construct x (exactly representable) with count sketch a_1 = constant, a_2 zero-sum => exact y = a1 (*) a2 = 0
in every lane and exact S = 0 (A_1 supported at m=0 only, A_2[0] = 0). Then compare twin, native(ltr conv)."""
import math
from fractions import Fraction
import numpy as np
from sklearn.kernel_approximation import PolynomialCountSketch
EPS = 2.0 ** -52
rng = np.random.default_rng(777)

def solve_exact(A, b, free_vals):
    """Solve A x = b over Q (A: list of lists of int), free vars set from free_vals."""
    m, n = len(A), len(A[0])
    M = [[Fraction(v) for v in row] + [Fraction(bb)] for row, bb in zip(A, b)]
    piv_cols, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, m) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]; M[r] = [v / pv for v in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]; M[i] = [a - f * bb for a, bb in zip(M[i], M[r])]
        piv_cols.append(c); r += 1
        if r == m: break
    assert all(all(v == 0 for v in M[i][:n]) and M[i][n] == 0 for i in range(r, m)), "inconsistent"
    x = [None] * n
    free = [c for c in range(n) if c not in piv_cols]
    for c, v in zip(free, free_vals): x[c] = Fraction(int(v))
    for i, c in enumerate(piv_cols):
        x[c] = M[i][n] - sum(M[i][k] * x[k] for k in free)
    return x

def conv_exact(a, b):
    D = len(a); return [sum(a[m] * b[(l - m) % D] for m in range(D)) for l in range(D)]
def conv_ltr(a, b):
    D = len(a); out = []
    for l in range(D):
        acc = a[0] * b[l % D]
        for m in range(1, D): acc = acc + a[m] * b[(l - m) % D]
        out.append(acc)
    return out

for D in (7, 15, 25, 50, 97, 100, 128):
    n = 6 * D
    for seed in range(1000):
        est = PolynomialCountSketch(degree=2, n_components=D, gamma=1.0, random_state=seed).fit(rng.normal(size=(50, n)))
        H, B = est.indexHash_, est.bitHash_
        if len(set(H[0].tolist())) == D: break
    # equations: a_1[b] - a_1[0] = 0 for b=1..D-1 ; a_1[0] = 1 ; sum_b a_2[b] = 0
    rows = []
    a1row = lambda b: [int(B[0, j]) if H[0, j] == b else 0 for j in range(n)]
    for b in range(D): rows.append(a1row(b))
    rhs = [1] * D
    rows.append([int(B[1, j]) for j in range(n)]); rhs.append(0)
    free_vals = rng.integers(-3, 4, size=n)
    xq = solve_exact(rows, rhs, free_vals)
    L = 1
    for v in xq: L = math.lcm(L, v.denominator)
    xq = [v * L for v in xq]                      # integers now: exactly representable
    x = [float(v) for v in xq]; assert all(Fraction(v) == q for v, q in zip(x, xq))
    tw = est.transform([x])[0]
    a = [[Fraction(0)] * D for _ in range(2)]; af = [[0.0] * D for _ in range(2)]
    for j in range(n):
        for d in range(2):
            a[d][H[d, j]] += int(B[d, j]) * xq[j]; af[d][H[d, j]] = af[d][H[d, j]] + float(B[d, j]) * x[j]
    yex = conv_exact(a[0], a[1]); ynat = conv_ltr(af[0], af[1])
    A = np.fft.fft(np.array(af), axis=1)
    S_num = float(np.prod(np.abs(A), axis=0).sum() / D)       # what a check would compute in floats
    norm_scale = math.sqrt(sum(float(v) ** 2 for v in a[0])) * math.sqrt(sum(float(v) ** 2 for v in a[1]))
    print(f"D={D:>3} n={n:>3}: a1 const={len(set(a[0]))==1} sum(a2)={sum(a[1])} | exact y all zero: {all(v == 0 for v in yex)}"
          f" | twin nonzero lanes {int((tw != 0).sum())}/{D}, max|twin|={np.abs(tw).max():.3g} | native nonzero {sum(v != 0 for v in ynat)}"
          f" | S exact=0, S in floats={S_num:.3g} | ||a1||*||a2||={norm_scale:.3g} -> K(row-wide S, float) = {np.abs(tw).max()/(EPS*S_num) if S_num else float('inf'):.3g}"
          f", K(normwise ||a1||||a2||) = {np.abs(tw).max()/(EPS*norm_scale):.3g}", flush=True)
