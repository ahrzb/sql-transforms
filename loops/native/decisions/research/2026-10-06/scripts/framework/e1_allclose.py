"""E1d: what an allclose (atol + rtol*|twin|) test needs on the PCA fixture
lanes, and what it then lets through."""
import pickle, numpy as np
EPS = 2.0**-52
cases = pickle.load(open("pca_cases.pkl", "rb")); T = np.load("twin_default.npz")["lanes"]
N = []; S = []
for seed, est, x in cases:
    C = est.components_; sc = np.sqrt(est.explained_variance_); sc[sc < EPS] = EPS
    M = (est.mean_.reshape(1, -1) @ C.T)[0]; xa = np.array(x)
    for j in range(C.shape[0]):
        acc = x[0] * float(C[j, 0])
        for i in range(1, len(x)): acc = acc + x[i] * float(C[j, i])
        N.append((acc - M[j]) / sc[j]); S.append((np.abs(xa * C[j]).sum() + abs(M[j])) / sc[j])
N = np.array(N); S = np.array(S); f = np.isfinite(T) & np.isfinite(N)
T, N, S = T[f], N[f], S[f]; d = np.abs(T - N)
for rtol in [1e-15, 1e-12, 1e-9, 1e-7]:
    need = max(0.0, float(np.max(d - rtol * np.abs(T))))
    print(f"rtol={rtol:g}: atol needed to pass every lane {need:.3g}; lanes with |twin| < that atol (test vacuous there): {int(np.sum(np.abs(T) < need))} of {len(T)}")
print(f"|twin| spans {np.min(np.abs(T[T != 0])):.3g} .. {np.max(np.abs(T)):.3g}; max |T-N| {d.max():.3g}; max |T-N|/(eps*S) {np.max(d / (EPS * S)):.3f}")
m = np.abs(T) < 1e100
print(f"excluding the 1e300 edge rows: max |T-N| {d[m].max():.3g}, atol needed at rtol=1e-12: {max(0.0, float(np.max(d[m] - 1e-12 * np.abs(T[m])))):.3g}")
