import math, pickle, numpy as np
EPS = 2.0 ** -52
L = pickle.load(open("v1_lanes.pkl", "rb"))
T = np.array([l["tw"] for l in L]); N = np.array([l["ent"] for l in L]); S = np.array([l["S"] for l in L])
f = np.isfinite(T) & np.isfinite(N); T, N, S = T[f], N[f], S[f]; d = np.abs(T - N)
print("lanes", len(T), "|T| range", np.abs(T[T != 0]).min(), np.abs(T).max(), "S range", S.min(), S.max())
for rtol in [0, 1e-15, 1e-12, 1e-9, 1e-7]:
    need = max(0.0, float(np.max(d - rtol * np.abs(T))))
    print(f"rtol={rtol:g}: atol needed {need:.3g}; lanes with |T| < atol: {int(np.sum(np.abs(T) < need))}; lanes where atol >= K=3 bound (3 eps S): {int(np.sum(need >= 3*EPS*S))}")
big = S > 1e100
print("lanes with S > 1e100:", int(big.sum()))
m = ~big
for rtol in [1e-15, 1e-12]:
    need = max(0.0, float(np.max(d[m] - rtol * np.abs(T[m]))))
    print(f"  excl S>1e100, rtol={rtol:g}: atol needed {need:.3g}")
# default numpy assert_allclose (rtol 1e-7, atol 0) and sklearn's (rtol 1e-7, atol 1e-7): failures?
for rt, at in [(1e-7, 0), (1e-7, 1e-7), (1e-7, 1e-9)]:
    print(f"assert_allclose(rtol={rt}, atol={at}) failing lanes: {int(np.sum(d > at + rt*np.abs(T)))}")
