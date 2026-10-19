import sys, numpy as np, scipy.stats as st
rng = np.random.default_rng(99)
lam = np.concatenate([rng.uniform(-3, 5, 200), [0.0, 2.0, 1.0, 0.5, -1.0, 1e-17, 2 + 1e-15, 3.0, -2.5]])
y = np.concatenate([rng.uniform(-1e3, 1e3, 1000), rng.normal(0, 1, 1000), np.exp(rng.uniform(-35, 35, 1500)) * rng.choice([-1, 1], 1500),
                    rng.normal(0, 1e-8, 500)])
T = np.vstack([st.yeojohnson(y, float(l)) for l in lam])
# row-by-row (1-element arrays), on a subset, as the served twin calls it
sub = [(i, k) for i in range(0, len(lam), 23) for k in range(0, len(y), 7)]
R = np.array([st.yeojohnson(y[k:k + 1], float(lam[i]))[0] for i, k in sub])
np.savez(sys.argv[1], lam=lam, y=y, T=T, R=R, sub=np.array(sub))
print("1-elem vs batch differ:", int((R != np.array([T[i, k] for i, k in sub])).sum()), "of", len(sub))
