import sys, math
import numpy as np
fma = math.fma
rng = np.random.default_rng(5)
V = {
 'A: y+fma(a0,x0,a1x1)->fma(a2,x2,.)': lambda p: 0.0 + (fma(p[2][0],p[2][1], fma(p[0][0],p[0][1],p[1][0]*p[1][1])) if len(p)==3 else fma(p[0][0],p[0][1],p[1][0]*p[1][1])),
 'B: fma chain from a0x0': lambda p: 0.0 + (fma(p[2][0],p[2][1], fma(p[1][0],p[1][1],p[0][0]*p[0][1])) if len(p)==3 else fma(p[1][0],p[1][1],p[0][0]*p[0][1])),
 'C: fma(a0,x0, fma(a1,x1,a2x2))': lambda p: 0.0 + (fma(p[0][0],p[0][1], fma(p[1][0],p[1][1],p[2][0]*p[2][1])) if len(p)==3 else fma(p[0][0],p[0][1],p[1][0]*p[1][1])),
 'D: y=fma chain into y': lambda p: (fma(p[2][0],p[2][1], fma(p[1][0],p[1][1], fma(p[0][0],p[0][1],0.0))) if len(p)==3 else fma(p[1][0],p[1][1], fma(p[0][0],p[0][1],0.0))),
 'E: plain': lambda p: 0.0 + (((p[0][0]*p[0][1] + p[1][0]*p[1][1]) + p[2][0]*p[2][1]) if len(p)==3 else (p[0][0]*p[0][1] + p[1][0]*p[1][1])),
}
for m in (2, 3):
  for k in (2, 3, 4, 5, 8):
    hits = {v: 0 for v in V}; tot = 0
    for t in range(300):
        C = rng.standard_normal((k, m)); x = rng.standard_normal(m)
        y = (x[None, :] @ C.T)[0]
        for j in range(k):
            p = [(C[j, q], x[q]) for q in range(m)]
            tot += 1
            for v, f in V.items(): hits[v] += f(p) == y[j]
    print(m, k, {v[:1]: h for v, h in hits.items()}, tot)
