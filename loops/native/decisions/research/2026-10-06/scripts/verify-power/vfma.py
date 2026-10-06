import sys, math, random, struct, array
random.seed(7)
N = 400000
xs = {
 'exp': [random.uniform(-700, 709) for _ in range(N)],
 'log': [10.0 ** random.uniform(-300, 300) if i % 2 else random.uniform(0.5, 2) for i in range(N)],
 'log1p': [random.uniform(0, 1e3) if i % 3 == 0 else (10.0 ** random.uniform(-15, 5) if i % 3 == 1 else random.uniform(-0.9, 1)) for i in range(N)],
 'expm1': [random.uniform(-40, 709) if i % 2 else random.uniform(-3, 3) for i in range(N)],
}
out = {}
for f, x in xs.items():
    g = getattr(math, f)
    out[f] = array.array('d', [g(v) for v in x])
import pickle
pickle.dump(out, open(sys.argv[1], 'wb'))
