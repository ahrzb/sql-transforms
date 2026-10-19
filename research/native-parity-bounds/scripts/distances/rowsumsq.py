"""Verbatim logic of packages/sql-transform/sql_transform/native/_helpers.py:116-171
(row_sumsq and its probe), on Python floats only."""
import numpy as np


def row_sumsq(xs):
    lanes = [None, None]

    def mac(v, acc):
        sq = v * v
        return sq if acc is None else sq + acc

    n, i = len(xs), 0
    while n - i >= 8:
        for lane in (0, 1):
            for k in (3, 2, 1, 0):
                lanes[lane] = mac(xs[i + 2 * k + lane], lanes[lane])
        i += 8
    while i < n:
        for lane in (0, 1):
            if i + lane < n:
                lanes[lane] = mac(xs[i + lane], lanes[lane])
        i += 2
    a, b = lanes
    return a if b is None else a + b


def row_sumsq_is_numpys():
    from sklearn.utils.extmath import row_norms
    rng = np.random.default_rng(20261005)
    for n in range(1, 41):
        for _ in range(4):
            x = rng.normal(size=(1, n)) * 10.0 ** rng.integers(-8, 8, size=(1, n))
            if row_sumsq([float(v) for v in x[0]]) != float(row_norms(x, squared=True)[0]):
                return False
    return True
