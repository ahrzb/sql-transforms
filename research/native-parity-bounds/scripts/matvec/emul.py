"""Emulations of summation orders, in plain IEEE double (CPython float ops are
correctly rounded, never contracted; math.fma is a correctly rounded fused
multiply-add, Python >= 3.13)."""

from __future__ import annotations

import math

fma = math.fma
NBMAX = 2048


def ltr(x, c):
    """The entry: sum(x_i * c_i) left to right, products rounded (no FMA)."""
    acc = x[0] * c[0]
    for i in range(1, len(x)):
        acc = acc + x[i] * c[i]
    return acc


def _k4x4(a, x, lo, hi):
    # AVX2 FMA: 4 interleaved accumulators, then (s0+s2)+(s1+s3)
    s = [0.0, 0.0, 0.0, 0.0]
    for i in range(lo, hi):
        l = (i - lo) & 3
        s[l] = fma(a[i], x[i], s[l])
    return (s[0] + s[2]) + (s[1] + s[3])


def _k4x2(a, x, lo, hi):
    # SSE2, unfused: 2 interleaved accumulators, then hadd
    s = [0.0, 0.0]
    for i in range(lo, hi):
        l = (i - lo) & 1
        s[l] = s[l] + a[i] * x[i]
    return s[0] + s[1]


def _k4x1(a, x, lo, hi):
    # SSE2, unfused: xmm10 holds lanes i%4 in {0,1}, xmm9 holds {2,3}
    s = [0.0, 0.0, 0.0, 0.0]
    for i in range(lo, hi):
        l = (i - lo) & 3
        s[l] = s[l] + a[i] * x[i]
    return (s[0] + s[2]) + (s[1] + s[3])


def _k4x4_generic(a, x, lo, hi):
    # dgemv_t_4.c C fallback: temp += a0*x0 + a1*x1 + a2*x2 + a3*x3 (no FMA)
    t = 0.0
    for i in range(lo, hi, 4):
        g = ((a[i] * x[i] + a[i + 1] * x[i + 1]) + a[i + 2] * x[i + 2]) + a[i + 3] * x[i + 3]
        t = t + g
    return t


def gemv_t(x, C, *, haswell=True, tail="fma_chain"):
    """OpenBLAS 0.3.33 kernel/x86_64/dgemv_t_4.c, y = C @ x with C (k, m) rows
    contiguous, alpha=1, beta=0, inc 1. Order depends on the lane's index
    (groups of 4 lanes -> 4x4 kernel, then a 4x2, then a 4x1)."""
    m = len(x)
    k = len(C)
    y = [0.0] * k
    m3 = m & 3
    m1 = m & -4
    m2 = (m & (NBMAX - 1)) - m3
    NB = NBMAX
    start = 0
    k4 = 4 * (k // 4)
    while NB == NBMAX:
        m1 -= NB
        if m1 < 0:
            if m2 == 0:
                break
            NB = m2
        lo, hi = start, start + NB
        for j in range(k):
            a = C[j]
            if j < k4:
                t = _k4x4(a, x, lo, hi) if haswell else _k4x4_generic(a, x, lo, hi)
            elif j < k4 + 2 and (k & 2):
                t = _k4x2(a, x, lo, hi)
            else:
                t = _k4x1(a, x, lo, hi)
            y[j] = y[j] + t
        start += NB
    if m3:
        i0 = start
        for j in range(k):
            a = C[j]
            ps = [(a[i0 + q], x[i0 + q]) for q in range(m3)]
            if tail == "fma_chain" and haswell:
                # GCC contraction of y += a0*x0 + a1*x1 (+ a2*x2):
                # fma(a0, x0, a1*x1), then fma(a2, x2, .), then y + .
                if m3 == 1:
                    y[j] = fma(ps[0][0], ps[0][1], y[j])
                else:
                    t = fma(ps[0][0], ps[0][1], ps[1][0] * ps[1][1])
                    if m3 == 3:
                        t = fma(ps[2][0], ps[2][1], t)
                    y[j] = y[j] + t
            else:
                t = ps[0][0] * ps[0][1]
                for aa, xx in ps[1:]:
                    t = t + aa * xx
                y[j] = y[j] + t
    return y


def ddot_skx(x, y):
    """kernel/x86_64/ddot.c + ddot_microk_skylakex-2.c (AVX-512, GCC-contracted FMA)."""
    n = len(x)
    n1 = n & -16
    dot = 0.0
    if n1:
        a5 = [[0.0] * 8 for _ in range(4)]
        n32 = n1 & ~31
        i = 0
        while i < n32:
            for q in range(4):
                for l in range(8):
                    a5[q][l] = fma(x[i + 8 * q + l], y[i + 8 * q + l], a5[q][l])
            i += 32
        acc = [[a5[q][l] + a5[q][l + 4] for l in range(4)] for q in range(4)]
        while i < n1:
            for q in range(4):
                for l in range(4):
                    acc[q][l] = fma(x[i + 4 * q + l], y[i + 4 * q + l], acc[q][l])
            i += 16
        v = [((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l] for l in range(4)]
        dot = (v[0] + v[2]) + (v[1] + v[3])
    for i in range(n1, n):
        dot = fma(y[i], x[i], dot)
    return dot


def ddot_hsw(x, y):
    """kernel/x86_64/ddot.c + ddot_microk_haswell-2.c (AVX2 FMA, 16 accumulators)."""
    n = len(x)
    n1 = n & -16
    dot = 0.0
    if n1:
        acc = [[0.0] * 4 for _ in range(4)]
        for i in range(0, n1, 16):
            for q in range(4):
                for l in range(4):
                    acc[q][l] = fma(x[i + 4 * q + l], y[i + 4 * q + l], acc[q][l])
        h = [[acc[q][l] + acc[q][l + 2] for l in range(2)] for q in range(4)]
        a01 = [h[0][l] + h[1][l] for l in range(2)]
        a23 = [h[2][l] + h[3][l] for l in range(2)]
        a = [a01[l] + a23[l] for l in range(2)]
        dot = a[0] + a[1]
    for i in range(n1, n):
        dot = dot + y[i] * x[i]  # the Haswell build does not contract (measured)
    return dot


def numpy_row_matvec(x, C, arch="SkylakeX"):
    """What numpy's (1,m) @ C.T computes: ddot for one output, gemv_t otherwise."""
    if len(C) == 1:
        d = ddot_skx if arch == "SkylakeX" else ddot_hsw
        return [0.0 + d(x, C[0])]
    if arch == "SkylakeX":
        return gemv_t(x, C)
    if arch == "Haswell":
        return gemv_t(x, C)  # gemv tail contracted, ddot tail not (measured)
    return gemv_t(x, C, haswell=False, tail="plain")
