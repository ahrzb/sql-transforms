// Q4: SVML log8_ha's first stage, DblRcp = roundscale(rcp14(mant), 0x58),
// as a function of the mantissa m in [1,2): how many values, and where it
// parts from round-to-nearest(1/m) on the same 2^-5 grid.
#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
int main(void) {
    const uint64_t N = 1ull << 26;  // m = 1 + k/N, plus a jitter of low bits
    uint64_t differ = 0, nvals = 0; double vals[64]; double first_diff = 0, last_diff = 0;
    double maxrel = 0;
    for (uint64_t k = 0; k < N; k += 8) {
        double m8[8], r8[8], d8[8];
        for (int i = 0; i < 8; i++) { m8[i] = 1.0 + (double)(k + i) / (double)N + ((k * 2654435761u + i) % 4096) * 0x1p-52; }
        __m512d m = _mm512_loadu_pd(m8);
        __m512d r = _mm512_rcp14_pd(m);
        __m512d d = _mm512_roundscale_pd(r, 0x58);
        _mm512_storeu_pd(r8, r); _mm512_storeu_pd(d8, d);
        for (int i = 0; i < 8; i++) {
            double rel = fabs(r8[i] * m8[i] - 1.0); if (rel > maxrel) maxrel = rel;
            double ref = nearbyint(32.0 / m8[i]) / 32.0;
            if (d8[i] != ref) { differ++; if (!first_diff) first_diff = m8[i]; last_diff = m8[i]; }
            int seen = 0; for (uint64_t v = 0; v < nvals; v++) if (vals[v] == d8[i]) seen = 1;
            if (!seen && nvals < 64) vals[nvals++] = d8[i];
        }
    }
    printf("samples %llu, distinct DblRcp %llu, differ from RN(1/m) on 2^-5 grid: %llu (%.3g)\n",
           (unsigned long long)N, (unsigned long long)nvals, (unsigned long long)differ, (double)differ / N);
    printf("max |rcp14(m)*m - 1| = %.3g (2^-14 = %.3g)\n", maxrel, 0x1p-14);
    return 0;
}
