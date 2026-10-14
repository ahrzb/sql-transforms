//! The arithmetic of DuckDB's `avg` over integers and DECIMAL (DuckDB
//! 1.5.5, `avg.cpp`; measured 2026-10-14). DuckDB keeps the exact sum and
//! the count, and divides once at the end: in `double` for SMALLINT and a
//! DECIMAL of width 4 or less, in C `long double` for the other types
//! ([`plan::avg_in_long_double`]). On x86_64 Linux `long double` is the x87
//! extended format, a 64-bit significand rounded to nearest, ties to even.
//! This module computes that format exactly in integers, so that its answer
//! does not depend on the floating point of the machine that runs confit.

use crate::specializer::ir::Ty;
use crate::specializer::plan;

/// A nonzero extended value `sig * 2^exp`, `sig` normalized (top bit set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ext {
    neg: bool,
    sig: u64,
    exp: i32,
}

/// `sig + 1` as a normalized significand: on a carry out of 64 bits, the
/// next power of two.
fn bump(sig: u64, exp: i32) -> (u64, i32) {
    match sig.checked_add(1) {
        Some(s) => (s, exp),
        None => (1 << 63, exp + 1),
    }
}

/// `mag * 2^exp` rounded to 64 bits; `mag` is nonzero.
fn round(neg: bool, mag: u128, exp: i32) -> Ext {
    let bits = 128 - mag.leading_zeros() as i32;
    if bits <= 64 {
        let sh = 64 - bits;
        return Ext {
            neg,
            sig: (mag << sh) as u64,
            exp: exp - sh,
        };
    }
    let k = bits - 64;
    let (mut sig, mut exp) = ((mag >> k) as u64, exp + k);
    let rem = mag & ((1u128 << k) - 1);
    let half = 1u128 << (k - 1);
    if rem > half || (rem == half && sig & 1 == 1) {
        (sig, exp) = bump(sig, exp);
    }
    Ext { neg, sig, exp }
}

/// A finite positive double, exactly.
fn of_f64(x: f64) -> Ext {
    let bits = x.to_bits();
    let (e, m) = (((bits >> 52) & 0x7ff) as i32, bits & ((1 << 52) - 1));
    debug_assert!(e != 0 && e != 0x7ff && x > 0.0, "a normal positive double");
    round(false, u128::from(m | 1 << 52), e - 1075)
}

fn mul(a: Ext, b: Ext) -> Ext {
    round(
        a.neg != b.neg,
        u128::from(a.sig) * u128::from(b.sig),
        a.exp + b.exp,
    )
}

/// `a / b` rounded to 64 bits. The truncated quotient of the significands
/// has 64 or 65 bits; the remainder decides the rounding.
fn div(a: Ext, b: Ext) -> Ext {
    let (num, d) = (u128::from(a.sig) << 64, u128::from(b.sig));
    let (q, r) = (num / d, num % d);
    let neg = a.neg != b.neg;
    let exp = a.exp - b.exp - 64;
    let (mut sig, mut exp, up) = if q >> 64 == 0 {
        // The quotient is q + r/d: round on r/d against one half.
        let sig = q as u64;
        (sig, exp, 2 * r > d || (2 * r == d && sig & 1 == 1))
    } else {
        // Drop the low bit of q: it and r/d make the rest.
        let sig = (q >> 1) as u64;
        (sig, exp + 1, q & 1 == 1 && (r != 0 || sig & 1 == 1))
    };
    if up {
        (sig, exp) = bump(sig, exp);
    }
    Ext { neg, sig, exp }
}

/// The double nearest to `x`, ties to even. `x` is in the normal range of
/// a double here: an average of values below 2^128 over a divisor of at
/// least 1, and above 10^-38 in magnitude.
fn to_f64(x: Ext) -> f64 {
    let low = x.sig & 0x7ff;
    let (mut m, mut exp) = (x.sig >> 11, x.exp + 11);
    if low > 0x400 || (low == 0x400 && m & 1 == 1) {
        m += 1;
        if m == 1 << 53 {
            (m, exp) = (1 << 52, exp + 1);
        }
    }
    // 2^exp, exactly: m * 2^exp is in the normal range.
    let pow = f64::from_bits(((exp + 1023) as u64) << 52);
    let v = m as f64 * pow;
    if x.neg {
        -v
    } else {
        v
    }
}

/// `avg` of `n` (at least 1) values of a column of `src` whose exact sum is
/// `sum`, in units of the column: `10^-s` for a DECIMAL(p, s).
pub fn finalize(src: Ty, sum: i128, n: u64) -> f64 {
    let scale = match src {
        Ty::Dec(_, s) => s,
        _ => 0,
    };
    if plan::avg_in_long_double(src) {
        long_double(sum, n, scale)
    } else {
        double(sum, n, scale)
    }
}

/// `avg` of `n` values whose exact sum is `sum` units of `10^-scale`, the
/// way DuckDB finalizes it in `long double`: the sum rounded to 64 bits,
/// divided by `n * (double) 10^scale` rounded to 64 bits, the quotient
/// rounded to 64 bits and then to a double.
fn long_double(sum: i128, n: u64, scale: u8) -> f64 {
    if sum == 0 {
        return 0.0;
    }
    let num = round(sum < 0, sum.unsigned_abs(), 0);
    let mut den = round(false, u128::from(n), 0);
    if scale > 0 {
        den = mul(den, of_f64(pow10_f64(scale)));
    }
    to_f64(div(num, den))
}

/// `avg` as DuckDB finalizes it in `double`, for SMALLINT and a DECIMAL of
/// width 4 or less: `(double) sum / ((double) n * (double) 10^scale)`. The
/// sum fits an i64, DuckDB's state for these, and converts as one does.
fn double(sum: i128, n: u64, scale: u8) -> f64 {
    let den = if scale > 0 {
        n as f64 * pow10_f64(scale)
    } else {
        n as f64
    };
    sum as f64 / den
}

/// `10^scale` rounded to a double once, as DuckDB's
/// `Hugeint::Cast<double>(POWERS_OF_TEN[scale])` gives it.
fn pow10_f64(scale: u8) -> f64 {
    10u128.pow(u32::from(scale)) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_quotients_are_the_plain_division() {
        assert_eq!(long_double(7, 2, 0), 3.5);
        assert_eq!(long_double(-7, 2, 0), -3.5);
        assert_eq!(long_double(1, 3, 0), 1.0 / 3.0);
        assert_eq!(long_double(12345, 1, 2), 123.45);
        assert_eq!(double(7, 2, 0), 3.5);
        assert_eq!(double(12345, 1, 2), 123.45);
    }

    // Groups whose x87 answer, DuckDB's, differs from the exact quotient
    // rounded once (in the comment) and from the double division: (sum, n,
    // scale, bits of DuckDB 1.5.5's answer), measured 2026-10-14.
    #[test]
    fn long_double_answers_what_duckdb_answers() {
        let cases: &[(i128, u64, u8, u64)] = &[
            // BIGINT, one rounding 0x43dfff89ea4d70cf
            (110674232335275571201, 12, 0, 0x43dfff89ea4d70ce),
            // BIGINT, one rounding 0x43dfff7fb7cefc25
            (83005270619952858622, 9, 0, 0x43dfff7fb7cefc26),
            // BIGINT, one rounding 0xc3dfff5ff2381873
            (-73781344901274685441, 8, 0, 0xc3dfff5ff2381872),
            // INTEGER, one rounding 0x41b70769a658e8cb
            (1410606808934, 3651, 0, 0x41b70769a658e8ca),
            // INTEGER, one rounding 0xc1b319deddb5c627
            (-876465095337, 2735, 0, 0xc1b319deddb5c628),
            // INTEGER, one rounding 0x41c9cde876872541
            (2326537824297, 2687, 0, 0x41c9cde876872540),
            // DECIMAL(38,25), one rounding 0x41803e4076804818
            (6131591786274750890610199031913676, 18, 25, 0x41803e4076804817),
            // DECIMAL(38,25), one rounding 0xc18768842e79b857
            (-13254487328547079665731872093156085, 27, 25, 0xc18768842e79b856),
            // DECIMAL(18,4), one rounding 0xc2c5f55b1bedf71b
            (-1446198029699000134508, 2995, 4, 0xc2c5f55b1bedf71c),
            // DECIMAL(18,4), one rounding 0x42a8a834f1ef165b
            (542078883250112316643, 3999, 4, 0x42a8a834f1ef165c),
        ];
        for &(sum, n, scale, bits) in cases {
            let got = long_double(sum, n, scale).to_bits();
            assert_eq!(got, bits, "{sum} / {n} at scale {scale}: {got:#x}");
        }
    }

    // SMALLINT and DECIMAL(4, s): DuckDB divides in double, where x87
    // would answer otherwise for these groups.
    #[test]
    fn double_answers_what_duckdb_answers() {
        let cases: &[(i128, u64, u8, u64)] = &[
            (-16145859, 2471, 0, 0xc0b98623be1bf8f5),
            (-37651911, 2495, 0, 0xc0cd7979201d8ce1),
            (-18943537, 3936, 2, 0xc048107ff557ff55),
            (-10315309, 2639, 2, 0xc0438b41d220a9fd),
        ];
        for &(sum, n, scale, bits) in cases {
            assert_eq!(double(sum, n, scale).to_bits(), bits, "{sum} / {n}");
            assert_ne!(long_double(sum, n, scale).to_bits(), bits);
        }
    }
}
