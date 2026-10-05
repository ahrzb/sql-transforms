//! Kernels of the i128 integer lane: HUGEINT, and UBIGINT as its narrow
//! width. Both backends call these (the cranelift helpers delegate here),
//! so the semantics live in one place, as for the decimal kernels.
//!
//! Sources are DuckDB 1.5.5's: `Hugeint::TryAddInPlace` and friends
//! (hugeint.cpp), `TryIntegerCast` / `IntegerCastLoop`
//! (integer_cast_operator.hpp), `HugeIntCastData` / `HugeIntegerCastOperation`
//! (cast_operators.cpp:1800-2023), `TryCastWithOverflowCheckFloat`
//! (numeric_cast.hpp:74-84) and `ConvertFloatingToBigint` (hugeint.cpp).

use super::kernels::{duck_exp_loop, duck_is_space, duck_stoi, duck_stou64};
use crate::specializer::ir::{BinOp, Ty};

/// `+ - * // %` and the bitwise family on the i128 lane, checked where
/// DuckDB checks (texts verbatim, measured). A zero divisor never reaches
/// here: the lowering turns it into the NULL flag and feeds 1.
pub fn hugeint_arith(op: BinOp, a: i128, b: i128) -> Result<i128, String> {
    let r = match op {
        BinOp::Hadd => a.checked_add(b),
        BinOp::Hsub => a.checked_sub(b),
        BinOp::Hmul => a.checked_mul(b),
        BinOp::Hdiv => a.checked_div(b),
        BinOp::Hrem => a.checked_rem(b),
        BinOp::Hand => Some(a & b),
        BinOp::Hor => Some(a | b),
        BinOp::Hxor => Some(a ^ b),
        _ => unreachable!("{} is not an i128-lane op", op.name()),
    };
    r.ok_or_else(|| match op {
        BinOp::Hadd => format!("Overflow in addition of INT128 ({a} + {b})!"),
        BinOp::Hsub => format!("Overflow in subtraction of INT128 ({a} - {b})!"),
        BinOp::Hmul => format!("Overflow in multiplication of INT128 ({a} * {b})!"),
        // `%` too: DuckDB computes it through the checked division.
        _ => format!("Overflow in division of {a} / {b}"),
    })
}

/// abs on HUGEINT: the minimum has no positive counterpart.
pub fn hugeint_abs(v: i128) -> Result<i128, String> {
    v.checked_abs().ok_or_else(|| format!("Overflow on abs({v})"))
}

/// DOUBLE -> the i128 lane, for a value the lowering already range-checked
/// (`-2^127 < x < 2^127`, or `0 <= x < 2^64` for UBIGINT): `nearbyint`,
/// half to even, then `ConvertFloatingToBigint`, which is exact on an
/// integral double. Saturates outside the range, where it is never fed.
pub fn f64_to_hugeint(x: f64) -> i128 {
    x.round_ties_even() as i128
}

/// `TryIntegerCast`'s unsigned rule: a minus sign is accepted only when
/// every character after it is a `0` (`-0` and `-00` parse; `-0.4`, `-0 `
/// and `-1` fail), checked after the leading spaces are skipped.
pub fn unsigned_sign_ok(s: &str) -> bool {
    let b = s.as_bytes();
    let mut lo = 0;
    while lo < b.len() && duck_is_space(b[lo]) {
        lo += 1;
    }
    match b.get(lo) {
        Some(b'-') => b[lo + 1..].iter().all(|&c| c == b'0'),
        _ => true,
    }
}

/// VARCHAR -> the integer width `to`, where DuckDB's parse differs from the
/// signed int64 one (`duck_stoi`): HUGEINT parses with `HugeIntCastData`,
/// UBIGINT at a `uint64_t` store, and every unsigned width takes the sign
/// rule first. None where DuckDB's cast fails, a value outside `to`
/// included.
pub fn duck_ston(s: &str, to: Ty) -> Option<i128> {
    match to {
        Ty::I128 => duck_stohuge(s),
        Ty::U64 => {
            if !unsigned_sign_ok(s) {
                return None;
            }
            duck_stou64(s).map(i128::from)
        }
        Ty::U8 | Ty::U16 | Ty::U32 => {
            if !unsigned_sign_ok(s) {
                return None;
            }
            let (lo, hi) = to.int_range()?;
            duck_stoi(s).filter(|v| (lo..=hi).contains(v)).map(i128::from)
        }
        _ => unreachable!("ston to {}", to.name()),
    }
}

const POW10: [i128; 39] = {
    let mut t = [1i128; 39];
    let mut i = 1;
    while i < 39 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// `HugeIntCastData<hugeint_t, Hugeint, int64_t>`: digits gather in an
/// int64 that is flushed into the hugeint when full, and the fraction is
/// kept whole (its first digit rounds, half away from zero).
#[derive(Default)]
struct HugeState {
    result: i128,
    intermediate: i64,
    digits: u8,
    decimal: i128,
    decimal_total_digits: u16,
    decimal_intermediate: i128,
    decimal_intermediate_digits: u16,
}

impl HugeState {
    fn flush(&mut self) -> Option<()> {
        if self.digits == 0 && self.intermediate == 0 {
            return Some(());
        }
        if self.result != 0 {
            if self.digits > 38 {
                return None;
            }
            self.result = self.result.checked_mul(POW10[self.digits as usize])?;
        }
        self.result = self.result.checked_add(i128::from(self.intermediate))?;
        self.digits = 0;
        self.intermediate = 0;
        Some(())
    }

    fn flush_decimal(&mut self) -> Option<()> {
        if self.decimal_intermediate_digits == 0 && self.decimal_intermediate == 0 {
            return Some(());
        }
        if self.decimal != 0 {
            if self.decimal_intermediate_digits > 38 {
                return None;
            }
            self.decimal = self
                .decimal
                .checked_mul(POW10[self.decimal_intermediate_digits as usize])?;
        }
        self.decimal = self.decimal.checked_add(self.decimal_intermediate)?;
        self.decimal_total_digits += self.decimal_intermediate_digits;
        self.decimal_intermediate_digits = 0;
        self.decimal_intermediate = 0;
        Some(())
    }

    fn handle_digit(&mut self, d: u8, neg: bool) -> Option<()> {
        let d64 = i64::from(d);
        if neg {
            if self.intermediate < (i64::MIN + d64) / 10 {
                self.flush()?;
            }
            self.intermediate = self.intermediate * 10 - d64;
        } else {
            if self.intermediate > (i64::MAX - d64) / 10 {
                self.flush()?;
            }
            self.intermediate = self.intermediate * 10 + d64;
        }
        self.digits += 1;
        Some(())
    }

    fn handle_decimal(&mut self, d: u8) -> Option<()> {
        self.flush()?;
        if self.decimal_intermediate > i128::from((i64::MAX - i64::from(d)) / 10) {
            self.flush_decimal()?;
        }
        self.decimal_intermediate = self.decimal_intermediate * 10 + i128::from(d);
        self.decimal_intermediate_digits += 1;
        Some(())
    }

    fn finalize(&mut self, neg: bool) -> Option<()> {
        self.flush()?;
        self.flush_decimal()?;
        if self.decimal == 0 || self.decimal_total_digits == 0 {
            return Some(());
        }
        // CACHED_POWERS_OF_TEN - 1
        const MAX_DIGITS: u16 = 38;
        while self.decimal_total_digits > MAX_DIGITS {
            self.decimal /= POW10[MAX_DIGITS as usize];
            self.decimal_total_digits -= MAX_DIGITS;
        }
        self.decimal /= POW10[(self.decimal_total_digits - 1) as usize];
        if self.decimal >= 5 {
            self.result = if neg {
                self.result.checked_sub(1)?
            } else {
                self.result.checked_add(1)?
            };
        }
        Some(())
    }

    fn handle_exponent(&mut self, exponent: i16, neg: bool) -> Option<()> {
        self.flush()?;
        let mut e = i32::from(exponent);
        if e < -38 {
            self.result = 0;
            return Some(());
        }
        let mut remainder: i128 = 0;
        if e < 0 {
            let p = POW10[(-e) as usize];
            remainder = self.result % p;
            self.result /= p;
            if remainder < 0 {
                remainder = remainder.checked_neg()?;
            }
            self.decimal = remainder;
            self.decimal_total_digits = (-e) as u16;
            self.decimal_intermediate = 0;
            self.decimal_intermediate_digits = 0;
            return self.finalize(neg);
        }
        if self.result != 0 {
            if e > 38 {
                return None;
            }
            self.result = self.result.checked_mul(POW10[e as usize])?;
        }
        self.flush_decimal()?;
        if self.decimal == 0 {
            return self.finalize(neg);
        }
        e = i32::from(exponent) - i32::from(self.decimal_total_digits);
        if e < 0 {
            if e < -38 {
                return None;
            }
            let p = POW10[(-e) as usize];
            remainder = self.decimal % p;
            self.decimal /= p;
            self.decimal_total_digits = self.decimal_total_digits.wrapping_sub(exponent as u16);
        } else {
            if e > 38 {
                return None;
            }
            self.decimal = self.decimal.checked_mul(POW10[e as usize])?;
        }
        self.result = if neg {
            self.result.checked_sub(self.decimal)?
        } else {
            self.result.checked_add(self.decimal)?
        };
        self.decimal = remainder;
        self.finalize(neg)
    }
}

/// `TryCast::Operation(string_t, hugeint_t&)`, non-strict: `TryIntegerCast`
/// over `HugeIntegerCastOperation`. Hex and binary spellings fail (that
/// operation refuses their digits); an exponent below -38 is zero, its
/// trailing digits unread.
pub fn duck_stohuge(s: &str) -> Option<i128> {
    let b = s.as_bytes();
    let mut lo = 0;
    while lo < b.len() && duck_is_space(b[lo]) {
        lo += 1;
    }
    let b = &b[lo..];
    if b.is_empty() {
        return None;
    }
    let neg = b[0] == b'-';
    if !neg && b.len() > 1 && b[0] == b'0' && matches!(b[1] | 0x20, b'x' | b'b') {
        return None;
    }
    let start = if neg || b[0] == b'+' { 1 } else { 0 };
    let mut st = HugeState::default();
    let mut pos = start;
    while pos < b.len() {
        let c = b[pos];
        if !c.is_ascii_digit() {
            if c == b'.' {
                let number_before_period = pos > start;
                pos += 1;
                let start_digit = pos;
                while pos < b.len() && b[pos].is_ascii_digit() {
                    st.handle_decimal(b[pos] - b'0')?;
                    pos += 1;
                    if pos != b.len() && b[pos] == b'_' {
                        pos += 1;
                        if pos == b.len() || !b[pos].is_ascii_digit() {
                            return None;
                        }
                    }
                }
                if !(number_before_period || pos > start_digit) {
                    return None;
                }
                if pos >= b.len() {
                    break;
                }
            }
            if duck_is_space(b[pos]) {
                pos += 1;
                while pos < b.len() {
                    if !duck_is_space(b[pos]) {
                        return None;
                    }
                    pos += 1;
                }
                break;
            }
            if b[pos] == b'e' || b[pos] == b'E' {
                if pos == start {
                    return None;
                }
                pos += 1;
                if pos >= b.len() {
                    return None;
                }
                let exp = duck_exp_loop(&b[pos..])?;
                st.handle_exponent(exp, neg)?;
                return Some(st.result);
            }
            return None;
        }
        st.handle_digit(c - b'0', neg)?;
        pos += 1;
        if pos != b.len() && b[pos] == b'_' {
            pos += 1;
            if pos == b.len() || !b[pos].is_ascii_digit() {
                return None;
            }
        }
    }
    st.finalize(neg)?;
    if pos <= start {
        return None;
    }
    Some(st.result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each pinned against DuckDB 1.5.5 (`CAST(s AS HUGEINT)`, optimizer
    /// off): `None` is "Could not convert string".
    #[test]
    fn hugeint_parse_matches_duckdb() {
        let max = "170141183460469231731687303715884105727";
        for (s, want) in [
            (" 12 ", Some(12)),
            ("1.5", Some(2)),
            ("2.5", Some(3)),
            ("-0.5", Some(-1)),
            ("-1.5", Some(-2)),
            ("1e3", Some(1000)),
            ("1.5e1", Some(15)),
            ("1e38", Some(10i128.pow(38))),
            ("5.", Some(5)),
            (".5", Some(1)),
            ("1_000", Some(1000)),
            (max, Some(i128::MAX)),
            ("170141183460469231731687303715884105728", None),
            ("-170141183460469231731687303715884105728", Some(i128::MIN)),
            ("0x10", None),
            ("", None),
            (" ", None),
            (".", None),
            ("+7", Some(7)),
        ] {
            assert_eq!(duck_stohuge(s), want, "{s:?}");
        }
    }

    #[test]
    fn unsigned_parse_takes_the_sign_rule() {
        assert_eq!(duck_ston("-0", Ty::U64), Some(0));
        assert_eq!(duck_ston("-0.4", Ty::U64), None);
        assert_eq!(duck_ston("-0.4", Ty::U8), None);
        assert_eq!(duck_ston("18446744073709551614.5", Ty::U64), Some(u64::MAX as i128));
        assert_eq!(duck_ston("18446744073709551615.5", Ty::U64), None);
        assert_eq!(duck_ston("0xFFFFFFFFFFFFFFFF", Ty::U64), Some(u64::MAX as i128));
        assert_eq!(duck_ston("256", Ty::U8), None);
    }

    #[test]
    fn i128_overflow_traps_with_duckdbs_text() {
        assert_eq!(
            hugeint_arith(BinOp::Hadd, i128::MAX, 1).unwrap_err(),
            format!("Overflow in addition of INT128 ({} + 1)!", i128::MAX)
        );
        assert!(hugeint_arith(BinOp::Hrem, i128::MIN, -1).is_err());
        assert_eq!(hugeint_arith(BinOp::Hdiv, -7, 2), Ok(-3));
        assert_eq!(hugeint_arith(BinOp::Hrem, -7, 2), Ok(-1));
    }
}
