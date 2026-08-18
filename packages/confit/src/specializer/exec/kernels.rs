//! The shared runtime library: DuckDB-exact scalar semantics.
//!
//! These are the kernels BOTH backends run — the interpreter calls them
//! directly, and the Cranelift backend emits calls to them, exactly as a
//! C compiler emits a call to libm for `log()` rather than open-coding it.
//! Nothing here interprets anything: this is the semantics of the SQL
//! functions themselves, pinned against DuckDB, plus the small runtime
//! support a compiled program needs (output reservation, UDF dispatch).
//!
//! Every function here is oracle-pinned; changing one changes what the
//! engine SERVES, on both backends at once.

use super::super::ir::{BinOp, NumOp1, StrOp2, TrimSide, Ty};
use super::{OutCol, ScalarVal, Trap};


pub(super) fn reserve_out(out: &mut [OutCol], rows: usize) {
    for col in out {
        match col {
            OutCol::I1(v) => v.reserve(rows),
            OutCol::I64(v) => v.reserve(rows),
            OutCol::F64(v) => v.reserve(rows),
            OutCol::Str(v) => v.reserve(rows),
        }
    }
}

/// Execute one extern (UDF) call and enforce the declared return shape —
/// THE shared function behind `ecall` on both backends (the cranelift
/// helper delegates here, so the backends cannot drift). Returns the
/// whole-call validity plus one (validity, payload) pair per declared
/// return; payloads under a false flag are the type default. A raised
/// error or a result violating the declaration is a named trap.
pub(super) fn call_extern(
    imp: &super::ExternImpl,
    rets: &[Ty],
    args: &[Option<ScalarVal>],
) -> Result<(bool, Vec<(bool, ScalarVal)>), Trap> {
    let default = |ty: Ty| match ty {
        Ty::I1 => ScalarVal::I1(false),
        Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => ScalarVal::I64(0),
        Ty::F64 => ScalarVal::F64(0.0),
        Ty::Str => ScalarVal::Str(String::new()),
    };
    match (imp.fun)(args).map_err(Trap)? {
        // Whole-call NULL: every component flag false.
        None => Ok((false, rets.iter().map(|&t| (false, default(t))).collect())),
        Some(vals) => {
            if vals.len() != rets.len() {
                return Err(Trap(format!(
                    "udf '{}' returned {} value(s), declared {}",
                    imp.name,
                    vals.len(),
                    rets.len()
                )));
            }
            let mut out = Vec::with_capacity(rets.len());
            for (j, (v, &ty)) in vals.into_iter().zip(rets).enumerate() {
                match v {
                    None => out.push((false, default(ty))),
                    Some(sv) if sv.ty() == ty => out.push((true, sv)),
                    Some(sv) => {
                        return Err(Trap(format!(
                            "udf '{}' output {j} is {}, declared {}",
                            imp.name,
                            sv.ty().name(),
                            ty.name()
                        )))
                    }
                }
            }
            Ok((true, out))
        }
    }
}

/// DuckDB's substr window arithmetic (measured 1.5.5), on codepoints — NOT
/// grapheme clusters (substr slices inside ZWJ emoji). 1-based virtual
/// positions: negative start counts from the end (`start = n + start + 1`),
/// start <= 0 consumes length before character 1, negative len is "". A
/// missing SQL length arrives as i64::MAX; the saturating add makes that
/// "rest of the string".
/// DuckDB's substr window arithmetic — the VECTORIZED path, which columns
/// (and therefore every real query and the mined corpus) take; DuckDB's own
/// constant-fold path disagrees with it on negative starts (measured
/// 2026-07-26, see the builtin-pins spec). Codepoints, NOT grapheme
/// clusters. 1-based positions: a negative start counts from the end
/// (`rs = n + start + 1`, NOT clamped -- see the body) while start 0 stays
/// virtual;
/// a non-negative length runs forward `[rs, rs+len)`, a NEGATIVE length
/// slices BACKWARDS `[rs+len, rs)`; `len: None` is the 2-arg rest-of-string
/// form.
pub(super) fn substr_window(s: &str, start: i64, len: Option<i64>) -> std::ops::Range<usize> {
    let n = s.chars().count() as i64;
    // A negative start is END-RELATIVE, and the mapped position is NOT clamped
    // to 1 here: clamping before the window is applied would drop the window's
    // END reduction, so `substr('hello', -10, 8)` would answer 'hello' instead
    // of 'hel'. The intersection below is what clamps, exactly as it does for
    // the `start = 0` case which shares this rule.
    //
    // This used to clamp (`.max(1)`), pinned as builtin-pins §4 "negative start
    // clamps to 1 BEFORE the length window". That pin measured the one DuckDB
    // path that disagrees with its own other three. Measured 2026-08-17,
    // `substr(s, -10, 8)` over 'hello':
    //
    //   optimizer ON,  literal args   'hel'
    //   optimizer ON,  column args    'hello'   <- the outlier, and the pin
    //   optimizer OFF, literal args   'hel'
    //   optimizer OFF, column args    'hel'
    //
    // so the clamp reproduced a DuckDB self-inconsistency. The window rule
    // agrees with three of its four paths, including its own constant folder.
    let rs = if start < 0 { n + start + 1 } else { start };
    let (lo, hi) = match len {
        Some(l) if l >= 0 => (rs, rs.saturating_add(l)),
        Some(l) => (rs.saturating_add(l), rs),
        None => (rs, n + 1),
    };
    let (lo, hi) = (lo.max(1), hi.min(n + 1));
    if hi <= lo {
        return 0..0;
    }
    // Byte range of the char window — the output is a subview of `s`, so
    // callers slice instead of copying.
    let skip = (lo - 1) as usize;
    let take = (hi - lo) as usize;
    let b0 = s
        .char_indices()
        .nth(skip)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let b1 = s[b0..]
        .char_indices()
        .nth(take)
        .map(|(i, _)| b0 + i)
        .unwrap_or(s.len());
    b0..b1
}

/// The byte range of `s` that survives trimming `set` chars from the chosen
/// ends — pure arithmetic, the output aliases the input. Membership scans
/// `set` per char (a trim set is a handful of chars; no Vec).
pub(super) fn trim_bounds(s: &str, set: &str, side: TrimSide) -> std::ops::Range<usize> {
    let hit = |c: char| set.chars().any(|k| k == c);
    let t = match side {
        TrimSide::Both => s.trim_matches(hit),
        TrimSide::Lead => s.trim_start_matches(hit),
        TrimSide::Trail => s.trim_end_matches(hit),
    };
    let start = t.as_ptr() as usize - s.as_ptr() as usize;
    start..start + t.len()
}

/// The offset/length guard DuckDB applies before the window: values outside
/// [-2^32, 2^32-1] raise an Out of Range error (measured boundary-exactly).
pub(super) fn substr_range_ok(v: i64) -> bool {
    (-(1i64 << 32)..(1i64 << 32)).contains(&v)
}

/// DuckDB's DOUBLE -> VARCHAR text (measured 1.5.5): Rust's shortest
/// round-trip form, except the exponent carries an explicit sign and at
/// least two digits (`1e+300`, `1e-05`) and NaN is lowercase `nan`.
pub(super) struct DuckF64(pub(super) f64);

impl std::fmt::Display for DuckF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_nan() {
            return f.write_str("nan");
        }
        // Stack-render the shortest round-trip form (≤ 24 bytes for any
        // f64) so the hot path never builds a temp String.
        let mut buf = StackStr::<32>::default();
        {
            use std::fmt::Write;
            write!(buf, "{:?}", self.0).expect("f64 debug fits 32 bytes");
        }
        let s = buf.as_str();
        match s.find('e') {
            None => f.write_str(s),
            Some(pos) => {
                let exp: i64 = s[pos + 1..].parse().expect("float exponent");
                write!(
                    f,
                    "{}e{}{:02}",
                    &s[..pos],
                    if exp < 0 { '-' } else { '+' },
                    exp.abs()
                )
            }
        }
    }
}

/// Fixed-capacity ASCII scratch for `write!` — errors instead of growing.
struct StackStr<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> Default for StackStr<N> {
    fn default() -> Self {
        StackStr {
            buf: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> StackStr<N> {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len]).expect("writes were valid UTF-8")
    }
}

impl<const N: usize> std::fmt::Write for StackStr<N> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let b = s.as_bytes();
        if self.len + b.len() > N {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..self.len + b.len()].copy_from_slice(b);
        self.len += b.len();
        Ok(())
    }
}

fn log_guard(x: f64) -> Result<(), Trap> {
    // Comparison-based, not is_finite-based: NaN fails both checks and
    // flows through to libm; +inf passes (measured).
    if x == 0.0 {
        return Err(Trap("cannot take logarithm of zero".into()));
    }
    if x < 0.0 {
        return Err(Trap("cannot take logarithm of a negative number".into()));
    }
    Ok(())
}

pub(super) fn duck_ln(x: f64) -> Result<f64, Trap> {
    log_guard(x)?;
    Ok(if x.is_nan() { x } else { x.ln() })
}

pub(super) fn duck_log2(x: f64) -> Result<f64, Trap> {
    log_guard(x)?;
    Ok(if x.is_nan() { x } else { x.log2() })
}

pub(super) fn duck_log10(x: f64) -> Result<f64, Trap> {
    log_guard(x)?;
    Ok(if x.is_nan() { x } else { x.log10() })
}

/// log(base, x): bit-exactly log10(x)/log10(base) — NOT an ln ratio
/// (refuted on 20k fuzz samples). Base is domain-checked FIRST; base==1
/// carries DuckDB's message verbatim, typo included.
pub(super) fn duck_logb(base: f64, x: f64) -> Result<f64, Trap> {
    log_guard(base)?;
    if base == 1.0 {
        return Err(Trap("divison by zero in based logarithm".into()));
    }
    log_guard(x)?;
    Ok(x.log10() / base.log10())
}

pub(super) fn duck_exp(x: f64) -> Result<f64, Trap> {
    // TOTAL: overflow -> inf, underflow -> denormals -> +0.0 (measured).
    Ok(x.exp())
}

pub(super) fn duck_sqrt(x: f64) -> Result<f64, Trap> {
    if x < 0.0 {
        return Err(Trap("cannot take square root of a negative number".into()));
    }
    Ok(x.sqrt()) // sqrt(-0.0) = -0.0 (not negative, not a trap); NaN passes
}

pub(super) fn duck_cbrt(x: f64) -> Result<f64, Trap> {
    Ok(x.cbrt()) // TOTAL; cbrt(-8) = -2.0 exactly (NOT pow(x, 1/3))
}

fn trig_guard(x: f64) -> Result<(), Trap> {
    if x.is_infinite() {
        return Err(Trap(format!(
            "input value {} is out of range for numeric function",
            if x > 0.0 { "inf" } else { "-inf" }
        )));
    }
    Ok(())
}

pub(super) fn duck_sin(x: f64) -> Result<f64, Trap> {
    trig_guard(x)?;
    // NaN passes through BIT-EXACTLY (payload + sign) — never hand it to
    // libm (measured: DuckDB preserves even signaling patterns).
    Ok(if x.is_nan() { x } else { x.sin() })
}

pub(super) fn duck_cos(x: f64) -> Result<f64, Trap> {
    trig_guard(x)?;
    Ok(if x.is_nan() { x } else { x.cos() })
}

pub(super) fn duck_tan(x: f64) -> Result<f64, Trap> {
    trig_guard(x)?;
    Ok(if x.is_nan() { x } else { x.tan() })
}

pub(super) fn duck_pow(x: f64, y: f64) -> Result<f64, Trap> {
    // TOTAL, pure IEEE: pow(NaN,0)=1, pow(1,NaN)=1, pow(0,-1)=inf,
    // negative^fractional=NaN, overflow=inf (all measured).
    Ok(x.powf(y))
}

pub(super) fn duck_floor(x: f64) -> Result<f64, Trap> {
    Ok(x.floor())
}

pub(super) fn duck_ceil(x: f64) -> Result<f64, Trap> {
    Ok(x.ceil())
}

pub(super) fn duck_trunc(x: f64) -> Result<f64, Trap> {
    Ok(x.trunc())
}

/// The wave-1 f64 unaries as shared fn pointers (Iabs/Fabs/Fround keep
/// their original arms).
pub(super) fn math1_fn(op: NumOp1) -> fn(f64) -> Result<f64, Trap> {
    match op {
        NumOp1::Ln => duck_ln,
        NumOp1::Log2 => duck_log2,
        NumOp1::Log10 => duck_log10,
        NumOp1::Fexp => duck_exp,
        NumOp1::Fsqrt => duck_sqrt,
        NumOp1::Fcbrt => duck_cbrt,
        NumOp1::Fsin => duck_sin,
        NumOp1::Fcos => duck_cos,
        NumOp1::Ftan => duck_tan,
        NumOp1::Ffloor => duck_floor,
        NumOp1::Fceil => duck_ceil,
        NumOp1::Ftrunc => duck_trunc,
        NumOp1::Iabs | NumOp1::Fabs | NumOp1::Fround => {
            unreachable!("legacy unaries keep dedicated arms")
        }
    }
}

/// round(x, n) on f64 — scale-then-round with the oracle pow table.
/// Non-finite results fall back to the INPUT for n >= 0 and to +0.0 for
/// n < 0 (measured asymmetry: round(NaN, -2) = 0.0).
pub(super) fn round_prec_f64(x: f64, n: i64) -> f64 {
    if n >= 0 {
        let m = pow10(n);
        let r = (x * m).round() / m;
        if r.is_infinite() || r.is_nan() {
            x
        } else {
            r
        }
    } else {
        let m = pow10(n.unsigned_abs() as i64);
        let r = (x / m).round() * m;
        if r.is_infinite() || r.is_nan() {
            0.0
        } else {
            r
        }
    }
}

/// trunc(x, n) on f64: same shape as round, but BOTH branches fall back
/// to the input — the round/trunc asymmetry is measured, not a bug.
pub(super) fn trunc_prec_f64(x: f64, n: i64) -> f64 {
    if n >= 0 {
        let m = pow10(n);
        let r = (x * m).trunc() / m;
        if r.is_infinite() || r.is_nan() {
            x
        } else {
            r
        }
    } else {
        let m = pow10(n.unsigned_abs() as i64);
        let r = (x / m).trunc() * m;
        if r.is_infinite() || r.is_nan() {
            x
        } else {
            r
        }
    }
}

/// Integer round with digits: identity for n >= 0; n < 0 WRAPS at i64
/// (measured: round(i64::MAX, -2) = -9223372036854775700) — never traps.
pub(super) fn round_prec_i64(x: i64, n: i64) -> i64 {
    if n >= 0 {
        return x;
    }
    let p = n.unsigned_abs();
    if p >= 19 {
        return 0;
    }
    let power = 10i64.pow(p as u32);
    let half = power / 2;
    let y = if x >= 0 {
        x.wrapping_add(half)
    } else {
        x.wrapping_sub(half)
    };
    (y / power) * power
}

/// Integer trunc with digits: identity for n >= 0, truncating scale for
/// n < 0 — no half-add, never wraps.
pub(super) fn trunc_prec_i64(x: i64, n: i64) -> i64 {
    if n >= 0 {
        return x;
    }
    let p = n.unsigned_abs();
    if p >= 19 {
        return 0;
    }
    let power = 10i64.pow(p as u32);
    (x / power) * power
}

pub(super) fn like_match(s: &[u8], p: &[u8], esc: Option<u8>) -> Result<bool, Trap> {
    let (mut si, mut pi) = (0usize, 0usize);
    // Backtrack state: pattern index just past the last %, and the string
    // index its current attempt started at.
    let (mut star_p, mut star_s): (Option<usize>, usize) = (None, 0);
    loop {
        if pi < p.len() {
            let pc = p[pi];
            if Some(pc) == esc {
                // Escape intro beats % and _ (ESCAPE '%' de-wildcards it).
                if si == s.len() {
                    return Ok(false); // exhausted string: plain false
                }
                if pi + 1 == p.len() {
                    return Err(Trap(
                        "Like pattern must not end with escape character!".into(),
                    ));
                }
                if s[si] == p[pi + 1] {
                    si += 1;
                    pi += 2;
                    continue;
                }
            } else if pc == b'%' {
                while pi < p.len() && p[pi] == b'%' && Some(b'%') != esc {
                    pi += 1;
                }
                if pi == p.len() {
                    return Ok(true);
                }
                star_p = Some(pi);
                star_s = si;
                continue;
            } else if pc == b'_' {
                if si < s.len() {
                    si += utf8_width(s[si]);
                    pi += 1;
                    continue;
                }
            } else if si < s.len() && s[si] == pc {
                si += 1;
                pi += 1;
                continue;
            }
        } else if si == s.len() {
            return Ok(true);
        }
        // Mismatch: restart at the last % with one more byte consumed.
        match star_p {
            Some(sp) if star_s < s.len() => {
                star_s += 1;
                si = star_s;
                pi = sp;
            }
            _ => return Ok(false),
        }
    }
}

/// Validate an ESCAPE operand per row (AFTER NULL handling): empty means
/// no escape; the limit is one BYTE (a single 2-byte codepoint errors).
pub(super) fn like_escape_of(esc: &str) -> Result<Option<u8>, Trap> {
    match esc.len() {
        0 => Ok(None),
        1 => Ok(Some(esc.as_bytes()[0])),
        _ => Err(Trap(
            "Invalid escape string. Escape string must be empty or one character.".into(),
        )),
    }
}

/// levenshtein == editdist3: classic two-row DP over bytes.
pub(super) fn duck_levenshtein(a: &[u8], b: &[u8]) -> i64 {
    if a.is_empty() || b.is_empty() {
        return (a.len() + b.len()) as i64;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ac) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &bc) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ac != bc);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] as i64
}

/// damerau_levenshtein: the UNRESTRICTED DL variant (NOT OSA — witness
/// ('ca','abc') = 2), transposition cost 1, over bytes. Full matrix plus
/// a last-occurrence table, as the unrestricted recurrence requires.
// ponytail: O(n*m) memory; corpus strings are short — stream it if huge
// inputs ever matter.
pub(super) fn duck_damerau(a: &[u8], b: &[u8]) -> i64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return (n + m) as i64;
    }
    let w = m + 2;
    let inf = n + m;
    let mut d = vec![0usize; (n + 2) * w];
    d[0] = inf;
    for i in 0..=n {
        d[(i + 1) * w] = inf;
        d[(i + 1) * w + 1] = i;
    }
    for j in 0..=m {
        d[j + 1] = inf;
        d[w + j + 1] = j;
    }
    let mut last_a = [0usize; 256]; // last row where each byte occurred in a
    for i in 1..=n {
        let mut last_b = 0usize; // last column where a[i-1] matched in b
        for j in 1..=m {
            let (i1, j1) = (last_a[b[j - 1] as usize], last_b);
            let cost = usize::from(a[i - 1] != b[j - 1]);
            if cost == 0 {
                last_b = j;
            }
            let sub = d[i * w + j] + cost;
            let ins = d[(i + 1) * w + j] + 1;
            let del = d[i * w + j + 1] + 1;
            let trans = d[i1 * w + j1] + (i - i1 - 1) + 1 + (j - j1 - 1);
            d[(i + 1) * w + j + 1] = sub.min(ins).min(del).min(trans);
        }
        last_a[a[i - 1] as usize] = i;
    }
    d[(n + 1) * w + m + 1] as i64
}

/// jaccard: |A∩B|/|A∪B| over single-BYTE sets; empty either side traps.
pub(super) fn duck_jaccard(a: &[u8], b: &[u8]) -> Result<f64, Trap> {
    if a.is_empty() || b.is_empty() {
        return Err(Trap("Jaccard Function: An argument too short!".into()));
    }
    let (mut in_a, mut in_b) = ([false; 256], [false; 256]);
    for &c in a {
        in_a[c as usize] = true;
    }
    for &c in b {
        in_b[c as usize] = true;
    }
    let (mut inter, mut union) = (0i64, 0i64);
    for i in 0..256 {
        inter += i64::from(in_a[i] && in_b[i]);
        union += i64::from(in_a[i] || in_b[i]);
    }
    Ok(inter as f64 / union as f64)
}

/// hamming == mismatches: byte-wise; empty inputs and length mismatch
/// trap — the messages say "Mismatch Function" even for hamming
/// (measured; the error text leaks the shared implementation).
pub(super) fn duck_hamming(a: &[u8], b: &[u8]) -> Result<i64, Trap> {
    if a.is_empty() || b.is_empty() {
        return Err(Trap(
            "Mismatch Function: Strings must be of length > 0!".into(),
        ));
    }
    if a.len() != b.len() {
        return Err(Trap(
            "Mismatch Function: Strings must be of equal length!".into(),
        ));
    }
    Ok(a.iter().zip(b).filter(|(x, y)| x != y).count() as i64)
}

/// repeat(s, n): n <= 0 -> '' silently.
pub(super) fn duck_repeat(s: &str, n: i64) -> Result<String, Trap> {
    if n <= 0 {
        return Ok(String::new());
    }
    match (s.len() as u64).checked_mul(n as u64) {
        Some(sz) if sz <= STR_BUILD_CAP => Ok(s.repeat(n as usize)),
        _ => Err(Trap("string builder result exceeds 1 GiB".into())),
    }
}

/// lpad/rpad(s, l, pad): l counts CODEPOINTS; truncation keeps the FIRST
/// l codepoints for BOTH sides; the pad cycles cut at codepoint
/// boundaries; empty pad traps ONLY when growth is needed.
pub(super) fn duck_pad(left: bool, s: &str, l: i64, pad: &str) -> Result<String, Trap> {
    if l <= 0 {
        return Ok(String::new());
    }
    let l = l as usize;
    let n = s.chars().count();
    if l <= n {
        return Ok(s.chars().take(l).collect());
    }
    if pad.is_empty() {
        return Err(Trap(format!(
            "Insufficient padding in {}.",
            if left { "LPAD" } else { "RPAD" }
        )));
    }
    if l as u64 > STR_BUILD_CAP / 4 {
        return Err(Trap("string builder result exceeds 1 GiB".into()));
    }
    let fill: String = pad.chars().cycle().take(l - n).collect();
    Ok(if left {
        fill + s
    } else {
        let mut out = s.to_string();
        out.push_str(&fill);
        out
    })
}

/// replace(s, from, to): empty needle is a strict NO-OP; leftmost
/// non-overlapping single pass (Rust `str::replace` is exactly that).
pub(super) fn duck_replace(s: &str, from: &str, to: &str) -> String {
    if from.is_empty() {
        return s.to_string();
    }
    s.replace(from, to)
}

/// translate(s, from, to): per-codepoint map, FIRST occurrence in `from`
/// wins, from-chars beyond |to| are deleted.
pub(super) fn duck_translate(s: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match from.chars().position(|f| f == c) {
            None => out.push(c),
            Some(i) => {
                if let Some(t) = to.chars().nth(i) {
                    out.push(t);
                }
            }
        }
    }
    out
}

/// array_extract/list_extract/s[i] on VARCHAR: 1-based codepoint,
/// negative resolves len+1+i, out-of-range/0 -> '' (NOT NULL). Returns
/// the byte range so callers can subview instead of copying.
pub(super) fn extract_window(s: &str, i: i64) -> std::ops::Range<usize> {
    let n = s.chars().count() as i64;
    let pos = if i < 0 { (n + 1).saturating_add(i) } else { i };
    if pos < 1 || pos > n {
        return 0..0;
    }
    let skip = (pos - 1) as usize;
    let b0 = s
        .char_indices()
        .nth(skip)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let b1 = s[b0..]
        .chars()
        .next()
        .map(|c| b0 + c.len_utf8())
        .unwrap_or(s.len());
    b0..b1
}

/// array_slice/list_slice/s[a:b] on VARCHAR: 1-based, both-ends-INCLUSIVE
/// codepoints, negative from-end (-1 = last), lo <= 0 clamps to start,
/// hi > len clamps to end, reversed/out-of-range -> ''. Byte range out.
pub(super) fn slice_window(s: &str, lo: i64, hi: i64) -> std::ops::Range<usize> {
    let n = s.chars().count() as i64;
    let rlo = (if lo < 0 {
        (n + 1).saturating_add(lo)
    } else {
        lo
    })
    .max(1);
    let rhi = (if hi < 0 {
        (n + 1).saturating_add(hi)
    } else {
        hi
    })
    .min(n);
    if rlo > rhi {
        return 0..0;
    }
    let skip = (rlo - 1) as usize;
    let take = (rhi - rlo + 1) as usize;
    let b0 = s
        .char_indices()
        .nth(skip)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    let b1 = s[b0..]
        .char_indices()
        .nth(take)
        .map(|(i, _)| b0 + i)
        .unwrap_or(s.len());
    b0..b1
}

/// unicode/ord ('' -> -1) and ascii ('' -> 0 — the sole divergence):
/// first codepoint as i64.
pub(super) fn duck_ord(s: &str, empty_zero: bool) -> i64 {
    match s.chars().next() {
        Some(c) => c as i64,
        None => {
            if empty_zero {
                0
            } else {
                -1
            }
        }
    }
}

/// strip_accents: all-ASCII passes VERBATIM (NULs preserved); otherwise
/// truncate at the first NUL (the measured context-dependent quirk),
/// per-codepoint oracle map, then Hangul jamo composition. TOTAL.
/// DuckDB reverse() (pins-waveA/reverse-graphemes.json): an all-ASCII
/// string BYTE-reverses (this splits CRLF — measured, not a bug to "fix");
/// anything else reverses UAX-29 EXTENDED grapheme clusters, each cluster
/// byte-preserved, no normalization.
pub(super) fn duck_reverse(s: &str) -> String {
    if s.is_ascii() {
        s.bytes().rev().map(|b| b as char).collect()
    } else {
        use unicode_segmentation::UnicodeSegmentation;
        s.graphemes(true).rev().collect()
    }
}

pub(super) fn duck_strip_accents(s: &str) -> Option<String> {
    if s.is_ascii() {
        return None; // caller keeps the input span — no copy
    }
    let cut = s.find('\0').map(|i| &s[..i]).unwrap_or(s);
    let mut out = String::with_capacity(cut.len());
    for c in cut.chars() {
        let mapped = match super::strip_accents::strip_map(c) {
            None => Some(c),
            Some(m) => m,
        };
        let Some(mc) = mapped else { continue };
        // Hangul compose against the last emitted codepoint: L+V -> LV,
        // LV+T -> LVT (formulaic; wrong-order jamo never compose).
        if let Some(p) = out.chars().next_back() {
            let (pu, cu) = (p as u32, mc as u32);
            if (0x1100..=0x1112).contains(&pu) && (0x1161..=0x1175).contains(&cu) {
                out.pop();
                let lv = 0xAC00 + (pu - 0x1100) * 588 + (cu - 0x1161) * 28;
                out.push(char::from_u32(lv).expect("Hangul LV in range"));
                continue;
            }
            if (0xAC00..0xAC00 + 11172).contains(&pu)
                && (pu - 0xAC00) % 28 == 0
                && (0x11A8..=0x11C2).contains(&cu)
            {
                out.pop();
                let lvt = pu + (cu - 0x11A7);
                out.push(char::from_u32(lvt).expect("Hangul LVT in range"));
                continue;
            }
        }
        out.push(mc);
    }
    Some(out)
}

/// SQL fdiv(x, y) = floor(x / y) — TOTAL, ±inf on zero divisor.
pub(super) fn duck_fdiv(x: f64, y: f64) -> f64 {
    (x / y).floor()
}

/// SQL fmod(x, y) = x − floor(x/y)·y — FLOORED (divisor's sign), NaN on
/// zero or infinite divisor. NOT C fmod (that is SQL `%`).
pub(super) fn duck_fmod(x: f64, y: f64) -> f64 {
    x - (x / y).floor() * y
}

/// C nextafter, bit-exact; x == y returns y (incl. signed zeros).
pub(super) fn duck_nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        // toward y from zero: the smallest denormal with y's direction.
        return f64::from_bits(1).copysign(y - x);
    }
    let bits = x.to_bits();
    let up = (y > x) == (x > 0.0); // move away from zero?
    f64::from_bits(if up { bits + 1 } else { bits - 1 })
}

/// i64 `<<` per the wave-5 pins ladder: negative value first (even << 0),
/// then negative count, then the zero-value shortcut, then count range,
/// then overflow (value >= 2^(63-count), computed in i128 because
/// 1 << 63 doesn't fit i64). Texts DuckDB-verbatim sans the class prefix.
pub(in crate::specializer) fn duck_shl(x: i64, y: i64) -> Result<i64, Trap> {
    if x < 0 {
        return Err(Trap(format!("Cannot left-shift negative number {x}")));
    }
    if y < 0 {
        return Err(Trap(format!("Cannot left-shift by negative number {y}")));
    }
    if x == 0 {
        return Ok(0);
    }
    if y >= 64 {
        return Err(Trap(format!("Left-shift value {y} is out of range")));
    }
    if (x as i128) >= (1i128 << (63 - y)) {
        return Err(Trap(format!("Overflow in left shift ({x} << {y})")));
    }
    Ok(x << y)
}

pub(super) fn overflow_msg(op: BinOp, x: i64, y: i64) -> String {
    match op {
        BinOp::Iadd => format!("Overflow in addition of INT64 ({x} + {y})!"),
        BinOp::Isub => format!("Overflow in subtraction of INT64 ({x} - {y})!"),
        BinOp::Imul => format!("Overflow in multiplication of INT64 ({x} * {y})!"),
        _ => format!("Overflow in division of {x} / {y}"),
    }
}

/// DuckDB's abs(i64::MIN) trap text, verbatim (measured 2026-07-26 — no
/// trailing '!', unlike the binary-op overflow family).
pub(super) fn abs_overflow_msg(x: i64) -> String {
    format!("Overflow on abs({x})")
}

/// Wave-1 string search (pins: 1-based CODEPOINT positions, empty needle
/// matches everything, byte-wise comparison, zero unicode intelligence).
pub(super) fn str_find(s: &str, n: &str) -> i64 {
    if n.is_empty() {
        return 1;
    }
    match s.find(n) {
        None => 0,
        Some(byte) => s[..byte].chars().count() as i64 + 1,
    }
}

pub(super) fn str_pred(op: StrOp2, s: &str, n: &str) -> bool {
    match op {
        StrOp2::Contains => s.contains(n),
        StrOp2::Starts => s.starts_with(n),
        StrOp2::Ends => s.ends_with(n),
        StrOp2::Glob => duck_glob(s, n),
        _ => unreachable!("non-predicate str2 ops have dedicated arms"),
    }
}

/// DuckDB's pow(10, k) — the oracle-extracted table; inf beyond 308.
fn pow10(k: i64) -> f64 {
    if (0..=308).contains(&k) {
        super::pow10::POW10[k as usize]
    } else {
        f64::INFINITY
    }
}

fn utf8_width(b: u8) -> usize {
    match b {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        // Continuation/invalid lead: advance one byte (spans are valid
        // UTF-8; this arm is reachable only from a % restart landing on a
        // continuation byte, where the subsequent literal compare fails
        // anyway — pinned by the multibyte %_ duck_checks).
        _ => 1,
    }
}

/// Engine guard for string-building ops: DuckDB errors on absurd result
/// sizes too (exact threshold/message unpinned — huge-n was deliberately
/// not probed); 1 GiB keeps the engine alive without shadowing any pin.
const STR_BUILD_CAP: u64 = 1 << 30;

/// DuckDB's int-overflow trap texts, verbatim (wave-3 pins: measured via
/// both the operators and their function aliases). Division covers `//`
/// AND `%` on i64::MIN op -1 — DuckDB's own % message says "division",
/// and only add/sub/mul carry the trailing '!'.
/// One parsed GLOB pattern element (bytes, not codepoints — measured).
enum GTok {
    Star,
    Any,
    Byte(u8),
    Class { neg: bool, set: Vec<(u8, u8)> },
}

/// Parse a GLOB pattern per the wave-5 pins. `None` = dead pattern
/// (dangling `\`, unclosed class — incl. `[a-]` whose `]` is eaten as the
/// range endpoint): matches nothing, never errors.
fn parse_glob(p: &[u8]) -> Option<Vec<GTok>> {
    let mut toks = Vec::new();
    let mut i = 0;
    while i < p.len() {
        match p[i] {
            b'*' => {
                toks.push(GTok::Star);
                i += 1;
            }
            b'?' => {
                toks.push(GTok::Any);
                i += 1;
            }
            b'\\' => {
                // Escape OUTSIDE classes only; dangling -> dead.
                let &b = p.get(i + 1)?;
                toks.push(GTok::Byte(b));
                i += 2;
            }
            b'[' => {
                i += 1;
                let mut neg = false;
                if p.get(i) == Some(&b'!') {
                    // Only '!' negates; '^' is a literal member.
                    neg = true;
                    i += 1;
                }
                let mut set: Vec<(u8, u8)> = Vec::new();
                let mut first = true;
                loop {
                    let &b = p.get(i)?; // unclosed -> dead
                    if b == b']' && !first {
                        i += 1;
                        break;
                    }
                    first = false;
                    // ']' is a literal if first; '-' is literal when not
                    // followed by a range endpoint (which may be ']').
                    if p.get(i + 1) == Some(&b'-') && p.get(i + 2).is_some() && b != b'-' {
                        set.push((b, p[i + 2])); // inverted range = empty
                        i += 3;
                    } else {
                        set.push((b, b));
                        i += 1;
                    }
                }
                toks.push(GTok::Class { neg, set });
            }
            b => {
                toks.push(GTok::Byte(b));
                i += 1;
            }
        }
    }
    Some(toks)
}

/// GLOB (wave-5 pins): raw-byte matcher, `*` = any run, `?` = ONE byte,
/// case-sensitive, malformed patterns match nothing. NOT expressible via
/// LIKE (classes; `?` is byte-based while `_` is codepoint-based).
pub(in crate::specializer) fn duck_glob(s: &str, p: &str) -> bool {
    let Some(toks) = parse_glob(p.as_bytes()) else {
        return false;
    };
    let s = s.as_bytes();
    let (mut ti, mut si) = (0usize, 0usize);
    let (mut bt_t, mut bt_s) = (usize::MAX, 0usize);
    while si < s.len() {
        let stepped = match toks.get(ti) {
            Some(GTok::Star) => {
                bt_t = ti;
                ti += 1;
                bt_s = si;
                continue;
            }
            Some(GTok::Any) => true,
            Some(GTok::Byte(b)) => *b == s[si],
            Some(GTok::Class { neg, set }) => {
                set.iter().any(|(lo, hi)| (*lo..=*hi).contains(&s[si])) != *neg
            }
            None => false,
        };
        if stepped {
            ti += 1;
            si += 1;
        } else if bt_t != usize::MAX {
            bt_s += 1;
            si = bt_s;
            ti = bt_t + 1;
        } else {
            return false;
        }
    }
    while matches!(toks.get(ti), Some(GTok::Star)) {
        ti += 1;
    }
    ti == toks.len()
}

/// i64 `>>` — total: out-of-range counts (either direction) give 0.
pub(in crate::specializer) fn duck_shr(x: i64, y: i64) -> i64 {
    if (0..64).contains(&y) {
        x >> y
    } else {
        0
    }
}
