//! Shared test helpers: batch builders and output snapshots. cfg(test)-only.

use super::super::ir::Program;
use super::interp::InterpFn;
use super::{Batch, ColData, OutCol, RunState, Trap};

// Column builders: `None` is a NULL cell, and its payload is the type
// default rather than garbage — matching what the boundaries hand the
// engine, so a test cannot pass by reading through a false validity flag.

pub fn c_i1(vals: &[Option<bool>]) -> ColData {
    ColData::I1 {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or(false)).collect(),
    }
}

pub fn c_i64(vals: &[Option<i64>]) -> ColData {
    ColData::I64 {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or(0)).collect(),
    }
}

pub fn c_f64(vals: &[Option<f64>]) -> ColData {
    ColData::F64 {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or(0.0)).collect(),
    }
}

pub fn c_str(vals: &[Option<&str>]) -> ColData {
    let mut col = ColData::new(super::super::ir::Ty::Str);
    for v in vals {
        col.push_str_cell(v.is_some(), v.unwrap_or(""));
    }
    col
}

pub fn batch(rows: usize, cols: Vec<ColData>) -> Batch {
    Batch { rows, cols }
}

/// Parse + verify a fixture text into a Program, panicking with context.
pub fn built(text: &str) -> Program {
    let p = match super::super::ir::parse::parse(text) {
        Ok(p) => p,
        Err(e) => panic!(
            "parse failed: {e}
---
{text}"
        ),
    };
    if let Err(errs) = super::super::ir::verify::verify(&p) {
        let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
        panic!(
            "verify failed: {}
---
{text}",
            msgs.join("; ")
        );
    }
    p
}

/// The two quiet NaNs, by their bits. A test that asserts which side of a
/// `-` a NaN lands on may not spell `f64::NAN`: Rust does not contract that
/// value's own sign bit (the same reason ir/parse.rs takes the sign from the
/// token). See `Lit` in ir/mod.rs for why the sign is carried at all.
pub const POS_NAN: f64 = f64::from_bits(0x7FF8_0000_0000_0000);
pub const NEG_NAN: f64 = f64::from_bits(0xFFF8_0000_0000_0000);

/// Snapshot the output as strings, masking NULL payloads (a NULL's payload
/// is meaningless downstream by contract). Allocates — test side only.
///
/// A double renders as Rust's `{:?}`, which spells every NaN `NaN`. That is
/// the right form for a PIN: most NaNs a program can reach come out of a
/// libm or the hardware, whose choice of sign is the platform's and not
/// ours to freeze. It is the wrong form for comparing two of OUR runs
/// against each other — see [`snapshot_bits`].
pub fn snapshot(st: &RunState) -> Vec<Vec<String>> {
    snap(st, |x| format!("{x:?}"))
}

/// [`snapshot`], except a NaN carries its bit pattern, so a sign or payload
/// that differs between two runs is visible as different text (see `Lit` in
/// ir/mod.rs for why the sign is worth seeing). Its first use is
/// run-against-run comparison (cranelift vs the interpreter, run vs re-run).
/// A CONSTANT expectation may pin this form only where the IR itself defines
/// the bits — literals and total bit operations such as `fneg`, whose sign
/// flip is ours and not the platform's — never where a libm or the hardware
/// chose them, for the reason above.
pub fn snapshot_bits(st: &RunState) -> Vec<Vec<String>> {
    snap(st, |x| {
        if x.is_nan() {
            format!("nan:{:#018x}", x.to_bits())
        } else {
            format!("{x:?}")
        }
    })
}

fn snap(st: &RunState, f64_text: impl Fn(f64) -> String) -> Vec<Vec<String>> {
    let ncols = st.out.len();
    let nrows = st.out.first().map(|c| c.len()).unwrap_or(0);
    (0..nrows)
        .map(|r| {
            (0..ncols)
                .map(|c| match &st.out[c] {
                    OutCol::I1(v) => render(v[r].0, format!("{}", v[r].1)),
                    OutCol::I64(v) => render(v[r].0, format!("{}", v[r].1)),
                    OutCol::F64(v) => render(v[r].0, f64_text(v[r].1)),
                    OutCol::Str(v) => render(v[r].0, st.arena.get(v[r].1).to_string()),
                    // The SCALED integer, which is what the lane holds; the
                    // decimal point is the boundary's business.
                    OutCol::Dec(v) => render(v[r].0, format!("{}", v[r].1)),
                })
                .collect()
        })
        .collect()
}

fn render(valid: bool, s: String) -> String {
    if valid {
        s
    } else {
        "NULL".to_string()
    }
}

pub fn run_snapshot(f: &InterpFn, input: &Batch) -> Result<Vec<Vec<String>>, Trap> {
    let mut st = f.new_state();
    f.run(input, &mut st)?;
    Ok(snapshot(&st))
}

/// The expected-value side of [`snapshot`]: rows of already-rendered cell
/// text, with `"NULL"` spelling a NULL cell.
pub fn rows(v: &[&[&str]]) -> Vec<Vec<String>> {
    v.iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect()
}
