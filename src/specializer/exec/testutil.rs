//! Shared test helpers: batch builders and output snapshots. cfg(test)-only.

use super::super::ir::Program;
use super::interp::InterpFn;
use super::{Batch, ColData, OutCol, RunState, Trap};

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
    ColData::Str {
        valid: vals.iter().map(|v| v.is_some()).collect(),
        data: vals.iter().map(|v| v.unwrap_or("").to_string()).collect(),
    }
}

pub fn batch(rows: usize, cols: Vec<ColData>) -> Batch {
    Batch { rows, cols }
}

/// Parse + verify a fixture text into a Program, panicking with context.
pub fn built(text: &str) -> Program {
    let p = match super::super::ir::parse::parse(text) {
        Ok(p) => p,
        Err(e) => panic!("parse failed: {e}
---
{text}"),
    };
    if let Err(errs) = super::super::ir::verify::verify(&p) {
        let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
        panic!("verify failed: {}
---
{text}", msgs.join("; "));
    }
    p
}

/// Snapshot the output as strings, masking NULL payloads (a NULL's payload
/// is meaningless downstream by contract). Allocates — test side only.
pub fn snapshot(st: &RunState) -> Vec<Vec<String>> {
    let ncols = st.out.len();
    let nrows = st.out.first().map(|c| c.len()).unwrap_or(0);
    (0..nrows)
        .map(|r| {
            (0..ncols)
                .map(|c| match &st.out[c] {
                    OutCol::I1(v) => render(v[r].0, format!("{}", v[r].1)),
                    OutCol::I64(v) => render(v[r].0, format!("{}", v[r].1)),
                    OutCol::F64(v) => render(v[r].0, format!("{:?}", v[r].1)),
                    OutCol::Str(v) => render(v[r].0, st.arena.get(v[r].1).to_string()),
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

pub fn rows(v: &[&[&str]]) -> Vec<Vec<String>> {
    v.iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect()
}
