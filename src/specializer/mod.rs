//! The SQL specializer: a partial evaluator that turns (fixed SQL + static
//! tables) into a specialized native function `f : Rows -> Rows`, prepared
//! once and invoked millions of times with a small dynamic input relation.
//!
//! Design: docs/superpowers/specs/2026-07-25-sql-specializer-design.md.
//! Build order (backlog milestone m-7): imperative IR (M-ir, done), the
//! closure-compiled interpreter oracle (M-interp, done), then this layer —
//! frontend + BTA + lowering (M-lower, in progress) — followed by the
//! Cranelift backend and the generated Python-boundary marshaller.

pub mod exec;
pub mod frontend;
pub mod ir;
pub mod lower;
pub mod plan;

#[cfg(test)]
mod tests;

pub use frontend::PrepareError;

/// STAGE 1 for the v0 ribbon: SQL text + the dynamic table's schema -> a
/// verified imperative-IR program. The returned program is always verified —
/// a lowering bug becomes [`PrepareError::Internal`], never an executable.
pub fn prepare(sql: &str, in_cols: &[ir::Col]) -> Result<ir::Program, PrepareError> {
    let (rel, out_cols) = frontend::frontend(sql, in_cols)?;
    let mut program = lower::lower(&rel, in_cols, out_cols, "run")?;
    // Block-splitting lowerings mint ids out of text order; renumber so
    // every prepared program is exactly canonical (parse(print(p)) == p).
    ir::canonicalize(&mut program);
    if let Err(errs) = ir::verify::verify(&program) {
        let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
        return Err(PrepareError::Internal(format!(
            "lowered program failed verification: {}",
            msgs.join("; ")
        )));
    }
    Ok(program)
}
