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
pub mod fold;
pub mod frontend;
pub mod ir;
pub mod lower;
pub mod plan;
mod retrans;
mod rewrite;

#[cfg(test)]
mod tests;

pub use frontend::PrepareError;

/// How to materialize map static `@N` from the static table it came from:
/// build one entry per table row, keyed by `key_cols` (converted to the
/// declared key types — an int column joined against a float expression
/// becomes f64 here), valued by `val_cols`. Rows with a NULL key are dropped
/// (a NULL never equi-matches); a NULL in a value column is an error.
pub struct StaticSpec {
    pub table: String,
    pub key_cols: Vec<String>,
    pub val_cols: Vec<String>,
    /// Per val_col: a declared-nullable column's map value is a
    /// (validity i1, payload) PAIR in the flattened StaticTy::Map values
    /// (TASK-55 — NULL join values flow through as NULL, not errors).
    pub val_nullable: Vec<bool>,
}

/// The output of stage 1: a verified program plus, per map static, the
/// recipe the caller uses to turn its table data into `StaticData`.
pub struct Prepared {
    pub program: ir::Program,
    pub statics: Vec<StaticSpec>,
}

/// STAGE 1 for the v0 ribbon: SQL text + the dynamic table's name and schema
/// + the static-table catalog -> a verified imperative-IR program. The
/// returned program is always verified — a lowering bug becomes
/// [`PrepareError::Internal`], never an executable.
pub fn prepare(
    sql: &str,
    this_name: &str,
    in_cols: &[ir::Col],
    statics: &[plan::StaticTable],
) -> Result<Prepared, PrepareError> {
    let (rel, joins, out_cols, regexes) = frontend::frontend(sql, this_name, in_cols, statics)?;
    let mut program = lower::lower(&rel, &joins, statics, in_cols, out_cols, regexes, "run")?;
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
    let specs = joins
        .iter()
        .map(|j| {
            let t = &statics[j.table];
            StaticSpec {
                table: t.name.clone(),
                key_cols: j
                    .key_cols
                    .iter()
                    .map(|&c| t.cols[c as usize].name.clone())
                    .collect(),
                val_cols: j
                    .val_cols
                    .iter()
                    .map(|&c| t.cols[c as usize].name.clone())
                    .collect(),
                val_nullable: j
                    .val_cols
                    .iter()
                    .map(|&c| t.cols[c as usize].ty.nullable)
                    .collect(),
            }
        })
        .collect();
    Ok(Prepared {
        program,
        statics: specs,
    })
}
