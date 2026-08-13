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
mod sig;

#[cfg(test)]
mod tests;

pub use frontend::PrepareError;

/// One wide UDF output field: the marshaller assembles model field `name`
/// from out columns `[first, first + 1 + width)` — a whole-call validity
/// lane (false = the field is NULL, distinct from a container of NULLs)
/// followed by `width` nullable component lanes.
#[derive(Debug, Clone)]
pub struct WideOut {
    pub name: String,
    pub first: u32,
    pub width: u32,
    /// Declared output field names — empty for an unnamed width-k extern
    /// (the DRAFT-22 `list | None` boundary); non-empty for a NAMED extern
    /// at every width, where the boundary assembles a STRUCT keyed by these
    /// names (slice 5), matching DuckDB's struct registration.
    pub names: Vec<String>,
}

/// How to materialize map static `@N` from the static table it came from:
/// build one entry per table row, keyed by `key_cols` (converted to the
/// declared key types — an int column joined against a float expression
/// becomes f64 here), valued by `val_cols`. Rows with a NULL key are dropped
/// (a NULL never equi-matches); a NULL in a value column is an error.
#[derive(Debug)]
pub struct StaticSpec {
    /// Stage-B self-join: no materialization — the build side is the
    /// BATCH, assembled per call by the executor.
    pub batch: bool,
    pub table: String,
    pub key_cols: Vec<String>,
    /// Per key_col: true when the key joins under IS NOT DISTINCT FROM.
    /// Such a key occupies TWO flattened map-key lanes — (validity i1,
    /// payload) — and the materializer KEEPS NULL-key rows as
    /// (false, type default) instead of dropping them.
    pub key_indf: Vec<bool>,
    pub val_cols: Vec<String>,
    /// Per val_col: a declared-nullable column's map value is a
    /// (validity i1, payload) PAIR in the flattened StaticTy::Map values
    /// (TASK-55 — NULL join values flow through as NULL, not errors).
    pub val_nullable: Vec<bool>,
}

/// The output of stage 1: a verified program plus, per map static, the
/// recipe the caller uses to turn its table data into `StaticData`.
#[derive(Debug)]
pub struct Prepared {
    pub program: ir::Program,
    pub statics: Vec<StaticSpec>,
    /// Width-k UDF output fields, in projection order (see [`WideOut`]).
    pub wide_outputs: Vec<WideOut>,
    /// `None` when the query provably emits EXACTLY one output row per
    /// input row (out[i] <-> in[i]); otherwise names the first construct
    /// that can drop a row. The static proof behind `shape="map"`
    /// (TASK-58): no WHERE, and every join is LEFT (unique keys are
    /// already the map contract, so LEFT never drops or duplicates).
    pub one_row_blocker: Option<String>,
    /// Referenced model sets by name, in `model<...>` static order. These
    /// statics sit AFTER every map static, so the caller materializes the
    /// map recipes first and then one ensemble per name here.
    pub models: Vec<String>,
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
    prepare_opaque(sql, this_name, in_cols, &[], &[], statics, false, &[], &[], &[])
}

/// [`prepare`] plus the row-model columns that have no plain scalar lane:
/// `opaque` (model position + name — timestamps, lists; reject on
/// REFERENCE, star expansion included, instead of blocking construction)
/// and `structs` (flattened to leaf lanes appended after the plain
/// columns in `in_cols` — see [`plan::StructCol`]) — plus the declared UDF
/// externs (DRAFT-22): an unknown function matching a declaration binds as
/// an opaque `ecall`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_opaque(
    sql: &str,
    this_name: &str,
    in_cols: &[ir::Col],
    opaque: &[(usize, String)],
    structs: &[plan::StructCol],
    statics: &[plan::StaticTable],
    many: bool,
    udfs: &[ir::ExternSpec],
    models: &[plan::ModelTable],
    // TASK-101: the udf callables, decl-order-aligned with `udfs`, for
    // the bind-time fold of pure externs. Empty disables the fold.
    bind_eval: &[exec::ExternImpl],
) -> Result<Prepared, PrepareError> {
    let (rel, joins, out_cols, regexes, wide_outputs, model_refs) = frontend::frontend(
        sql, this_name, in_cols, opaque, structs, statics, many, udfs, models, bind_eval,
    )?;
    let one_row_blocker = one_row_blocker(&rel, &joins, statics);
    let mut program = lower::lower(
        &rel, &joins, statics, in_cols, out_cols, regexes, udfs, "run", many, models,
        &model_refs,
    )?;
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
            if j.batch {
                return StaticSpec {
                    batch: true,
                    table: String::new(),
                    key_cols: Vec::new(),
                    key_indf: Vec::new(),
                    val_cols: Vec::new(),
                    val_nullable: Vec::new(),
                };
            }
            let t = &statics[j.table];
            StaticSpec {
                batch: false,
                table: t.name.clone(),
                key_cols: j
                    .key_cols
                    .iter()
                    .map(|&c| t.cols[c as usize].name.clone())
                    .collect(),
                key_indf: j.key_indf.clone(),
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
        wide_outputs,
        one_row_blocker,
        models: model_refs
            .iter()
            .map(|r| models[*r as usize].name.clone())
            .collect(),
    })
}

/// The static exactly-one-row proof behind `shape="map"`: a Filter node or
/// a non-LEFT join can drop input rows; everything else the engine serves
/// is row-preserving (unique join keys are already the map contract, so a
/// LEFT join never drops or duplicates).
fn one_row_blocker(
    rel: &plan::Rel,
    joins: &[plan::JoinSpec],
    statics: &[plan::StaticTable],
) -> Option<String> {
    fn has_filter(r: &plan::Rel) -> bool {
        match r {
            plan::Rel::Filter { .. } => true,
            plan::Rel::Project { input, .. } => has_filter(input),
            plan::Rel::Scan => false,
        }
    }
    if has_filter(rel) {
        return Some("a WHERE clause can drop rows".into());
    }
    for j in joins {
        if j.kind != plan::JoinKind::Left {
            if j.batch {
                return Some("an INNER self-join drops rows on a miss".to_string());
            }
            return Some(format!(
                "INNER JOIN '{}' drops rows on a key miss (use LEFT JOIN)",
                statics[j.table].name
            ));
        }
        if j.batch {
            // A LEFT self-join still multiplies rows — never exactly-one.
            return Some("a self-join multiplies rows".to_string());
        }
    }
    None
}
