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
pub(crate) mod rewrite;
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
/// One map key at the BOUNDARY: where to read it out of a build row, and
/// its slot layout. Replaces four parallel vectors — the paths, the INDF
/// flags, the presence flags, and the key types the materializer used to
/// walk out of `StaticTy::Map` with an iterator skip.
#[derive(Debug)]
pub struct StaticKey {
    /// The column's SEGMENT path (TASK-132) — one segment for a plain
    /// column (dots and all: a name is not a path), the struct walk for a
    /// leaf lane. When `present`, it names a struct NODE. Display = the
    /// segments dotted.
    pub path: Vec<String>,
    /// TASK-133: the key is the node's PRESENCE, not its value — `TRUE if
    /// that node is non-NULL else NULL`, so the ordinary plain /
    /// IS-NOT-DISTINCT machinery gives it DuckDB's nested semantics.
    pub present: bool,
    pub map: plan::MapKey,
}

/// One map value at the boundary: its path plus its slot layout.
#[derive(Debug)]
pub struct StaticVal {
    pub path: Vec<String>,
    pub map: plan::MapVal,
}

#[derive(Debug)]
pub struct StaticSpec {
    /// Stage-B self-join: no materialization — the build side is the
    /// BATCH, assembled per call by the executor.
    pub batch: bool,
    pub table: String,
    pub keys: Vec<StaticKey>,
    pub vals: Vec<StaticVal>,
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
    /// See [`Prepared::input_lanes`] — the AUTHORITY on the program's row
    /// input, of which `program.in_cols` is the projection.
    input_lanes: Vec<plan::InputLane>,
}

impl Prepared {
    /// Every input lane, in IR order: caller lanes first, then the lanes the
    /// binder minted (TASK-133: struct-node presence — appended, so no
    /// caller lane index moves, and empty for every query without a struct
    /// join key). `input_lanes()[i].col() == program.in_cols[i]` for every
    /// `i`: the program's list is the projection, this one is the authority,
    /// and this is the ONLY place either is built.
    pub fn input_lanes(&self) -> &[plan::InputLane] {
        &self.input_lanes
    }
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
    let (rel, joins, out_cols, regexes, wide_outputs, model_refs, minted_lanes) =
        frontend::frontend(
            sql, this_name, in_cols, opaque, structs, statics, many, udfs, models, bind_eval,
        )?;
    let one_row_blocker = one_row_blocker(&rel, &joins, statics);
    // THE one producer of the lane list. A minted lane is an ordinary input
    // column from here down — appended, so no caller lane index shifts
    // (TASK-133) — and `all_in` is this vector's projection, not a second
    // list built from the same parts somewhere else.
    let mut input_lanes = plan::input_lanes(in_cols, structs);
    input_lanes.extend(minted_lanes);
    let all_in: Vec<ir::Col> = input_lanes.iter().map(plan::InputLane::col).collect();
    let mut program = lower::lower(
        &rel, &joins, statics, &all_in, out_cols, regexes, udfs, "run", many, models,
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
                    keys: Vec::new(),
                    vals: Vec::new(),
                };
            }
            let t = &statics[j.table];
            let paths = plan::lane_paths(&t.cols, &t.structs);
            StaticSpec {
                batch: false,
                table: t.name.clone(),
                keys: j
                    .key_cols
                    .iter()
                    .zip(plan::map_keys(&j.keys, &j.key_cols))
                    .map(|(c, map)| {
                        let (path, present) = match &c.src {
                            plan::KeySrc::Lane(c) => (paths[*c as usize].clone(), false),
                            plan::KeySrc::Present(p) => (p.clone(), true),
                        };
                        // A presence key's declared type is built as I1 by
                        // `present_key`, and the build side short-circuits
                        // the type dispatch on `present` — which is only
                        // consistent while the two agree. Debug-only, so
                        // release behavior is untouched.
                        debug_assert!(
                            !present || map.ty == ir::Ty::I1,
                            "a presence key is always I1"
                        );
                        StaticKey { path, present, map }
                    })
                    .collect(),
                vals: j
                    .val_cols
                    .iter()
                    .zip(plan::map_vals(&t.cols, &j.val_cols))
                    .map(|(&c, map)| StaticVal {
                        path: paths[c as usize].clone(),
                        map,
                    })
                    .collect(),
            }
        })
        .collect();
    // DOCUMENTATION, not the guarantee. `all_in` above is this very vector's
    // projection, taken a few lines apart, so this compares a list against
    // itself. What actually keeps the boundary's lanes and the program's
    // columns in step is that there is now exactly ONE producer — the
    // boundary's own second append is deleted. A reviewer who reads this
    // assert as the guarantee would accept a patch that re-introduces a
    // second producer and keeps the assert green.
    debug_assert!(
        input_lanes.len() == program.in_cols.len()
            && input_lanes
                .iter()
                .zip(&program.in_cols)
                .all(|(l, c)| &l.col() == c),
        "input lanes do not project to program.in_cols"
    );
    Ok(Prepared {
        program,
        statics: specs,
        wide_outputs,
        one_row_blocker,
        models: model_refs
            .iter()
            .map(|r| models[*r as usize].name.clone())
            .collect(),
        input_lanes,
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
