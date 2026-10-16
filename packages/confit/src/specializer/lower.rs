//! Produce/consume lowering: relational IR -> imperative IR.
//!
//! The central mechanism is the env-threading function builder [`FB`]:
//! CASE and guarded CASTs split the CFG, and the IR's strict block-param SSA
//! (values never cross blocks except as branch args) means every live value
//! must ride the branch when a split happens. `emit` therefore carries a
//! live stack: recursive children push their partial lanes before a sibling
//! is evaluated, and any block transition rewrites the stack in place to the
//! target block's params. Columns are NOT threaded — loads are pure and each
//! block re-loads through its own cache.
//!
//! Null-lane discipline: an SExpr with `nullable == false` lowers to a bare
//! payload register (no flag anywhere); a nullable one carries an `i1` flag,
//! combined with `and` where NULL propagates. Kleene AND/OR are lowered
//! branchless from flag algebra wherever the value is what is wanted;
//! SQL's observable short-circuit belongs to SELECTION context and branches
//! instead, in `FB::emit_truth`. WHERE keeps a row iff the predicate is
//! TRUE — flag && value.
//!
//! CASE lowers to a condition chain with a join block; branch results are
//! evaluated only on their taken path (a guarded `1/0`-style branch must not
//! trap eagerly — SQL semantics, verified by test). A CASE whose parts
//! cannot trap lowers to selects instead when it is small or a NULL guard
//! (`select_case`): computing every part on every row is invisible there,
//! and costs less than blocks that carry the live stack and load their
//! columns again. CAST failure is a conditional trap block; TRY_CAST folds
//! the failure into the null lane.
//!
//! ponytail: blocks re-load columns instead of receiving them as branch args
//! — pure loads, identical semantics, simpler lowering. Thread them when the
//! Cranelift backend makes the extra reads measurable.

use std::collections::HashMap;

use super::frontend::PrepareError;
use super::ir::{DecOp, 
    BinOp, Block, BlockId, Builder, CmpPred, Col, Inst, Lit, NumOp1, Program, StaticTy, StrOp1,
    StrOp2, Term, Ty, Value,
};
use super::plan::{
    self, AggFunc, ArithOp, JoinKind, JoinSpec, KeyCmp, Plan, SExpr, SKind, StaticTable,
};


/// Can this kind's NARROW result sit outside its width's range?
///
/// The exempt list is an allowlist, and each entry earns it: `Col` and
/// `StaticCol` are range-checked on the way IN (the ingest boundary mirrors
/// `narrow_check`); a `Slot` was checked where its stage produced it; a `Lit` outside its type's range refuses at build;
/// `NullOf` is a typed default; `JoinHit` is i1; a `Shared` value was checked
/// where it was computed; and `Case` only forwards a
/// value one of its arms already produced and checked. Everything else —
/// arithmetic, casts, abs, extern returns, anything added later — is checked.
fn narrow_result_can_escape(k: &SKind) -> bool {
    !matches!(
        k,
        SKind::Col(_)
            | SKind::Slot(_)
            | SKind::StaticCol { .. }
            | SKind::Lit(_)
            | SKind::NullOf
            | SKind::JoinHit(_)
            | SKind::Shared(_)
            | SKind::Case { .. }
    )
}

/// The bit width of a narrow integer type; `None` for BIGINT and the rest.
pub(super) fn narrow_bits(ty: Ty) -> Option<i64> {
    match ty {
        Ty::I8 => Some(8),
        Ty::I16 => Some(16),
        Ty::I32 => Some(32),
        Ty::U8 => Some(8),
        Ty::U16 => Some(16),
        Ty::U32 => Some(32),
        _ => None,
    }
}

/// DuckDB's spelling of a narrow width, for the trap message.
fn duck_narrow_name(ty: Ty) -> &'static str {
    match ty {
        Ty::I8 => "TINYINT",
        Ty::I16 => "SMALLINT",
        Ty::I32 => "INTEGER",
        Ty::U8 => "UTINYINT",
        Ty::U16 => "USMALLINT",
        Ty::U32 => "UINTEGER",
        Ty::U64 => "UBIGINT",
        Ty::I128 => "HUGEINT",
        _ => "BIGINT",
    }
}

/// The arrow spelling of the same width — the vocabulary the schemas and the
/// boundary refusals use, so a reader greps one word for both.
fn arrow_narrow_name(ty: Ty) -> &'static str {
    match ty {
        Ty::I8 => "int8",
        Ty::I16 => "int16",
        Ty::I32 => "int32",
        Ty::U8 => "uint8",
        Ty::U16 => "uint16",
        Ty::U32 => "uint32",
        Ty::U64 => "uint64",
        _ => "int64",
    }
}

/// The bound plan as one imperative-IR program named `name`: its stages, the
/// joins they run, and the statics those probe.
///
/// `in_cols` is the row input in LANE order (struct leaves already
/// flattened) and `out_cols` its declared output, both positional — the last
/// stage's i-th projection stores to `out_cols[i]`. `joins` indexes
/// `catalog`, and join `j` probes static `@j`, so the `model<...>` statics
/// for `model_refs` (indices into `models`) are appended after them all.
/// `many` selects the multiplicity loops of `shape="many"`, one per join,
/// which are limited to a single stage.
///
/// Stages lower in order into one program. A stage's projection values stay
/// on the live stack, below whatever the next stage pushes, so every block
/// transition carries them as block arguments: a slot is an SSA value
/// computed once per row reaching its stage, and reading it evaluates
/// nothing.
///
/// The result is well-formed but NOT yet canonical or verified; the caller
/// does both. What the loop form cannot serve refuses here as
/// [`PrepareError::Unsupported`]; a shape the frontend should have rejected
/// surfaces as [`PrepareError::Internal`], never as a wrong program.
#[allow(clippy::too_many_arguments)]
pub fn lower(
    plan: &Plan,
    joins: &[JoinSpec],
    catalog: &[StaticTable],
    in_cols: &[Col],
    out_cols: Vec<Col>,
    regexes: Vec<super::ir::ReSpec>,
    udfs: &[super::ir::ExternSpec],
    name: &str,
    many: bool,
    models: &[super::plan::ModelTable],
    model_refs: &[u32],
) -> Result<Program, PrepareError> {
    // One `model<...>` static per referenced tree transform, APPENDED after
    // every join static: join index IS the static id (see emit_probe below),
    // so a model static anywhere earlier would shift every probe's @N.
    let model_statics: Vec<StaticTy> = model_refs
        .iter()
        .map(|r| StaticTy::Model {
            n_features: models[*r as usize].takes.len() as u32,
        })
        .collect();
    let Some(last) = plan.stages.last() else {
        return Err(PrepareError::Internal("plan has no stage".to_string()));
    };
    if last.project.len() != out_cols.len() {
        return Err(PrepareError::Internal(
            "the last stage's projection does not match the output".to_string(),
        ));
    }

    // Both paths emit one static per join.
    let mut fb = FB::new(in_cols, joins, catalog, udfs, joins.len());

    // shape='many': joins lower as multiplicity LOOPS over multimap row ranges
    // — 0..N output rows per input row, one loop per join, nested in FROM
    // order. A map's key-uniqueness is unknown at prepare, so under 'many'
    // every join takes the loop form.
    if many && joins.iter().skip(1).any(|j| j.batch) {
        // A loop runs per row that reaches it, but DuckDB builds the batch
        // side of a self-join over the whole batch, reached or not.
        return Err(PrepareError::Unsupported(
            "a self-join after another join under shape='many'".to_string(),
        ));
    }
    if many
        && joins
            .iter()
            .any(|j| j.key_cols.iter().any(|k| k.cmp == KeyCmp::NotDistinct))
    {
        return Err(PrepareError::Unsupported(
            "IS NOT DISTINCT FROM join keys under shape='many' \
             (params joins are the map/filter shapes)"
                .to_string(),
        ));
    }
    if many && !joins.is_empty() {
        let [stage] = plan.stages.as_slice() else {
            return Err(PrepareError::Unsupported(
                "a subquery under a shape='many' join".to_string(),
            ));
        };
        // The loop does not share: the binder read every let's value in
        // place (`frontend/lets.rs`).
        fb.lower_many_loops(&stage.joins, &stage.project, stage.pred.as_ref(), &out_cols)?;
        let statics: Vec<StaticTy> = joins
            .iter()
            .map(|spec| {
                if spec.batch {
                    StaticTy::BatchMap {
                        values: plan::map_vals(in_cols, &spec.val_cols)
                            .iter()
                            .flat_map(|v| v.slots())
                            .collect(),
                    }
                } else {
                    StaticTy::MultiMap {
                        // The INDF refusal above means every key reaching
                        // here compares `Eq`, so `slots()` returns exactly
                        // `[ty.lane()]` today; lift that refusal and this
                        // line starts carrying pairs unedited.
                        keys: plan::map_keys(&spec.keys, &spec.key_cols)
                            .iter()
                            .flat_map(|k| k.slots())
                            .collect(),
                        values: plan::map_vals(&catalog[spec.table].cols, &spec.val_cols)
                            .iter()
                            .flat_map(|v| v.slots())
                            .collect(),
                    }
                }
            })
            .collect();
        let statics = [statics, model_statics].concat();
        return fb.finish(name, statics, in_cols, out_cols, regexes, udfs.to_vec());
    }

    // The previous stage's slots: the bottom of the live stack, always.
    let mut live: Live = Vec::new();
    for (si, stage) in plan.stages.iter().enumerate() {
        let n_slots = live.len();
        fb.n_slots = n_slots;
        fb.lower_stage_head(stage, &mut live)?;
        // The projection's shared values go on the live stack above the
        // slots, each computed just before the first item that reads it
        // (directly or through another shared value) and dropped after the
        // last step that reads it, so a value lives only across the items
        // that use it: computing every one before the first item carried
        // each across every later item's blocks (a catalog step's 256
        // per-feature values rode all 6,000 blocks). A shared value cannot
        // trap, and items are evaluated in order at the top level, so the
        // value is ready wherever it is read.
        let exprs: Vec<SExpr> = stage.project.iter().map(|(_, e)| e.clone()).collect();
        let (defs, mut items) = match super::share::share(&exprs, &stage.lets) {
            Some(sh) => {
                fb.shared_at = vec![None; sh.defs.len()];
                (sh.defs, sh.items)
            }
            None => {
                fb.shared_at.clear();
                (Vec::new(), exprs)
            }
        };
        let first = first_item(&defs, &mut items);
        let last_use = until_item(&defs, &mut items, &first)
            .into_iter()
            .map(|u| u.map_or(usize::MAX, |ci| defs.len() + ci))
            .collect::<Vec<_>>();
        let n_defs = last_use.len();
        if si + 1 < plan.stages.len() {
            // Every column, read or not, in SELECT order: its traps fire
            // for every row reaching the stage, exactly as DuckDB evaluates
            // a subquery. Each lane joins the live stack as soon as it
            // exists, so a later column's block splits carry it.
            for (ci, e) in items.iter().enumerate() {
                fb.emit_shared_for(&defs, &first, ci, &mut live)?;
                let lane = fb.emit(e, &mut live)?;
                live.push((lane, e.ty));
                fb.drop_dead_shared(&last_use, n_defs + ci, &mut live);
            }
            // The next stage sees only these: the inner scope is closed.
            live.drain(..n_slots);
            continue;
        }
        for (ci, e) in items.iter().enumerate() {
            fb.emit_shared_for(&defs, &first, ci, &mut live)?;
            // Not a `debug_assert`: this runs once per output column at
            // prepare, never per row, and it is the ONLY thing that catches
            // an arm which pushes lanes and forgets to truncate. A leak
            // leaves dead values riding every later block transition —
            // well-formed IR, so `verify` says nothing, and a release-only
            // test run sees nothing either (measured: the whole suite passes
            // green with a `live.truncate` deleted from an arm).
            let n_shared = fb.shared_at.iter().flatten().count();
            assert!(
                live.len() == n_slots + n_shared,
                "live stack leaked before column {ci}"
            );
            let lane = fb.emit(e, &mut live)?;
            let col = ci as u32;
            if out_cols[ci].ty.nullable {
                let flag = match lane.flag {
                    Some(f) => f,
                    // Nullability contract slack (e.g. an infallible
                    // TRY_CAST): the column is declared nullable, the lane is
                    // provably valid — store with a constant true flag.
                    None => fb.const_i1(true),
                };
                fb.inst(Inst::StoreOpt {
                    col,
                    flag,
                    val: lane.val,
                });
            } else {
                debug_assert!(lane.flag.is_none(), "non-nullable column with a flag lane");
                fb.inst(Inst::Store { col, val: lane.val });
            }
            fb.drop_dead_shared(&last_use, n_defs + ci, &mut live);
        }
    }
    fb.term(Term::Emit);

    // Map static @j belongs to join j: keyed by the (already promoted) key
    // expression types, valued by the join's value columns. The rows that
    // a fold scans are a multimap: a key can hold more than one row.
    let statics = joins
        .iter()
        .map(|spec| {
            let keys = plan::map_keys(&spec.keys, &spec.key_cols)
                .iter()
                .flat_map(|k| k.slots())
                .collect();
            let values = plan::map_vals(&catalog[spec.table].cols, &spec.val_cols)
                .iter()
                .flat_map(|v| v.slots())
                .collect();
            match spec.scan {
                Some(_) => StaticTy::MultiMap { keys, values },
                None => StaticTy::Map { keys, values },
            }
        })
        .collect();
    let statics = [statics, model_statics].concat();

    fb.finish(name, statics, in_cols, out_cols, regexes, udfs.to_vec())
}

/// Per shared value, the first item that reads it, directly or through a
/// later shared value (which reads only earlier ones); `None` when none does.
fn first_item(defs: &[SExpr], items: &mut [SExpr]) -> Vec<Option<usize>> {
    fn reads(e: &mut SExpr, out: &mut Vec<u32>) {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || {
                if let SKind::Shared(k) = e.kind {
                    out.push(k);
                    return;
                }
                for c in e.children_mut() {
                    reads(c, out);
                }
            },
        )
    }
    let mut first: Vec<Option<usize>> = vec![None; defs.len()];
    for (ci, e) in items.iter_mut().enumerate() {
        let mut out = Vec::new();
        reads(e, &mut out);
        for k in out {
            let f = &mut first[k as usize];
            *f = Some(f.map_or(ci, |x| x.min(ci)));
        }
    }
    for j in (0..defs.len()).rev() {
        let Some(fj) = first[j] else { continue };
        let mut out = Vec::new();
        reads(&mut defs[j].clone(), &mut out);
        for k in out {
            let f = &mut first[k as usize];
            *f = Some(f.map_or(fj, |x| x.min(fj)));
        }
    }
    first
}

/// Per shared value, the last item before or at which it is read: by an
/// item, or by a later shared value, which is computed just before its own
/// first reader (`first_item`). It is dropped after that item.
fn until_item(
    defs: &[SExpr],
    items: &mut [SExpr],
    first: &[Option<usize>],
) -> Vec<Option<usize>> {
    fn reads(e: &mut SExpr, out: &mut Vec<u32>) {
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || {
                if let SKind::Shared(k) = e.kind {
                    out.push(k);
                    return;
                }
                for c in e.children_mut() {
                    reads(c, out);
                }
            },
        )
    }
    let mut until: Vec<Option<usize>> = vec![None; defs.len()];
    let mut bump = |k: u32, at: usize| {
        let u = &mut until[k as usize];
        *u = Some(u.map_or(at, |x| x.max(at)));
    };
    for (ci, e) in items.iter_mut().enumerate() {
        let mut out = Vec::new();
        reads(e, &mut out);
        for k in out {
            bump(k, ci);
        }
    }
    for (j, d) in defs.iter().enumerate() {
        let Some(at) = first[j] else { continue };
        let mut out = Vec::new();
        reads(&mut d.clone(), &mut out);
        for k in out {
            bump(k, at);
        }
    }
    until
}

/// A value in the null-lane representation: payload + optional validity.
/// `flag: None` means provably non-NULL (no flag register exists).
#[derive(Clone, Copy)]
struct Lane {
    flag: Option<Value>,
    val: Value,
}

/// The live stack: lanes (with their payload types) that must survive block
/// transitions triggered while a sibling expression is being emitted.
type Live = Vec<(Lane, Ty)>;

/// A block under construction.
struct PB {
    params: Vec<(Value, Ty)>,
    insts: Vec<Inst>,
    term: Option<Term>,
    /// Input columns already loaded in this block, by column index.
    cache: HashMap<u32, Lane>,
    /// Probes already emitted in this block: join index -> (valid hit —
    /// map hit AND every key flag — and the value-column registers).
    probes: HashMap<u32, (Value, Vec<Value>)>,
    /// Extern calls already emitted in this block: call site -> (whole
    /// validity, per-return (flag, payload) registers flattened). The k+1
    /// lanes of one width-k call share a site, so the callable runs once.
    externs: HashMap<u32, (Value, Vec<Value>)>,
}

impl PB {
    fn new(params: Vec<(Value, Ty)>) -> PB {
        PB {
            params,
            insts: vec![],
            term: None,
            cache: HashMap::new(),
            probes: HashMap::new(),
            externs: HashMap::new(),
        }
    }
}

/// Env-threading function builder over the strict block-param SSA IR.
struct FB<'a> {
    b: Builder,
    blocks: Vec<PB>,
    cur: usize,
    in_cols: &'a [Col],
    joins: &'a [JoinSpec],
    catalog: &'a [StaticTable],
    udfs: &'a [super::ir::ExternSpec],
    /// Index of the first `model<...>` static: model statics are appended
    /// after every join static, so `predict @N` is `model_base + model`.
    model_base: usize,
    /// The many-joins whose probe caches must be re-created in every block a
    /// split creates, while one expression is being emitted under them:
    /// every loop open around the expression, outermost first. `base` is
    /// where a join's value lanes sit on the live stack — NOT
    /// `live.len() - nd`, which holds only at an expression boundary,
    /// before operands are pushed on top of them.
    many: Vec<ManySeed>,
    /// Scalar joins whose residual is being emitted right now, innermost
    /// last. Same hazard as `many`, different cache: the join's value lanes
    /// ride the live stack but the cache ENTRY is per block, and a split
    /// inside the residual would drop it, miss, and re-enter `emit_probe`
    /// without bound — a stack overflow that kills the process rather than
    /// raising. A stack, not an Option: a residual may probe another join.
    probe_seeds: Vec<ProbeSeed>,
    /// How many slots the stage being lowered can read: the previous
    /// stage's projection width, sitting at `live[..n_slots]`.
    n_slots: usize,
    /// Where each of the stage's shared values (`share.rs`) sits on the
    /// live stack, `None` once it is dead.
    shared_at: Vec<Option<usize>>,
}

/// Where an active many-join's probe lanes ride: `nd` live entries starting
/// at `base`, `nd` being the FLATTENED value width (`val_flat_tys`), plus
/// the hit flag to seed the cache with.
#[derive(Clone, Copy)]
struct ManySeed {
    join: u32,
    hit: ManyHit,
    base: usize,
    nd: usize,
}

/// A many-join's hit flag: a constant where it is known — TRUE in the
/// join's own residual and for an INNER join, whose loop has no
/// null-extension — or the live lane at that index, which is FALSE on the
/// LEFT null-extension and TRUE on a matched row.
#[derive(Clone, Copy)]
enum ManyHit {
    Const(bool),
    Live(usize),
}

/// The same for a scalar join being probed inside its own residual, where
/// the hit is TRUE by construction and needs no field.
#[derive(Clone, Copy)]
struct ProbeSeed {
    join: u32,
    base: usize,
    nd: usize,
}

impl<'a> FB<'a> {
    fn new(
        in_cols: &'a [Col],
        joins: &'a [JoinSpec],
        catalog: &'a [StaticTable],
        udfs: &'a [super::ir::ExternSpec],
        model_base: usize,
    ) -> FB<'a> {
        FB {
            b: Builder::new(),
            blocks: vec![PB::new(vec![])],
            cur: 0,
            in_cols,
            joins,
            catalog,
            udfs,
            model_base,
            many: Vec::new(),
            probe_seeds: Vec::new(),
            n_slots: 0,
            shared_at: Vec::new(),
        }
    }

    fn fresh(&mut self) -> Value {
        self.b.fresh()
    }

    fn inst(&mut self, i: Inst) {
        self.blocks[self.cur].insts.push(i);
    }

    fn term(&mut self, t: Term) {
        let slot = &mut self.blocks[self.cur].term;
        debug_assert!(slot.is_none(), "block terminated twice");
        *slot = Some(t);
    }

    fn switch(&mut self, blk: usize) {
        self.cur = blk;
    }

    fn create_block(&mut self, param_tys: &[Ty]) -> (usize, Vec<Value>) {
        let params: Vec<(Value, Ty)> = param_tys.iter().map(|t| (self.b.fresh(), *t)).collect();
        let vals = params.iter().map(|(v, _)| *v).collect();
        self.blocks.push(PB::new(params));
        (self.blocks.len() - 1, vals)
    }

    fn finish(
        self,
        name: &str,
        statics: Vec<StaticTy>,
        in_cols: &[Col],
        out_cols: Vec<Col>,
        regexes: Vec<super::ir::ReSpec>,
        externs: Vec<super::ir::ExternSpec>,
    ) -> Result<Program, PrepareError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for (i, pb) in self.blocks.into_iter().enumerate() {
            let term = pb
                .term
                .ok_or_else(|| PrepareError::Internal(format!("block b{i} left unterminated")))?;
            blocks.push(Block {
                params: pb.params,
                insts: pb.insts,
                term,
            });
        }
        Ok(Program {
            statics,
            regexes,
            externs,
            name: name.to_string(),
            in_cols: in_cols.to_vec(),
            out_cols,
            blocks,
        })
    }

    // ------------------------------------------------------ conveniences --

    fn const_lit(&mut self, lit: Lit) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Const { dst, lit });
        dst
    }

    fn const_i1(&mut self, v: bool) -> Value {
        self.const_lit(Lit::I1(v))
    }

    /// Trapping instructions must never fire on the garbage payload under a
    /// false NULL flag — computed garbage is unbounded (`x + MAX + MAX` with
    /// x NULL overflows its payload). Mask nullable payloads to the type
    /// default right before any trapping instruction.
    fn masked(&mut self, l: Lane, ty: Ty) -> Value {
        match l.flag {
            None => l.val,
            Some(f) => {
                let d = self.default_of(ty);
                self.select_of(f, l.val, d)
            }
        }
    }

    /// Encode one PROBE-side key into its flattened slots, appending them to
    /// `out`. Returns, for an `Eq` key only, the validity flag the CALLER
    /// must fold — into `keys_valid` for a map probe, into the range gate
    /// for the fan-out loop. The plain-key NULL rule is per-site and stays
    /// out of the layout; what the layout owns is the slot count and the
    /// masked payload, which must match the build side bit for bit
    /// (`exec::null_key_slots`).
    fn encode_probe_key(
        &mut self,
        k: plan::MapKey,
        lane: Lane,
        out: &mut Vec<Value>,
    ) -> Option<Value> {
        match k.cmp {
            KeyCmp::NotDistinct => {
                let (valid, payload) = match lane.flag {
                    None => (self.const_i1(true), lane.val),
                    Some(f) => (f, self.masked(lane, k.ty)),
                };
                out.push(valid);
                out.push(payload);
                None
            }
            KeyCmp::Eq => {
                out.push(lane.val);
                lane.flag
            }
        }
    }

    /// Like `masked`, but to an op-specific SAFE constant: the type default
    /// (0.0) is itself in the trap domain of ln/logb, so those mask to a
    /// value the op accepts (the result is discarded under the false flag).
    fn masked_to(&mut self, l: Lane, safe: Lit) -> Value {
        match l.flag {
            None => l.val,
            Some(f) => {
                let d = self.const_lit(safe);
                self.select_of(f, l.val, d)
            }
        }
    }

    fn select_of(&mut self, cond: Value, a: Value, b: Value) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Select { dst, cond, a, b });
        dst
    }

    fn default_of(&mut self, ty: Ty) -> Value {
        self.const_lit(match ty {
            Ty::I1 => Lit::I1(false),
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 => Lit::I64(0),
            Ty::F64 => Lit::F64(0.0),
            Ty::Str => Lit::Str(String::new()),
            // The LEFT-miss payload of a decimal lane: a zero at the
            // column's own scale, exactly like the other lanes' defaults.
            Ty::Dec(p, s) => Lit::Dec(0, p, s),
            Ty::I128 | Ty::U64 => Lit::I128(0),
        })
    }

    fn bin(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Bin { op, dst, a, b });
        dst
    }

    fn not(&mut self, a: Value) -> Value {
        let dst = self.fresh();
        self.inst(Inst::Not { dst, a });
        dst
    }

    /// The SQL truth of a boolean lane: TRUE iff valid AND value.
    fn truthy(&mut self, lane: Lane) -> Value {
        match lane.flag {
            None => lane.val,
            Some(f) => self.bin(BinOp::And, f, lane.val),
        }
    }

    /// SELECTION context: the value is consumed only for TRUE-ness, so the
    /// result is a bare i1 with no null flag. Entered from the WHERE root
    /// (scalar and many paths) and from every CASE condition; exits to
    /// `emit` at every operator that can tell NULL from FALSE.
    ///
    /// The measured model: AND is the ONLY lazy
    /// operator -- its LEFT always runs, its RIGHT is skipped when the left
    /// is not TRUE, recursively through both children. OR always evaluates
    /// both sides but passes the context through, which is how a conjunction
    /// under `OR TRUE` keeps its laziness. A CASE's condition is selection
    /// context wherever the CASE sits; its ARMS are value context (a taken
    /// arm traps on DuckDB even under WHERE), which is why there is no Case
    /// arm here -- the default `emit` route is the measured behaviour.
    ///
    /// Skipping the right on a not-TRUE left yields FALSE where Kleene may
    /// say NULL (`NULL AND FALSE`), and that is sound HERE because the
    /// consumer only asks "is it TRUE" -- FALSE and NULL answer alike, and
    /// OR-of-truths preserves the answer. `NOT` tells the two apart, which
    /// is exactly where DuckDB reverts to value context, and so do we.
    fn emit_truth(&mut self, e: &SExpr, live: &mut Live) -> Result<Value, PrepareError> {
        match &e.kind {
            SKind::And { a, b } => {
                let at = self.emit_truth(a, live)?;
                let shape = Self::live_types(live);
                let mut join_tys = shape.clone();
                join_tys.push(Ty::I1);
                let (join, jp) = self.create_block(&join_tys);
                let (rhs, rp) = self.create_block(&shape);
                let mut skip_args = Self::live_args(live);
                let f = self.const_i1(false);
                skip_args.push(f);
                self.term(Term::Brif {
                    cond: at,
                    then_to: BlockId(rhs as u32),
                    then_args: Self::live_args(live),
                    else_to: BlockId(join as u32),
                    else_args: skip_args,
                });
                self.switch(rhs);
                self.enter_block(live, &rp);
                let bt = self.emit_truth(b, live)?;
                let mut args = Self::live_args(live);
                args.push(bt);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args,
                });
                self.switch(join);
                let lw = shape.len();
                self.enter_block(live, &jp[..lw]);
                Ok(jp[lw])
            }
            SKind::Or { a, b } => {
                // The exact dual of And: a TRUE left decides TRUE and the
                // right is SKIPPED (measured: `WHERE TRUE OR trap` and
                // `WHERE b OR trap` with b true both serve; b NULL or FALSE
                // evaluates the right and traps -- NULL OR x is TRUE exactly
                // when x is).
                let at = self.emit_truth(a, live)?;
                let shape = Self::live_types(live);
                let mut join_tys = shape.clone();
                join_tys.push(Ty::I1);
                let (join, jp) = self.create_block(&join_tys);
                let (rhs, rp) = self.create_block(&shape);
                let mut skip_args = Self::live_args(live);
                let tr = self.const_i1(true);
                skip_args.push(tr);
                self.term(Term::Brif {
                    cond: at,
                    then_to: BlockId(join as u32),
                    then_args: skip_args,
                    else_to: BlockId(rhs as u32),
                    else_args: Self::live_args(live),
                });
                self.switch(rhs);
                self.enter_block(live, &rp);
                let bt = self.emit_truth(b, live)?;
                let mut args = Self::live_args(live);
                args.push(bt);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args,
                });
                self.switch(join);
                let lw = shape.len();
                self.enter_block(live, &jp[..lw]);
                Ok(jp[lw])
            }
            _ => {
                let l = self.emit(e, live)?;
                Ok(self.truthy(l))
            }
        }
    }

    /// [`Self::emit_truth`] under the many-joins `seeds` -- the same wrapper
    /// shape as [`Self::emit_many`].
    fn emit_truth_many(
        &mut self,
        e: &SExpr,
        live: &mut Live,
        seeds: &[ManySeed],
    ) -> Result<Value, PrepareError> {
        self.many = seeds.to_vec();
        self.seed_many(live);
        let out = self.emit_truth(e, live);
        self.many.clear();
        out
    }

    // -------------------------------------------------- live threading --

    /// The flattened param type shape of a live stack: per lane, an i1 flag
    /// (when present) then the payload.
    fn live_types(live: &Live) -> Vec<Ty> {
        let mut tys = Vec::new();
        for (lane, ty) in live {
            if lane.flag.is_some() {
                tys.push(Ty::I1);
            }
            // Block params are SSA values: narrow widths erase to the lane.
            tys.push(ty.lane());
        }
        tys
    }

    fn live_args(live: &Live) -> Vec<Value> {
        let mut args = Vec::new();
        for (lane, _) in live {
            if let Some(f) = lane.flag {
                args.push(f);
            }
            args.push(lane.val);
        }
        args
    }

    /// Drop every shared value whose last read is evaluation step `step`
    /// (see [`super::share::Shared::last_use`]) off the live stack, so it
    /// stops riding later block transitions. Called between top-level
    /// steps, when nothing above the stage's own entries is pending.
    /// Emit the shared values whose first reader is item `ci`, in order
    /// (each reads only earlier ones, emitted by then).
    fn emit_shared_for(
        &mut self,
        defs: &[SExpr],
        first: &[Option<usize>],
        ci: usize,
        live: &mut Live,
    ) -> Result<(), PrepareError> {
        for (k, d) in defs.iter().enumerate() {
            if first[k] != Some(ci) {
                continue;
            }
            let lane = self.emit(d, live)?;
            live.push((lane, d.ty));
            self.shared_at[k] = Some(live.len() - 1);
        }
        Ok(())
    }

    fn drop_dead_shared(&mut self, last_use: &[usize], step: usize, live: &mut Live) {
        for (k, &last) in last_use.iter().enumerate() {
            if last != step {
                continue;
            }
            let Some(at) = self.shared_at[k].take() else {
                continue;
            };
            live.remove(at);
            for p in self.shared_at.iter_mut().flatten() {
                if *p > at {
                    *p -= 1;
                }
            }
        }
    }

    /// Rebind the live stack to the params of the block just switched to.
    /// The shape (lane count + flag pattern) is invariant across transitions.
    fn rebind_live(live: &mut Live, params: &[Value]) {
        let mut i = 0;
        for (lane, _) in live.iter_mut() {
            if lane.flag.is_some() {
                lane.flag = Some(params[i]);
                i += 1;
            }
            lane.val = params[i];
            i += 1;
        }
        debug_assert_eq!(i, params.len());
    }

    // ------------------------------------------------------- expressions --

    /// Every expression, plus the NARROW-WIDTH RANGE TRAP its type implies.
    ///
    /// I8/I16/I32 erase to the i64 lane, and that erasure is only sound
    /// because DuckDB's narrow ints are checked-never-wrapping: i64 compute
    /// is bit-identical to real int32 exactly when the range trap fires
    /// wherever DuckDB's would. Checking at the OUTPUT boundary is not that.
    /// `CAST((i + 1) AS BIGINT)` over INT32_MAX served 2147483648, a value
    /// DuckDB never produces, because the widening cast consumed the i32
    /// result before any boundary saw it; a comparison, a function
    /// argument, or a float promotion hid it the same way.
    ///
    /// So the check lands on the RESULT, at the point of production. The
    /// opt-out is an ALLOWLIST rather than a denylist, so an operator added
    /// later is checked by default and only what provably cannot leave the
    /// range is exempt.
    fn emit(&mut self, e: &SExpr, live: &mut Live) -> Result<Lane, PrepareError> {
        // Recursive per level: grow the stack rather than overflow it.
        stacker::maybe_grow(
            super::frontend::RED_ZONE,
            super::frontend::STACK_SEGMENT,
            || self.emit_here(e, live),
        )
    }

    fn emit_here(&mut self, e: &SExpr, live: &mut Live) -> Result<Lane, PrepareError> {
        let lane = self.emit_kind(e, live)?;
        match e.ty.int_range() {
            Some((lo, hi)) if narrow_result_can_escape(&e.kind) => {
                self.narrow_trap(lane, Lit::I64(lo), Lit::I64(hi), e.ty, live)
            }
            // UBIGINT, the narrow width of the i128 lane.
            _ if e.ty == Ty::U64 && narrow_result_can_escape(&e.kind) => {
                let (lo, hi) = e.ty.int_range128().expect("an integer width");
                self.narrow_trap(lane, Lit::I128(lo), Lit::I128(hi), e.ty, live)
            }
            _ => Ok(lane),
        }
    }

    /// Trap unless the lane's payload fits `[lo, hi]`. A NULL row never
    /// traps: its payload is a masked default, and DuckDB has no value there
    /// to overflow either.
    fn narrow_trap(
        &mut self,
        l: Lane,
        lo: Lit,
        hi: Lit,
        ty: Ty,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        let lane = ty.lane();
        let lo_v = self.const_lit(lo);
        let hi_v = self.const_lit(hi);
        let ge = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Ge,
            ty: lane,
            dst: ge,
            a: l.val,
            b: lo_v,
        });
        let le = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Le,
            ty: lane,
            dst: le,
            a: l.val,
            b: hi_v,
        });
        let in_range = self.bin(BinOp::And, ge, le);
        let out = self.not(in_range);
        let bad = match l.flag {
            Some(f) => self.bin(BinOp::And, f, out),
            None => out,
        };
        let msg = format!(
            "Out of Range Error: value out of range for {} (arrow {})",
            duck_narrow_name(ty),
            arrow_narrow_name(ty)
        );
        Ok(self.trap_if(bad, msg, l, ty, live))
    }

    /// Branch to a trap when `bad` (already flag-masked) is set; otherwise
    /// thread `l` and the live set through a continuation block.
    fn trap_if(&mut self, bad: Value, msg: String, l: Lane, ty: Ty, live: &mut Live) -> Lane {
        let (trap_b, _) = self.create_block(&[]);
        let live_width = Self::live_types(live).len();
        let mut cont_tys = Self::live_types(live);
        let flag_in_shape = l.flag.is_some();
        if flag_in_shape {
            cont_tys.push(Ty::I1);
        }
        cont_tys.push(ty.lane());
        let (cont_b, cont_p) = self.create_block(&cont_tys);
        let mut args = Self::live_args(live);
        if let Some(f) = l.flag {
            args.push(f);
        }
        args.push(l.val);
        self.term(Term::Brif {
            cond: bad,
            then_to: BlockId(trap_b as u32),
            then_args: vec![],
            else_to: BlockId(cont_b as u32),
            else_args: args,
        });
        self.switch(trap_b);
        self.term(Term::Trap { msg });
        self.switch(cont_b);
        self.enter_block(live, &cont_p[..live_width]);
        let tail = &cont_p[live_width..];
        if flag_in_shape {
            Lane {
                flag: Some(tail[0]),
                val: tail[1],
            }
        } else {
            Lane {
                flag: None,
                val: tail[0],
            }
        }
    }

    /// The per-kind body of [`Self::emit`], which is its only caller: the
    /// narrow-width range trap belongs to `emit`, so reaching a kind from
    /// anywhere else would skip the trap.
    fn emit_kind(&mut self, e: &SExpr, live: &mut Live) -> Result<Lane, PrepareError> {
        match &e.kind {
            // The previous stage's column: the bottom of the live stack,
            // rebound at every block transition, so this is whatever
            // register holds it in the current block. Nothing is evaluated.
            SKind::Slot(i) if (*i as usize) < self.n_slots => Ok(live[*i as usize].0),
            // A shared value: computed before the projection's first item,
            // riding the live stack like a slot.
            SKind::Shared(k) => match self.shared_at.get(*k as usize).copied().flatten() {
                Some(at) => Ok(live[at].0),
                None => Err(PrepareError::Internal(format!(
                    "shared value {k} read where it is not live"
                ))),
            },
            // `share.rs` reads every let in its place before lowering.
            SKind::Let(n) => Err(PrepareError::Internal(format!(
                "let {n} read where share did not place it"
            ))),
            SKind::Slot(i) => Err(PrepareError::Internal(format!(
                "slot {i} read outside the previous stage's {} columns",
                self.n_slots
            ))),
            SKind::Col(idx) => {
                if let Some(lane) = self.blocks[self.cur].cache.get(idx) {
                    return Ok(*lane);
                }
                let col = &self.in_cols[*idx as usize];
                let lane = if col.ty.nullable {
                    let flag = self.fresh();
                    let val = self.fresh();
                    self.inst(Inst::LoadOpt {
                        flag,
                        dst: val,
                        col: *idx,
                    });
                    Lane {
                        flag: Some(flag),
                        val,
                    }
                } else {
                    let val = self.fresh();
                    self.inst(Inst::Load {
                        dst: val,
                        col: *idx,
                    });
                    Lane { flag: None, val }
                };
                self.blocks[self.cur].cache.insert(*idx, lane);
                Ok(lane)
            }
            SKind::StaticCol { join, col } => {
                let (valid_hit, dsts) = self.emit_probe(*join, live)?;
                let slots = self.val_slots(*join);
                let (validity, payload) = slots[*col as usize];
                let hit_flag = match self.joins[*join as usize].kind {
                    // INNER: a miss already skipped the row before any
                    // expression could look — the lane is provably valid.
                    JoinKind::Inner => None,
                    JoinKind::Left => Some(valid_hit),
                };
                // A nullable static column carries its own validity dst; on
                // a LEFT miss the probe defaults it to false, so the AND is
                // correct without extra guards.
                let vflag = validity.map(|vi| dsts[vi]);
                let flag = self.combine_flags(hit_flag, vflag);
                Ok(Lane {
                    flag,
                    val: dsts[payload],
                })
            }
            SKind::Lit(lit) => Ok(Lane {
                flag: None,
                val: self.const_lit(lit.clone()),
            }),
            SKind::NullOf => {
                let flag = self.const_i1(false);
                let val = self.default_of(e.ty);
                Ok(Lane {
                    flag: Some(flag),
                    val,
                })
            }
            SKind::Extreme { greatest, args } => {
                // A running extreme: each argument in order, replacing the
                // held one when it is not NULL and the held one is NULL or
                // strictly beaten, so ties keep the first. The held lane
                // stays live while the next argument may branch to a trap.
                let pred = if *greatest { CmpPred::Gt } else { CmpPred::Lt };
                let lane_ty = e.ty.lane();
                let mut held = self.emit(&args[0], live)?;
                for a in &args[1..] {
                    live.push((held, e.ty));
                    let l = self.emit(a, live)?;
                    let (m, _) = live.pop().expect("pushed above");
                    let beats = self.fresh();
                    self.inst(Inst::Cmp {
                        pred,
                        ty: lane_ty,
                        dst: beats,
                        a: l.val,
                        b: m.val,
                    });
                    let take = match m.flag {
                        None => beats,
                        Some(mv) => {
                            let m_null = self.not(mv);
                            self.bin(BinOp::Or, m_null, beats)
                        }
                    };
                    let take = match l.flag {
                        None => take,
                        Some(lv) => self.bin(BinOp::And, lv, take),
                    };
                    let val = self.fresh();
                    self.inst(Inst::Select {
                        dst: val,
                        cond: take,
                        a: l.val,
                        b: m.val,
                    });
                    let flag = match (l.flag, m.flag) {
                        (Some(lv), Some(mv)) => Some(self.bin(BinOp::Or, lv, mv)),
                        _ => None,
                    };
                    held = Lane { flag, val };
                }
                Ok(held)
            }
            SKind::Seq { items, pick } => {
                // In order, so the first trap is DuckDB's field order's. The
                // answer stays live across the items after it, which may
                // branch to a trap.
                let mut held = false;
                for (i, it) in items.iter().enumerate() {
                    let l = self.emit(it, live)?;
                    if i == *pick {
                        live.push((l, it.ty));
                        held = true;
                    }
                }
                debug_assert!(held, "pick is an item");
                let (l, _) = live.pop().expect("pushed above");
                Ok(l)
            }
            SKind::Raise(msg) => {
                // Reaching it traps; the continuation (a NULL lane) is
                // unreachable but keeps the block shape every caller expects.
                let flag = self.const_i1(false);
                let val = self.default_of(e.ty);
                let bad = self.const_i1(true);
                Ok(self.trap_if(bad, msg.clone(), Lane { flag: Some(flag), val }, e.ty, live))
            }
            SKind::DecToFloat(inner) => {
                let (p, s) = inner.ty.dec().expect("dtof operand is a decimal");
                let l = self.emit(inner, live)?;
                let dst = self.fresh();
                self.inst(Inst::Dtof { p, s, dst, a: l.val });
                Ok(Lane {
                    val: dst,
                    flag: l.flag,
                })
            }
            SKind::DecArith { op, check, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let flag = self.combine_flags(la.flag, lb.flag);
                // `%` by a zero DECIMAL is NULL, the integer `%` rule.
                let nonzero_lit = matches!(b.kind, SKind::Lit(Lit::Dec(n, ..)) if n != 0);
                let (bp, bs) = b.ty.dec().expect("decimal operand");
                let flag = if *op == DecOp::Rem && !nonzero_lit {
                    let zero = self.const_lit(Lit::Dec(0, bp, bs));
                    let nz = self.fresh();
                    self.inst(Inst::Cmp {
                        pred: CmpPred::Ne,
                        ty: b.ty,
                        dst: nz,
                        a: lb.val,
                        b: zero,
                    });
                    self.combine_flags(flag, Some(nz))
                } else {
                    flag
                };
                // A NULL row computes 0 op 0 (0 % 1), which never traps.
                let (va, vb) = match flag {
                    Some(f) => {
                        let (ap, asc) = a.ty.dec().expect("decimal operand");
                        let da = self.const_lit(Lit::Dec(0, ap, asc));
                        let safe = if *op == DecOp::Rem { 10i128.pow(bs as u32) } else { 0 };
                        let db = self.const_lit(Lit::Dec(safe, bp, bs));
                        (self.select_of(f, la.val, da), self.select_of(f, lb.val, db))
                    }
                    None => (la.val, lb.val),
                };
                let dst = self.fresh();
                self.inst(Inst::Dop {
                    op: *op,
                    check: *check,
                    ty: e.ty,
                    dst,
                    a: va,
                    b: vb,
                });
                Ok(Lane { flag, val: dst })
            }
            SKind::DecCast(inner) => {
                let l = self.emit(inner, live)?;
                let dst = self.fresh();
                if e.ty == Ty::Str {
                    let (p, s) = inner.ty.dec().expect("dtos operand is a decimal");
                    self.inst(Inst::Dtos {
                        p,
                        s,
                        dst,
                        a: l.val,
                    });
                } else {
                    // Mask the payload under a NULL flag: 0 converts to
                    // anything, so a NULL row never traps.
                    let a = match l.flag {
                        Some(f) => {
                            let d = self.default_of(inner.ty);
                            self.select_of(f, l.val, d)
                        }
                        None => l.val,
                    };
                    self.inst(Inst::Dcast {
                        from: inner.ty.lane(),
                        to: e.ty,
                        dst,
                        a,
                    });
                }
                Ok(Lane {
                    val: dst,
                    flag: l.flag,
                })
            }
            SKind::DecTryCast(inner) => {
                let l = self.emit(inner, live)?;
                let a = match l.flag {
                    Some(f) => {
                        let d = self.default_of(inner.ty);
                        self.select_of(f, l.val, d)
                    }
                    None => l.val,
                };
                let (from, to) = (inner.ty.lane(), e.ty);
                let ok = self.fresh();
                self.inst(Inst::DcastOk { from, to, dst: ok, a });
                // A failing row converts 0 instead, and its flag says NULL.
                let zero = self.default_of(inner.ty);
                let safe = self.select_of(ok, a, zero);
                let dst = self.fresh();
                self.inst(Inst::Dcast {
                    from,
                    to,
                    dst,
                    a: safe,
                });
                let flag = self.combine_flags(l.flag, Some(ok));
                Ok(Lane { val: dst, flag })
            }
            SKind::DecUnary { op, k, m, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Dunary {
                    op: *op,
                    k: *k,
                    m: *m,
                    ty: e.ty,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    val: dst,
                    flag: l.flag,
                })
            }
            SKind::IntToDec { s: _, a: inner } => {
                let (p, s) = e.ty.dec().expect("itod result is a decimal");
                let l = self.emit(inner, live)?;
                let dst = self.fresh();
                self.inst(Inst::Itod { p, s, dst, a: l.val });
                Ok(Lane {
                    val: dst,
                    flag: l.flag,
                })
            }
            SKind::IntToFloat(inner) | SKind::IntToFloat32(inner) => {
                let narrow = matches!(e.kind, SKind::IntToFloat32(_));
                let l = self.emit(inner, live)?;
                let dst = self.fresh();
                if inner.ty.lane() == Ty::I128 {
                    if narrow {
                        return Err(PrepareError::Unsupported(format!(
                            "a {} tree feature (sklearn's float32 narrowing of a 128-bit \
                             integer is not modelled)",
                            duck_narrow_name(inner.ty)
                        )));
                    }
                    self.inst(Inst::Htof { dst, a: l.val });
                } else {
                    self.inst(Inst::Itof {
                        narrow,
                        dst,
                        a: l.val,
                    });
                }
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Arith { op, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let ir_op = match (op, e.ty.lane()) {
                    (ArithOp::Add, Ty::I64) => BinOp::Iadd,
                    (ArithOp::Sub, Ty::I64) => BinOp::Isub,
                    (ArithOp::Mul, Ty::I64) => BinOp::Imul,
                    (ArithOp::Rem, Ty::I64) => BinOp::Irem,
                    (ArithOp::IDiv, Ty::I64) => BinOp::Idiv,
                    (ArithOp::Add, Ty::F64) => BinOp::Fadd,
                    (ArithOp::Sub, Ty::F64) => BinOp::Fsub,
                    (ArithOp::Mul, Ty::F64) => BinOp::Fmul,
                    (ArithOp::Div, Ty::F64) => BinOp::Fdiv,
                    // `//` on doubles is PLAIN division (wave-3 pins); the
                    // zero-divisor NULL is the result flag set below.
                    (ArithOp::IDiv, Ty::F64) => BinOp::Fdiv,
                    (ArithOp::Rem, Ty::F64) => BinOp::Frem,
                    (ArithOp::Shl, Ty::I64) => BinOp::Ishl,
                    (ArithOp::Shr, Ty::I64) => BinOp::Ishr,
                    (ArithOp::BitAnd, Ty::I64) => BinOp::Iand,
                    (ArithOp::BitOr, Ty::I64) => BinOp::Ior,
                    (ArithOp::BitXor, Ty::I64) => BinOp::Ixor,
                    (ArithOp::Add, Ty::I128) => BinOp::Hadd,
                    (ArithOp::Sub, Ty::I128) => BinOp::Hsub,
                    (ArithOp::Mul, Ty::I128) => BinOp::Hmul,
                    (ArithOp::IDiv, Ty::I128) => BinOp::Hdiv,
                    (ArithOp::Rem, Ty::I128) => BinOp::Hrem,
                    (ArithOp::BitAnd, Ty::I128) => BinOp::Hand,
                    (ArithOp::BitOr, Ty::I128) => BinOp::Hor,
                    (ArithOp::BitXor, Ty::I128) => BinOp::Hxor,
                    (op, ty) => {
                        return Err(PrepareError::Internal(format!(
                            "arith {op:?} on {} escaped the frontend",
                            ty.name()
                        )))
                    }
                };
                // Integer arithmetic traps (overflow, shift ranges, % edge
                // cases): when EITHER operand is NULL, mask BOTH payloads, so
                // a NULL row computes 0 op 0 (0 op 1 for / and %) and can
                // never fire the trap. Masking each operand by its own flag
                // is not enough: a NULL left operand masked to 0 still
                // overflows `0 - i64::MIN` and still traps `0 << -1`, where
                // DuckDB answers NULL. A zero divisor clears the flag below,
                // so the masked divisor is 1, never 0. Float ops are total —
                // no masking.
                let flag = self.combine_flags(la.flag, lb.flag);
                // Integer `%` and `//` on either lane answer NULL for a zero
                // divisor: the divisor's non-zero test joins the result
                // flag, so the masking below feeds the op 0 / 1, never 0 / 0.
                // Both operands were evaluated above whatever the divisor, as
                // DuckDB evaluates them, so a trap in the dividend still fires.
                let nonzero_lit = matches!(b.kind, SKind::Lit(Lit::I64(n)) if n != 0)
                    || matches!(b.kind, SKind::Lit(Lit::I128(n)) if n != 0)
                    || matches!(b.kind, SKind::Lit(Lit::F64(x)) if x != 0.0);
                let flag = if plan::zero_divisor_nulls(*op, e.ty) && !nonzero_lit {
                    let lane = e.ty.lane();
                    let zero = self.const_lit(match lane {
                        Ty::F64 => Lit::F64(0.0),
                        Ty::I128 => Lit::I128(0),
                        _ => Lit::I64(0),
                    });
                    let nz = self.fresh();
                    self.inst(Inst::Cmp {
                        pred: CmpPred::Ne,
                        ty: lane,
                        dst: nz,
                        a: lb.val,
                        b: zero,
                    });
                    self.combine_flags(flag, Some(nz))
                } else {
                    flag
                };
                let (va, vb) = match (e.ty.lane(), flag) {
                    (Ty::I64, Some(f)) => {
                        let safe_b = if matches!(ir_op, BinOp::Idiv | BinOp::Irem) {
                            1
                        } else {
                            0
                        };
                        let da = self.const_lit(Lit::I64(0));
                        let db = self.const_lit(Lit::I64(safe_b));
                        (self.select_of(f, la.val, da), self.select_of(f, lb.val, db))
                    }
                    (Ty::I128, Some(f)) => {
                        let safe_b = i128::from(matches!(ir_op, BinOp::Hdiv | BinOp::Hrem));
                        let da = self.const_lit(Lit::I128(0));
                        let db = self.const_lit(Lit::I128(safe_b));
                        (self.select_of(f, la.val, da), self.select_of(f, lb.val, db))
                    }
                    _ => (la.val, lb.val),
                };
                let mut val = self.bin(ir_op, va, vb);
                // A right shift by the width or more is 0 at a narrow width
                // too (`(-1)::TINYINT >> 8` is 0 on DuckDB), where the i64
                // shift would keep the sign bits; negative counts are
                // already 0 in the kernel.
                if let (ArithOp::Shr, Some(bits)) = (op, narrow_bits(e.ty)) {
                    let w = self.const_lit(Lit::I64(bits));
                    let inrange = self.fresh();
                    self.inst(Inst::Cmp {
                        pred: CmpPred::Lt,
                        ty: Ty::I64,
                        dst: inrange,
                        a: vb,
                        b: w,
                    });
                    let zero = self.const_lit(Lit::I64(0));
                    val = self.select_of(inrange, val, zero);
                }
                let lane = Lane { flag, val };
                // MIN % -1 at a NARROW width. DuckDB computes the modulo
                // through the checked division, which overflows at the
                // width even though the mathematical result (0) is in range —
                // the one narrow overflow a RESULT-range check cannot see.
                // Masked payloads (defaults) can never equal a negative MIN,
                // so a NULL row cannot fire this.
                if let (ArithOp::Rem, Some((lo, _))) = (op, e.ty.int_range()) {
                    let lo_v = self.const_lit(Lit::I64(lo));
                    let m1 = self.const_lit(Lit::I64(-1));
                    let at_min = self.fresh();
                    self.inst(Inst::Cmp {
                        pred: CmpPred::Eq,
                        ty: Ty::I64,
                        dst: at_min,
                        a: va,
                        b: lo_v,
                    });
                    let div_m1 = self.fresh();
                    self.inst(Inst::Cmp {
                        pred: CmpPred::Eq,
                        ty: Ty::I64,
                        dst: div_m1,
                        a: vb,
                        b: m1,
                    });
                    let bad = self.bin(BinOp::And, at_min, div_m1);
                    let msg = format!("Out of Range Error: Overflow in division of {lo} / -1");
                    return Ok(self.trap_if(bad, msg, lane, e.ty, live));
                }
                Ok(lane)
            }
            SKind::Cmp { pred, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: *pred,
                    ty: a.ty.lane(),
                    dst,
                    a: la.val,
                    b: lb.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lb.flag),
                    val: dst,
                })
            }
            SKind::Not(inner) => {
                let l = self.emit(inner, live)?;
                let val = self.not(l.val);
                Ok(Lane { flag: l.flag, val })
            }
            SKind::And { a, b } => self.kleene(a, b, live, true, e.nullable),
            SKind::Or { a, b } => self.kleene(a, b, live, false, e.nullable),
            SKind::IsNull { negated, inner } => {
                let l = self.emit(inner, live)?;
                let val = match l.flag {
                    None => self.const_i1(*negated),
                    Some(f) => {
                        if *negated {
                            f
                        } else {
                            self.not(f)
                        }
                    }
                };
                Ok(Lane { flag: None, val })
            }
            SKind::Case { arms, default } => self.case(e, arms, default.as_deref(), live),
            SKind::Cast { inner, trying } => self.cast(e, inner, *trying, live),
            SKind::StrCase { upper, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                let op = if *upper { StrOp1::Upper } else { StrOp1::Lower };
                self.inst(Inst::Str1 { op, dst, a: l.val });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Trim { side, a, chars } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lc = self.emit(chars, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Strim {
                    side: *side,
                    dst,
                    a: la.val,
                    chars: lc.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lc.flag),
                    val: dst,
                })
            }
            SKind::Substr { a, start, len } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let ls = self.emit(start, live)?;
                live.push((ls, Ty::I64));
                let ll = match len {
                    Some(l) => Some(self.emit(l, live)?),
                    None => None,
                };
                let (ls, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                // The range guards trap; mask nullable position payloads.
                let start_v = self.masked(ls, Ty::I64);
                let len_v = ll.map(|l| self.masked(l, Ty::I64));
                let dst = self.fresh();
                self.inst(Inst::Ssubstr {
                    dst,
                    a: la.val,
                    start: start_v,
                    len: len_v,
                });
                let flag = self.combine_flags(la.flag, ls.flag);
                Ok(Lane {
                    flag: self.combine_flags(flag, ll.and_then(|l| l.flag)),
                    val: dst,
                })
            }
            SKind::Abs(a) => {
                let l = self.emit(a, live)?;
                // iabs / habs trap on the lane's minimum; mask the nullable
                // payload.
                let (op, av) = match e.ty.lane() {
                    Ty::I64 => (NumOp1::Iabs, self.masked(l, Ty::I64)),
                    Ty::I128 => (NumOp1::Habs, self.masked(l, Ty::I128)),
                    _ => (NumOp1::Fabs, l.val),
                };
                let dst = self.fresh();
                self.inst(Inst::Num1 { op, dst, a: av });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Like { ci, a, p, esc } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lp = self.emit(p, live)?;
                live.push((lp, Ty::Str));
                let le = match esc {
                    Some(e) => Some(self.emit(e, live)?),
                    None => None,
                };
                let (lp, _) = {
                    let x = live.pop().expect("pushed above");
                    let _ = &x;
                    x
                };
                let (la, _) = live.pop().expect("pushed above");
                // Trapping op: mask every nullable payload to "" — the
                // empty string/pattern/escape are all outside the trap
                // domain (like("", "", no-escape) is a clean false/true).
                let empty = self.const_lit(Lit::Str(String::new()));
                let mask = |fbb: &mut Self, l: Lane| match l.flag {
                    None => l.val,
                    Some(f) => fbb.select_of(f, l.val, empty),
                };
                let av = mask(self, la);
                let pv = mask(self, lp);
                let (ev, eflag) = match le {
                    Some(l) => (Some(mask(self, l)), l.flag),
                    None => (None, None),
                };
                let dst = self.fresh();
                self.inst(Inst::Slike {
                    ci: *ci,
                    dst,
                    a: av,
                    p: pv,
                    esc: ev,
                });
                let f1 = self.combine_flags(la.flag, lp.flag);
                Ok(Lane {
                    flag: self.combine_flags(f1, eflag),
                    val: dst,
                })
            }
            SKind::Round2 { trunc, a, n } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let ln = self.emit(n, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                if a.ty == Ty::F64 {
                    self.inst(Inst::Round2f {
                        trunc: *trunc,
                        dst,
                        a: la.val,
                        n: ln.val,
                    });
                } else {
                    self.inst(Inst::Round2i {
                        trunc: *trunc,
                        dst,
                        a: la.val,
                        n: ln.val,
                    });
                }
                Ok(Lane {
                    flag: self.combine_flags(la.flag, ln.flag),
                    val: dst,
                })
            }
            SKind::MathF1 { op, a } => {
                let l = self.emit(a, live)?;
                // Safe mask per op: the value must be OUTSIDE the trap
                // domain (1.0 for logs; 0.0 is fine for sqrt/trig; total
                // ops need no mask at all).
                let av = match op {
                    NumOp1::Ln | NumOp1::Log2 | NumOp1::Log10 => self.masked_to(l, Lit::F64(1.0)),
                    NumOp1::Fsqrt | NumOp1::Fsin | NumOp1::Fcos | NumOp1::Ftan => {
                        self.masked_to(l, Lit::F64(0.0))
                    }
                    _ => l.val,
                };
                let dst = self.fresh();
                self.inst(Inst::Num1 {
                    op: *op,
                    dst,
                    a: av,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::MathF2 { op, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, a.ty));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                // Flogb traps on base<=0 / base==1 / x<=0. NULL pre-empts
                // EVERY domain check (pinned: log(-2.0, NULL) is NULL, not
                // an error) — so both payloads mask under the COMBINED
                // flag: either side NULL makes both sides safe.
                let flag = self.combine_flags(la.flag, lb.flag);
                let (av, bv) = match op {
                    BinOp::Flogb => {
                        let a = Lane { flag, val: la.val };
                        let b = Lane { flag, val: lb.val };
                        (
                            self.masked_to(a, Lit::F64(10.0)),
                            self.masked_to(b, Lit::F64(1.0)),
                        )
                    }
                    _ => (la.val, lb.val),
                };
                let val = self.bin(*op, av, bv);
                Ok(Lane { flag, val })
            }
            SKind::Round(a) => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Num1 {
                    op: NumOp1::Fround,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Str2 { op, a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let flag = self.combine_flags(la.flag, lb.flag);
                // Jaccard/Hamming trap on empty inputs / length mismatch —
                // and the "" mask default IS in the trap domain. NULL
                // pre-empts the checks (jaccard(NULL, '') is NULL, not an
                // error), so BOTH payloads mask to "a" under the COMBINED
                // flag, the Flogb pattern. The edit distances trap where
                // their table is refused, and a NULL row's payload can be
                // long (`s || repeat('a', 30000)` over a NULL `s`), so they
                // mask too.
                let (av, bv) = match op {
                    StrOp2::Jaccard
                    | StrOp2::Hamming
                    | StrOp2::Levenshtein
                    | StrOp2::Damerau => {
                        let a = Lane { flag, val: la.val };
                        let b = Lane { flag, val: lb.val };
                        (
                            self.masked_to(a, Lit::Str("a".into())),
                            self.masked_to(b, Lit::Str("a".into())),
                        )
                    }
                    _ => (la.val, lb.val),
                };
                let dst = self.fresh();
                self.inst(Inst::Str2 {
                    op: *op,
                    dst,
                    a: av,
                    b: bv,
                });
                Ok(Lane { flag, val: dst })
            }
            SKind::Str3 { op, a, b, c } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lb = self.emit(b, live)?;
                live.push((lb, Ty::Str));
                let lc = self.emit(c, live)?;
                let (lb, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Str3 {
                    op: *op,
                    dst,
                    a: la.val,
                    b: lb.val,
                    c: lc.val,
                });
                let f = self.combine_flags(la.flag, lb.flag);
                Ok(Lane {
                    flag: self.combine_flags(f, lc.flag),
                    val: dst,
                })
            }
            SKind::Str2i { op, a, n } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let ln = self.emit(n, live)?;
                let (la, _) = live.pop().expect("pushed above");
                // Total under masked defaults: repeat("", any n) allocates
                // nothing (the cap guard sees 0 bytes) and extract is total.
                let dst = self.fresh();
                self.inst(Inst::Str2i {
                    op: *op,
                    dst,
                    a: la.val,
                    n: ln.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, ln.flag),
                    val: dst,
                })
            }
            SKind::Spad { left, a, len, pad } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let ll = self.emit(len, live)?;
                live.push((ll, Ty::I64));
                let lp = self.emit(pad, live)?;
                let (ll, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                // The empty-pad trap fires only when growth is needed; a
                // VALID len with a NULL pad must not trap (NULL pre-empts).
                // Mask len to 0 under the COMBINED flag — len 0 returns ''
                // before the pad is ever examined.
                let flag = {
                    let f = self.combine_flags(la.flag, ll.flag);
                    self.combine_flags(f, lp.flag)
                };
                let len_v = self.masked_to(Lane { flag, val: ll.val }, Lit::I64(0));
                let dst = self.fresh();
                self.inst(Inst::Spad {
                    left: *left,
                    dst,
                    a: la.val,
                    len: len_v,
                    pad: lp.val,
                });
                Ok(Lane { flag, val: dst })
            }
            SKind::Sslice { a, lo, hi } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let llo = self.emit(lo, live)?;
                live.push((llo, Ty::I64));
                let lhi = self.emit(hi, live)?;
                let (llo, _) = live.pop().expect("pushed above");
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Sslice {
                    dst,
                    a: la.val,
                    lo: llo.val,
                    hi: lhi.val,
                });
                let f = self.combine_flags(la.flag, llo.flag);
                Ok(Lane {
                    flag: self.combine_flags(f, lhi.flag),
                    val: dst,
                })
            }
            SKind::Sord { empty_zero, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Sord {
                    empty_zero: *empty_zero,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::StripAccents(a) | SKind::Reverse(a) => {
                let op = match e.kind {
                    SKind::StripAccents(_) => StrOp1::StripAccents,
                    _ => StrOp1::Reverse,
                };
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::Str1 {
                    op,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::JoinHit(j) => {
                let (match_v, _) = self.emit_probe(*j, live)?;
                Ok(Lane {
                    flag: None,
                    val: match_v,
                })
            }
            SKind::ExternCall {
                site,
                ext,
                args,
                ret,
                whole,
            } => {
                let (whole_v, lanes) = self.emit_extern(*site, *ext, args, live)?;
                if *whole {
                    Ok(Lane {
                        flag: None,
                        val: whole_v,
                    })
                } else {
                    Ok(Lane {
                        flag: Some(lanes[2 * *ret as usize]),
                        val: lanes[2 * *ret as usize + 1],
                    })
                }
            }
            SKind::SLen { bytes, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::SLen {
                    bytes: *bytes,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::TreePredict { model, id, feats } => {
                // Every operand rides the `live` stack while its siblings are
                // emitted. A CASE or guarded CAST inside a later feature
                // splits the CFG, and `rebind_live` then rewrites the earlier
                // lanes to the new block's params — so a `Lane` still held in
                // a local from before the split names a register that block
                // cannot see (measured: `COALESCE(x, 0.0)` around a nullable
                // feature stranded the id). Reading the operands back OUT of
                // `live` afterwards is the point, not the pushing; this is
                // the same shape as `emit_probe` and `emit_extern`.
                let idl = self.emit(id, live)?;
                live.push((idl, Ty::I64));
                for f in feats {
                    let lane = self.emit(f, live)?;
                    live.push((lane, f.ty));
                }
                let start = live.len() - (feats.len() + 1);
                let (idl, _) = live[start];
                let id_flag = idl.flag;
                // MASK the id, as every other trapping op does. Under a false
                // validity flag the payload is arbitrary, and `predict` traps
                // on an id it does not know — so an unmasked payload turns a
                // NULL id into a hard error the moment anything arithmetic
                // sits between the probe and the call (`p.est - 1` on a LEFT
                // miss). Model 0 always exists (an empty model set is refused
                // at build), so the type default is a safe id whose result
                // the flag discards anyway.
                let id_val = self.masked(idl, Ty::I64);
                // A NULL feature is handed to the model as NaN — it has a
                // defined answer for missing (the node's learned direction),
                // so its validity is deliberately NOT folded into the result
                // flag. Only the id's is: an unseen group has no model.
                //
                // Both the NaN constant and the selects are emitted HERE, in
                // whatever block the operands ended up in, so the hoisted
                // constant cannot be stranded either.
                let mut nan: Option<Value> = None;
                let mut vals = Vec::with_capacity(feats.len());
                for i in 0..feats.len() {
                    let (l, _) = live[start + 1 + i];
                    // A non-nullable feature is already the value; only a
                    // nullable one needs the NULL-to-NaN select.
                    let Some(flag) = l.flag else {
                        vals.push(l.val);
                        continue;
                    };
                    let n = match nan {
                        Some(n) => n,
                        None => {
                            let n = self.fresh();
                            self.inst(Inst::Const {
                                dst: n,
                                // Sign chosen, not inherited from `f64::NAN`
                                // (see `Lit` in ir/mod.rs).
                                lit: super::ir::Lit::F64(f64::NAN.copysign(1.0)),
                            });
                            nan = Some(n);
                            n
                        }
                    };
                    let v = self.fresh();
                    self.inst(Inst::Select {
                        dst: v,
                        cond: flag,
                        a: l.val,
                        b: n,
                    });
                    vals.push(v);
                }
                live.truncate(start);
                let dst = self.fresh();
                self.inst(Inst::Predict {
                    static_id: (self.model_base + *model as usize) as u32,
                    dst,
                    id: id_val,
                    feats: vals,
                });
                Ok(Lane {
                    flag: id_flag,
                    val: dst,
                })
            }
            SKind::ReMatch { re, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::ReMatch {
                    re: *re,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::ReExtract { re, group, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::ReExtract {
                    re: *re,
                    group: *group,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::ReReplace { re, global, a } => {
                let l = self.emit(a, live)?;
                let dst = self.fresh();
                self.inst(Inst::ReReplace {
                    re: *re,
                    global: *global,
                    dst,
                    a: l.val,
                });
                Ok(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            SKind::Concat { a, b } => {
                let la = self.emit(a, live)?;
                live.push((la, Ty::Str));
                let lb = self.emit(b, live)?;
                let (la, _) = live.pop().expect("pushed above");
                let dst = self.fresh();
                self.inst(Inst::Sconcat {
                    dst,
                    a: la.val,
                    b: lb.val,
                });
                Ok(Lane {
                    flag: self.combine_flags(la.flag, lb.flag),
                    val: dst,
                })
            }
        }
    }

    /// Flattened probe-dst layout for join `j`: per value column either
    /// `(None, payload_idx)` or `(Some(validity_idx), payload_idx)` —
    /// nullable static value columns ride as validity+payload pairs,
    /// mirroring the StaticTy::Map flattening.
    fn val_slots(&self, j: u32) -> Vec<(Option<usize>, usize)> {
        plan::slot_pairs(&self.map_vals(j))
    }

    /// Flattened probe-dst TYPES for join `j` (same order as val_slots).
    fn val_flat_tys(&self, j: u32) -> Vec<Ty> {
        self.map_vals(j).iter().flat_map(|v| v.slots()).collect()
    }

    /// Join `j`'s value layouts, read from whichever column list this join's
    /// build side actually is: the batch's own input columns under `batch`
    /// (where `spec.table` means nothing), the static catalog otherwise.
    fn map_vals(&self, j: u32) -> Vec<plan::MapVal> {
        let spec = &self.joins[j as usize];
        let cols: &[Col] = if spec.batch {
            self.in_cols
        } else {
            &self.catalog[spec.table].cols
        };
        plan::map_vals(cols, &spec.val_cols)
    }

    /// A stage's joins, then its WHERE: joins run before WHERE (SQL order),
    /// each in FROM order — probe, and for INNER skip the row on a miss. A
    /// LEFT join's probe is also forced here so its key expressions are
    /// evaluated (and can trap) for every row that reaches it, exactly as
    /// the join would — even if no value column is ever referenced. Later
    /// references re-probe per block (pure, cached), same ponytail trade as
    /// column re-loads. A subquery's hidden join that the WHERE does not
    /// read is forced after the WHERE instead ([`plan::JoinSpec::after_where`]).
    /// `live` holds the previous stage's slots and is left as it was found.
    fn lower_stage_head(&mut self, stage: &plan::Stage, live: &mut Live) -> Result<(), PrepareError> {
        for &j in &stage.joins {
            if self.joins[j as usize].after_where || self.unforced(j) {
                continue;
            }
            let (valid_hit, _) = self.emit_probe(j, live)?;
            if self.joins[j as usize].kind == JoinKind::Inner {
                let tys = Self::live_types(live);
                let (keep, keep_params) = self.create_block(&tys);
                let (miss, _) = self.create_block(&tys);
                let args = Self::live_args(live);
                self.term(Term::Brif {
                    cond: valid_hit,
                    then_to: BlockId(keep as u32),
                    then_args: args.clone(),
                    else_to: BlockId(miss as u32),
                    else_args: args,
                });
                self.switch(miss);
                self.term(Term::Skip);
                self.switch(keep);
                self.enter_block(live, &keep_params);
            }
        }
        if let Some(pred) = &stage.pred {
            // The WHERE root is SELECTION context: emit_truth is where AND's
            // left-to-right laziness lives, recursively, at every depth
            // rather than only along a top-level conjunct spine. A filter
            // asks `pred IS TRUE`, and FALSE and NULL answer that question
            // alike, so the bare i1 is the whole story.
            let cond = self.emit_truth(pred, live)?;
            let tys = Self::live_types(live);
            let (keep, keep_params) = self.create_block(&tys);
            let (drop, _) = self.create_block(&tys);
            let args = Self::live_args(live);
            self.term(Term::Brif {
                cond,
                then_to: BlockId(keep as u32),
                then_args: args.clone(),
                else_to: BlockId(drop as u32),
                else_args: args,
            });
            self.switch(drop);
            self.term(Term::Skip);
            self.switch(keep);
            self.enter_block(live, &keep_params);
        }
        for &j in &stage.joins {
            if self.joins[j as usize].after_where && !self.unforced(j) {
                self.emit_probe(j, live)?;
            }
        }
        Ok(())
    }

    /// Whether the stage head leaves join `j` alone: the rows that a fold
    /// scans are read only by the fold, and a fold that cannot trap runs
    /// where a value reads it ([`Self::emit_fold`]). A fold that can trap
    /// runs here, for every row that reaches its join ([`plan::Fold::traps`]).
    fn unforced(&self, j: u32) -> bool {
        let spec = &self.joins[j as usize];
        spec.scan.is_some() || spec.fold.as_ref().is_some_and(|f| !f.traps)
    }

    /// Multiplicity loops for the joins under shape='many': one loop per
    /// join, nested in FROM order, the way DuckDB's left-deep plan feeds
    /// join k with every row that joins 0..k-1 produce, matched or
    /// null-extended (measured, 1.5.5 optimizer off: a key of the second
    /// join traps for a row that a LEFT first join null-extends, and not
    /// for one that an INNER first join drops).
    ///
    ///   entry_k:  join k's keys (they trap for every row reaching join k),
    ///             ProbeRange -> [lo, hi) (NULL keys force an EMPTY range:
    ///             a NULL never matches), jump header_k(lo, hi, any=false)
    ///   header_k(i, end, any): i < end ? body_k : after_k(any)
    ///   body_k:   ProbeRead lanes; the RESIDUAL gates match-ness (and the
    ///             `any` flag): pass -> next_k(i + 1, end, hit=true, lanes);
    ///             fail -> header_k(i + 1, end, any)
    ///   after_k:  LEFT and !any -> next_k(0, 0, hit=false, default lanes),
    ///             the null-extension; otherwise done_k
    ///   next_k:   entry_{k+1}; after the last join, WHERE gates emission
    ///             only (DuckDB joins first, filters second): pass ->
    ///             stores -> emit.to header_k(i, end, any=true); fail ->
    ///             header_k(i, end, any=true)
    ///   done_k:   header_{k-1}(i, end, any=true), the next row of the
    ///             enclosing loop; the outermost loop skips, as the input
    ///             row is finished. The null-extension rides the empty
    ///             range [0, 0), so its continuation leaves its loop at once.
    ///
    /// Each open loop's state rides the live stack as [i, end, hit,
    /// lanes...], outermost first. Expression emission may SPLIT blocks
    /// (CASE machinery), and each new block starts with an empty probe
    /// cache, so every open join's cache is re-seeded from those lanes
    /// before every emission ([`ManySeed`]).
    fn lower_many_loops(
        &mut self,
        joins: &[u32],
        exprs: &[(String, SExpr)],
        filter_pred: Option<&SExpr>,
        out_cols: &[Col],
    ) -> Result<(), PrepareError> {
        // Per open loop, outermost first: its header and where its state
        // starts on the live stack.
        let mut open: Vec<(usize, usize)> = Vec::new();
        let mut seeds: Vec<ManySeed> = Vec::new();
        let mut live: Live = Vec::new();
        for &j in joins {
            if self.joins[j as usize].scan.is_some() {
                continue;
            }
            if self.joins[j as usize].fold.is_some() {
                self.many_fold(j, &mut open, &mut seeds, &mut live)?;
                continue;
            }
            let spec = &self.joins[j as usize];
            let (kind, residual, keys_expr) = (spec.kind, spec.residual.clone(), spec.keys.clone());
            let key_layout = plan::map_keys(&keys_expr, &spec.key_cols);
            let dst_tys: Vec<Ty> = self.val_flat_tys(j);
            let nd = dst_tys.len();
            let at = live.len();

            // entry: the keys, each riding the live stack until the probe
            // (a later key's emission may split blocks), then the range.
            for key in &keys_expr {
                let lane = self.emit_many(key, &mut live, &seeds)?;
                live.push((lane, key.ty));
            }
            let mut keys_valid: Option<Value> = None;
            let mut key_vals = Vec::with_capacity(keys_expr.len());
            for (i, k) in key_layout.iter().enumerate() {
                let (lane, _) = live[at + i];
                // Every key here is plain (INDF is refused under `many`),
                // so every flag comes back and every one gates the range.
                if let Some(fl) = self.encode_probe_key(*k, lane, &mut key_vals) {
                    keys_valid = self.combine_flags(keys_valid, Some(fl));
                }
            }
            live.truncate(at);
            let lo = self.fresh();
            let hi = self.fresh();
            self.inst(Inst::ProbeRange {
                static_id: j,
                start: lo,
                end: hi,
                keys: key_vals,
            });
            let (lo, hi) = match keys_valid {
                None => (lo, hi),
                Some(fl) => {
                    let zero = self.const_lit(Lit::I64(0));
                    (self.select_of(fl, lo, zero), self.select_of(fl, hi, zero))
                }
            };

            let ctx_tys = Self::live_types(&live);
            let w = ctx_tys.len();
            let loop_tys = [ctx_tys.clone(), vec![Ty::I64, Ty::I64, Ty::I1]].concat();
            let (header, hp) = self.create_block(&loop_tys);
            let (body, bp) = self.create_block(&loop_tys);
            let (after, ap) = self.create_block(&[ctx_tys.clone(), vec![Ty::I1]].concat());
            let mut state: Live = vec![
                (Lane { flag: None, val: lo }, Ty::I64),
                (Lane { flag: None, val: hi }, Ty::I64),
                (Lane { flag: None, val: lo }, Ty::I1),
            ];
            for &ty in &dst_tys {
                state.push((Lane { flag: None, val: lo }, ty));
            }
            let (next, np) = self.create_block(&[ctx_tys, Self::live_types(&state)].concat());
            let f0 = self.const_i1(false);
            let mut args = Self::live_args(&live);
            args.extend([lo, hi, f0]);
            self.term(Term::Jump {
                to: BlockId(header as u32),
                args,
            });

            // header(i, end, any)
            self.switch(header);
            self.enter_block(&mut live, &hp[..w]);
            let (h_i, h_end, h_any) = (hp[w], hp[w + 1], hp[w + 2]);
            let more = self.fresh();
            self.inst(Inst::Cmp {
                pred: CmpPred::Lt,
                ty: Ty::I64,
                dst: more,
                a: h_i,
                b: h_end,
            });
            let ctx = Self::live_args(&live);
            self.term(Term::Brif {
                cond: more,
                then_to: BlockId(body as u32),
                then_args: [ctx.clone(), vec![h_i, h_end, h_any]].concat(),
                else_to: BlockId(after as u32),
                else_args: [ctx, vec![h_any]].concat(),
            });

            // body: the lanes and the residual, everything riding live as
            // [ctx..., inext, end, any, dsts...].
            self.switch(body);
            self.enter_block(&mut live, &bp[..w]);
            let (b_i, b_end, b_any) = (bp[w], bp[w + 1], bp[w + 2]);
            let dsts: Vec<Value> = (0..nd).map(|_| self.fresh()).collect();
            self.inst(Inst::ProbeRead {
                static_id: j,
                idx: b_i,
                dsts: dsts.clone(),
            });
            let one = self.const_lit(Lit::I64(1));
            let inext = self.bin(BinOp::Iadd, b_i, one);
            live.push((Lane { flag: None, val: inext }, Ty::I64));
            live.push((Lane { flag: None, val: b_end }, Ty::I64));
            live.push((Lane { flag: None, val: b_any }, Ty::I1));
            for (&d, &ty) in dsts.iter().zip(dst_tys.iter()) {
                live.push((Lane { flag: None, val: d }, ty));
            }
            let rv = match &residual {
                None => None,
                Some(res) => {
                    let mut own = seeds.clone();
                    own.push(ManySeed {
                        join: j,
                        hit: ManyHit::Const(true),
                        base: at + 3,
                        nd,
                    });
                    let rl = self.emit_many(res, &mut live, &own)?;
                    Some(self.truthy(rl))
                }
            };
            let vals = Self::live_args(&live);
            let t = self.const_i1(true);
            let mut match_args = vals[..w + 2].to_vec();
            match_args.push(t);
            match_args.extend(&vals[w + 3..]);
            match rv {
                None => self.term(Term::Jump {
                    to: BlockId(next as u32),
                    args: match_args,
                }),
                Some(cond) => self.term(Term::Brif {
                    cond,
                    then_to: BlockId(next as u32),
                    then_args: match_args,
                    else_to: BlockId(header as u32),
                    else_args: vals[..w + 3].to_vec(),
                }),
            }
            live.truncate(at);

            // after(any): the LEFT null-extension, then done.
            self.switch(after);
            self.enter_block(&mut live, &ap[..w]);
            if kind == JoinKind::Left {
                let a_any = ap[w];
                let (done, dp) = self.create_block(&Self::live_types(&live));
                let (miss, mp) = self.create_block(&Self::live_types(&live));
                let ctx = Self::live_args(&live);
                self.term(Term::Brif {
                    cond: a_any,
                    then_to: BlockId(done as u32),
                    then_args: ctx.clone(),
                    else_to: BlockId(miss as u32),
                    else_args: ctx,
                });
                self.switch(miss);
                self.enter_block(&mut live, &mp);
                let zero = self.const_lit(Lit::I64(0));
                let f = self.const_i1(false);
                let mut args = Self::live_args(&live);
                args.extend([zero, zero, f]);
                for &ty in &dst_tys {
                    let d = self.default_of(ty);
                    args.push(d);
                }
                self.term(Term::Jump {
                    to: BlockId(next as u32),
                    args,
                });
                self.switch(done);
                self.enter_block(&mut live, &dp);
            }
            // done: the next row of the enclosing loop, or the row is over.
            match open.last() {
                None => self.term(Term::Skip),
                Some(&(outer, oat)) => {
                    let args = self.continue_args(&live, oat);
                    self.term(Term::Jump {
                        to: BlockId(outer as u32),
                        args,
                    });
                }
            }

            // next: this loop's row, matched or null-extended, open for the
            // joins after it.
            self.switch(next);
            live.extend(state);
            self.enter_block(&mut live, &np);
            open.push((header, at));
            seeds.push(ManySeed {
                join: j,
                // An INNER loop has no null-extension: its hit is TRUE.
                hit: match kind {
                    JoinKind::Inner => ManyHit::Const(true),
                    JoinKind::Left => ManyHit::Live(at + 2),
                },
                base: at + 3,
                nd,
            });
        }
        let Some(&(inner, iat)) = open.last() else {
            return Err(PrepareError::Internal(
                "the many loops lowered with no join".to_string(),
            ));
        };
        if let Some(pred) = filter_pred {
            // SELECTION context, same as the scalar WHERE; it sees the
            // null-extended rows too (measured).
            let pv = self.emit_truth_many(pred, &mut live, &seeds)?;
            let (keep, kp) = self.create_block(&Self::live_types(&live));
            let args = self.continue_args(&live, iat);
            self.term(Term::Brif {
                cond: pv,
                then_to: BlockId(keep as u32),
                then_args: Self::live_args(&live),
                else_to: BlockId(inner as u32),
                else_args: args,
            });
            self.switch(keep);
            self.enter_block(&mut live, &kp);
        }
        self.store_out_row(exprs, out_cols, &seeds, &mut live)?;
        let args = self.continue_args(&live, iat);
        self.term(Term::EmitTo {
            to: BlockId(inner as u32),
            args,
        });
        Ok(())
    }

    /// A fold join ([`Self::emit_fold`]) among the loops of
    /// [`Self::lower_many_loops`]: one row, matched or not, as the LEFT
    /// join to the one group of a scalar subquery. Its state rides the live
    /// stack as a loop's does, [0, 0, any, values...], and its continuation
    /// leaves at once for the next row of the loop around it.
    fn many_fold(
        &mut self,
        j: u32,
        open: &mut Vec<(usize, usize)>,
        seeds: &mut Vec<ManySeed>,
        live: &mut Live,
    ) -> Result<(), PrepareError> {
        let at = live.len();
        self.many = seeds.clone();
        self.seed_many(live);
        let folded = self.emit_fold(j, live);
        self.many.clear();
        let (any, outs) = folded?;
        let tys = self.val_flat_tys(j);
        // cont(ctx, i, end, any): the level is over.
        let here = self.cur;
        let kept = live.clone();
        let ctx_tys = Self::live_types(live);
        let w = ctx_tys.len();
        let (cont, cp) = self.create_block(&[ctx_tys, vec![Ty::I64, Ty::I64, Ty::I1]].concat());
        self.switch(cont);
        self.enter_block(live, &cp[..w]);
        match open.last() {
            None => self.term(Term::Skip),
            Some(&(outer, oat)) => {
                let args = self.continue_args(live, oat);
                self.term(Term::Jump {
                    to: BlockId(outer as u32),
                    args,
                });
            }
        }
        self.switch(here);
        *live = kept;
        let zero = self.const_lit(Lit::I64(0));
        live.push((Lane { flag: None, val: zero }, Ty::I64));
        live.push((Lane { flag: None, val: zero }, Ty::I64));
        live.push((Lane { flag: None, val: any }, Ty::I1));
        for (&o, &ty) in outs.iter().zip(&tys) {
            live.push((Lane { flag: None, val: o }, ty));
        }
        open.push((cont, at));
        seeds.push(ManySeed {
            join: j,
            hit: ManyHit::Live(at + 2),
            base: at + 3,
            nd: tys.len(),
        });
        Ok(())
    }

    /// The arguments that continue the many loop whose state starts at
    /// `at` on the live stack with its next row: the state of the loops
    /// around it, then its own (i, end) and any=TRUE (a row got here).
    fn continue_args(&mut self, live: &Live, at: usize) -> Vec<Value> {
        let mut args = Self::live_args(&live[..at].to_vec());
        let t = self.const_i1(true);
        args.extend([live[at].0.val, live[at + 1].0.val, t]);
        args
    }

    /// Emit `e` with the probe caches of the many-joins `seeds` seeded in the
    /// current block AND re-created in every block a split inside `e`
    /// creates.
    ///
    /// The joins' values ride the live stack, so they survive a transition
    /// on their own — but the CACHE that says "@j is already probed, here are
    /// its registers" is per block (`PB::new` starts it empty). Without the
    /// re-creation, a joined column read inside a CASE arm misses the cache
    /// and falls through to the scalar `Inst::Probe`, which `ir::verify`
    /// rejects for a multimap: "@N is a multimap: use probe.range".
    fn emit_many(
        &mut self,
        e: &SExpr,
        live: &mut Live,
        seeds: &[ManySeed],
    ) -> Result<Lane, PrepareError> {
        self.many = seeds.to_vec();
        self.seed_many(live);
        let out = self.emit(e, live);
        self.many.clear();
        out
    }

    /// (Re)create the current block's probe cache entry for every active
    /// many-join, from the live lanes its values (and a live hit flag) ride
    /// on. No-op outside a many-join.
    fn seed_many(&mut self, live: &Live) {
        for i in 0..self.many.len() {
            let ManySeed {
                join,
                hit,
                base,
                nd,
            } = self.many[i];
            let h = match hit {
                ManyHit::Const(b) => self.const_i1(b),
                ManyHit::Live(at) => live[at].0.val,
            };
            let ds: Vec<Value> = live[base..base + nd].iter().map(|(l, _)| l.val).collect();
            self.blocks[self.cur].probes.insert(join, (h, ds));
        }
    }

    /// Every block transition goes through here: rebind the live stack to the
    /// block just switched to, then restore what that block cannot inherit.
    fn enter_block(&mut self, live: &mut Live, params: &[Value]) {
        Self::rebind_live(live, params);
        self.seed_many(live);
        self.seed_probes(live);
    }

    /// (Re)create this block's probe cache entry for every scalar join whose
    /// residual is currently being emitted. The hit is TRUE by construction —
    /// we are inside the `brif raw_hit -> eval` branch — but the constant is
    /// a register like any other and has to be re-materialised here.
    fn seed_probes(&mut self, live: &Live) {
        for i in 0..self.probe_seeds.len() {
            let ProbeSeed { join, base, nd } = self.probe_seeds[i];
            let t = self.const_i1(true);
            let ds: Vec<Value> = live[base..base + nd].iter().map(|(l, _)| l.val).collect();
            self.blocks[self.cur].probes.insert(join, (t, ds));
        }
    }

    /// Emit every output expression and store it. The probe caches of the
    /// many-joins `seeds` are re-seeded from `live` before EVERY
    /// expression — emission may split blocks, and each new block starts
    /// with an empty cache.
    fn store_out_row(
        &mut self,
        exprs: &[(String, SExpr)],
        out_cols: &[Col],
        seeds: &[ManySeed],
        live: &mut Live,
    ) -> Result<(), PrepareError> {
        for (ci, (_, e)) in exprs.iter().enumerate() {
            let lane = self.emit_many(e, live, seeds)?;
            let col = ci as u32;
            if out_cols[ci].ty.nullable {
                let flag = match lane.flag {
                    Some(fl) => fl,
                    None => self.const_i1(true),
                };
                self.inst(Inst::StoreOpt {
                    col,
                    flag,
                    val: lane.val,
                });
            } else {
                debug_assert!(lane.flag.is_none(), "non-nullable column with a flag lane");
                self.inst(Inst::Store { col, val: lane.val });
            }
        }
        Ok(())
    }

    /// Emit (or reuse) join `j`'s probe in the current block. The probe is
    /// pure — same keys, same row, same result — so blocks re-probe rather
    /// than thread probe lanes through branch args. Returns the valid-hit
    /// flag (map hit AND every nullable key's validity: a NULL key never
    /// matches, and a garbage payload under a false flag must not spuriously
    /// hit) plus the value-column registers, laid out by `val_slots`.
    fn emit_probe(&mut self, j: u32, live: &mut Live) -> Result<(Value, Vec<Value>), PrepareError> {
        if let Some((valid_hit, dsts)) = self.blocks[self.cur].probes.get(&j) {
            return Ok((*valid_hit, dsts.clone()));
        }
        // Re-entering a join while emitting its OWN residual means the cache
        // entry was lost across a block transition. Unchecked, that recurses
        // until the process dies of stack overflow; the seeding above
        // prevents it, and this turns any remaining hole into a named error
        // rather than a dead serving process.
        if self.probe_seeds.iter().any(|p| p.join == j) {
            return Err(PrepareError::Internal(format!(
                "probe cache for @{j} lost inside its own residual"
            )));
        }
        if self.joins[j as usize].fold.is_some() {
            return self.emit_fold(j, live);
        }
        if self.joins[j as usize].scan.is_some() {
            return Err(PrepareError::Internal(format!(
                "@{j} holds the rows of a scan, which only its fold reads"
            )));
        }
        let spec = &self.joins[j as usize];
        let nkeys = spec.keys.len();
        for key in &spec.keys {
            let lane = self.emit(key, live)?;
            live.push((lane, key.ty));
        }
        let mut keys_valid: Option<Value> = None;
        let mut key_vals = Vec::with_capacity(nkeys);
        let start = live.len() - nkeys;
        for (i, k) in plan::map_keys(&spec.keys, &spec.key_cols).iter().enumerate() {
            let (lane, _) = live[start + i];
            // A plain key's flag is the caller's business: a NULL key never
            // matches, and here that is spelled by ANDing it into the hit.
            if let Some(f) = self.encode_probe_key(*k, lane, &mut key_vals) {
                keys_valid = self.combine_flags(keys_valid, Some(f));
            }
        }
        live.truncate(start);

        let flat_len = self.val_flat_tys(j).len();
        let hit = self.fresh();
        let mut dsts: Vec<Value> = (0..flat_len).map(|_| self.b.fresh()).collect();
        self.inst(Inst::Probe {
            static_id: j,
            hit,
            dsts: dsts.clone(),
            keys: key_vals,
        });
        let valid_hit = match keys_valid {
            None => hit,
            Some(f) => self.bin(BinOp::And, f, hit),
        };
        // Cache the RAW probe first: the residual below references this
        // join's own columns, and its StaticCol emissions must find the
        // lanes instead of recursing.
        self.blocks[self.cur]
            .probes
            .insert(j, (valid_hit, dsts.clone()));
        let residual = self.joins[j as usize].residual.clone();
        let match_v = match residual {
            None => valid_hit,
            Some(res) => {
                // Hit-guarded lazy evaluation (wave-4 pins: DuckDB
                // evaluates both-sides residuals per candidate PAIR) —
                // the mini-case shape: brif raw_hit -> eval, else false.
                // The strict block-param SSA means the probe's value lanes
                // must ride the params (and, during residual emission, the
                // live stack — nested CASE machinery rebinds them).
                let dst_tys: Vec<Ty> = self.val_flat_tys(j);
                let live_width = Self::live_types(live).len();
                let mut join_tys = Self::live_types(live);
                join_tys.extend(dst_tys.iter().copied());
                join_tys.push(Ty::I1);
                let (join, join_params) = self.create_block(&join_tys);
                let mut eval_tys = Self::live_types(live);
                eval_tys.extend(dst_tys.iter().copied());
                let (eval, eval_params) = self.create_block(&eval_tys);
                let mut then_args = Self::live_args(live);
                then_args.extend(dsts.iter().copied());
                let mut miss_args = Self::live_args(live);
                miss_args.extend(dsts.iter().copied());
                let f = self.const_i1(false);
                miss_args.push(f);
                self.term(Term::Brif {
                    cond: valid_hit,
                    then_to: BlockId(eval as u32),
                    then_args,
                    else_to: BlockId(join as u32),
                    else_args: miss_args,
                });
                self.switch(eval);
                self.enter_block(live, &eval_params[..live_width]);
                // Ride the dsts on the live stack through the residual —
                // and seed the cache with hit = TRUE (by construction in
                // the guarded branch) so StaticCol/JoinHit resolve here.
                let eval_dsts: Vec<Value> = eval_params[live_width..].to_vec();
                for (&d, &ty) in eval_dsts.iter().zip(dst_tys.iter()) {
                    live.push((Lane { flag: None, val: d }, ty));
                }
                let t = self.const_i1(true);
                self.blocks[self.cur].probes.insert(j, (t, eval_dsts));
                // Keep that cache entry alive across any CFG split INSIDE the
                // residual — a CASE arm, a COALESCE over a nullable joined
                // column, a guarded CAST.
                self.probe_seeds.push(ProbeSeed {
                    join: j,
                    base: live.len() - dst_tys.len(),
                    nd: dst_tys.len(),
                });
                let rl = self.emit(&res, live);
                self.probe_seeds.pop();
                let rl = rl?;
                // 3VL collapse: NULL residual is a non-match.
                let rv = match rl.flag {
                    None => rl.val,
                    Some(f) => self.bin(BinOp::And, f, rl.val),
                };
                let cur_dsts: Vec<Value> = live
                    .drain(live.len() - dst_tys.len()..)
                    .map(|(l, _)| l.val)
                    .collect();
                let mut hit_args = Self::live_args(live);
                hit_args.extend(cur_dsts.iter().copied());
                hit_args.push(rv);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args: hit_args,
                });
                self.switch(join);
                self.enter_block(live, &join_params[..live_width]);
                dsts = join_params[live_width..live_width + dst_tys.len()].to_vec();
                join_params[live_width + dst_tys.len()]
            }
        };
        // Downstream consumers (INNER skip, LEFT flags, StaticCol, JoinHit)
        // read the cache: store the final MATCH there.
        self.blocks[self.cur].probes.insert(j, (match_v, dsts.clone()));
        Ok((match_v, dsts))
    }

    /// The values of the fold join `j` ([`plan::Fold`]) for the current
    /// row: a loop over the rows of its row join whose keys equal the
    /// row's, which keeps each row on which the fold's predicate is TRUE
    /// and adds it to the accumulators. The accumulators have the layout of
    /// the join's value lanes ([`Self::val_slots`]): a count is one lane,
    /// and a sum, min or max is a lane that says whether a value was added
    /// and the value. Returns, as [`Self::emit_probe`] does, whether a row
    /// was kept and the value lanes, which it caches in the block after
    /// the loop.
    ///
    ///   entry:  the keys, ProbeRange -> [lo, hi) (a NULL key: empty),
    ///           jump header(lo, hi, any=false, accumulators at 0)
    ///   header(i, end, any, accs): i < end ? body : after(any, accs)
    ///   body:   ProbeRead the row, then the predicate (SELECTION context):
    ///           TRUE -> step(i + 1, end, any, accs, row), else
    ///           header(i + 1, end, any, accs)
    ///   step:   the accumulators over the row ->
    ///           header(i + 1, end, any=true, accs')
    ///   after:  the value lanes are the accumulators.
    ///
    /// The loop's state and the row ride the live stack, so a split inside
    /// the predicate carries them, and the row join's cache entry is
    /// re-created in each new block ([`ProbeSeed`]).
    fn emit_fold(&mut self, j: u32, live: &mut Live) -> Result<(Value, Vec<Value>), PrepareError> {
        let Some(fold) = self.joins[j as usize].fold.clone() else {
            return Err(PrepareError::Internal(format!("@{j} is not a fold")));
        };
        let r = fold.rows;
        let rspec = &self.joins[r as usize];
        let keys_expr = rspec.keys.clone();
        let key_layout = plan::map_keys(&keys_expr, &rspec.key_cols);
        let at = live.len();
        for key in &keys_expr {
            let lane = self.emit(key, live)?;
            live.push((lane, key.ty));
        }
        let mut keys_valid: Option<Value> = None;
        let mut key_vals = Vec::with_capacity(keys_expr.len());
        for (i, k) in key_layout.iter().enumerate() {
            let (lane, _) = live[at + i];
            if let Some(f) = self.encode_probe_key(*k, lane, &mut key_vals) {
                keys_valid = self.combine_flags(keys_valid, Some(f));
            }
        }
        live.truncate(at);
        let lo = self.fresh();
        let hi = self.fresh();
        self.inst(Inst::ProbeRange {
            static_id: r,
            start: lo,
            end: hi,
            keys: key_vals,
        });
        let (lo, hi) = match keys_valid {
            None => (lo, hi),
            Some(fl) => {
                let zero = self.const_lit(Lit::I64(0));
                (self.select_of(fl, lo, zero), self.select_of(fl, hi, zero))
            }
        };

        let ctx_tys = Self::live_types(live);
        let w = ctx_tys.len();
        // The accumulators: the value lanes of the aggregates, before the
        // item's ([`plan::Fold::post`]).
        let acc_tys: Vec<Ty> = self.map_vals(j)[..fold.aggs.len()]
            .iter()
            .flat_map(|v| v.slots())
            .collect();
        let na = acc_tys.len();
        let row_tys = self.val_flat_tys(r);
        let nr = row_tys.len();
        let state_tys = [ctx_tys.clone(), vec![Ty::I64, Ty::I64, Ty::I1], acc_tys.clone()].concat();
        let (header, hp) = self.create_block(&state_tys);
        let (body, bp) = self.create_block(&state_tys);
        let (step, sp) = self.create_block(&[state_tys.clone(), row_tys.clone()].concat());
        let (after, ap) = self.create_block(&[ctx_tys, vec![Ty::I1], acc_tys.clone()].concat());
        let f0 = self.const_i1(false);
        let mut args = Self::live_args(live);
        args.extend([lo, hi, f0]);
        for &ty in &acc_tys {
            let d = self.default_of(ty);
            args.push(d);
        }
        self.term(Term::Jump {
            to: BlockId(header as u32),
            args,
        });

        // header(i, end, any, accs)
        self.switch(header);
        self.enter_block(live, &hp[..w]);
        let more = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Lt,
            ty: Ty::I64,
            dst: more,
            a: hp[w],
            b: hp[w + 1],
        });
        let ctx = Self::live_args(live);
        self.term(Term::Brif {
            cond: more,
            then_to: BlockId(body as u32),
            then_args: [ctx.clone(), hp[w..].to_vec()].concat(),
            else_to: BlockId(after as u32),
            else_args: [ctx, hp[w + 2..].to_vec()].concat(),
        });

        // body: the row and the predicate, everything riding live as
        // [ctx..., inext, end, any, accs..., row...].
        self.switch(body);
        self.enter_block(live, &bp[..w]);
        let dsts: Vec<Value> = (0..nr).map(|_| self.fresh()).collect();
        self.inst(Inst::ProbeRead {
            static_id: r,
            idx: bp[w],
            dsts: dsts.clone(),
        });
        let one = self.const_lit(Lit::I64(1));
        let inext = self.bin(BinOp::Iadd, bp[w], one);
        live.push((Lane { flag: None, val: inext }, Ty::I64));
        live.push((Lane { flag: None, val: bp[w + 1] }, Ty::I64));
        live.push((Lane { flag: None, val: bp[w + 2] }, Ty::I1));
        for (&v, &ty) in bp[w + 3..].iter().zip(&acc_tys) {
            live.push((Lane { flag: None, val: v }, ty));
        }
        for (&d, &ty) in dsts.iter().zip(&row_tys) {
            live.push((Lane { flag: None, val: d }, ty));
        }
        let cond = match &fold.pred {
            None => None,
            Some(p) => {
                let t = self.const_i1(true);
                self.blocks[self.cur].probes.insert(r, (t, dsts));
                self.probe_seeds.push(ProbeSeed {
                    join: r,
                    base: live.len() - nr,
                    nd: nr,
                });
                let c = self.emit_truth(p, live);
                self.probe_seeds.pop();
                Some(c?)
            }
        };
        let vals = Self::live_args(live);
        match cond {
            None => self.term(Term::Jump {
                to: BlockId(step as u32),
                args: vals,
            }),
            Some(c) => self.term(Term::Brif {
                cond: c,
                then_to: BlockId(step as u32),
                then_args: vals.clone(),
                else_to: BlockId(header as u32),
                else_args: vals[..w + 3 + na].to_vec(),
            }),
        }
        live.truncate(at);

        // step: the accumulators over the kept row.
        self.switch(step);
        self.enter_block(live, &sp[..w]);
        let (accs, row) = (&sp[w + 3..w + 3 + na], &sp[w + 3 + na..]);
        let slots = self.val_slots(j);
        let row_slots = self.val_slots(r);
        let row_cols = self.map_vals(r);
        let mut next = Vec::with_capacity(na);
        for (&(func, arg), &(seen, acc)) in fold.aggs.iter().zip(&slots) {
            let arg = arg.map(|a| {
                let (validity, payload) = row_slots[a as usize];
                (validity.map(|v| row[v]), row[payload], row_cols[a as usize].ty)
            });
            let acc_v = accs[acc];
            match (func, arg) {
                (AggFunc::CountStar, _) | (AggFunc::Count, Some((None, _, _))) => {
                    let one = self.const_lit(Lit::I64(1));
                    next.push(self.bin(BinOp::Iadd, acc_v, one));
                }
                (AggFunc::Count, Some((Some(flag), _, _))) => {
                    let one = self.const_lit(Lit::I64(1));
                    let added = self.bin(BinOp::Iadd, acc_v, one);
                    next.push(self.select_of(flag, added, acc_v));
                }
                (AggFunc::Sum, Some((flag, x, ty))) => {
                    let Some(seen) = seen else {
                        return Err(PrepareError::Internal("a sum without its flag".into()));
                    };
                    let x = match ty {
                        Ty::I64 => {
                            let dst = self.fresh();
                            self.inst(Inst::Dcast {
                                from: Ty::I64,
                                to: Ty::I128,
                                dst,
                                a: x,
                            });
                            dst
                        }
                        Ty::I1 => {
                            let one = self.const_lit(Lit::I128(1));
                            let zero = self.const_lit(Lit::I128(0));
                            self.select_of(x, one, zero)
                        }
                        _ => x,
                    };
                    // A DECIMAL adds at its own scale; at most SCAN_LIMIT
                    // values of at most 36 digits stay below 10^38.
                    let added = match ty {
                        Ty::Dec(..) => {
                            let dst = self.fresh();
                            self.inst(Inst::Dop {
                                op: DecOp::Add,
                                check: 0,
                                ty: acc_tys[acc],
                                dst,
                                a: acc_v,
                                b: x,
                            });
                            dst
                        }
                        _ => self.bin(BinOp::Hadd, acc_v, x),
                    };
                    let (seen_v, added) = match flag {
                        None => (self.const_i1(true), added),
                        Some(f) => {
                            let seen_v = self.bin(BinOp::Or, accs[seen], f);
                            (seen_v, self.select_of(f, added, acc_v))
                        }
                    };
                    next.push(seen_v);
                    next.push(added);
                }
                // The first value starts the accumulator, and the operator
                // joins each later one to it. `min` and `max` of a BOOLEAN
                // are its AND and its OR.
                (
                    AggFunc::BitAnd
                    | AggFunc::BitOr
                    | AggFunc::BitXor
                    | AggFunc::BoolAnd
                    | AggFunc::BoolOr,
                    Some((flag, x, ty)),
                )
                | (AggFunc::Min | AggFunc::Max, Some((flag, x, ty @ Ty::I1))) => {
                    let Some(seen) = seen else {
                        return Err(PrepareError::Internal(format!(
                            "a fold of {func:?} without its flag"
                        )));
                    };
                    let op = match (func, ty) {
                        (AggFunc::BitAnd, Ty::I64) => BinOp::Iand,
                        (AggFunc::BitOr, Ty::I64) => BinOp::Ior,
                        (AggFunc::BitXor, Ty::I64) => BinOp::Ixor,
                        (AggFunc::BitAnd, Ty::I128) => BinOp::Hand,
                        (AggFunc::BitOr, Ty::I128) => BinOp::Hor,
                        (AggFunc::BitXor, Ty::I128) => BinOp::Hxor,
                        (AggFunc::BoolAnd | AggFunc::Min, Ty::I1) => BinOp::And,
                        (AggFunc::BoolOr | AggFunc::Max, Ty::I1) => BinOp::Or,
                        _ => {
                            return Err(PrepareError::Internal(format!(
                                "a fold of {func:?} over {ty:?}"
                            )))
                        }
                    };
                    let joined = self.bin(op, acc_v, x);
                    let joined = self.select_of(accs[seen], joined, x);
                    let (seen_v, val) = match flag {
                        None => (self.const_i1(true), joined),
                        Some(f) => (
                            self.bin(BinOp::Or, accs[seen], f),
                            self.select_of(f, joined, acc_v),
                        ),
                    };
                    next.push(seen_v);
                    next.push(val);
                }
                (AggFunc::Min | AggFunc::Max, Some((flag, x, ty))) => {
                    let Some(seen) = seen else {
                        return Err(PrepareError::Internal("an extreme without its flag".into()));
                    };
                    let better = self.fresh();
                    self.inst(Inst::Cmp {
                        pred: if func == AggFunc::Min {
                            CmpPred::Lt
                        } else {
                            CmpPred::Gt
                        },
                        ty,
                        dst: better,
                        a: x,
                        b: acc_v,
                    });
                    let first = self.not(accs[seen]);
                    let take = self.bin(BinOp::Or, first, better);
                    let (take, seen_v) = match flag {
                        None => (take, self.const_i1(true)),
                        Some(f) => (self.bin(BinOp::And, f, take), self.bin(BinOp::Or, accs[seen], f)),
                    };
                    next.push(seen_v);
                    next.push(self.select_of(take, x, acc_v));
                }
                _ => {
                    return Err(PrepareError::Internal(format!(
                        "a fold of {func:?} over {arg:?}",
                        arg = arg.map(|a| a.2)
                    )))
                }
            }
        }
        let t = self.const_i1(true);
        let mut args = Self::live_args(live);
        args.extend([sp[w], sp[w + 1], t]);
        args.extend(next);
        self.term(Term::Jump {
            to: BlockId(header as u32),
            args,
        });

        // after(any, accs): the value lanes.
        self.switch(after);
        self.enter_block(live, &ap[..w]);
        let any = ap[w];
        let accs = ap[w + 1..].to_vec();
        let Some(post) = &fold.post else {
            self.blocks[self.cur].probes.insert(j, (any, accs.clone()));
            return Ok((any, accs));
        };
        let maps = self.map_vals(j);
        if maps.len() != fold.aggs.len() + 1 || !maps[fold.aggs.len()].nullable {
            return Err(PrepareError::Internal(format!(
                "the item of the fold @{j} is not one nullable value after its aggregates"
            )));
        }
        // The item over the aggregates, which it reads through this join.
        // Over no row, a count's lane is 0 and every other aggregate's
        // validity FALSE, so the hit is TRUE: for these reads, and for the
        // item's own value, which is not NULL over no row in general
        // (`count(*) + 1`). The accumulators ride the live stack, and the
        // cache entry is re-created in each block that the item splits.
        let at = live.len();
        for (&v, &ty) in accs.iter().zip(&acc_tys) {
            live.push((Lane { flag: None, val: v }, ty));
        }
        let t = self.const_i1(true);
        self.blocks[self.cur].probes.insert(j, (t, accs));
        self.probe_seeds.push(ProbeSeed {
            join: j,
            base: at,
            nd: na,
        });
        let lane = self.emit(post, live);
        self.probe_seeds.pop();
        let lane = lane?;
        let mut outs: Vec<Value> = live.drain(at..).map(|(l, _)| l.val).collect();
        let valid = match lane.flag {
            None => self.const_i1(true),
            Some(f) => f,
        };
        outs.extend([valid, lane.val]);
        let t = self.const_i1(true);
        self.blocks[self.cur].probes.insert(j, (t, outs.clone()));
        Ok((t, outs))
    }

    /// Emit (or reuse, per block, keyed by call site) one extern call:
    /// args lower to (validity, payload) pairs — a provably non-NULL arg
    /// gets a constant-true flag; payloads under a false flag are never
    /// read by the trampoline. Returns the whole-call validity plus the
    /// flattened per-return (flag, payload) registers.
    fn emit_extern(
        &mut self,
        site: u32,
        ext: u32,
        args: &[SExpr],
        live: &mut Live,
    ) -> Result<(Value, Vec<Value>), PrepareError> {
        if let Some(hit) = self.blocks[self.cur].externs.get(&site) {
            return Ok(hit.clone());
        }
        let nargs = args.len();
        for a in args {
            let lane = self.emit(a, live)?;
            live.push((lane, a.ty));
        }
        let start = live.len() - nargs;
        let mut arg_vals = Vec::with_capacity(2 * nargs);
        for i in 0..nargs {
            let (lane, _) = live[start + i];
            let (f, v) = match lane.flag {
                None => (self.const_i1(true), lane.val),
                Some(f) => (f, lane.val),
            };
            arg_vals.push(f);
            arg_vals.push(v);
        }
        live.truncate(start);
        // A fold lowers with no extern table (`eval_closed`): a call there
        // is a lowering that fails, not an index out of bounds.
        let Some(spec) = self.udfs.get(ext as usize) else {
            return Err(PrepareError::Internal(format!("extern {ext} is not declared")));
        };
        let nrets = spec.rets.len();
        let dsts: Vec<Value> = (0..1 + 2 * nrets).map(|_| self.fresh()).collect();
        self.inst(Inst::ExternCall {
            ext,
            dsts: dsts.clone(),
            args: arg_vals,
        });
        let res = (dsts[0], dsts[1..].to_vec());
        self.blocks[self.cur].externs.insert(site, res.clone());
        Ok(res)
    }

    /// Kleene AND/OR in VALUE context, where the three-valued RESULT is what
    /// the consumer wants: both operands are evaluated, always, and the flag
    /// algebra combines them branchlessly.
    ///
    /// Nothing short-circuits here. SQL's observable laziness — `WHERE
    /// k = 0 AND <trap>` returns `[]` on DuckDB rather than failing the
    /// request — is SELECTION context and lives in [`Self::emit_truth`],
    /// which asks only whether the answer is TRUE.
    ///
    /// With the lane contract (payloads under a false flag may be garbage),
    /// every value read is guarded by its flag.
    fn kleene(
        &mut self,
        a: &SExpr,
        b: &SExpr,
        live: &mut Live,
        is_and: bool,
        res_nullable: bool,
    ) -> Result<Lane, PrepareError> {
        let la = self.emit(a, live)?;
        let _ = res_nullable;
        live.push((la, Ty::I1));
        let lb = self.emit(b, live)?;
        let (la, _) = live.pop().expect("pushed above");
        Ok(self.kleene_combine(la, lb, is_and))
    }

    /// The flag algebra itself: when a Kleene AND/OR result is KNOWN, and
    /// what it is.
    fn kleene_combine(&mut self, la: Lane, lb: Lane, is_and: bool) -> Lane {
        let op = if is_and { BinOp::And } else { BinOp::Or };
        let val = self.bin(op, la.val, lb.val);
        let flag = match (la.flag, lb.flag) {
            (None, None) => None,
            // One side definite: the result is known when the definite side
            // decides (false for AND, true for OR), or when the other side
            // is valid.
            (Some(f), None) => {
                let decides = if is_and { self.not(lb.val) } else { lb.val };
                Some(self.bin(BinOp::Or, f, decides))
            }
            (None, Some(f)) => {
                let decides = if is_and { self.not(la.val) } else { la.val };
                Some(self.bin(BinOp::Or, f, decides))
            }
            (Some(fa), Some(fb)) => {
                // Known when: both valid, or either side validly decides.
                let both = self.bin(BinOp::And, fa, fb);
                let da = if is_and { self.not(la.val) } else { la.val };
                let da = self.bin(BinOp::And, fa, da);
                let db = if is_and { self.not(lb.val) } else { lb.val };
                let db = self.bin(BinOp::And, fb, db);
                let t = self.bin(BinOp::Or, both, da);
                Some(self.bin(BinOp::Or, t, db))
            }
        };
        Lane { flag, val }
    }


    /// CASE chain: one condition block per arm, results evaluated only on
    /// their taken path, all paths joining with the result lane as params.
    fn case(
        &mut self,
        e: &SExpr,
        arms: &[(SExpr, SExpr)],
        default: Option<&SExpr>,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        if let Some((table, prefix)) = dispatch_table(arms) {
            return self.case_dispatch(e, arms, prefix, default, &table, live);
        }
        if select_case(e) {
            return self.case_select(e, arms, default, live);
        }
        let res_ty = e.ty.lane();
        let res_nullable = e.nullable;

        let mut join_tys = Self::live_types(live);
        if res_nullable {
            join_tys.push(Ty::I1);
        }
        join_tys.push(res_ty);
        let live_width = Self::live_types(live).len();
        let (join, join_params) = self.create_block(&join_tys);

        let finish_branch = |fb: &mut FB<'a>, lane: Lane, live: &Live| -> Term {
            let mut args = Self::live_args(live);
            if res_nullable {
                let flag = match lane.flag {
                    Some(f) => f,
                    None => fb.const_i1(true),
                };
                args.push(flag);
            }
            args.push(lane.val);
            Term::Jump {
                to: BlockId(join as u32),
                args,
            }
        };

        for (cond, result) in arms {
            // A CASE condition is SELECTION context wherever the CASE sits
            // (measured in a projection too); the ARMS below stay value
            // context -- a taken arm traps on DuckDB even under WHERE.
            let keep = self.emit_truth(cond, live)?;
            let shape = Self::live_types(live);
            let (then_b, then_p) = self.create_block(&shape);
            let (else_b, else_p) = self.create_block(&shape);
            let args = Self::live_args(live);
            self.term(Term::Brif {
                cond: keep,
                then_to: BlockId(then_b as u32),
                then_args: args.clone(),
                else_to: BlockId(else_b as u32),
                else_args: args,
            });

            self.switch(then_b);
            self.enter_block(live, &then_p);
            let rl = self.emit(result, live)?;
            let jump = finish_branch(self, rl, live);
            self.term(jump);

            self.switch(else_b);
            self.enter_block(live, &else_p);
        }

        let dl = match default {
            Some(d) => self.emit(d, live)?,
            None => {
                let flag = self.const_i1(false);
                let val = self.default_of(res_ty);
                Lane {
                    flag: Some(flag),
                    val,
                }
            }
        };
        let jump = finish_branch(self, dl, live);
        self.term(jump);

        self.switch(join);
        self.enter_block(live, &join_params[..live_width]);
        let tail = &join_params[live_width..];
        Ok(if res_nullable {
            Lane {
                flag: Some(tail[0]),
                val: tail[1],
            }
        } else {
            Lane {
                flag: None,
                val: tail[0],
            }
        })
    }

    /// A CASE that [`select_case`] admits, without branches: every
    /// condition, every result and the default on every row, then one
    /// select per arm, the last arm first, picks the result that the chain
    /// of arms would take.
    fn case_select(
        &mut self,
        e: &SExpr,
        arms: &[(SExpr, SExpr)],
        default: Option<&SExpr>,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        // Each part rides the live stack while the next one is emitted,
        // since a part may still split blocks (a CASE of its own).
        let base = live.len();
        for (cond, result) in arms {
            let lc = self.emit(cond, live)?;
            let keep = self.truthy(lc);
            live.push((
                Lane {
                    flag: None,
                    val: keep,
                },
                Ty::I1,
            ));
            let lr = self.emit(result, live)?;
            live.push((lr, result.ty));
        }
        let mut acc = match default {
            Some(d) => self.emit(d, live)?,
            None => {
                let flag = self.const_i1(false);
                let val = self.default_of(e.ty.lane());
                Lane {
                    flag: Some(flag),
                    val,
                }
            }
        };
        let parts: Vec<Lane> = live.drain(base..).map(|(l, _)| l).collect();
        for pair in parts.chunks(2).rev() {
            let (keep, lr) = (pair[0].val, pair[1]);
            let val = self.select_of(keep, lr.val, acc.val);
            let flag = if e.nullable {
                let taken = match lr.flag {
                    Some(f) => f,
                    None => self.const_i1(true),
                };
                let passed = match acc.flag {
                    Some(f) => f,
                    None => self.const_i1(true),
                };
                Some(self.select_of(keep, taken, passed))
            } else {
                None
            };
            acc = Lane { flag, val };
        }
        if !e.nullable {
            acc.flag = None;
        }
        Ok(acc)
    }

    /// A CASE whose first `prefix` conditions are `x = <integer constant>`
    /// over one plain value `x` (a column, slot or static column), as a
    /// binary search over the sorted constants instead of a test per arm:
    /// what a native transform's per-instance selection compiles to. The
    /// same arm is taken: those conditions are side-effect free and mutually
    /// exclusive once a repeated constant keeps its FIRST arm, and when none
    /// matches (a NULL `x` included) the remaining arms and the default run
    /// as a CASE of their own, as they would arm by arm.
    fn case_dispatch(
        &mut self,
        e: &SExpr,
        arms: &[(SExpr, SExpr)],
        prefix: usize,
        default: Option<&SExpr>,
        table: &[(i64, usize)],
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        let res_ty = e.ty.lane();
        let res_nullable = e.nullable;
        let base_tys = Self::live_types(live);
        let live_width = base_tys.len();
        let mut join_tys = base_tys;
        if res_nullable {
            join_tys.push(Ty::I1);
        }
        join_tys.push(res_ty);
        let (join, join_params) = self.create_block(&join_tys);

        let x = dispatch_scrutinee(&arms[0].0);
        let lx = self.emit(x, live)?;
        live.push((lx, x.ty));
        let shape = Self::live_types(live);
        let mut arm_blocks: Vec<Option<(usize, Vec<Value>)>> = vec![None; arms.len()];
        for &(_, i) in table {
            if arm_blocks[i].is_none() {
                arm_blocks[i] = Some(self.create_block(&shape));
            }
        }
        let (dflt, dflt_p) = self.create_block(&shape);

        // A NULL x matches no arm.
        if let Some(valid) = lx.flag {
            let (root, root_p) = self.create_block(&shape);
            let args = Self::live_args(live);
            self.term(Term::Brif {
                cond: valid,
                then_to: BlockId(root as u32),
                then_args: args.clone(),
                else_to: BlockId(dflt as u32),
                else_args: args,
            });
            self.switch(root);
            self.enter_block(live, &root_p);
        }
        let to_arm: Vec<(i64, usize)> = table
            .iter()
            .map(|&(v, i)| (v, arm_blocks[i].as_ref().expect("made above").0))
            .collect();
        self.dispatch_search(&to_arm, dflt, &shape, live);

        let finish = |fb: &mut FB<'a>, lane: Lane, live: &Live| -> Term {
            let mut args = Self::live_args(live);
            if res_nullable {
                let flag = match lane.flag {
                    Some(f) => f,
                    None => fb.const_i1(true),
                };
                args.push(flag);
            }
            args.push(lane.val);
            Term::Jump {
                to: BlockId(join as u32),
                args,
            }
        };
        for (i, blk) in arm_blocks.iter().enumerate() {
            let Some((b, params)) = blk else { continue };
            self.switch(*b);
            self.enter_block(live, params);
            let held = live.pop().expect("x is live");
            let rl = self.emit(&arms[i].1, live)?;
            let jump = finish(self, rl, live);
            self.term(jump);
            live.push(held);
        }
        self.switch(dflt);
        self.enter_block(live, &dflt_p);
        live.pop().expect("x is live");
        let dl = match default {
            _ if prefix < arms.len() => self.case(e, &arms[prefix..], default, live)?,
            Some(d) => self.emit(d, live)?,
            None => {
                let flag = self.const_i1(false);
                let val = self.default_of(res_ty);
                Lane {
                    flag: Some(flag),
                    val,
                }
            }
        };
        let jump = finish(self, dl, live);
        self.term(jump);

        self.switch(join);
        self.enter_block(live, &join_params[..live_width]);
        let tail = &join_params[live_width..];
        Ok(if res_nullable {
            Lane {
                flag: Some(tail[0]),
                val: tail[1],
            }
        } else {
            Lane {
                flag: None,
                val: tail[0],
            }
        })
    }

    /// Branch to the arm block whose constant equals `x` (the top of
    /// `live`), else to `dflt`: halve on `x < c` until a few constants are
    /// left, then test those in turn. Terminates the current block.
    fn dispatch_search(&mut self, table: &[(i64, usize)], dflt: usize, shape: &[Ty], live: &mut Live) {
        if table.len() <= 4 {
            for (j, &(v, arm)) in table.iter().enumerate() {
                // Re-read per block: entering one rebinds the live values.
                let xv = live.last().expect("x is live").0.val;
                let c = self.const_lit(Lit::I64(v));
                let eq = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Eq,
                    ty: Ty::I64,
                    dst: eq,
                    a: xv,
                    b: c,
                });
                let args = Self::live_args(live);
                if j + 1 == table.len() {
                    self.term(Term::Brif {
                        cond: eq,
                        then_to: BlockId(arm as u32),
                        then_args: args.clone(),
                        else_to: BlockId(dflt as u32),
                        else_args: args,
                    });
                    return;
                }
                let (next, next_p) = self.create_block(shape);
                self.term(Term::Brif {
                    cond: eq,
                    then_to: BlockId(arm as u32),
                    then_args: args.clone(),
                    else_to: BlockId(next as u32),
                    else_args: args,
                });
                self.switch(next);
                self.enter_block(live, &next_p);
            }
            return;
        }
        let xv = live.last().expect("x is live").0.val;
        let mid = table.len() / 2;
        let c = self.const_lit(Lit::I64(table[mid].0));
        let lt = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Lt,
            ty: Ty::I64,
            dst: lt,
            a: xv,
            b: c,
        });
        let (lo, lo_p) = self.create_block(shape);
        let (hi, hi_p) = self.create_block(shape);
        let args = Self::live_args(live);
        self.term(Term::Brif {
            cond: lt,
            then_to: BlockId(lo as u32),
            then_args: args.clone(),
            else_to: BlockId(hi as u32),
            else_args: args,
        });
        self.switch(lo);
        self.enter_block(live, &lo_p);
        self.dispatch_search(&table[..mid], dflt, shape, live);
        self.switch(hi);
        self.enter_block(live, &hi_p);
        self.dispatch_search(&table[mid..], dflt, shape, live);
    }

    /// CAST/TRY_CAST lowering. NULL input never traps; CAST failure traps
    /// with a conversion message; TRY_CAST failure becomes NULL.
    fn cast(
        &mut self,
        e: &SExpr,
        inner: &SExpr,
        trying: bool,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        // Casts convert LANES, so a width-only int cast is the (a == b)
        // no-op below: its range check is the result trap `emit` applies
        // over this call, not anything emitted here.
        let from = inner.ty.lane();
        let to = e.ty.lane();
        let l = self.emit(inner, live)?;

        // The i128 lane, and the parse into an unsigned width, which keeps
        // DuckDB's sign rule (`ston.opt`).
        if from == Ty::I128 || to == Ty::I128 || (from == Ty::Str && e.ty.is_unsigned()) {
            return self.wide_cast(e, inner, l, trying, live);
        }

        // Branchless conversions first.
        let simple: Option<Lane> = match (from, to) {
            (a, b) if a == b => Some(l),
            (Ty::I64, Ty::F64) => {
                let dst = self.fresh();
                self.inst(Inst::Itof {
                    narrow: false,
                    dst,
                    a: l.val,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::F64, Ty::I64) if !trying => {
                // ftoi.nearest is half-to-even, which is DuckDB's cast
                // rounding (the round() BUILTIN is the other one,
                // half-away-from-zero, and lowers elsewhere). Its own range
                // trap stands in for DuckDB's conversion error. Nullable
                // payloads are masked: computed garbage under a false flag
                // (x * 1e300 * 1e300 with x NULL) must not fire the trap.
                let a = self.masked(l, Ty::F64);
                let dst = self.fresh();
                self.inst(Inst::Ftoi {
                    mode: super::ir::RoundMode::Nearest,
                    dst,
                    a,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I64, Ty::Str) => {
                let dst = self.fresh();
                self.inst(Inst::Itos { dst, a: l.val });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::F64, Ty::Str) => {
                let dst = self.fresh();
                self.inst(Inst::Ftos { dst, a: l.val });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I1, Ty::Str) => {
                let t = self.const_lit(Lit::Str("true".to_string()));
                let f = self.const_lit(Lit::Str("false".to_string()));
                let dst = self.fresh();
                self.inst(Inst::Select {
                    dst,
                    cond: l.val,
                    a: t,
                    b: f,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I1, Ty::I64) => {
                let one = self.const_lit(Lit::I64(1));
                let zero = self.const_lit(Lit::I64(0));
                let dst = self.fresh();
                self.inst(Inst::Select {
                    dst,
                    cond: l.val,
                    a: one,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I1, Ty::F64) => {
                let one = self.const_lit(Lit::F64(1.0));
                let zero = self.const_lit(Lit::F64(0.0));
                let dst = self.fresh();
                self.inst(Inst::Select {
                    dst,
                    cond: l.val,
                    a: one,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::I64, Ty::I1) => {
                let zero = self.const_lit(Lit::I64(0));
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ne,
                    ty: Ty::I64,
                    dst,
                    a: l.val,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            (Ty::F64, Ty::I1) => {
                // DuckDB: nonzero -> true (measured 2.5::BOOLEAN = true).
                let zero = self.const_lit(Lit::F64(0.0));
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ne,
                    ty: Ty::F64,
                    dst,
                    a: l.val,
                    b: zero,
                });
                Some(Lane {
                    flag: l.flag,
                    val: dst,
                })
            }
            _ => None,
        };
        if let Some(lane) = simple {
            // TRY_CAST of an infallible conversion: the declared nullability
            // is `true`, so surface a flag even though nothing can fail.
            if trying && lane.flag.is_none() {
                let t = self.const_i1(true);
                return Ok(Lane {
                    flag: Some(t),
                    val: lane.val,
                });
            }
            return Ok(lane);
        }

        match (from, to) {
            // String parses: shared shape, differing parse op.
            (Ty::Str, Ty::I64) | (Ty::Str, Ty::F64) | (Ty::Str, Ty::I1) => {
                let ok = self.fresh();
                let parsed = self.fresh();
                if to == Ty::I1 {
                    // DuckDB's TryCastToBoolean, measured: exactly true/t/1/
                    // yes/y and false/f/0/no/n, ASCII case-insensitive, no
                    // trimming. Unicode lowering is equivalent here: no
                    // non-ASCII codepoint lowers to a single ASCII letter of
                    // these words.
                    let lo = self.fresh();
                    self.inst(Inst::Str1 {
                        op: StrOp1::Lower,
                        dst: lo,
                        a: l.val,
                    });
                    let any = |words: &[&str], this: &mut Self| {
                        let mut acc: Option<Value> = None;
                        for w in words {
                            let c = this.const_lit(Lit::Str((*w).to_string()));
                            let eq = this.fresh();
                            this.inst(Inst::Cmp {
                                pred: CmpPred::Eq,
                                ty: Ty::Str,
                                dst: eq,
                                a: lo,
                                b: c,
                            });
                            acc = Some(match acc {
                                Some(x) => this.bin(BinOp::Or, x, eq),
                                None => eq,
                            });
                        }
                        acc.expect("non-empty word list")
                    };
                    let is_t = any(&["true", "t", "1", "yes", "y"], self);
                    let is_f = any(&["false", "f", "0", "no", "n"], self);
                    let either = self.bin(BinOp::Or, is_t, is_f);
                    self.inst(Inst::Bin {
                        op: BinOp::And,
                        dst: ok,
                        a: either,
                        b: either,
                    });
                    self.inst(Inst::Bin {
                        op: BinOp::And,
                        dst: parsed,
                        a: is_t,
                        b: is_t,
                    });
                } else if to == Ty::I64 {
                    self.inst(Inst::StoiOpt {
                        flag: ok,
                        dst: parsed,
                        a: l.val,
                    });
                } else {
                    self.inst(Inst::StofOpt {
                        flag: ok,
                        dst: parsed,
                        a: l.val,
                    });
                }
                // The string parser rounds and the WIDTH check applies to
                // the ROUNDED value, in the cast itself -- DuckDB says
                // "Could not convert string" for '300'::TINYINT, not an
                // out-of-range error, and TRY_CAST is NULL, not a trap. The
                // in-arm check folds into `ok`, so both spellings follow.
                let ok = match (to, e.ty.int_range()) {
                    (Ty::I64, Some((lo, hi))) => {
                        let lo_v = self.const_lit(Lit::I64(lo));
                        let hi_v = self.const_lit(Lit::I64(hi));
                        let ge = self.fresh();
                        self.inst(Inst::Cmp {
                            pred: CmpPred::Ge,
                            ty: Ty::I64,
                            dst: ge,
                            a: parsed,
                            b: lo_v,
                        });
                        let le = self.fresh();
                        self.inst(Inst::Cmp {
                            pred: CmpPred::Le,
                            ty: Ty::I64,
                            dst: le,
                            a: parsed,
                            b: hi_v,
                        });
                        let in_range = self.bin(BinOp::And, ge, le);
                        self.bin(BinOp::And, ok, in_range)
                    }
                    _ => ok,
                };
                if trying {
                    let flag = match l.flag {
                        Some(f) => self.bin(BinOp::And, f, ok),
                        None => ok,
                    };
                    return Ok(Lane {
                        flag: Some(flag),
                        val: parsed,
                    });
                }
                // CAST: trap iff the input is a real (non-NULL) string that
                // does not parse.
                let not_ok = self.not(ok);
                let bad = match l.flag {
                    Some(f) => self.bin(BinOp::And, f, not_ok),
                    None => not_ok,
                };
                let (trap_b, _) = self.create_block(&[]);
                let mut cont_tys = Self::live_types(live);
                let flag_in_shape = l.flag.is_some();
                if flag_in_shape {
                    cont_tys.push(Ty::I1);
                }
                cont_tys.push(to);
                let live_width = Self::live_types(live).len();
                let (cont_b, cont_p) = self.create_block(&cont_tys);
                let mut args = Self::live_args(live);
                if let Some(f) = l.flag {
                    args.push(f);
                }
                args.push(parsed);
                self.term(Term::Brif {
                    cond: bad,
                    then_to: BlockId(trap_b as u32),
                    then_args: vec![],
                    else_to: BlockId(cont_b as u32),
                    else_args: args,
                });
                self.switch(trap_b);
                self.term(Term::Trap {
                    msg: if to == Ty::I64 {
                        // DuckDB spells INT8/16/32/64 in THIS message
                        // (measured), unlike its TINYINT-style DDL names.
                        format!(
                            "Conversion Error: Could not convert string to {}",
                            match e.ty {
                                Ty::I8 => "INT8",
                                Ty::I16 => "INT16",
                                Ty::I32 => "INT32",
                                Ty::U8 => "UINT8",
                                Ty::U16 => "UINT16",
                                Ty::U32 => "UINT32",
                                _ => "INT64",
                            }
                        )
                    } else if to == Ty::I1 {
                        "Conversion Error: Could not convert string to BOOL".to_string()
                    } else {
                        "Conversion Error: could not cast VARCHAR to DOUBLE".to_string()
                    },
                });
                self.switch(cont_b);
                self.enter_block(live, &cont_p[..live_width]);
                let tail = &cont_p[live_width..];
                Ok(if flag_in_shape {
                    Lane {
                        flag: Some(tail[0]),
                        val: tail[1],
                    }
                } else {
                    Lane {
                        flag: None,
                        val: tail[0],
                    }
                })
            }
            // TRY_CAST(f64 -> i64): guard the range so ftoi cannot trap; the
            // payload rides the branch on the live stack.
            (Ty::F64, Ty::I64) => {
                debug_assert!(trying);
                // Exactly ±2^63; NaN fails both compares -> NULL. A NULL
                // input's default payload passes the range check but its
                // false flag forces the null path anyway.
                let min = self.const_lit(Lit::F64(-9223372036854775808.0));
                let max = self.const_lit(Lit::F64(9223372036854775808.0));
                let ge = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ge,
                    ty: Ty::F64,
                    dst: ge,
                    a: l.val,
                    b: min,
                });
                let lt = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Lt,
                    ty: Ty::F64,
                    dst: lt,
                    a: l.val,
                    b: max,
                });
                let in_range = self.bin(BinOp::And, ge, lt);
                let ok = match l.flag {
                    Some(f) => self.bin(BinOp::And, f, in_range),
                    None => in_range,
                };

                live.push((l, Ty::F64));
                let live_width = Self::live_types(live).len();
                let mut join_tys = Self::live_types(live);
                join_tys.push(Ty::I1);
                join_tys.push(Ty::I64);
                let (join, join_p) = self.create_block(&join_tys);
                let shape = Self::live_types(live);
                let (conv_b, conv_p) = self.create_block(&shape);
                let (null_b, null_p) = self.create_block(&shape);
                let args = Self::live_args(live);
                self.term(Term::Brif {
                    cond: ok,
                    then_to: BlockId(conv_b as u32),
                    then_args: args.clone(),
                    else_to: BlockId(null_b as u32),
                    else_args: args,
                });

                self.switch(conv_b);
                self.enter_block(live, &conv_p);
                let payload = live.last().expect("pushed above").0.val;
                let r = self.fresh();
                self.inst(Inst::Ftoi {
                    mode: super::ir::RoundMode::Nearest,
                    dst: r,
                    a: payload,
                });
                let t = self.const_i1(true);
                let mut args = Self::live_args(live);
                args.push(t);
                args.push(r);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args,
                });

                self.switch(null_b);
                self.enter_block(live, &null_p);
                let f = self.const_i1(false);
                let d = self.const_lit(Lit::I64(0));
                let mut args = Self::live_args(live);
                args.push(f);
                args.push(d);
                self.term(Term::Jump {
                    to: BlockId(join as u32),
                    args,
                });

                self.switch(join);
                self.enter_block(live, &join_p[..live_width]);
                live.pop();
                let tail = &join_p[live_width..];
                Ok(Lane {
                    flag: Some(tail[0]),
                    val: tail[1],
                })
            }
            (from, to) => Err(PrepareError::Internal(format!(
                "cast {} -> {} escaped the frontend",
                from.name(),
                to.name()
            ))),
        }
    }

    /// A cast with the i128 lane on either side (or a parse into an
    /// unsigned width): integer <-> integer through the checked `dcast`,
    /// DOUBLE in through a range check and `ftoh`, DOUBLE / VARCHAR /
    /// BOOLEAN out, VARCHAR in through `ston.opt`. CAST traps where DuckDB
    /// errors; TRY_CAST answers NULL there instead.
    fn wide_cast(
        &mut self,
        e: &SExpr,
        inner: &SExpr,
        l: Lane,
        trying: bool,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        let (from, to) = (inner.ty, e.ty);
        let fresh_lane = |this: &mut Self, flag: Option<Value>, val: Value| -> Lane {
            match (trying, flag) {
                (true, None) => Lane {
                    flag: Some(this.const_i1(true)),
                    val,
                },
                _ => Lane { flag, val },
            }
        };
        // `ok` is the conversion's success on this row's (masked) payload;
        // CAST traps on a real row that fails, TRY_CAST flags it NULL.
        let finish = |this: &mut Self, ok: Value, val: Value, msg: String, live: &mut Live| {
            if trying {
                let flag = match l.flag {
                    Some(f) => this.bin(BinOp::And, f, ok),
                    None => ok,
                };
                return Lane {
                    flag: Some(flag),
                    val,
                };
            }
            let not_ok = this.not(ok);
            let bad = match l.flag {
                Some(f) => this.bin(BinOp::And, f, not_ok),
                None => not_ok,
            };
            this.trap_if(bad, msg, Lane { flag: l.flag, val }, to, live)
        };
        let phys = |t: Ty| match t {
            Ty::U8 => "UINT8",
            Ty::U16 => "UINT16",
            Ty::U32 => "UINT32",
            Ty::U64 => "UINT64",
            Ty::I128 => "INT128",
            Ty::I8 => "INT8",
            Ty::I16 => "INT16",
            Ty::I32 => "INT32",
            _ => "INT64",
        };
        match (from.lane(), to.lane()) {
            (f, t) if from.is_integer() && to.is_integer() => {
                let (flo, fhi) = from.int_range128().expect("an integer width");
                let (tlo, thi) = to.int_range128().expect("an integer width");
                let total = tlo <= flo && fhi <= thi;
                if total && f == t {
                    return Ok(fresh_lane(self, l.flag, l.val));
                }
                // A NULL row converts 0, which every width holds.
                let a = self.masked(l, from);
                let (from_lane, dst) = (f, self.fresh());
                if total {
                    self.inst(Inst::Dcast {
                        from: from_lane,
                        to,
                        dst,
                        a,
                    });
                    return Ok(fresh_lane(self, l.flag, dst));
                }
                let ok = self.fresh();
                self.inst(Inst::DcastOk {
                    from: from_lane,
                    to,
                    dst: ok,
                    a,
                });
                // A failing row converts 0 instead: its flag says NULL or
                // the trap below fires first.
                let zero = self.default_of(from);
                let safe = self.select_of(ok, a, zero);
                self.inst(Inst::Dcast {
                    from: from_lane,
                    to,
                    dst,
                    a: safe,
                });
                let msg = format!(
                    "Conversion Error: Type {} can't be cast because the value is out of \
                     range for the destination type {}",
                    phys(from),
                    phys(to)
                );
                Ok(finish(self, ok, dst, msg, live))
            }
            (Ty::I1, Ty::I128) => {
                let one = self.const_lit(Lit::I128(1));
                let zero = self.const_lit(Lit::I128(0));
                let dst = self.select_of(l.val, one, zero);
                Ok(fresh_lane(self, l.flag, dst))
            }
            (Ty::I128, Ty::I1) => {
                let zero = self.const_lit(Lit::I128(0));
                let dst = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Ne,
                    ty: Ty::I128,
                    dst,
                    a: l.val,
                    b: zero,
                });
                Ok(fresh_lane(self, l.flag, dst))
            }
            (Ty::I128, Ty::F64) => {
                let dst = self.fresh();
                self.inst(Inst::Htof { dst, a: l.val });
                Ok(fresh_lane(self, l.flag, dst))
            }
            (Ty::I128, Ty::Str) => {
                let dst = self.fresh();
                self.inst(Inst::Htos { dst, a: l.val });
                Ok(fresh_lane(self, l.flag, dst))
            }
            (Ty::F64, Ty::I128) => {
                // `TryCastWithOverflowCheckFloat` / `ConvertFloatingToBigint`:
                // the range is checked on the UNROUNDED value, NaN out.
                // UBIGINT's bound is the double 18446744073709551615.0,
                // which is 2^64.
                let (lo, lo_pred, hi) = if to == Ty::U64 {
                    (0.0, CmpPred::Ge, 18446744073709551616.0)
                } else {
                    (-170141183460469231731687303715884105728.0, CmpPred::Gt, 170141183460469231731687303715884105728.0)
                };
                let lo_v = self.const_lit(Lit::F64(lo));
                let hi_v = self.const_lit(Lit::F64(hi));
                let ge = self.fresh();
                self.inst(Inst::Cmp {
                    pred: lo_pred,
                    ty: Ty::F64,
                    dst: ge,
                    a: l.val,
                    b: lo_v,
                });
                let lt = self.fresh();
                self.inst(Inst::Cmp {
                    pred: CmpPred::Lt,
                    ty: Ty::F64,
                    dst: lt,
                    a: l.val,
                    b: hi_v,
                });
                let ok = self.bin(BinOp::And, ge, lt);
                let zero = self.const_lit(Lit::F64(0.0));
                let safe = self.select_of(ok, l.val, zero);
                let dst = self.fresh();
                self.inst(Inst::Ftoh { dst, a: safe });
                let msg = format!(
                    "Conversion Error: Type DOUBLE can't be cast because the value is out of \
                     range for the destination type {}",
                    phys(to)
                );
                Ok(finish(self, ok, dst, msg, live))
            }
            (Ty::Str, _) => {
                let (ok, dst) = (self.fresh(), self.fresh());
                self.inst(Inst::StonOpt {
                    to,
                    flag: ok,
                    dst,
                    a: l.val,
                });
                let msg = format!("Conversion Error: Could not convert string to {}", phys(to));
                Ok(finish(self, ok, dst, msg, live))
            }
            (f, t) => Err(PrepareError::Internal(format!(
                "cast {} -> {} escaped the frontend",
                f.name(),
                t.name()
            ))),
        }
    }

    /// NULL propagation: the result is valid iff every nullable input is.
    fn combine_flags(&mut self, a: Option<Value>, b: Option<Value>) -> Option<Value> {
        match (a, b) {
            (None, None) => None,
            (Some(f), None) | (None, Some(f)) => Some(f),
            (Some(fa), Some(fb)) => Some(self.bin(BinOp::And, fa, fb)),
        }
    }
}


/// Arms below this many keep the test-per-arm lowering.
const DISPATCH_MIN_ARMS: usize = 8;

/// The value an equality condition tests and its constant, when the
/// condition is `x = c` (either side) with `x` a plain integer value.
fn dispatch_key(cond: &SExpr) -> Option<(&SExpr, i64)> {
    let SKind::Cmp {
        pred: CmpPred::Eq,
        a,
        b,
    } = &cond.kind
    else {
        return None;
    };
    let (x, c) = match (&a.kind, &b.kind) {
        (_, SKind::Lit(Lit::I64(c))) => (a.as_ref(), *c),
        (SKind::Lit(Lit::I64(c)), _) => (b.as_ref(), *c),
        _ => return None,
    };
    let plain = matches!(
        x.kind,
        SKind::Col(_) | SKind::Slot(_) | SKind::StaticCol { .. }
    );
    (plain && x.ty.is_int()).then_some((x, c))
}

fn same_plain(a: &SExpr, b: &SExpr) -> bool {
    a.ty == b.ty
        && match (&a.kind, &b.kind) {
            (SKind::Col(i), SKind::Col(j)) | (SKind::Slot(i), SKind::Slot(j)) => i == j,
            (
                SKind::StaticCol { join: j1, col: c1 },
                SKind::StaticCol { join: j2, col: c2 },
            ) => j1 == j2 && c1 == c2,
            _ => false,
        }
}

fn dispatch_scrutinee(cond: &SExpr) -> &SExpr {
    dispatch_key(cond).expect("checked by dispatch_table").0
}

/// `(constant, arm)` sorted by constant, each constant keeping its first
/// arm, over the longest run of leading conditions that test one plain
/// value against an integer constant, and that run's length; `None` when
/// the run is too short to pay.
fn dispatch_table(arms: &[(SExpr, SExpr)]) -> Option<(Vec<(i64, usize)>, usize)> {
    let (x0, _) = dispatch_key(&arms.first()?.0)?;
    let mut table: Vec<(i64, usize)> = Vec::with_capacity(arms.len());
    let mut seen = std::collections::HashSet::new();
    let mut prefix = 0;
    for (i, (cond, _)) in arms.iter().enumerate() {
        let Some((x, c)) = dispatch_key(cond) else { break };
        if !same_plain(x, x0) {
            break;
        }
        if seen.insert(c) {
            table.push((c, i));
        }
        prefix = i + 1;
    }
    if prefix < DISPATCH_MIN_ARMS {
        return None;
    }
    table.sort_unstable();
    Some((table, prefix))
}

/// The most operations a CASE lowered to selects holds ([`select_case`]),
/// itself included. Reading a column, a constant or a computed value is not
/// an operation.
const SELECT_CASE_OPS: usize = 8;

/// Whether a CASE lowers to selects ([`FB::case_select`]) instead of a
/// chain of blocks: no condition, result or default can trap
/// ([`plan::can_trap`], which also keeps out integer arithmetic and
/// everything impure), and either it holds at most [`SELECT_CASE_OPS`]
/// operations, each a comparison, AND, OR, NOT, IS NULL, arithmetic, abs,
/// an integer read as DOUBLE or a CASE, or it is a NULL guard
/// ([`null_guard`]). Each part is then computed on every row, which no row
/// can see. A chain splits three blocks for one arm, and each carries the
/// whole live stack: with a `coalesce(x, 'NaN')` at each of 4,096 reads of
/// 256 features, as the native catalog reads them, 64 rows served in 15 ms,
/// against 2 ms for plain reads (2026-10-14).
fn select_case(e: &SExpr) -> bool {
    let SKind::Case { arms, default } = &e.kind else {
        return false;
    };
    let mut budget = SELECT_CASE_OPS;
    (few_ops(e, &mut budget) || null_guard(arms, default.as_deref()))
        && arms
            .iter()
            .all(|(c, r)| !plan::can_trap(c) && !plan::can_trap(r))
        && default.as_deref().is_none_or(|d| !plan::can_trap(d))
}

/// Whether a CASE is `CASE WHEN c THEN NULL ELSE v END` with a `c` of at
/// most [`SELECT_CASE_OPS`] operations: each field read of a SQL function
/// with a `null_when` has this shape. As selects, `v` is computed on every
/// row whatever its size, which costs only the rows where `c` holds. As a
/// chain, each field read is a branch, and each block it starts loads again
/// every column that it reads: 64 field reads of 256 features served 64
/// rows in 28 ms, against 4 ms without `null_when` (2026-10-14).
fn null_guard(arms: &[(SExpr, SExpr)], default: Option<&SExpr>) -> bool {
    let [(c, r)] = arms else {
        return false;
    };
    let mut budget = SELECT_CASE_OPS;
    matches!(r.kind, SKind::NullOf) && default.is_some() && few_ops(c, &mut budget)
}

/// Whether `e` holds at most `budget` operations, each of a kind that
/// [`select_case`] admits, counting them off `budget`.
fn few_ops(e: &SExpr, budget: &mut usize) -> bool {
    let leaf = matches!(
        e.kind,
        SKind::Col(_)
            | SKind::Slot(_)
            | SKind::StaticCol { .. }
            | SKind::JoinHit(_)
            | SKind::Shared(_)
            | SKind::Lit(_)
            | SKind::NullOf
    );
    if leaf {
        return true;
    }
    let Some(left) = budget.checked_sub(1) else {
        return false;
    };
    *budget = left;
    match &e.kind {
        SKind::Cmp { a, b, .. }
        | SKind::And { a, b }
        | SKind::Or { a, b }
        | SKind::Arith { a, b, .. } => few_ops(a, budget) && few_ops(b, budget),
        SKind::Not(a) | SKind::IsNull { inner: a, .. } | SKind::IntToFloat(a) | SKind::Abs(a) => {
            few_ops(a, budget)
        }
        SKind::Case { arms, default } => {
            arms.iter()
                .all(|(c, r)| few_ops(c, budget) && few_ops(r, budget))
                && default.as_deref().is_none_or(|d| few_ops(d, budget))
        }
        _ => false,
    }
}
