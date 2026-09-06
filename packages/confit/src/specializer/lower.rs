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
//! trap eagerly — SQL semantics, verified by test). CAST failure is a
//! conditional trap block; TRY_CAST folds the failure into the null lane.
//!
//! ponytail: blocks re-load columns instead of receiving them as branch args
//! — pure loads, identical semantics, simpler lowering. Thread them when the
//! Cranelift backend makes the extra reads measurable.

use std::collections::HashMap;

use super::frontend::PrepareError;
use super::ir::{
    BinOp, Block, BlockId, Builder, CmpPred, Col, Inst, Lit, NumOp1, Program, StaticTy, StrOp1,
    StrOp2, Term, Ty, Value,
};
use super::plan::{self, ArithOp, JoinKind, JoinSpec, KeyCmp, Rel, SExpr, SKind, StaticTable};


/// Can this kind's NARROW result sit outside its width's range?
///
/// The exempt list is an allowlist, and each entry earns it: `Col` and
/// `StaticCol` are range-checked on the way IN (the ingest boundary mirrors
/// `narrow_check`); a `Lit` outside its type's range refuses at build;
/// `NullOf` is a typed default; `JoinHit` is i1; and `Case` only forwards a
/// value one of its arms already produced and checked. Everything else —
/// arithmetic, casts, abs, extern returns, anything added later — is checked.
fn narrow_result_can_escape(k: &SKind) -> bool {
    !matches!(
        k,
        SKind::Col(_)
            | SKind::StaticCol { .. }
            | SKind::Lit(_)
            | SKind::NullOf
            | SKind::JoinHit(_)
            | SKind::Case { .. }
    )
}

/// DuckDB's spelling of a narrow width, for the trap message.
fn duck_narrow_name(ty: Ty) -> &'static str {
    match ty {
        Ty::I8 => "TINYINT",
        Ty::I16 => "SMALLINT",
        Ty::I32 => "INTEGER",
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
        _ => "int64",
    }
}

/// The bound plan as one imperative-IR program named `name`: the projection
/// `rel` carries, its joins, and the statics they probe.
///
/// `in_cols` is the row input in LANE order (struct leaves already
/// flattened) and `out_cols` its declared output, both positional — the
/// projection's i-th expression stores to `out_cols[i]`. `joins` indexes
/// `catalog`, and join `j` probes static `@j`, so the `model<...>` statics
/// for `model_refs` (indices into `models`) are appended after them all.
/// `many` selects the multiplicity loop of `shape="many"`, which stage B
/// limits to a single join.
///
/// The result is well-formed but NOT yet canonical or verified; the caller
/// does both. What the loop form cannot serve refuses here as
/// [`PrepareError::Unsupported`]; a shape the frontend should have rejected
/// surfaces as [`PrepareError::Internal`], never as a wrong program.
#[allow(clippy::too_many_arguments)]
pub fn lower(
    rel: &Rel,
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
    let (exprs, filter_pred) = match rel {
        Rel::Project { input, exprs } => match input.as_ref() {
            Rel::Filter { input: scan, pred } => {
                debug_assert!(matches!(scan.as_ref(), Rel::Scan));
                (exprs, Some(pred))
            }
            Rel::Scan => (exprs, None),
            Rel::Project { .. } => {
                return Err(PrepareError::Internal("nested projection".to_string()))
            }
        },
        _ => {
            return Err(PrepareError::Internal(
                "plan root is not a projection".to_string(),
            ))
        }
    };

    // The many path emits exactly one join static (it is guarded to a single
    // join); the normal path emits one per join.
    let model_base = if many && joins.len() == 1 { 1 } else { joins.len() };
    let mut fb = FB::new(in_cols, joins, catalog, udfs, model_base);

    // shape='many' (stage B): joins lower as multiplicity LOOPS over
    // multimap row ranges — 0..N output rows per input row. One join per
    // query for now; a map's key-uniqueness is unknown at prepare, so
    // under 'many' every join takes the loop form.
    if many && joins.len() > 1 {
        return Err(PrepareError::Unsupported(
            "multiple joins under shape='many' (one join per query in stage B)".to_string(),
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
    if many && joins.len() == 1 {
        fb.lower_many_loop(exprs, filter_pred, &out_cols)?;
        let statics = vec![if joins[0].batch {
            StaticTy::BatchMap {
                values: plan::map_vals(in_cols, &joins[0].val_cols)
                    .iter()
                    .flat_map(|v| v.slots())
                    .collect(),
            }
        } else {
            StaticTy::MultiMap {
                // The INDF refusal above means every key reaching here
                // compares `Eq`, so `slots()` returns exactly
                // `[ty.lane()]` today; lift that refusal and this line
                // starts carrying pairs unedited.
                keys: plan::map_keys(&joins[0].keys, &joins[0].key_cols)
                    .iter()
                    .flat_map(|k| k.slots())
                    .collect(),
                values: plan::map_vals(&catalog[joins[0].table].cols, &joins[0].val_cols)
                    .iter()
                    .flat_map(|v| v.slots())
                    .collect(),
            }
        }];
        let statics = [statics, model_statics].concat();
        return fb.finish(name, statics, in_cols, out_cols, regexes, udfs.to_vec());
    }

    // Joins run before WHERE (SQL order), each in FROM order: probe, and for
    // INNER skip the row on a miss. A LEFT join's probe is also forced here
    // so its key expressions are evaluated (and can trap) for every row that
    // reaches it, exactly as the join would — even if no value column is
    // ever referenced. Later references re-probe per block (pure, cached),
    // same ponytail trade as column re-loads.
    for (j, spec) in joins.iter().enumerate() {
        let mut live = Vec::new();
        let (valid_hit, _) = fb.emit_probe(j as u32, &mut live)?;
        if spec.kind == JoinKind::Inner {
            let (keep, _) = fb.create_block(&[]);
            let (miss, _) = fb.create_block(&[]);
            fb.term(Term::Brif {
                cond: valid_hit,
                then_to: BlockId(keep as u32),
                then_args: vec![],
                else_to: BlockId(miss as u32),
                else_args: vec![],
            });
            fb.switch(miss);
            fb.term(Term::Skip);
            fb.switch(keep);
        }
    }

    if let Some(pred) = filter_pred {
        // The WHERE root is SELECTION context: emit_truth is where AND's
        // left-to-right laziness lives, recursively, at every depth rather
        // than only along a top-level conjunct spine. A filter asks
        // `pred IS TRUE`, and FALSE and NULL answer that question alike, so
        // the bare i1 is the whole story.
        let mut live = Vec::new();
        let cond = fb.emit_truth(pred, &mut live)?;
        let (keep, _) = fb.create_block(&[]);
        let (drop, _) = fb.create_block(&[]);
        fb.term(Term::Brif {
            cond,
            then_to: BlockId(keep as u32),
            then_args: vec![],
            else_to: BlockId(drop as u32),
            else_args: vec![],
        });
        fb.switch(drop);
        fb.term(Term::Skip);
        fb.switch(keep);
    }

    let mut live = Vec::new();
    for (ci, (_, e)) in exprs.iter().enumerate() {
        // Not a `debug_assert`: this runs once per output column at prepare,
        // never per row, and it is the ONLY thing that catches an arm which
        // pushes lanes and forgets to truncate. A leak leaves dead values
        // riding every later block transition — well-formed IR, so `verify`
        // says nothing, and a release-only test run sees nothing either
        // (measured: the whole suite passed green with a `live.truncate`
        // deleted from the arm that had just been fixed to add it).
        assert!(live.is_empty(), "live stack leaked before column {ci}");
        let lane = fb.emit(e, &mut live)?;
        let col = ci as u32;
        if out_cols[ci].ty.nullable {
            let flag = match lane.flag {
                Some(f) => f,
                // Nullability contract slack (e.g. an infallible TRY_CAST):
                // the column is declared nullable, the lane is provably
                // valid — store with a constant true flag.
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
    }
    fb.term(Term::Emit);

    // Map static @j belongs to join j: keyed by the (already promoted) key
    // expression types, valued by the join's value columns.
    let statics = joins
        .iter()
        .map(|spec| StaticTy::Map {
            keys: plan::map_keys(&spec.keys, &spec.key_cols)
                .iter()
                .flat_map(|k| k.slots())
                .collect(),
            values: plan::map_vals(&catalog[spec.table].cols, &spec.val_cols)
                .iter()
                .flat_map(|v| v.slots())
                .collect(),
        })
        .collect();
    let statics = [statics, model_statics].concat();

    fb.finish(name, statics, in_cols, out_cols, regexes, udfs.to_vec())
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
    /// The many-join whose probe cache must be re-created in every block a
    /// split creates, while one expression is being emitted under it.
    /// `base` is where the join's value lanes sit on the live stack — NOT
    /// `live.len() - nd`, which holds only at an expression boundary,
    /// before operands are pushed on top of them.
    many: Option<ManySeed>,
    /// Scalar joins whose residual is being emitted right now, innermost
    /// last. Same hazard as `many`, different cache: the join's value lanes
    /// ride the live stack but the cache ENTRY is per block, and a split
    /// inside the residual used to drop it, miss, and re-enter `emit_probe`
    /// without bound — a stack overflow that killed the process rather than
    /// raising. A stack, not an Option: a residual may probe another join.
    probe_seeds: Vec<ProbeSeed>,
}

/// Where an active many-join's probe lanes ride: `nd` live entries starting
/// at `base`, `nd` being the FLATTENED value width (`val_flat_tys`), plus
/// the hit flag to seed the cache with — true in the loop body, false on the
/// LEFT null-extension path.
#[derive(Clone, Copy)]
struct ManySeed {
    join: u32,
    hit: bool,
    base: usize,
    nd: usize,
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
            many: None,
            probe_seeds: Vec::new(),
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
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 => Lit::I64(0),
            Ty::F64 => Lit::F64(0.0),
            Ty::Str => Lit::Str(String::new()),
            // The LEFT-miss payload of a decimal lane: a zero at the
            // column's own scale, exactly like the other lanes' defaults.
            Ty::Dec(p, s) => Lit::Dec(0, p, s),
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
    /// The measured model (the 2026-08-19 spec): AND is the ONLY lazy
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

    /// [`Self::emit_truth`] under a many-join probe context -- the same
    /// wrapper shape as [`Self::emit_many`].
    fn emit_truth_many(
        &mut self,
        e: &SExpr,
        live: &mut Live,
        j: u32,
        hit: bool,
        nd: usize,
    ) -> Result<Value, PrepareError> {
        self.many = Some(ManySeed {
            join: j,
            hit,
            base: live.len() - nd,
            nd,
        });
        self.seed_many(live);
        let out = self.emit_truth(e, live);
        self.many = None;
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
        let lane = self.emit_kind(e, live)?;
        match e.ty.int_range() {
            Some((lo, hi)) if narrow_result_can_escape(&e.kind) => {
                self.narrow_trap(lane, lo, hi, e.ty, live)
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
        lo: i64,
        hi: i64,
        ty: Ty,
        live: &mut Live,
    ) -> Result<Lane, PrepareError> {
        let lo_v = self.const_lit(Lit::I64(lo));
        let hi_v = self.const_lit(Lit::I64(hi));
        let ge = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Ge,
            ty: Ty::I64,
            dst: ge,
            a: l.val,
            b: lo_v,
        });
        let le = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Le,
            ty: Ty::I64,
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
                self.inst(Inst::Itof {
                    narrow,
                    dst,
                    a: l.val,
                });
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
                    // zero-divisor NULL guard is the frontend's CASE wrap.
                    (ArithOp::IDiv, Ty::F64) => BinOp::Fdiv,
                    (ArithOp::Rem, Ty::F64) => BinOp::Frem,
                    (ArithOp::Shl, Ty::I64) => BinOp::Ishl,
                    (ArithOp::Shr, Ty::I64) => BinOp::Ishr,
                    (ArithOp::BitAnd, Ty::I64) => BinOp::Iand,
                    (ArithOp::BitOr, Ty::I64) => BinOp::Ior,
                    (ArithOp::BitXor, Ty::I64) => BinOp::Ixor,
                    (op, ty) => {
                        return Err(PrepareError::Internal(format!(
                            "arith {op:?} on {} escaped the frontend",
                            ty.name()
                        )))
                    }
                };
                // Integer arithmetic traps (overflow, % edge cases): mask
                // nullable payloads so garbage under a false flag can never
                // fire the trap. Float ops are total — no masking needed.
                let (va, vb) = if e.ty.lane() == Ty::I64 {
                    (self.masked(la, Ty::I64), self.masked(lb, Ty::I64))
                } else {
                    (la.val, lb.val)
                };
                let val = self.bin(ir_op, va, vb);
                let lane = Lane {
                    flag: self.combine_flags(la.flag, lb.flag),
                    val,
                };
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
                // iabs traps on i64::MIN; mask the nullable payload.
                let (op, av) = if e.ty.lane() == Ty::I64 {
                    (NumOp1::Iabs, self.masked(l, Ty::I64))
                } else {
                    (NumOp1::Fabs, l.val)
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
                // flag, the Flogb pattern.
                let (av, bv) = match op {
                    StrOp2::Jaccard | StrOp2::Hamming => {
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

    /// Stage-B loop lowering for the (single) join under shape='many':
    ///
    ///   entry:  key exprs (trap per input row), ProbeRange -> [lo, hi)
    ///           (NULL keys force an EMPTY range — a NULL never matches),
    ///           jump header(lo, hi, any=false)
    ///   header(i, end, any): i < end ? body : after(any)
    ///   body:   ProbeRead lanes; the RESIDUAL gates match-ness (and the
    ///           `any` flag); WHERE only gates emission (DuckDB joins
    ///           first, filters second); pass -> stores -> emit.to header
    ///           (any=true); fail -> continue.
    ///   after:  INNER -> skip. LEFT -> any ? skip : null-extension row
    ///           (default lanes, hit=false) gated by WHERE, then emit.
    ///
    /// Predicate/expression emission may SPLIT blocks (CASE machinery), so
    /// the loop state and probe lanes ride the LIVE stack (auto-rebound
    /// across splits) with the invariant: the probe lanes are always the
    /// LAST `nd` live entries; the per-block probe cache is re-seeded from
    /// them before every emission that can reference join columns.
    fn lower_many_loop(
        &mut self,
        exprs: &[(String, SExpr)],
        filter_pred: Option<&SExpr>,
        out_cols: &[Col],
    ) -> Result<(), PrepareError> {
        let j = 0u32;
        let kind = self.joins[0].kind;
        let residual = self.joins[0].residual.clone();
        let keys_expr = self.joins[0].keys.clone();
        let dst_tys: Vec<Ty> = self.val_flat_tys(j);
        let nd = dst_tys.len();

        // entry: keys + range.
        let mut live: Live = Vec::new();
        let mut keys_valid: Option<Value> = None;
        let mut key_vals = Vec::with_capacity(keys_expr.len());
        let key_layout = plan::map_keys(&keys_expr, &self.joins[0].key_cols);
        for (key, k) in keys_expr.iter().zip(&key_layout) {
            let lane = self.emit(key, &mut live)?;
            // Every key here is plain (INDF is refused under `many` above),
            // so every flag comes back and every one gates the range.
            if let Some(fl) = self.encode_probe_key(*k, lane, &mut key_vals) {
                keys_valid = self.combine_flags(keys_valid, Some(fl));
            }
        }
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
                let l2 = self.fresh();
                self.inst(Inst::Select {
                    dst: l2,
                    cond: fl,
                    a: lo,
                    b: zero,
                });
                let h2 = self.fresh();
                self.inst(Inst::Select {
                    dst: h2,
                    cond: fl,
                    a: hi,
                    b: zero,
                });
                (l2, h2)
            }
        };

        let (header, hp) = self.create_block(&[Ty::I64, Ty::I64, Ty::I1]);
        let f0 = self.const_i1(false);
        self.term(Term::Jump {
            to: BlockId(header as u32),
            args: vec![lo, hi, f0],
        });

        self.switch(header);
        let (h_i, h_end, h_any) = (hp[0], hp[1], hp[2]);
        let more = self.fresh();
        self.inst(Inst::Cmp {
            pred: CmpPred::Lt,
            ty: Ty::I64,
            dst: more,
            a: h_i,
            b: h_end,
        });
        let (body, bp) = self.create_block(&[Ty::I64, Ty::I64, Ty::I1]);
        let (after, ap) = self.create_block(&[Ty::I1]);
        self.term(Term::Brif {
            cond: more,
            then_to: BlockId(body as u32),
            then_args: vec![h_i, h_end, h_any],
            else_to: BlockId(after as u32),
            else_args: vec![h_any],
        });

        // body: lanes + the two gates, everything riding live as
        // [inext, end, any, dsts...].
        self.switch(body);
        let (b_i, b_end, b_any) = (bp[0], bp[1], bp[2]);
        let dsts: Vec<Value> = (0..nd).map(|_| self.fresh()).collect();
        self.inst(Inst::ProbeRead {
            static_id: j,
            idx: b_i,
            dsts: dsts.clone(),
        });
        let one = self.const_lit(Lit::I64(1));
        let inext = self.bin(BinOp::Iadd, b_i, one);
        let mut live: Live = Vec::new();
        live.push((Lane { flag: None, val: inext }, Ty::I64));
        live.push((Lane { flag: None, val: b_end }, Ty::I64));
        live.push((Lane { flag: None, val: b_any }, Ty::I1));
        for (&d, &ty) in dsts.iter().zip(dst_tys.iter()) {
            live.push((Lane { flag: None, val: d }, ty));
        }
        let rv = match &residual {
            None => None,
            Some(res) => {
                let rl = self.emit_many(res, &mut live, j, true, nd)?;
                Some(self.truthy(rl))
            }
        };
        let vals: Vec<Value> = live.iter().map(|(l, _)| l.val).collect();
        let (r_inext, r_end, r_any) = (vals[0], vals[1], vals[2]);
        let r_dsts: Vec<Value> = vals[3..].to_vec();
        live.clear();
        let mut match_tys = vec![Ty::I64, Ty::I64];
        match_tys.extend(dst_tys.iter().copied());
        let (matched_blk, mp) = self.create_block(&match_tys);
        let mut match_args = vec![r_inext, r_end];
        match_args.extend(r_dsts.iter().copied());
        match rv {
            None => self.term(Term::Jump {
                to: BlockId(matched_blk as u32),
                args: match_args,
            }),
            Some(cond) => self.term(Term::Brif {
                cond,
                then_to: BlockId(matched_blk as u32),
                then_args: match_args,
                else_to: BlockId(header as u32),
                else_args: vec![r_inext, r_end, r_any],
            }),
        }

        // matched_blk: `any` is TRUE from here on; WHERE gates emission
        // only. live = [inext, end, dsts...].
        self.switch(matched_blk);
        let mut live: Live = Vec::new();
        live.push((Lane { flag: None, val: mp[0] }, Ty::I64));
        live.push((Lane { flag: None, val: mp[1] }, Ty::I64));
        for (&d, &ty) in mp[2..].iter().zip(dst_tys.iter()) {
            live.push((Lane { flag: None, val: d }, ty));
        }
        if let Some(pred) = filter_pred {
            // SELECTION context, same as the scalar WHERE.
            let pv = self.emit_truth_many(pred, &mut live, j, true, nd)?;
            let vals: Vec<Value> = live.iter().map(|(l, _)| l.val).collect();
            let (keep, kp) = {
                let mut tys = vec![Ty::I64, Ty::I64];
                tys.extend(dst_tys.iter().copied());
                self.create_block(&tys)
            };
            let mut keep_args = vec![vals[0], vals[1]];
            keep_args.extend(vals[2..].iter().copied());
            let tt = self.const_i1(true);
            self.term(Term::Brif {
                cond: pv,
                then_to: BlockId(keep as u32),
                then_args: keep_args,
                else_to: BlockId(header as u32),
                else_args: vec![vals[0], vals[1], tt],
            });
            self.switch(keep);
            live.clear();
            live.push((Lane { flag: None, val: kp[0] }, Ty::I64));
            live.push((Lane { flag: None, val: kp[1] }, Ty::I64));
            for (&d, &ty) in kp[2..].iter().zip(dst_tys.iter()) {
                live.push((Lane { flag: None, val: d }, ty));
            }
        }
        self.store_out_row(exprs, out_cols, Some((j, true)), &mut live, nd)?;
        let vals: Vec<Value> = live.iter().map(|(l, _)| l.val).collect();
        let t3 = self.const_i1(true);
        self.term(Term::EmitTo {
            to: BlockId(header as u32),
            args: vec![vals[0], vals[1], t3],
        });

        // after: LEFT null-extension or skip.
        self.switch(after);
        let a_any = ap[0];
        if kind == JoinKind::Left {
            let (done, _) = self.create_block(&[]);
            let (miss, _) = self.create_block(&[]);
            self.term(Term::Brif {
                cond: a_any,
                then_to: BlockId(done as u32),
                then_args: vec![],
                else_to: BlockId(miss as u32),
                else_args: vec![],
            });
            self.switch(miss);
            // live = [dsts(defaults)...] only.
            let mut live: Live = Vec::new();
            for &ty in dst_tys.iter() {
                let d = self.default_of(ty);
                live.push((Lane { flag: None, val: d }, ty));
            }
            if let Some(pred) = filter_pred {
                // WHERE sees the null-extended row too (measured), and it
                // is SELECTION context, same as the scalar path.
                let pv = self.emit_truth_many(pred, &mut live, j, false, nd)?;
                let vals: Vec<Value> = live.iter().map(|(l, _)| l.val).collect();
                let (keep, kp) = self.create_block(&dst_tys);
                let (drop, _) = self.create_block(&[]);
                self.term(Term::Brif {
                    cond: pv,
                    then_to: BlockId(keep as u32),
                    then_args: vals,
                    else_to: BlockId(drop as u32),
                    else_args: vec![],
                });
                self.switch(drop);
                self.term(Term::Skip);
                self.switch(keep);
                live.clear();
                for (&d, &ty) in kp.iter().zip(dst_tys.iter()) {
                    live.push((Lane { flag: None, val: d }, ty));
                }
            }
            self.store_out_row(exprs, out_cols, Some((j, false)), &mut live, nd)?;
            self.term(Term::Emit);
            self.switch(done);
            self.term(Term::Skip);
        } else {
            self.term(Term::Skip);
        }
        Ok(())
    }

    /// Emit `e` with many-join `j`'s probe cache seeded in the current block
    /// AND re-created in every block a split inside `e` creates.
    ///
    /// The join's values ride the live stack, so they survive a transition on
    /// their own — but the CACHE that says "@j is already probed, here are its
    /// registers" is per block (`PB::new` starts it empty). Without the
    /// re-creation, a joined column read inside a CASE arm misses the cache
    /// and falls through to the scalar `Inst::Probe`, which `ir::verify`
    /// rejects for a multimap: "@N is a multimap: use probe.range".
    fn emit_many(
        &mut self,
        e: &SExpr,
        live: &mut Live,
        j: u32,
        hit: bool,
        nd: usize,
    ) -> Result<Lane, PrepareError> {
        self.many = Some(ManySeed {
            join: j,
            hit,
            base: live.len() - nd,
            nd,
        });
        self.seed_many(live);
        let out = self.emit(e, live);
        self.many = None;
        out
    }

    /// (Re)create the current block's probe cache entry for the active
    /// many-join, from the live lanes its values ride on. No-op outside a
    /// many-join.
    fn seed_many(&mut self, live: &Live) {
        let Some(ManySeed {
            join,
            hit,
            base,
            nd,
        }) = self.many
        else {
            return;
        };
        let h = self.const_i1(hit);
        let ds: Vec<Value> = live[base..base + nd].iter().map(|(l, _)| l.val).collect();
        self.blocks[self.cur].probes.insert(join, (h, ds));
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

    /// Emit every output expression and store it. Under a many-join
    /// (`seed` present) the probe cache is re-seeded from `live`'s
    /// trailing `nd` lanes before EVERY expression — emission may split
    /// blocks, and each new block starts with an empty cache.
    fn store_out_row(
        &mut self,
        exprs: &[(String, SExpr)],
        out_cols: &[Col],
        seed: Option<(u32, bool)>,
        live: &mut Live,
        nd: usize,
    ) -> Result<(), PrepareError> {
        for (ci, (_, e)) in exprs.iter().enumerate() {
            let lane = match seed {
                Some((j, hit)) => self.emit_many(e, live, j, hit, nd)?,
                None => self.emit(e, live)?,
            };
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
        // entry was lost across a block transition. That used to recurse until
        // the process died of stack overflow; the seeding above prevents it,
        // and this turns any FUTURE hole into a named error rather than a
        // dead serving process.
        if self.probe_seeds.iter().any(|p| p.join == j) {
            return Err(PrepareError::Internal(format!(
                "probe cache for @{j} lost inside its own residual"
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
        let nrets = self.udfs[ext as usize].rets.len();
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
            (Ty::Str, Ty::I64) | (Ty::Str, Ty::F64) => {
                let ok = self.fresh();
                let parsed = self.fresh();
                if to == Ty::I64 {
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
                                _ => "INT64",
                            }
                        )
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

    /// NULL propagation: the result is valid iff every nullable input is.
    fn combine_flags(&mut self, a: Option<Value>, b: Option<Value>) -> Option<Value> {
        match (a, b) {
            (None, None) => None,
            (Some(f), None) | (None, Some(f)) => Some(f),
            (Some(fa), Some(fb)) => Some(self.bin(BinOp::And, fa, fb)),
        }
    }
}
