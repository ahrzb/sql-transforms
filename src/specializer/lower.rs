//! Produce/consume lowering: relational IR -> imperative IR. For the
//! stretch-1 ribbon (scan -> filter? -> project) the pipeline is a straight
//! line: one entry block computes the predicate, `keep`/`drop` blocks handle
//! the filter, projections store on the keep path.
//!
//! Null-lane discipline: an SExpr with `nullable == false` lowers to a bare
//! payload register (no flag anywhere); a nullable one carries an `i1` flag
//! alongside, combined with `and` as NULL propagates. WHERE keeps a row iff
//! the predicate is TRUE — flag && value — which is exactly one `and`.
//!
//! ponytail: the keep block re-loads its input columns instead of receiving
//! them as branch args — pure loads, identical semantics, simpler lowering.
//! Switch to branch-arg threading when the Cranelift backend makes the extra
//! column reads measurable.

use std::collections::HashMap;

use super::frontend::PrepareError;
use super::ir::{
    BinOp, Block, BlockId, Builder, Inst, Program, Term, Ty, Value,
};
use super::plan::{ArithOp, Rel, SExpr, SKind};

/// Lower a bound relational tree to a complete (unverified) program.
/// `prepare` in mod.rs runs the verifier on the result — lowering bugs
/// surface as `PrepareError::Internal`, never as executable programs.
pub fn lower(rel: &Rel, in_cols: &[super::ir::Col], out_cols: Vec<super::ir::Col>, name: &str) -> Result<Program, PrepareError> {
    // Peel the fixed v0 shape: Project [Filter?] Scan.
    let (exprs, filtered) = match rel {
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
        _ => return Err(PrepareError::Internal("plan root is not a projection".to_string())),
    };

    let mut b = Builder::new();
    let mut blocks = Vec::new();

    match filtered {
        None => {
            // Single block: compute projections, store, emit.
            let mut ctx = BlockCtx::new(&mut b, in_cols);
            store_projections(&mut ctx, exprs, &out_cols);
            blocks.push(Block { params: vec![], insts: ctx.insts, term: Term::Emit });
        }
        Some(pred) => {
            // entry: predicate -> brif keep, drop
            let mut entry = BlockCtx::new(&mut b, in_cols);
            let p = emit_expr(&mut entry, pred);
            // Keep iff pred IS TRUE: NULL (flag=false) and FALSE both drop.
            let cond = match p.flag {
                None => p.val,
                Some(flag) => {
                    let c = entry.b.fresh();
                    entry.insts.push(Inst::Bin { op: BinOp::And, dst: c, a: flag, b: p.val });
                    c
                }
            };
            blocks.push(Block {
                params: vec![],
                insts: entry.insts,
                term: Term::Brif {
                    cond,
                    then_to: BlockId(1),
                    then_args: vec![],
                    else_to: BlockId(2),
                    else_args: vec![],
                },
            });
            // keep: recompute inputs (see module note), store, emit.
            let mut keep = BlockCtx::new(&mut b, in_cols);
            store_projections(&mut keep, exprs, &out_cols);
            blocks.push(Block { params: vec![], insts: keep.insts, term: Term::Emit });
            blocks.push(Block { params: vec![], insts: vec![], term: Term::Skip });
        }
    }

    Ok(Program {
        statics: vec![],
        name: name.to_string(),
        in_cols: in_cols.to_vec(),
        out_cols,
        blocks,
    })
}

/// A value in the null-lane representation: payload + optional validity.
struct Lane {
    flag: Option<Value>,
    val: Value,
}

/// Per-block emission context. The column cache keeps one load per column
/// per block (SSA values are block-local, so the cache is too).
struct BlockCtx<'a> {
    b: &'a mut Builder,
    in_cols: &'a [super::ir::Col],
    insts: Vec<Inst>,
    col_cache: HashMap<u32, (Option<Value>, Value)>,
}

impl<'a> BlockCtx<'a> {
    fn new(b: &'a mut Builder, in_cols: &'a [super::ir::Col]) -> BlockCtx<'a> {
        BlockCtx { b, in_cols, insts: Vec::new(), col_cache: HashMap::new() }
    }
}

fn store_projections(ctx: &mut BlockCtx<'_>, exprs: &[(String, SExpr)], out_cols: &[super::ir::Col]) {
    for (ci, (_, e)) in exprs.iter().enumerate() {
        let lane = emit_expr(ctx, e);
        let col = ci as u32;
        if out_cols[ci].ty.nullable {
            let flag = match lane.flag {
                Some(f) => f,
                // Declared nullable but provably non-null cannot happen: the
                // out column's nullability IS the expression's derivation.
                None => unreachable!("out column nullable without a flag lane"),
            };
            ctx.insts.push(Inst::StoreOpt { col, flag, val: lane.val });
        } else {
            debug_assert!(lane.flag.is_none());
            ctx.insts.push(Inst::Store { col, val: lane.val });
        }
    }
}

fn emit_expr(ctx: &mut BlockCtx<'_>, e: &SExpr) -> Lane {
    match &e.kind {
        SKind::Col(idx) => {
            if let Some((flag, val)) = ctx.col_cache.get(idx) {
                return Lane { flag: *flag, val: *val };
            }
            let col = &ctx.in_cols[*idx as usize];
            let lane = if col.ty.nullable {
                let flag = ctx.b.fresh();
                let val = ctx.b.fresh();
                ctx.insts.push(Inst::LoadOpt { flag, dst: val, col: *idx });
                Lane { flag: Some(flag), val }
            } else {
                let val = ctx.b.fresh();
                ctx.insts.push(Inst::Load { dst: val, col: *idx });
                Lane { flag: None, val }
            };
            ctx.col_cache.insert(*idx, (lane.flag, lane.val));
            lane
        }
        SKind::Lit(lit) => {
            let dst = ctx.b.fresh();
            ctx.insts.push(Inst::Const { dst, lit: lit.clone() });
            Lane { flag: None, val: dst }
        }
        SKind::IntToFloat(inner) => {
            let l = emit_expr(ctx, inner);
            let dst = ctx.b.fresh();
            ctx.insts.push(Inst::Itof { dst, a: l.val });
            Lane { flag: l.flag, val: dst }
        }
        SKind::Arith { op, a, b } => {
            let la = emit_expr(ctx, a);
            let lb = emit_expr(ctx, b);
            let ir_op = match (op, e.ty) {
                (ArithOp::Add, Ty::I64) => BinOp::Iadd,
                (ArithOp::Sub, Ty::I64) => BinOp::Isub,
                (ArithOp::Mul, Ty::I64) => BinOp::Imul,
                (ArithOp::Rem, Ty::I64) => BinOp::Irem,
                (ArithOp::Add, Ty::F64) => BinOp::Fadd,
                (ArithOp::Sub, Ty::F64) => BinOp::Fsub,
                (ArithOp::Mul, Ty::F64) => BinOp::Fmul,
                (ArithOp::Div, Ty::F64) => BinOp::Fdiv,
                (ArithOp::Rem, Ty::F64) => {
                    // DuckDB fmod: defer until pinned against the oracle.
                    // The frontend currently only produces int Rem; guard it.
                    unreachable!("float % not produced by the v0 frontend")
                }
                (ArithOp::Div, _) => unreachable!("/ is always f64 after promotion"),
                (op, ty) => unreachable!("arith {op:?} on {}", ty.name()),
            };
            let dst = ctx.b.fresh();
            ctx.insts.push(Inst::Bin { op: ir_op, dst, a: la.val, b: lb.val });
            Lane { flag: combine_flags(ctx, la.flag, lb.flag), val: dst }
        }
        SKind::Cmp { pred, a, b } => {
            let la = emit_expr(ctx, a);
            let lb = emit_expr(ctx, b);
            let dst = ctx.b.fresh();
            ctx.insts.push(Inst::Cmp { pred: *pred, ty: a.ty, dst, a: la.val, b: lb.val });
            Lane { flag: combine_flags(ctx, la.flag, lb.flag), val: dst }
        }
    }
}

/// NULL propagation: the result is valid iff every nullable input is.
fn combine_flags(ctx: &mut BlockCtx<'_>, a: Option<Value>, b: Option<Value>) -> Option<Value> {
    match (a, b) {
        (None, None) => None,
        (Some(f), None) | (None, Some(f)) => Some(f),
        (Some(fa), Some(fb)) => {
            let dst = ctx.b.fresh();
            ctx.insts.push(Inst::Bin { op: BinOp::And, dst, a: fa, b: fb });
            Some(dst)
        }
    }
}
