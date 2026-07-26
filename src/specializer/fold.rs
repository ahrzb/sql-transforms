//! Prepare-time scalar folding: the expression half of binding-time
//! analysis. Every all-constant subtree collapses to `Lit`/`NullOf` before
//! lowering, so a static computation never reaches the per-row IR.
//!
//! The folder is deliberately conservative — it folds ONLY when the result
//! provably equals what the interpreter would compute at run time:
//! * every operand must itself be a constant (`Lit`/`NullOf`); a constant
//!   that merely dominates (`FALSE AND dynamic`) is never folded, because
//!   dropping the dynamic side could drop its trap;
//! * an operation that would trap at run time (integer overflow, `% 0`) is
//!   left unfolded — the trap stays a run-time trap, same timing as before;
//! * f64 arithmetic and comparisons mirror exec/interp.rs exactly (IEEE:
//!   `x/0 = inf`, NaN compares false except `!=`);
//! * CASE and CAST nodes fold their children but never themselves — CAST
//!   can trap and CASE guards branch traps, both wrong to evaluate eagerly.

use super::ir::{CmpPred, Lit, Ty};
use super::plan::{ArithOp, SExpr, SKind};

/// A constant operand: a payload or a typed NULL.
enum K {
    Val(Lit),
    Null,
}

fn as_const(e: &SExpr) -> Option<K> {
    match &e.kind {
        SKind::Lit(l) => Some(K::Val(l.clone())),
        SKind::NullOf => Some(K::Null),
        _ => None,
    }
}

fn lit(l: Lit, ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::Lit(l),
        ty,
        nullable: false,
    }
}

fn null(ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::NullOf,
        ty,
        nullable: true,
    }
}

/// Fold every all-constant subtree of `e`, bottom-up.
pub fn fold(e: SExpr) -> SExpr {
    let SExpr { kind, ty, nullable } = e;
    let e = |kind| SExpr { kind, ty, nullable };
    match kind {
        SKind::Col(_) | SKind::StaticCol { .. } | SKind::Lit(_) | SKind::NullOf => e(kind),
        SKind::IntToFloat(inner) => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(Lit::I64(i))) => lit(Lit::F64(i as f64), ty),
                Some(K::Null) => null(ty),
                _ => e(SKind::IntToFloat(Box::new(inner))),
            }
        }
        // Wave-1 math: fold children only — the ops themselves stay
        // runtime so constant domain errors trap per row exactly like the
        // vectorized path we pin against (no fold/vector divergence).
        SKind::Str2 { op, a, b } => {
            let a = fold(*a);
            let b = fold(*b);
            e(SKind::Str2 {
                op,
                a: Box::new(a),
                b: Box::new(b),
            })
        }
        SKind::SLen { bytes, a } => {
            let a = fold(*a);
            e(SKind::SLen {
                bytes,
                a: Box::new(a),
            })
        }
        SKind::Like { ci, a, p, esc } => {
            let a = fold(*a);
            let p = fold(*p);
            let esc = esc.map(|e| Box::new(fold(*e)));
            e(SKind::Like {
                ci,
                a: Box::new(a),
                p: Box::new(p),
                esc,
            })
        }
        SKind::Round2 { trunc, a, n } => {
            let a = fold(*a);
            let n = fold(*n);
            e(SKind::Round2 {
                trunc,
                a: Box::new(a),
                n: Box::new(n),
            })
        }
        SKind::MathF1 { op, a } => {
            let a = fold(*a);
            e(SKind::MathF1 { op, a: Box::new(a) })
        }
        SKind::MathF2 { op, a, b } => {
            let a = fold(*a);
            let b = fold(*b);
            e(SKind::MathF2 {
                op,
                a: Box::new(a),
                b: Box::new(b),
            })
        }
        SKind::Not(inner) => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(Lit::I1(b))) => lit(Lit::I1(!b), ty),
                Some(K::Null) => null(ty),
                _ => e(SKind::Not(Box::new(inner))),
            }
        }
        SKind::IsNull { negated, inner } => {
            let inner = fold(*inner);
            match as_const(&inner) {
                Some(K::Val(_)) => lit(Lit::I1(negated), ty),
                Some(K::Null) => lit(Lit::I1(!negated), ty),
                None => e(SKind::IsNull {
                    negated,
                    inner: Box::new(inner),
                }),
            }
        }
        SKind::Arith { op, a, b } => {
            let (a, b) = (fold(*a), fold(*b));
            match (as_const(&a), as_const(&b)) {
                (Some(K::Null), Some(_)) | (Some(_), Some(K::Null)) => null(ty),
                (Some(K::Val(x)), Some(K::Val(y))) => match arith(op, &x, &y) {
                    Some(l) => lit(l, ty),
                    // Would trap at run time — keep the node, keep the trap.
                    None => e(SKind::Arith {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    }),
                },
                _ => e(SKind::Arith {
                    op,
                    a: Box::new(a),
                    b: Box::new(b),
                }),
            }
        }
        SKind::Cmp { pred, a, b } => {
            let (a, b) = (fold(*a), fold(*b));
            match (as_const(&a), as_const(&b)) {
                (Some(K::Null), Some(_)) | (Some(_), Some(K::Null)) => null(ty),
                (Some(K::Val(x)), Some(K::Val(y))) => lit(Lit::I1(cmp(pred, &x, &y)), ty),
                _ => e(SKind::Cmp {
                    pred,
                    a: Box::new(a),
                    b: Box::new(b),
                }),
            }
        }
        SKind::And { a, b } => kleene(true, *a, *b, ty, nullable),
        SKind::Or { a, b } => kleene(false, *a, *b, ty, nullable),
        SKind::Case { arms, default } => e(SKind::Case {
            arms: arms.into_iter().map(|(c, r)| (fold(c), fold(r))).collect(),
            default: default.map(|d| Box::new(fold(*d))),
        }),
        SKind::Cast { inner, trying } => e(SKind::Cast {
            inner: Box::new(fold(*inner)),
            trying,
        }),
        // Builtin nodes fold children only (ponytail: constant upper('a')
        // etc. can fold later if a corpus query ever cares).
        SKind::StrCase { upper, a } => e(SKind::StrCase {
            upper,
            a: Box::new(fold(*a)),
        }),
        SKind::Trim { side, a, chars } => e(SKind::Trim {
            side,
            a: Box::new(fold(*a)),
            chars: Box::new(fold(*chars)),
        }),
        SKind::Substr { a, start, len } => e(SKind::Substr {
            a: Box::new(fold(*a)),
            start: Box::new(fold(*start)),
            len: len.map(|l| Box::new(fold(*l))),
        }),
        SKind::Abs(a) => e(SKind::Abs(Box::new(fold(*a)))),
        SKind::Round(a) => e(SKind::Round(Box::new(fold(*a)))),
        SKind::Concat { a, b } => e(SKind::Concat {
            a: Box::new(fold(*a)),
            b: Box::new(fold(*b)),
        }),
    }
}

/// Kleene AND/OR, folded only when BOTH sides are constants — the full
/// three-valued table, including `FALSE AND NULL = FALSE`.
fn kleene(is_and: bool, a: SExpr, b: SExpr, ty: Ty, nullable: bool) -> SExpr {
    let (a, b) = (fold(a), fold(b));
    let tri = |e: &SExpr| match as_const(e) {
        Some(K::Val(Lit::I1(v))) => Some(Some(v)),
        Some(K::Null) => Some(None),
        _ => None,
    };
    if let (Some(x), Some(y)) = (tri(&a), tri(&b)) {
        let r = if is_and {
            match (x, y) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            }
        } else {
            match (x, y) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            }
        };
        return match r {
            Some(v) => lit(Lit::I1(v), ty),
            None => null(ty),
        };
    }
    let (a, b) = (Box::new(a), Box::new(b));
    let kind = if is_and {
        SKind::And { a, b }
    } else {
        SKind::Or { a, b }
    };
    SExpr { kind, ty, nullable }
}

/// `None` = the interpreter would trap on this — do not fold.
fn arith(op: ArithOp, a: &Lit, b: &Lit) -> Option<Lit> {
    match (a, b) {
        (Lit::I64(x), Lit::I64(y)) => match op {
            ArithOp::Add => x.checked_add(*y).map(Lit::I64),
            ArithOp::Sub => x.checked_sub(*y).map(Lit::I64),
            ArithOp::Mul => x.checked_mul(*y).map(Lit::I64),
            ArithOp::Rem => x.checked_rem(*y).map(Lit::I64),
            ArithOp::Div => unreachable!("/ is promoted to f64 by the frontend"),
        },
        (Lit::F64(x), Lit::F64(y)) => Some(Lit::F64(match op {
            ArithOp::Add => x + y,
            ArithOp::Sub => x - y,
            ArithOp::Mul => x * y,
            ArithOp::Div => x / y,
            // IEEE, exactly as exec/interp.rs: x % 0.0 is NaN, never traps.
            ArithOp::Rem => x % y,
        })),
        _ => unreachable!("operands are promoted to a common type at bind"),
    }
}

fn cmp(pred: CmpPred, a: &Lit, b: &Lit) -> bool {
    use std::cmp::Ordering;
    let ord = |o: Ordering| match pred {
        CmpPred::Eq => o == Ordering::Equal,
        CmpPred::Ne => o != Ordering::Equal,
        CmpPred::Lt => o == Ordering::Less,
        CmpPred::Le => o != Ordering::Greater,
        CmpPred::Gt => o == Ordering::Greater,
        CmpPred::Ge => o != Ordering::Less,
    };
    match (a, b) {
        (Lit::I64(x), Lit::I64(y)) => ord(x.cmp(y)),
        (Lit::Str(x), Lit::Str(y)) => ord(x.cmp(y)),
        // DuckDB DOUBLE order, exactly as exec/interp.rs computes it.
        (Lit::F64(x), Lit::F64(y)) => ord(super::exec::duck_fcmp(*x, *y)),
        _ => unreachable!("cmp operands share a type; i1 cmp rejected at bind"),
    }
}
