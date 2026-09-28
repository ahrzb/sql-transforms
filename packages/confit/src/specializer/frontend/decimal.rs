//! DECIMAL expressions: DuckDB's binding rules, read from its source
//! (docs/specs/decimal-expressions.md). Values are scaled integers on the
//! i128 lane, so every rule here is exact.

use super::*;
use super::super::ir::{DecOp, DecUnary};

/// DuckDB's storage tier for a width (`DecimalType` physical types):
/// what decides whether its binder casts an argument whose scale already
/// matches.
fn storage(p: u8) -> u8 {
    match p {
        0..=4 => 16,
        5..=9 => 32,
        10..=18 => 64,
        _ => 128,
    }
}

/// The (width, scale) a numeric operand has as a DECIMAL argument (§2):
/// a DECIMAL its own, an integer its `GetDecimalProperties` width.
pub(super) fn dec_props(t: Ty) -> Option<(u8, u8)> {
    match t {
        Ty::Dec(p, s) => Some((p, s)),
        t if t.is_int() => Some((int_dec_width(t), 0)),
        _ => None,
    }
}

/// A checked conversion to `to` (§8). A typed NULL retypes.
pub(super) fn dec_cast_node(e: SExpr, to: Ty) -> SExpr {
    if e.ty == to {
        return e;
    }
    if matches!(e.kind, SKind::NullOf) {
        return null_of(to);
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::DecCast(Box::new(e)),
        ty: to,
        nullable,
    }
}

/// DuckDB's argument cast for a decimal operator (`BindDecimalArithmetic`):
/// skipped when the scale and the storage tier already match, since the
/// kernel then reads the value as it is.
fn arg_to(e: SExpr, to: Ty) -> SExpr {
    let (tp, ts) = to.dec().expect("a decimal target");
    match e.ty {
        Ty::Dec(p, s) if s == ts && storage(p) == storage(tp) => e,
        _ => dec_cast_node(e, to),
    }
}

impl Binder<'_> {
    /// `+ - * %` with a DECIMAL operand and the other a DECIMAL or an
    /// integer (§3-§5). The caller has routed DOUBLE mixes, `/` and `//` to
    /// the DOUBLE path and taken the NULL-operand collapse.
    pub(super) fn dec_arith(
        &self,
        op: ArithOp,
        a: SExpr,
        b: SExpr,
    ) -> Result<SExpr, PrepareError> {
        let (Some((pa, sa)), Some((pb, sb))) = (dec_props(a.ty), dec_props(b.ty)) else {
            return Err(PrepareError::Bind(format!(
                "No function matches the given name and argument types '{}({}, {})'",
                arith_sym(op),
                duck_ty_name(a.ty),
                duck_ty_name(b.ty)
            )));
        };
        let max_w = pa.max(pb);
        let (dop, ty, check, a, b) = match op {
            ArithOp::Add | ArithOp::Sub | ArithOp::Rem => {
                // `BindDecimalArithmetic`, arithmetic.cpp:193-246.
                let s = sa.max(sb);
                let mut p = (s + (pa - sa).max(pb - sb)).max(max_w);
                let mut check = 0;
                if op != ArithOp::Rem {
                    p += 1;
                    if p > 18 && max_w <= 18 {
                        check = 18;
                        p = 18;
                    }
                }
                if p > 38 {
                    // A remainder is never larger than its divisor, so the
                    // capped `%` needs no result check.
                    check = if op == ArithOp::Rem { 0 } else { 38 };
                    p = 38;
                }
                let ty = Ty::Dec(p, s);
                let dop = match op {
                    ArithOp::Add => DecOp::Add,
                    ArithOp::Sub => DecOp::Sub,
                    _ => DecOp::Rem,
                };
                (dop, ty, check, arg_to(a, ty), arg_to(b, ty))
            }
            ArithOp::Mul => {
                // `BindDecimalMultiply`, arithmetic.cpp:836-895.
                let s = sa + sb;
                if s > 38 {
                    return Err(PrepareError::Bind(format!(
                        "Needed scale {s} to accurately represent the multiplication \
                         result, but this is out of range of the DECIMAL type. Max scale \
                         is 38; could not perform an accurate multiplication. Either add \
                         a cast to DOUBLE, or add an explicit cast to a decimal with a \
                         lower scale."
                    )));
                }
                let mut p = pa + pb;
                let mut check = 0;
                if p > 18 && max_w <= 18 && s < 18 {
                    check = 18;
                    p = 18;
                }
                if p > 38 {
                    check = 38;
                    p = 38;
                }
                // No rescale: the product of the scaled integers is already
                // at scale sa + sb. Only the storage tier may change.
                let widen = |e: SExpr, s: u8| match e.ty {
                    Ty::Dec(ep, _) if storage(ep) == storage(p) => e,
                    _ => dec_cast_node(e, Ty::Dec(p, s)),
                };
                (DecOp::Mul, Ty::Dec(p, s), check, widen(a, sa), widen(b, sb))
            }
            _ => unreachable!("the caller routes only + - * % here"),
        };
        let nullable = a.nullable || b.nullable || dop == DecOp::Rem;
        Ok(SExpr {
            kind: SKind::DecArith {
                op: dop,
                check,
                a: Box::new(a),
                b: Box::new(b),
            },
            ty,
            nullable,
        })
    }

    /// Unary minus over a DECIMAL: `0 - x` in the operand's own type, which
    /// cannot overflow (DuckDB keeps the type: `-2.5` is DECIMAL(2,1)).
    pub(super) fn dec_negate(&self, e: SExpr) -> SExpr {
        let (p, s) = e.ty.dec().expect("a decimal operand");
        let nullable = e.nullable;
        SExpr {
            kind: SKind::DecArith {
                op: DecOp::Sub,
                check: 0,
                a: Box::new(SExpr {
                    kind: SKind::Lit(Lit::Dec(0, p, s)),
                    ty: e.ty,
                    nullable: false,
                }),
                b: Box::new(e),
            },
            ty: Ty::Dec(p, s),
            nullable,
        }
    }

    /// A builtin's DECIMAL overload (numeric.cpp): abs keeps the type;
    /// ceil/floor/round/trunc drop the scale, and round/trunc with a
    /// constant precision `n` keep min(n, s) digits of it (§9 of the spec).
    pub(super) fn dec_overload(
        &self,
        name: &str,
        e: SExpr,
        rest: &[&SqlExpr],
    ) -> Result<SExpr, PrepareError> {
        let (w, s) = e.ty.dec().expect("a decimal argument");
        // A constant NULL DECIMAL argument binds DuckDB's SQLNULL, as the
        // decimal operators do (measured: `floor(CAST(NULL AS
        // DECIMAL(4,2)))` is INTEGER).
        let (e, is_null) = fold_operand(e);
        if is_null {
            return Ok(null_of(Ty::I32));
        }
        let op = match name {
            "abs" => DecUnary::Abs,
            "ceil" | "ceiling" => DecUnary::Ceil,
            "floor" => DecUnary::Floor,
            "round" => DecUnary::Round,
            _ => DecUnary::Trunc,
        };
        let node = |op, k: u8, m: u8, ty: Ty, e: SExpr| {
            let nullable = e.nullable;
            fold(SExpr {
                kind: SKind::DecUnary {
                    op,
                    k,
                    m,
                    a: Box::new(e),
                },
                ty,
                nullable,
            })
        };
        match (op, rest) {
            (DecUnary::Abs, []) => Ok(node(op, 0, 0, e.ty, e)),
            (_, []) if op != DecUnary::Abs => Ok(node(op, s, 0, Ty::Dec(w, 0), e)),
            (DecUnary::Round | DecUnary::Trunc, [n]) => {
                // `BindDecimalRoundPrecision`: the precision must fold to a
                // constant at bind.
                let fname = name.to_uppercase();
                let not_const = || {
                    Err(PrepareError::Bind(format!(
                        "{fname}(DECIMAL, INTEGER) with non-constant precision is not supported"
                    )))
                };
                let Some(n) = self.expr_or_null(n)? else {
                    return not_const();
                };
                let n = fold(n);
                let SKind::Lit(Lit::I64(n)) = n.kind else {
                    return not_const();
                };
                if n < 0 {
                    // Past the integer digits the whole column is 0, a
                    // constant even for a NULL row (numeric.cpp sets a
                    // constant INTEGER 0).
                    if n <= -i64::from(w - s) {
                        return Ok(SExpr {
                            kind: SKind::Lit(Lit::Dec(0, w, 0)),
                            ty: Ty::Dec(w, 0),
                            nullable: false,
                        });
                    }
                    let up = (-n) as u8;
                    Ok(node(op, s + up, up, Ty::Dec(w, 0), e))
                } else if n >= i64::from(s) {
                    Ok(e)
                } else {
                    Ok(node(op, s - n as u8, 0, Ty::Dec(w, n as u8), e))
                }
            }
            _ => Err(PrepareError::Bind(format!(
                "no function matches {name} over DECIMAL({w},{s}) with {} more argument(s)",
                rest.len()
            ))),
        }
    }

    /// CAST with a DECIMAL on either side (§8). DOUBLE and VARCHAR out of a
    /// DECIMAL cannot fail; integer <-> DECIMAL and DECIMAL -> DECIMAL are
    /// checked conversions. What is not served refuses by name: DOUBLE,
    /// VARCHAR and BOOLEAN into a DECIMAL, a DECIMAL into BOOLEAN, and a
    /// TRY_CAST that could fail.
    pub(super) fn dec_cast_expr(
        &self,
        inner: SExpr,
        to: Ty,
        trying: bool,
    ) -> Result<SExpr, PrepareError> {
        let refuse = |what: String| Err(unsup(what));
        match (inner.ty, to) {
            (Ty::Dec(..), Ty::F64) => Ok(dec_to_float(inner)),
            (Ty::Dec(..), Ty::Str) => Ok(dec_cast_node(inner, to)),
            // A TRY_CAST that can fail: NULL where CAST would trap.
            (f, t) if trying && (f.dec().is_some() || f.is_int()) && (t.dec().is_some() || t.is_int()) => {
                if matches!(inner.kind, SKind::NullOf) {
                    return Ok(null_of(to));
                }
                Ok(fold(SExpr {
                    kind: SKind::DecTryCast(Box::new(inner)),
                    ty: to,
                    nullable: true,
                }))
            }
            (Ty::Dec(..), t) if t.is_int() || t.dec().is_some() => Ok(dec_cast_node(inner, to)),
            (f, Ty::Dec(..)) if f.is_int() => Ok(dec_cast_node(inner, to)),
            (f, t) => refuse(format!(
                "CAST from {} to {} -- DECIMAL casts served are to and from the \
                 integer widths and other DECIMALs, and out to DOUBLE and VARCHAR",
                duck_ty_name(f),
                duck_ty_name(t)
            )),
        }
    }
}

/// The common type of two numeric types where at least one is a DECIMAL
/// (§7, `MaxLogicalType` / `DecimalSizeCheck`): DOUBLE with a DOUBLE,
/// otherwise the DECIMAL that holds both. `None` for a non-numeric pair.
pub(super) fn dec_common(x: Ty, y: Ty) -> Option<Ty> {
    match (x, y) {
        (Ty::Dec(..), Ty::F64) | (Ty::F64, Ty::Dec(..)) => Some(Ty::F64),
        (Ty::Dec(p1, s1), Ty::Dec(p2, s2)) => {
            let int = (p1 - s1).max(p2 - s2);
            let mut s = s1.max(s2);
            let mut p = int + s;
            if p > 38 {
                p = 38;
                s = 38 - int;
            }
            Some(Ty::Dec(p, s))
        }
        (Ty::Dec(p, s), i) | (i, Ty::Dec(p, s)) if i.is_int() => {
            let w = int_dec_width(i);
            Some(if w > p - s {
                Ty::Dec((w + s).min(38), s)
            } else {
                Ty::Dec(p, s)
            })
        }
        _ => None,
    }
}

/// An operand at the §7 common type: a DOUBLE through `dec_to_float`, a
/// DECIMAL or integer through the checked conversion.
pub(super) fn to_common(e: SExpr, to: Ty) -> SExpr {
    match (e.ty, to) {
        (x, y) if x == y => e,
        (Ty::Dec(..), Ty::F64) => dec_to_float(e),
        (x, Ty::F64) if x.is_int() => promote_f64(e),
        _ => dec_cast_node(e, to),
    }
}
