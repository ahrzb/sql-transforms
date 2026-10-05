//! Literals, widths, promotion and unification.

use super::*;

pub(super) fn dec_to_float(e: SExpr) -> SExpr {
    let nullable = e.nullable;
    SExpr {
        kind: SKind::DecToFloat(Box::new(e)),
        ty: Ty::F64,
        nullable,
    }
}

/// The DECIMAL width DuckDB gives an integer type when the two meet —
/// `LogicalType::GetDecimalProperties`, src/common/types.cpp:730-795.
pub(super) fn int_dec_width(t: Ty) -> u8 {
    match t {
        Ty::I8 => 3,
        Ty::I16 => 5,
        Ty::I32 => 10,
        Ty::U8 => 3,
        Ty::U16 => 5,
        Ty::U32 => 10,
        _ => 19,
    }
}

/// The bound subject of a regex op must be VARCHAR (no implicit casts —
/// pins-waveB/; DuckDB binder errors name the function).
pub(super) fn str_only(name: &str, e: SExpr) -> Result<SExpr, PrepareError> {
    if e.ty != Ty::Str {
        return Err(PrepareError::Bind(format!(
            "no function matches {name}({})",
            e.ty.name()
        )));
    }
    Ok(e)
}

/// `''` for non-NULL subjects, NULL for NULL ones (the pinned NULL-group
/// result of regexp_extract).
pub(super) fn empty_for_nonnull(subject: SExpr) -> SExpr {
    if !subject.nullable {
        return lit_str("");
    }
    let is_null = SExpr {
        kind: SKind::IsNull {
            negated: false,
            inner: Box::new(subject),
        },
        ty: Ty::I1,
        nullable: false,
    };
    SExpr {
        kind: SKind::Case {
            arms: vec![(is_null, null_of(Ty::Str))],
            default: Some(Box::new(lit_str(""))),
        },
        ty: Ty::Str,
        nullable: true,
    }
}

/// Fold an arithmetic operand, and say whether it folded to a typed NULL.
///
/// Every arithmetic operator shares one rule, and this is it; comparisons
/// deliberately do not elide a NULL operand, and `cmp` says why. Folding
/// comes FIRST so that a NULL PRODUCED by constant folding (a
/// constant-condition CASE landing on NULL) is visible to the caller's NULL
/// check, exactly as DuckDB's folder sees it; `fold` is pure and idempotent.
/// A folded NULL then goes back to the caller's consumer as a typed NULL and
/// not as a live node, because that is what DuckDB's binder hands upward and
/// what the consumers above read: `NULL || 'x'` is INTEGER-typed SQLNULL,
/// while wrapping the NULL in a live op would type the `||` VARCHAR instead.
///
/// Only what DuckDB's binder can fold is folded here (`bind_foldable`): our
/// fold dead-arm-eliminates a CASE holding a COLUMN, which DuckDB's binder
/// does not, and folding it at bind time typed
/// `- (CASE WHEN false THEN x END) || 'y'` INTEGER where DuckDB answers
/// VARCHAR. Such an operand stays live here; the projection's own fold
/// still simplifies it after typing.
pub(super) fn fold_operand(e: SExpr) -> (SExpr, bool) {
    let e = bind_fold(e);
    let is_null = matches!(e.kind, SKind::NullOf);
    (e, is_null)
}

/// `fold`, for an operand the binder keeps building on: only what DuckDB's
/// binder can fold is folded, so a column erased by dead-arm elimination
/// cannot reach a later `bind_foldable` test (the whole-call NULL, the
/// `||` collapse, a pure UDF's bind-time call) as a foldable NULL.
/// Measured, fuzz seed 18995:
/// `repeat(CAST((CASE WHEN FALSE THEN c1 END) AS VARCHAR), <overflow>)`
/// runs per row on DuckDB and traps; a folding CAST made it a NULL call.
///
/// A closed constant our fold cannot finish (`'0' LIKE 'a_c'` under a CASE)
/// is evaluated, and a NULL result becomes the NULL it is on DuckDB, whose
/// binder elides a trapping sibling under it (nightly seed 2323487). A value
/// or a trap keeps the node: only the NULL decides anything at bind.
pub(super) fn bind_fold(e: SExpr) -> SExpr {
    if !bind_foldable(&e) {
        return e;
    }
    let e = fold(e);
    // Only a subtree that can be NULL can decide anything here.
    if !e.nullable || matches!(e.kind, SKind::Lit(_) | SKind::NullOf) {
        return e;
    }
    // A value leaves the node as it is, its nullability included: the
    // lowering still gives a TRY_CAST under it a flag, and a non-nullable
    // node over a flagged lane breaks the output store (campaign seed
    // 5001244).
    match eval_closed(&e, Vec::new()) {
        Some(None) => null_of(e.ty),
        _ => e,
    }
}

/// The value of a closed constant expression, computed by the interpreter
/// over one row of no columns: the runtime's own answer, so fold and run
/// cannot disagree. DuckDB's binder evaluates a foldable expression the same
/// way (`TryEvaluateScalar`). `None` = not evaluable here (lowering refused,
/// a trap, a regex this table does not carry); `Some(None)` = NULL.
#[allow(clippy::option_option)]
pub(super) fn eval_closed(
    e: &SExpr,
    regexes: Vec<super::super::ir::ReSpec>,
) -> Option<Option<ScalarVal>> {
    use super::super::exec::{interp, Batch, OutCol};
    use super::super::ir::ColTy;
    let plan = super::super::plan::Plan {
        stages: vec![super::super::plan::Stage {
            joins: vec![],
            pred: None,
            project: vec![("v".to_string(), e.clone())],
        }],
    };
    let out = vec![Col {
        name: "v".to_string(),
        ty: ColTy {
            ty: e.ty,
            nullable: true,
        },
    }];
    let mut p = super::super::lower::lower(
        &plan, &[], &[], &[], out, regexes, &[], "fold", false, &[], &[],
    )
    .ok()?;
    super::super::ir::canonicalize(&mut p);
    let f = interp::compile(&p, vec![]).ok()?;
    let mut st = f.new_state();
    f.run(&Batch { rows: 1, cols: vec![] }, &mut st).ok()?;
    if st.emitted != 1 {
        return None;
    }
    Some(match &st.out[0] {
        OutCol::I1(v) => v[0].0.then(|| ScalarVal::I1(v[0].1)),
        OutCol::I64(v) => v[0].0.then(|| ScalarVal::I64(v[0].1)),
        OutCol::F64(v) => v[0].0.then(|| ScalarVal::F64(v[0].1)),
        OutCol::Str(v) => v[0].0.then(|| ScalarVal::Str(st.arena.get(v[0].1).to_string())),
        OutCol::Dec(v) => match e.ty {
            Ty::Dec(p, s) => v[0].0.then(|| ScalarVal::Dec(v[0].1, p, s)),
            _ => return None,
        },
    })
}


pub(super) fn math1_node(op: NumOp1, inner: SExpr) -> SExpr {
    let nullable = inner.nullable;
    SExpr {
        kind: SKind::MathF1 {
            op,
            a: Box::new(inner),
        },
        ty: Ty::F64,
        nullable,
    }
}

pub(super) fn is_flat_bitop(op: &BinaryOperator) -> bool {
    matches!(
        op,
        BinaryOperator::PGBitwiseShiftLeft
            | BinaryOperator::PGBitwiseShiftRight
            | BinaryOperator::BitwiseAnd
            | BinaryOperator::BitwiseOr
    )
}

pub(super) fn flat_bitop(op: &BinaryOperator) -> ArithOp {
    match op {
        BinaryOperator::PGBitwiseShiftLeft => ArithOp::Shl,
        BinaryOperator::PGBitwiseShiftRight => ArithOp::Shr,
        BinaryOperator::BitwiseAnd => ArithOp::BitAnd,
        _ => ArithOp::BitOr,
    }
}

/// In-order collect of a maximal run of flat-tier bit operators: yields the
/// operands and operators in SOURCE order regardless of how sqlparser
/// grouped them.
pub(super) fn flatten_bitops<'e>(
    e: &'e SqlExpr,
    ops: &mut Vec<&'e BinaryOperator>,
    operands: &mut Vec<&'e SqlExpr>,
) {
    match e {
        SqlExpr::BinaryOp { left, op, right } if is_flat_bitop(op) => {
            flatten_bitops(left, ops, operands);
            ops.push(op);
            flatten_bitops(right, ops, operands);
        }
        other => operands.push(other),
    }
}

/// AST constructors for the BETWEEN/IN desugars.
pub(super) fn ast_bin(op: BinaryOperator, l: SqlExpr, r: SqlExpr) -> SqlExpr {
    SqlExpr::BinaryOp {
        left: Box::new(l),
        op,
        right: Box::new(r),
    }
}

pub(super) fn ast_not_if(negated: bool, e: SqlExpr) -> SqlExpr {
    if negated {
        SqlExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr: Box::new(e),
        }
    } else {
        e
    }
}

/// DuckDB's default trim set (adversarial census, 1.5.5): exactly the
/// Unicode Zs space separators — NOT tab/newline/ZWSP/BOM/LS/PS/NEL.
pub(super) const ZS_SPACES: &str = "\u{20}\u{A0}\u{1680}\u{2000}\u{2001}\u{2002}\u{2003}\u{2004}\u{2005}\
                         \u{2006}\u{2007}\u{2008}\u{2009}\u{200A}\u{202F}\u{205F}\u{3000}";

pub(super) fn lit_str(s: &str) -> SExpr {
    SExpr {
        kind: SKind::Lit(Lit::Str(s.to_string())),
        ty: Ty::Str,
        nullable: false,
    }
}

pub(super) fn lit_i64(n: i64) -> SExpr {
    SExpr {
        kind: SKind::Lit(Lit::I64(n)),
        ty: Ty::I64,
        nullable: false,
    }
}

/// DuckDB coerces numeric values to BOOLEAN in conditional contexts —
/// WHERE, AND/OR/NOT operands, CASE WHEN conditions (measured 1.5.5:
/// nonzero -> true including NaN, 0 and -0.0 -> false, NULL -> NULL).
/// Strings stay a bind error (DuckDB errors at runtime; such queries never
/// mine into the corpus).
pub(super) fn bool_context(e: SExpr, what: &str) -> Result<SExpr, PrepareError> {
    match e.ty {
        Ty::I1 => Ok(e),
        t if t.is_int() || t == Ty::F64 => {
            if matches!(e.kind, SKind::NullOf) {
                return Ok(null_of(Ty::I1));
            }
            let nullable = e.nullable;
            Ok(SExpr {
                kind: SKind::Cast {
                    inner: Box::new(e),
                    trying: false,
                },
                ty: Ty::I1,
                nullable,
            })
        }
        other => Err(PrepareError::Bind(format!(
            "{what} must be BOOLEAN, got {}",
            other.name()
        ))),
    }
}

/// DuckDB's implicit VARCHAR coercion for concatenation: ints, floats and
/// bools all render through the same conversion CAST uses.
pub(super) fn to_varchar(e: SExpr) -> SExpr {
    if e.ty == Ty::Str {
        return e;
    }
    if matches!(e.kind, SKind::NullOf) {
        return null_of(Ty::Str);
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::Cast {
            inner: Box::new(e),
            trying: false,
        },
        ty: Ty::Str,
        nullable,
    }
}

/// A bind-fold result value as a literal of the declared type.
pub(super) fn scalar_lit(v: ScalarVal, ty: Ty) -> SExpr {
    let lit = match v {
        ScalarVal::I1(x) => Lit::I1(x),
        ScalarVal::I64(x) => Lit::I64(x),
        ScalarVal::F64(x) => Lit::F64(x),
        ScalarVal::Str(x) => Lit::Str(x),
        ScalarVal::Dec(x, p, s) => Lit::Dec(x, p, s),
    };
    SExpr {
        kind: SKind::Lit(lit),
        ty,
        nullable: false,
    }
}

pub(super) fn null_of(ty: Ty) -> SExpr {
    SExpr {
        kind: SKind::NullOf,
        ty,
        nullable: true,
    }
}

/// Checked-int32 evaluation of a LITERAL-shaped integer subtree, mirroring
/// DuckDB's typing: an int32-range number literal is INTEGER there, INTEGER
/// op INTEGER stays INTEGER, and overflow is a runtime ERROR. `NotShaped`
/// means some leaf isn't an int32 literal (column, cast, big literal, `/`
/// which is DOUBLE there) — 64-bit semantics apply and nothing refuses.
/// `Fine(None)` is a NULL-valued but trap-free subtree (INTEGER % 0 is NULL
/// on DuckDB, measured in the pins).
pub(super) enum I32Fold {
    NotShaped,
    Traps,
    Fine(Option<i32>),
}

pub(super) fn eval_i32_literal(e: &SqlExpr) -> I32Fold {
    use I32Fold::{Fine, NotShaped, Traps};
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => match text.parse::<i64>() {
                Ok(n) => match i32::try_from(n) {
                    Ok(n) => Fine(Some(n)),
                    Err(_) => NotShaped, // BIGINT literal there too
                },
                Err(_) => NotShaped,
            },
            _ => NotShaped,
        },
        SqlExpr::Nested(inner) => eval_i32_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => eval_i32_literal(expr),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match eval_i32_literal(expr) {
            Fine(Some(n)) => n.checked_neg().map_or(Traps, |n| Fine(Some(n))),
            other => other,
        },
        SqlExpr::BinaryOp { left, op, right } => eval_i32_binary(left, op, right),
        _ => NotShaped,
    }
}

/// [`eval_i32_literal`] of `left op right`, without building the node.
pub(super) fn eval_i32_binary(left: &SqlExpr, op: &BinaryOperator, right: &SqlExpr) -> I32Fold {
    use I32Fold::{Fine, NotShaped, Traps};
    let f = match op {
        BinaryOperator::Plus => i32::checked_add,
        BinaryOperator::Minus => i32::checked_sub,
        BinaryOperator::Multiply => i32::checked_mul,
        BinaryOperator::Modulo => i32::checked_rem,
        _ => return NotShaped,
    };
    match (eval_i32_literal(left), eval_i32_literal(right)) {
        (Fine(x), Fine(y)) => match (x, y) {
            (Some(_), Some(0)) if matches!(op, BinaryOperator::Modulo) => {
                Fine(None) // INTEGER % 0 is NULL, not a trap
            }
            (Some(x), Some(y)) => f(x, y).map_or(Traps, |n| Fine(Some(n))),
            _ => Fine(None), // NULL propagates trap-free
        },
        (Traps, _) | (_, Traps) => Traps,
        _ => NotShaped,
    }
}

/// The type a bare NULL adopts next to a typed operand.
pub(super) fn null_context_ty(op: &BinaryOperator, other: Ty) -> Ty {
    match op {
        BinaryOperator::And | BinaryOperator::Or => Ty::I1,
        _ => other,
    }
}

pub(super) fn cast_target(dt: &sqlparser::ast::DataType) -> Result<Ty, PrepareError> {
    let name = dt.to_string().to_uppercase();
    // DuckDB's spellings of the widths that have a lane, matched exactly
    // (INT8 is BIGINT: eight BYTES). Every other target -- HUGEINT, the
    // unsigned family, FLOAT/REAL (f32), DECIMAL/NUMERIC, INTERVAL, dates --
    // refuses: computing it in the nearest lane serves values DuckDB does
    // not (measured: CAST(-1 AS UINTEGER) errors there, CAST(16777217 AS
    // FLOAT) rounds to 16777216, CAST(1.25 AS DECIMAL(3,1)) is 1.3).
    let base = name.split('(').next().unwrap_or("").trim();
    Ok(match base {
        "TINYINT" | "INT1" => Ty::I8,
        "SMALLINT" | "INT2" | "SHORT" => Ty::I16,
        "INTEGER" | "INT" | "INT4" | "SIGNED" => Ty::I32,
        "BIGINT" | "INT8" | "LONG" => Ty::I64,
        "UTINYINT" | "UINT8" => Ty::U8,
        "USMALLINT" | "UINT16" => Ty::U16,
        "UINTEGER" | "UINT32" => Ty::U32,
        "DOUBLE" | "DOUBLE PRECISION" | "FLOAT8" => Ty::F64,
        "VARCHAR" | "TEXT" | "STRING" | "CHAR" | "CHARACTER" | "CHARACTER VARYING" | "BPCHAR" => {
            Ty::Str
        }
        "BOOLEAN" | "BOOL" | "LOGICAL" => Ty::I1,
        // DECIMAL(p,s); DECIMAL(p) is scale 0 and a bare DECIMAL is DuckDB's
        // default DECIMAL(18,3) (`LogicalType::DECIMAL` defaults).
        "DECIMAL" | "NUMERIC" | "DEC" => {
            let args: Vec<u32> = name
                .split_once('(')
                .map(|(_, rest)| {
                    rest.trim_end_matches(')')
                        .split(',')
                        .filter_map(|x| x.trim().parse().ok())
                        .collect()
                })
                .unwrap_or_default();
            let (p, sc) = match args[..] {
                [] => (18, 3),
                [p] => (p, 0),
                [p, sc] => (p, sc),
                _ => return Err(PrepareError::Bind(format!("bad DECIMAL type {name}"))),
            };
            if !(1..=38).contains(&p) {
                return Err(PrepareError::Bind("Width must be between 1 and 38!".into()));
            }
            if sc > p {
                return Err(PrepareError::Bind("Scale cannot be bigger than width".into()));
            }
            Ty::Dec(p as u8, sc as u8)
        }
        _ => {
            return Err(unsup(format!(
                "CAST target type {name} -- served targets are TINYINT, SMALLINT, \
                 INTEGER, BIGINT, DOUBLE, DECIMAL, VARCHAR and BOOLEAN"
            )))
        }
    })
}

/// DuckDB's typing of a numeric literal with a point and no exponent
/// (transform_constant.cpp, `T_PGFloat`): DECIMAL(digits, digits after
/// the point) when it has at most 38 digits, leading zeros included and
/// underscores not. `None` for anything else (DOUBLE or an integer).
fn decimal_literal(text: &str) -> Option<(Lit, Ty)> {
    if text.contains(['e', 'E']) {
        return None;
    }
    let (int, frac) = text.split_once('.')?;
    let int: String = int.chars().filter(|c| *c != '_').collect();
    let frac: String = frac.chars().filter(|c| *c != '_').collect();
    let (p, s) = (int.len() + frac.len(), frac.len());
    if p == 0 || p > 38 {
        return None;
    }
    let v: i128 = format!("{int}{frac}").parse().ok()?;
    Some((Lit::Dec(v, p as u8, s as u8), Ty::Dec(p as u8, s as u8)))
}

pub(super) fn literal(v: &SqlValue) -> Result<SExpr, PrepareError> {
    let (lit, ty) = match v {
        SqlValue::Number(text, _) => {
            if let Some(d) = decimal_literal(text) {
                d
            } else if text.contains('.') || text.to_ascii_lowercase().contains('e') {
                let f = text
                    .parse::<f64>()
                    .map_err(|_| PrepareError::Bind(format!("bad numeric literal '{text}'")))?;
                (Lit::F64(f), Ty::F64)
            } else {
                let i = text
                    .parse::<i64>()
                    .map_err(|_| PrepareError::Bind(format!("bad integer literal '{text}'")))?;
                // DuckDB types a bare integer literal by magnitude: INTEGER
                // when it fits (never narrower), else BIGINT. `-2147483648`
                // parses as -(2147483648) and stays BIGINT there too — that
                // falls out of unary minus binding, not of this rule.
                let ty = if i32::try_from(i).is_ok() { Ty::I32 } else { Ty::I64 };
                (Lit::I64(i), ty)
            }
        }
        SqlValue::SingleQuotedString(s) => (Lit::Str(s.clone()), Ty::Str),
        SqlValue::Boolean(b) => (Lit::I1(*b), Ty::I1),
        SqlValue::Null => unreachable!("NULL handled by expr_or_null"),
        other => return Err(unsup(format!("literal {other}"))),
    };
    Ok(SExpr {
        kind: SKind::Lit(lit),
        ty,
        nullable: false,
    })
}

/// The operator-table symbol for an [`ArithOp`]: the key into
/// `sig::OPS`, where the RESULT-TYPE rules live.
pub(super) fn arith_sym(op: ArithOp) -> &'static str {
    match op {
        ArithOp::Add => "+",
        ArithOp::Sub => "-",
        ArithOp::Mul => "*",
        ArithOp::Div => "/",
        ArithOp::IDiv => "//",
        ArithOp::Rem => "%",
        ArithOp::Shl => "<<",
        ArithOp::Shr => ">>",
        ArithOp::BitAnd => "&",
        ArithOp::BitOr => "|",
        ArithOp::BitXor => "xor",
    }
}

/// The operator-table symbol for a [`CmpPred`].
pub(super) fn cmp_sym(pred: CmpPred) -> &'static str {
    match pred {
        CmpPred::Eq => "=",
        CmpPred::Ne => "<>",
        CmpPred::Lt => "<",
        CmpPred::Le => "<=",
        CmpPred::Gt => ">",
        CmpPred::Ge => ">=",
    }
}

/// DuckDB numeric promotion. The result-type RULE is the operator's
/// `sig::OPS` row — `/` is Fixed(F64), everything else Widens across the
/// integer width lattice; this function is the rule's
/// consumer and owns the promotion nodes.
pub(super) fn numeric_promote(
    op: ArithOp,
    a: SExpr,
    b: SExpr,
    lits: (Option<i64>, Option<i64>),
) -> Result<(SExpr, SExpr, Ty), PrepareError> {
    let numeric = |e: &SExpr| e.ty.is_int() || e.ty == Ty::F64;
    if !numeric(&a) || !numeric(&b) {
        return Err(PrepareError::Bind(format!(
            "arithmetic needs numeric operands, got {} and {}",
            a.ty.name(),
            b.ty.name()
        )));
    }
    let ty = match sig::op_ret(arith_sym(op)) {
        Ret::Fixed(t) => t,
        Ret::Widen => {
            if a.ty == Ty::F64 || b.ty == Ty::F64 {
                Ty::F64
            } else {
                int_width_promote(a.ty, lits.0, b.ty, lits.1)
            }
        }
        Ret::Arg(_) | Ret::Unify => unreachable!("not an operator rule"),
    };
    if ty == Ty::F64 {
        // promote_f64 is identity on an already-F64 operand.
        Ok((promote_f64(a), promote_f64(b), Ty::F64))
    } else {
        Ok((a, b, ty))
    }
}

pub(super) fn width_rank(t: Ty) -> u8 {
    match t {
        Ty::I8 | Ty::U8 => 0,
        Ty::I16 | Ty::U16 => 1,
        Ty::I32 | Ty::U32 => 2,
        _ => 3,
    }
}

/// The literal half of DuckDB's integer promotion (measured): a syntactic
/// literal whose VALUE fits a non-literal side takes that side's type, so
/// `c8 + 127` is TINYINT and `u8 + 1` UTINYINT, while `c8 + 128` and
/// `u8 + (-1)` keep the literal's INTEGER. `None` when it does not apply.
fn literal_fit(a_ty: Ty, a_lit: Option<i64>, b_ty: Ty, b_lit: Option<i64>) -> Option<Ty> {
    match (a_lit, b_lit) {
        (Some(v), None) if fits_width(b_ty, v) && width_rank(a_ty) >= width_rank(b_ty) => {
            Some(b_ty)
        }
        (None, Some(v)) if fits_width(a_ty, v) && width_rank(b_ty) >= width_rank(a_ty) => {
            Some(a_ty)
        }
        _ => None,
    }
}

/// One width-combine step of DuckDB's integer promotion for an OPERATOR
/// (`+ - * // %` and the bitwise family), measured on 1.5.5: equal types
/// keep; a literal that fits the other side narrows to it
/// ([`literal_fit`]); signed with signed, the wider; unsigned with
/// unsigned, the wider; signed with unsigned, the signed side when it is
/// strictly wider, else BIGINT (`u8 + i8` and `u16 + i16` are BIGINT,
/// `u8 + i16` SMALLINT).
pub(super) fn int_width_promote(a_ty: Ty, a_lit: Option<i64>, b_ty: Ty, b_lit: Option<i64>) -> Ty {
    if a_ty == b_ty {
        return a_ty;
    }
    if let Some(t) = literal_fit(a_ty, a_lit, b_ty, b_lit) {
        return t;
    }
    match (a_ty.is_unsigned(), b_ty.is_unsigned()) {
        (false, false) | (true, true) => {
            if width_rank(a_ty) >= width_rank(b_ty) {
                a_ty
            } else {
                b_ty
            }
        }
        (s_u, _) => {
            let (signed, unsigned) = if s_u { (b_ty, a_ty) } else { (a_ty, b_ty) };
            if width_rank(signed) > width_rank(unsigned) {
                signed
            } else {
                Ty::I64
            }
        }
    }
}

/// The same step for a FAMILY (CASE / COALESCE / greatest / least):
/// identical but for signed with unsigned, which takes the smallest signed
/// type holding both (`coalesce(u8, i8)` is SMALLINT, `coalesce(u16, i8)`
/// INTEGER, `coalesce(u32, i8)` BIGINT; measured).
pub(super) fn int_family_promote(a_ty: Ty, a_lit: Option<i64>, b_ty: Ty, b_lit: Option<i64>) -> Ty {
    if a_ty == b_ty || a_ty.is_unsigned() == b_ty.is_unsigned() {
        return int_width_promote(a_ty, a_lit, b_ty, b_lit);
    }
    if let Some(t) = literal_fit(a_ty, a_lit, b_ty, b_lit) {
        return t;
    }
    let (signed, unsigned) = if a_ty.is_unsigned() { (b_ty, a_ty) } else { (a_ty, b_ty) };
    let bits = signed
        .int_bits()
        .unwrap_or(64)
        .max(2 * unsigned.int_bits().unwrap_or(64));
    match bits {
        8 => Ty::I8,
        16 => Ty::I16,
        32 => Ty::I32,
        _ => Ty::I64,
    }
}

/// Whether the SPELLING is a DECIMAL literal (a dot, no exponent —
/// `2.5`, `-2.681`; `1.5e0` is DOUBLE), optionally under parens or unary
/// minus. DuckDB folds a strict op over one of these and a bare NULL to
/// SQLNULL (INTEGER), discarding the decimal entirely.
pub(super) fn ast_decimal_literal(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => {
                text.contains('.') && !text.to_ascii_lowercase().contains('e')
            }
            _ => false,
        },
        SqlExpr::Nested(inner) => ast_decimal_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => ast_decimal_literal(expr),
        // DuckDB types a CASE over DECIMAL arms DECIMAL, so the
        // SQLNULL collapse follows it through -- every RESULT arm (ELSE
        // included, when present) must be decimal-spelled.
        SqlExpr::Case {
            conditions,
            else_result,
            ..
        } => {
            !conditions.is_empty()
                && conditions.iter().all(|w| ast_decimal_literal(&w.result))
                && else_result
                    .as_ref()
                    .is_none_or(|e| ast_decimal_literal(e))
        }
        _ => false,
    }
}

/// Whether DuckDB types `e` DECIMAL: a decimal-spelled literal (see
/// [`ast_decimal_literal`]), or `+`/`-`/`*` over decimal-typed operands and
/// integer literals with at least one decimal side.
pub(super) fn ast_decimal_typed(e: &SqlExpr) -> bool {
    match e {
        SqlExpr::Nested(inner) => ast_decimal_typed(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => ast_decimal_typed(expr),
        SqlExpr::BinaryOp { left, op, right }
            if matches!(
                op,
                BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Multiply
            ) =>
        {
            let side = |x: &SqlExpr| ast_decimal_typed(x) || ast_int_literal(x).is_some();
            (ast_decimal_typed(left) || ast_decimal_typed(right)) && side(left) && side(right)
        }
        e => ast_decimal_literal(e),
    }
}

/// The value of a SYNTACTIC integer literal, from the SQL AST: a bare
/// Number, optionally under parentheses or unary MINUS. Never unary plus
/// (DuckDB's `+` is a real function that erases literal-ness), never a
/// function call, never a cast, never anything bound — every SExpr-shape
/// heuristic leaks (verbatim
/// family returns, `0 - N` user spellings, retyped degenerations), so the
/// hint comes from the spelling alone. This is DuckDB's own notion for
/// its value-fits promotion.
pub(super) fn ast_int_literal(e: &SqlExpr) -> Option<i64> {
    match e {
        SqlExpr::Value(v) => match &v.value {
            SqlValue::Number(text, _) => text.parse::<i64>().ok(),
            _ => None,
        },
        SqlExpr::Nested(inner) => ast_int_literal(inner),
        SqlExpr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => ast_int_literal(expr).and_then(i64::checked_neg),
        _ => None,
    }
}

/// Whether `v` is representable at width `t` (always true for lane types).
pub(super) fn fits_width(t: Ty, v: i64) -> bool {
    t.int_range().map_or(true, |(lo, hi)| (lo..=hi).contains(&v))
}

/// DuckDB's name for an integer width, for refusal messages.
pub(super) fn duck_int_name(t: Ty) -> &'static str {
    match t {
        Ty::I8 => "TINYINT",
        Ty::I16 => "SMALLINT",
        Ty::I32 => "INTEGER",
        Ty::U8 => "UTINYINT",
        Ty::U16 => "USMALLINT",
        Ty::U32 => "UINTEGER",
        Ty::F64 => "DOUBLE",
        _ => "BIGINT",
    }
}

/// Whether DuckDB's binder would fold `e` to NULL: a foldable argument that
/// evaluates to NULL turns a default-NULL-handling call into a NULL of its
/// return type at bind, so its other arguments never run.
pub(super) fn folds_to_null(e: &SExpr) -> bool {
    bind_foldable(e) && matches!(bind_fold(e.clone()).kind, SKind::NullOf)
}

/// A BOOLEAN as the INTEGER DuckDB casts it to for a comparison: 0 or 1,
/// NULL staying NULL. A typed NULL retypes, like `promote_f64`.
/// DuckDB's VARCHAR -> BOOLEAN parse (measured): true/t/1/yes/y and
/// false/f/0/no/n, ASCII case-insensitive, no trimming.
pub(super) fn duck_stob(s: &str) -> Option<bool> {
    match s.to_ascii_lowercase().as_str() {
        "true" | "t" | "1" | "yes" | "y" => Some(true),
        "false" | "f" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

pub(super) fn bool_to_int(e: SExpr) -> SExpr {
    if matches!(e.kind, SKind::NullOf) {
        return SExpr {
            kind: SKind::NullOf,
            ty: Ty::I64,
            nullable: true,
        };
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::Cast {
            inner: Box::new(e),
            trying: false,
        },
        ty: Ty::I64,
        nullable,
    }
}

/// DuckDB's name for a type, for bind-error messages.
pub(super) fn duck_ty_name(t: Ty) -> String {
    match t {
        Ty::I1 => "BOOLEAN".into(),
        Ty::Str => "VARCHAR".into(),
        Ty::Dec(p, s) => format!("DECIMAL({p},{s})"),
        t => duck_int_name(t).into(),
    }
}

/// Widen an integer arm to the unified width, the way DuckDB does: it casts
/// the arm's FINISHED result and never re-types the arm's operands. Setting
/// `ty` on a computed node would move its own range check to the wider width
/// (`c0 * c0` over TINYINT would stop overflowing at 127), so a computed arm
/// is wrapped in a width-only cast; a literal or typed NULL, which carries no
/// check, is re-typed in place.
pub(super) fn widen_int(e: SExpr, to: Ty) -> SExpr {
    if e.ty == to {
        return e;
    }
    if matches!(e.kind, SKind::NullOf | SKind::Lit(_)) {
        return SExpr { ty: to, ..e };
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::Cast {
            inner: Box::new(e),
            trying: false,
        },
        ty: to,
        nullable,
    }
}

pub(super) fn promote_f64(e: SExpr) -> SExpr {
    if e.ty == Ty::F64 {
        return e;
    }
    // A typed NULL promotes by retyping — no conversion node needed.
    if matches!(e.kind, SKind::NullOf) {
        return SExpr {
            kind: SKind::NullOf,
            ty: Ty::F64,
            nullable: true,
        };
    }
    let nullable = e.nullable;
    SExpr {
        kind: SKind::IntToFloat(Box::new(e)),
        ty: Ty::F64,
        nullable,
    }
}

/// Turn a just-built `promote_f64` node into the f32-narrowing one, so an
/// integer `tree_predict` feature rounds ONCE the way sklearn does.
/// Anything else — an f64 expression, a typed NULL, a folded
/// literal — is already on the grid or has no integer to narrow, and passes
/// through untouched.
pub(super) fn narrow_f32(e: SExpr) -> SExpr {
    match e.kind {
        SKind::IntToFloat(inner) => SExpr {
            kind: SKind::IntToFloat32(inner),
            ..e
        },
        _ => e,
    }
}

impl Binder<'_> {
    /// DuckDB unifies BETWEEN/IN across the WHOLE construct (pins-wave1/):
    /// one common type for the subject and every bound/element, so a single
    /// f64 side promotes all sides. Numeric-with-string/bool mixing has
    /// exec-time cast semantics we don't model — clean-unsupported.
    pub(super) fn unify_family(&self, exprs: &[&SqlExpr]) -> Result<Vec<SqlExpr>, PrepareError> {
        let (mut any_f64, mut any_num) = (false, false);
        // A DECIMAL member unifies the family at the common DECIMAL
        // (docs/specs/decimal-expressions.md §7). Each comparison below
        // meets its own pair's common type, which reads the same values
        // exactly as the family-wide one does, EXCEPT at a family capped at
        // 38 digits (scale truncated), which refuses by name. A DOUBLE
        // family is cast to DOUBLE whole, below.
        let mut any_dec: Option<SExpr> = None;
        let mut family: Option<Ty> = None;
        for e in exprs {
            if let Some(b) = self.expr_or_null(e)? {
                if b.ty.is_int() || b.ty == Ty::F64 || b.ty.dec().is_some() {
                    family = Some(match family {
                        None => b.ty,
                        Some(f) if f == b.ty => f,
                        Some(f) if f.is_int() && b.ty.is_int() => Ty::I64,
                        Some(f) => match dec_common(f, b.ty) {
                            Some(t) => t,
                            None => Ty::F64,
                        },
                    });
                }
                match b.ty {
                    Ty::F64 => (any_f64, any_num) = (true, true),
                    Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 => any_num = true,
                    Ty::Dec(..) => any_dec = Some(b.clone()),
                    Ty::Str | Ty::I1 => {}
                }
            }
        }
        if let Some(d) = any_dec {
            let capped = |t: Option<Ty>| {
                exprs.iter().any(|e| {
                    self.expr_or_null(e).ok().flatten().is_some_and(|b| {
                        matches!((b.ty, t), (Ty::Dec(p, s), Some(Ty::Dec(_, fs))) if fs < s || p - s > 38)
                    })
                })
            };
            // A DOUBLE family casts every member to DOUBLE below, decimals
            // included, which is DuckDB's own reading.
            if family != Some(Ty::F64) && capped(family) {
                return Err(self.dec_refusal("BETWEEN/IN unification at 38 digits", &d));
            }
            // A string or boolean member would convert to the DECIMAL
            // (DuckDB's VARCHAR -> DECIMAL parse), which is not served.
            if !any_f64 {
                for e in exprs {
                    if let Some(b) = self.expr_or_null(e)? {
                        if matches!(b.ty, Ty::Str | Ty::I1) {
                            return Err(self.dec_refusal(
                                "BETWEEN/IN mixing a string or boolean with",
                                &d,
                            ));
                        }
                    }
                }
            }
        }
        // pins-wave5/: mixing casts the string/bool side to the NUMERIC
        // side (strings numerically with half-away-from-zero rounding to
        // ints; bool -> 0/1; non-numeric strings are DuckDB Conversion
        // Errors). Only literals convert at bind — a string/bool COLUMN
        // would need runtime cast traps and stays unsupported.
        let mut owned: Vec<SqlExpr> = Vec::with_capacity(exprs.len());
        for e in exprs {
            let b = self.expr_or_null(e)?;
            let needs_cast =
                any_num && b.as_ref().is_some_and(|b| matches!(b.ty, Ty::Str | Ty::I1));
            if !needs_cast {
                owned.push((*e).clone());
                continue;
            }
            let lit = match b.map(|b| b.kind) {
                Some(SKind::Lit(Lit::Str(s))) => {
                    // Non-numeric strings convert (and fail) at EXECUTION
                    // time in DuckDB — an empty input succeeds — so a
                    // bind-time error would be over-eager; stay clean.
                    let Ok(x) = s.trim().parse::<f64>() else {
                        return Err(unsup(
                            "BETWEEN/IN mixing non-numeric string literals with numbers \
                             (exec-time conversion)",
                        ));
                    };
                    if any_f64 {
                        s.trim().to_string()
                    } else {
                        let r = if x >= 0.0 {
                            (x + 0.5).floor()
                        } else {
                            (x - 0.5).ceil()
                        };
                        if r < i64::MIN as f64 || r > i64::MAX as f64 {
                            return Err(unsup(
                                "BETWEEN/IN mixing out-of-range string literals with numbers \
                                 (exec-time conversion)",
                            ));
                        }
                        format!("{}", r as i64)
                    }
                }
                Some(SKind::Lit(Lit::I1(v))) => format!("{}", v as i64),
                _ => {
                    return Err(unsup(
                        "BETWEEN/IN mixing strings or booleans with numbers \
                         (non-literal side needs exec-time cast semantics)",
                    ))
                }
            };
            owned.push(SqlExpr::Value(sqlparser::ast::ValueWithSpan {
                value: SqlValue::Number(lit, false),
                span: sqlparser::tokenizer::Span::empty(),
            }));
        }
        Ok(owned
            .into_iter()
            .map(|e| {
                if any_f64 {
                    // CAST is a no-op on already-f64 sides and types NULL
                    // literals from context; exactly DuckDB's unification.
                    SqlExpr::Cast {
                        kind: CastKind::Cast,
                        expr: Box::new(e),
                        data_type: sqlparser::ast::DataType::Double(
                            sqlparser::ast::ExactNumberInfo::None,
                        ),
                        format: None,
                        array: false,
                    }
                } else {
                    e
                }
            })
            .collect())
    }
}
