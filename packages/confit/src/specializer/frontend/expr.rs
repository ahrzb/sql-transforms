//! Expression binding: the dispatcher and the operator families
//! (comparison, arithmetic, CASE, CAST, IS NULL).

use super::*;

/// DuckDB's `max_expression_depth` default: its parser refuses an
/// expression nested deeper (a 999-term `+` chain already is). Binding
/// recurses per level, so past this the native stack is at risk too.
const MAX_EXPRESSION_DEPTH: u32 = 1000;

thread_local! {
    static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// The deepest level reached since [`measure_depth`] last reset it.
    static PEAK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn too_deep() -> PrepareError {
    PrepareError::Parse(format!(
        "Max expression depth limit of {MAX_EXPRESSION_DEPTH} exceeded"
    ))
}

/// One level of [`Binder::expr_or_null`]'s recursion, released on drop.
struct DepthGuard;

impl DepthGuard {
    fn enter() -> Result<Self, PrepareError> {
        let d = DEPTH.with(|c| {
            c.set(c.get() + 1);
            c.get()
        });
        PEAK.with(|p| p.set(p.get().max(d)));
        let g = DepthGuard;
        if d > MAX_EXPRESSION_DEPTH {
            return Err(too_deep());
        }
        Ok(g)
    }
}

/// `bind`, and how many levels below the current one it reached.
pub(super) fn measure_depth<T>(bind: impl FnOnce() -> T) -> (T, u32) {
    let base = DEPTH.with(|c| c.get());
    let saved = PEAK.with(|p| p.replace(base));
    let out = bind();
    let peak = PEAK.with(|p| p.get());
    PEAK.with(|p| p.set(saved.max(peak)));
    (out, peak.saturating_sub(base))
}

/// Refuse as a binding `height` levels below the current one would.
pub(super) fn check_depth(height: u32) -> Result<(), PrepareError> {
    let d = DEPTH.with(|c| c.get()).saturating_add(height);
    PEAK.with(|p| p.set(p.get().max(d)));
    if d > MAX_EXPRESSION_DEPTH {
        return Err(too_deep());
    }
    Ok(())
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|c| c.set(c.get() - 1));
    }
}

/// Chains longer than this rebalance (see [`rebalance_chain`]); shorter
/// ones bind exactly as written.
const LONG_CHAIN: usize = 64;

/// A left-deep `a AND b AND c ...` (or OR) chain of more than
/// [`LONG_CHAIN`] terms, rebuilt as a balanced tree of the same terms in the
/// same order. AND and OR are associative under three-valued logic, and
/// evaluating left to right until a decisive operand stops at the same
/// operand in either shape, so the meaning is unchanged; the depth drops
/// from n to log n, which every later recursive pass needs (DuckDB builds
/// these n-ary and serves 20000 terms). A balanced tree's own spine is
/// shorter than the threshold, so this applies once.
fn rebalance_chain(e: &SqlExpr) -> Option<SqlExpr> {
    let SqlExpr::BinaryOp { op, .. } = e else {
        return None;
    };
    if !matches!(op, BinaryOperator::And | BinaryOperator::Or) {
        return None;
    }
    let mut terms = Vec::new();
    let mut cur = e;
    while let SqlExpr::BinaryOp { left, op: o, right } = cur {
        if o != op {
            break;
        }
        terms.push(right.as_ref());
        cur = left;
    }
    terms.push(cur);
    if terms.len() <= LONG_CHAIN {
        return None;
    }
    terms.reverse();
    fn build(terms: &[&SqlExpr], op: &BinaryOperator) -> SqlExpr {
        match terms {
            [one] => (*one).clone(),
            _ => {
                let (l, r) = terms.split_at(terms.len() / 2);
                SqlExpr::BinaryOp {
                    left: Box::new(build(l, op)),
                    op: op.clone(),
                    right: Box::new(build(r, op)),
                }
            }
        }
    }
    Some(build(&terms, op))
}

/// `inner IS NULL`.
fn is_null_test(inner: SExpr) -> SExpr {
    SExpr {
        kind: SKind::IsNull {
            negated: false,
            inner: Box::new(inner),
        },
        ty: Ty::I1,
        nullable: false,
    }
}

/// A NULL of `to` that evaluates `read`, which reads a column: an SQLNULL
/// DuckDB cannot fold, where it keeps a type (see `Binder::typed_null`).
fn null_reading(read: SExpr, to: Ty) -> SExpr {
    SExpr {
        kind: SKind::Case {
            arms: vec![(is_null_test(read), null_of(to))],
            default: None,
        },
        ty: to,
        nullable: true,
    }
}

fn unfoldable_null_refusal() -> PrepareError {
    unsup(
        "a NULL that reads a column, where DuckDB keeps it a typed value that \
         does not fold (under a CAST, a comparison, CASE, COALESCE, a list or \
         a UDF argument)",
    )
}

impl Binder<'_> {
    /// Bind an expression that must have a definite type on its own. A bare
    /// NULL with no adopting context takes DuckDB's SQLNULL default:
    /// INTEGER.
    pub(super) fn expr(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        Ok(self.expr_or_null(e)?.unwrap_or_else(|| null_of(Ty::I32)))
    }

    /// Like `expr`, but a bare NULL literal comes back as `None` for the
    /// caller to type from context.
    pub(super) fn expr_or_null(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        // Binding recurses per level of the tree, and a level's frames are
        // large (tens of KiB unoptimized): grow the stack on the heap rather
        // than overflow it.
        stacker::maybe_grow(RED_ZONE, STACK_SEGMENT, || self.expr_or_null_here(e))
    }

    fn expr_or_null_here(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        // A let read stands where its text would: it is not a level.
        if let Some((id, lets)) = lets::marker_let(e) {
            return self.let_read(id, &lets[id]);
        }
        if let Some(balanced) = rebalance_chain(e) {
            return self.expr_or_null_here(&balanced);
        }
        // DuckDB's parser builds AND/OR chains n-ary, so they do not count
        // toward its depth limit (a 20000-term AND serves there).
        let _depth = match e {
            SqlExpr::BinaryOp {
                op: BinaryOperator::And | BinaryOperator::Or,
                ..
            } => None,
            _ => Some(DepthGuard::enter()?),
        };
        match e {
            SqlExpr::Value(v) if matches!(v.value, SqlValue::Null) => Ok(None),
            // A bare NULL column of the level below is DuckDB's SQLNULL, as
            // a NULL literal is (see `BoundQuery::null_cols`); a whole item
            // keeps it a column.
            SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)
                if !self.whole_item.get() && self.null_col_ref(e).is_some() =>
            {
                Ok(None)
            }
            SqlExpr::Nested(inner) => self.expr_or_null(inner),
            SqlExpr::Function(f) => {
                if f.name.to_string().eq_ignore_ascii_case(structs::SEQ_MARKER) {
                    return self.seq(f);
                }
                // A call marker outside a field read: its expansion.
                if let Some((id, calls)) = calls::marker_call(e) {
                    return self.expr_or_null(&calls[id].1);
                }
                if self.nullif_sqlnull(f)? {
                    return Ok(None);
                }
                // struct_extract over struct_pack desugars HERE too, so a
                // bare-NULL field keeps its adopting context.
                if let Some(sub) = self.desugar_struct_extract(f)? {
                    return self.expr_or_null(&sub);
                }
                self.bind_or_sqlnull(e)
            }
            other => {
                if let Some(read) = self.call_field(other)? {
                    return Ok(read);
                }
                if let Some(sub) = self.desugar_struct_field(other)? {
                    return self.expr_or_null(&sub);
                }
                self.bind_or_sqlnull(other)
            }
        }
    }

    /// Bind `e`; if the result is DuckDB's SQLNULL surface — the ||
    /// collapse, or a pure-udf field fold to whole-call NULL —
    /// answer the ADOPTABLE channel instead: `- ((udf(1, NULL)).f1)` is
    /// BIGINT on DuckDB, `abs(s || NULL)` BIGINT too, because SQLNULL
    /// re-promotes by signature. The gate is the SPELLING, not the bound
    /// type: a builtin's whole-call NULL is a COMMITTED int32 there
    /// (measured: `-(ascii(NULL))` is INTEGER and `upper(ascii(NULL))` a
    /// binder error), and within the gated shapes NullOf(int32) has no
    /// other producer (a udf cannot declare an int32 field; || otherwise
    /// types Str).
    pub(super) fn bind_or_sqlnull(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        let bound = self.bind(e)?;
        let sqlnull = matches!(bound.kind, SKind::NullOf)
            && bound.ty == Ty::I32
            && self.sqlnull_capable_shape(e);
        Ok((!sqlnull).then_some(bound))
    }

    /// The spellings whose bind may yield the SQLNULL surface: `||`, and
    /// field access over a DECLARED udf (dot form and struct_extract).
    pub(super) fn sqlnull_capable_shape(&self, e: &SqlExpr) -> bool {
        let over_udf = |root: &SqlExpr| {
            let mut b = root;
            while let SqlExpr::Nested(i) = b {
                b = i;
            }
            matches!(b, SqlExpr::Function(f) if self.find_udf(&f.name.to_string()).is_some())
        };
        match e {
            SqlExpr::BinaryOp {
                op: BinaryOperator::StringConcat,
                ..
            } => true,
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                matches!(access_chain.as_slice(), [AccessExpr::Dot(_)]) && over_udf(root)
            }
            SqlExpr::Function(f) => {
                f.name.to_string().eq_ignore_ascii_case("struct_extract")
                    || self.all_null_spelling(e)
            }
            SqlExpr::Case { .. } => self.all_null_spelling(e),
            // A DECIMAL operator over a constant NULL operand, typed or bare,
            // and unary minus over a NULL DECIMAL, bind SQLNULL (measured:
            // `- (NULL * 2.5)` is BIGINT, `coalesce(1.75 + 41.7, NULL -
            // 1.75)` DECIMAL(6,3)).
            SqlExpr::BinaryOp {
                left,
                op:
                    BinaryOperator::Plus
                    | BinaryOperator::Minus
                    | BinaryOperator::Multiply
                    | BinaryOperator::Modulo,
                right,
            } => self.dec_null_operator(&[left, right]),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => self.dec_null_operator(&[expr]),
            _ => false,
        }
    }

    /// Whether `e` is spelled as NULL all the way down: a NULL literal,
    /// `nullif(NULL, x)`, or a CASE / COALESCE / IFNULL / least / greatest
    /// whose every result or argument is such a spelling. DuckDB types each
    /// of these as a bare NULL (measured under unary minus, abs, upper, ||,
    /// concat, coalesce and CASE unification).
    pub(super) fn all_null_spelling(&self, e: &SqlExpr) -> bool {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        match e {
            SqlExpr::Value(v) => matches!(v.value, SqlValue::Null),
            SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => {
                self.null_col_ref(e).is_some()
            }
            SqlExpr::Nested(i) => self.all_null_spelling(i),
            SqlExpr::Case {
                conditions,
                else_result,
                ..
            } => {
                conditions.iter().all(|w| self.all_null_spelling(&w.result))
                    && else_result.as_deref().is_none_or(|x| self.all_null_spelling(x))
            }
            SqlExpr::Function(f) => {
                if self.nullif_sqlnull(f).unwrap_or(false) {
                    return true;
                }
                let name = f.name.to_string().to_ascii_lowercase();
                if !matches!(name.as_str(), "coalesce" | "ifnull" | "least" | "greatest")
                    || f.over.is_some()
                    || f.filter.is_some()
                {
                    return false;
                }
                let FunctionArguments::List(list) = &f.args else {
                    return false;
                };
                !list.args.is_empty()
                    && list.args.iter().all(|a| match a {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(x)) => {
                            self.all_null_spelling(x)
                        }
                        _ => false,
                    })
            }
            _ => false,
        }
    }

    /// Whether these operands bind a DECIMAL operator (one is a DECIMAL,
    /// the rest DECIMALs, integers or bare NULLs) and one of them is a
    /// constant NULL. A DOUBLE operand makes it the DOUBLE operator, which
    /// keeps its type.
    fn dec_null_operator(&self, operands: &[&SqlExpr]) -> bool {
        let bound: Vec<Option<SExpr>> = match operands
            .iter()
            .map(|e| self.expr_or_null(e))
            .collect::<Result<_, _>>()
        {
            Ok(b) => b,
            Err(_) => return false,
        };
        let any_dec = bound.iter().flatten().any(|e| e.ty.dec().is_some());
        let all_dec_or_int = bound
            .iter()
            .flatten()
            .all(|e| e.ty.dec().is_some() || e.ty.is_integer());
        let any_null = bound.iter().any(|e| {
            e.as_ref()
                .is_none_or(folds_to_null)
        });
        any_dec && all_dec_or_int && any_null
    }

    /// Whether `f` is `nullif(NULL, x)` — which propagates DuckDB's
    /// SQLNULL: the whole call is an ADOPTABLE bare NULL, not a committed
    /// int32 (`- nullif(NULL, 1)` is BIGINT there, `nullif(NULL, 1) *
    /// 1::SMALLINT` SMALLINT). The second argument still
    /// binds so its own errors fire. Builtin names cannot be UDF-shadowed
    /// (see [`BUILTIN_NAMES`]), so the name test is enough.
    pub(super) fn nullif_sqlnull(
        &self,
        f: &sqlparser::ast::Function,
    ) -> Result<bool, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if !f.name.to_string().eq_ignore_ascii_case("nullif")
            || f.uses_odbc_syntax
            || !matches!(f.parameters, FunctionArguments::None)
            || f.filter.is_some()
            || f.null_treatment.is_some()
            || f.over.is_some()
            || !f.within_group.is_empty()
        {
            return Ok(false);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(false);
        };
        if !list.clauses.is_empty() || list.duplicate_treatment.is_some() {
            return Ok(false);
        }
        let plain: Vec<&SqlExpr> = list
            .args
            .iter()
            .filter_map(|a| match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Some(e),
                _ => None,
            })
            .collect();
        if plain.len() != 2 || plain.len() != list.args.len() {
            return Ok(false);
        }
        if self.expr_or_null(plain[0])?.is_some() {
            return Ok(false);
        }
        // DuckDB evaluates the second operand, and it can trap there
        // (`nullif(NULL, x + x)` overflows on optimizer-off DuckDB; nightly
        // seeds 3058298, 3722953, campaign seed 992226). The bare NULL has
        // no node that evaluates an operand for its trap only, so such an
        // operand refuses by name.
        if self.expr_or_null(plain[1])?.as_ref().is_some_and(can_trap) {
            return Err(unsup(
                "nullif(NULL, x) where x can trap (DuckDB evaluates x; a bare NULL cannot)",
            ));
        }
        Ok(true)
    }

    pub(super) fn bind(&self, e: &SqlExpr) -> Result<SExpr, PrepareError> {
        match e {
            SqlExpr::Identifier(ident) => {
                reserved_head(ident)?;
                self.column(&ident.value)
            }
            SqlExpr::CompoundIdentifier(parts) => {
                reserved_head(&parts[0])?;
                self.compound(parts)
            }
            SqlExpr::Nested(inner) => self.expr(inner),
            SqlExpr::Value(v) => literal(&v.value),
            // DuckDB puts << >> & | in ONE flat left-associative tier;
            // sqlparser tiers them (& above << >> above |), so 4|1&1 would
            // silently parse as 4|(1&1)=5 where DuckDB computes (4|1)&1=1.
            // Re-associate: in-order traversal of the parsed run recovers
            // source order, then left-fold. User parens are Nested nodes,
            // which the flatten treats as leaves.
            // ~ / !~ are FULL regex match in DuckDB (measured: the binder
            // error names regexp_full_match — NOT the Postgres search).
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::PGRegexMatch,
                right,
            } => self.regex_full_predicate("~", left, right, false),
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::PGRegexNotMatch,
                right,
            } => self.regex_full_predicate("!~", left, right, true),
            SqlExpr::BinaryOp { op, .. } if is_flat_bitop(op) => {
                let (mut ops, mut operands) = (Vec::new(), Vec::new());
                flatten_bitops(e, &mut ops, &mut operands);
                let mut acc = self.expr_or_null(operands[0])?;
                // An integer literal adapts to its partner's width, as for
                // `+` (`tiny >> 8` is TINYINT); only the run's first operand
                // can be a literal on the left.
                let mut left_lit = ast_int_literal(operands[0]);
                for (o, rhs) in ops.iter().zip(&operands[1..]) {
                    let b = self.expr_or_null(rhs)?;
                    let (av, bv) = match (acc, b) {
                        (Some(x), Some(y)) => (x, y),
                        (Some(x), None) => {
                            let n = null_of(x.ty);
                            (x, n)
                        }
                        (None, Some(y)) => {
                            let n = null_of(y.ty);
                            (n, y)
                        }
                        (None, None) => {
                            return Err(unsup("NULL <op> NULL without a typing context"))
                        }
                    };
                    let lits = (left_lit.take(), ast_int_literal(rhs));
                    acc = Some(self.arith(flat_bitop(o), av, bv, lits)?);
                }
                Ok(acc.expect("a flat-bitop run has at least one operator"))
            }
            SqlExpr::BinaryOp { left, op, right } => self.binary(op, left, right),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => {
                // DuckDB's grammar folds a minus into an integer literal too
                // big for INTEGER (it lexes as a numeric string), parens or
                // spaces between notwithstanding, as many minuses as there
                // are: `-9223372036854775808` is the BIGINT i64::MIN, and
                // `- -9223372036854775808` is 9223372036854775808 again,
                // HUGEINT. Below i64's edge folding and computing agree, so
                // only a literal past it is folded here.
                if let Some((minuses, text)) = ast_signed_number(e) {
                    let integral = text.bytes().all(|c| c.is_ascii_digit());
                    if integral && text.parse::<i64>().is_err() {
                        let signed = if minuses % 2 == 1 {
                            format!("-{text}")
                        } else {
                            text
                        };
                        return integer_text_literal(&signed);
                    }
                }
                // A DOUBLE negates with Fneg, a sign-bit flip; no
                // subtraction reproduces it (`NumOp1` in ir/mod.rs).
                // Integers keep 0 - x and its i64::MIN trap, which is
                // DuckDB's own overflow behaviour.
                // sqlparser parses `-a % b` as `-(a % b)`; DuckDB binds
                // `(-a) % b` (its unary minus is tighter than mul/div/mod).
                // The minus distributes over these ops so VALUES agree, but
                // INT32_MIN's literal type doesn't — mirror DuckDB's tree.
                // Explicit parens arrive as Nested and are untouched.
                if let SqlExpr::BinaryOp { left, op, right } = &**expr {
                    if matches!(
                        op,
                        BinaryOperator::Multiply
                            | BinaryOperator::Divide
                            | BinaryOperator::Modulo
                            | BinaryOperator::DuckIntegerDivide
                    ) {
                        let rewritten = SqlExpr::BinaryOp {
                            left: Box::new(SqlExpr::UnaryOp {
                                op: UnaryOperator::Minus,
                                expr: left.clone(),
                            }),
                            op: op.clone(),
                            right: right.clone(),
                        };
                        return self.expr(&rewritten);
                    }
                }
                // Measured: unary +/- on a BARE NULL is BIGINT on DuckDB
                // (the SQLNULL INTEGER default does not survive negation);
                // a typed NULL — CAST(NULL AS INTEGER) — keeps its width.
                let Some(inner) = self.expr_or_null(expr)? else {
                    return Ok(null_of(Ty::I64));
                };
                if inner.ty.is_unsigned() {
                    // DuckDB wraps it: -(200::UTINYINT) is 56.
                    return Err(unsup(format!(
                        "unary minus over an unsigned integer ({})",
                        duck_int_name(inner.ty)
                    )));
                }
                // Unary minus over a DECIMAL-spelled operand that folds to
                // NULL is SQLNULL/INTEGER on DuckDB; a DOUBLE-spelled one
                // stays DOUBLE.
                if ast_decimal_literal(expr)
                    && bind_foldable(&inner)
                    && matches!(fold(inner.clone()).kind, SKind::NullOf)
                {
                    return Ok(null_of(Ty::I32));
                }
                if inner.ty.dec().is_some() {
                    // DECIMAL negation keeps the type (`-2.5` is
                    // DECIMAL(2,1)); a constant NULL is SQLNULL (measured:
                    // `-(NULL::DECIMAL(3,1))` is INTEGER).
                    let (inner, is_null) = fold_operand(inner);
                    if is_null {
                        return Ok(null_of(Ty::I32));
                    }
                    return Ok(bind_fold(self.dec_negate(inner)));
                }
                if inner.ty == Ty::F64 {
                    // The shared strict-NULL rule (`fold_operand`), which
                    // `arith` applies to every other operator: the negate is
                    // for VALUES, and a NULL has no sign to flip.
                    let (inner, is_null) = fold_operand(inner);
                    if is_null {
                        return Ok(null_of(Ty::F64));
                    }
                    let neg = math1_node(NumOp1::Fneg, inner);
                    // DuckDB types a decimal-spelled operand DECIMAL, and a
                    // DECIMAL has no negative zero: -0.0 and -(1.5 - 1.5)
                    // are +0.0 once they reach a DOUBLE. Adding +0.0 after
                    // the sign flip is exactly that (IEEE: -0.0 + 0.0 is
                    // +0.0, every other value is unchanged). A DOUBLE operand
                    // (-0.0e0, -0.0::DOUBLE, a column) keeps its sign.
                    if ast_decimal_typed(expr) {
                        let nullable = neg.nullable;
                        return Ok(SExpr {
                            kind: SKind::Arith {
                                op: ArithOp::Add,
                                a: Box::new(neg),
                                b: Box::new(SExpr {
                                    kind: SKind::Lit(Lit::F64(0.0)),
                                    ty: Ty::F64,
                                    nullable: false,
                                }),
                            },
                            ty: Ty::F64,
                            nullable,
                        });
                    }
                    return Ok(neg);
                }
                let zero = SExpr {
                    kind: SKind::Lit(Lit::I64(0)),
                    // A zero literal's natural width; the value-fits
                    // promotion hands -x its operand's own width.
                    ty: Ty::I32,
                    nullable: false,
                };
                self.arith(
                    ArithOp::Sub,
                    zero,
                    inner,
                    (Some(0), ast_int_literal(expr)),
                )
            }
            SqlExpr::UnaryOp {
                op: UnaryOperator::Plus,
                expr,
            } => match self.expr_or_null(expr)? {
                // Same BIGINT rule as unary minus (measured: +NULL).
                None => Ok(null_of(Ty::I64)),
                // DuckDB's + is a real unary function over numerics only:
                // +'a' / +TRUE are binder errors there.
                Some(e) if e.ty.is_integer() || e.ty == Ty::F64 || e.ty.dec().is_some() => Ok(e),
                Some(e) => Err(PrepareError::Bind(format!(
                    "no function matches +({})",
                    e.ty.name()
                ))),
            },
            SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr,
            } => {
                // DuckDB's transformer makes NOT over a comparison the
                // negated comparison before anything binds: `NOT (a = b)` is
                // `a <> b`, and in a JOIN ON a join condition, each side
                // evaluated over its whole table (measured, 1.5.5 optimizer
                // off; nightly seed 4873273).
                if let Some((l, op, r)) = naming::as_cmp(e) {
                    return self.binary(&op, l, r);
                }
                let inner = match self.expr_or_null(expr)? {
                    Some(x) => x,
                    None => self.typed_null(expr, Ty::I1)?,
                };
                let inner = bool_context(inner, "NOT operand")?;
                if inner.ty != Ty::I1 {
                    return Err(PrepareError::Bind(format!(
                        "NOT requires BOOLEAN, got {}",
                        inner.ty.name()
                    )));
                }
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Not(Box::new(inner)),
                    ty: Ty::I1,
                    nullable,
                })
            }
            SqlExpr::UnaryOp { op, .. } => Err(unsup(format!("unary operator {op:?}"))),
            SqlExpr::IsNull(inner) => self.is_null(inner, false),
            SqlExpr::IsNotNull(inner) => self.is_null(inner, true),
            SqlExpr::IsDistinctFrom(l, r) => self.not_distinct(l, r, true),
            SqlExpr::IsNotDistinctFrom(l, r) => self.not_distinct(l, r, false),
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } => self.case(operand.as_deref(), conditions, else_result.as_deref()),
            // Exhaustive on purpose: `array` and `format`
            // are real modifiers, not spelling flags, and the `..` that used
            // to sit here dropped both. Same class as the swallowed
            // TABLESAMPLE — refuse rather than answer a different query.
            SqlExpr::Cast {
                kind,
                expr,
                data_type,
                array,
                format,
            } => {
                if *array {
                    return Err(unsup("CAST(... AS <type> ARRAY)"));
                }
                if format.is_some() {
                    return Err(unsup("CAST(... FORMAT ...)"));
                }
                let trying = match kind {
                    CastKind::Cast | CastKind::DoubleColon => false,
                    CastKind::TryCast | CastKind::SafeCast => true,
                };
                self.cast(expr, data_type, trying)
            }
            SqlExpr::Function(f) => self.function(f),
            SqlExpr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => {
                let side = match trim_where {
                    None | Some(sqlparser::ast::TrimWhereField::Both) => TrimSide::Both,
                    Some(sqlparser::ast::TrimWhereField::Leading) => TrimSide::Lead,
                    Some(sqlparser::ast::TrimWhereField::Trailing) => TrimSide::Trail,
                };
                let chars: Option<&SqlExpr> = match (trim_what, trim_characters) {
                    (Some(w), _) => Some(w),
                    (None, Some(cs)) if cs.len() == 1 => Some(&cs[0]),
                    (None, Some(cs)) if cs.is_empty() => None,
                    (None, Some(_)) => return Err(unsup("TRIM with multiple character args")),
                    (None, None) => None,
                };
                self.trim_node(side, expr, chars)
            }
            SqlExpr::Substring {
                expr,
                substring_from,
                substring_for,
                ..
            } => self.substr_node(expr, substring_from.as_deref(), substring_for.as_deref()),
            // SQL-standard position(needle IN haystack) — needle-first,
            // same op as instr/strpos (measured).
            SqlExpr::Position { expr, r#in } => self.str2("position", StrOp2::Find, r#in, expr),
            // sqlparser gives FLOOR/CEIL dedicated AST nodes, not Function
            // calls; the datetime `CEIL(x TO field)` form rejects by name.
            SqlExpr::Floor { expr, field } => match field {
                sqlparser::ast::CeilFloorKind::Scale(_) => Err(unsup("floor with scale argument")),
                sqlparser::ast::CeilFloorKind::DateTimeField(
                    sqlparser::ast::DateTimeField::NoDateTime,
                ) => self.math1("floor", NumOp1::Ffloor, expr),
                _ => Err(unsup("FLOOR(x TO datetime-field)")),
            },
            SqlExpr::Ceil { expr, field } => match field {
                sqlparser::ast::CeilFloorKind::Scale(_) => Err(unsup("ceil with scale argument")),
                sqlparser::ast::CeilFloorKind::DateTimeField(
                    sqlparser::ast::DateTimeField::NoDateTime,
                ) => self.math1("ceil", NumOp1::Fceil, expr),
                _ => Err(unsup("CEIL(x TO datetime-field)")),
            },
            // BETWEEN and IN are exact K3 desugars (pins-wave1/): DuckDB's
            // truth tables over NULL/NaN fall out of Kleene AND/OR of the
            // duck_fcmp comparisons with zero special cases. DuckDB unifies
            // types across the WHOLE construct (one common type for the
            // subject and every bound/element), so any f64 side promotes
            // all sides before the pairwise desugar.
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => {
                let mut u = self.unify_family(&[expr, low, high])?;
                let (e, lo, hi) = (u.remove(0), u.remove(0), u.remove(0));
                // NO dead-range short circuit here, deliberately.
                // Optimizer-ON DuckDB folds a constant dead range (lo > hi)
                // to FALSE in a FILTER and never evaluates the subject. The
                // ORACLE evaluates it:
                //
                //   SELECT s FROM t WHERE CAST(s AS BIGINT) BETWEEN 22 AND 10
                //   oracle: Conversion Error: Could not convert string 'one'
                //
                // The PROJECTION form evaluates the subject under both
                // readings.
                let both = ast_bin(
                    BinaryOperator::And,
                    ast_bin(BinaryOperator::GtEq, e.clone(), lo),
                    ast_bin(BinaryOperator::LtEq, e, hi),
                );
                self.bind(&ast_not_if(*negated, both))
            }
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => {
                let mut family: Vec<&SqlExpr> = vec![expr];
                family.extend(list.iter());
                let mut unified = self.unify_family(&family)?;
                let subject = unified.remove(0);
                let mut chain: Option<SqlExpr> = None;
                for item in unified {
                    let eq = ast_bin(BinaryOperator::Eq, subject.clone(), item);
                    chain = Some(match chain {
                        None => eq,
                        Some(prev) => ast_bin(BinaryOperator::Or, prev, eq),
                    });
                }
                let chain = chain.ok_or_else(|| unsup("empty IN list"))?;
                self.bind(&ast_not_if(*negated, chain))
            }
            SqlExpr::Like {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            }
            | SqlExpr::ILike {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            } => {
                let ci = matches!(e, SqlExpr::ILike { .. });
                if *any {
                    return Err(unsup("LIKE ANY"));
                }
                // GLOB arrives as LIKE with the pattern wrapped in the
                // __glob_pat identity marker (rewrite.rs); unwrap anywhere
                // in the pattern tree so `s GLOB 'a' || x` still binds the
                // full concat as the pattern.
                if let Some(pat) = strip_glob_marker(pattern) {
                    if ci || *negated || escape_char.is_some() {
                        return Err(unsup("GLOB in a LIKE-variant position"));
                    }
                    let (ba, bp) = (self.expr_or_null(expr)?, self.expr_or_null(&pat)?);
                    let (Some(ba), Some(bp)) = (ba, bp) else {
                        return Ok(null_of(Ty::I1));
                    };
                    for side in [&ba, &bp] {
                        if side.ty != Ty::Str {
                            // GLOB has NO implicit casts (pins-wave5/;
                            // DuckDB's scalar name for it is ~~~).
                            return Err(PrepareError::Bind(format!(
                                "no function matches ~~~({}, {})",
                                ba.ty.name(),
                                bp.ty.name()
                            )));
                        }
                    }
                    let nullable = ba.nullable || bp.nullable;
                    return Ok(SExpr {
                        kind: SKind::Str2 {
                            op: StrOp2::Glob,
                            a: Box::new(ba),
                            b: Box::new(bp),
                        },
                        ty: Ty::I1,
                        nullable,
                    });
                }
                let (ba, bp) = (self.expr_or_null(expr)?, self.expr_or_null(pattern)?);
                let (Some(ba), Some(bp)) = (ba, bp) else {
                    // NULL on either side is NULL before any validation
                    // (even a bad ESCAPE never raises on NULL rows).
                    return Ok(null_of(Ty::I1));
                };
                for side in [&ba, &bp] {
                    if side.ty != Ty::Str {
                        return Err(PrepareError::Bind(format!(
                            "no function matches {}({})",
                            if ci { "ilike" } else { "like" },
                            side.ty.name()
                        )));
                    }
                }
                let esc = match escape_char {
                    None => None,
                    Some(v) => match &v.value {
                        SqlValue::SingleQuotedString(s) => Some(Box::new(lit_str(s))),
                        SqlValue::Null => return Ok(null_of(Ty::I1)),
                        other => return Err(unsup(format!("ESCAPE {other} (non-string escape)"))),
                    },
                };
                let nullable = ba.nullable || bp.nullable;
                let like = SExpr {
                    kind: SKind::Like {
                        ci,
                        a: Box::new(ba),
                        p: Box::new(bp),
                        esc,
                    },
                    ty: Ty::I1,
                    nullable,
                };
                Ok(if *negated {
                    SExpr {
                        kind: SKind::Not(Box::new(like)),
                        ty: Ty::I1,
                        nullable,
                    }
                } else {
                    like
                })
            }
            // SIMILAR TO on VALUES is exactly regexp_full_match on the RAW
            // pattern — DuckDB translates NO wildcards ('h%o' is literal %,
            // 'h.llo' is a live regex dot). pins-waveB/.
            SqlExpr::SimilarTo {
                negated,
                expr,
                pattern,
                escape_char,
            } => {
                if escape_char.is_some() {
                    return Err(unsup("Custom escape in SIMILAR TO (DuckDB: not implemented)"));
                }
                self.regex_full_predicate("SIMILAR TO", expr, pattern, *negated)
            }
            // Bracket syntax s[i] / s[a:b] — exactly array_extract /
            // array_slice in DuckDB (one shared implementation, measured:
            // pins-wave5/{subscripts-extended,slices}.json).
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                // Field read over struct_pack: the bind-time desugar — the
                // field's own expression binds in place.
                if let Some(sub) = self.desugar_struct_field(e)? {
                    return self.expr(&sub);
                }
                // Any access over a CASE reads into the arm taken: the chain
                // moves into each arm (`structs::case_field`).
                {
                    let mut base: &SqlExpr = root;
                    while let SqlExpr::Nested(i) = base {
                        base = i;
                    }
                    if matches!(base, SqlExpr::Case { .. }) {
                        return self.expr(&structs::case_field(base, access_chain));
                    }
                }
                // An element read over a list literal.
                if let (Some(elems), [AccessExpr::Subscript(Subscript::Index { index })]) =
                    (list_literal(root), access_chain.as_slice())
                {
                    return self.list_element(&elems, index);
                }
                // Field read over a declared wide extern: a lane off one
                // shared ecall.
                if let [AccessExpr::Dot(SqlExpr::Identifier(id))] = access_chain.as_slice() {
                    let mut base: &SqlExpr = root;
                    while let SqlExpr::Nested(i) = base {
                        base = i;
                    }
                    if let SqlExpr::Function(func) = base {
                        if let Some(lane) = self.extern_field_lane(func, &id.value)? {
                            return Ok(lane);
                        }
                    }
                }
                // `s['f']`, `s['n'].x`, `(s).f` over a struct COLUMN: the
                // same path as the dotted spelling.
                if let Some(path) = self.struct_access_path(e) {
                    return self.expr(&SqlExpr::CompoundIdentifier(path));
                }
                // A bare-NULL root types Str here and the chain still
                // applies; DuckDB agrees for (NULL)[2] (VARCHAR) but types
                // (NULL)[1:2] INTEGER (its SQLNULL fallback) — value parity
                // holds (NULL either way).
                let mut cur = match self.expr_or_null(root)? {
                    Some(b) => b,
                    None => null_of(Ty::Str),
                };
                for acc in access_chain {
                    let AccessExpr::Subscript(sub) = acc else {
                        return Err(unsup("struct field access in a subscript chain"));
                    };
                    cur = match sub {
                        Subscript::Index { index } => {
                            self.apply_extract("array_extract", cur, index)?
                        }
                        Subscript::Slice {
                            lower_bound,
                            upper_bound,
                            stride,
                        } => {
                            if stride.is_some() {
                                // DuckDB rejects step slicing on VARCHAR for
                                // EVERY step value, including 1 (measured).
                                return Err(unsup(
                                    "slice with step (DuckDB: not implemented for string types)",
                                ));
                            }
                            self.apply_slice(
                                "array_slice",
                                cur,
                                lower_bound.as_ref(),
                                upper_bound.as_ref(),
                            )?
                        }
                    };
                }
                Ok(cur)
            }
            other => Err(unsup(expr_refusal(other))),
        }
    }

    pub(super) fn binary(
        &self,
        op: &BinaryOperator,
        left: &SqlExpr,
        right: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        // DuckDB types integer literals INTEGER and computes their
        // arithmetic in 32 bits, so `-6 * (- 2147483647)` ERRORS there while
        // a single i64 width serves an answer. A literal-shaped integer
        // subtree is re-evaluated here in checked int32 — DuckDB's own
        // semantics — and refuses at build if any step would trap. A BIGINT
        // operand anywhere (column, cast, out-of-int32 literal) makes the
        // whole tree 64-bit on both engines and is untouched. This is the
        // CONSTANT half only: `CAST(k AS INTEGER) * 2`, trapping
        // data-dependently at row time, is the lowering's narrow-width
        // range trap instead.
        // NO i128 comparison fold here, deliberately. Optimizer-ON DuckDB
        // answers an all-literal integer
        // comparison through wide range analysis, without ever performing the
        // overflowing multiply. The ORACLE does not:
        //
        //   SELECT 9223372036854775807 > (9223372036854775807 * -50)
        //   oracle:  Out of Range Error: Overflow in multiplication of INT64
        //   opt-on:  true
        //
        // The operand on its own errors at BIND under both readings; only
        // the comparison wrapper differs, and only because of the
        // optimizer.
        let a = self.expr_or_null(left)?;
        let b = self.expr_or_null(right)?;
        // After the operands, so the depth guard refuses a deep tree
        // before this walks it.
        if let I32Fold::Traps = eval_i32_binary(left, op, right) {
            return Err(PrepareError::Bind(format!(
                "integer literal arithmetic overflows INTEGER on DuckDB \
                 ({left} {op} {right}) — int-literal math runs in 32 bits there; \
                 make an operand BIGINT (CAST(.. AS BIGINT)) for 64-bit \
                 arithmetic"
            )));
        }
        // DuckDB folds a strict op over a DECIMAL literal and a bare NULL
        // to SQLNULL — INTEGER — discarding the decimal (measured:
        // -2.681 + NULL and 2.5 * NULL are INTEGER; / stays DOUBLE). Our
        // decimal literals are f64 (the documented narrowing), so
        // adoption would answer double.
        if matches!(
            op,
            BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Multiply
                | BinaryOperator::Modulo
        ) && {
            // The rule is operand FOLDABILITY, not literal
            // spelling. A decimal-spelled operand that BINDS and folds to
            // NULL (CASE WHEN FALSE THEN 1.25 END) collapses exactly like a
            // bare NULL next to a decimal literal; DOUBLE-spelled foldable
            // NULLs do not (measured control).
            let folds_null = |x: &Option<SExpr>| x.as_ref().is_none_or(folds_to_null);
            let dec_l = ast_decimal_literal(left);
            let dec_r = ast_decimal_literal(right);
            ((a.is_none() || (dec_l && folds_null(&a))) && b.is_some() && dec_r
                || (b.is_none() || (dec_r && folds_null(&b))) && a.is_some() && dec_l)
                && (folds_null(&a) || folds_null(&b))
        } {
            return Ok(null_of(Ty::I32));
        }
        // A NULL literal adopts the other side's type; the op itself is not
        // folded (NULL AND FALSE is FALSE, so folding would be wrong). A
        // comparison, AND and OR keep an SQLNULL operand a node, so one that
        // reads a column does not fold (see `typed_null`); the other
        // operators are calls, which make it a NULL constant.
        let keeps = matches!(
            op,
            BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::LtEq
                | BinaryOperator::Gt
                | BinaryOperator::GtEq
                | BinaryOperator::And
                | BinaryOperator::Or
        );
        let adopt = |e: &SqlExpr, ty: Ty| {
            if keeps {
                self.typed_null(e, ty)
            } else {
                Ok(null_of(ty))
            }
        };
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => {
                let n = adopt(right, null_context_ty(op, a.ty))?;
                (a, n)
            }
            (None, Some(b)) => {
                let n = adopt(left, null_context_ty(op, b.ty))?;
                (n, b)
            }
            (None, None) => {
                // pins-wave5/: NULL <op> NULL types by the operator —
                // + - * % -> BIGINT, / -> DOUBLE, comparisons -> BOOLEAN
                // (via I64 operands), AND/OR -> BOOLEAN.
                let ty = match op {
                    BinaryOperator::Divide => Ty::F64,
                    BinaryOperator::And | BinaryOperator::Or => Ty::I1,
                    BinaryOperator::PGStartsWith | BinaryOperator::StringConcat => Ty::Str,
                    _ => Ty::I64,
                };
                (adopt(left, ty)?, adopt(right, ty)?)
            }
        };
        let lits = (ast_int_literal(left), ast_int_literal(right));
        match op {
            BinaryOperator::Plus => self.arith(ArithOp::Add, a, b, lits),
            BinaryOperator::Minus => self.arith(ArithOp::Sub, a, b, lits),
            BinaryOperator::Multiply => self.arith(ArithOp::Mul, a, b, lits),
            BinaryOperator::Divide => self.arith(ArithOp::Div, a, b, lits),
            BinaryOperator::DuckIntegerDivide => self.arith(ArithOp::IDiv, a, b, lits),
            BinaryOperator::Modulo => self.arith(ArithOp::Rem, a, b, lits),
            BinaryOperator::Eq => self.cmp(CmpPred::Eq, a, b),
            BinaryOperator::NotEq => self.cmp(CmpPred::Ne, a, b),
            BinaryOperator::Lt => self.cmp(CmpPred::Lt, a, b),
            BinaryOperator::LtEq => self.cmp(CmpPred::Le, a, b),
            BinaryOperator::Gt => self.cmp(CmpPred::Gt, a, b),
            BinaryOperator::GtEq => self.cmp(CmpPred::Ge, a, b),
            BinaryOperator::And | BinaryOperator::Or => {
                let a = bool_context(a, "AND/OR operand")?;
                let b = bool_context(b, "AND/OR operand")?;
                let nullable = a.nullable || b.nullable;
                let (a, b) = (Box::new(a), Box::new(b));
                let kind = if matches!(op, BinaryOperator::And) {
                    SKind::And { a, b }
                } else {
                    SKind::Or { a, b }
                };
                Ok(SExpr {
                    kind,
                    ty: Ty::I1,
                    nullable,
                })
            }
            BinaryOperator::StringConcat => {
                // DuckDB: || is ALWAYS string concat (1 || 2 = '12',
                // true || true = 'truetrue'), NULL-propagating; operands
                // implicitly cast to VARCHAR.
                let (a, b) = (to_varchar(a), to_varchar(b));
                // DuckDB's binder collapses || to an SQLNULL
                // constant (int32 at the boundary, ADOPTABLE upstream —
                // see expr_or_null's shape gate) when an operand its
                // binder can fold evaluates to NULL — any spelling, a
                // column on the OTHER side notwithstanding, since ||
                // propagates NULL to every row. Concat-specific: +, LIKE
                // and function calls keep their promoted type, and
                // concat() skips NULLs instead. The bind_foldable gate is
                // load-bearing: our own fold dead-arm-eliminates a CASE
                // whose column sits in an untaken arm, which DuckDB's
                // binder never folds — that spelling stays Str. Pure
                // extern operands TRY-fold by execution, a folded VALUE
                // baking in as a literal.
                let (a, a_null) = self.bind_fold_concat_operand(a);
                let (b, b_null) = self.bind_fold_concat_operand(b);
                if a_null || b_null {
                    return Ok(null_of(Ty::I32));
                }
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::Concat {
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            // s ^@ p is exactly starts_with(s, p): byte-prefix compare,
            // VARCHAR-only with no implicit casts (pins-wave5/).
            BinaryOperator::PGStartsWith => {
                for side in [&a, &b] {
                    if side.ty != Ty::Str {
                        return Err(PrepareError::Bind(format!(
                            "no function matches ^@({}, {})",
                            a.ty.name(),
                            b.ty.name()
                        )));
                    }
                }
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::Str2 {
                        op: StrOp2::Starts,
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty: Ty::I1,
                    nullable,
                })
            }
            // DuckDB's ^ IS pow — but sqlparser parses ^ BELOW * while
            // DuckDB binds it above (measured: duck 2*x^y = 2*(x^y),
            // sqlparser tree = (2*x)^y). Mapping it would silently compute
            // the wrong tree, so the operator stays cleanly unsupported;
            // pow()/power() cover the semantics.
            BinaryOperator::BitwiseXor => Err(unsup(
                "operator ^ (sqlparser precedence differs from DuckDB pow)",
            )),
            other => Err(unsup(format!("operator {other}"))),
        }
    }

    pub(super) fn is_null(&self, inner: &SqlExpr, negated: bool) -> Result<SExpr, PrepareError> {
        // A struct is NULL where the struct itself is, never because its
        // fields are.
        if let Some(valid) = self.struct_valid(inner)? {
            return Ok(if negated {
                valid
            } else {
                SExpr {
                    kind: SKind::Not(Box::new(valid)),
                    ty: Ty::I1,
                    nullable: false,
                }
            });
        }
        // NULL IS NULL is legal and constant; type the literal as i64
        // arbitrarily (only its flag matters).
        let inner = match self.expr_or_null(inner)? {
            Some(e) => e,
            None => self.typed_null(inner, Ty::I64)?,
        };
        // NO nullness rewrite here, deliberately. `<arithmetic> IS [NOT] NULL`
        // answers without evaluating the arithmetic on optimizer-ON DuckDB —
        // `statistics_propagation` proves the predicate from the column's null
        // statistic and deletes the expression. The ORACLE is optimizer-OFF
        // DuckDB, which evaluates and traps:
        //
        //   SELECT (c0 * 32) IS NOT NULL FROM t   -- c0 TINYINT, one row -128
        //   oracle: Out of Range Error: Overflow in multiplication of INT8
        //
        // so we evaluate and trap too. That pass's answer is not a
        // function of the query at all (it changes with
        // the table's insert history — see
        // known_divergences/test_trap_elision.py), which is exactly why the
        // oracle excludes it.
        Ok(SExpr {
            kind: SKind::IsNull {
                negated,
                inner: Box::new(inner),
            },
            ty: Ty::I1,
            nullable: false,
        })
    }

    /// `__cf_seq(pick, f1, ..., fn)`, a field read over struct_pack (see
    /// [`structs::SEQ_MARKER`]): the read field binds where the read stood,
    /// its siblings bind for their traps, and only those that can trap are
    /// kept. With none, the read is just the field, so a bare-NULL field
    /// keeps its adoptable channel.
    pub(super) fn seq(
        &self,
        f: &sqlparser::ast::Function,
    ) -> Result<Option<SExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let FunctionArguments::List(list) = &f.args else {
            return Err(PrepareError::Internal("__cf_seq without arguments".into()));
        };
        let mut args = list.args.iter().map(|a| match a {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => Ok(e),
            _ => Err(PrepareError::Internal("__cf_seq argument form".into())),
        });
        let pick: usize = match args.next().transpose()? {
            Some(SqlExpr::Value(v)) => match &v.value {
                SqlValue::Number(n, _) => n.parse().ok(),
                _ => None,
            },
            _ => None,
        }
        .ok_or_else(|| PrepareError::Internal("__cf_seq without its index".into()))?;
        let values: Vec<&SqlExpr> = args.collect::<Result<_, _>>()?;
        let mut items = Vec::with_capacity(values.len());
        let mut at = None;
        for (i, v) in values.iter().enumerate() {
            if i == pick {
                at = Some(items.len());
                match self.expr_or_null(v)? {
                    Some(e) => items.push(e),
                    None => items.push(null_of(Ty::I32)),
                }
                continue;
            }
            if let Some(e) = self.expr_or_null(v)? {
                // Only its traps are read: its skeleton, once (an equal
                // skeleton already kept traps first, with the same message).
                if let Some(s) = trap_skeleton(&e) {
                    let seen = items
                        .iter()
                        .enumerate()
                        .any(|(j, x)| Some(j) != at && *x == s);
                    if !seen {
                        items.push(s);
                    }
                }
            }
        }
        let at = at.ok_or_else(|| PrepareError::Internal("__cf_seq index out of range".into()))?;
        if items.len() == 1 {
            // No sibling can trap: the read is the field, bound in place.
            return self.expr_or_null(values[pick]);
        }
        let v = &items[at];
        if matches!(v.kind, SKind::NullOf) && self.expr_or_null(values[pick])?.is_none() {
            return Err(unsup(
                "a bare-NULL struct field read beside a field that can trap",
            ));
        }
        let (ty, nullable) = (v.ty, v.nullable);
        Ok(Some(SExpr {
            kind: SKind::Seq { items, pick: at },
            ty,
            nullable,
        }))
    }

    /// One CASE result: bound as usual, or, for `error(msg)`, a NULL arm
    /// plus the message it raises with. DuckDB types `error()` as SQLNULL and
    /// never folds it, so in a CASE it adopts the CASE's type and raises
    /// exactly when its arm is taken. The message must be a constant: a
    /// computed one refuses by name, as does `error()` anywhere else.
    fn case_result(&self, e: &SqlExpr) -> Result<(Option<SExpr>, Option<String>), PrepareError> {
        let mut inner = e;
        while let SqlExpr::Nested(x) = inner {
            inner = x;
        }
        let SqlExpr::Function(f) = inner else {
            return Ok((self.expr_or_null(e)?, None));
        };
        if !(f.name.0.len() == 1 && f.name.to_string().eq_ignore_ascii_case("error")) {
            return Ok((self.expr_or_null(e)?, None));
        }
        let args = match &f.args {
            sqlparser::ast::FunctionArguments::List(l)
                if l.duplicate_treatment.is_none() && l.clauses.is_empty() =>
            {
                &l.args
            }
            _ => return Err(unsup("error() with this argument form")),
        };
        let [sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(arg))] =
            args.as_slice()
        else {
            return Err(PrepareError::Bind("error takes exactly 1 argument".into()));
        };
        if f.over.is_some() || f.filter.is_some() || !f.within_group.is_empty() {
            return Err(unsup("error() with OVER, FILTER or WITHIN GROUP"));
        }
        let Some(msg) = self.expr_or_null(arg)? else {
            // error(NULL) is NULL.
            return Ok((None, None));
        };
        if msg.ty != Ty::Str {
            return Err(PrepareError::Bind(format!(
                "error takes a VARCHAR message, got {}",
                duck_ty_name(msg.ty)
            )));
        }
        if !bind_foldable(&msg) {
            return Err(unsup("error() with a message computed per row"));
        }
        // The message is closed: its value is the interpreter's answer for it.
        match eval_closed(&msg, Vec::new()) {
            Some(Some(ScalarVal::Str(m))) => Ok((None, Some(format!("Invalid Input Error: {m}")))),
            Some(None) => Ok((None, None)),
            _ => Err(unsup("error() with a message that does not evaluate to a constant")),
        }
    }

    /// A CASE's WHEN conditions, bound: the searched form directly, the
    /// simple form desugared to `operand = value` per arm (operand re-bound
    /// per arm via clone — pure re-evaluation, same result).
    pub(super) fn case_conditions(
        &self,
        operand: Option<&SqlExpr>,
        conditions: &[sqlparser::ast::CaseWhen],
    ) -> Result<Vec<SExpr>, PrepareError> {
        if conditions.is_empty() {
            return Err(PrepareError::Bind("CASE with no WHEN arms".to_string()));
        }
        // Each NULL here is a comparison's operand or a condition: a node
        // (see `typed_null`).
        let bound_operand = match operand {
            Some(op) => Some(match self.expr_or_null(op)? {
                Some(x) => x,
                None => self.typed_null(op, Ty::I32)?,
            }),
            None => None,
        };
        let mut conds = Vec::with_capacity(conditions.len());
        for when in conditions {
            let c = match &bound_operand {
                Some(op) => {
                    let v = match self.expr_or_null(&when.condition)? {
                        Some(v) => v,
                        None => self.typed_null(&when.condition, op.ty)?,
                    };
                    self.cmp(CmpPred::Eq, op.clone(), v)?
                }
                None => match self.expr_or_null(&when.condition)? {
                    Some(c) => bool_context(c, "CASE WHEN condition")?,
                    None => self.typed_null(&when.condition, Ty::I1)?,
                },
            };
            conds.push(c);
        }
        Ok(conds)
    }

    pub(super) fn case(
        &self,
        operand: Option<&SqlExpr>,
        conditions: &[sqlparser::ast::CaseWhen],
        else_result: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        // CASE arms are guarded: plan-time trapping-constant refusals are
        // suspended inside (see `in_guarded`).
        self.in_guarded.set(self.in_guarded.get() + 1);
        let _guard = GuardScope(&self.in_guarded);
        let conds = self.case_conditions(operand, conditions)?;

        // Bind results (NULL allowed), then unify their types. An
        // `error('msg')` result is typed like a NULL arm (DuckDB's SQLNULL)
        // and raises when its arm is taken.
        let mut results: Vec<Option<SExpr>> = Vec::with_capacity(conditions.len());
        let mut raises: Vec<Option<String>> = Vec::with_capacity(conditions.len());
        for when in conditions {
            let (r, raise) = self.case_result(&when.result)?;
            results.push(r);
            raises.push(raise);
        }
        let (else_bound, else_raise) = match else_result {
            Some(e) => {
                let (r, raise) = self.case_result(e)?;
                (Some(r), raise)
            }
            None => (None, None),
        };

        // Width unification — DuckDB's fold: SEED from the ELSE
        // (its syntactic-literal hint intact); no ELSE — or ELSE NULL —
        // seeds as an implicit non-literal NULL. Then combine WHEN arms in
        // order; every combine makes the accumulator computed, so only the
        // seed's literal-ness ever survives. A NULL arm never widens, but
        // it is not a pure skip either: Max(acc, SQLNULL) =
        // NormalizeType(acc) in types.cpp, so a NULL arm HARDENS a
        // literal-seeded accumulator to the literal's base width — that is
        // DuckDB's "NULL floors the CASE at INTEGER".
        let mut unified: Option<Ty> = None;
        let mut acc_lit: Option<i128> = None;
        let mut dec_arm: Option<SExpr> = None;
        if let Some(Some(r)) = &else_bound {
            unified = Some(r.ty);
            acc_lit = else_result.and_then(ast_int_literal);
            if r.ty.dec().is_some() {
                dec_arm = Some(r.clone());
            }
        }
        for (r, when) in results.iter().zip(conditions) {
            let Some(r) = r else {
                acc_lit = None;
                continue;
            };
            let new_lit = ast_int_literal(&when.result);
            if r.ty.dec().is_some() {
                dec_arm.get_or_insert_with(|| r.clone());
            }
            unified = Some(match unified {
                None => r.ty,
                Some(u) if u == r.ty => u,
                Some(u) if u.is_integer() && r.ty.is_integer() => {
                    int_family_promote(u, acc_lit, r.ty, new_lit)
                }
                Some(u) if u.is_integer() && r.ty == Ty::F64 => Ty::F64,
                Some(Ty::F64) if r.ty.is_integer() => Ty::F64,
                Some(u) if dec_common(u, r.ty).is_some() => dec_common(u, r.ty).expect("checked"),
                Some(u) => {
                    if let Some(d) = &dec_arm {
                        return Err(refuse_dec(
                            "CASE unification",
                            d.ty,
                            self.dec_col_name(d).as_deref(),
                        ));
                    }
                    return Err(PrepareError::Bind(format!(
                        "CASE branches disagree: {} vs {}",
                        u.name(),
                        r.ty.name()
                    )));
                }
            });
            acc_lit = None;
        }
        if unified.is_none() && (else_raise.is_some() || raises.iter().any(Option::is_some)) {
            return Err(unsup(
                "CASE whose every branch is NULL or error(): its type is DuckDB's \
                 SQLNULL, which a query level cannot carry",
            ));
        }
        let Some(unified) = unified else {
            // Every branch NULL: DuckDB types the CASE as a bare NULL (the
            // adoptable SQLNULL, measured in every context) but still
            // evaluates its conditions, so a condition that can trap keeps
            // the refusal.
            if conds.iter().any(may_trap) {
                return Err(unsup(
                    "CASE where every branch is NULL, over a condition that can trap",
                ));
            }
            return Ok(null_of(Ty::I32));
        };

        // A NULL result is a node of the CASE (see `typed_null`).
        let coerce = |r: Option<SExpr>, spelled: &SqlExpr| -> Result<SExpr, PrepareError> {
            Ok(match r {
                None => self.typed_null(spelled, unified)?,
                Some(e) if unified.dec().is_some() || (unified == Ty::F64 && e.ty.dec().is_some()) => {
                    to_common(e, unified)
                }
                Some(e) if e.ty.is_integer() && unified == Ty::F64 => promote_f64(e),
                Some(e) if e.ty.is_integer() && unified.is_integer() && e.ty != unified => {
                    widen_int(e, unified)
                }
                Some(e) => e,
            })
        };
        let raise_or = |r: Option<SExpr>, raise: Option<String>, spelled: &SqlExpr| match raise {
            Some(msg) => Ok(SExpr {
                kind: SKind::Raise(msg),
                ty: unified,
                nullable: true,
            }),
            None => coerce(r, spelled),
        };
        let results: Vec<SExpr> = results
            .into_iter()
            .zip(raises)
            .zip(conditions)
            .map(|((r, raise), w)| raise_or(r, raise, &w.result))
            .collect::<Result<_, _>>()?;
        let default = match (else_bound, else_result) {
            (Some(r), Some(spelled)) => Some(raise_or(r, else_raise, spelled)?),
            _ => None,
        };

        let nullable = default.is_none()
            || results.iter().any(|r| r.nullable)
            || default.as_ref().is_some_and(|d| d.nullable);
        let arms = conds.into_iter().zip(results).collect();
        Ok(SExpr {
            kind: SKind::Case {
                arms,
                default: default.map(Box::new),
            },
            ty: unified,
            nullable,
        })
    }

    /// `e`, a spelling that binds DuckDB's SQLNULL (`expr_or_null` answered
    /// `None`), as a value of `to` where DuckDB keeps it a node of its own:
    /// under a CAST, a comparison, AND, OR, NOT, IS NULL, a CASE, COALESCE,
    /// least, greatest, a list or a UDF argument. (A call with the default
    /// NULL handling, such as `+`, `abs` or `LIKE`, is what turns an SQLNULL
    /// operand into a NULL constant instead.) Where DuckDB folds `e` (a NULL
    /// literal, `a + NULL`), the value is a NULL constant of `to`. Where `e`
    /// reads a column (a bare NULL column, an all-NULL CASE over a column
    /// condition, `nullif(NULL, c)`), DuckDB folds neither `e` nor the node
    /// over it, so the value is a NULL of `to` that reads the column too: a
    /// strict operator over the node still evaluates its other operand, and
    /// a pure UDF over it runs per row (measured, 1.5.5: `CAST(i0 AS BIGINT)
    /// + (k + 1)`, `coalesce(i0, CAST(NULL AS BIGINT)) + (k + 1)` and
    /// `CAST(i0 = 1 AS INTEGER) + (k + 1)` trap on the overflow of `k + 1`
    /// where the same with a NULL literal does not, and the field of
    /// `udf(1.0, i0)` is DOUBLE where that of `udf(1.0, NULL)` is the SQLNULL
    /// INTEGER). Any other such spelling refuses by name.
    pub(super) fn typed_null(&self, e: &SqlExpr, to: Ty) -> Result<SExpr, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if DuckNulls::new(self).closed(e)? {
            return Ok(null_of(to));
        }
        let mut x = e;
        while let SqlExpr::Nested(i) = x {
            x = i;
        }
        if let Some((id, lets)) = lets::marker_let(x) {
            return self.typed_null(&lets[id].ast, to);
        }
        if let Some((id, calls)) = calls::marker_call(x) {
            return self.typed_null(&calls[id].1, to);
        }
        if let Some(col) = self.null_col_ref(x) {
            return Ok(null_reading(col, to));
        }
        match x {
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } if self.all_null_spelling(x) => {
                // Bound as `case` binds them: its arms are guarded.
                self.in_guarded.set(self.in_guarded.get() + 1);
                let _guard = GuardScope(&self.in_guarded);
                let conds = self.case_conditions(operand.as_deref(), conditions)?;
                let mut arms = Vec::with_capacity(conds.len());
                for (c, w) in conds.into_iter().zip(conditions) {
                    arms.push((c, self.typed_null(&w.result, to)?));
                }
                let default = match else_result {
                    Some(r) => Some(Box::new(self.typed_null(r, to)?)),
                    None => None,
                };
                Ok(SExpr {
                    kind: SKind::Case { arms, default },
                    ty: to,
                    nullable: true,
                })
            }
            SqlExpr::Function(f) => {
                let args: Vec<&SqlExpr> = match &f.args {
                    FunctionArguments::List(list) => list
                        .args
                        .iter()
                        .filter_map(|a| match a {
                            FunctionArg::Unnamed(FunctionArgExpr::Expr(a)) => Some(a),
                            _ => None,
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                // `nullif(NULL, v)` is `CASE WHEN NULL = v THEN NULL END`
                // there: it evaluates v.
                if self.nullif_sqlnull(f)? {
                    let v = args[1];
                    let read = match self.expr_or_null(v)? {
                        Some(b) => b,
                        None => self.typed_null(v, Ty::I32)?,
                    };
                    return Ok(null_reading(read, to));
                }
                // An all-NULL COALESCE, least or greatest evaluates each
                // argument in turn.
                let name = f.name.to_string().to_ascii_lowercase();
                if matches!(name.as_str(), "coalesce" | "ifnull" | "least" | "greatest")
                    && self.all_null_spelling(x)
                {
                    let mut acc: Option<SExpr> = None;
                    for a in args.iter().rev() {
                        let t = self.typed_null(a, to)?;
                        acc = Some(match acc {
                            None => t,
                            Some(rest) => SExpr {
                                kind: SKind::Case {
                                    arms: vec![(is_null_test(t), rest)],
                                    default: None,
                                },
                                ty: to,
                                nullable: true,
                            },
                        });
                    }
                    if let Some(acc) = acc {
                        return Ok(acc);
                    }
                }
                Err(unfoldable_null_refusal())
            }
            _ => Err(unfoldable_null_refusal()),
        }
    }

    pub(super) fn cast(
        &self,
        expr: &SqlExpr,
        data_type: &sqlparser::ast::DataType,
        trying: bool,
    ) -> Result<SExpr, PrepareError> {
        let to = cast_target(data_type)?;
        let inner = match self.expr_or_null(expr)? {
            Some(e) => e,
            None => return self.typed_null(expr, to),
        };
        if inner.ty == to && !trying {
            // A string LITERAL cast to VARCHAR is no longer a literal on
            // DuckDB (it binds only = / <> against a number, measured), so
            // it stays a node the comparison rule can tell apart.
            if matches!(inner.kind, SKind::Lit(Lit::Str(_))) {
                let nullable = inner.nullable;
                return Ok(SExpr {
                    kind: SKind::Cast {
                        inner: Box::new(inner),
                        trying: false,
                    },
                    ty: to,
                    nullable,
                });
            }
            return Ok(inner);
        }
        // DuckDB range-checks a DOUBLE against an unsigned target BEFORE
        // rounding, and wraps the one value that rounds up past it
        // (-0.4 errors, 255.5 becomes 0 as UTINYINT). UBIGINT has no such
        // value (the largest double below 2^64 is an integer), so its check
        // is the lowering's own.
        if to.is_unsigned() && to != Ty::U64 && inner.ty == Ty::F64 {
            return Err(unsup(format!(
                "CAST from DOUBLE to {} (DuckDB checks the range before rounding)",
                duck_int_name(to)
            )));
        }
        // A DECIMAL on either side: the checked conversions of
        // docs/specs/decimal-expressions.md §8.
        if inner.ty.dec().is_some() || to.dec().is_some() {
            return self.dec_cast_expr(inner, to, trying);
        }
        // A constant cast that FAILS is a plan-time error
        // on DuckDB — measured to fire even over zero rows and under a
        // constant-false WHERE — while a row-driven engine never evaluates
        // it. Evaluate the constant here with the interpreter's EXACT parse
        // (`kernels::duck_stoi`, the same grammar `Inst::StoiOpt` runs, so
        // the fold and the runtime cannot drift) and refuse a failure by
        // name. TRY_CAST stays lazy: it yields NULL. A numeric string that
        // parses-and-rounds ('1.5', '0x1A', '150e-1') SERVES.
        let inner = bind_fold(inner);
        if !trying && self.in_guarded.get() == 0 {
            if let SKind::Lit(Lit::Str(s)) = &inner.kind {
                let ok = match to {
                    t if t.is_integer() => duck_parses_as(s, t),
                    Ty::F64 => s.trim_ascii().parse::<f64>().is_ok(),
                    Ty::I1 => duck_stob(s).is_some(),
                    _ => true,
                };
                if !ok {
                    return Err(PrepareError::Bind(format!(
                        "constant cast fails on every row: CAST('{s}' AS                          {}) — DuckDB errors at plan time; TRY_CAST is the                          NULL-yielding spelling",
                        duck_int_name(to)
                    )));
                }
            }
        }
        // A constant that misses a NARROW target's range: TRY_CAST is NULL;
        // CAST refuses — DuckDB's plan-time conversion error, and there is
        // no CONSTANT-fold path to the lowering's narrow runtime trap (that
        // trap fires on emitted lanes, not on a folded literal), so the
        // refusal is NOT in_guarded-suspended (refusing a query DuckDB could
        // run lazily beats serving a value it would never produce).
        // The same for a wide constant (HUGEINT, past BIGINT) against any
        // narrower width, and any integer constant against UBIGINT.
        if to.is_integer() {
            let (lo, hi) = to.int_range128().expect("an integer width");
            let v = match &inner.kind {
                SKind::Lit(Lit::I128(v)) => Some(*v),
                SKind::Lit(Lit::I64(v)) if to.lane() == Ty::I128 => Some(i128::from(*v)),
                _ => None,
            };
            match v {
                Some(v) if !(lo..=hi).contains(&v) && trying => return Ok(null_of(to)),
                Some(v) if !(lo..=hi).contains(&v) => {
                    return Err(PrepareError::Bind(format!(
                        "constant cast overflows {}: DuckDB errors at plan \
                         time; TRY_CAST is the NULL-yielding spelling",
                        duck_int_name(to)
                    )))
                }
                _ => {}
            }
        }
        if let Some((lo, hi)) = to.int_range() {
            let const_out = match &inner.kind {
                SKind::Lit(Lit::I64(v)) => Some(!(lo..=hi).contains(v)),
                SKind::Lit(Lit::F64(f)) => {
                    let r = f.round_ties_even();
                    Some(!(lo as f64..=hi as f64).contains(&r))
                }
                _ => None,
            };
            match const_out {
                Some(true) if trying => return Ok(null_of(to)),
                Some(true) => {
                    return Err(PrepareError::Bind(format!(
                        "constant cast overflows {}: DuckDB errors at plan \
                         time; TRY_CAST is the NULL-yielding spelling",
                        duck_int_name(to)
                    )))
                }
                _ => {}
            }
            // TRY_CAST to a narrow width NULLs outside the range at row
            // time. No trap machinery: convert on the lane (NULL on
            // failure), then a range-guard CASE — pure re-evaluation
            // clones, like the %-by-zero guard.
            // A VARCHAR into an unsigned width parses with the sign rule
            // (`ston.opt`), range included: no guard needed.
            let parsed_unsigned = inner.ty == Ty::Str && to.is_unsigned();
            if trying && const_out.is_none() && !parsed_unsigned {
                let wide = if inner.ty.is_int() {
                    inner
                } else {
                    SExpr {
                        kind: SKind::Cast {
                            inner: Box::new(inner),
                            trying: true,
                        },
                        ty: Ty::I64,
                        nullable: true,
                    }
                };
                let bound = |n: i64| SExpr {
                    kind: SKind::Lit(Lit::I64(n)),
                    ty: Ty::I64,
                    nullable: false,
                };
                let ge = self.cmp(CmpPred::Ge, wide.clone(), bound(lo))?;
                let le = self.cmp(CmpPred::Le, wide.clone(), bound(hi))?;
                let nullable_cond = ge.nullable || le.nullable;
                let cond = SExpr {
                    kind: SKind::And {
                        a: Box::new(ge),
                        b: Box::new(le),
                    },
                    ty: Ty::I1,
                    nullable: nullable_cond,
                };
                let val = SExpr {
                    kind: SKind::Cast {
                        inner: Box::new(wide.clone()),
                        trying: false,
                    },
                    ty: to,
                    nullable: wide.nullable,
                };
                return Ok(SExpr {
                    kind: SKind::Case {
                        arms: vec![(cond, val)],
                        default: Some(Box::new(null_of(to))),
                    },
                    ty: to,
                    nullable: true,
                });
            }
        }
        let nullable = trying || inner.nullable;
        Ok(SExpr {
            kind: SKind::Cast {
                inner: Box::new(inner),
                trying,
            },
            ty: to,
            nullable,
        })
    }

    /// `lits` are the operands' SYNTACTIC-literal hints, computed by the
    /// caller from the SQL AST (`ast_int_literal`) — never from bound
    /// nodes (every SExpr-shape heuristic leaks).
    pub(super) fn arith(
        &self,
        op: ArithOp,
        a: SExpr,
        b: SExpr,
        lits: (Option<i128>, Option<i128>),
    ) -> Result<SExpr, PrepareError> {
        // The shared strict-NULL rule (`fold_operand`). Folding to NULL
        // ELIMINATES the sibling subexpression, so a trapping ln/overflow/
        // giant-string under it never executes on DuckDB and must not here.
        // Measured: DuckDB's elision is literal-NULL only; a
        // runtime NULL does not spare the trap on either engine, so eager
        // per-row evaluation stays as it is. The nullness is read BEFORE
        // promotion — promote_f64 wraps NullOf in a cast, hiding it — but
        // acted on after, since the NULL takes the PROMOTED type. Type
        // errors still refuse first, below, exactly as DuckDB binder-errors
        // before it folds.
        let (a, a_null) = fold_operand(a);
        let (b, b_null) = fold_operand(b);
        let null_operand = a_null || b_null;
        // DECIMAL arithmetic (docs/specs/decimal-expressions.md): `+ - * %`
        // against a DECIMAL or an integer stay DECIMAL; against a DOUBLE,
        // and under `/` and `//` (plain division, measured `2.5 // 2` is
        // 1.25), the DECIMAL side becomes a DOUBLE and the float path below
        // takes over. A constant NULL operand of the DECIMAL overloads, typed
        // or bare, binds DuckDB's SQLNULL (measured: `2.5 + CAST(NULL AS
        // DECIMAL(9,4))` is INTEGER).
        let (a, b) = if dec_operand(&a, &b).is_some() {
            let float = a.ty == Ty::F64 || b.ty == Ty::F64;
            match op {
                ArithOp::Add | ArithOp::Sub | ArithOp::Mul | ArithOp::Rem if !float => {
                    if null_operand {
                        return Ok(null_of(Ty::I32));
                    }
                    return self.dec_arith(op, a, b);
                }
                ArithOp::Add
                | ArithOp::Sub
                | ArithOp::Mul
                | ArithOp::Rem
                | ArithOp::Div
                | ArithOp::IDiv => {
                    let f = |e: SExpr| if e.ty.dec().is_some() { dec_to_float(e) } else { e };
                    (f(a), f(b))
                }
                _ => {
                    return Err(PrepareError::Bind(format!(
                        "No function matches the given name and argument types '{}({}, {})'",
                        arith_sym(op),
                        duck_ty_name(a.ty),
                        duck_ty_name(b.ty)
                    )))
                }
            }
        } else {
            (a, b)
        };
        if matches!(
            op,
            ArithOp::Shl | ArithOp::Shr | ArithOp::BitAnd | ArithOp::BitOr | ArithOp::BitXor
        ) {
            // Bitwise is integer-only (pins-wave5/: non-integer operands
            // are binder errors) and width-polymorphic (1 & 2 is INTEGER,
            // i & k is BIGINT — measured); compute is i64 either
            // way.
            for e in [&a, &b] {
                if !e.ty.is_integer() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches bitwise op on ({}, {})",
                        a.ty.name(),
                        b.ty.name()
                    )));
                }
            }
            // DuckDB shifts at the type's own width, with guards this
            // engine models for BIGINT only.
            if matches!(op, ArithOp::Shl | ArithOp::Shr)
                && (a.ty.is_unsigned() || b.ty.is_unsigned() || a.ty.is_wide() || b.ty.is_wide())
            {
                let what = if a.ty.is_unsigned() || b.ty.is_unsigned() {
                    "an unsigned integer"
                } else {
                    "a HUGEINT"
                };
                return Err(unsup(format!(
                    "a shift over {what} ({} {} {})",
                    duck_int_name(a.ty),
                    arith_sym(op),
                    duck_int_name(b.ty)
                )));
            }
            let ty = int_width_promote(a.ty, lits.0, b.ty, lits.1);
            let (a, b) = onto_lane(a, b, ty);
            if null_operand {
                return Ok(null_of(ty));
            }
            let nullable = a.nullable || b.nullable;
            return Ok(SExpr {
                kind: SKind::Arith {
                    op,
                    a: Box::new(a),
                    b: Box::new(b),
                },
                ty,
                nullable,
            });
        }
        let (a, b, ty) = numeric_promote(op, a, b, lits)?;
        if null_operand {
            return Ok(null_of(ty));
        }
        // DuckDB evaluates constants at plan time, so an
        // all-literal integer operation that TRAPS errors on every
        // execution there — even over zero rows — while a row-driven
        // engine would serve. Refuse by name. % and // are guarded below;
        // float arithmetic is IEEE and always folds.
        if let (SKind::Lit(Lit::I64(x)), SKind::Lit(Lit::I64(y))) =
            (&a.kind, &b.kind)
        {
            // Width-aware: the op traps in the RESULT's width (an i32 lane
            // overflows at ±2^31 on DuckDB, not ±2^63).
            let fits = |r: i64| fits_width(ty, r);
            let trapped = self.in_guarded.get() == 0
                && match op {
                    ArithOp::Add => !x.checked_add(*y).is_some_and(fits),
                    ArithOp::Sub => !x.checked_sub(*y).is_some_and(fits),
                    ArithOp::Mul => !x.checked_mul(*y).is_some_and(fits),
                    _ => false,
                };
            if trapped {
                return Err(PrepareError::Bind(format!(
                    "constant integer arithmetic overflows {} \
                     ({x} {op:?} {y}) — DuckDB evaluates constants at plan \
                     time and errors on every execution; this engine \
                     refuses instead of serving rows the oracle never would",
                    if ty == Ty::I64 { "BIGINT" } else { "INTEGER" }
                )));
            }
        }
        // The same rule on the i128 lane, at the result's own range.
        let lit128 = |k: &SKind| match k {
            SKind::Lit(Lit::I128(v)) => Some(*v),
            SKind::Lit(Lit::I64(v)) => Some(i128::from(*v)),
            _ => None,
        };
        if let (Ty::I128 | Ty::U64, Some(x), Some(y)) = (ty, lit128(&a.kind), lit128(&b.kind)) {
            let fits = |r: i128| fits_width(ty, r);
            let trapped = self.in_guarded.get() == 0
                && match op {
                    ArithOp::Add => !x.checked_add(y).is_some_and(fits),
                    ArithOp::Sub => !x.checked_sub(y).is_some_and(fits),
                    ArithOp::Mul => !x.checked_mul(y).is_some_and(fits),
                    _ => false,
                };
            if trapped {
                return Err(PrepareError::Bind(format!(
                    "constant integer arithmetic overflows {} ({x} {op:?} {y}) — DuckDB \
                     evaluates constants at plan time and errors on every execution; \
                     this engine refuses instead of serving rows the oracle never would",
                    duck_int_name(ty)
                )));
            }
        }
        let nullable = a.nullable || b.nullable;
        // DuckDB pins (pins-wave1/, pins-wave3/): integer % by zero is NULL,
        // and `//`/divide() by zero is NULL on BOTH ints and doubles. The
        // zero/NULL-divisor rule is the lowering's (`zero_divisor_nulls`):
        // the node stays a plain Arith so its DIVIDEND is always evaluated,
        // as DuckDB evaluates it, and a trap inside it fires even when the
        // divisor is 0 or NULL (fuzz seeds 23097, 20523, 46043). A CASE
        // guard here skipped the dividend. The idiv/irem traps stay
        // reachable only for MIN op -1, where DuckDB traps too. Float % is
        // IEEE (x % 0.0 = NaN), no rule.
        let nonzero_lit = matches!(b.kind, SKind::Lit(Lit::I64(n)) if n != 0)
            || matches!(b.kind, SKind::Lit(Lit::I128(n)) if n != 0)
            || matches!(b.kind, SKind::Lit(Lit::F64(x)) if x != 0.0);
        let nullable = nullable || (super::super::plan::zero_divisor_nulls(op, ty) && !nonzero_lit);
        Ok(SExpr {
            kind: SKind::Arith {
                op,
                a: Box::new(a),
                b: Box::new(b),
            },
            ty,
            nullable,
        })
    }

    /// The static column a DECIMAL value came from, for the refusal to
    /// name. Nothing but a static column produces a Dec, so the walk finds
    /// one whenever the expression is Dec-typed.
    pub(super) fn dec_col_name(&self, e: &SExpr) -> Option<String> {
        fn walk(b: &Binder<'_>, e: &SExpr) -> Option<String> {
            match &e.kind {
                SKind::StaticCol { join, col } => {
                    let sj = b.joins.get(*join as usize)?;
                    let ci = *sj.val_cols.get(*col as usize)? as usize;
                    Some(sj.table.cols.get(ci)?.name.clone())
                }
                SKind::Case { arms, default } => arms
                    .iter()
                    .find_map(|(_, r)| walk(b, r))
                    .or_else(|| default.as_deref().and_then(|d| walk(b, d))),
                _ => None,
            }
        }
        walk(self, e)
    }

    /// [`refuse_dec`] with the offending column resolved.
    pub(super) fn dec_refusal(&self, op: &str, e: &SExpr) -> PrepareError {
        refuse_dec(op, e.ty, self.dec_col_name(e).as_deref())
    }

    /// VARCHAR -> `to` (an integer width, DOUBLE or BOOLEAN) for a
    /// comparison operand, as DuckDB casts it. A constant that cannot
    /// convert is DuckDB's plan-time conversion error, refused by name as
    /// the constant CAST refuses it.
    pub(super) fn str_to(&self, to: Ty, e: SExpr) -> Result<SExpr, PrepareError> {
        if let SKind::Lit(Lit::Str(s)) = &e.kind {
            let ok = match to {
                Ty::I1 => duck_stob(s).is_some(),
                Ty::F64 => super::super::exec::kernels::duck_stof(s).is_some(),
                t => duck_parses_as(s, t),
            };
            if self.in_guarded.get() == 0 && !ok {
                return Err(PrepareError::Bind(format!(
                    "constant cast fails on every row: '{s}' to {} -- DuckDB \
                     errors at plan time",
                    duck_ty_name(to)
                )));
            }
        }
        if matches!(e.kind, SKind::NullOf) {
            return Ok(null_of(to));
        }
        let nullable = e.nullable;
        Ok(SExpr {
            kind: SKind::Cast {
                inner: Box::new(e),
                trying: false,
            },
            ty: to,
            nullable,
        })
    }

    /// `l IS [NOT] DISTINCT FROM r`: NULL-safe equality, never NULL
    /// (measured). The equality itself is `=`'s, with its type rules; the
    /// NULL cases wrap it: both NULL are not distinct, one NULL is.
    pub(super) fn not_distinct(
        &self,
        l: &SqlExpr,
        r: &SqlExpr,
        distinct: bool,
    ) -> Result<SExpr, PrepareError> {
        let is_null = |e: SExpr| SExpr {
            kind: SKind::IsNull {
                negated: false,
                inner: Box::new(e),
            },
            ty: Ty::I1,
            nullable: false,
        };
        let same = match (self.expr_or_null(l)?, self.expr_or_null(r)?) {
            (None, None) => SExpr {
                kind: SKind::Lit(Lit::I1(true)),
                ty: Ty::I1,
                nullable: false,
            },
            (Some(e), None) | (None, Some(e)) => is_null(e),
            (Some(a), Some(b)) => {
                let eq = self.cmp(CmpPred::Eq, a.clone(), b.clone())?;
                let f = SExpr {
                    kind: SKind::Lit(Lit::I1(false)),
                    ty: Ty::I1,
                    nullable: false,
                };
                SExpr {
                    kind: SKind::Case {
                        arms: vec![(is_null(a), is_null(b.clone())), (is_null(b), f)],
                        default: Some(Box::new(eq)),
                    },
                    ty: Ty::I1,
                    nullable: false,
                }
            }
        };
        Ok(if distinct {
            SExpr {
                kind: SKind::Not(Box::new(same)),
                ty: Ty::I1,
                nullable: false,
            }
        } else {
            same
        })
    }

    pub(super) fn cmp(&self, pred: CmpPred, a: SExpr, b: SExpr) -> Result<SExpr, PrepareError> {
        // The comparison result type is the operator table's rule.
        let Ret::Fixed(ret) = sig::op_ret(cmp_sym(pred)) else {
            unreachable!("comparisons are Fixed rows")
        };
        // Folded as `arith` folds. The NULL elision `arith` then performs on
        // the folded operands is deliberately absent here; see below.
        // A string LITERAL (not any VARCHAR) casts to the other side under
        // every operator; any other VARCHAR only under = and <> (measured).
        let str_lit = |e: &SExpr| matches!(e.kind, SKind::Lit(Lit::Str(_)));
        let (a_lit, b_lit) = (str_lit(&a), str_lit(&b));
        let (a, b) = (bind_fold(a), bind_fold(b));
        let equality = matches!(pred, CmpPred::Eq | CmpPred::Ne);
        let castable = |t: Ty| t.is_integer() || t == Ty::F64 || t == Ty::I1;
        let (a, b) = match (a.ty, b.ty) {
            (x, y) if x == y => (a, b),
            // Mixed integer widths compare in a shared lane: the i64 one,
            // or the i128 one when either side is UBIGINT or HUGEINT
            // (DuckDB's common type is then HUGEINT, which holds both
            // exactly: `1::UBIGINT = -1` is false, not a wrap).
            (x, y) if x.is_int() && y.is_int() => (a, b),
            (x, y) if x.is_integer() && y.is_integer() => {
                (widen_int(a, x.lane().max_lane(y)), widen_int(b, y.lane().max_lane(x)))
            }
            (x, Ty::F64) if x.is_integer() => (promote_f64(a), b),
            (Ty::F64, y) if y.is_integer() => (a, promote_f64(b)),
            // DECIMAL vs INTEGER: DuckDB casts the INTEGER up, exactly, so
            // the comparison stays in the decimal's scale. The one shape it
            // refuses is the CAPPED width, where the integer's per-row cast
            // can fail — reproducing that needs a row-time trap.
            // DECIMAL vs DECIMAL, DOUBLE or an integer: both sides at the
            // common type (docs/specs/decimal-expressions.md §7). Against a
            // DOUBLE only decimal->double is a legal implicit cast
            // (cast_rules.cpp:196-204), so the DECIMAL side casts DOWN and
            // the comparison is lossy — DuckDB's loss, reproduced. Against
            // an integer at a capped width the integer's cast can fail per
            // row, as it does on DuckDB.
            (x, y) if dec_common(x, y).is_some() => {
                let t = dec_common(x, y).expect("checked");
                (to_common(a, t), to_common(b, t))
            }
            // BOOLEAN vs an integer: DuckDB casts the BOOLEAN to INTEGER
            // (EXPLAIN: `CAST(a AS INTEGER) = i`), so it compares as 0/1 in
            // the integer lane. Against DOUBLE it refuses at bind, below.
            (Ty::I1, y) if y.is_integer() => (widen_int(bool_to_int(a), y.lane()), b),
            (x, Ty::I1) if x.is_integer() => (a, widen_int(bool_to_int(b), x.lane())),
            // A number or BOOLEAN vs VARCHAR: DuckDB casts the VARCHAR to
            // the other side's exact type (`i = '3000000000'` fails to
            // INT32); a constant that cannot convert errors at plan time, a
            // column that fails traps per row.
            (x, Ty::Str) if castable(x) && (b_lit || equality) => (a, self.str_to(x, b)?),
            (Ty::Str, y) if castable(y) && (a_lit || equality) => (self.str_to(y, a)?, b),
            (x, y) => {
                return Err(PrepareError::Bind(format!(
                    "cannot compare {} with {}",
                    duck_ty_name(x),
                    duck_ty_name(y)
                )))
            }
        };
        // BOOLEAN vs BOOLEAN orders false < true: compare as 0/1.
        let (a, b) = if a.ty == Ty::I1 {
            (bool_to_int(a), bool_to_int(b))
        } else {
            (a, b)
        };
        // NO constant shift and NO NULL-operand elision here, both
        // deliberately.
        //
        // `x ± c <cmp> k` is simplified to `x <cmp> k∓c` by
        // `expression_rewriter`, so on optimizer-ON DuckDB `(i + 1) > 5`
        // serves TRUE over INT32_MAX while `(i + 1)` alone traps. And a
        // comparison against a literal NULL folds to NULL there before either
        // side runs, so `ln(-2.0) < NULL` serves NULL.
        //
        // The ORACLE is optimizer-OFF DuckDB, and it does neither:
        //
        //   SELECT (i + 1) > 5 FROM t        -- i INTEGER = 2147483647
        //   oracle: Out of Range Error: Overflow in addition of INT32
        //   SELECT ln(x) < NULL FROM t       -- x DOUBLE = -2.0
        //   oracle: Out of Range Error: cannot take logarithm of a negative
        //
        // so neither do we. Note the ASYMMETRY this leaves, which is measured
        // rather than chosen: `arith` still folds a literal-NULL operand,
        // because THAT one is the binder's own constant folding and survives
        // the optimizer being off (`ln(x) + (x - NULL)` is NULL on the oracle).
        // Comparison folding does not. Same-looking rules, different layers.
        let nullable = a.nullable || b.nullable;
        Ok(SExpr {
            kind: SKind::Cmp {
                pred,
                a: Box::new(a),
                b: Box::new(b),
            },
            ty: ret,
            nullable,
        })
    }
}

/// DuckDB's reserved keywords (`duckdb_keywords()`, category 'reserved',
/// DuckDB 1.5.5). Unquoted, one cannot START a column reference there: its
/// grammar rejects `SELECT (from).f0` as a syntax error, where sqlparser
/// reads an identifier. After a dot any keyword is a label (`t.from`), so
/// only the head is checked; quoting (`"from"`) makes any name a name.
const DUCKDB_RESERVED: &[&str] = &[
    "all", "analyse", "analyze", "and", "any", "array", "as", "asc", "asymmetric", "both",
    "case", "cast", "check", "collate", "column", "constraint", "create", "default",
    "deferrable", "desc", "describe", "distinct", "do", "else", "end", "except", "false",
    "fetch", "for", "foreign", "from", "group", "having", "in", "initially", "intersect",
    "into", "lambda", "lateral", "leading", "limit", "not", "null", "offset", "on", "only",
    "or", "order", "pivot", "pivot_longer", "pivot_wider", "placing", "primary", "qualify",
    "references", "returning", "select", "show", "some", "summarize", "symmetric", "table",
    "then", "to", "trailing", "true", "union", "unique", "unpivot", "using", "variadic",
    "when", "where", "window", "with",
];

fn reserved_head(head: &sqlparser::ast::Ident) -> Result<(), PrepareError> {
    if head.quote_style.is_none()
        && DUCKDB_RESERVED
            .iter()
            .any(|k| k.eq_ignore_ascii_case(&head.value))
    {
        return Err(PrepareError::Parse(format!(
            "syntax error at or near \"{}\" (a reserved keyword; quote it to use it \
             as a name)",
            head.value
        )));
    }
    Ok(())
}
