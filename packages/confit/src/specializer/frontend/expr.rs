//! Expression binding: the dispatcher and the operator families
//! (comparison, arithmetic, CASE, CAST, IS NULL).

use super::*;

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
        match e {
            SqlExpr::Value(v) if matches!(v.value, SqlValue::Null) => Ok(None),
            SqlExpr::Nested(inner) => self.expr_or_null(inner),
            SqlExpr::Function(f) => {
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
        let _second_binds = self.expr_or_null(plain[1])?;
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
                    acc = Some(self.arith(flat_bitop(o), av, bv, (None, None))?);
                }
                Ok(acc.expect("a flat-bitop run has at least one operator"))
            }
            SqlExpr::BinaryOp { left, op, right } => self.binary(op, left, right),
            SqlExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } => {
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
                    return Ok(fold(self.dec_negate(inner)));
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
                Some(e) if e.ty.is_int() || e.ty == Ty::F64 || e.ty.dec().is_some() => Ok(e),
                Some(e) => Err(PrepareError::Bind(format!(
                    "no function matches +({})",
                    e.ty.name()
                ))),
            },
            SqlExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr,
            } => {
                let inner = bool_context(self.expr(expr)?, "NOT operand")?;
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
        if matches!(
            op,
            BinaryOperator::Plus
                | BinaryOperator::Minus
                | BinaryOperator::Multiply
                | BinaryOperator::Modulo
        ) {
            let probe = SqlExpr::BinaryOp {
                left: Box::new(left.clone()),
                op: op.clone(),
                right: Box::new(right.clone()),
            };
            if let I32Fold::Traps = eval_i32_literal(&probe) {
                return Err(PrepareError::Bind(format!(
                    "integer literal arithmetic overflows INTEGER on DuckDB \
                     ({} {op} {}) — int-literal math runs in 32 bits there; \
                     make an operand BIGINT (CAST(.. AS BIGINT)) for 64-bit \
                     arithmetic",
                    left, right
                )));
            }
        }
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
            let folds_null = |x: &Option<SExpr>| {
                x.as_ref().is_none_or(|e| {
                    bind_foldable(e) && matches!(fold(e.clone()).kind, SKind::NullOf)
                })
            };
            let dec_l = ast_decimal_literal(left);
            let dec_r = ast_decimal_literal(right);
            ((a.is_none() || (dec_l && folds_null(&a))) && b.is_some() && dec_r
                || (b.is_none() || (dec_r && folds_null(&b))) && a.is_some() && dec_l)
                && (folds_null(&a) || folds_null(&b))
        } {
            return Ok(null_of(Ty::I32));
        }
        // A NULL literal adopts the other side's type; the op itself is not
        // folded (NULL AND FALSE is FALSE, so folding would be wrong).
        let (a, b) = match (a, b) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => {
                let n = null_of(null_context_ty(op, a.ty));
                (a, n)
            }
            (None, Some(b)) => {
                let n = null_of(null_context_ty(op, b.ty));
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
                (null_of(ty), null_of(ty))
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
        // NULL IS NULL is legal and constant; type the literal as i64
        // arbitrarily (only its flag matters).
        let inner = match self.expr_or_null(inner)? {
            Some(e) => e,
            None => null_of(Ty::I64),
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
        if conditions.is_empty() {
            return Err(PrepareError::Bind("CASE with no WHEN arms".to_string()));
        }
        // Bind conditions: searched form directly; simple form desugars to
        // `operand = value` per arm (operand re-bound per arm via clone —
        // pure re-evaluation, same result).
        let bound_operand = operand.map(|op| self.expr(op)).transpose()?;
        let mut conds = Vec::with_capacity(conditions.len());
        for when in conditions {
            let c = match &bound_operand {
                Some(op) => {
                    let v = match self.expr_or_null(&when.condition)? {
                        Some(v) => v,
                        None => null_of(op.ty),
                    };
                    self.cmp(CmpPred::Eq, op.clone(), v)?
                }
                None => match self.expr_or_null(&when.condition)? {
                    Some(c) => bool_context(c, "CASE WHEN condition")?,
                    None => null_of(Ty::I1),
                },
            };
            conds.push(c);
        }

        // Bind results (NULL allowed), then unify their types.
        let mut results: Vec<Option<SExpr>> = Vec::with_capacity(conditions.len());
        for when in conditions {
            results.push(self.expr_or_null(&when.result)?);
        }
        let else_bound: Option<Option<SExpr>> =
            else_result.map(|e| self.expr_or_null(e)).transpose()?;

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
        let mut acc_lit: Option<i64> = None;
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
                Some(u) if u.is_int() && r.ty.is_int() => {
                    int_width_promote(u, acc_lit, r.ty, new_lit)
                }
                Some(u) if u.is_int() && r.ty == Ty::F64 => Ty::F64,
                Some(Ty::F64) if r.ty.is_int() => Ty::F64,
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

        let coerce = |r: Option<SExpr>| -> SExpr {
            match r {
                None => null_of(unified),
                Some(e) if unified.dec().is_some() || (unified == Ty::F64 && e.ty.dec().is_some()) => {
                    to_common(e, unified)
                }
                Some(e) if e.ty.is_int() && unified == Ty::F64 => promote_f64(e),
                Some(e) if e.ty.is_int() && unified.is_int() && e.ty != unified => {
                    widen_int(e, unified)
                }
                Some(e) => e,
            }
        };
        let results: Vec<SExpr> = results.into_iter().map(coerce).collect();
        let default = else_bound.map(coerce);

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

    pub(super) fn cast(
        &self,
        expr: &SqlExpr,
        data_type: &sqlparser::ast::DataType,
        trying: bool,
    ) -> Result<SExpr, PrepareError> {
        let to = cast_target(data_type)?;
        let inner = match self.expr_or_null(expr)? {
            Some(e) => e,
            // CAST(NULL AS T) is just a typed NULL, both forms.
            None => return Ok(null_of(to)),
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
        let inner = fold(inner);
        if !trying && self.in_guarded.get() == 0 {
            if let SKind::Lit(Lit::Str(s)) = &inner.kind {
                let ok = match to {
                    t if t.is_int() => super::super::exec::kernels::duck_stoi(s)
                        .is_some_and(|v| fits_width(t, v)),
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
            if trying && const_out.is_none() {
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
        lits: (Option<i64>, Option<i64>),
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
                if !e.ty.is_int() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches bitwise op on ({}, {})",
                        a.ty.name(),
                        b.ty.name()
                    )));
                }
            }
            let ty = int_width_promote(a.ty, lits.0, b.ty, lits.1);
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
                t => super::super::exec::kernels::duck_stoi(s).is_some_and(|v| fits_width(t, v)),
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
        let (a, b) = (fold(a), fold(b));
        let equality = matches!(pred, CmpPred::Eq | CmpPred::Ne);
        let castable = |t: Ty| t.is_int() || t == Ty::F64 || t == Ty::I1;
        let (a, b) = match (a.ty, b.ty) {
            (x, y) if x == y => (a, b),
            // Mixed integer widths compare in the shared i64 lane.
            (x, y) if x.is_int() && y.is_int() => (a, b),
            (x, Ty::F64) if x.is_int() => (promote_f64(a), b),
            (Ty::F64, y) if y.is_int() => (a, promote_f64(b)),
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
            (Ty::I1, y) if y.is_int() => (bool_to_int(a), b),
            (x, Ty::I1) if x.is_int() => (a, bool_to_int(b)),
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
