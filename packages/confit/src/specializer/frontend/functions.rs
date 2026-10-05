//! The builtin function catalogue and its signature resolution.

use super::*;

/// What the signature-table resolution head hands a table-resolved arm.
pub(super) enum SigArgs {
    /// A bare-NULL argument made the whole call NULL of the result type.
    Null(SExpr),
    /// Typed (checked, promoted) arguments plus the resolved result type.
    Bound(Vec<SExpr>, Ty),
}

/// Every name the builtin catalogue in [`Binder::function`] claims.
///
/// A declared UDF may not take one of these. `function()` matches the
/// catalogue on `name.as_str()` BEFORE it ever consults `find_tree` /
/// `find_udf` (they live in the `_` arm), so the builtin would win here
/// silently — while DuckDB, which lets a registered function shadow its own
/// builtin, binds the UDF. Same SQL, two engines, different answers: the
/// contract's forbidden third mode.
///
/// Refusing is the only resolution that is right at every arity. Matching
/// DuckDB by resolving UDFs first would fix the common case and break
/// another: DuckDB overload-resolves, so `least(a, b, c)` against a
/// two-argument UDF falls back to its builtin, where a UDF-first binder
/// refuses on arity. One divergence traded for another.
///
/// `builtin_names_match_the_catalogue` re-derives this list from the match
/// itself, so a new arm cannot silently escape the guard.
pub const BUILTIN_NAMES: &[&str] = &[
    "abs", "add", "any_value", "array_extract", "array_slice", "ascii", "avg",
    "bit_length", "cbrt", "ceil", "ceiling", "char_length", "character_length",
    "coalesce", "concat", "concat_ws", "contains", "cos", "count",
    "damerau_levenshtein", "divide", "editdist3", "ends_with", "error", "exp", "fdiv",
    "first", "floor", "fmod", "geomean", "greatest", "hamming", "if",
    "ifnull", "instr",
    "jaccard", "last", "lcase", "least", "len", "length", "levenshtein",
    "list_extract", "list_slice", "ln", "log", "log10", "log2", "lower",
    "lpad", "ltrim", "max", "min", "mismatches", "mod", "multiply",
    "nextafter", "nullif", "ord", "pi", "position", "pow", "power", "prefix",
    "product", "regexp_extract", "regexp_extract_all", "regexp_full_match",
    "regexp_matches", "regexp_replace", "regexp_split_to_array", "repeat",
    "replace", "reverse", "round", "rpad", "rtrim", "sin", "sqrt",
    "starts_with", "string_agg", "strip_accents", "strlen", "strpos",
    "struct_extract", "subtract", "suffix", "sum", "tan", "translate",
    "trunc", "ucase", "unicode", "upper", "xor",
];

/// Whether a call-site name is claimed by the builtin catalogue. Matching is
/// ASCII-case-insensitive, like `function()`'s own lowercasing of the name.
pub fn is_builtin(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    BUILTIN_NAMES.contains(&lower.as_str())
}

/// The builtins DuckDB overloads for DECIMAL itself, returning a DECIMAL
/// (extension/core_functions/scalar/math/numeric.cpp); every other numeric
/// builtin reads a DECIMAL as DOUBLE.
const DECIMAL_OVERLOADS: &[&str] = &["abs", "ceil", "ceiling", "floor", "trunc", "round"];

impl Binder<'_> {
    /// One argument of a call whose foldable NULL argument makes it NULL,
    /// bound as a guarded arm. DuckDB's fold swallows an argument whose
    /// evaluation errors, so a trapping constant must not refuse before a
    /// NULL sibling makes the call NULL (measured:
    /// `repeat(CAST(NULL AS VARCHAR), 9223372036854775807 * 34)` is NULL). In
    /// a live call it stays a per-row trap, which is when DuckDB raises it:
    /// never over zero rows.
    fn null_call_arg(&self, a: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        self.in_guarded.set(self.in_guarded.get() + 1);
        let _guard = GuardScope(&self.in_guarded);
        self.expr_or_null(a)
    }

    /// The signature-table resolution head for `WholeCallNull` rows: arity,
    /// eager argument binding, the bare-NULL whole-call short-circuit,
    /// per-arg type checks (byte-identical error strings), promotion into
    /// the f64 lane for the DOUBLE-returning math rows, and the result
    /// type. The arm then only builds its node.
    ///
    /// The NULL short-circuit runs BEFORE the type checks, DuckDB's
    /// dominant pattern — replace(NULL, 1, 2) binds NULL::VARCHAR and
    /// pow(s, NULL) binds NULL::DOUBLE (the latter is looser than DuckDB,
    /// which refuses the VARCHAR sibling; kept deliberately).
    pub(super) fn sig_resolve(
        &self,
        name: &str,
        sig: &Sig,
        args: &[&SqlExpr],
    ) -> Result<SigArgs, PrepareError> {
        let n = sig.params.len();
        if (!sig.variadic && args.len() != n) || (sig.variadic && args.len() < n) {
            // reverse alone spells its arity error
            // "takes one argument"; every sibling says "exactly 1".
            return Err(PrepareError::Bind(if name == "reverse" {
                format!("{name} takes one argument")
            } else {
                match n {
                    0 => format!("{name} takes no arguments"),
                    1 => format!("{name} takes exactly 1 argument"),
                    _ => format!("{name} takes exactly {n} arguments"),
                }
            }));
        }
        let mut bound: Vec<Option<SExpr>> = Vec::with_capacity(args.len());
        for a in args {
            bound.push(self.null_call_arg(a)?);
        }
        if bound.iter().any(Option::is_none) {
            // A bare NULL adopts the result type; Arg(_) rows adopt BIGINT
            // (measured: abs(NULL) binds abs(BIGINT) in DuckDB).
            let ty = match sig.ret {
                Ret::Fixed(t) => t,
                _ => Ty::I64,
            };
            return Ok(SigArgs::Null(null_of(ty)));
        }
        let mut out = Vec::with_capacity(bound.len());
        for (p, e) in sig.params.iter().zip(bound) {
            let e = e.expect("checked above");
            // A DECIMAL argument: DuckDB casts it to DOUBLE for the
            // DOUBLE-only math functions (decimal->double is its implicit
            // cast). The functions with a DECIMAL overload of their own
            // (numeric.cpp: abs, ceil, floor, trunc, round) return a DECIMAL
            // there, which is not served.
            let e = if e.ty.dec().is_some() && matches!(p, ArgTy::Num | ArgTy::Exact(Ty::F64)) {
                if DECIMAL_OVERLOADS.contains(&name) || sig.ret != Ret::Fixed(Ty::F64) {
                    return Err(self.dec_refusal(name, &e));
                }
                dec_to_float(e)
            } else {
                e
            };
            if !sig::arg_ok(*p, e.ty) {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    e.ty.name()
                )));
            }
            // Num args feed the f64 lane when the row returns DOUBLE (the
            // math1/math2 families); Arg(0) rows (abs) keep their type.
            out.push(
                if matches!(p, ArgTy::Num) && sig.ret == Ret::Fixed(Ty::F64) {
                    promote_f64(e)
                } else {
                    e
                },
            );
        }
        let ret = match sig.ret {
            Ret::Fixed(t) => t,
            Ret::Arg(i) => out[i].ty,
            Ret::Widen | Ret::Unify => {
                unreachable!("no WholeCallNull table row uses these")
            }
        };
        // Not only a bare NULL: DuckDB's binder evaluates every FOLDABLE
        // argument of a default-NULL-handling function and, if one is NULL,
        // replaces the whole call with a NULL of its return type -- so the
        // other arguments never run. Measured: lpad(CAST(NULL AS VARCHAR),
        // c0.f0 + c0.f0, 'a') is NULL where c0.f0 + c0.f0 overflows INT32.
        if out.iter().any(folds_to_null) {
            return Ok(SigArgs::Null(null_of(ret)));
        }
        Ok(SigArgs::Bound(out, ret))
    }

    /// The builtin catalogue. Everything here follows the measured pins
    /// in packages/confit/docs/specs/2026-07-26-stretch4-builtin-pins.md; names
    /// not listed reject as clean unsupported.
    pub(super) fn function(&self, f: &sqlparser::ast::Function) -> Result<SExpr, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        // DuckDB refuses every call-node modifier on a scalar call (OVER is
        // a catalog error, FILTER invalid input, IGNORE NULLS a parser
        // error); ignoring these fields would serve the bare call where
        // the oracle errors. Destructured EXHAUSTIVELY (no `..`) for
        // the same reason as [`refuse_unhandled_query`]: a modifier field
        // added to sqlparser must break this build, not the answers.
        let sqlparser::ast::Function {
            name: _,
            uses_odbc_syntax,
            parameters,
            args: _,
            filter,
            null_treatment,
            over,
            within_group,
        } = f;
        if *uses_odbc_syntax
            || !matches!(parameters, FunctionArguments::None)
            || filter.is_some()
            || null_treatment.is_some()
            || over.is_some()
            || !within_group.is_empty()
        {
            return Err(unsup(format!(
                "modifier on scalar call {} (FILTER, OVER, IGNORE NULLS and \
                 WITHIN GROUP apply to aggregates and window functions, \
                 which this engine does not serve)",
                f.name
            )));
        }
        let name = f.name.to_string().to_lowercase();
        let FunctionArguments::List(list) = &f.args else {
            return Err(unsup(format!(
                "function {} without an argument list",
                f.name
            )));
        };
        if !list.clauses.is_empty() || list.duplicate_treatment.is_some() {
            return Err(unsup(format!("function {} argument clauses", f.name)));
        }
        let mut args: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
        for a in &list.args {
            match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => args.push(e),
                _ => return Err(unsup(format!("function {} argument form", f.name))),
            }
        }
        // The builtins with a DECIMAL overload of their own take a DECIMAL
        // argument here, before any DOUBLE reading of it.
        if DECIMAL_OVERLOADS.contains(&name.as_str()) && !args.is_empty() {
            if let Some(Some(e)) = args.first().map(|a| self.expr_or_null(a)).transpose()? {
                if e.ty.dec().is_some() {
                    return self.dec_overload(&name, e, &args[1..]);
                }
            }
        }
        // Names with a WholeCallNull signature row resolve here
        // (sig.rs is the catalogue of what they accept and return); their
        // arms below only build nodes. Custom rows and CUSTOM_NAMES keep
        // every gate in their arm, verbatim.
        let resolved: Option<(Vec<SExpr>, Ty)> = match sig::lookup(&name) {
            Some(s) if s.null_arg == NullArg::WholeCallNull => {
                match self.sig_resolve(&name, s, &args)? {
                    SigArgs::Null(e) => return Ok(e),
                    SigArgs::Bound(bound, ret) => Some((bound, ret)),
                }
            }
            _ => None,
        };
        match name.as_str() {
            // ucase/lcase are alias-identical to upper/lower (pins-wave3/:
            // exhaustive all-codepoint sweep, zero mismatches).
            "upper" | "lower" | "ucase" | "lcase" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::StrCase {
                        upper: matches!(name.as_str(), "upper" | "ucase"),
                        a: Box::new(inner),
                    },
                    ty,
                    nullable,
                })
            }
            "ltrim" | "rtrim" => {
                let side = if name == "ltrim" {
                    TrimSide::Lead
                } else {
                    TrimSide::Trail
                };
                match args[..] {
                    [s] => self.trim_node(side, s, None),
                    [s, c] => self.trim_node(side, s, Some(c)),
                    _ => Err(PrepareError::Bind(format!("{name} takes 1 or 2 arguments"))),
                }
            }
            // String search (pins-wave1/): instr/strpos/2-arg position are
            // one op with (haystack, needle) order; prefix/suffix alias
            // starts_with/ends_with; positions are 1-based codepoints.
            "instr" | "strpos" | "position" | "starts_with" | "prefix" | "ends_with"
            | "suffix" => {
                let op = match name.as_str() {
                    "instr" | "strpos" | "position" => StrOp2::Find,
                    "starts_with" | "prefix" => StrOp2::Starts,
                    _ => StrOp2::Ends,
                };
                let (bound, ty) = resolved.expect("signature row");
                let Ok([h, n]) = <[SExpr; 2]>::try_from(bound) else {
                    unreachable!("arity 2")
                };
                let nullable = h.nullable || n.nullable;
                Ok(SExpr {
                    kind: SKind::Str2 {
                        op,
                        a: Box::new(h),
                        b: Box::new(n),
                    },
                    ty,
                    nullable,
                })
            }
            // Custom row: the bare-NULL-needle ambiguity gate (MAP/LIST
            // overloads) lives in str2 and must see the raw args.
            "contains" => {
                let [h, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                self.str2(&name, StrOp2::Contains, h, n)
            }
            "length" | "len" | "char_length" | "character_length" | "strlen" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::SLen {
                        bytes: name == "strlen",
                        a: Box::new(inner),
                    },
                    ty,
                    nullable,
                })
            }
            // f64 unary math (pins: 2026-07-26-wave1-builtin-pins.md).
            // 1-arg log IS base 10 in DuckDB — handled under "log" below.
            "ln" | "log2" | "log10" | "exp" | "sqrt" | "cbrt" | "sin" | "cos" | "tan" | "floor"
            | "ceil" | "ceiling" => {
                let op = match name.as_str() {
                    "ln" => NumOp1::Ln,
                    "log2" => NumOp1::Log2,
                    "log10" => NumOp1::Log10,
                    "exp" => NumOp1::Fexp,
                    "sqrt" => NumOp1::Fsqrt,
                    "cbrt" => NumOp1::Fcbrt,
                    "sin" => NumOp1::Fsin,
                    "cos" => NumOp1::Fcos,
                    "tan" => NumOp1::Ftan,
                    "floor" => NumOp1::Ffloor,
                    _ => NumOp1::Fceil,
                };
                let (bound, _ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                Ok(math1_node(op, inner))
            }
            "log" => match args[..] {
                [x] => self.math1("log", NumOp1::Log10, x),
                [b, x] => self.math2("log", BinOp::Flogb, b, x),
                _ => Err(PrepareError::Bind("log takes 1 or 2 arguments".to_string())),
            },
            // One table row: the binary DOUBLE lane (math2's NULL ordering
            // lives in the resolution head). fdiv/fmod are
            // the FLOOR pair — always DOUBLE, even for two int args.
            "pow" | "power" | "fdiv" | "fmod" | "nextafter" => {
                let op = match name.as_str() {
                    "pow" | "power" => BinOp::Fpow,
                    "fdiv" => BinOp::Ffloordiv,
                    "fmod" => BinOp::Ffloormod,
                    _ => BinOp::Fnextafter,
                };
                let (bound, ty) = resolved.expect("signature row");
                let Ok([a, b]) = <[SExpr; 2]>::try_from(bound) else {
                    unreachable!("arity 2")
                };
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::MathF2 {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty,
                    nullable,
                })
            }
            "pi" => {
                let _ = resolved.expect("signature row"); // arity 0 checked
                // Bit-equal to DuckDB's pi() (measured 0x400921FB54442D18).
                Ok(SExpr {
                    kind: SKind::Lit(Lit::F64(std::f64::consts::PI)),
                    ty: Ty::F64,
                    nullable: false,
                })
            }
            "trunc" => match args[..] {
                [arg] => {
                    let Some(inner) = self.expr_or_null(arg)? else {
                        return Ok(null_of(Ty::I64));
                    };
                    match inner.ty {
                        // Measured: integer trunc is identity, WIDTH preserved.
                        t if t.is_int() => Ok(inner),
                        Ty::F64 => Ok(math1_node(NumOp1::Ftrunc, inner)),
                        Ty::Dec(..) => Err(self.dec_refusal("trunc", &inner)),
                        other => Err(PrepareError::Bind(format!(
                            "no function matches trunc({})",
                            other.name()
                        ))),
                    }
                }
                [x, n] => self.round2(true, x, n),
                _ => Err(PrepareError::Bind(
                    "trunc takes 1 or 2 arguments".to_string(),
                )),
            },
            "abs" => {
                // abs(NULL) binds to abs(BIGINT) in DuckDB — the head's
                // Arg(0)-row NULL rule.
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                // Width-polymorphic via the Arg(0) row: abs follows its
                // argument (measured).
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Abs(Box::new(inner)),
                    ty,
                    nullable,
                })
            }
            "round" => match args[..] {
                [arg] => {
                    let Some(inner) = self.expr_or_null(arg)? else {
                        return Ok(null_of(Ty::I64));
                    };
                    match inner.ty {
                        // Measured: integer round is identity, WIDTH preserved.
                        t if t.is_int() => Ok(inner),
                        Ty::F64 => {
                            let nullable = inner.nullable;
                            Ok(SExpr {
                                kind: SKind::Round(Box::new(inner)),
                                ty: Ty::F64,
                                nullable,
                            })
                        }
                        Ty::Dec(..) => Err(self.dec_refusal("round", &inner)),
                        other => Err(PrepareError::Bind(format!(
                            "no function matches round({})",
                            other.name()
                        ))),
                    }
                }
                [x, n] => self.round2(false, x, n),
                _ => Err(PrepareError::Bind(
                    "round takes 1 or 2 arguments".to_string(),
                )),
            },
            "concat" => {
                if args.is_empty() {
                    return Err(PrepareError::Bind(
                        "concat needs at least 1 argument".to_string(),
                    ));
                }
                // CONCAT skips NULLs (measured): a literal NULL contributes
                // nothing, a nullable arg becomes CASE WHEN x IS NULL THEN ''
                // ELSE x END, and the all-NULL call is ''.
                let mut acc: Option<SExpr> = None;
                for arg in &args {
                    let Some(e) = self.expr_or_null(arg)? else {
                        continue;
                    };
                    let e = to_varchar(e);
                    let piece = if e.nullable {
                        let cond = SExpr {
                            kind: SKind::IsNull {
                                negated: false,
                                inner: Box::new(e.clone()),
                            },
                            ty: Ty::I1,
                            nullable: false,
                        };
                        SExpr {
                            kind: SKind::Case {
                                arms: vec![(cond, lit_str(""))],
                                default: Some(Box::new(e)),
                            },
                            ty: Ty::Str,
                            // Never NULL: either arm produces a value. The
                            // default's flag is provably true on its path.
                            nullable: false,
                        }
                    } else {
                        e
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(p) => SExpr {
                            kind: SKind::Concat {
                                a: Box::new(p),
                                b: Box::new(piece),
                            },
                            ty: Ty::Str,
                            nullable: false,
                        },
                    });
                }
                Ok(acc.unwrap_or_else(|| lit_str("")))
            }
            // if() IS the ternary CASE (audited: same type
            // unification, same lazy arms) — rebuild the AST and re-enter,
            // the `-a %% b` precedent, so every CASE rule applies verbatim,
            // selection-context conditions included.
            "if" => {
                let [c, a, b] = args[..] else {
                    return Err(PrepareError::Bind(
                        "if takes exactly 3 arguments".to_string(),
                    ));
                };
                let rewritten = SqlExpr::Case {
                    case_token: sqlparser::tokenizer::TokenWithSpan::wrap(
                        sqlparser::tokenizer::Token::EOF,
                    )
                    .into(),
                    end_token: sqlparser::tokenizer::TokenWithSpan::wrap(
                        sqlparser::tokenizer::Token::EOF,
                    )
                    .into(),
                    operand: None,
                    conditions: vec![sqlparser::ast::CaseWhen {
                        condition: c.clone(),
                        result: a.clone(),
                    }],
                    else_result: Some(Box::new(b.clone())),
                };
                self.expr(&rewritten)
            }
            // ifnull is 2-arg coalesce on DuckDB (audited), so it
            // shares the arm below — only the arity is its own.
            "coalesce" | "ifnull" => {
                if name == "ifnull" && args.len() != 2 {
                    return Err(PrepareError::Bind(
                        "ifnull takes exactly 2 arguments".to_string(),
                    ));
                }
                // Lazy per-row (measured: untaken erroring arms don't fire) —
                // guaranteed here because CASE branches run only when taken.
                // Stricter than DuckDB twice, deliberately — it binds
                // coalesce(NULL, NULL) as INTEGER and unifies BOOLEAN with
                // ints (coalesce(b, i) -> INTEGER); both refuse here
                // (BOOLEAN+DOUBLE refuses on both engines).
                self.in_guarded.set(self.in_guarded.get() + 1);
                let _guard = GuardScope(&self.in_guarded);
                // Seed-then-combine (DuckDB's fold): the seed keeps its
                // literal hint; every combine makes the accumulator
                // computed. Literal NULL args never produce a value — they
                // drop from evaluation — but they still fold as SQLNULL:
                // Max(acc, SQLNULL) = NormalizeType(acc), so a NULL hardens
                // a literal accumulator (and a NULL BEFORE the seed strips
                // the seed's hint the same way).
                let mut bound: Vec<SExpr> = Vec::with_capacity(args.len());
                let mut unified: Option<Ty> = None;
                let mut acc_lit: Option<i64> = None;
                let mut seen_null = false;
                for arg in &args {
                    let Some(e) = self.expr_or_null(arg)? else {
                        seen_null = true;
                        acc_lit = None;
                        continue;
                    };
                    // The hint rides with the arg's own SPELLING (never
                    // the bound node).
                    let new_lit = ast_int_literal(arg);
                    match unified {
                        None => {
                            unified = Some(e.ty);
                            acc_lit = if seen_null { None } else { new_lit };
                        }
                        Some(u) => {
                            unified = Some(match (u, e.ty) {
                                (u, t) if u == t => u,
                                (u, t) if u.is_int() && t.is_int() => {
                                    int_width_promote(u, acc_lit, t, new_lit)
                                }
                                (u, t) if u.is_int() && t == Ty::F64 => Ty::F64,
                                (Ty::F64, t) if t.is_int() => Ty::F64,
                                (u, t) if dec_common(u, t).is_some() => {
                                    dec_common(u, t).expect("checked")
                                }
                                (u, t) => {
                                    if let Some(d) =
                                        bound.iter().chain([&e]).find(|x| x.ty.dec().is_some())
                                    {
                                        return Err(refuse_dec(
                                            "COALESCE unification",
                                            d.ty,
                                            self.dec_col_name(d).as_deref(),
                                        ));
                                    }
                                    return Err(PrepareError::Bind(format!(
                                        "COALESCE arguments disagree: {} vs {}",
                                        u.name(),
                                        t.name()
                                    )));
                                }
                            });
                            acc_lit = None;
                        }
                    }
                    bound.push(e);
                }
                let Some(unified) = unified else {
                    // Only NULLs: a bare NULL on DuckDB (adoptable SQLNULL).
                    return Ok(null_of(Ty::I32));
                };
                let mut bound: Vec<SExpr> = bound
                    .into_iter()
                    .map(|mut e| {
                        if unified.dec().is_some() || (unified == Ty::F64 && e.ty.dec().is_some()) {
                            to_common(e, unified)
                        } else if e.ty.is_int() && unified == Ty::F64 {
                            promote_f64(e)
                        } else if e.ty.is_int() && unified.is_int() {
                            // Fold may select this arm whole, and the OUTPUT
                            // width is the unified one.
                            widen_int(e, unified)
                        } else {
                            e.ty = unified;
                            e
                        }
                    })
                    .collect();
                // Args after the first non-nullable one are unreachable.
                if let Some(stop) = bound.iter().position(|e| !e.nullable) {
                    bound.truncate(stop + 1);
                }
                let mut it = bound.into_iter().rev();
                let mut acc = it.next().expect("non-empty");
                for a in it {
                    let nullable = a.nullable && acc.nullable;
                    let cond = SExpr {
                        kind: SKind::IsNull {
                            negated: true,
                            inner: Box::new(a.clone()),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    };
                    acc = SExpr {
                        kind: SKind::Case {
                            arms: vec![(cond, a)],
                            default: Some(Box::new(acc)),
                        },
                        ty: unified,
                        nullable,
                    };
                }
                Ok(acc)
            }
            // least/greatest: NULL-IGNORING (result NULL only when every
            // arg is), ties return the FIRST argument, NaN sorts above
            // +inf — all of which the CASE + duck-order-cmp composition
            // reproduces exactly (pins-wave1/), so no IR op exists.
            // Stricter than DuckDB twice, deliberately — it binds
            // least(NULL, NULL) as INTEGER and unifies BOOLEAN with ints
            // (least(b, k) -> BIGINT); both refuse here (BOOLEAN+DOUBLE
            // refuses on both engines).
            "least" | "greatest" => {
                if args.is_empty() {
                    return Err(PrepareError::Bind(format!(
                        "{name} needs at least 1 argument"
                    )));
                }
                let mut bound = Vec::new();
                for arg in &args {
                    // Literal NULL args contribute nothing (NULL-ignoring);
                    // hints ride with the SPELLING.
                    if let Some(e) = self.expr_or_null(arg)? {
                        bound.push((e, ast_int_literal(arg)));
                    }
                }
                if bound.is_empty() {
                    // Only NULLs: a bare NULL on DuckDB (adoptable SQLNULL).
                    return Ok(null_of(Ty::I32));
                }
                // Seed-then-combine, same fold as COALESCE above.
                let mut unified = bound[0].0.ty;
                let mut acc_lit = bound[0].1;
                for (e, new_lit) in &bound[1..] {
                    unified = match (unified, e.ty) {
                        (u, t) if u == t => u,
                        (u, t) if u.is_int() && t.is_int() => {
                            int_width_promote(u, acc_lit, t, *new_lit)
                        }
                        (u, t) if u.is_int() && t == Ty::F64 => Ty::F64,
                        (Ty::F64, t) if t.is_int() => Ty::F64,
                        (u, t) if dec_common(u, t).is_some() => dec_common(u, t).expect("checked"),
                        (u, t) => {
                            if let Some((d, _)) =
                                bound.iter().find(|(x, _)| x.ty.dec().is_some())
                            {
                                return Err(refuse_dec(
                                    &format!("{name} unification"),
                                    d.ty,
                                    self.dec_col_name(d).as_deref(),
                                ));
                            }
                            return Err(PrepareError::Bind(format!(
                                "{name} arguments disagree: {} vs {}",
                                u.name(),
                                t.name()
                            )));
                        }
                    };
                    acc_lit = None;
                }
                let bound: Vec<SExpr> = bound
                    .into_iter()
                    .map(|(e, _)| e)
                    .map(|mut e| {
                        if unified.dec().is_some() || (unified == Ty::F64 && e.ty.dec().is_some()) {
                            to_common(e, unified)
                        } else if e.ty.is_int() && unified == Ty::F64 {
                            promote_f64(e)
                        } else if e.ty.is_int() && unified.is_int() {
                            widen_int(e, unified)
                        } else {
                            e.ty = unified;
                            e
                        }
                    })
                    .collect();
                let pred = if name == "greatest" {
                    CmpPred::Ge
                } else {
                    CmpPred::Le
                };
                let mut it = bound.into_iter();
                let mut acc = it.next().expect("non-empty");
                for b in it {
                    let cmp = self.cmp(pred, acc.clone(), b.clone())?;
                    let is_null = |e: &SExpr| SExpr {
                        kind: SKind::IsNull {
                            negated: false,
                            inner: Box::new(e.clone()),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    };
                    let nullable = acc.nullable && b.nullable;
                    acc = SExpr {
                        kind: SKind::Case {
                            arms: vec![
                                (is_null(&acc), b.clone()),
                                (is_null(&b), acc.clone()),
                                (cmp, acc),
                            ],
                            default: Some(Box::new(b)),
                        },
                        ty: unified,
                        nullable,
                    };
                }
                Ok(acc)
            }
            n if n == super::structs::SEQ_MARKER => {
                Ok(self.seq(f)?.unwrap_or_else(|| null_of(Ty::I32)))
            }
            // Served as a CASE result only (`Binder::case_result`), where
            // DuckDB's SQLNULL typing of it has a type to adopt.
            "error" => Err(unsup("error() outside a CASE result")),
            "nullif" => {
                // Stricter than DuckDB wherever cmp(Eq) refuses a mix,
                // deliberately — nullif(s, i) -> VARCHAR and nullif(b, b)
                // -> BOOLEAN both bind there (compare at the promoted type,
                // result keeps arg 1's type).
                let [a, b] = args[..] else {
                    return Err(PrepareError::Bind(
                        "nullif takes exactly 2 arguments".to_string(),
                    ));
                };
                match (self.expr_or_null(a)?, self.expr_or_null(b)?) {
                    // DuckDB types the bare NULL first argument INTEGER,
                    // nullif's output
                    // takes the first argument's type, and NULL = b is
                    // never TRUE — the whole call IS an int32 NULL.
                    (None, _) => Ok(null_of(Ty::I32)),
                    // a = NULL is never TRUE, so nullif(a, NULL) is a.
                    (Some(a), None) => Ok(a),
                    (Some(a), Some(b)) => {
                        // Comparison at the promoted type; result keeps a's
                        // ORIGINAL type (measured: nullif(1, 1.0) -> INTEGER).
                        let cond = self.cmp(CmpPred::Eq, a.clone(), b)?;
                        let ty = a.ty;
                        Ok(SExpr {
                            kind: SKind::Case {
                                arms: vec![(cond, null_of(ty))],
                                default: Some(Box::new(a)),
                            },
                            ty,
                            nullable: true,
                        })
                    }
                }
            }
            // Similarity (pins-wave3/): all raw UTF-8 BYTE-based (measured);
            // editdist3 == levenshtein and mismatches == hamming exactly.
            "levenshtein" | "editdist3" | "damerau_levenshtein" | "jaccard" | "hamming"
            | "mismatches" => {
                let op = match name.as_str() {
                    "levenshtein" | "editdist3" => StrOp2::Levenshtein,
                    "damerau_levenshtein" => StrOp2::Damerau,
                    "jaccard" => StrOp2::Jaccard,
                    _ => StrOp2::Hamming,
                };
                let (bound, ty) = resolved.expect("signature row");
                let Ok([a, b]) = <[SExpr; 2]>::try_from(bound) else {
                    unreachable!("arity 2")
                };
                let nullable = a.nullable || b.nullable;
                Ok(SExpr {
                    kind: SKind::Str2 {
                        op,
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty,
                    nullable,
                })
            }
            "repeat" => {
                let [s, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                let (bs, bn) = (self.null_call_arg(s)?, self.null_call_arg(n)?);
                // A bare NULL string picks DuckDB's BLOB overload,
                // so the answer is BLOB there and string here — and every
                // OUTER call binding the result splits (strpos/ltrim/lower/
                // levenshtein/LIKE refuse BLOB on DuckDB while building
                // here). A NULL COUNT stays: both engines type that VARCHAR.
                let Some(bs) = bs else {
                    return Err(unsup(
                        "bare NULL as repeat's string (DuckDB picks the BLOB \
                         overload; spell it CAST(NULL AS VARCHAR))",
                    ));
                };
                let Some(bn) = bn else {
                    return Ok(null_of(Ty::Str));
                };
                if bs.ty != Ty::Str {
                    // No implicit numeric->VARCHAR cast (measured).
                    return Err(PrepareError::Bind(format!(
                        "no function matches repeat({})",
                        bs.ty.name()
                    )));
                }
                if !bn.ty.is_int() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}(str, {})",
                        bn.ty.name()
                    )));
                }
                if folds_to_null(&bs) || folds_to_null(&bn) {
                    return Ok(null_of(Ty::Str));
                }
                refuse_budget_breaking_count(&name, &bn)?;
                let nullable = bs.nullable || bn.nullable;
                Ok(SExpr {
                    kind: SKind::Str2i {
                        op: StrOp2i::Repeat,
                        a: Box::new(bs),
                        n: Box::new(bn),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            "array_extract" | "list_extract" => {
                let [s, n] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 2 arguments"
                    )));
                };
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                self.apply_extract(&name, bs, n)
            }
            "array_slice" | "list_slice" => {
                if args.len() == 4 {
                    // DuckDB rejects step slicing on VARCHAR for EVERY step
                    // value, including 1 (measured).
                    return Err(unsup(
                        "slice with step (DuckDB: not implemented for string types)",
                    ));
                }
                let [s, lo, hi] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 3 arguments"
                    )));
                };
                // A bare-NULL subject types Str here; DuckDB's SQLNULL
                // fallback types the slice INTEGER (while
                // array_extract(NULL, 2) agrees at VARCHAR). Value parity
                // holds.
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                self.apply_slice(&name, bs, Some(lo), Some(hi))
            }
            "lpad" | "rpad" => {
                let [s, l, pad] = args[..] else {
                    return Err(PrepareError::Bind(format!(
                        "{name} takes exactly 3 arguments"
                    )));
                };
                let (bs, bl, bp) = (
                    self.null_call_arg(s)?,
                    self.null_call_arg(l)?,
                    self.null_call_arg(pad)?,
                );
                // DuckDB's {l,r}pad count is INTEGER and its binder
                // does NOT downcast — a BIGINT count is a binder error
                // there. The gate IS the type: INTEGER or narrower binds,
                // BIGINT refuses. The NULL short-circuit below must not skip
                // this check; a bare-NULL count itself is fine: DuckDB types
                // it INTEGER.
                let count_is_int32 =
                    |e: &SExpr| e.ty.is_int() && e.ty != Ty::I64;
                let bad_count = format!(
                    "no function matches {name}(VARCHAR, BIGINT, VARCHAR) — \
                     DuckDB's {name} count is INTEGER and a BIGINT does not \
                     implicitly narrow; spell a constant count as a plain \
                     literal or CAST(.. AS INTEGER)"
                );
                if bl.as_ref().is_some_and(|e| e.ty == Ty::I64)
                    && (bs.is_none() || bp.is_none())
                {
                    return Err(PrepareError::Bind(bad_count));
                }
                let (Some(bs), Some(bl), Some(bp)) = (bs, bl, bp) else {
                    return Ok(null_of(Ty::Str));
                };
                if bs.ty != Ty::Str || bp.ty != Ty::Str || !bl.ty.is_int() {
                    return Err(PrepareError::Bind(format!(
                        "no function matches {name}({}, {}, {})",
                        bs.ty.name(),
                        bl.ty.name(),
                        bp.ty.name()
                    )));
                }
                if !count_is_int32(&bl) {
                    return Err(PrepareError::Bind(bad_count));
                }
                // After the overload gates, before the budget: a foldable NULL
                // argument makes the call NULL at bind (see `folds_to_null`).
                if [&bs, &bl, &bp].into_iter().any(folds_to_null) {
                    return Ok(null_of(Ty::Str));
                }
                refuse_budget_breaking_count(&name, &bl)?;
                let nullable = bs.nullable || bl.nullable || bp.nullable;
                Ok(SExpr {
                    kind: SKind::Spad {
                        left: name == "lpad",
                        a: Box::new(bs),
                        len: Box::new(bl),
                        pad: Box::new(bp),
                    },
                    ty: Ty::Str,
                    nullable,
                })
            }
            "replace" | "translate" => {
                let op = if name == "replace" {
                    StrOp3::Replace
                } else {
                    StrOp3::Translate
                };
                let (bound, ty) = resolved.expect("signature row");
                let Ok([bs, bx, by]) = <[SExpr; 3]>::try_from(bound) else {
                    unreachable!("arity 3")
                };
                let nullable = bs.nullable || bx.nullable || by.nullable;
                Ok(SExpr {
                    kind: SKind::Str3 {
                        op,
                        a: Box::new(bs),
                        b: Box::new(bx),
                        c: Box::new(by),
                    },
                    ty,
                    nullable,
                })
            }
            // unicode('') = ord('') = -1, but ascii('') = 0 — the measured
            // sole divergence; all return the FIRST codepoint otherwise.
            "unicode" | "ord" | "ascii" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Sord {
                        empty_zero: name == "ascii",
                        a: Box::new(inner),
                    },
                    // Fixed(I32) row: a codepoint is INTEGER on DuckDB (the
                    // length family, by contrast, is BIGINT).
                    ty,
                    nullable,
                })
            }
            // bit_length = 8 * strlen exactly (measured) — pure desugar.
            "bit_length" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                let slen = SExpr {
                    kind: SKind::SLen {
                        bytes: true,
                        a: Box::new(inner),
                    },
                    ty,
                    nullable,
                };
                self.arith(ArithOp::Mul, lit_i64(8), slen, (None, None))
            }
            "strip_accents" => {
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::StripAccents(Box::new(inner)),
                    ty,
                    nullable,
                })
            }
            // concat_ws: NULL args are SKIPPED with their separator; NULL
            // sep -> NULL; all-args-NULL -> '' (measured). Desugars onto
            // Case/Or/Concat — the separator appears before arg i iff some
            // earlier arg was non-NULL.
            "concat_ws" => {
                if args.len() < 2 {
                    return Err(PrepareError::Bind(
                        "concat_ws needs a separator and at least 1 argument".to_string(),
                    ));
                }
                let sep = match self.expr_or_null(args[0])? {
                    // NULL separator -> NULL result, regardless of args.
                    None => return Ok(null_of(Ty::Str)),
                    Some(e) => e,
                };
                if sep.ty != Ty::Str {
                    // The separator does NOT implicitly cast (measured —
                    // unlike the value args).
                    return Err(PrepareError::Bind(format!(
                        "no function matches concat_ws({}, ...)",
                        sep.ty.name()
                    )));
                }
                let is_null = |e: &SExpr| SExpr {
                    kind: SKind::IsNull {
                        negated: false,
                        inner: Box::new(e.clone()),
                    },
                    ty: Ty::I1,
                    nullable: false,
                };
                let sconcat = |a: SExpr, b: SExpr| SExpr {
                    kind: SKind::Concat {
                        a: Box::new(a),
                        b: Box::new(b),
                    },
                    ty: Ty::Str,
                    nullable: false,
                };
                // The body only evaluates when sep is non-NULL (the outer
                // CASE guards it), so pieces use a provably-non-null view
                // of the separator — the concat() precedent shape.
                let sep_body = if sep.nullable {
                    SExpr {
                        kind: SKind::Case {
                            arms: vec![(is_null(&sep), lit_str(""))],
                            default: Some(Box::new(sep.clone())),
                        },
                        ty: Ty::Str,
                        nullable: false,
                    }
                } else {
                    sep.clone()
                };
                // prior_nullable: IS-NOT-NULL exprs of earlier nullable
                // args; prior_sure: an earlier arg is provably non-NULL.
                let mut prior_nullable: Vec<SExpr> = Vec::new();
                let mut prior_sure = false;
                let mut acc: Option<SExpr> = None;
                for arg in &args[1..] {
                    let Some(e) = self.expr_or_null(arg)? else {
                        continue; // literal NULL: skipped entirely
                    };
                    let e = to_varchar(e);
                    let joined = sconcat(sep_body.clone(), e.clone());
                    let with_sep = if prior_sure {
                        joined
                    } else if prior_nullable.is_empty() {
                        e.clone()
                    } else {
                        let mut it = prior_nullable.iter();
                        let mut some_prior = SExpr {
                            kind: SKind::IsNull {
                                negated: true,
                                inner: Box::new(it.next().expect("non-empty").clone()),
                            },
                            ty: Ty::I1,
                            nullable: false,
                        };
                        for p in it {
                            let not_null = SExpr {
                                kind: SKind::IsNull {
                                    negated: true,
                                    inner: Box::new(p.clone()),
                                },
                                ty: Ty::I1,
                                nullable: false,
                            };
                            some_prior = SExpr {
                                kind: SKind::Or {
                                    a: Box::new(some_prior),
                                    b: Box::new(not_null),
                                },
                                ty: Ty::I1,
                                nullable: false,
                            };
                        }
                        SExpr {
                            kind: SKind::Case {
                                arms: vec![(some_prior, joined)],
                                default: Some(Box::new(e.clone())),
                            },
                            ty: Ty::Str,
                            nullable: false,
                        }
                    };
                    let piece = if e.nullable {
                        SExpr {
                            kind: SKind::Case {
                                arms: vec![(is_null(&e), lit_str(""))],
                                default: Some(Box::new(with_sep)),
                            },
                            ty: Ty::Str,
                            nullable: false,
                        }
                    } else {
                        with_sep
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(p) => sconcat(p, piece),
                    });
                    if e.nullable {
                        prior_nullable.push(e);
                    } else {
                        prior_sure = true;
                    }
                }
                let body = acc.unwrap_or_else(|| lit_str(""));
                if !sep.nullable {
                    return Ok(body);
                }
                // NULL separator -> NULL result (measured), even though
                // every piece is individually total.
                Ok(SExpr {
                    kind: SKind::Case {
                        arms: vec![(is_null(&sep), null_of(Ty::Str))],
                        default: Some(Box::new(body)),
                    },
                    ty: Ty::Str,
                    nullable: true,
                })
            }
            // Math tail (pins-wave3/): add/subtract/multiply/divide/mod are EXACT
            // aliases of + - * // % (measured: same values, types, and
            // error texts); fdiv/fmod are the FLOOR pair (always DOUBLE);
            // nextafter is C nextafter, total.
            "add" | "subtract" | "multiply" | "divide" | "mod" | "xor" => {
                let [x, y] = args[..] else {
                    return Err(unsup(format!("{name} with {} arguments", args.len())));
                };
                let op = match name.as_str() {
                    "add" => ArithOp::Add,
                    "subtract" => ArithOp::Sub,
                    "multiply" => ArithOp::Mul,
                    "divide" => ArithOp::IDiv,
                    // xor is FUNCTION-only in DuckDB; `#`/`^` are not it
                    // (pins-wave5/ — `^` is pow and stays unsupported).
                    "xor" => ArithOp::BitXor,
                    _ => ArithOp::Rem,
                };
                let (bx, by) = (self.expr_or_null(x)?, self.expr_or_null(y)?);
                let (bx, by) = match (bx, by) {
                    (Some(a), Some(b)) => (a, b),
                    (Some(a), None) => {
                        let n = null_of(a.ty);
                        (a, n)
                    }
                    (None, Some(b)) => {
                        let n = null_of(b.ty);
                        (n, b)
                    }
                    (None, None) => (null_of(Ty::I64), null_of(Ty::I64)),
                };
                self.arith(op, bx, by, (ast_int_literal(x), ast_int_literal(y)))
            }
            // Named rejects: each states WHY, not just what.
            "sum" | "count" | "avg" | "min" | "max" | "geomean" | "product" | "string_agg"
            | "first" | "last" | "any_value" => Err(unsup(format!(
                "aggregate function {name} (aggregation is not served)"
            ))),
            // Regexp family (pins: 2026-07-27-waveB-regexp-pins.md).
            "regexp_matches" | "regexp_full_match" => {
                let (s, p, opts) = match args[..] {
                    [s, p] => (s, p, None),
                    [s, p, o] => (s, p, Some(o)),
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "{name} takes 2 or 3 arguments"
                        )))
                    }
                };
                let o = self.regex_options(opts, false)?;
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::I1));
                };
                let full = name == "regexp_full_match";
                let bs = str_only(&name, bs)?;
                match self.regex_pattern(p, o, full)? {
                    None => Ok(null_of(Ty::I1)),
                    Some(re) => Ok(SExpr {
                        nullable: bs.nullable,
                        kind: SKind::ReMatch {
                            re,
                            a: Box::new(bs),
                        },
                        ty: Ty::I1,
                    }),
                }
            }
            "regexp_extract" => {
                // Stricter than DuckDB — its third arg
                // also accepts a constant NAME-LIST returning a STRUCT
                // (regexp_extract(s, '(a)(b)', ['x','y'])); absent here.
                let (s, p, group, opts) = match args[..] {
                    [s, p] => (s, p, None, None),
                    [s, p, g] => (s, p, Some(g), None),
                    [s, p, g, o] => (s, p, Some(g), Some(o)),
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "{name} takes 2 to 4 arguments"
                        )))
                    }
                };
                let o = self.regex_options(opts, false)?;
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                let bs = str_only(&name, bs)?;
                // Group index: constant, flat 0..9 range check unrelated to
                // the pattern; NULL group -> '' for non-NULL subjects.
                let group = match group {
                    None => 0u32,
                    Some(g) => match self.expr_or_null(g)? {
                        None => return Ok(empty_for_nonnull(bs)),
                        Some(bg) => match bg.kind {
                            SKind::Lit(Lit::I64(n)) if (0..=9).contains(&n) => n as u32,
                            SKind::Lit(Lit::I64(_)) => {
                                return Err(PrepareError::Bind(
                                    "Group index must be between 0 and 9!".into(),
                                ))
                            }
                            _ => {
                                return Err(unsup(
                                    "non-constant regexp_extract group index",
                                ))
                            }
                        },
                    },
                };
                match self.regex_pattern(p, o, false)? {
                    None => Ok(null_of(Ty::Str)),
                    Some(re) => Ok(SExpr {
                        nullable: bs.nullable,
                        kind: SKind::ReExtract {
                            re,
                            group,
                            a: Box::new(bs),
                        },
                        ty: Ty::Str,
                    }),
                }
            }
            "regexp_replace" => {
                let (s, p, r, opts) = match args[..] {
                    [s, p, r] => (s, p, r, None),
                    [s, p, r, o] => (s, p, r, Some(o)),
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "{name} takes 3 or 4 arguments"
                        )))
                    }
                };
                // Pinned asymmetry: for regexp_replace ANY NULL argument
                // (including the options string) -> NULL result.
                let Some(o) = self.regex_options_nullable(opts)? else {
                    return Ok(null_of(Ty::Str));
                };
                let Some(bs) = self.expr_or_null(s)? else {
                    return Ok(null_of(Ty::Str));
                };
                let bs = str_only(&name, bs)?;
                let Some(br) = self.expr_or_null(r)? else {
                    return Ok(null_of(Ty::Str));
                };
                if matches!(br.kind, SKind::NullOf) && br.ty == Ty::Str {
                    // CAST(NULL AS VARCHAR) replacement — NULL result.
                    return Ok(null_of(Ty::Str));
                }
                let SKind::Lit(Lit::Str(rw)) = br.kind else {
                    return Err(unsup("non-constant regexp_replace replacement"));
                };
                let Some((re, group_count)) = self.regex_pattern_counted(p, o, false)? else {
                    return Ok(null_of(Ty::Str));
                };
                match super::super::retrans::translate_rewrite(&rw, group_count, o.global) {
                    // Invalid rewrites never error (measured RE2 quirks).
                    super::super::retrans::Rewrite::Identity => Ok(bs),
                    super::super::retrans::Rewrite::Template(t)
                    | super::super::retrans::Rewrite::ConsumeWithPrefix(t) => {
                        self.regexes.borrow_mut()[re as usize].rewrite = Some(t);
                        Ok(SExpr {
                            nullable: bs.nullable,
                            kind: SKind::ReReplace {
                                re,
                                global: o.global,
                                a: Box::new(bs),
                            },
                            ty: Ty::Str,
                        })
                    }
                }
            }
            "regexp_split_to_array" | "regexp_extract_all" => Err(unsup(format!(
                "function {name} (list-valued, non-scalar)"
            ))),
            "reverse" => {
                // ASCII byte path + UAX-29 extended grapheme path
                // (pins-waveA). No implicit casts — reverse(123) is a
                // DuckDB binder error.
                let (bound, ty) = resolved.expect("signature row");
                let Ok([inner]) = <[SExpr; 1]>::try_from(bound) else {
                    unreachable!("arity 1")
                };
                let nullable = inner.nullable;
                Ok(SExpr {
                    kind: SKind::Reverse(Box::new(inner)),
                    ty,
                    nullable,
                })
            }
            // The FUNCTION spelling of field access over a wide extern —
            // DuckDB serializes it distinct from the dot form. Over
            // struct_pack it is the struct_pack desugar instead.
            "struct_extract" => {
                if let Some(sub) = self.desugar_struct_extract(f)? {
                    return self.expr(&sub);
                }
                if let Some(path) = self.struct_access_path(&SqlExpr::Function(f.clone())) {
                    return self.expr(&SqlExpr::CompoundIdentifier(path));
                }
                if let [target, SqlExpr::Value(v)] = args[..] {
                    if let SqlValue::SingleQuotedString(field) = &v.value {
                        let mut base = target;
                        while let SqlExpr::Nested(i) = base {
                            base = i;
                        }
                        if let SqlExpr::Function(func) = base {
                            if let Some(lane) = self.extern_field_lane(func, field)? {
                                return Ok(lane);
                            }
                        }
                    }
                }
                Err(unsup(format!(
                    "function {} (not in the builtin catalogue)",
                    f.name
                )))
            }
            _ => {
                // DuckDB 1.5.5 HAS if() and ifnull() (iif/nvl are absent
                // there too); neither is in the catalogue, so a UDF/tree may
                // claim those two names and silently diverge from oracle
                // semantics.
                // A declared tree transform: same namespace as the ecall
                // UDFs, but it lowers to the native kernel rather than a
                // callback, so it is resolved before them.
                if let Some(cat) = self.find_tree(&f.name.to_string()) {
                    return self.tree_call(cat, &args);
                }
                // Declared UDF externs: width-1 is an ordinary
                // scalar expression; width-k is bare-item-only (handled in
                // the projection loop), so reaching it here is refused.
                if let Some((ext, spec)) = self.find_udf(&f.name.to_string()) {
                    if !spec.ret_names.is_empty() {
                        // Struct-valued at every width (the subtraction
                        // loop): a named extern MID-EXPRESSION has no
                        // scalar reading — DuckDB's struct registration
                        // would binder-error. Bare items take the struct
                        // boundary in the projection loop.
                        return Err(unsup(format!(
                            "udf '{}' is struct-valued (declared field names) \
                             — serve it as its own SELECT item or address an \
                             output field ({}(...).name)",
                            spec.name, spec.name
                        )));
                    }
                    if spec.rets.len() != 1 {
                        return Err(unsup(format!(
                            "width-{} udf '{}' used as a scalar expression \
                             (multi-output transformer calls must be bare SELECT items)",
                            spec.rets.len(),
                            spec.name
                        )));
                    }
                    let args = self.bind_udf_args(f, spec)?;
                    // A pure udf over constant args folds at bind (see
                    // `try_extern_bind_fold`). A NULL result is a typed NULL
                    // constant there, so a strict parent collapses without
                    // running its other operand: `udf0(NULL) * (a + 1)` is
                    // NULL on DuckDB, not an overflow (nightly seed
                    // 1118442). Measured: the constant keeps the declared
                    // type. A non-NULL result stays a run-time call; the
                    // value is the same either way.
                    let site = self.fresh_site();
                    if let Some(Ok(lanes)) = self.site_bind_fold(site, ext as usize, spec, &args) {
                        if lanes.is_none_or(|l| l.first().is_none_or(Option::is_none)) {
                            return Ok(null_of(spec.rets[0]));
                        }
                    }
                    return Ok(SExpr {
                        kind: SKind::ExternCall {
                            site,
                            ext,
                            args,
                            ret: 0,
                            whole: false,
                        },
                        ty: spec.rets[0],
                        nullable: true,
                    });
                }
                Err(unsup(format!(
                    "function {} (not in the builtin catalogue)",
                    f.name
                )))
            }
        }
    }

    /// String search (pins-wave1/): both args must be Str (no implicit numeric
    /// casts — measured binder errors). A literal NULL binds to the typed
    /// NULL result for every member EXCEPT contains, where DuckDB's
    /// overloads (MAP/LIST) make a bare NULL a binder error — mirrored.
    pub(super) fn str2(
        &self,
        name: &str,
        op: StrOp2,
        h: &SqlExpr,
        n: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let (bh, bn) = (self.expr_or_null(h)?, self.expr_or_null(n)?);
        // contains has MAP/LIST overloads; a NULL literal NEEDLE binds only
        // when a NON-literal Str haystack anchors resolution (measured:
        // contains(s, NULL) and contains(NULL, 'o') work, contains('abc',
        // NULL) and contains(NULL, NULL) are binder errors — mirrored
        // exactly).
        if name == "contains" && bn.is_none() {
            let anchored = matches!(&bh, Some(e) if !matches!(e.kind, SKind::Lit(_)));
            if !anchored {
                return Err(PrepareError::Bind(
                    "contains with a NULL literal is ambiguous (VARCHAR/MAP/LIST overloads)"
                        .to_string(),
                ));
            }
        }
        let (Some(bh), Some(bn)) = (bh, bn) else {
            return Ok(null_of(op.result_ty()));
        };
        for e in [&bh, &bn] {
            if e.ty != Ty::Str {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    e.ty.name()
                )));
            }
        }
        let nullable = bh.nullable || bn.nullable;
        Ok(SExpr {
            kind: SKind::Str2 {
                op,
                a: Box::new(bh),
                b: Box::new(bn),
            },
            ty: op.result_ty(),
            nullable,
        })
    }

    /// round(x, n) / trunc(x, n): result type == subject type; the digits
    /// slot maxes at INTEGER — a BIGINT digits expression (column or wide
    /// literal) is a binder error on DuckDB too (measured).
    /// Total on both types (i64 wraps — pinned).
    ///
    /// Looser than DuckDB in one case — a bare-NULL subject returns
    /// before the digits slot is even bound (round(NULL, s) is NULL here, a
    /// binder error there). This interleaved order is also why round/trunc
    /// stay Custom rows.
    pub(super) fn round2(&self, trunc: bool, x: &SqlExpr, n: &SqlExpr) -> Result<SExpr, PrepareError> {
        let name = if trunc { "trunc" } else { "round" };
        let Some(subject) = self.expr_or_null(x)? else {
            return Ok(null_of(Ty::I64));
        };
        if subject.ty.dec().is_some() {
            return Err(self.dec_refusal(name, &subject));
        }
        if !subject.ty.is_int() && subject.ty != Ty::F64 {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}({}, digits)",
                subject.ty.name()
            )));
        }
        let ty = subject.ty;
        let Some(digits) = self.expr_or_null(n)? else {
            return Ok(null_of(ty));
        };
        if !matches!(digits.ty, Ty::I8 | Ty::I16 | Ty::I32) {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}({}, {})",
                ty.name(),
                digits.ty.name()
            )));
        }
        let nullable = subject.nullable || digits.nullable;
        Ok(SExpr {
            kind: SKind::Round2 {
                trunc,
                a: Box::new(subject),
                n: Box::new(digits),
            },
            ty,
            nullable,
        })
    }

    /// Unary f64 math (pins-wave1/): numeric args promote to DOUBLE, VARCHAR and
    /// BOOLEAN columns are binder errors (no implicit cast — measured), a
    /// literal NULL binds to the DOUBLE overload.
    pub(super) fn math1(&self, name: &str, op: NumOp1, arg: &SqlExpr) -> Result<SExpr, PrepareError> {
        let Some(inner) = self.expr_or_null(arg)? else {
            return Ok(null_of(Ty::F64));
        };
        let inner = match inner.ty {
            Ty::F64 => inner,
            t if t.is_int() => promote_f64(inner),
            // floor/ceil (sqlparser's FLOOR/CEIL forms land here) have a
            // DECIMAL overload of their own; the rest read it as DOUBLE,
            // DuckDB's implicit decimal->double cast.
            Ty::Dec(..) if DECIMAL_OVERLOADS.contains(&name) => {
                return self.dec_overload(name, inner, &[])
            }
            Ty::Dec(..) => dec_to_float(inner),
            other => {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    other.name()
                )))
            }
        };
        Ok(math1_node(op, inner))
    }

    /// Binary f64 math (pins-wave1/; just log(base, x) — the fixed-arity
    /// members read the signature table). A literal NULL in either slot
    /// pre-empts every domain check (measured: log(-2.0, NULL) is NULL,
    /// not an error).
    ///
    /// The NULL short-circuit preceding the type checks is looser than
    /// DuckDB — log(s, NULL) binds NULL::DOUBLE here where it refuses the
    /// VARCHAR sibling (the resolution head reproduces the same order for
    /// the table rows).
    pub(super) fn math2(
        &self,
        name: &str,
        op: BinOp,
        a: &SqlExpr,
        b: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        let (ba, bb) = (self.expr_or_null(a)?, self.expr_or_null(b)?);
        let (Some(ba), Some(bb)) = (ba, bb) else {
            return Ok(null_of(Ty::F64));
        };
        let promote = |e: SExpr| -> Result<SExpr, PrepareError> {
            match e.ty {
                Ty::F64 => Ok(e),
                t if t.is_int() => Ok(promote_f64(e)),
                Ty::Dec(..) => Ok(dec_to_float(e)),
                other => Err(PrepareError::Bind(format!(
                    "no function matches {name}({})",
                    other.name()
                ))),
            }
        };
        let (ba, bb) = (promote(ba)?, promote(bb)?);
        let nullable = ba.nullable || bb.nullable;
        Ok(SExpr {
            kind: SKind::MathF2 {
                op,
                a: Box::new(ba),
                b: Box::new(bb),
            },
            ty: Ty::F64,
            nullable,
        })
    }

    /// All TRIM forms plus ltrim/rtrim. `chars` is the optional trim-set
    /// expression; absent means DuckDB's default — the single space (only
    /// 0x20 is trimmed, never tabs/newlines).
    pub(super) fn trim_node(
        &self,
        side: TrimSide,
        s: &SqlExpr,
        chars: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        let Some(s) = self.expr_or_null(s)? else {
            return Ok(null_of(Ty::Str));
        };
        if s.ty != Ty::Str {
            return Err(PrepareError::Bind(format!(
                "trim needs VARCHAR, got {}",
                s.ty.name()
            )));
        }
        let chars = match chars {
            Some(c) => match self.expr_or_null(c)? {
                // A NULL trim-set propagates NULL (measured).
                None => return Ok(null_of(Ty::Str)),
                Some(c) if c.ty == Ty::Str => c,
                Some(c) => {
                    return Err(PrepareError::Bind(format!(
                        "trim characters must be VARCHAR, got {}",
                        c.ty.name()
                    )))
                }
            },
            None => lit_str(ZS_SPACES),
        };
        let nullable = s.nullable || chars.nullable;
        Ok(SExpr {
            kind: SKind::Trim {
                side,
                a: Box::new(s),
                chars: Box::new(chars),
            },
            ty: Ty::Str,
            nullable,
        })
    }

    /// SUBSTR / SUBSTRING (both syntaxes). Missing start means 1; missing
    /// length means i64::MAX ("rest of the string" under the saturating
    /// window arithmetic in the interpreter).
    pub(super) fn substr_node(
        &self,
        s: &SqlExpr,
        from: Option<&SqlExpr>,
        for_: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        let Some(s) = self.expr_or_null(s)? else {
            return Ok(null_of(Ty::Str));
        };
        if s.ty != Ty::Str {
            return Err(PrepareError::Bind(format!(
                "substr needs VARCHAR, got {}",
                s.ty.name()
            )));
        }
        // Ok(None) = a literal NULL argument: the whole call is NULL.
        let num = |e: &SqlExpr| -> Result<Option<SExpr>, PrepareError> {
            match self.expr_or_null(e)? {
                None => Ok(None),
                Some(x) if x.ty.is_int() => Ok(Some(x)),
                Some(x) => Err(PrepareError::Bind(format!(
                    "substr position/length must be INTEGER, got {}",
                    x.ty.name()
                ))),
            }
        };
        let start = match from {
            Some(e) => match num(e)? {
                Some(x) => x,
                None => return Ok(null_of(Ty::Str)),
            },
            None => lit_i64(1),
        };
        let len = match for_ {
            Some(e) => match num(e)? {
                Some(x) => Some(Box::new(x)),
                None => return Ok(null_of(Ty::Str)),
            },
            None => None,
        };
        let nullable = s.nullable || start.nullable || len.as_ref().is_some_and(|l| l.nullable);
        Ok(SExpr {
            kind: SKind::Substr {
                a: Box::new(s),
                start: Box::new(start),
                len,
            },
            ty: Ty::Str,
            nullable,
        })
    }
}
