//! UDF and tree-model calls: argument binding, the bind-time fold,
//! wide and unnested outputs.

use super::*;

impl Binder<'_> {
    /// The declared UDF matching `name` (case-insensitive), if any.
    pub(super) fn find_udf(&self, name: &str) -> Option<(u32, &super::super::ir::ExternSpec)> {
        self.udfs
            .iter()
            .enumerate()
            .find(|(_, u)| u.name.eq_ignore_ascii_case(name))
            .map(|(i, u)| (i as u32, u))
    }

    pub(super) fn fresh_site(&self) -> u32 {
        let s = self.sites.get();
        self.sites.set(s + 1);
        s
    }

    /// The shared ecall site for a UDF call AST node: every mention of the
    /// SAME call — field reads and the whole item — binds one site (P16
    /// single-eval: the k field reads of one call execute the callable
    /// once per row, exactly as DuckDB's CSE does).
    pub(super) fn site_for(&self, f: &sqlparser::ast::Function) -> u32 {
        let mut cache = self.extern_sites.borrow_mut();
        match cache.iter().find(|(k, _)| k == f) {
            Some((_, s)) => *s,
            None => {
                let s = self.fresh_site();
                cache.push((f.clone(), s));
                s
            }
        }
    }

    /// Try to execute a pure extern at BIND, DuckDB's bind fold
    /// `None` = not foldable here
    /// (side_effects declared, no evaluator, or a non-constant argument);
    /// `Some(Err(msg))` = the callable raised, and the CONTEXT decides
    /// (field access fails the build, || swallows — both measured);
    /// `Some(Ok(..))` = the folded result (`None` = whole-call NULL).
    #[allow(clippy::type_complexity)]
    pub(super) fn try_extern_bind_fold(
        &self,
        ext: usize,
        spec: &super::super::ir::ExternSpec,
        args: &[SExpr],
    ) -> Option<Result<Option<Vec<Option<ScalarVal>>>, String>> {
        if spec.side_effects {
            return None;
        }
        let eval = self.bind_eval.get(ext)?;
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            // A pure udf over constants is itself a constant to DuckDB's
            // binder, which folds the inner call first (nightly seed
            // 4234049): fold it here too, reading the lane this argument
            // takes off it.
            if let SKind::ExternCall {
                ext: inner,
                args: inner_args,
                ret,
                whole,
                ..
            } = &a.kind
            {
                let inner_spec = self.udfs.get(*inner as usize)?;
                let lanes = match self.try_extern_bind_fold(*inner as usize, inner_spec, inner_args)? {
                    Ok(lanes) => lanes,
                    // It raises: DuckDB leaves the call to run.
                    Err(_) => return None,
                };
                vals.push(match lanes {
                    None => None,
                    Some(_) if *whole => return None,
                    Some(l) => l.into_iter().nth(*ret as usize).flatten(),
                });
                continue;
            }
            if !bind_foldable(a) {
                return None;
            }
            vals.push(match fold(a.clone()).kind {
                SKind::NullOf => None,
                SKind::Lit(Lit::I1(v)) => Some(ScalarVal::I1(v)),
                SKind::Lit(Lit::I64(v)) => Some(ScalarVal::I64(v)),
                SKind::Lit(Lit::F64(v)) => Some(ScalarVal::F64(v)),
                SKind::Lit(Lit::Str(s)) => Some(ScalarVal::Str(s)),
                // `CAST('0' AS VARCHAR)`: the binder keeps a string literal
                // cast to VARCHAR as a node (it compares unlike a bare
                // literal), but as an argument it is just its value
                // (fuzz seed 49961).
                SKind::Cast { inner, .. }
                    if a.ty == Ty::Str && matches!(inner.kind, SKind::Lit(Lit::Str(_))) =>
                {
                    let SKind::Lit(Lit::Str(s)) = inner.kind else {
                        unreachable!("matched above")
                    };
                    Some(ScalarVal::Str(s))
                }
                // A constant spelling our fold leaves as runtime ops over
                // literals (`reverse('x')`, `'a' LIKE '_'`): DuckDB's binder
                // evaluates it, so we do too (nightly seeds 1801793,
                // 3811339). A trap means no fold, as DuckDB's
                // TryEvaluateScalar gives up and leaves the call to run.
                _ => self.eval_closed(a)?,
            });
        }
        Some((eval.fun)(&vals))
    }

    fn eval_closed(&self, e: &SExpr) -> Option<Option<ScalarVal>> {
        eval_closed(e, self.regexes.borrow().clone())
    }

    /// [`Self::try_extern_bind_fold`], at most once per call site.
    #[allow(clippy::type_complexity)]
    pub(super) fn site_bind_fold(
        &self,
        site: u32,
        ext: usize,
        spec: &super::super::ir::ExternSpec,
        args: &[SExpr],
    ) -> Option<Result<Option<Vec<Option<ScalarVal>>>, String>> {
        if let Some((_, r)) = self.bind_folds.borrow().iter().find(|(s, _)| *s == site) {
            return r.clone();
        }
        let r = self.try_extern_bind_fold(ext, spec, args);
        self.bind_folds.borrow_mut().push((site, r.clone()));
        r
    }

    /// Bind-fold one || operand, on top of [`Self::try_extern_bind_fold`].
    /// `(_, true)` = the operand folds to NULL, so the whole || collapses
    /// to SQLNULL. Otherwise the (possibly rewritten) operand comes back:
    /// a pure extern's folded VALUE is baked as a literal — DuckDB
    /// executes once at bind, never per row, and a non-deterministic
    /// "pure" udf gets one baked sample there too — while a raising
    /// callable keeps the runtime call (DuckDB's fold swallows
    /// exceptions uniformly).
    pub(super) fn bind_fold_concat_operand(&self, e: SExpr) -> (SExpr, bool) {
        if folds_to_null(&e) {
            return (e, true);
        }
        // Peel unary wrappers down to a possible pure extern: to_varchar
        // adds one Cast, and a composition pin showed upper()/lower()
        // between the extern and the || is the same shape. Bake the extern,
        // rebuild the wrappers over the literal, and FOLD the result — the
        // StrCase/Abs fold arms finish what the bake started, so
        // `upper(us9(..))` collapses exactly like the bare call.
        enum Frame {
            Cast { ty: Ty, trying: bool },
            Case { upper: bool, ty: Ty },
        }
        let mut frames: Vec<Frame> = Vec::new();
        let mut cur = e.clone();
        loop {
            match cur.kind {
                SKind::Cast { inner, trying } => {
                    frames.push(Frame::Cast { ty: cur.ty, trying });
                    cur = *inner;
                }
                SKind::StrCase { upper, a } => {
                    frames.push(Frame::Case { upper, ty: cur.ty });
                    cur = *a;
                }
                _ => break,
            }
        }
        if let SKind::ExternCall {
            site,
            ext,
            ref args,
            whole: false,
            ..
        } = cur.kind
        {
            if let Some(spec) = self.udfs.get(ext as usize) {
                if spec.rets.len() == 1 {
                    match self.site_bind_fold(site, ext as usize, spec, args) {
                        Some(Ok(None)) => return (e, true),
                        Some(Ok(Some(lanes))) => match lanes.into_iter().next() {
                            Some(Some(v)) => {
                                let mut rebuilt = scalar_lit(v, cur.ty);
                                for f in frames.into_iter().rev() {
                                    rebuilt = match f {
                                        Frame::Cast { ty, trying } => SExpr {
                                            nullable: false,
                                            kind: SKind::Cast {
                                                inner: Box::new(rebuilt),
                                                trying,
                                            },
                                            ty,
                                        },
                                        Frame::Case { upper, ty } => SExpr {
                                            nullable: rebuilt.nullable,
                                            kind: SKind::StrCase {
                                                upper,
                                                a: Box::new(rebuilt),
                                            },
                                            ty,
                                        },
                                    };
                                }
                                let rebuilt = fold(rebuilt);
                                if matches!(rebuilt.kind, SKind::NullOf) {
                                    return (e, true);
                                }
                                return (rebuilt, false);
                            }
                            // A NULL-valued fold result collapses exactly
                            // like any constant NULL operand (measured:
                            // s || CAST(NULL AS VARCHAR) is INTEGER).
                            _ => return (e, true),
                        },
                        _ => {}
                    }
                }
            }
        }
        (e, false)
    }

    /// Field access over a declared width-k extern call: bind the named
    /// lane of ONE shared ecall. `Ok(None)` when this isn't
    /// that shape — callers fall through to their own handling.
    pub(super) fn extern_field_lane(
        &self,
        f: &sqlparser::ast::Function,
        field: &str,
    ) -> Result<Option<SExpr>, PrepareError> {
        let Some((ext, spec)) = self.find_udf(&f.name.to_string()) else {
            return Ok(None);
        };
        if spec.ret_names.is_empty() {
            return Err(unsup(format!(
                "field access on udf '{}': no declared output field names",
                spec.name
            )));
        }
        let Some(ret) = spec
            .ret_names
            .iter()
            .position(|n| n.eq_ignore_ascii_case(field))
        else {
            return Err(PrepareError::Bind(format!(
                "udf '{}' has no output field '{}' (declared: {})",
                spec.name,
                field,
                spec.ret_names
                    .iter()
                    .map(|n| format!("'{n}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        };
        let args = self.bind_udf_args(f, spec)?;
        // Field access is a fold context — a pure udf with
        // constant args EXECUTES here at bind, special null handling
        // honored (the real result is used, never assumed). Whole-call
        // None is DuckDB's SQLNULL (surfaced int32, ADOPTED by consumers
        // via expr_or_null's shape gate); a real struct keeps its
        // declared field types, NULL fields included. A raised exception
        // is SWALLOWED and the runtime call stays — DuckDB's fold gives
        // up uniformly (DESCRIBE succeeds, the error fires at RUN with
        // rows, a zero-row batch answers empty; a FROM-less query that
        // appears to error at bind is eager constant evaluation, not the
        // binder).
        match self.try_extern_bind_fold(ext as usize, spec, &args) {
            Some(Err(_)) => {}
            Some(Ok(None)) => return Ok(Some(null_of(Ty::I32))),
            Some(Ok(Some(lanes))) => {
                let ty = spec.rets[ret];
                return Ok(Some(match lanes.into_iter().nth(ret) {
                    Some(Some(v)) => scalar_lit(v, ty),
                    _ => null_of(ty),
                }));
            }
            None => {}
        }
        let site = self.site_for(f);
        Ok(Some(SExpr {
            kind: SKind::ExternCall {
                site,
                ext,
                args,
                ret: ret as u32,
                whole: false,
            },
            ty: spec.rets[ret],
            nullable: true,
        }))
    }

    /// Bind and type-check a UDF call's arguments against its declared
    /// params. Bare NULLs adopt the param type; an i64 argument against a
    /// declared f64 promotes exactly like DuckDB's implicit cast; anything
    /// else refuses by name.
    pub(super) fn bind_udf_args(
        &self,
        f: &sqlparser::ast::Function,
        spec: &super::super::ir::ExternSpec,
    ) -> Result<Vec<SExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let FunctionArguments::List(list) = &f.args else {
            return Err(unsup(format!(
                "function {} without an argument list",
                f.name
            )));
        };
        if !list.clauses.is_empty() || list.duplicate_treatment.is_some() {
            return Err(unsup(format!("function {} argument clauses", f.name)));
        }
        // Every caller of a declared UDF routes through here, so the
        // call-node modifiers the oracle rejects are screened once
        // (review round: IGNORE NULLS rode in on the INNER call of an
        // unnest item and expanded as if unadorned).
        if f.filter.is_some()
            || f.over.is_some()
            || f.null_treatment.is_some()
            || !f.within_group.is_empty()
        {
            return Err(unsup(format!(
                "modifier on udf call {} (FILTER, OVER, IGNORE NULLS and \
                 WITHIN GROUP apply to aggregates and window functions)",
                f.name
            )));
        }
        let mut raw: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
        for a in &list.args {
            match a {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) => raw.push(e),
                _ => return Err(unsup(format!("function {} argument form", f.name))),
            }
        }
        if raw.len() != spec.params.len() {
            return Err(PrepareError::Bind(format!(
                "udf '{}' takes {} argument(s), the call passes {}",
                spec.name,
                spec.params.len(),
                raw.len()
            )));
        }
        let mut out = Vec::with_capacity(raw.len());
        for (i, (arg, &pt)) in raw.iter().zip(&spec.params).enumerate() {
            let bound = match self.expr_or_null(arg)? {
                None => null_of(pt),
                Some(e) => bind_fold(e),
            };
            let bound = match (bound.ty, pt) {
                (a, b) if a == b => bound,
                // A narrow int upcasts into a declared int64 param (the
                // lane is shared) — DuckDB's implicit INTEGER -> BIGINT.
                (a, Ty::I64) if a.is_int() => widen_int(bound, Ty::I64),
                (a, Ty::F64) if a.is_int() => promote_f64(bound),
                // decimal->double, DuckDB's implicit cast to a DOUBLE param.
                (Ty::Dec(..), Ty::F64) => dec_to_float(bound),
                (a, b) => {
                    return Err(PrepareError::Bind(format!(
                        "udf '{}' argument {} is {}, declared {}",
                        spec.name,
                        i + 1,
                        a.name(),
                        b.name()
                    )))
                }
            };
            out.push(bound);
        }
        Ok(out)
    }

    /// `unnest(<declared udf>(..))` as a projection item: one plain scalar
    /// column per declared output field, named by the field names, each
    /// reading a lane of ONE shared ecall. `None` for any other shape;
    /// modifiers on the UNNEST itself refuse by name.
    pub(super) fn unnest_extern_columns(
        &self,
        e: &SqlExpr,
    ) -> Result<Option<Vec<(String, SExpr)>>, PrepareError> {
        let mut outer = e;
        while let SqlExpr::Nested(i) = outer {
            outer = i;
        }
        let SqlExpr::Function(uf) = outer else {
            return Ok(None);
        };
        if !uf.name.to_string().eq_ignore_ascii_case("unnest") {
            return Ok(None);
        }
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let FunctionArguments::List(list) = &uf.args else {
            return Ok(None);
        };
        // Measured: the oracle rejects every modifier on UNNEST itself
        // (DISTINCT/FILTER/in-call ORDER BY "not applicable to UNNEST",
        // OVER a catalog error, IGNORE NULLS a parser error) — refuse by
        // name rather than expanding as if unadorned (review round).
        if !list.clauses.is_empty()
            || list.duplicate_treatment.is_some()
            || uf.filter.is_some()
            || uf.over.is_some()
            || uf.null_treatment.is_some()
            || !uf.within_group.is_empty()
        {
            return Err(unsup(
                "modifier on UNNEST (DISTINCT, FILTER, ORDER BY, OVER and \
                 IGNORE NULLS are not applicable to UNNEST)"
                    .to_string(),
            ));
        }
        let [FunctionArg::Unnamed(FunctionArgExpr::Expr(arg))] = &list.args[..] else {
            return Ok(None);
        };
        let mut inner = arg;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        let SqlExpr::Function(f) = inner else {
            return Ok(None);
        };
        let Some((ext, spec)) = self.find_udf(&f.name.to_string()) else {
            return Ok(None);
        };
        if spec.ret_names.is_empty() {
            return Err(unsup(format!(
                "unnest of udf '{}': no declared output field names",
                spec.name
            )));
        }
        let args = self.bind_udf_args(f, spec)?;
        let site = self.site_for(f);
        Ok(Some(
            spec.ret_names
                .iter()
                .zip(&spec.rets)
                .enumerate()
                .map(|(j, (n, &rt))| {
                    (
                        n.clone(),
                        SExpr {
                            kind: SKind::ExternCall {
                                site,
                                ext,
                                args: args.clone(),
                                ret: j as u32,
                                whole: false,
                            },
                            ty: rt,
                            nullable: true,
                        },
                    )
                })
                .collect(),
        ))
    }

    /// A bare wide UDF call as a projection item expands to a whole-validity
    /// lane plus per-return nullable component lanes sharing one call site:
    /// a width-k (k >= 2) unnamed extern (the list boundary), or a
    /// NAMED extern at EVERY width (DuckDB registers named
    /// externs as STRUCT, so the boundary assembles a struct keyed by the
    /// returned declared names; empty names = list). `None` for anything
    /// else (width-1 unnamed calls stay ordinary scalar expressions). Lane
    /// names carry U+0001 (reserved at the SQL gate, so no user column can
    /// collide).
    pub(super) fn wide_extern_lanes(
        &self,
        e: &SqlExpr,
        base: &str,
    ) -> Result<Option<(Vec<(String, SExpr)>, Vec<String>)>, PrepareError> {
        let mut inner = e;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        let SqlExpr::Function(f) = inner else {
            return Ok(None);
        };
        let Some((ext, spec)) = self.find_udf(&f.name.to_string()) else {
            return Ok(None);
        };
        if spec.rets.len() < 2 && spec.ret_names.is_empty() {
            return Ok(None);
        }
        let args = self.bind_udf_args(f, spec)?;
        let site = self.site_for(f);
        let mut lanes = Vec::with_capacity(1 + spec.rets.len());
        lanes.push((
            format!("{base}\u{1}valid"),
            SExpr {
                kind: SKind::ExternCall {
                    site,
                    ext,
                    args: args.clone(),
                    ret: 0,
                    whole: true,
                },
                ty: Ty::I1,
                nullable: false,
            },
        ));
        for (j, &rt) in spec.rets.iter().enumerate() {
            lanes.push((
                format!("{base}\u{1}{j}"),
                SExpr {
                    kind: SKind::ExternCall {
                        site,
                        ext,
                        args: args.clone(),
                        ret: j as u32,
                        whole: false,
                    },
                    ty: rt,
                    nullable: true,
                },
            ));
        }
        Ok(Some((lanes, spec.ret_names.clone())))
    }

    /// A declared tree transform, called `name(<i64 id>, feat, feat, ..)` —
    /// the same shape as any other transform (`PythonTransform`'s implicit
    /// leading instance id, then the features), the difference being that
    /// this one lowers to the native kernel instead of an ecall.
    ///
    /// Features bind by POSITION, in the order the transform declared its
    /// `takes`. A call site is free to name its columns anything.
    pub(super) fn tree_call(
        &self,
        cat: usize,
        args: &[&SqlExpr],
    ) -> Result<SExpr, PrepareError> {
        let decl = &self.models[cat];
        let name = &decl.name;
        let Some((id, feats)) = args.split_first() else {
            return Err(PrepareError::Bind(format!(
                "udf '{name}' takes {} argument(s), the call passes 0",
                decl.takes.len() + 1
            )));
        };
        if feats.len() != decl.takes.len() {
            return Err(PrepareError::Bind(format!(
                "udf '{name}' takes {} argument(s), the call passes {}",
                decl.takes.len() + 1,
                args.len()
            )));
        }

        let mut bound: Vec<SExpr> = Vec::with_capacity(feats.len());
        for (i, fexpr) in feats.iter().enumerate() {
            let want = decl.takes[i];
            // A bare NULL feature is legal and means "missing" — the model
            // has an answer for that. It types as f64 like any other.
            let e = match self.expr_or_null(fexpr)? {
                None => null_of(Ty::F64),
                // The DECLARED type decides, not the argument's: DuckDB casts
                // the argument to the declaration before calling, so a BIGINT
                // column in a declared-DOUBLE lane reaches the model as
                // `float64(n)` and narrows from there. An i64 argument in a
                // declared-BIGINT lane is the only one that narrows in one
                // step, and only on a float32 grid.
                Some(e) => match (e.ty, want) {
                    (Ty::F64, Ty::F64) => e,
                    // DuckDB's implicit widening, exactly as `bind_udf_args`
                    // does it for every other UDF.
                    (Ty::I64, Ty::F64) => promote_f64(e),
                    (Ty::Dec(..), Ty::F64) => dec_to_float(e),
                    // How an integer reaches the compare depends on the grid
                    // the model set declared.
                    //
                    // On a float32 grid (sklearn) it must narrow in ONE step:
                    // `_validate_X_predict` does `int64 -> float32`, whereas
                    // `promote_f64` would give `float32(float64(n))` — two
                    // roundings, a whole float32 ULP off above 2**53. Below
                    // 2**53 `float64(n)` is exact and the two agree, which is
                    // what makes the narrowing safe for every integer feature
                    // rather than only large ones.
                    //
                    // On a float64 grid the integer reaches the compare
                    // exactly, and narrowing it would throw away precision
                    // that library had every right to keep — the grid is the
                    // PACKER's property, so it is declared, not assumed.
                    (Ty::I64, Ty::I64) => match decl.grid {
                        CompareGrid::F32 => narrow_f32(promote_f64(e)),
                        CompareGrid::F64 => promote_f64(e),
                    },
                    (a, b) => {
                        return Err(PrepareError::Bind(format!(
                            "udf '{name}' argument {} is {}, declared {}",
                            i + 2,
                            a.name(),
                            b.name()
                        )))
                    }
                },
            };
            bound.push(e);
        }

        // The instance id gates the RESULT: an unseen group has no model.
        // Feature nullability deliberately does not propagate.
        let bid = match self.expr_or_null(id)? {
            Some(b) => b,
            // A bare NULL id: DuckDB folds the call only when every
            // argument is a constant (then it is the NULL the binder elides
            // its siblings under). Over a column feature the call runs per
            // row and answers NULL, and a trapping sibling still traps
            // (nightly seed 4316677: `trees(NULL, c0, ..) * ln(c0)`).
            None if bound.iter().all(bind_foldable) => return Ok(null_of(Ty::F64)),
            None => null_of(Ty::I64),
        };
        let bid = match bid.ty {
            t if t.is_int() => bid,
            other => {
                return Err(PrepareError::Bind(format!(
                    "udf '{name}' argument 1 is {}, declared the instance id (i64)",
                    other.name()
                )))
            }
        };
        let nullable = bid.nullable;

        let mut refs = self.model_refs.borrow_mut();
        let model = match refs.iter().position(|r| *r == cat as u32) {
            Some(i) => i as u32,
            None => {
                refs.push(cat as u32);
                (refs.len() - 1) as u32
            }
        };
        Ok(SExpr {
            kind: SKind::TreePredict {
                model,
                id: Box::new(bid),
                feats: bound,
            },
            ty: Ty::F64,
            nullable,
        })
    }

    /// Index of a declared tree transform by call-site name, matched
    /// case-insensitively like [`Self::find_udf`] — they share a namespace.
    pub(super) fn find_tree(&self, name: &str) -> Option<usize> {
        self.models
            .iter()
            .position(|m| m.name.eq_ignore_ascii_case(name))
    }
}
