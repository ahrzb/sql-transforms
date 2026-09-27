//! Struct values: field access, subscripts, struct_pack.

use super::*;

impl Binder<'_> {
    /// s[i] / array_extract / list_extract on a bound VARCHAR subject:
    /// exec handles negatives (len+1+i), 0/out-of-range -> '' and the
    /// runtime +-2^32 offset trap (pins-wave5/subscripts-extended.json).
    pub(super) fn apply_extract(
        &self,
        name: &str,
        bs: SExpr,
        n: &SqlExpr,
    ) -> Result<SExpr, PrepareError> {
        if bs.ty != Ty::Str {
            // The LIST overload has different out-of-range semantics
            // (NULL, not '') — only the VARCHAR path is served.
            return Err(unsup(format!(
                "{name} on {} (only VARCHAR subscripts are served)",
                bs.ty.name()
            )));
        }
        let Some(bn) = self.expr_or_null(n)? else {
            return Ok(null_of(Ty::Str));
        };
        if !bn.ty.is_int() {
            return Err(PrepareError::Bind(format!(
                "no function matches {name}(str, {})",
                bn.ty.name()
            )));
        }
        let nullable = bs.nullable || bn.nullable;
        Ok(SExpr {
            kind: SKind::Str2i {
                op: StrOp2i::Extract,
                a: Box::new(bs),
                n: Box::new(bn),
            },
            ty: Ty::Str,
            nullable,
        })
    }

    /// s[a:b] / array_slice / list_slice on a bound VARCHAR subject. Open
    /// bounds are pure syntax ([:b] == [1:b], [a:] == [a:-1]); a NULL bound
    /// is NOT open — it nulls the result (pins-wave5/slices.json).
    pub(super) fn apply_slice(
        &self,
        name: &str,
        bs: SExpr,
        lo: Option<&SqlExpr>,
        hi: Option<&SqlExpr>,
    ) -> Result<SExpr, PrepareError> {
        if bs.ty != Ty::Str {
            return Err(unsup(format!(
                "{name} on {} (only VARCHAR subscripts are served)",
                bs.ty.name()
            )));
        }
        let bind_bound = |e: Option<&SqlExpr>, open: i64| -> Result<Option<SExpr>, PrepareError> {
            match e {
                None => Ok(Some(lit_i64(open))),
                Some(e) => self.expr_or_null(e),
            }
        };
        let (blo, bhi) = (bind_bound(lo, 1)?, bind_bound(hi, -1)?);
        let (Some(blo), Some(bhi)) = (blo, bhi) else {
            return Ok(null_of(Ty::Str));
        };
        for e in [&blo, &bhi] {
            if !e.ty.is_int() {
                return Err(PrepareError::Bind(format!(
                    "no function matches {name}(str, {}, {})",
                    blo.ty.name(),
                    bhi.ty.name()
                )));
            }
        }
        let nullable = bs.nullable || blo.nullable || bhi.nullable;
        Ok(SExpr {
            kind: SKind::Sslice {
                a: Box::new(bs),
                lo: Box::new(blo),
                hi: Box::new(bhi),
            },
            ty: Ty::Str,
            nullable,
        })
    }

    /// Field access over struct_pack is a pure bind-time desugar —
    /// extracting a field of a just-packed struct IS binding that field's
    /// expression. Handles the dot form `(struct_pack(a := e)).a` (chains
    /// peel one Dot per pass; re-entry desugars the rest) and returns the
    /// SUBSTITUTE AST. `Ok(None)` when the shape isn't
    /// field-access-over-struct-pack; the missing-key refusal uses DuckDB's
    /// wording. A bare-NULL field rides the adoptable-SQLNULL channel by
    /// construction — the substitute AST re-binds wherever the ORIGINAL
    /// stood (`- (struct_pack(a := NULL)).a` is BIGINT on DuckDB, the bare
    /// field INTEGER; measured).
    pub(super) fn desugar_struct_field(
        &self,
        e: &SqlExpr,
    ) -> Result<Option<SqlExpr>, PrepareError> {
        let SqlExpr::CompoundFieldAccess { root, access_chain } = e else {
            return Ok(None);
        };
        let Some((AccessExpr::Dot(SqlExpr::Identifier(id)), rest)) =
            access_chain.split_first()
        else {
            return Ok(None);
        };
        let mut base: &SqlExpr = root;
        while let SqlExpr::Nested(i) = base {
            base = i;
        }
        let SqlExpr::Function(f) = base else {
            return Ok(None);
        };
        let Some(field) = self.struct_pack_field(f, &id.value)? else {
            return Ok(None);
        };
        if rest.is_empty() {
            return Ok(Some(field.clone()));
        }
        Ok(Some(SqlExpr::CompoundFieldAccess {
            root: Box::new(SqlExpr::Nested(Box::new(field.clone()))),
            access_chain: rest.to_vec(),
        }))
    }

    /// Field access over a struct COLUMN in its non-dotted spellings --
    /// `s['f']`, `s.n['x']`, `s['n'].x`, `(s).f`, `struct_extract(s, 'f')`,
    /// and chains of them -- as the segment path the dotted spelling binds
    /// through, so each reads exactly what `s.f` reads (same walk, same
    /// NULL propagation, same missing-key error; field matching is
    /// case-insensitive in both, measured). Only a string-literal key is a
    /// field name; an integer key on a named struct is DuckDB's binder
    /// error and stays on the refusing path.
    ///
    /// The ROOT's dotted run resolves by the ordinary rules, but a key after
    /// it is always a FIELD, never a column of a relation: `v['x']` with
    /// `v` a relation in scope is not `v.x`. Such a root is left alone.
    pub(super) fn struct_access_path(&self, e: &SqlExpr) -> Option<Vec<Ident>> {
        self.struct_access_path_beside(e, None)
    }

    /// [`Self::struct_access_path`] with one more relation name in scope:
    /// the static table a JOIN ON binds against, which is not in
    /// `self.joins` until its ON has bound.
    pub(super) fn struct_access_path_beside(
        &self,
        e: &SqlExpr,
        rel: Option<&str>,
    ) -> Option<Vec<Ident>> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let (base, fields) = match e {
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                // Leading dots on an unparenthesized root belong to its
                // dotted run; everything from the first key on is a field.
                let mut root_run = Vec::new();
                let mut chain = access_chain.as_slice();
                if matches!(
                    root.as_ref(),
                    SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_)
                ) {
                    while let [AccessExpr::Dot(SqlExpr::Identifier(i)), rest @ ..] = chain {
                        root_run.push(i.clone());
                        chain = rest;
                    }
                }
                let base = self.struct_access_base(root, root_run, rel)?;
                (base, chain_fields(chain)?)
            }
            SqlExpr::Function(f) if f.name.to_string().eq_ignore_ascii_case("struct_extract") => {
                let FunctionArguments::List(list) = &f.args else {
                    return None;
                };
                let [
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(target)),
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(v))),
                ] = list.args.as_slice()
                else {
                    return None;
                };
                let SqlValue::SingleQuotedString(field) = &v.value else {
                    return None;
                };
                (self.struct_access_base(target, Vec::new(), rel)?, vec![Ident::new(field)])
            }
            _ => return None,
        };
        if fields.is_empty() {
            return None;
        }
        Some([base, fields].concat())
    }

    /// The path a struct access starts from: a column reference's dotted
    /// run (plus `more`, dots that continue it), or a nested access. A run
    /// ending on a relation name in scope is refused (see
    /// [`Self::struct_access_path`]).
    pub(super) fn struct_access_base(
        &self,
        root: &SqlExpr,
        more: Vec<Ident>,
        rel: Option<&str>,
    ) -> Option<Vec<Ident>> {
        let mut path = match root {
            SqlExpr::Nested(i) if more.is_empty() => return self.struct_access_base(i, more, rel),
            SqlExpr::Identifier(i) => vec![i.clone()],
            SqlExpr::CompoundIdentifier(p) => p.clone(),
            SqlExpr::CompoundFieldAccess { .. } | SqlExpr::Function(_) if more.is_empty() => {
                return self.struct_access_path_beside(root, rel);
            }
            _ => return None,
        };
        path.extend(more);
        let last = path.last()?;
        let is_rel = last.value.eq_ignore_ascii_case(&self.this_name)
            || self.joins.iter().any(|sj| last.value.eq_ignore_ascii_case(&sj.name))
            || rel.is_some_and(|r| last.value.eq_ignore_ascii_case(r));
        (!is_rel).then_some(path)
    }

    /// The struct_extract SPELLING of the same desugar:
    /// `struct_extract(struct_pack(a := e), 'a')` -> the field's AST.
    pub(super) fn desugar_struct_extract(
        &self,
        f: &sqlparser::ast::Function,
    ) -> Result<Option<SqlExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if !f.name.to_string().eq_ignore_ascii_case("struct_extract") {
            return Ok(None);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        let [
            FunctionArg::Unnamed(FunctionArgExpr::Expr(target)),
            FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(v))),
        ] = list.args.as_slice()
        else {
            return Ok(None);
        };
        let SqlValue::SingleQuotedString(field) = &v.value else {
            return Ok(None);
        };
        let mut base: &SqlExpr = target;
        while let SqlExpr::Nested(i) = base {
            base = i;
        }
        let SqlExpr::Function(inner) = base else {
            return Ok(None);
        };
        Ok(self.struct_pack_field(inner, field)?.cloned())
    }

    /// The named field of a plain `struct_pack(...)` call, ASCII-case-
    /// insensitively (DuckDB's struct key matching). `Ok(None)` when `f`
    /// isn't a plain struct_pack; a struct_pack MISSING the key refuses
    /// with DuckDB's wording.
    pub(super) fn struct_pack_field<'e>(
        &self,
        f: &'e sqlparser::ast::Function,
        field: &str,
    ) -> Result<Option<&'e SqlExpr>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        if !f.name.to_string().eq_ignore_ascii_case("struct_pack")
            || f.uses_odbc_syntax
            || !matches!(f.parameters, FunctionArguments::None)
            || f.filter.is_some()
            || f.null_treatment.is_some()
            || f.over.is_some()
        {
            return Ok(None);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        for a in &list.args {
            if let FunctionArg::Named { name, arg, .. } = a {
                if name.value.eq_ignore_ascii_case(field) {
                    let FunctionArgExpr::Expr(v) = arg else {
                        return Ok(None);
                    };
                    return Ok(Some(v));
                }
            }
        }
        Err(PrepareError::Bind(format!(
            "Could not find key \"{field}\" in struct"
        )))
    }

    /// A struct-VALUED projection item — `struct_pack(n := e, ...)`, or that
    /// guarded by `CASE WHEN g IS NULL THEN NULL ELSE ... END` (θ export).
    /// Lowered to the same wide-lane shape a named extern uses: a
    /// whole-validity lane (false = the whole struct is NULL, distinct from
    /// a struct of NULLs) plus one component lane per field. `None` when
    /// this isn't that shape.
    ///
    /// Stricter than DuckDB, deliberately — unnamed args
    /// (`struct_pack(i)` infers the field name there) and
    /// leading-underscore fields bind on the oracle; the recognizer
    /// refuses both (projection-loop-only, pydantic model boundary).
    /// Field ACCESS over struct_pack is `desugar_struct_field`'s
    /// bind-time desugar and never reaches this recognizer.
    pub(super) fn struct_pack_lanes(
        &self,
        e: &SqlExpr,
        base: &str,
    ) -> Result<Option<(Vec<(String, SExpr)>, Vec<String>)>, PrepareError> {
        let mut inner = e;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        // The guard arm: exactly the shape the θ rewrite emits. Anything
        // else with a struct in it falls through and refuses by name.
        let (guard, packed) = match inner {
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } if operand.is_none() && conditions.len() == 1 => {
                let arm = &conditions[0];
                // The oracle's own serialization parenthesizes both arms.
                let mut res = &arm.result;
                while let SqlExpr::Nested(i) = res {
                    res = i;
                }
                let is_null_lit = matches!(
                    res,
                    SqlExpr::Value(v) if matches!(v.value, SqlValue::Null)
                );
                match (is_null_lit, else_result) {
                    (true, Some(alt)) => (Some(&arm.condition), &**alt),
                    _ => return Ok(None),
                }
            }
            other => (None, other),
        };
        let mut packed_inner = packed;
        while let SqlExpr::Nested(i) = packed_inner {
            packed_inner = i;
        }
        let SqlExpr::Function(f) = packed_inner else {
            return Ok(None);
        };
        if !f.name.to_string().eq_ignore_ascii_case("struct_pack") {
            return Ok(None);
        }
        use sqlparser::ast::{FunctionArg, FunctionArguments};
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        // Every field must be NAMED — an unnamed struct_pack arg is a
        // binder error in the oracle too.
        let mut names: Vec<String> = Vec::with_capacity(list.args.len());
        let mut values: Vec<&SqlExpr> = Vec::with_capacity(list.args.len());
        for a in &list.args {
            match a {
                FunctionArg::Named { name, arg, .. } => {
                    let sqlparser::ast::FunctionArgExpr::Expr(v) = arg else {
                        return Ok(None);
                    };
                    names.push(name.value.clone());
                    values.push(v);
                }
                _ => return Ok(None),
            }
        }
        if names.is_empty() {
            return Ok(None);
        }
        for (i, n) in names.iter().enumerate() {
            // Measured: DuckDB's binder rejects a duplicate struct entry
            // name, case-insensitively — never serve what batch cannot.
            if names[..i].iter().any(|m| m.eq_ignore_ascii_case(n)) {
                return Err(PrepareError::Bind(format!(
                    "duplicate struct entry name \"{n}\""
                )));
            }
            // A _-leading field becomes a pydantic private attribute, so
            // the row model would silently drop it while batch serves it.
            if n.starts_with('_') {
                return Err(unsup(format!(
                    "struct field '{n}' cannot cross the row-path model \
                     boundary (a leading underscore is private) — rename it"
                )));
            }
        }
        // Whole-struct validity: the guard's IS NOT NULL, else always-true.
        let valid = match guard {
            None => SExpr {
                kind: SKind::Lit(Lit::I1(true)),
                ty: Ty::I1,
                nullable: false,
            },
            Some(g) => {
                let mut guard_inner = g;
                while let SqlExpr::Nested(i) = guard_inner {
                    guard_inner = i;
                }
                let SqlExpr::IsNull(target) = guard_inner else {
                    return Ok(None);
                };
                let bound = match self.expr_or_null(target)? {
                    None => {
                        // A provably-NULL guard: the struct is always NULL.
                        SExpr {
                            kind: SKind::Lit(Lit::I1(false)),
                            ty: Ty::I1,
                            nullable: false,
                        }
                    }
                    Some(t) => SExpr {
                        kind: SKind::IsNull {
                            negated: true,
                            inner: Box::new(t),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    },
                };
                bound
            }
        };
        let mut lanes = Vec::with_capacity(1 + values.len());
        lanes.push((format!("{base}\u{1}valid"), valid));
        for (j, v) in values.iter().enumerate() {
            // struct_pack(a := NULL) is STRUCT(a INTEGER) on DuckDB —
            // SQLNULL's int32 home.
            let bound = match self.expr_or_null(v)? {
                None => null_of(Ty::I32),
                Some(x) => fold(x),
            };
            lanes.push((format!("{base}\u{1}{j}"), bound));
        }
        Ok(Some((lanes, names)))
    }
}
