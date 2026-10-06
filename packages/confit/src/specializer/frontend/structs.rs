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
        let field: Option<SqlExpr> = match base {
            // A read of a struct DuckDB folds to NULL is a bare NULL.
            SqlExpr::Case { .. } if self.struct_folds_to_null(base) => return Ok(Some(null_literal())),
            // The whole chain moves into each arm: `(CASE .. END).q.r`.
            SqlExpr::Case { .. } => return Ok(Some(case_field(base, access_chain))),
            // The rest of the chain reads into the picked field.
            SqlExpr::Function(f) => self.struct_pack_field(f, &id.value, rest)?,
            _ => return Ok(None),
        };
        Ok(field)
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
    /// The ROOT's dotted run resolves by the ordinary rules, and a key after
    /// it is a FIELD. A bare root binds as a column first, as on DuckDB; only
    /// a relation name no column shares is that relation's row struct, so
    /// `t['a']` reads `t.a` (see [`Self::struct_access_base`]).
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
        // A bare relation name no column shares is that relation's row
        // struct on DuckDB, so `t['a']` and `(t).a` read `t.a` (measured,
        // LEFT misses included). A column of the same name wins there, and
        // that spelling keeps refusing: the path below would resolve the
        // relation first.
        if is_rel && path.len() == 1 && !self.binds_as_column(&last.value) {
            return Some(path);
        }
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
        match base {
            SqlExpr::Case { .. } if self.struct_folds_to_null(base) => Ok(Some(null_literal())),
            SqlExpr::Case { .. } => Ok(Some(case_field(
                base,
                &[AccessExpr::Dot(SqlExpr::Identifier(Ident::new(field.clone())))],
            ))),
            SqlExpr::Function(inner) => self.struct_pack_field(inner, field, &[]),
            _ => Ok(None),
        }
    }

    /// The read of a named field of a plain `struct_pack(...)` call, as the
    /// AST it binds (see [`field_read`]), matching ASCII-case-insensitively
    /// (DuckDB's struct key matching). `Ok(None)` when `f` isn't a plain
    /// struct_pack; a struct_pack MISSING the key refuses with DuckDB's
    /// wording.
    pub(super) fn struct_pack_field(
        &self,
        f: &sqlparser::ast::Function,
        field: &str,
        rest: &[AccessExpr],
    ) -> Result<Option<SqlExpr>, PrepareError> {
        Ok(self
            .struct_pack_values(f, field)?
            .map(|(values, pick)| field_read(f, &values, pick, rest)))
    }

    /// A plain `struct_pack(...)`'s field values in order, and the index of
    /// `field` among them; see [`Self::struct_pack_field`].
    pub(super) fn struct_pack_values<'f>(
        &self,
        f: &'f sqlparser::ast::Function,
        field: &str,
    ) -> Result<Option<(Vec<&'f SqlExpr>, usize)>, PrepareError> {
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
        let mut values = Vec::with_capacity(list.args.len());
        let mut pick = None;
        for a in &list.args {
            let (FunctionArg::Named { name, arg: FunctionArgExpr::Expr(v), .. }
            | FunctionArg::ExprNamed {
                name: SqlExpr::Identifier(name),
                arg: FunctionArgExpr::Expr(v),
                ..
            }) = a
            else {
                return Ok(None);
            };
            if pick.is_none() && name.value.eq_ignore_ascii_case(field) {
                pick = Some(values.len());
            }
            values.push(v);
        }
        let Some(pick) = pick else {
            return Err(PrepareError::Bind(format!(
                "Could not find key \"{field}\" in struct"
            )));
        };
        Ok(Some((values, pick)))
    }

    /// A list- or struct-valued projection item as its out lanes and their
    /// shape: a list literal, or a struct value ([`Self::struct_value`]).
    /// `None` when `e` is neither.
    pub(super) fn wide_item(
        &self,
        e: &SqlExpr,
        base: &str,
    ) -> Result<Option<(Vec<(String, SExpr)>, super::super::WideShape)>, PrepareError> {
        if let Some(r) = self.list_item(e, base)? {
            return Ok(Some(r));
        }
        Ok(self.struct_value(e)?.map(|v| {
            debug_assert!(matches!(v, OutVal::Struct { .. }), "a struct value");
            v.into_lanes(base)
        }))
    }

    /// A list literal as an output column — unnamed lanes, the boundary an
    /// extern's list return crosses — bare or guarded by `CASE WHEN g IS
    /// NULL THEN NULL ELSE [...] END`: the whole-value validity, then one
    /// lane per element. `None` when `e` is not that shape.
    fn list_item(
        &self,
        e: &SqlExpr,
        base: &str,
    ) -> Result<Option<(Vec<(String, SExpr)>, super::super::WideShape)>, PrepareError> {
        let mut inner = e;
        while let SqlExpr::Nested(i) = inner {
            inner = i;
        }
        // The guard arm: exactly the shape the θ rewrite emits.
        let (guard, packed) = match inner {
            SqlExpr::Case {
                operand: None,
                conditions,
                else_result: Some(alt),
                ..
            } if conditions.len() == 1 => {
                let arm = &conditions[0];
                // The oracle's own serialization parenthesizes both arms.
                let mut res = &arm.result;
                while let SqlExpr::Nested(i) = res {
                    res = i;
                }
                if !matches!(res, SqlExpr::Value(v) if matches!(v.value, SqlValue::Null)) {
                    return Ok(None);
                }
                (Some(&arm.condition), &**alt)
            }
            other => (None, other),
        };
        let Some(elems) = list_literal(packed) else {
            return Ok(None);
        };
        if elems.len() < 2 {
            return Err(unsup(
                "a one-element list as an output column (a width-1 lane is a scalar \
                 at the boundary)",
            ));
        }
        // Whole-value validity: the guard's IS NOT NULL, else always-true.
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
                match self.expr_or_null(target)? {
                    // A provably-NULL guard: the list is always NULL.
                    None => SExpr {
                        kind: SKind::Lit(Lit::I1(false)),
                        ty: Ty::I1,
                        nullable: false,
                    },
                    Some(t) => SExpr {
                        kind: SKind::IsNull {
                            negated: true,
                            inner: Box::new(t),
                        },
                        ty: Ty::I1,
                        nullable: false,
                    },
                }
            }
        };
        let mut lanes = Vec::with_capacity(1 + elems.len());
        for (j, b) in self.list_elements(&elems)?.into_iter().map(fold).enumerate() {
            // DuckDB evaluates the ELSE arm only on the rows that reach it.
            let b = if guard.is_some() {
                let ty = b.ty;
                SExpr {
                    kind: SKind::Case {
                        arms: vec![(valid.clone(), b)],
                        default: None,
                    },
                    ty,
                    nullable: true,
                }
            } else {
                b
            };
            lanes.push((format!("{base}\u{1}{j}"), b));
        }
        let width = lanes.len() as u32;
        lanes.insert(0, (format!("{base}\u{1}valid"), valid));
        Ok(Some((lanes, super::super::WideShape::List(width))))
    }
}


/// The internal call a field read over `struct_pack` binds as when the
/// struct has other fields: `__cf_seq(pick, f1, ..., fn)`, every field in
/// its order and the index of the one read. DuckDB builds every field of a
/// struct_pack, so a sibling's trap fires even though one field is read
/// (`(struct_pack(p := a, q := a * 9223372036854775807)).p` traps, measured
/// on 1.5.5); the binder keeps the siblings that can trap
/// (`Binder::seq`). The name is reserved: user SQL spelling it refuses.
pub(super) const SEQ_MARKER: &str = "__cf_seq";

fn field_read(
    f: &sqlparser::ast::Function,
    values: &[&SqlExpr],
    pick: usize,
    rest: &[AccessExpr],
) -> SqlExpr {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments, ObjectName};
    let picked = if rest.is_empty() {
        values[pick].clone()
    } else {
        SqlExpr::CompoundFieldAccess {
            root: Box::new(SqlExpr::Nested(Box::new(values[pick].clone()))),
            access_chain: rest.to_vec(),
        }
    };
    if values.len() == 1 {
        return picked;
    }
    let mut seq = f.clone();
    seq.name = ObjectName::from(vec![Ident::new(SEQ_MARKER)]);
    let FunctionArguments::List(list) = &mut seq.args else {
        unreachable!("a struct_pack has an argument list");
    };
    let index = SqlExpr::Value(SqlValue::Number(pick.to_string(), false).into());
    let fields = values
        .iter()
        .enumerate()
        .map(|(i, v)| if i == pick { picked.clone() } else { (*v).clone() });
    list.args = std::iter::once(index)
        .chain(fields)
        .map(|e| FunctionArg::Unnamed(FunctionArgExpr::Expr(e)))
        .collect();
    SqlExpr::Function(seq)
}

/// The literal `NULL`.
pub(super) fn null_literal() -> SqlExpr {
    SqlExpr::Value(SqlValue::Null.into())
}

/// A field read over a CASE, as the CASE of the field reads:
/// `(CASE WHEN c THEN s1 ELSE s2 END).f` is `CASE WHEN c THEN (s1).f ELSE
/// (s2).f END`, a longer access chain moving into each arm whole. The arm taken is the arm whose struct DuckDB would build, and
/// a NULL arm (or no ELSE) is a NULL struct, whose field is NULL; each arm's
/// read then binds by the ordinary rules (over struct_pack: every field
/// built, see [`SEQ_MARKER`]).
pub(super) fn case_field(case: &SqlExpr, chain: &[AccessExpr]) -> SqlExpr {
    let read = |r: &SqlExpr| -> SqlExpr {
        if matches!(r, SqlExpr::Value(v) if matches!(v.value, SqlValue::Null)) {
            return r.clone();
        }
        SqlExpr::CompoundFieldAccess {
            root: Box::new(SqlExpr::Nested(Box::new(r.clone()))),
            access_chain: chain.to_vec(),
        }
    };
    let mut out = case.clone();
    if let SqlExpr::Case {
        conditions,
        else_result,
        ..
    } = &mut out
    {
        for w in conditions.iter_mut() {
            w.result = read(&w.result);
        }
        if let Some(e) = else_result {
            **e = read(e);
        }
    }
    out
}
