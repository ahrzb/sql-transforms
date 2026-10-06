//! Star expansion, COLUMNS(...), and output naming.

use super::*;

/// DuckDB names an unaliased projection after the identifier it selects
/// (spelling preserved), else after the expression's text.
pub(super) fn default_name(e: &SqlExpr) -> String {
    match e {
        SqlExpr::Identifier(ident) => ident.value.clone(),
        SqlExpr::CompoundIdentifier(parts) if !parts.is_empty() => {
            parts.last().unwrap().value.clone()
        }
        // DuckDB parenthesizes a dot access whose left side is already a
        // subscripted expression (measured: `s['n'].x` is named
        // `(s['n']).x`, while `s.n['x']` and `s['n']['x']` print as written).
        SqlExpr::CompoundFieldAccess { root, access_chain } => {
            if let Some(n) = super::naming::duck_name(e) {
                return n;
            }
            let mut name = root.to_string();
            let mut keyed = false;
            for acc in access_chain {
                match acc {
                    AccessExpr::Dot(d) if keyed => name = format!("({name}).{d}"),
                    AccessExpr::Dot(d) => name = format!("{name}.{d}"),
                    AccessExpr::Subscript(sub) => {
                        keyed = true;
                        name = format!("{name}[{sub}]");
                    }
                }
            }
            name
        }
        // DuckDB prints the parsed expression; a form the printer does not
        // model keeps the SQL text.
        other => super::naming::duck_name(other).unwrap_or_else(|| other.to_string()),
    }
}

/// DuckDB's boundary rename for duplicate output names
/// (pins-wave5/dup-names-client-contract.json): left-to-right after star
/// expansion, first occurrence keeps its name, later ones get
/// `<own-original-case-name>_N` with the smallest free N; the collision
/// check is case-insensitive and covers generated candidates too
/// (id,ID -> id,ID_1; id,id,id_1 -> id,id_1,id_1_1). Identical to what
/// DuckDB itself does at every subquery/CTE/CTAS boundary and in .df().
pub(super) fn dedup_output_names(cols: &mut [Col]) {
    dedup_names(cols.iter_mut().map(|c| &mut c.name));
}

/// [`dedup_output_names`] over the output FIELDS: a list or struct field
/// counts once, under its own name, where its first lane stands.
pub(super) fn dedup_field_names(cols: &mut [Col], wide: &mut [super::super::WideOut]) {
    let mut names: Vec<&mut String> = Vec::new();
    let mut w = wide.iter_mut().peekable();
    let mut cols = cols.iter_mut().enumerate();
    while let Some((i, c)) = cols.next() {
        if w.peek().is_some_and(|wo| wo.first as usize == i) {
            let wo = w.next().expect("peeked");
            for _ in 1..wo.shape.lanes() {
                cols.next();
            }
            names.push(&mut wo.name);
            continue;
        }
        names.push(&mut c.name);
    }
    dedup_names(names.into_iter());
}

fn dedup_names<'n>(names: impl Iterator<Item = &'n mut String>) {
    let mut seen = std::collections::HashSet::new();
    for name in names {
        if seen.insert(name.to_lowercase()) {
            continue;
        }
        let mut n = 1;
        loop {
            let cand = format!("{name}_{n}");
            if seen.insert(cand.to_lowercase()) {
                *name = cand;
                break;
            }
            n += 1;
        }
    }
}

/// A star-expansion lane: a real scalar lane, a struct column (its node,
/// turned into lanes only if it survives EXCLUDE and the name filters, so
/// an excluded struct mints no presence lane), a REPLACE value, or an
/// opaque column (kept under its ORIGINAL name so EXCLUDE / REPLACE / name
/// filters can still remove it). An opaque one surviving to the output is
/// the named unsupported error — deferred so COLUMNS('re') can filter
/// first.
pub(super) enum StarLane {
    Real(SExpr),
    Struct(NodeRef),
    Value(OutVal),
    Opaque(String),
}

/// `AS x(p, q)` on a joined relation: a positional rename over the DECLARED
/// columns (the star list), the same rule the driving-table arm applies. A
/// PARTIAL list is legal (prefix rename); more names than declared columns is
/// DuckDB's arity bind error; a name landing on a struct or non-vocabulary
/// column has no plain lane to rename and refuses. Returns None when there is
/// no column list (plain `AS x` renames only the scope, handled by callers).
pub(super) fn apply_column_alias(
    st: &StaticTable,
    alias: Option<&sqlparser::ast::TableAlias>,
) -> Result<Option<StaticTable>, PrepareError> {
    let Some(a) = alias else { return Ok(None) };
    if a.columns.is_empty() {
        return Ok(None);
    }
    if a.columns.len() > st.star.len() {
        return Err(PrepareError::Bind(format!(
            "table \"{}\" has {} columns available but {} columns specified",
            st.name,
            st.star.len(),
            a.columns.len()
        )));
    }
    let mut t = st.clone();
    for (def, sc) in a.columns.iter().zip(&t.star) {
        match sc {
            super::super::plan::StarCol::Real(ci) => {
                t.cols[*ci as usize].name = def.name.value.clone()
            }
            super::super::plan::StarCol::Opaque(oname) => {
                return Err(unsup(format!(
                    "column-list alias over non-scalar column '{oname}'"
                )))
            }
        }
    }
    Ok(Some(t))
}


/// Bind-time LIKE over column NAMES for star filters: `%`/`_` over
/// codepoints, no ESCAPE (an ESCAPE clause after a star filter does not
/// parse), ci = ILIKE's Unicode case fold.
pub(super) fn like_match(s: &str, p: &str, ci: bool) -> bool {
    let norm = |x: &str| {
        if ci {
            x.to_lowercase()
        } else {
            x.to_string()
        }
    };
    let s: Vec<char> = norm(s).chars().collect();
    let p: Vec<char> = norm(p).chars().collect();
    let (mut si, mut pi) = (0usize, 0usize);
    let (mut bt_p, mut bt_s) = (usize::MAX, 0usize);
    while si < s.len() {
        if pi < p.len() && p[pi] == '%' {
            bt_p = pi;
            pi += 1;
            bt_s = si;
        } else if pi < p.len() && (p[pi] == '_' || p[pi] == s[si]) {
            pi += 1;
            si += 1;
        } else if bt_p != usize::MAX {
            bt_s += 1;
            si = bt_s;
            pi = bt_p + 1;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '%' {
        pi += 1;
    }
    pi == p.len()
}

/// A star name filter, decoded from the ILIKE slot (rewrite.rs encodes
/// LIKE / NOT LIKE / GLOB / NOT ILIKE there with a \u{1} marker; an
/// unmarked pattern is a genuine * ILIKE).
pub(super) enum StarFilter {
    Like { ci: bool, neg: bool },
    Glob,
    /// pins-waveB/: positive = unanchored RE2 SEARCH over names; NOT =
    /// NOT full-match — independent predicates, never complements.
    Similar { neg: bool },
}

/// Decoded star filter + any EXCLUDE entries the rewrite absorbed into the
/// marker (sqlparser parses ILIKE and EXCLUDE as mutually exclusive).
pub(super) struct DecodedFilter {
    op: StarFilter,
    pat: String,
    excludes: Vec<(Option<String>, String)>,
}

pub(super) fn decode_star_filter(pattern: &str) -> DecodedFilter {
    if let Some(rest) = pattern.strip_prefix('\u{1}') {
        for (code, op) in [
            ("L:", StarFilter::Like { ci: false, neg: false }),
            ("NL:", StarFilter::Like { ci: false, neg: true }),
            ("NI:", StarFilter::Like { ci: true, neg: true }),
            ("G:", StarFilter::Glob),
            ("S:", StarFilter::Similar { neg: false }),
            ("NS:", StarFilter::Similar { neg: true }),
        ] {
            let Some(body) = rest.strip_prefix(code) else {
                continue;
            };
            let (exc, pat) = body.split_once('\u{2}').unwrap_or(("", body));
            let excludes = exc
                .split(',')
                .filter(|e| !e.is_empty())
                .map(|e| match e.split_once('.') {
                    Some((t, c)) => (Some(t.to_string()), c.to_string()),
                    None => (None, e.to_string()),
                })
                .collect();
            return DecodedFilter {
                op,
                pat: pat.to_string(),
                excludes,
            };
        }
    }
    DecodedFilter {
        op: StarFilter::Like { ci: true, neg: false },
        pat: pattern.to_string(),
        excludes: Vec::new(),
    }
}

/// Find the `__glob_pat(...)` identity marker (rewrite.rs) in a LIKE
/// pattern tree and return the tree with the marker unwrapped; None means
/// "no marker — a plain LIKE".
pub(super) fn strip_glob_marker(e: &SqlExpr) -> Option<SqlExpr> {
    fn marker_arg(e: &SqlExpr) -> Option<&SqlExpr> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let SqlExpr::Function(f) = e else { return None };
        if !f.name.to_string().eq_ignore_ascii_case("__glob_pat") {
            return None;
        }
        let FunctionArguments::List(list) = &f.args else {
            return None;
        };
        let [FunctionArg::Unnamed(FunctionArgExpr::Expr(inner))] = &list.args[..] else {
            return None;
        };
        Some(inner)
    }
    if let Some(inner) = marker_arg(e) {
        return Some(inner.clone());
    }
    match e {
        SqlExpr::BinaryOp { left, op, right } => {
            if let Some(l) = strip_glob_marker(left) {
                Some(SqlExpr::BinaryOp {
                    left: Box::new(l),
                    op: op.clone(),
                    right: right.clone(),
                })
            } else {
                strip_glob_marker(right).map(|r| SqlExpr::BinaryOp {
                    left: left.clone(),
                    op: op.clone(),
                    right: Box::new(r),
                })
            }
        }
        SqlExpr::Nested(inner) => strip_glob_marker(inner).map(|i| SqlExpr::Nested(Box::new(i))),
        _ => None,
    }
}

impl Binder<'_> {
    /// Expand `*` / `tbl.*` per DuckDB's measured semantics (1.5.5,
    /// pins-wave5/): FROM order, declared column order within a table; grammar
    /// order EXCLUDE -> REPLACE -> RENAME with the name filter applying
    /// after EXCLUDE only. Duplicate output names across the star survive
    /// here and are renamed by [`dedup_output_names`] (DuckDB's own
    /// boundary-rename contract).
    pub(super) fn expand_star(
        &self,
        qualifier: Option<&str>,
        opts: &sqlparser::ast::WildcardAdditionalOptions,
    ) -> Result<Vec<(String, OutVal)>, PrepareError> {
        self.finalize_star(self.expand_star_lanes(qualifier, opts)?)
    }

    /// The surviving star columns as output values: a struct column
    /// becomes its whole value, and an opaque one refuses by name.
    pub(super) fn finalize_star(
        &self,
        cols: Vec<(String, StarLane)>,
    ) -> Result<Vec<(String, OutVal)>, PrepareError> {
        cols.into_iter()
            .map(|(n, l)| match l {
                StarLane::Real(e) => Ok((n, OutVal::Scalar(e))),
                StarLane::Struct(node) => Ok((n, self.node_value(&node)?)),
                StarLane::Value(v) => Ok((n, v)),
                StarLane::Opaque(orig) => Err(unsup(format!(
                    "column '{orig}' has a non-scalar type"
                ))),
            })
            .collect()
    }

    pub(super) fn expand_star_lanes(
        &self,
        qualifier: Option<&str>,
        opts: &sqlparser::ast::WildcardAdditionalOptions,
    ) -> Result<Vec<(String, StarLane)>, PrepareError> {
        use sqlparser::ast::{ExcludeSelectItem, RenameSelectItem};
        if opts.opt_except.is_some() {
            return Err(unsup("SELECT * EXCEPT"));
        }
        // EXCLUDE entries: (optional table qualifier, column name).
        fn exclude_name(
            n: &sqlparser::ast::ObjectName,
        ) -> Result<(Option<&str>, &str), PrepareError> {
            fn ident(p: &sqlparser::ast::ObjectNamePart) -> Option<&str> {
                p.as_ident().map(|i| i.value.as_str())
            }
            match n.0.as_slice() {
                [part] => ident(part)
                    .map(|c| (None, c))
                    .ok_or_else(|| unsup("EXCLUDE list entry form")),
                [t, part] => match (ident(t), ident(part)) {
                    (Some(t), Some(c)) => Ok((Some(t), c)),
                    _ => Err(unsup("EXCLUDE list entry form")),
                },
                _ => Err(unsup("EXCLUDE list entry form")),
            }
        }
        let mut exclude: Vec<(Option<String>, String)> = match &opts.opt_exclude {
            None => Vec::new(),
            Some(ExcludeSelectItem::Single(id)) => vec![exclude_name(id)?],
            Some(ExcludeSelectItem::Multiple(ids)) => ids
                .iter()
                .map(exclude_name)
                .collect::<Result<Vec<_>, _>>()?,
        }
        .into_iter()
        .map(|(t, c)| (t.map(str::to_string), c.to_string()))
        .collect();
        // Name filter, decoded from the ILIKE slot (rewrite.rs) — it may
        // carry EXCLUDE entries the rewrite absorbed.
        let filter = opts
            .opt_ilike
            .as_ref()
            .map(|il| decode_star_filter(&il.pattern));
        if let Some(f) = &filter {
            exclude.extend(f.excludes.iter().cloned());
        }
        // Measured: two entries clash when the names match and either is
        // unqualified or both carry the same qualifier -- (i, i), (i, a.i)
        // and (A.i, a.I) are duplicates, (a.i, b.i) is not.
        for (i, (qa, a)) in exclude.iter().enumerate() {
            if exclude[..i].iter().any(|(qb, b)| {
                b.eq_ignore_ascii_case(a)
                    && match (qa, qb) {
                        (Some(x), Some(y)) => x.eq_ignore_ascii_case(y),
                        _ => true,
                    }
            }) {
                // DuckDB rejects this at parse; ours surfaces at bind.
                return Err(PrepareError::Bind(format!(
                    "duplicate entry \"{a}\" in EXCLUDE list"
                )));
            }
        }
        // A qualified EXCLUDE of a merged USING column UNMERGES it on DuckDB,
        // whichever side the qualifier names (measured: EXCLUDE (i1.i) and
        // EXCLUDE (__THIS__.k) both put the key back at the right side's
        // position, with the right side's values) -- not modeled, refused.
        let using_key = |sj: &ScopeJoin, e: &str| {
            sj.merged.iter().any(|m| m.eq_ignore_ascii_case(e))
                || (sj.using
                    && sj.key_cols.iter().any(|k| match k.src {
                        KeySrc::Lane(ci) => sj.table.cols[ci as usize].name.eq_ignore_ascii_case(e),
                        _ => false,
                    }))
        };
        if exclude
            .iter()
            .any(|(q, e)| q.is_some() && self.joins.iter().any(|sj| using_key(sj, e)))
        {
            return Err(unsup("EXCLUDE of a USING-merged column (DuckDB unmerges it)"));
        }
        let excluded_lists_conflict = |list: &str, name: &str| -> Result<(), PrepareError> {
            if exclude.iter().any(|(_, e)| e.eq_ignore_ascii_case(name)) {
                // DuckDB: Parser Error — same clean class via Parse.
                return Err(PrepareError::Parse(format!(
                    "Column \"{name}\" cannot occur in both EXCLUDE and {list} list"
                )));
            }
            Ok(())
        };
        if filter.is_some() && opts.opt_replace.is_some() {
            return Err(PrepareError::Bind(
                "Replace list cannot be combined with a filtering operation".into(),
            ));
        }
        if filter.is_some() && opts.opt_rename.is_some() {
            return Err(PrepareError::Bind(
                "Rename list cannot be combined with a filtering operation".into(),
            ));
        }

        // (origin table, output name, lane) — the origin drives qualified
        // EXCLUDE; it is dropped on return.
        let mut cols: Vec<(String, String, StarLane)> = Vec::new();
        let mut matched = false;
        if qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&self.this_name)) {
            matched = true;
            // Interleave scalar lanes with opaque and struct columns back
            // into MODEL order (their positions are model positions;
            // scalars fill the rest in order). A struct column expands as
            // its whole value, as on DuckDB.
            let mut scalars = self.in_cols[..self.n_plain].iter().enumerate();
            for pos in 0..self.n_plain + self.opaque.len() + self.structs.len() {
                if let Some((_, oname)) = self.opaque.iter().find(|(p, _)| *p == pos) {
                    cols.push((
                        self.this_name.clone(),
                        oname.clone(),
                        StarLane::Opaque(oname.clone()),
                    ));
                } else if let Some(sc) = self.structs.iter().find(|s| s.pos == pos) {
                    cols.push((
                        self.this_name.clone(),
                        sc.name.clone(),
                        StarLane::Struct(NodeRef {
                            src: NodeSrc::Row,
                            path: vec![sc.name.clone()],
                            refusal: format!("column '{}' has a non-scalar type", sc.name),
                        }),
                    ));
                } else {
                    let (i, c) = scalars.next().expect("scalar count matches positions");
                    cols.push((
                        self.this_name.clone(),
                        c.name.clone(),
                        StarLane::Real(SExpr {
                            kind: SKind::Col(i as u32),
                            ty: c.ty.ty,
                            nullable: c.ty.nullable,
                        }),
                    ));
                }
            }
        }
        // Joined tables expand in FROM order, columns in DECLARED order
        // (measured): value columns as probe lanes, key columns via the
        // dynamic-side reconstruction, USING keys suppressed under `*`
        // (merged into the left occurrence).
        for (j, sj) in self.joins.iter().enumerate() {
            if !qualifier.is_none_or(|q| q.eq_ignore_ascii_case(&sj.name)) {
                continue;
            }
            matched = true;
            // Declared order: a struct column expands as ONE entry, its whole
            // value, and a non-vocabulary column as one opaque entry —
            // EXCLUDE removes it, surviving is the named refusal. Iterating
            // `cols` here would expand a struct's flattened leaves as
            // phantom columns and silently DROP an opaque column, a column
            // set DuckDB never produces. A USING/NATURAL struct key merges
            // into the left occurrence like a scalar key; under its own
            // table's star it stays opaque (its whole value is not served).
            for sc in sj.table.star.iter() {
                let ci = match sc {
                    super::super::plan::StarCol::Opaque(oname)
                        if qualifier.is_none() && key_struct(sj, oname) =>
                    {
                        if exclude.iter().any(|(t, e)| {
                            t.as_deref()
                                .is_some_and(|t| t.eq_ignore_ascii_case(&sj.name))
                                && e.eq_ignore_ascii_case(oname)
                        }) {
                            return Err(unsup(
                                "EXCLUDE of a USING-merged column (DuckDB unmerges it)",
                            ));
                        }
                        continue;
                    }
                    super::super::plan::StarCol::Opaque(oname) => {
                        let is_struct = sj
                            .table
                            .structs
                            .iter()
                            .any(|s| s.name == *oname && !key_struct(sj, oname));
                        cols.push((
                            sj.name.clone(),
                            oname.clone(),
                            if is_struct {
                                StarLane::Struct(NodeRef {
                                    src: NodeSrc::Static(j),
                                    path: vec![oname.clone()],
                                    refusal: format!("column '{oname}' has a non-scalar type"),
                                })
                            } else {
                                StarLane::Opaque(oname.clone())
                            },
                        ));
                        continue;
                    }
                    super::super::plan::StarCol::Real(ci) => *ci,
                };
                let c = &sj.table.cols[ci as usize];
                // KEY first: a lossy key column ALSO appears in
                // `val_cols` as a shadow lane, and taking that branch here
                // would unmerge a USING key. `key_lane` reads the shadow.
                let kp = sj.key_cols.iter().position(|k| k.src == KeySrc::Lane(ci));
                if qualifier.is_none() && sj.merged.iter().any(|m| m.eq_ignore_ascii_case(&c.name)) {
                    continue; // a self-join's merged USING column
                }
                if kp.is_none() {
                    let pos = sj
                        .val_cols
                        .iter()
                        .position(|&v| v == ci)
                        .expect("column is key or value");
                    cols.push((
                        sj.name.clone(),
                        c.name.clone(),
                        StarLane::Real(self.static_lane(j, pos)),
                    ));
                } else {
                    let kp = kp.expect("checked above");
                    // A USING key merges into the left occurrence under `*`;
                    // its own table's star keeps it, with that table's
                    // values (measured: `d.*` lists d's key, NULL on a miss).
                    if !sj.using || qualifier.is_some() {
                        cols.push((
                            sj.name.clone(),
                            c.name.clone(),
                            StarLane::Real(self.key_lane(j, kp)?),
                        ));
                    } else if exclude.iter().any(|(t, e)| {
                        t.as_deref()
                            .is_some_and(|t| t.eq_ignore_ascii_case(&sj.name))
                            && e.eq_ignore_ascii_case(&c.name)
                    }) {
                        // Measured: EXCLUDE (right.key) on a USING join
                        // UNMERGES the column (it reappears at the right
                        // table's position with right values) — not modeled.
                        return Err(unsup(
                            "EXCLUDE of a USING-merged column (DuckDB unmerges it)",
                        ));
                    }
                }
            }
        }
        // Struct-star `a.*` — checked AFTER tables: a table alias with the
        // same name WINS over the struct column (measured, silently). The
        // struct may be the driving table's or a joined static table's; a
        // head in both is DuckDB's ambiguity error, as for `a.f`.
        if !matched {
            if let Some(q) = qualifier {
                use super::super::plan::StructNode;
                let row = self.structs.iter().find(|s| s.name.eq_ignore_ascii_case(q));
                let statics: Vec<(usize, &super::super::plan::StructCol)> = self
                    .joins
                    .iter()
                    .enumerate()
                    .filter(|(_, sj)| !key_struct(sj, q))
                    .filter_map(|(j, sj)| {
                        sj.table
                            .structs
                            .iter()
                            .find(|s| s.name.eq_ignore_ascii_case(q))
                            .map(|sc| (j, sc))
                    })
                    .collect();
                if usize::from(row.is_some()) + statics.len() > 1 {
                    return Err(PrepareError::Bind(format!(
                        "ambiguous column '{q}' (qualify it)"
                    )));
                }
                if row.is_some() || !statics.is_empty() {
                    matched = true;
                    if filter.is_some() || opts.opt_rename.is_some() {
                        return Err(unsup(
                            "struct star with a name filter or RENAME (unpinned)",
                        ));
                    }
                }
                if let Some(sc) = row {
                    for f in &sc.fields {
                        let lane = match &f.node {
                            StructNode::Leaf(l) => {
                                let c = &self.in_cols[*l as usize];
                                StarLane::Real(SExpr {
                                    kind: SKind::Col(*l),
                                    ty: c.ty.ty,
                                    nullable: c.ty.nullable,
                                })
                            }
                            // A nested struct field expands as its whole
                            // value.
                            StructNode::Nested(_) => StarLane::Struct(NodeRef {
                                src: NodeSrc::Row,
                                path: vec![sc.name.clone(), f.name.clone()],
                                refusal: format!("column '{}' has a non-scalar type", f.name),
                            }),
                            // An unmappable field expands as a non-scalar
                            // entry: EXCLUDE removes it, surviving is the
                            // named error.
                            StructNode::Opaque => StarLane::Opaque(f.name.clone()),
                        };
                        cols.push((sc.name.clone(), f.name.clone(), lane));
                    }
                }
                for (j, sc) in statics {
                    let table = self.joins[j].name.clone();
                    for f in &sc.fields {
                        let lane = match &f.node {
                            StructNode::Leaf(_) => StarLane::Real(self.static_struct_lane(
                                j,
                                sc,
                                &table,
                                &sc.name,
                                &[Ident::new(&f.name)],
                            )?),
                            StructNode::Nested(_) => StarLane::Struct(NodeRef {
                                src: NodeSrc::Static(j),
                                path: vec![sc.name.clone(), f.name.clone()],
                                refusal: format!("column '{}' has a non-scalar type", f.name),
                            }),
                            StructNode::Opaque => StarLane::Opaque(f.name.clone()),
                        };
                        cols.push((sc.name.clone(), f.name.clone(), lane));
                    }
                }
            }
        }
        if !matched {
            return Err(PrepareError::Bind(format!(
                "table '{}' in wildcard does not exist in FROM",
                qualifier.unwrap_or("?")
            )));
        }

        for (t, ex) in &exclude {
            let hit = cols.iter().any(|(ct, cn, _)| {
                cn.eq_ignore_ascii_case(ex)
                    && t.as_deref().is_none_or(|t| t.eq_ignore_ascii_case(ct))
            });
            if !hit {
                let disp = match t {
                    Some(t) => format!("{t}.{ex}"),
                    None => ex.to_string(),
                };
                // Name the scope that was actually searched, as DuckDB does
                // (measured): a QUALIFIED star searched one relation, an
                // unqualified one searched every relation in FROM.
                let scope = match qualifier {
                    Some(q) => format!("'{q}'"),
                    None => "FROM clause".to_string(),
                };
                return Err(PrepareError::Bind(format!(
                    "column \"{disp}\" in EXCLUDE list not found in {scope}"
                )));
            }
        }
        // Unqualified EXCLUDE strips ALL same-named copies; qualified
        // strips one table's (measured).
        cols.retain(|(ct, cn, _)| {
            !exclude.iter().any(|(t, ex)| {
                cn.eq_ignore_ascii_case(ex)
                    && t.as_deref().is_none_or(|t| t.eq_ignore_ascii_case(ct))
            })
        });

        // REPLACE (expr AS col): position and name kept, type may change,
        // expr sees the full original scope (incl. EXCLUDEd columns).
        if let Some(rep) = &opts.opt_replace {
            for (i, it) in rep.items.iter().enumerate() {
                let name = &it.column_name.value;
                if rep.items[..i]
                    .iter()
                    .any(|p| p.column_name.value.eq_ignore_ascii_case(name))
                {
                    return Err(PrepareError::Parse(format!(
                        "Duplicate entry \"{name}\" in REPLACE list"
                    )));
                }
                excluded_lists_conflict("REPLACE", name)?;
                let hits: Vec<usize> = cols
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, cn, _))| cn.eq_ignore_ascii_case(name))
                    .map(|(i, _)| i)
                    .collect();
                match hits[..] {
                    [] => {
                        return Err(PrepareError::Bind(format!(
                            "column \"{name}\" in REPLACE list not found in FROM clause"
                        )))
                    }
                    [pos] => {
                        cols[pos].2 = match self.struct_value(&it.expr)? {
                            Some(v) => StarLane::Value(v),
                            None => StarLane::Real(fold(self.expr(&it.expr)?)),
                        };
                        // The output name takes the REPLACE alias's exact
                        // case (measured on struct-star; the match itself
                        // stays case-insensitive).
                        cols[pos].1 = it.column_name.value.clone();
                    }
                    _ => {
                        return Err(PrepareError::Bind(format!(
                            "ambiguous reference to column name \"{name}\" in REPLACE list"
                        )))
                    }
                }
            }
        }

        // RENAME (a AS b): position kept; a NONEXISTENT target is silently
        // ignored and an ambiguous name renames ALL copies (measured);
        // collisions become duplicates for dedup_output_names.
        if let Some(ren) = &opts.opt_rename {
            let items: Vec<&sqlparser::ast::IdentWithAlias> = match ren {
                RenameSelectItem::Single(i) => vec![i],
                RenameSelectItem::Multiple(v) => v.iter().collect(),
            };
            for it in items {
                let name = &it.ident.value;
                excluded_lists_conflict("RENAME", name)?;
                if let Some(rep) = &opts.opt_replace {
                    if rep
                        .items
                        .iter()
                        .any(|p| p.column_name.value.eq_ignore_ascii_case(name))
                    {
                        return Err(PrepareError::Parse(format!(
                            "Column \"{name}\" cannot occur in both REPLACE and RENAME list"
                        )));
                    }
                }
                for c in cols.iter_mut() {
                    if c.1.eq_ignore_ascii_case(name) {
                        c.1 = it.alias.value.clone();
                    }
                }
            }
        }

        // Name filter LAST in our pipeline but semantically after EXCLUDE
        // only (REPLACE/RENAME + filter were rejected above).
        if let Some(DecodedFilter { op, pat, .. }) = &filter {
            let similar = match op {
                // Positive SIMILAR TO searches names UNANCHORED; the NOT
                // form negates a FULL match — measured to not be
                // complements ('a.*': "Weird Name" is in BOTH results).
                StarFilter::Similar { neg } => Some(self.name_regex(pat, *neg)?),
                _ => None,
            };
            cols.retain(|(_, cn, _)| match (op, &similar) {
                (StarFilter::Like { ci, neg }, _) => like_match(cn, pat, *ci) != *neg,
                (StarFilter::Glob, _) => super::super::exec::kernels::duck_glob(cn, pat),
                (StarFilter::Similar { neg }, Some(rx)) => rx.is_match(cn) != *neg,
                (StarFilter::Similar { .. }, None) => unreachable!(),
            });
            if cols.is_empty() {
                return Err(PrepareError::Bind(format!(
                    "star expression with name filter '{pat}' resulted in an empty set of columns"
                )));
            }
        }
        Ok(cols.into_iter().map(|(_, n, l)| (n, l)).collect())
    }

    /// Compile a column-NAME regex for star filters / COLUMNS: unanchored
    /// for the positive search, `\A..\z`-wrapped for the NOT-full-match
    /// form. Bind-time only — never reaches the exec regex table.
    pub(super) fn name_regex(&self, pat: &str, full: bool) -> Result<regex::Regex, PrepareError> {
        let translated = super::super::retrans::translate_pattern(pat)?;
        let pattern = if full {
            format!("\\A(?:{translated})\\z")
        } else {
            translated
        };
        regex::RegexBuilder::new(&pattern)
            .octal(true)
            .build()
            .map_err(|e| {
                PrepareError::Bind(format!("Failed to compile regex \"{pat}\": {e}"))
            })
    }

    /// Expand a `COLUMNS('re')` / `COLUMNS(*)` SELECT item (pins-waveB/):
    /// unanchored RE2 search over declared-case names, table-declaration
    /// order. Returns None when `e` is not a COLUMNS call; expression
    /// forms (COLUMNS(..) + 1) stay unsupported upstream.
    pub(super) fn expand_columns_item(
        &self,
        e: &SqlExpr,
    ) -> Result<Option<Vec<(String, OutVal)>>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let SqlExpr::Function(f) = e else {
            return Ok(None);
        };
        if !f.name.to_string().eq_ignore_ascii_case("columns") {
            return Ok(None);
        }
        let FunctionArguments::List(list) = &f.args else {
            return Ok(None);
        };
        let all =
            self.expand_star_lanes(None, &sqlparser::ast::WildcardAdditionalOptions::default())?;
        match &list.args[..] {
            [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)] => Ok(Some(self.finalize_star(all)?)),
            // COLUMNS(* EXCLUDE/REPLACE/... ) — measured identical to the
            // bare `* <modifiers>` select item (names, order, values;
            // pins-waveA/columns-replace.json), so route through the same
            // star expansion.
            [FunctionArg::Unnamed(FunctionArgExpr::WildcardWithOptions(opts))] => {
                Ok(Some(self.expand_star(None, opts)?))
            }
            [FunctionArg::Unnamed(FunctionArgExpr::Expr(p))] => {
                let Some(bp) = self.expr_or_null(p)? else {
                    return Err(PrepareError::Bind(
                        "COLUMNS requires a constant pattern".into(),
                    ));
                };
                let SKind::Lit(Lit::Str(pat)) = bp.kind else {
                    return Err(unsup("COLUMNS with a non-constant or list argument"));
                };
                let rx = self.name_regex(&pat, false)?;
                // Filter BEFORE the opaque check: a regex that never
                // matches an opaque column must not reject the query.
                let cols: Vec<(String, StarLane)> =
                    all.into_iter().filter(|(n, _)| rx.is_match(n)).collect();
                if cols.is_empty() {
                    return Err(PrepareError::Bind(format!(
                        "No matching columns found that match regex \"{pat}\""
                    )));
                }
                Ok(Some(self.finalize_star(cols)?))
            }
            _ => Err(unsup("COLUMNS argument form")),
        }
    }
}
