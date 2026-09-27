//! FROM clause: relation names, the driving table, joins as written.

use super::*;

/// `a CROSS JOIN b` IS `a, b` on DuckDB (measured: same rows, same column
/// order). The driving relation's TRAILING cross joins become comma
/// relations, in order, so the comma path binds them -- its WHERE-key
/// extraction and multiplicity rules included -- and a star expands in the
/// same order. A cross join followed by another join stays where it is,
/// and refuses by name.
pub(super) fn cross_joins_as_commas(
    from: &[sqlparser::ast::TableWithJoins],
) -> Vec<sqlparser::ast::TableWithJoins> {
    use sqlparser::ast::TableWithJoins;
    let Some((first, rest)) = from.split_first() else {
        return Vec::new();
    };
    let is_cross = |j: &sqlparser::ast::Join| {
        matches!(j.join_operator, JoinOperator::CrossJoin(JoinConstraint::None))
    };
    let tail = first.joins.iter().rev().take_while(|j| is_cross(j)).count();
    let keep = first.joins.len() - tail;
    let mut out = vec![TableWithJoins {
        relation: first.relation.clone(),
        joins: first.joins[..keep].to_vec(),
    }];
    out.extend(first.joins[keep..].iter().map(|j| TableWithJoins {
        relation: j.relation.clone(),
        joins: Vec::new(),
    }));
    out.extend(rest.iter().cloned());
    out
}

/// A relation's name as its identifier VALUES, quotes resolved: `"Dim Table"`
/// is the one part `Dim Table`, `main.d` the parts `main`, `d`. Nothing
/// splits the SQL spelling, so a quoted name may contain a space or a dot.
#[derive(Debug, Clone)]
pub(super) struct RelName(pub(super) Vec<String>);

impl RelName {
    /// The table part: the name the relation is in scope under.
    pub(super) fn bare(&self) -> &str {
        self.0.last().map_or("", String::as_str)
    }

    /// The schema the relation lives in as named in FROM: the part before
    /// the table (`main.d`, `memory.main.d`), `main` when unqualified. A
    /// column may be qualified through exactly this schema (measured:
    /// `main.t.c` binds over `FROM t`, `other.t.c` is DuckDB's
    /// `Referenced table "other.t" not found`).
    pub(super) fn schema(&self) -> String {
        match self.0.len() {
            0 | 1 => "main".to_string(),
            n => self.0[n - 2].clone(),
        }
    }
}

impl std::fmt::Display for RelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0.join("."))
    }
}

pub(super) fn plain_table(
    tf: &TableFactor,
) -> Result<Option<(RelName, Option<&TableAlias>)>, PrepareError> {
    let TableFactor::Table {
        name,
        alias,
        args,
        with_hints,
        version,
        with_ordinality,
        partitions,
        json_path,
        sample,
        index_hints,
    } = tf
    else {
        return Ok(None);
    };
    let modifier = if args.is_some() {
        Some("table-valued function arguments")
    } else if !with_hints.is_empty() {
        Some("WITH (...) table hints")
    } else if version.is_some() {
        Some("a table version qualifier")
    } else if *with_ordinality {
        Some("WITH ORDINALITY")
    } else if !partitions.is_empty() {
        Some("PARTITION (...) selection")
    } else if json_path.is_some() {
        Some("a JSON path on a relation")
    } else if sample.is_some() {
        Some("TABLESAMPLE")
    } else if !index_hints.is_empty() {
        Some("index hints")
    } else {
        None
    };
    if let Some(m) = modifier {
        return Err(unsup(format!("{m} on relation '{name}'")));
    }
    let Some(parts) = name
        .0
        .iter()
        .map(|p| p.as_ident().map(|i| i.value.clone()))
        .collect::<Option<Vec<_>>>()
    else {
        return Err(unsup(format!("relation name '{name}' built by a function")));
    };
    Ok(Some((RelName(parts), alias.as_ref())))
}

/// Parse and bind the FROM clause: the dynamic table, then zero or more
/// equi-joins to static tables. Returns the fully-scoped binder (every join
/// visible) and the join specs in FROM order.
#[allow(clippy::too_many_arguments)]
/// Where a relation sits in FROM: its position decides only the wording of
/// two refusals (an unparseable relation, an unresolvable table).
#[derive(Clone, Copy)]
pub(super) enum RelPosition {
    Join,
    Comma,
}

pub(super) enum RelSource {
    /// The request table again: a self-join against the batch.
    SelfJoin,
    /// A provided static table, by index.
    Static(usize),
}

/// A JOIN or comma relation after name resolution: everything both paths
/// need before they bind keys. One resolver, so a naming rule (schema
/// qualifiers, quoting, aliases, duplicate names) is written once.
pub(super) struct ResolvedRel<'q> {
    pub(super) alias: Option<&'q TableAlias>,
    /// The name the relation is in scope under: its alias, else its bare
    /// table name (`main.d` and `memory.main.d` are in scope as `d`, as on
    /// DuckDB, measured).
    pub(super) scope_name: String,
    /// The schema it was named through; empty when aliased, since `main.x.a`
    /// over `... AS x` is a binder error on DuckDB (measured).
    pub(super) schema: String,
    pub(super) source: RelSource,
}

impl ResolvedRel<'_> {
    /// The scope entry of a self-join: the build side is the BATCH, every
    /// plain row column a value lane, no probe keys.
    pub(super) fn batch_scope<'b>(
        &self,
        in_cols: &[Col],
        n_plain: usize,
        kind: JoinKind,
        merged: Vec<String>,
    ) -> ScopeJoin<'b> {
        ScopeJoin {
            name: self.scope_name.clone(),
            schema: self.schema.clone(),
            table: std::borrow::Cow::Owned(StaticTable::all_scalar(
                self.scope_name.clone(),
                in_cols[..n_plain].to_vec(),
            )),
            kind,
            key_cols: Vec::new(),
            val_cols: (0..n_plain as u32).collect(),
            keys: Vec::new(),
            using: false,
            merged,
        }
    }
}

/// The spec of a self-join: a keyless batchmap built per call, the whole ON
/// (if any) as its residual.
pub(super) fn batch_spec(n_plain: usize, kind: JoinKind, residual: Option<SExpr>) -> JoinSpec {
    JoinSpec {
        table: 0,
        batch: true,
        kind,
        keys: Vec::new(),
        key_cols: Vec::new(),
        val_cols: (0..n_plain as u32).collect(),
        residual,
    }
}

/// Resolve one JOIN or comma relation against the scope built so far: its
/// name and alias, the self-join guards, the static table it names, and the
/// duplicate-name check, in that order (the order fixes which error a query
/// with several faults reports).
pub(super) fn resolve_relation<'q>(
    binder: &Binder<'_>,
    factor: &'q TableFactor,
    position: RelPosition,
    statics: &[StaticTable],
    this_name: &str,
    many: bool,
    driving: &Driving,
) -> Result<ResolvedRel<'q>, PrepareError> {
    let (raw_name, alias) = match plain_table(factor)? {
        Some(named) => named,
        None => {
            return Err(unsup(match position {
                RelPosition::Join => format!("JOIN {factor}"),
                RelPosition::Comma => relation_refusal(factor),
            }))
        }
    };
    let scope_name = alias
        .map(|a| a.name.value.clone())
        .unwrap_or_else(|| raw_name.bare().to_string());
    let schema = if alias.is_some() { String::new() } else { raw_name.schema() };
    let source = if raw_name.to_string().eq_ignore_ascii_case(this_name) {
        if let Driving::Derived { .. } = driving {
            return Err(unsup("joining the request table beside a derived table"));
        }
        if alias.is_some_and(|a| !a.columns.is_empty()) {
            // Dropping it answered a query with the WRONG names in
            // scope; serving the rename on a self-join is unpinned.
            return Err(unsup("column-list alias on a self-join"));
        }
        if !many {
            return Err(unsup("joining the dynamic table to itself"));
        }
        if !binder.opaque.is_empty() || !binder.structs.is_empty() {
            return Err(unsup("self-join over a row model with non-scalar columns"));
        }
        RelSource::SelfJoin
    } else {
        RelSource::Static(match position {
            RelPosition::Join => resolve_static(statics, &raw_name)?,
            // Unresolvable comma tables (schema-qualified names, table
            // functions we didn't get as statics) stay CLEAN.
            RelPosition::Comma => resolve_static(statics, &raw_name).map_err(|_| {
                unsup(format!(
                    "comma-joined table '{raw_name}' is not a provided static table"
                ))
            })?,
        })
    };
    if binder.this_name.eq_ignore_ascii_case(&scope_name)
        || binder
            .joins
            .iter()
            .any(|j| j.name.eq_ignore_ascii_case(&scope_name))
    {
        return Err(PrepareError::Bind(format!(
            "duplicate table name '{scope_name}' in FROM"
        )));
    }
    Ok(ResolvedRel {
        alias,
        scope_name,
        schema,
        source,
    })
}

pub(super) fn bind_from<'a>(
    select: &sqlparser::ast::Select,
    this_name: &str,
    in_cols: &'a [Col],
    opaque: &'a [(usize, String)],
    structs: &'a [super::super::plan::StructCol],
    statics: &'a [StaticTable],
    many: bool,
    udfs: &'a [super::super::ir::ExternSpec],
    models: &'a [super::super::plan::ModelTable],
    // Installed at construction so JOIN keys and residuals, which bind in
    // here, fold pure externs exactly like the projection.
    bind_eval: &'a [ExternImpl],
    // What this level reads: the request table, or a derived table whose
    // output columns are `in_cols`.
    driving: Driving,
    ctx: QueryCtx,
) -> Result<(Binder<'a>, Vec<JoinSpec>, Option<SqlExpr>), PrepareError> {
    // Plain scalar columns occupy in_cols[..n_plain]; struct leaf lanes
    // follow and are addressable ONLY through their struct paths.
    let n_plain = in_cols.len() - structs.iter().map(|s| s.leaf_count()).sum::<usize>();
    let from = cross_joins_as_commas(&select.from);
    let Some((table, comma_rels)) = from.split_first() else {
        return Err(unsup("FROM-less SELECT -- a query must read the request table"));
    };
    let dyn_name = match (&driving, plain_table(&table.relation)?) {
        // A derived table is always named (its alias, else
        // `unnamed_subquery`); its column-alias list was applied by the
        // caller, so `in_cols` already carries the names in scope.
        (Driving::Derived { name }, _) => (name.clone(), None),
        (Driving::Request, Some((n, alias))) => {
            // The engine's registry is SCHEMA-LESS: a single schema
            // qualifier is accepted when the table part matches the
            // registered bare name (DuckDB's schema-existence errors are
            // unknowable to a schema-less registry; documented in
            // known-limitations.md §5).
            let bare = n.bare();
            if !bare.eq_ignore_ascii_case(this_name) {
                return Err(unsup(format!(
                    "table '{n}' as the driving relation (must be the dynamic table '{this_name}')"
                )));
            }
            let n = bare.to_string();
            match alias {
                // Measured: an alias REPLACES the original name entirely
                // (qualified refs through the original are binder errors in
                // DuckDB) — making the alias the binder's this_name gives
                // exactly that scoping.
                Some(a) if a.columns.is_empty() => (a.name.value.clone(), None),
                // `t AS u(x, y)`: a PARTIAL list is legal (prefix rename,
                // remaining columns keep their names); too many names is
                // the pinned bind error; old names are fully shadowed
                // (pins-wave5/).
                Some(a) => {
                    let model_cols = n_plain + opaque.len() + structs.len();
                    if a.columns.len() > model_cols {
                        return Err(PrepareError::Bind(format!(
                            "table \"{n}\" has {model_cols} columns available but {} columns specified",
                            a.columns.len()
                        )));
                    }
                    // The rename is positional over the FULL model; a name
                    // landing on an opaque/struct column has no plain lane
                    // to rename.
                    if let Some((_, oname)) = opaque.iter().find(|(p, _)| *p < a.columns.len()) {
                        return Err(unsup(format!(
                            "row column '{oname}' has a non-scalar type"
                        )));
                    }
                    if let Some(sc) = structs.iter().find(|s| s.pos < a.columns.len()) {
                        return Err(unsup(format!(
                            "column-list alias over struct column '{}'",
                            sc.name
                        )));
                    }
                    let mut renamed = in_cols.to_vec();
                    for (c, def) in renamed.iter_mut().zip(&a.columns) {
                        c.name = def.name.value.clone();
                    }
                    (a.name.value.clone(), Some(renamed))
                }
                None => (n.to_string(), None),
            }
        }
        (Driving::Request, None) => return Err(unsup(relation_refusal(&table.relation))),
    };
    let (dyn_name, renamed_cols) = dyn_name;

    // An aliased relation has no schema: `main.x.a` over `FROM t AS x` is a
    // binder error on DuckDB (measured), so nothing can match "".
    let this_schema = match plain_table(&table.relation)? {
        // Nor does a derived table: it is never schema-qualified.
        _ if matches!(driving, Driving::Derived { .. }) => String::new(),
        Some((_, Some(_))) => String::new(),
        Some((n, None)) => n.schema(),
        None => "main".to_string(),
    };
    let mut binder = Binder {
        this_name: dyn_name,
        this_schema,
        in_cols: match renamed_cols {
            Some(v) => std::borrow::Cow::Owned(v),
            None => std::borrow::Cow::Borrowed(in_cols),
        },
        n_plain,
        opaque,
        structs,
        joins: Vec::new(),
        select_aliases: select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(alias.value.clone()),
                _ => None,
            })
            .collect(),
        bound_aliases: std::cell::RefCell::new(Vec::new()),
        regexes: std::cell::RefCell::new(ctx.regexes),
        udfs,
        bind_eval,
        models,
        model_refs: std::cell::RefCell::new(ctx.model_refs),
        sites: std::cell::Cell::new(ctx.sites),
        // Per level: the same call text at two levels has different
        // arguments, so it is a different call site.
        extern_sites: std::cell::RefCell::new(Vec::new()),
        in_guarded: std::cell::Cell::new(0),
        minted_lanes: std::cell::RefCell::new(ctx.minted_lanes),
    };
    let mut specs: Vec<JoinSpec> = Vec::new();

    for join in &table.joins {
        let (kind, constraint) = match &join.join_operator {
            JoinOperator::Join(c) | JoinOperator::Inner(c) => (JoinKind::Inner, c),
            JoinOperator::Left(c) | JoinOperator::LeftOuter(c) => (JoinKind::Left, c),
            other => return Err(unsup(join_refusal(other))),
        };
        let rel = resolve_relation(
            &binder,
            &join.relation,
            RelPosition::Join,
            statics,
            this_name,
            many,
            &driving,
        )?;
        if let RelSource::SelfJoin = rel.source {
            // Self-join: the build side is the BATCH — a keyless
            // batchmap (built per call) with the WHOLE ON as residual.
            // USING/NATURAL is the equality residual `left.c = right.c` per
            // merged name, plus the merge itself (see `ScopeJoin::merged`).
            // NATURAL over the same table merges every column.
            let row_names: Vec<String> = in_cols[..binder.n_plain]
                .iter()
                .map(|c| c.name.clone())
                .collect();
            let (on, merged) = match constraint {
                JoinConstraint::On(e) => (Some(e.clone()), Vec::new()),
                JoinConstraint::Using(cols) => {
                    let mut names: Vec<String> = Vec::new();
                    for obj in cols {
                        let [part] = obj.0.as_slice() else {
                            return Err(unsup("qualified name in JOIN USING"));
                        };
                        let name = part
                            .as_ident()
                            .map(|i| i.value.clone())
                            .ok_or_else(|| unsup("JOIN USING entry form"))?;
                        let Some(c) = row_names.iter().find(|c| c.eq_ignore_ascii_case(&name))
                        else {
                            return Err(PrepareError::Bind(format!(
                                "column \"{name}\" does not exist on right side of join!"
                            )));
                        };
                        if !names.iter().any(|n| n.eq_ignore_ascii_case(c)) {
                            names.push(c.clone()); // USING (a, a) dedupes
                        }
                    }
                    (Some(using_equalities(&rel.scope_name, &names)?), names)
                }
                JoinConstraint::Natural => (
                    Some(using_equalities(&rel.scope_name, &row_names)?),
                    row_names.clone(),
                ),
                JoinConstraint::None => {
                    return Err(unsup("JOIN without ON (cross join)"))
                }
            };
            binder.joins.push(rel.batch_scope(in_cols, binder.n_plain, kind, merged));
            let residual = match &on {
                None => None,
                Some(e) => Some(fold(bool_context(binder.expr(e)?, "JOIN condition")?)),
            };
            specs.push(batch_spec(binder.n_plain, kind, residual));
            continue;
        }
        let RelSource::Static(table_idx) = rel.source else {
            unreachable!("handled above")
        };
        let renamed = apply_column_alias(&statics[table_idx], rel.alias)?;
        let st = renamed.as_ref().unwrap_or(&statics[table_idx]);
        let (keys, key_cols, residual_raw, using) = match constraint {
            JoinConstraint::On(e) => {
                let (keys, key_cols, res) = bind_on(&binder, st, &rel.scope_name, &rel.schema, e)?;
                (keys, key_cols, res, false)
            }
            // USING desugar (pins-wave4/): each column pairs the LEFT
            // scope's binding with this table's column; duplicates in the
            // list dedupe silently; ambiguity in the left scope (e.g.
            // after a prior ON join) errors exactly like DuckDB.
            JoinConstraint::Using(cols) => {
                let mut keys = Vec::new();
                let mut key_cols = Vec::new();
                let mut heads: Vec<String> = Vec::new();
                let offered = static_head_names(st);
                for obj in cols {
                    let [part] = obj.0.as_slice() else {
                        return Err(unsup("qualified name in JOIN USING"));
                    };
                    let name = part
                        .as_ident()
                        .map(|i| i.value.clone())
                        .ok_or_else(|| unsup("JOIN USING entry form"))?;
                    // The scan is over every HEAD the table offers, structs
                    // and opaque columns included: claiming a struct column
                    // "does not exist on right side" is a lie about a
                    // column that plainly does.
                    if !offered.iter().any(|h| h.eq_ignore_ascii_case(&name)) {
                        return Err(PrepareError::Bind(format!(
                            "column \"{name}\" does not exist on right side of join!"
                        )));
                    }
                    if heads.iter().any(|h| h.eq_ignore_ascii_case(&name)) {
                        continue; // USING (a, a) dedupes silently (measured)
                    }
                    heads.push(name.clone());
                    let ks = shared_key(&binder, st, &name, many)?;
                    if ks.is_empty() {
                        // Present on the right, absent on the LEFT: the
                        // ordinary column bind is the error to report.
                        return Err(binder.column(&name).err().unwrap_or_else(|| {
                            PrepareError::Bind(format!(
                                "column \"{name}\" does not exist on left side of join!"
                            ))
                        }));
                    }
                    for (key, col) in ks {
                        keys.push(key);
                        key_cols.push(col);
                    }
                }
                (keys, key_cols, Vec::new(), true)
            }
            // NATURAL = USING(all common column names), case-insensitive,
            // merged output like USING with the LEFT spelling; NO common
            // columns is a hard error, never a cross product (pins-wave5/).
            JoinConstraint::Natural => {
                let mut keys = Vec::new();
                let mut key_cols = Vec::new();
                let mut heads: Vec<String> = Vec::new();
                // DuckDB intersects NAME SETS with no type inspection, so
                // the scan is over every head, not just the scalar lanes.
                // Skipping a struct or opaque head would drop the column OUT
                // of the key set and emit rows DuckDB never produces, so
                // such a head refuses BY NAME instead.
                for name in static_head_names(st) {
                    let ks = shared_key(&binder, st, &name, many)?;
                    if ks.is_empty() {
                        continue; // not a shared name at all
                    }
                    heads.push(name);
                    for (key, col) in ks {
                        keys.push(key);
                        key_cols.push(col);
                    }
                }
                if heads.is_empty() {
                    return Err(PrepareError::Bind(
                        "No columns found to join on in NATURAL JOIN.\n\
                         Use CROSS JOIN if you intended for this to be a cross-product."
                            .into(),
                    ));
                }
                (keys, key_cols, Vec::new(), true)
            }
            JoinConstraint::None => return Err(unsup("JOIN without ON (cross join)")),
        };
        let val_cols = val_cols_for(st, &key_cols, &keys);

        binder.joins.push(ScopeJoin {
            name: rel.scope_name,
            schema: rel.schema,
            table: match renamed {
                Some(t) => std::borrow::Cow::Owned(t),
                None => std::borrow::Cow::Borrowed(&statics[table_idx]),
            },
            kind,
            key_cols: key_cols.clone(),
            val_cols: val_cols.clone(),
            keys: keys.clone(),
            using,
            merged: Vec::new(),
        });
        // Residual conjuncts bind with THIS join in scope.
        let j = (binder.joins.len() - 1) as u32;
        let residual = bind_residual(&binder, j, &residual_raw)?;
        specs.push(JoinSpec {
            table: table_idx,
            batch: false,
            kind,
            keys,
            key_cols,
            val_cols,
            residual,
        });
    }

    // Comma relations (measured: FROM t, u WHERE t.k = u.k is bit-identical
    // to the INNER probe, star order included; residual WHERE placement is
    // free under INNER). Equi conjuncts pairing the current scope with a
    // comma table's column are consumed as its probe keys; everything else
    // stays WHERE. A keyless comma table is a cross join — correct for a
    // 1-row static via the empty-key map (the duplicate-key check enforces
    // single-entry-ness at compile; a 0-row static annihilates, also
    // measured).
    let mut conjuncts: Vec<&SqlExpr> = Vec::new();
    if let Some(sel) = &select.selection {
        collect_conjuncts(sel, &mut conjuncts);
    }
    let mut consumed = vec![false; conjuncts.len()];
    for rel in comma_rels {
        if !rel.joins.is_empty() {
            return Err(unsup("JOIN attached to a comma-joined relation"));
        }
        let rel = resolve_relation(
            &binder,
            &rel.relation,
            RelPosition::Comma,
            statics,
            this_name,
            many,
            &driving,
        )?;
        let RelSource::Static(table_idx) = rel.source else {
            // Comma self-join = pure cross against the batch; equi
            // conjuncts stay in WHERE (cross-then-filter is bit-identical
            // under multiplicity — measured, pins-stageB).
            binder.joins.push(rel.batch_scope(in_cols, binder.n_plain, JoinKind::Inner, Vec::new()));
            specs.push(batch_spec(binder.n_plain, JoinKind::Inner, None));
            continue;
        };
        let renamed = apply_column_alias(&statics[table_idx], rel.alias)?;
        let st = renamed.as_ref().unwrap_or(&statics[table_idx]);
        let mut keys = Vec::new();
        let mut key_cols = Vec::new();
        for (ci, c) in conjuncts.iter().enumerate() {
            if consumed[ci] {
                continue;
            }
            let SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } = c
            else {
                continue;
            };
            let l = static_col_of(left, st, &rel.scope_name, &rel.schema, &binder)?;
            let r = static_col_of(right, st, &rel.scope_name, &rel.schema, &binder)?;
            let (col, dyn_side, static_side) = match (l, r) {
                (Some(c), None) => (c, right.as_ref(), left.as_ref()),
                (None, Some(c)) => (c, left.as_ref(), right.as_ref()),
                _ => continue, // stays WHERE
            };
            if let SqlExpr::Identifier(id) = static_side {
                if binder.column(&id.value).is_ok() {
                    return Err(PrepareError::Bind(format!(
                        "ambiguous column '{}' in WHERE (qualify it)",
                        id.value
                    )));
                }
            }
            // The dynamic side must bind in the scope BEFORE this table —
            // if it references this or a later comma table, leave the
            // conjunct in WHERE (a later table may consume it).
            let Ok(key) = binder.expr(dyn_side) else {
                continue;
            };
            keys.push(promote_key(fold(key), st, col)?);
            // A comma equi-conjunct is always plain `=`.
            key_cols.push(JoinKey {
                src: KeySrc::Lane(col),
                cmp: KeyCmp::Eq,
            });
            consumed[ci] = true;
        }
        let val_cols = val_cols_for(st, &key_cols, &keys);
        binder.joins.push(ScopeJoin {
            name: rel.scope_name,
            schema: rel.schema,
            table: match renamed {
                Some(t) => std::borrow::Cow::Owned(t),
                None => std::borrow::Cow::Borrowed(&statics[table_idx]),
            },
            kind: JoinKind::Inner,
            key_cols: key_cols.clone(),
            val_cols: val_cols.clone(),
            keys: keys.clone(),
            using: false,
            merged: Vec::new(),
        });
        specs.push(JoinSpec {
            table: table_idx,
            batch: false,
            kind: JoinKind::Inner,
            keys,
            key_cols,
            val_cols,
            residual: None,
        });
    }

    // Rebuild the WHERE from unconsumed conjuncts (identity when nothing
    // was consumed — the single-relation path always takes this shape).
    let leftover = if comma_rels.is_empty() {
        select.selection.clone()
    } else {
        let mut acc: Option<SqlExpr> = None;
        for (ci, c) in conjuncts.iter().enumerate() {
            if consumed[ci] {
                continue;
            }
            acc = Some(match acc {
                None => (*c).clone(),
                Some(p) => ast_bin(BinaryOperator::And, p, (*c).clone()),
            });
        }
        acc
    };
    Ok((binder, specs, leftover))
}

/// `left.c = right.c AND ...` over `names`, as SQL: the residual a
/// USING/NATURAL self-join binds. The LEFT reference is the name as the
/// scope to the left of this join resolves it (so a USING chain keys on
/// the merged column, as DuckDB does); the right is qualified by the
/// self-join's own scope name.
pub(super) fn using_equalities(right: &str, names: &[String]) -> Result<SqlExpr, PrepareError> {
    let ident = |s: &str| sqlparser::ast::Ident::new(s);
    let mut conj: Option<SqlExpr> = None;
    for n in names {
        let eq = SqlExpr::BinaryOp {
            left: Box::new(SqlExpr::Identifier(ident(n))),
            op: BinaryOperator::Eq,
            right: Box::new(SqlExpr::CompoundIdentifier(vec![ident(right), ident(n)])),
        };
        conj = Some(match conj {
            None => eq,
            Some(c) => SqlExpr::BinaryOp {
                left: Box::new(c),
                op: BinaryOperator::And,
                right: Box::new(eq),
            },
        });
    }
    conj.ok_or_else(|| unsup("JOIN USING with no columns"))
}
