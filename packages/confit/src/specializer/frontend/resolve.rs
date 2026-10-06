//! Column resolution: bare, qualified and compound names against the
//! scope, static lanes and join keys.

use super::*;

impl Binder<'_> {
    /// The lane of value column `pos` of join `j`: NULL-able exactly when
    /// the join is LEFT (a miss makes it NULL); INNER misses never reach an
    /// expression (the row was skipped).
    pub(super) fn static_lane(&self, j: usize, pos: usize) -> SExpr {
        let sj = &self.joins[j];
        let col = &sj.table.cols[sj.val_cols[pos] as usize];
        SExpr {
            kind: SKind::StaticCol {
                join: j as u32,
                col: pos as u32,
            },
            ty: col.ty.ty,
            // NULL-able on a LEFT miss OR when the static column itself is
            // declared nullable (NULL values ride as validity+payload
            // pairs through the probe).
            nullable: sj.kind == JoinKind::Left || col.ty.nullable,
        }
    }

    /// The probe-side KEY expression for "the struct node at `path` is
    /// non-NULL": NULL when the node is absent, TRUE when it is present.
    /// Mints the boundary lane on first use and reuses it after.
    ///
    /// Encoding the node's validity as a key VALUE this way is what lets
    /// the existing key machinery carry DuckDB's nested semantics with no
    /// new runtime: as a PLAIN key it annihilates (top level, where `=`
    /// propagates NULL), as an IS-NOT-DISTINCT key it matches its own kind
    /// (every level below, where a node's NULL is a value).
    pub(super) fn present_key(&self, path: &[String]) -> SExpr {
        let lane = self.present_lane(path);
        SExpr {
            kind: SKind::Case {
                arms: vec![(
                    lane,
                    SExpr {
                        kind: SKind::Lit(Lit::I1(true)),
                        ty: Ty::I1,
                        nullable: false,
                    },
                )],
                default: None,
            },
            ty: Ty::I1,
            nullable: true,
        }
    }

    /// The input lane "the row struct node at `path` is non-NULL", as a
    /// non-nullable boolean. Mints it on first use and reuses it after.
    pub(super) fn present_lane(&self, path: &[String]) -> SExpr {
        let mut lanes = self.minted_lanes.borrow_mut();
        let idx = match lanes.iter().position(|l| l.path == path) {
            Some(i) => i,
            None => {
                lanes.push(super::super::plan::InputLane {
                    name: format!("{} (present)", path.join(".")),
                    path: path.to_vec(),
                    kind: super::super::plan::LaneKind::Present,
                });
                lanes.len() - 1
            }
        };
        SExpr {
            kind: SKind::Col((self.in_cols.len() + idx) as u32),
            ty: Ty::I1,
            nullable: false,
        }
    }

    /// KEY column `key_pos` of join `j`, reconstructed from the dynamic
    /// side (measured: on a match the static key equals the probe key;
    /// on a LEFT miss it is NULL): INNER rows all matched, so the key
    /// expression itself is exact; LEFT wraps it in CASE match THEN key
    /// ELSE NULL.
    ///
    /// The reconstruction is sound about the VALUE and easy to get wrong
    /// about the TYPE. Adopting the ROW column's declaration projected int8
    /// where DuckDB projects int64 for `int8 row key = int64 static key`,
    /// and int64 where DuckDB projects int8 for the reverse pairing. The
    /// STATIC column's own declaration is the answer; between two integer
    /// widths that is a pure re-declaration, because DuckDB compares across
    /// widths NUMERICALLY, so a match already proves the value fits both.
    pub(super) fn key_lane(&self, j: usize, key_pos: usize) -> Result<SExpr, PrepareError> {
        let sj = &self.joins[j];
        let key = sj.keys[key_pos].clone();
        let KeySrc::Lane(ci) = sj.key_cols[key_pos].src else {
            return Err(PrepareError::Internal(
                "a struct presence key has no projectable column".into(),
            ));
        };
        let col = &sj.table.cols[ci as usize];
        if self.classify_keys.get() {
            return Ok(SExpr {
                kind: SKind::StaticCol {
                    join: j as u32,
                    col: ci,
                },
                ty: col.ty.ty,
                nullable: true,
            });
        }
        // `promote_key`'s F64-probe-against-integer-column arm compares
        // in double space, so no reconstruction can name the i64
        // back (two build rows can collide on one double). That column rides
        // as a shadow VALUE lane — read the real value, at the static
        // column's own declared width, and let the ordinary LEFT-miss
        // nullability of a value lane carry the miss.
        if key_is_lossy(key.ty, col.ty.ty) {
            let pos = sj
                .val_cols
                .iter()
                .position(|&v| v == ci)
                .expect("a lossy key column has a shadow value lane");
            return Ok(self.static_lane(j, pos));
        }
        let key = match (key.ty, col.ty.ty) {
            (a, b) if a == b => key,
            (a, b) if a.is_integer() && b.is_integer() && a.lane() == b.lane() => {
                SExpr { ty: b, ..key }
            }
            // A key compared on the i128 lane against a narrower column:
            // where the join matched, the value is the column's, so the
            // conversion back cannot fail.
            (a, b) if a.is_integer() && b.is_integer() => {
                let nullable = key.nullable;
                SExpr {
                    kind: SKind::Cast {
                        inner: Box::new(key),
                        trying: false,
                    },
                    ty: b,
                    nullable,
                }
            }
            (a, b) => {
                return Err(unsup(format!(
                    "projecting join key column '{}.{}': it is declared {} \
                     but the join compares it as {}, and the original value \
                     is not recoverable from the comparison",
                    sj.name,
                    col.name,
                    b.name(),
                    a.name()
                )))
            }
        };
        if sj.kind == JoinKind::Inner {
            return Ok(key);
        }
        let ty = key.ty;
        let hit = SExpr {
            kind: SKind::JoinHit(j as u32),
            ty: Ty::I1,
            nullable: false,
        };
        Ok(SExpr {
            kind: SKind::Case {
                arms: vec![(hit, key)],
                default: None,
            },
            ty,
            nullable: true,
        })
    }

    /// n-part dotted reference (pins-waveA/struct-nested.json): qualifier
    /// prefixes longest-first — (schema.table).column, then
    /// (table|alias).column, then bare column — committing to the longest
    /// prefix whose COLUMN part binds, remaining parts becoming struct
    /// field extractions. A failed column bind BACKTRACKS to the next
    /// shorter interpretation; a failed FIELD walk after a bound column is
    /// a hard error (measured). The schema part follows the registry-noise
    /// rule (known-limitations §5).
    pub(super) fn compound(&self, parts: &[sqlparser::ast::Ident]) -> Result<SExpr, PrepareError> {
        self.compound_res(parts).and_then(Resolved::scalar)
    }

    /// [`Self::compound`], with a whole struct answered as its node.
    pub(super) fn compound_res(
        &self,
        parts: &[sqlparser::ast::Ident],
    ) -> Result<Resolved, PrepareError> {
        // R0: memory.schema.(this|join).column[.fields...] -- the default
        // catalog, then the same rung as R1.
        if parts.len() >= 4 && parts[0].value.eq_ignore_ascii_case("memory") {
            let (schema, rel) = (&parts[1].value, &parts[2].value);
            if rel.eq_ignore_ascii_case(&self.this_name)
                && schema.eq_ignore_ascii_case(&self.this_schema)
            {
                if let Some(r) = self.this_col_res(&parts[3].value, &parts[4..]) {
                    return r;
                }
            } else if self.joins.iter().any(|sj| {
                sj.name.eq_ignore_ascii_case(rel) && sj.schema.eq_ignore_ascii_case(schema)
            }) {
                if let Some(r) = self.qualified_path(rel, &parts[3..]) {
                    return r;
                }
            }
        }
        // R1: schema.(this|join).column[.fields...], the schema being the one
        // the relation was named through.
        if parts.len() >= 3 {
            if parts[1].value.eq_ignore_ascii_case(&self.this_name)
                && parts[0].value.eq_ignore_ascii_case(&self.this_schema)
            {
                if let Some(r) = self.this_col_res(&parts[2].value, &parts[3..]) {
                    return r;
                }
            } else if
            // parts[0] must be a SCHEMA here, not a relation. `__THIS__.w.mean`
            // reads as (this).w.mean, never as (schema __THIS__).(table w).mean
            // — otherwise a static table sharing a struct column's name would
            // make the qualified spelling unable to reach the struct, and that
            // spelling is the one DuckDB gives you to force it. Only this
            // branch is guarded: `t.t.t.t FROM t.t` is a real
            // schema-qualified corpus case and it lands in the branch above.
            !parts[0].value.eq_ignore_ascii_case(&self.this_name)
                && self.joins.iter().any(|sj| {
                    sj.name.eq_ignore_ascii_case(&parts[1].value)
                        && sj.schema.eq_ignore_ascii_case(&parts[0].value)
                })
            {
                if let Some(r) = self.qualified_path(&parts[1].value, &parts[2..]) {
                    return r;
                }
            }
        }
        // R2: this.column[.fields...] / join.column
        if parts.len() >= 2 {
            if parts[0].value.eq_ignore_ascii_case(&self.this_name) {
                if let Some(r) = self.this_col_res(&parts[1].value, &parts[2..]) {
                    return r;
                }
            } else if let Some(r) = self.qualified_path(&parts[0].value, &parts[1..]) {
                return r;
            }
        }
        // R3: bare column[.fields...]
        if parts.len() >= 2 {
            if let Some(r) = self.bare_col_res(&parts[0].value, &parts[1..]) {
                return r;
            }
            // Nothing bound anywhere. DuckDB surfaces the error of the
            // LONGEST rung whose RELATION existed (measured: `s.nope.x` and
            // `main.s.nope.x.y` both say `Table "s" does not have a column
            // named "nope"`); with no relation at all, the pre-struct
            // shapes stand.
            let relation = |i: usize| {
                parts[i].value.eq_ignore_ascii_case(&self.this_name)
                    || self
                        .joins
                        .iter()
                        .any(|sj| sj.name.eq_ignore_ascii_case(&parts[i].value))
            };
            if parts.len() >= 3 && relation(1) {
                let schema_ok = if parts[1].value.eq_ignore_ascii_case(&self.this_name) {
                    parts[0].value.eq_ignore_ascii_case(&self.this_schema)
                } else {
                    self.joins.iter().any(|sj| {
                        sj.name.eq_ignore_ascii_case(&parts[1].value)
                            && sj.schema.eq_ignore_ascii_case(&parts[0].value)
                    })
                };
                if !schema_ok {
                    return Err(PrepareError::Bind(format!(
                        "Referenced table \"{}.{}\" not found",
                        parts[0].value, parts[1].value
                    )));
                }
                return self.qualified_res(&parts[1].value, &parts[2].value);
            }
            if relation(0) {
                return self.qualified_res(&parts[0].value, &parts[1].value);
            }
            return match parts.len() {
                2 => self.qualified_res(&parts[0].value, &parts[1].value),
                3 => self.qualified_res(&parts[1].value, &parts[2].value),
                _ => Err(PrepareError::Bind(format!(
                    "Referenced table \"{}.{}\" not found",
                    parts[0].value, parts[1].value
                ))),
            };
        }
        self.column_res(&parts[0].value)
    }

    /// A static table's column addressed by `parts`: the head resolves
    /// among the table's struct TREES first,
    /// and the remaining parts walk the tree to a leaf lane — the path is
    /// ORDERED and either walks exactly or misses, so `w.x.y.z.a` and
    /// `w.z.y.x.a` are different lanes and `w.a` finds nothing. No dotted
    /// string is ever built: a quoted `"w.mean"` is ONE part and lands in
    /// the single-part branch, where it can only mean a literal column.
    ///
    /// `None` means BACKTRACK — the relation is in scope but the
    /// head is not one of its columns, DuckDB's only fall-through between
    /// ladder rungs (bind_context.cpp:360-363). Everything the relation
    /// does bind is answered here, ambiguous and unservable included.
    pub(super) fn qualified_path(
        &self,
        table: &str,
        parts: &[sqlparser::ast::Ident],
    ) -> Option<Result<Resolved, PrepareError>> {
        let head = &parts[0].value;
        let (j, sj) = self
            .joins
            .iter()
            .enumerate()
            .find(|(_, sj)| sj.name.eq_ignore_ascii_case(table))?;
        if !binds_in_join(sj, head) {
            return None;
        }
        if parts.len() == 1 {
            return Some(self.qualified_res(table, head));
        }
        if let Some(sc) = sj
            .table
            .structs
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(head))
        {
            return Some(self.static_struct_res(j, sc, table, head, &parts[1..]));
        }
        Some(match self.qualified_res(table, head) {
            Ok(_) => Err(PrepareError::Bind(format!(
                "Cannot extract field '{}' from expression \"{head}\" \
                 because it is not a struct, union, map, or json",
                parts[1].value
            ))),
            Err(e) => Err(e),
        })
    }

    /// Walk one static struct TREE of join `j` down `fields` to its leaf
    /// lane. Shared by the qualified head and the unqualified one
    /// — `table` and `head` are the spellings the user typed, because the
    /// refusals quote them back.
    pub(super) fn static_struct_lane(
        &self,
        j: usize,
        sc: &super::super::plan::StructCol,
        table: &str,
        head: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Result<SExpr, PrepareError> {
        self.static_struct_res(j, sc, table, head, fields)
            .and_then(Resolved::scalar)
    }

    /// The lane of static struct leaf `ci` of join `j`.
    pub(super) fn static_leaf(&self, j: usize, ci: u32, table: &str) -> Result<SExpr, PrepareError> {
        let sj = &self.joins[j];
        if let Some(pos) = sj.val_cols.iter().position(|&v| v == ci) {
            return Ok(self.static_lane(j, pos));
        }
        // The struct is a NATURAL/USING key, so its leaves are
        // key lanes, not value lanes — and a key column
        // reconstructs from the dynamic side exactly like a scalar
        // key does (on a match the two are equal; a LEFT miss is
        // NULL).
        let kp = sj
            .key_cols
            .iter()
            .position(|k| k.src == KeySrc::Lane(ci))
            .ok_or_else(|| {
                PrepareError::Internal(format!(
                    "struct leaf lane {ci} of '{table}' is neither a value \
                     nor a key lane"
                ))
            })?;
        self.key_lane(j, kp)
    }

    /// [`Self::static_struct_lane`], with a nested struct field answered
    /// as its node.
    pub(super) fn static_struct_res(
        &self,
        j: usize,
        sc: &super::super::plan::StructCol,
        table: &str,
        head: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Result<Resolved, PrepareError> {
        match walk_fields(&sc.fields, fields) {
            Ok(ci) => self.static_leaf(j, ci, table).map(Resolved::Lane),
            Err(WalkStop::Missing { at }) => Err(PrepareError::Bind(format!(
                "Could not find key \"{}\" in struct",
                fields[at].value
            ))),
            Err(WalkStop::NotStruct { at, field }) => Err(PrepareError::Bind(format!(
                "Cannot extract field '{}' from expression \"{field}\" \
                 because it is not a struct, union, map, or json",
                fields[at + 1].value
            ))),
            Err(WalkStop::OpaqueField { field }) => Err(unsup(format!(
                "struct field '{field}' has a non-scalar type"
            ))),
            // An INTERMEDIATE node: the path is real and only its VALUE is
            // a struct. Same refusal as the whole column, not a "no such
            // key" lie. The dotted spelling here is DISPLAY, rebuilt from
            // the parts.
            Err(WalkStop::Nested { at, .. }) => {
                let display: Vec<&str> = std::iter::once(head)
                    .chain(fields[..=at].iter().map(|f| f.value.as_str()))
                    .collect();
                Ok(Resolved::Node(NodeRef {
                    src: NodeSrc::Static(j),
                    path: declared_path(sc, &fields[..=at]),
                    refusal: format!(
                        "static table '{table}' column '{}' is a struct where a \
                         scalar is needed (read one of its fields)",
                        display.join(".")
                    ),
                }))
            }
        }
    }

    /// The driving table's column `name` followed by struct-field `fields`.
    /// `None` = no such column (the caller backtracks); `Some(Err)` = the
    /// column bound but the reference is an error (hard, per pins).
    pub(super) fn this_col_with_fields(
        &self,
        name: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Option<Result<SExpr, PrepareError>> {
        self.this_col_res(name, fields)
            .map(|r| r.and_then(Resolved::scalar))
    }

    /// [`Self::this_col_with_fields`], with a whole struct answered as its
    /// node.
    pub(super) fn this_col_res(
        &self,
        name: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Option<Result<Resolved, PrepareError>> {
        for (i, c) in self.in_cols[..self.n_plain].iter().enumerate() {
            if c.name.eq_ignore_ascii_case(name) {
                if let Some(f) = fields.first() {
                    return Some(Err(PrepareError::Bind(format!(
                        "Cannot extract field '{}' from expression \"{}\" \
                         because it is not a struct, union, map, or json",
                        f.value, c.name
                    ))));
                }
                return Some(self.null_col(Resolved::Lane(SExpr {
                    kind: SKind::Col(i as u32),
                    ty: c.ty.ty,
                    nullable: c.ty.nullable,
                })));
            }
        }
        if let Some((_, n)) = self
            .opaque
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
        {
            return Some(Err(unsup(format!(
                "row column '{n}' has a non-scalar type"
            ))));
        }
        let sc = self
            .structs
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))?;
        Some(self.walk_struct(sc, fields))
    }

    /// A bare first part, resolved over the WHOLE scope: the
    /// driving table's columns (incl. structs and opaque) and every joined
    /// table's — a static struct head is a binding here exactly as it is
    /// under a qualifier, which is what lets an unqualified `w.mean` reach
    /// a lane instead of hunting for a table called `w`.
    pub(super) fn bare_col_res(
        &self,
        name: &str,
        fields: &[sqlparser::ast::Ident],
    ) -> Option<Result<Resolved, PrepareError>> {
        // DuckDB decides AMBIGUITY before it looks at the fields —
        // a head that binds in the driving table AND in a join scope refuses
        // even when only one side is a struct the path could walk. Resolving
        // the struct first would answer a query DuckDB rejects.
        // GetMatchingBinding THROWS and no rung catches it, so the verdict
        // is on the head name alone, whatever the heads hold.
        let row = self.this_col_res(name, fields);
        let hits = usize::from(row.is_some())
            + self
                .joins
                .iter()
                .map(|sj| head_hits_in_join(sj, name))
                .sum::<usize>();
        if hits == 0 {
            return None;
        }
        // The static table whose ON key is binding has the head too.
        if self.beside.borrow().iter().any(|b| b.eq_ignore_ascii_case(name)) {
            return Some(Err(PrepareError::Bind(format!(
                "ambiguous column '{name}' in JOIN ON (qualify it)"
            ))));
        }
        if hits > 1 {
            return Some(Err(PrepareError::Bind(format!(
                "ambiguous column '{name}' (qualify it)"
            ))));
        }
        if row.is_some() {
            return row;
        }
        let (j, sj) = self
            .joins
            .iter()
            .enumerate()
            .find(|(_, sj)| head_hits_in_join(sj, name) > 0)
            .expect("the single hit is in a join");
        if let Some(sc) = sj
            .table
            .structs
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
        {
            return Some(self.static_struct_res(j, sc, &sj.name, name, fields));
        }
        if let Some(e) = opaque_static_refusal(&sj.table, name, &sj.name) {
            return Some(Err(e));
        }
        // A scalar lane (value or key): a field asked of it is DuckDB's
        // not-a-struct error.
        Some(Err(PrepareError::Bind(format!(
            "Cannot extract field '{}' from expression \"{name}\" \
             because it is not a struct, union, map, or json",
            fields[0].value
        ))))
    }

    /// Walk `fields` down a row struct column to a scalar leaf lane. Empty
    /// fields, or fields that end on a nested struct, are the whole struct:
    /// its node.
    pub(super) fn walk_struct(
        &self,
        sc: &super::super::plan::StructCol,
        fields: &[sqlparser::ast::Ident],
    ) -> Result<Resolved, PrepareError> {
        if fields.is_empty() {
            return Ok(Resolved::Node(NodeRef {
                src: NodeSrc::Row,
                path: vec![sc.name.clone()],
                refusal: format!(
                    "struct column '{}' where a scalar is needed (read one of its fields)",
                    sc.name
                ),
            }));
        }
        match walk_fields(&sc.fields, fields) {
            Ok(lane) => {
                let c = &self.in_cols[lane as usize];
                Ok(Resolved::Lane(SExpr {
                    kind: SKind::Col(lane),
                    ty: c.ty.ty,
                    nullable: c.ty.nullable,
                }))
            }
            Err(WalkStop::Missing { at }) => Err(PrepareError::Bind(format!(
                "Could not find key \"{}\" in struct",
                fields[at].value
            ))),
            Err(WalkStop::NotStruct { at, field }) => Err(PrepareError::Bind(format!(
                "Cannot extract field '{}' from expression \"{field}\" \
                 because it is not a struct, union, map, or json",
                fields[at + 1].value
            ))),
            Err(WalkStop::OpaqueField { field }) => Err(unsup(format!(
                "struct field '{field}' has a non-scalar type"
            ))),
            Err(WalkStop::Nested { field, at }) => Ok(Resolved::Node(NodeRef {
                src: NodeSrc::Row,
                path: declared_path(sc, &fields[..=at]),
                refusal: format!(
                    "struct field '{field}' where a scalar is needed (read one \
                     of its fields)"
                ),
            })),
        }
    }

    /// Case-insensitive, spelling-preserving bare-column bind over the whole
    /// scope: the dynamic table plus every joined static table's value
    /// columns (DuckDB semantics; ambiguity is an error).
    pub(super) fn column(&self, name: &str) -> Result<SExpr, PrepareError> {
        self.column_res(name).and_then(Resolved::scalar)
    }

    /// [`Self::column`], with a whole struct answered as its node.
    pub(super) fn column_res(&self, name: &str) -> Result<Resolved, PrepareError> {
        if self.beside.borrow().iter().any(|b| b.eq_ignore_ascii_case(name)) {
            let saved = self.beside.take();
            let here = self.column_res(name);
            *self.beside.borrow_mut() = saved;
            // Any binding at all on this side, a refused non-scalar one
            // included, makes the name ambiguous.
            let exists = match &here {
                Ok(_) => true,
                Err(PrepareError::Unsupported(_)) => true,
                Err(PrepareError::Bind(m)) => m.starts_with("ambiguous"),
                Err(_) => false,
            };
            if exists {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{name}' in JOIN ON (qualify it)"
                )));
            }
            return here;
        }
        let mut hits: Vec<Resolved> = Vec::new();
        for (i, c) in self.in_cols[..self.n_plain].iter().enumerate() {
            if c.name.eq_ignore_ascii_case(name) {
                hits.push(Resolved::Lane(SExpr {
                    kind: SKind::Col(i as u32),
                    ty: c.ty.ty,
                    nullable: c.ty.nullable,
                }));
            }
        }
        if let Some((_, n)) = self
            .opaque
            .iter()
            .find(|(_, n)| n.eq_ignore_ascii_case(name))
        {
            // The column exists in the model — it just has no lane. Same
            // precedence as a real-column hit (beats lateral aliases).
            return Err(unsup(format!("row column '{n}' has a non-scalar type")));
        }
        if let Some(sc) = self
            .structs
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
        {
            // A bare struct reference is its WHOLE value: a hit like any
            // other column, so a joined table with the same name still
            // makes it ambiguous.
            hits.push(self.walk_struct(sc, &[])?);
        }
        for (j, sj) in self.joins.iter().enumerate() {
            for pos in 0..sj.val_cols.len() {
                // Struct leaves bind by PATH only; a lossy key's shadow
                // lane binds not at all — the key arm below is that
                // column's one binding.
                if sj.table.is_leaf_lane(sj.val_cols[pos]) || is_shadow_lane(sj, pos) {
                    continue;
                }
                if sj.table.cols[sj.val_cols[pos] as usize]
                    .name
                    .eq_ignore_ascii_case(name)
                    && !sj.merged.iter().any(|m| m.eq_ignore_ascii_case(name))
                {
                    hits.push(Resolved::Lane(self.static_lane(j, pos)));
                }
            }
            // Key columns resolve via reconstruction. A USING join's key
            // is MERGED into the left occurrence (measured) — the static
            // side contributes no separate binding; an ON join's key
            // contributes one, so a bare shared key name is ambiguous,
            // exactly like DuckDB.
            if !sj.using {
                for (kp, k) in sj.key_cols.iter().enumerate() {
                    // A struct LEAF key binds by PATH only and a PRESENCE
                    // key is not a column at all.
                    let KeySrc::Lane(ci) = &k.src else { continue };
                    if !sj.table.is_leaf_lane(*ci)
                        && sj.table.cols[*ci as usize].name.eq_ignore_ascii_case(name)
                    {
                        hits.push(Resolved::Lane(self.key_lane(j, kp)?));
                    }
                }
            }
        }
        // A static STRUCT (or non-vocabulary) column's bare name lives
        // outside `cols` (in `structs` / `opaque`) — it still
        // BINDS on DuckDB, so it still counts for ambiguity, and as a sole
        // hit it is the named non-scalar refusal rather than a "does not
        // exist" lie.
        let join_opaque = self.joins.iter().enumerate().find_map(|(j, sj)| {
            sj.table
                .structs
                .iter()
                // A struct head consumed as a USING/NATURAL key is MERGED
                // into the left occurrence — it is not a second binding,
                // so it cannot make the name ambiguous.
                .find(|s| !key_struct(sj, &s.name) && s.name.eq_ignore_ascii_case(name))
                .map(|s| (j, s.name.clone(), "struct".to_string()))
                .or_else(|| {
                    sj.table
                        .opaque
                        .iter()
                        .find(|(c, _)| c.eq_ignore_ascii_case(name))
                        .map(|(c, aty)| (j, c.clone(), aty.clone()))
                })
        });
        if let Some((j, cname, aty)) = join_opaque {
            if !hits.is_empty() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{name}' (qualify it)"
                )));
            }
            let tname = &self.joins[j].name;
            if aty != "struct" {
                return Err(unsup(format!(
                    "static table '{tname}' column '{cname}' has a non-scalar type: {aty}"
                )));
            }
            // A static struct's whole value: its node.
            return Ok(Resolved::Node(NodeRef {
                src: NodeSrc::Static(j),
                path: vec![cname.clone()],
                refusal: format!(
                    "static table '{tname}' column '{cname}' is a struct where a \
                     scalar is needed (read one of its fields)"
                ),
            }));
        }
        match hits.len() {
            // The REAL column wins over a same-named select alias (measured
            // in both SELECT and WHERE — pins-wave5/).
            1 => self.null_col(hits.pop().expect("len checked")),
            0 if name.eq_ignore_ascii_case("rowid") => Err(unsup("rowid pseudo-column")),
            0 => {
                // Lateral aliases: an already-bound alias resolves to its
                // expression; a known-but-later alias is the pinned
                // forward-reference error. A name defined MORE THAN ONCE
                // falls through to that same error: DuckDB resolves lateral
                // refs to the LAST definition (so a between-definitions ref
                // is its forward-reference error, measured 1.5.5), while
                // our per-occurrence binding would take the first — and a
                // shared extern site bound through a mutating alias would
                // silently freeze it. Refusal keeps
                // binding time-invariant for every accepted query.
                let dup = self
                    .select_aliases
                    .iter()
                    .filter(|a| a.eq_ignore_ascii_case(name))
                    .count()
                    > 1;
                if !dup {
                    if let Some((_, e)) = self
                        .bound_aliases
                        .borrow()
                        .iter()
                        .rev()
                        .find(|(a, _)| a.eq_ignore_ascii_case(name))
                    {
                        return self.null_col(Resolved::Lane(e.clone()));
                    }
                }
                if self
                    .select_aliases
                    .iter()
                    .any(|a| a.eq_ignore_ascii_case(name))
                {
                    return Err(PrepareError::Bind(format!(
                        "column \"{name}\" referenced that exists in the SELECT clause - \
                         but this column cannot be referenced before it is defined"
                    )));
                }
                // No column has the name, but a relation does: DuckDB binds
                // it as that relation's whole row struct.
                if self.is_relation(name) {
                    return Err(unsup(format!(
                        "relation '{name}' as a value (its whole row struct); \
                         read its columns as {name}.col instead"
                    )));
                }
                Err(PrepareError::Bind(format!(
                    "column '{name}' does not exist in scope"
                )))
            }
            _ => Err(PrepareError::Bind(format!("ambiguous column '{name}' (qualify it)"))),
        }
    }

    /// Whether `name` is a relation in scope: the request table by its FROM
    /// spelling, or a joined table by its alias (or name).
    pub(super) fn is_relation(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case(&self.this_name)
            || self.joins.iter().any(|sj| name.eq_ignore_ascii_case(&sj.name))
    }

    /// `r`, unless it reads a bare NULL column of the level below anywhere
    /// but as a whole projection item (see `BoundQuery::null_cols`).
    pub(super) fn null_col(&self, r: Resolved) -> Result<Resolved, PrepareError> {
        if let Resolved::Lane(SExpr {
            kind: SKind::Col(i),
            ..
        }) = &r
        {
            if self.null_cols.get(*i as usize) == Some(&true) && !self.whole_item.get() {
                return Err(null_col_refusal());
            }
        }
        Ok(r)
    }

    /// Whether a bare `name` binds as a column (DuckDB's first choice,
    /// before the relation's row struct): anything but "does not exist".
    pub(super) fn binds_as_column(&self, name: &str) -> bool {
        !matches!(self.column(name), Err(PrepareError::Bind(m)) if m.contains("does not exist"))
            && !matches!(self.column(name), Err(PrepareError::Unsupported(m)) if m.starts_with("relation '"))
    }

    /// `table.col` bind: the dynamic table by its FROM spelling, a joined
    /// static table by its alias (or name). A whole struct answers as its
    /// node.
    pub(super) fn qualified_res(&self, table: &str, name: &str) -> Result<Resolved, PrepareError> {
        if table.eq_ignore_ascii_case(&self.this_name) {
            let mut hit = None;
            for (i, c) in self.in_cols[..self.n_plain].iter().enumerate() {
                if c.name.eq_ignore_ascii_case(name) {
                    if hit.is_some() {
                        return Err(PrepareError::Bind(format!("ambiguous column '{name}' (qualify it)")));
                    }
                    hit = Some((i, c));
                }
            }
            if let Some((_, n)) = self
                .opaque
                .iter()
                .find(|(_, n)| n.eq_ignore_ascii_case(name))
            {
                return Err(unsup(format!("row column '{n}' has a non-scalar type")));
            }
            if let Some(sc) = self
                .structs
                .iter()
                .find(|s| s.name.eq_ignore_ascii_case(name))
            {
                return self.walk_struct(sc, &[]);
            }
            if hit.is_none() && name.eq_ignore_ascii_case("rowid") {
                return Err(unsup("rowid pseudo-column"));
            }
            let (i, c) = hit.ok_or_else(|| {
                PrepareError::Bind(format!("column '{name}' does not exist in '{table}'"))
            })?;
            return self.null_col(Resolved::Lane(SExpr {
                kind: SKind::Col(i as u32),
                ty: c.ty.ty,
                nullable: c.ty.nullable,
            }));
        }
        for (j, sj) in self.joins.iter().enumerate() {
            if !sj.name.eq_ignore_ascii_case(table) {
                continue;
            }
            let mut hit = None;
            for pos in 0..sj.val_cols.len() {
                // A struct LEAF is reachable only through its path: its
                // dotted display name is not an identifier. A lossy key's
                // SHADOW lane is not addressable either — the key arm
                // below serves that name, reading this very lane.
                if sj.table.is_leaf_lane(sj.val_cols[pos]) || is_shadow_lane(sj, pos) {
                    continue;
                }
                if sj.table.cols[sj.val_cols[pos] as usize]
                    .name
                    .eq_ignore_ascii_case(name)
                {
                    if hit.is_some() {
                        return Err(PrepareError::Bind(format!("ambiguous column '{name}' (qualify it)")));
                    }
                    hit = Some(pos);
                }
            }
            if let Some(pos) = hit {
                return Ok(Resolved::Lane(self.static_lane(j, pos)));
            }
            // Qualified key access reconstructs from the dynamic side —
            // measured to stay addressable even after USING (NULL on a
            // LEFT miss, never coalesced).
            for (kp, k) in sj.key_cols.iter().enumerate() {
                let KeySrc::Lane(ci) = &k.src else { continue };
                if !sj.table.is_leaf_lane(*ci)
                    && sj.table.cols[*ci as usize].name.eq_ignore_ascii_case(name)
                {
                    return self.key_lane(j, kp).map(Resolved::Lane);
                }
            }
            if name.eq_ignore_ascii_case("rowid") {
                return Err(unsup("rowid pseudo-column"));
            }
            if let Some(e) = opaque_static_refusal(&sj.table, name, table) {
                // A static struct's whole value: its node. A USING/NATURAL
                // struct key's is not served.
                let sc = sj.table.structs.iter().find(|s| s.name.eq_ignore_ascii_case(name));
                if let (Some(sc), PrepareError::Unsupported(refusal)) = (sc, &e) {
                    if key_struct(sj, &sc.name) {
                        return Err(self.merged_key(j, &sc.name));
                    }
                    return Ok(Resolved::Node(NodeRef {
                        src: NodeSrc::Static(j),
                        path: vec![sc.name.clone()],
                        refusal: refusal.clone(),
                    }));
                }
                return Err(e);
            }
            return Err(PrepareError::Bind(format!(
                "column '{name}' does not exist in '{table}'"
            )));
        }
        Err(PrepareError::Bind(format!("unknown table '{table}'")))
    }
}

impl Binder<'_> {
    /// DuckDB qualifies the column names of a SELECT list item before it
    /// expands the SQL function calls in it, so a column in an argument
    /// that a body never reads must not be ambiguous either (measured: with
    /// `CREATE MACRO f(a, b) AS a`, `SELECT f(1, x)` over two tables with
    /// an `x` refuses, in any level's SELECT list but not in WHERE or ON).
    /// A name that does not resolve passes there, so only an ambiguous one
    /// refuses here. `e` is the item as written.
    pub(super) fn qualify_written(&self, e: &SqlExpr) -> Result<(), PrepareError> {
        self.walk_expr(e, &mut |x| match x {
            SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => match self.ref_res(x) {
                Err(PrepareError::Bind(m)) if m.starts_with("ambiguous") => {
                    Err(PrepareError::Bind(m))
                }
                _ => Ok(Walk::Over),
            },
            _ => Ok(Walk::Into),
        })
        .map(|_| ())
    }

    /// What column reference `r` (a bare or a dotted name) resolves to.
    pub(super) fn ref_res(&self, r: &SqlExpr) -> Result<Resolved, PrepareError> {
        match r {
            SqlExpr::CompoundIdentifier(parts) => self.compound_res(parts),
            SqlExpr::Identifier(id) => self.column_res(&id.value),
            other => Err(PrepareError::Internal(format!("not a column reference: {other}"))),
        }
    }

    /// Walk `e` in order, asking `f` at each node whether to go into it,
    /// over it, or to stop; false when `f` stopped. The expansion of a SQL
    /// function call and the text of a let are walked where they stand,
    /// each once: one walked before answered "go on", and would again (a
    /// recurrence reads the let before it twice, so walking each read would
    /// take exponential time). A field name, an argument's name and a
    /// subquery are not walked: none is a column of this scope.
    pub(super) fn walk_expr(
        &self,
        e: &SqlExpr,
        f: &mut dyn FnMut(&SqlExpr) -> Result<Walk, PrepareError>,
    ) -> Result<bool, PrepareError> {
        self.walk_in(e, f, &mut Walked::default())
    }

    fn walk_in(
        &self,
        e: &SqlExpr,
        f: &mut dyn FnMut(&SqlExpr) -> Result<Walk, PrepareError>,
        walked: &mut Walked,
    ) -> Result<bool, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments, Subscript};
        let all = |xs: &mut dyn Iterator<Item = &SqlExpr>,
                   f: &mut dyn FnMut(&SqlExpr) -> Result<Walk, PrepareError>,
                   walked: &mut Walked|
         -> Result<bool, PrepareError> {
            for x in xs {
                if !self.walk_in(x, f, walked)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        match f(e)? {
            Walk::Stop => return Ok(false),
            Walk::Over => return Ok(true),
            Walk::Into => {}
        }
        if let Some((id, lets)) = lets::marker_let(e) {
            if !walked.lets.insert(id) {
                return Ok(true);
            }
            return self.walk_in(&lets[id].ast, f, walked);
        }
        if let Some((id, calls)) = calls::marker_call(e) {
            if walked.calls.insert(id) && !self.walk_in(&calls[id].1, f, walked)? {
                return Ok(false);
            }
        }
        match e {
            SqlExpr::CompoundFieldAccess { root, access_chain } => {
                let mut parts: Vec<&SqlExpr> = vec![root];
                for a in access_chain {
                    match a {
                        AccessExpr::Subscript(Subscript::Index { index }) => parts.push(index),
                        AccessExpr::Subscript(Subscript::Slice {
                            lower_bound,
                            upper_bound,
                            stride,
                        }) => parts.extend([lower_bound, upper_bound, stride].into_iter().flatten()),
                        AccessExpr::Dot(_) => {}
                    }
                }
                all(&mut parts.into_iter(), f, walked)
            }
            SqlExpr::Function(func) => {
                let FunctionArguments::List(list) = &func.args else {
                    return Ok(true);
                };
                all(
                    &mut list.args.iter().filter_map(|a| match a {
                        FunctionArg::Unnamed(FunctionArgExpr::Expr(x))
                        | FunctionArg::Named {
                            arg: FunctionArgExpr::Expr(x),
                            ..
                        }
                        | FunctionArg::ExprNamed {
                            arg: FunctionArgExpr::Expr(x),
                            ..
                        } => Some(x),
                        _ => None,
                    }),
                    f,
                    walked,
                )
            }
            SqlExpr::Nested(x)
            | SqlExpr::UnaryOp { expr: x, .. }
            | SqlExpr::Cast { expr: x, .. }
            | SqlExpr::IsNull(x)
            | SqlExpr::IsNotNull(x)
            | SqlExpr::IsTrue(x)
            | SqlExpr::IsNotTrue(x)
            | SqlExpr::IsFalse(x)
            | SqlExpr::IsNotFalse(x)
            | SqlExpr::IsUnknown(x)
            | SqlExpr::IsNotUnknown(x)
            | SqlExpr::Extract { expr: x, .. }
            | SqlExpr::Ceil { expr: x, .. }
            | SqlExpr::Floor { expr: x, .. }
            | SqlExpr::Collate { expr: x, .. } => self.walk_in(x, f, walked),
            SqlExpr::BinaryOp { left, right, .. }
            | SqlExpr::IsDistinctFrom(left, right)
            | SqlExpr::IsNotDistinctFrom(left, right)
            | SqlExpr::Position {
                expr: left,
                r#in: right,
            }
            | SqlExpr::Like {
                expr: left,
                pattern: right,
                ..
            }
            | SqlExpr::ILike {
                expr: left,
                pattern: right,
                ..
            }
            | SqlExpr::SimilarTo {
                expr: left,
                pattern: right,
                ..
            } => all(&mut [&**left, &**right].into_iter(), f, walked),
            SqlExpr::Between { expr, low, high, .. } => {
                all(&mut [&**expr, &**low, &**high].into_iter(), f, walked)
            }
            SqlExpr::InList { expr, list, .. } => {
                all(&mut std::iter::once(&**expr).chain(list), f, walked)
            }
            SqlExpr::Case {
                operand,
                conditions,
                else_result,
                ..
            } => all(
                &mut operand
                    .as_deref()
                    .into_iter()
                    .chain(conditions.iter().flat_map(|w| [&w.condition, &w.result]))
                    .chain(else_result.as_deref()),
                f,
                walked,
            ),
            SqlExpr::Substring {
                expr,
                substring_from,
                substring_for,
                ..
            } => all(
                &mut std::iter::once(&**expr)
                    .chain(substring_from.as_deref())
                    .chain(substring_for.as_deref()),
                f,
                walked,
            ),
            SqlExpr::Trim {
                expr,
                trim_what,
                trim_characters,
                ..
            } => all(
                &mut std::iter::once(&**expr)
                    .chain(trim_what.as_deref())
                    .chain(trim_characters.iter().flatten()),
                f,
                walked,
            ),
            SqlExpr::Dictionary(fields) => all(&mut fields.iter().map(|d| &*d.value), f, walked),
            SqlExpr::Array(a) => all(&mut a.elem.iter(), f, walked),
            SqlExpr::Tuple(xs) => all(&mut xs.iter(), f, walked),
            // A name was asked about above; a subquery resolves in its own
            // scope; the rest holds no expression.
            _ => Ok(true),
        }
    }
}

/// What [`Binder::walk_expr`] does at a node.
pub(super) enum Walk {
    Into,
    Over,
    Stop,
}

/// The lets and calls one walk has walked.
#[derive(Default)]
struct Walked {
    lets: std::collections::HashSet<usize>,
    calls: std::collections::HashSet<usize>,
}
