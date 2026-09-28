//! Join keys and residuals: static-side resolution, key promotion,
//! struct keys, conjunct collection.

use super::*;

/// How many bindings of head `name` a joined relation contributes: value
/// lanes that are not struct LEAVES (a leaf's dotted display name is not
/// an identifier), key lanes — none under USING, where
/// the key is merged into the left occurrence — struct heads, and columns
/// this engine cannot serve. Ambiguity counts bindings, not lanes we can
/// answer with.
pub(super) fn head_hits_in_join(sj: &ScopeJoin, name: &str) -> usize {
    if sj.merged.iter().any(|m| m.eq_ignore_ascii_case(name)) {
        return 0; // merged into the left occurrence
    }
    let named = |ci: &&u32| sj.table.cols[**ci as usize].name.eq_ignore_ascii_case(name);
    sj.val_cols
        .iter()
        .enumerate()
        // A lossy key's SHADOW lane is not a binding — the key arm below
        // already counts that column exactly once.
        .filter(|(pos, _)| !is_shadow_lane(sj, *pos))
        .map(|(_, ci)| ci)
        .filter(|ci| !sj.table.is_leaf_lane(**ci) && named(ci))
        .count()
        + if sj.using {
            0
        } else {
            sj.key_cols
                .iter()
                .filter_map(|k| match &k.src {
                    KeySrc::Lane(ci) => Some(ci),
                    KeySrc::Present(_) => None,
                })
                // A struct LEAF key is reachable only by path; its dotted
                // display name is not a column name.
                .filter(|ci| !sj.table.is_leaf_lane(**ci) && named(ci))
                .count()
        }
        + sj.table
            .structs
            .iter()
            // A struct head that BECAME a USING/NATURAL key is merged into
            // the left occurrence exactly like a scalar one, so the static
            // side contributes no separate binding.
            .filter(|s| !key_struct(sj, &s.name) && s.name.eq_ignore_ascii_case(name))
            .count()
        + sj.table
            .opaque
            .iter()
            .filter(|(c, _)| c.eq_ignore_ascii_case(name))
            .count()
}

/// Is this static STRUCT head one of the join's MERGED (USING/NATURAL) key
/// columns? Its presence key carries the head's own name.
pub(super) fn key_struct(sj: &ScopeJoin, head: &str) -> bool {
    sj.using
        && sj.key_cols.iter().any(|k| match &k.src {
            KeySrc::Present(p) => p.len() == 1 && p[0].eq_ignore_ascii_case(head),
            KeySrc::Lane(_) => false,
        })
}

/// Whether `name` binds in this joined relation at all — the qualified
/// question, where a USING key is still addressable and `rowid` still
/// answers for itself. A head that does NOT bind is the one case DuckDB
/// retries as a shorter interpretation.
pub(super) fn binds_in_join(sj: &ScopeJoin, name: &str) -> bool {
    head_hits_in_join(sj, name) > 0
        || name.eq_ignore_ascii_case("rowid")
        || sj.key_cols.iter().any(|k| match &k.src {
            KeySrc::Lane(ci) => {
                !sj.table.is_leaf_lane(*ci)
                    && sj.table.cols[*ci as usize].name.eq_ignore_ascii_case(name)
            }
            KeySrc::Present(p) => p.len() == 1 && p[0].eq_ignore_ascii_case(name),
        })
}

/// A subscript/dot chain read as FIELD names: `['f']` and `.f` each name a
/// field; anything else (an integer key, a slice) is not a field access.
pub(super) fn chain_fields(chain: &[AccessExpr]) -> Option<Vec<Ident>> {
    chain
        .iter()
        .map(|acc| match acc {
            AccessExpr::Dot(SqlExpr::Identifier(i)) => Some(i.clone()),
            AccessExpr::Subscript(Subscript::Index {
                index: SqlExpr::Value(v),
            }) => match &v.value {
                SqlValue::SingleQuotedString(f) => Some(Ident::new(f)),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Where a struct-tree walk stopped: shared by the row and static
/// paths, which render the SAME stop into their own pinned error
/// spellings. `at` indexes the FIELD parts handed to [`walk_fields`];
/// `field` is the actual (declaration-cased) field name involved.
pub(super) enum WalkStop {
    /// parts[at] matched no child of the current node.
    Missing { at: usize },
    /// parts[at] hit a scalar LEAF but more parts follow.
    NotStruct { at: usize, field: String },
    /// parts[at] hit a field whose type this engine does not serve.
    OpaqueField { field: String },
    /// The parts ran out ON a nested node — the reference is a whole
    /// struct value, not a leaf.
    Nested { at: usize, field: String },
}

/// Walk `parts` down a struct field tree to a scalar leaf lane index.
/// Field matching is case-insensitive even when quoted (measured —
/// quoting does not opt into case sensitivity).
pub(super) fn walk_fields(
    fields: &[super::super::plan::StructField],
    parts: &[sqlparser::ast::Ident],
) -> Result<u32, WalkStop> {
    use super::super::plan::StructNode;
    debug_assert!(!parts.is_empty());
    let mut cur = fields;
    for (k, f) in parts.iter().enumerate() {
        let Some(sf) = cur.iter().find(|x| x.name.eq_ignore_ascii_case(&f.value)) else {
            return Err(WalkStop::Missing { at: k });
        };
        match &sf.node {
            StructNode::Leaf(lane) => {
                if k + 1 == parts.len() {
                    return Ok(*lane);
                }
                return Err(WalkStop::NotStruct {
                    at: k,
                    field: sf.name.clone(),
                });
            }
            StructNode::Opaque => {
                return Err(WalkStop::OpaqueField {
                    field: sf.name.clone(),
                });
            }
            StructNode::Nested(n) => {
                if k + 1 == parts.len() {
                    return Err(WalkStop::Nested {
                        at: k,
                        field: sf.name.clone(),
                    });
                }
                cur = n;
            }
        }
    }
    unreachable!("the loop returns on the last part")
}

pub(super) fn resolve_static(statics: &[StaticTable], raw_name: &RelName) -> Result<usize, PrepareError> {
    // Schema-less registry: an exact registered-name match wins;
    // otherwise a qualified SQL name (`s1.t1`) matches a registered bare
    // `t1`. Ambiguity stays an error.
    let whole = raw_name.to_string();
    let bare = (raw_name.0.len() > 1).then(|| raw_name.bare());
    let mut table_idx = None;
    for (i, st) in statics.iter().enumerate() {
        let hit = st.name.eq_ignore_ascii_case(&whole)
            || bare.is_some_and(|b| st.name.eq_ignore_ascii_case(b));
        if hit {
            if table_idx.is_some() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous static table '{raw_name}'"
                )));
            }
            table_idx = Some(i);
        }
    }
    table_idx.ok_or_else(|| {
        PrepareError::Bind(format!(
            "table '{raw_name}' was not provided as a static table"
        ))
    })
}

/// Bind the non-key ON conjuncts of join `j` and AND them into the spec's
/// residual, enforcing the evaluation-order rule (pins-wave4/): single-side
/// residuals must be conservatively trap-free (DuckDB scan-pushes them —
/// different error timing); both-sides residuals may trap (DuckDB
/// evaluates them per candidate pair, exactly our hit-guarded lowering).
pub(super) fn bind_residual(
    binder: &Binder<'_>,
    j: u32,
    raw: &[&SqlExpr],
) -> Result<Option<SExpr>, PrepareError> {
    let mut acc: Option<SExpr> = None;
    for c in raw {
        let bound = bool_context(fold(binder.expr(c)?), "JOIN ON condition")?;
        let (mut right, mut left, mut known) = (false, false, true);
        scan_residual(&bound, j, &mut right, &mut left, &mut known);
        let total = !may_trap(&bound);
        if !(total || (left && right && known)) {
            // Two different refusals wear one condition, and the message
            // has to name the one that actually binds. An unrecognised
            // node is always the cause where there is one: the scan does
            // not descend into it, so the sides it reports are the sides
            // OUTSIDE that node and cannot be used to tell the reader
            // which rule applies. Only where every node was recognised is
            // the trapping single-side rule the honest answer.
            return Err(unsup(if known {
                format!(
                    "JOIN ON condition '{c}' (single-side residual with \
                     trapping ops: DuckDB's scan-pushed evaluation order \
                     differs)"
                )
            } else {
                format!(
                    "JOIN ON condition '{c}' (contains an expression the \
                     residual classifier does not recognise, so which side \
                     it reads is unknown)"
                )
            }));
        }
        acc = Some(match acc {
            None => bound,
            Some(p) => {
                let nullable = p.nullable || bound.nullable;
                SExpr {
                    kind: SKind::And {
                        a: Box::new(p),
                        b: Box::new(bound),
                    },
                    ty: Ty::I1,
                    nullable,
                }
            }
        });
    }
    Ok(acc)
}

/// Bind a JOIN ... ON condition. Equalities (and IS NOT DISTINCT FROM)
/// pairing a dynamic-side expression with a static column become probe
/// keys; every OTHER conjunct (non-equalities, constant equalities,
/// both-sides-static equalities) is returned raw for residual binding once
/// the join is in scope (pins-wave4/: `match = key_hit AND residual`).
#[allow(clippy::type_complexity)]
pub(super) fn bind_on<'e>(
    binder: &Binder<'_>,
    st: &StaticTable,
    scope_name: &str,
    scope_schema: &str,
    on: &'e SqlExpr,
) -> Result<(Vec<SExpr>, Vec<JoinKey>, Vec<&'e SqlExpr>), PrepareError> {
    let mut conjuncts = Vec::new();
    collect_conjuncts(on, &mut conjuncts);
    let mut keys = Vec::new();
    let mut key_cols = Vec::new();
    let mut residual = Vec::new();
    for c in conjuncts {
        let (left, right, cmp) = match c {
            SqlExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
            } => (left, right, KeyCmp::Eq),
            // IS NOT DISTINCT FROM: NULL is an ordinary key value — NULL
            // joins NULL (the params-join contract).
            SqlExpr::IsNotDistinctFrom(left, right) => (left, right, KeyCmp::NotDistinct),
            _ => {
                residual.push(c);
                continue;
            }
        };
        let l = static_col_of(left, st, scope_name, scope_schema, binder)?;
        let r = static_col_of(right, st, scope_name, scope_schema, binder)?;
        let (col, dyn_side, static_side) = match (l, r) {
            (Some(c), None) => (c, right.as_ref(), left.as_ref()),
            (None, Some(c)) => (c, left.as_ref(), right.as_ref()),
            // Both sides this table (r.a = r.b) or neither (test.b = 2,
            // NULL = 2): a residual match condition, not a key (measured —
            // DuckDB binds these fine and they filter matches).
            _ => {
                residual.push(c);
                continue;
            }
        };
        // A bare identifier on the static side that also binds in the outer
        // scope is ambiguous (DuckDB rejects it too).
        if let SqlExpr::Identifier(id) = static_side {
            if binder.column(&id.value).is_ok() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{}' in JOIN ON (qualify it)",
                    id.value
                )));
            }
        }
        // The STRUCT-PATH spelling of the same rule: DuckDB decides
        // ambiguity on the HEAD name before it looks at the fields,
        // and the table being joined is not in `binder.joins` yet while its
        // ON binds — so the check has to happen here, against `st` itself.
        // `c0.f0 = s0.c0` where the row table has struct c0 AND s0 has
        // column c0 refuses; qualified spellings resolve normally.
        // (Every spelling of the path: `c0['f0']` is `c0.f0` here too.)
        let dyn_path = match dyn_side {
            SqlExpr::CompoundIdentifier(parts) => Some(parts.clone()),
            _ => binder.struct_access_path_beside(dyn_side, Some(scope_name)),
        };
        if let Some(parts) = dyn_path {
            let head = &parts[0].value;
            let head_in_outer = binder.column(head).is_ok()
                || binder
                    .structs
                    .iter()
                    .any(|s| s.name.eq_ignore_ascii_case(head));
            let head_in_static = st
                .cols
                .iter()
                .enumerate()
                .any(|(ci, c)| {
                    !st.is_leaf_lane(ci as u32) && c.name.eq_ignore_ascii_case(head)
                })
                || st
                    .structs
                    .iter()
                    .any(|s| s.name.eq_ignore_ascii_case(head))
                || st.opaque.iter().any(|(c, _)| c.eq_ignore_ascii_case(head));
            if head_in_outer && head_in_static {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{head}' in JOIN ON (qualify it)"
                )));
            }
        }
        let key = fold(binder.expr(dyn_side)?);
        let key = promote_key(key, st, col)?;
        keys.push(key);
        key_cols.push(JoinKey {
            src: KeySrc::Lane(col),
            cmp,
        });
    }
    Ok((keys, key_cols, residual))
}

/// Promote a dynamic-side key expression to the map's key type, or refuse
/// the pairing by name. An unchanged key means the BUILD side converts
/// instead, while the probe table is materialized (see
/// `duckdb::materialize_map`) — the arms below say which case is which.
pub(super) fn promote_key(key: SExpr, st: &StaticTable, col: u32) -> Result<SExpr, PrepareError> {
    let col_ty = st.cols[col as usize].ty.ty;
    match (key.ty, col_ty) {
        // A DECIMAL probe expression (`ON d.k = a * 0.5`): against a
        // DOUBLE build key DuckDB casts the probe to DOUBLE; every other
        // pairing needs a decimal probe lane the key path does not have.
        (Ty::Dec(..), Ty::F64) => Ok(dec_to_float(key)),
        (Ty::Dec(p, s), b) => Err(unsup(format!(
            "a DECIMAL({p},{s}) join key expression against {} -- DECIMAL probe \
             keys are served against DOUBLE build keys only",
            b.name()
        ))),
        (a, b) if a == b => Ok(key),
        // Integer widths share the key lane; the map stores i64 bits.
        (a, b) if a.is_int() && b.is_int() => Ok(key),
        (a, Ty::F64) if a.is_int() => Ok(promote_f64(key)),
        // Static-side ints promote at materialization: the map key type
        // (the key expression's type) becomes F64 and the build side is
        // converted while the probe table is built. EVERY integer width:
        // DuckDB compares all four against a DOUBLE in double space, so
        // refusing the narrow three refused joins DuckDB serves.
        (Ty::F64, b) if b.is_int() => Ok(key),
        // A DECIMAL build key, same precedent, no new key machinery: the
        // lane stays the PROBE's type and the build side converts while the
        // probe table is built (see duckdb::materialize_map).
        //
        // Against an INTEGER probe DuckDB casts the integer UP to the
        // decimal (ImplicitCastBigint, cast_rules.cpp:96-107), exactly — so
        // equality is decidable at build time and a non-integral build row
        // simply drops. The one shape that refuses is the CAPPED comparison
        // width: when the integer's decimal width plus the scale exceeds
        // 38, DuckDB caps the width and the integer's per-row cast can FAIL
        // (measured: CAST(1 AS DECIMAL(38,30)) = 10000000000::BIGINT is a
        // Conversion Error, = 1::BIGINT is true). Reproducing that needs a
        // row-time trap the key path has no shape for.
        (a, Ty::Dec(_, s)) if a.is_int() => {
            if u32::from(int_dec_width(a)) + u32::from(s) > 38 {
                return Err(PrepareError::Bind(format!(
                    "cannot join {} with {}: DuckDB compares these as DECIMAL(38,{s}) \
                     and the integer cast can fail per row",
                    a.name(),
                    col_ty.name()
                )));
            }
            Ok(key)
        }
        // Against a DOUBLE probe only decimal->double is a legal implicit
        // cast, so the DECIMAL side casts DOWN and the comparison is LOSSY.
        // The lane stays F64 and the build side converts with DuckDB's own
        // algorithm, reproducing the loss rather than hiding it.
        (Ty::F64, Ty::Dec(..)) => Ok(key),
        // Numeric probe vs VARCHAR build key: the build side converts while
        // the probe table is built (duckdb::materialize_map), to the probe's
        // type, as DuckDB casts it.
        (a, Ty::Str) if a.is_int() || a == Ty::F64 => Ok(key),
        // VARCHAR probe vs numeric build key: DuckDB casts the probe per row
        // to the build column's type; a value that cannot convert traps,
        // as CAST does.
        (Ty::Str, b) if b.is_int() || b == Ty::F64 => {
            let nullable = key.nullable;
            Ok(SExpr {
                kind: SKind::Cast {
                    inner: Box::new(key),
                    trying: false,
                },
                ty: b,
                nullable,
            })
        }
        (a, b) => Err(PrepareError::Bind(format!(
            "cannot join {} with {} (ON '{}')",
            a.name(),
            b.name(),
            st.cols[col as usize].name
        ))),
    }
}

/// Every shared-column NAME a static table offers a NATURAL/USING join.
/// DuckDB intersects case-insensitive NAME SETS with no type inspection at
/// all (`bind_joinref.cpp:185-208`), so a struct head and an opaque head
/// are shared names exactly like a scalar; only a struct LEAF lane is not a
/// name (its dotted spelling is display, never an identifier).
pub(super) fn static_head_names(st: &StaticTable) -> Vec<String> {
    let mut names: Vec<String> = st
        .cols
        .iter()
        .enumerate()
        .filter(|(i, _)| !st.is_leaf_lane(*i as u32))
        .map(|(_, c)| c.name.clone())
        .collect();
    names.extend(st.structs.iter().map(|s| s.name.clone()));
    names.extend(st.opaque.iter().map(|(c, _)| c.clone()));
    names
}

/// One shared NATURAL/USING column -> the map keys it becomes, as
/// `(probe expression, static key, IS NOT DISTINCT FROM)` triples.
///
/// A scalar is one key. A STRUCT expands into ordinary
/// composite keys: one PLAIN presence key for the whole struct (a NULL
/// struct never matches, on either side), one IS-NOT-DISTINCT presence key
/// per nested node, and one IS-NOT-DISTINCT key per leaf, pairing fields BY
/// NAME case-insensitively like DuckDB's struct cast. That is
/// `row_matcher.cpp:379-382` written out.
///
/// Everything the encoding cannot carry REFUSES BY NAME here; nothing is
/// ever dropped from the key set, because a dropped key emits rows DuckDB
/// never produces.
pub(super) fn shared_key(
    binder: &Binder<'_>,
    st: &StaticTable,
    name: &str,
    many: bool,
) -> Result<Vec<(SExpr, JoinKey)>, PrepareError> {
    // Both sides' SHAPE for this name. A name absent on either side is not
    // shared at all, and every refusal below is gated on it being shared —
    // refusing an unshared opaque column would refuse queries DuckDB serves.
    enum Side<'s> {
        Struct(&'s StructCol),
        Scalar,
        Opaque(String),
    }
    let st_side = if let Some(s) = st.structs.iter().find(|s| s.name.eq_ignore_ascii_case(name)) {
        Some(Side::Struct(s))
    } else if let Some((c, aty)) = st.opaque.iter().find(|(c, _)| c.eq_ignore_ascii_case(name)) {
        Some(Side::Opaque(format!(
            "type {aty} on static table '{}' (column '{c}')",
            st.name
        )))
    } else {
        st.cols
            .iter()
            .enumerate()
            .find(|(i, c)| !st.is_leaf_lane(*i as u32) && c.name.eq_ignore_ascii_case(name))
            .map(|(i, _)| i as u32)
            .map(|_| Side::Scalar)
    };
    // The left scope's shape. A bare struct head never binds as a COLUMN
    // (it is a whole value), and neither does an opaque one, so those are
    // looked up by name; `binder.column` answers for every plain scalar in
    // scope — the row model AND any earlier join.
    let row_side = if let Some(s) = binder
        .structs
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case(name))
    {
        Some(Side::Struct(s))
    } else if binder.opaque.iter().any(|(_, c)| c.eq_ignore_ascii_case(name)) {
        Some(Side::Opaque(format!(
            "a non-scalar type on row table '{}'",
            binder.this_name
        )))
    } else if let Some(sj) = binder.joins.iter().find(|sj| {
        sj.table
            .structs
            .iter()
            .any(|s| s.name.eq_ignore_ascii_case(name))
            || sj.table.opaque.iter().any(|(c, _)| c.eq_ignore_ascii_case(name))
    }) {
        Some(Side::Opaque(format!(
            "a non-scalar type on the already-joined table '{}'",
            sj.name
        )))
    } else {
        binder.column(name).ok().map(|_| Side::Scalar)
    };
    let (Some(row_side), Some(st_side)) = (row_side, st_side) else {
        // Not a shared name — the caller decides (NATURAL skips it, USING
        // says which side it is missing from).
        return Ok(Vec::new());
    };
    let cannot = |what: String| {
        Err(unsup(format!(
            "shared join column '{name}' has {what}, so this engine cannot key on it"
        )))
    };
    match (row_side, st_side) {
        (Side::Opaque(w), _) | (_, Side::Opaque(w)) => cannot(w),
        (Side::Struct(row_sc), Side::Struct(st_sc)) => {
            if many {
                // lower.rs refuses IS NOT DISTINCT FROM keys under the
                // fan-out loop, which implements plain equality only. The
                // generic message names the comparison; this one names the
                // column, which is what a reader can act on.
                return Err(unsup(format!(
                    "struct join column '{name}' as a join key under shape='many' \
                     (the fan-out loop implements plain equality only)"
                )));
            }
            struct_keys(binder, st, name, row_sc, st_sc)
        }
        (Side::Struct(_), Side::Scalar) | (Side::Scalar, Side::Struct(_)) => {
            cannot("a struct on one side of the join and a scalar on the other".to_string())
        }
        (Side::Scalar, Side::Scalar) => {
            let col = st
                .cols
                .iter()
                .enumerate()
                .find(|(i, c)| !st.is_leaf_lane(*i as u32) && c.name.eq_ignore_ascii_case(name))
                .map(|(i, _)| i as u32)
                .expect("Side::Scalar came from this lookup");
            let key = fold(binder.column(name)?);
            Ok(vec![(
                promote_key(key, st, col)?,
                JoinKey {
                    src: KeySrc::Lane(col),
                    cmp: KeyCmp::Eq,
                },
            )])
        }
    }
}

/// The struct arm of [`shared_key`]: the top-level presence key, then the
/// per-node / per-leaf walk.
pub(super) fn struct_keys(
    binder: &Binder<'_>,
    st: &StaticTable,
    name: &str,
    row_sc: &StructCol,
    st_sc: &StructCol,
) -> Result<Vec<(SExpr, JoinKey)>, PrepareError> {
    // PLAIN, not IS NOT DISTINCT FROM: level 0 propagates NULL, so a NULL
    // struct misses even another NULL struct. A plain key already does
    // exactly that — the build side drops NULL-key rows and the probe side
    // ANDs the validity into `keys_valid`.
    let mut out = vec![(
        binder.present_key(&[row_sc.name.clone()]),
        JoinKey {
            src: KeySrc::Present(vec![st_sc.name.clone()]),
            cmp: KeyCmp::Eq,
        },
    )];
    let mut rp = vec![row_sc.name.clone()];
    let mut sp = vec![st_sc.name.clone()];
    walk_key_fields(
        binder,
        st,
        name,
        &row_sc.fields,
        &st_sc.fields,
        &mut rp,
        &mut sp,
        &mut out,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn walk_key_fields(
    binder: &Binder<'_>,
    st: &StaticTable,
    name: &str,
    rf: &[StructField],
    sf: &[StructField],
    rp: &mut Vec<String>,
    sp: &mut Vec<String>,
    out: &mut Vec<(SExpr, JoinKey)>,
) -> Result<(), PrepareError> {
    let mismatch = |side: &str, at: &[String], field: &str| {
        Err(unsup(format!(
            "shared join column '{name}': field '{}' exists on the {side} side only, \
             so the two structs cannot be paired by name",
            at.iter()
                .cloned()
                .chain(std::iter::once(field.to_string()))
                .collect::<Vec<_>>()
                .join(".")
        )))
    };
    // Both directions: a field the OTHER side lacks would otherwise drop
    // silently out of the key set, and a dropped key emits rows DuckDB
    // never produces.
    for s in sf {
        if !rf.iter().any(|r| r.name.eq_ignore_ascii_case(&s.name)) {
            return mismatch("static", sp, &s.name);
        }
    }
    for r in rf {
        let Some(s) = sf.iter().find(|f| f.name.eq_ignore_ascii_case(&r.name)) else {
            return mismatch("row", rp, &r.name);
        };
        // Each side keeps its OWN spelling: the paths are read against that
        // side's data, where field lookup is case-SENSITIVE.
        rp.push(r.name.clone());
        sp.push(s.name.clone());
        match (&r.node, &s.node) {
            (StructNode::Leaf(l), StructNode::Leaf(c)) => {
                let rc = &binder.in_cols[*l as usize];
                let key = SExpr {
                    kind: SKind::Col(*l),
                    ty: rc.ty.ty,
                    nullable: rc.ty.nullable,
                };
                // IS NOT DISTINCT FROM: a NULL field equals a NULL field
                // (the nested walk's child predicate).
                out.push((
                    promote_key(key, st, *c)?,
                    JoinKey {
                        src: KeySrc::Lane(*c),
                        cmp: KeyCmp::NotDistinct,
                    },
                ));
            }
            (StructNode::Nested(rn), StructNode::Nested(sn)) => {
                // An inner node's NULL is a VALUE: NULL-node matches
                // NULL-node and misses a present node. This is the pair the
                // DISCRIMINATOR cells demand and that leaf lanes alone
                // cannot supply.
                out.push((
                    binder.present_key(rp),
                    JoinKey {
                        src: KeySrc::Present(sp.clone()),
                        cmp: KeyCmp::NotDistinct,
                    },
                ));
                walk_key_fields(binder, st, name, rn, sn, rp, sp, out)?;
            }
            (StructNode::Opaque, _) | (_, StructNode::Opaque) => {
                return Err(unsup(format!(
                    "shared join column '{name}': field '{}' has no scalar lane \
                     (a non-vocabulary type), so this \
                     engine cannot key on the struct",
                    rp.join(".")
                )))
            }
            _ => {
                return Err(unsup(format!(
                    "shared join column '{name}': field '{}' is a struct on one \
                     side of the join and a scalar on the other",
                    rp.join(".")
                )))
            }
        }
        rp.pop();
        sp.pop();
    }
    Ok(())
}

/// Does this key pairing LOSE the static column's value? A DOUBLE
/// probe against an integer column compares in double space, so the build
/// side is converted while the probe table is built and the comparison no
/// longer names one i64 (every i64 above 2^53 shares its double with a
/// neighbour). Such a column rides as a shadow VALUE lane as well, so
/// [`Binder::key_lane`] can read the real value instead of rebuilding it.
pub(super) fn key_is_lossy(key_ty: Ty, col_ty: Ty) -> bool {
    key_ty == Ty::F64 && col_ty.is_int()
}

/// The join's value lanes: every non-key column, then a shadow lane per
/// LOSSY key column (appended, so ordinary lane positions never shift).
pub(super) fn val_cols_for(st: &StaticTable, key_cols: &[JoinKey], keys: &[SExpr]) -> Vec<u32> {
    // A struct key's LEAF lanes leave `val_cols` with every other key
    // column, or the struct's leaves would be emitted as separate output
    // columns on the static side. A PRESENCE key has no lane, so it has
    // nothing of its own to exclude.
    let mut val: Vec<u32> = (0..st.cols.len() as u32)
        .filter(|c| !key_cols.iter().any(|k| k.src == KeySrc::Lane(*c)))
        .collect();
    for (kp, k) in key_cols.iter().enumerate() {
        if let KeySrc::Lane(c) = &k.src {
            if key_is_lossy(keys[kp].ty, st.cols[*c as usize].ty.ty) {
                val.push(*c);
            }
        }
    }
    val
}

/// Value-lane positions that are SHADOWS of a lossy key column: they carry
/// the key's real value for [`Binder::key_lane`] only and are NOT bindings —
/// name resolution and `*` must skip them or the column would bind twice.
pub(super) fn is_shadow_lane(sj: &ScopeJoin, pos: usize) -> bool {
    sj.key_cols
        .iter()
        .any(|k| k.src == KeySrc::Lane(sj.val_cols[pos]))
}

/// One traversal answering the residual SCOPE questions about a bound
/// ON residual: `right`/`left` — does it reference THIS join's columns / any
/// other scope; `known` — was every node classifiable at all, since an
/// unrecognised node's children are never visited and so `left`/`right` are
/// not trustworthy past it.
///
/// Trap-freeness is deliberately NOT computed here — it is
/// [`plan::may_trap`], whose one consumer is the JOIN ON residual rule
/// below. Acceptance rule at the call site:
/// `!may_trap(e) || (left && right && known)` — measured: DuckDB scan-pushes
/// single-side residuals (eager trap timing) but evaluates both-sides
/// residuals per candidate pair, which our hit-guarded lowering matches.
pub(super) fn scan_residual(e: &SExpr, j: u32, right: &mut bool, left: &mut bool, known: &mut bool) {
    match &e.kind {
        SKind::StaticCol { join, .. } | SKind::JoinHit(join) => {
            if *join == j {
                *right = true;
            } else {
                *left = true;
            }
        }
        SKind::Col(_) => *left = true,
        SKind::Lit(_) | SKind::NullOf => {}
        SKind::Cmp { a, b, .. }
        | SKind::And { a, b }
        | SKind::Or { a, b }
        | SKind::Arith { a, b, .. }
        | SKind::DecArith { a, b, .. } => {
            scan_residual(a, j, right, left, known);
            scan_residual(b, j, right, left, known);
        }
        SKind::Not(a)
        | SKind::IsNull { inner: a, .. }
        | SKind::IntToFloat(a)
        | SKind::DecToFloat(a)
        | SKind::IntToDec { a, .. }
        | SKind::DecCast(a)
        | SKind::DecTryCast(a)
        | SKind::DecUnary { a, .. }
        | SKind::IntToFloat32(a)
        // A DOUBLE unary minus is plain arithmetic and is scanned through.
        // The libm-backed f64 unaries are not classifiable.
        | SKind::MathF1 { op: NumOp1::Fneg, a } => {
            scan_residual(a, j, right, left, known);
        }
        SKind::Case { arms, default } => {
            for (c, r) in arms {
                scan_residual(c, j, right, left, known);
                scan_residual(r, j, right, left, known);
            }
            if let Some(d) = default {
                scan_residual(d, j, right, left, known);
            }
        }
        // Anything else: not classifiable — the caller must reject rather
        // than risk the permissive both-sides path on a wrong guess.
        _ => *known = false,
    }
}

pub(super) fn collect_conjuncts<'e>(e: &'e SqlExpr, out: &mut Vec<&'e SqlExpr>) {
    match e {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
        } => {
            collect_conjuncts(left, out);
            collect_conjuncts(right, out);
        }
        SqlExpr::Nested(inner) => collect_conjuncts(inner, out),
        other => out.push(other),
    }
}

/// A struct-LEAF spelling of this static table's column, as an ON-key
/// candidate: `v.x` (bare struct head), `d.v.x` (through the relation), or
/// `d.v` where `d` has no column `v` but a struct `d` (DuckDB retries a
/// qualifier that misses as a struct head). The walk is by segment, so a
/// leaf's dotted display name never matches. A bare head that names a
/// relation in scope is that relation (measured: `FROM t AS v ... v.x` is
/// t's column x), and one that also binds as a column elsewhere in scope is
/// DuckDB's ambiguity error, decided on the head alone. A path that does not
/// reach a leaf is not a key; the residual binder then words the refusal.
pub(super) fn static_leaf_of(
    parts: &[sqlparser::ast::Ident],
    st: &StaticTable,
    scope_name: &str,
    binder: &Binder<'_>,
) -> Result<Option<u32>, PrepareError> {
    let is = |a: &sqlparser::ast::Ident, b: &str| a.value.eq_ignore_ascii_case(b);
    let walk = |head: &sqlparser::ast::Ident, fields: &[sqlparser::ast::Ident]| {
        st.structs
            .iter()
            .find(|sc| is(head, &sc.name))
            .and_then(|sc| walk_fields(&sc.fields, fields).ok())
    };
    let (head, rest) = parts.split_first().expect("a compound identifier has parts");
    if is(head, scope_name) {
        // `d.c` with a real column `c` is the column, never a struct path.
        let column = rest.len() == 1
            && st
                .cols
                .iter()
                .enumerate()
                .any(|(ci, c)| !st.is_leaf_lane(ci as u32) && is(&rest[0], &c.name));
        if column {
            return Ok(None);
        }
        if rest.len() >= 2 {
            if let Some(leaf) = walk(&rest[0], &rest[1..]) {
                return Ok(Some(leaf));
            }
        }
        return Ok(if rest.len() == 1 { walk(head, rest) } else { None });
    }
    let Some(leaf) = walk(head, rest) else {
        return Ok(None);
    };
    let is_rel = is(head, &binder.this_name)
        || binder.joins.iter().any(|sj| is(head, &sj.name));
    if is_rel {
        return Ok(None);
    }
    let elsewhere = binder.this_col_with_fields(&head.value, rest).is_some()
        || binder.joins.iter().any(|sj| head_hits_in_join(sj, &head.value) > 0);
    if elsewhere {
        return Err(PrepareError::Bind(format!(
            "ambiguous column '{}' (qualify it)",
            head.value
        )));
    }
    Ok(Some(leaf))
}

/// Does `e` name a column of the static table being joined? Qualified form
/// matches on the join's scope name; a bare identifier matches if the table
/// has that column.
pub(super) fn static_col_of(
    e: &SqlExpr,
    st: &StaticTable,
    scope_name: &str,
    scope_schema: &str,
    binder: &Binder<'_>,
) -> Result<Option<u32>, PrepareError> {
    let name = match e {
        SqlExpr::Identifier(id) => &id.value,
        SqlExpr::CompoundIdentifier(parts) => {
            // Through the schema the relation lives in (and the `memory`
            // catalog): DuckDB's FIRST reading of `a.b.c` is schema.table.
            // column, so the stripped spelling is tried before any other.
            // An aliased relation has no schema (`scope_schema` empty).
            let is = |i: usize, name: &str| {
                !name.is_empty() && parts[i].value.eq_ignore_ascii_case(name)
            };
            let skip = if parts.len() >= 4
                && is(0, "memory")
                && is(1, scope_schema)
                && is(2, scope_name)
            {
                2
            } else if parts.len() >= 3 && is(0, scope_schema) && is(1, scope_name) {
                1
            } else {
                0
            };
            if skip > 0 {
                let stripped = SqlExpr::CompoundIdentifier(parts[skip..].to_vec());
                if let Ok(Some(c)) = static_col_of(&stripped, st, scope_name, "", binder) {
                    return Ok(Some(c));
                }
            }
            if let Some(leaf) = static_leaf_of(parts, st, scope_name, binder)? {
                return Ok(Some(leaf));
            }
            match parts.as_slice() {
                [t, c] if t.value.eq_ignore_ascii_case(scope_name) => &c.value,
                _ => return Ok(None),
            }
        }
        SqlExpr::Nested(inner) => {
            return static_col_of(inner, st, scope_name, scope_schema, binder)
        }
        // A struct leaf in a non-dotted spelling (`s.c['f']`, `(s.c).f`,
        // `struct_extract(s.c, 'f')`) is the same key as `s.c.f`.
        SqlExpr::CompoundFieldAccess { .. } | SqlExpr::Function(_) => {
            return match binder.struct_access_path_beside(e, Some(scope_name)) {
                Some(path) => static_col_of(
                    &SqlExpr::CompoundIdentifier(path),
                    st,
                    scope_name,
                    scope_schema,
                    binder,
                ),
                None => Ok(None),
            };
        }
        _ => return Ok(None),
    };
    let mut hit = None;
    for (i, c) in st.cols.iter().enumerate() {
        // A struct leaf is reached by PATH (`static_leaf_of` above), never
        // here: its dotted display name is not an identifier.
        if st.is_leaf_lane(i as u32) {
            continue;
        }
        if c.name.eq_ignore_ascii_case(name) {
            if hit.is_some() {
                return Err(PrepareError::Bind(format!(
                    "ambiguous column '{name}' in static table '{}'",
                    st.name
                )));
            }
            hit = Some(i as u32);
        }
    }
    // Qualified misses are errors; bare misses just mean "not the static
    // side" — the caller will try binding it dynamically. Either way, a
    // column the catalogue could not SERVE is present and must say so: a
    // join key is the likeliest place to meet one, since ids are exactly
    // where UBIGINT and friends show up.
    if hit.is_none() {
        if let Some(err) = opaque_static_refusal(st, name, scope_name) {
            return Err(err);
        }
        if let SqlExpr::CompoundIdentifier(_) = e {
            return Err(PrepareError::Bind(format!(
                "column '{name}' does not exist in '{scope_name}'"
            )));
        }
    }
    Ok(hit)
}
