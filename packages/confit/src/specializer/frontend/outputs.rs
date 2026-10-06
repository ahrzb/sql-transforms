//! Struct-valued outputs: a projection item, star column or struct field
//! whose value is a whole struct. No struct value exists at runtime. A
//! struct leaves the engine as lanes, its validity first and then each
//! field's lanes in field order, and the boundary assembles them
//! ([`super::super::WideShape`]).

use super::*;

/// A projection value: a scalar expression, or a struct assembled from
/// lanes.
pub(super) enum OutVal {
    Scalar(SExpr),
    Struct {
        /// A non-nullable boolean: false when the struct is NULL, which is
        /// not the same as a struct of NULLs.
        valid: SExpr,
        fields: Vec<(String, OutVal)>,
    },
}

/// What a column reference resolved to: a scalar lane, or a struct node
/// (the whole value of a struct column or of a nested struct field).
pub(super) enum Resolved {
    Lane(SExpr),
    Node(NodeRef),
}

impl Resolved {
    /// The scalar a scalar context needs. A struct node refuses there, with
    /// the refusal the reference carries.
    pub(super) fn scalar(self) -> Result<SExpr, PrepareError> {
        match self {
            Resolved::Lane(e) => Ok(e),
            Resolved::Node(n) => Err(unsup(n.refusal)),
        }
    }
}

/// A struct node a reference landed on.
#[derive(Clone)]
pub(super) struct NodeRef {
    pub(super) src: NodeSrc,
    /// The SEGMENT path from the column down, in the declared spelling.
    pub(super) path: Vec<String>,
    /// What a scalar context reports for this reference.
    pub(super) refusal: String,
}

#[derive(Clone, Copy)]
pub(super) enum NodeSrc {
    /// A struct column of the request table.
    Row,
    /// A struct column of join `j`'s static table.
    Static(usize),
}

/// `parts` walked down `sc` (ASCII-case-insensitively, like
/// [`walk_fields`]) as the declared field names, behind the column's own.
/// The caller has walked `parts` already, so every part matches.
pub(super) fn declared_path(sc: &StructCol, parts: &[Ident]) -> Vec<String> {
    let mut path = vec![sc.name.clone()];
    let mut cur = sc.fields.as_slice();
    for p in parts {
        let Some(f) = cur.iter().find(|f| f.name.eq_ignore_ascii_case(&p.value)) else {
            break;
        };
        path.push(f.name.clone());
        if let StructNode::Nested(n) = &f.node {
            cur = n;
        }
    }
    path
}

fn always(valid: bool) -> SExpr {
    SExpr {
        kind: SKind::Lit(Lit::I1(valid)),
        ty: Ty::I1,
        nullable: false,
    }
}

fn is_not_null(e: SExpr) -> SExpr {
    SExpr {
        kind: SKind::IsNull {
            negated: true,
            inner: Box::new(e),
        },
        ty: Ty::I1,
        nullable: false,
    }
}

impl OutVal {
    /// This value read only where `guard` holds: a scalar that can trap
    /// becomes `CASE WHEN guard THEN value END`, which the engine evaluates
    /// only on the rows the guard selects, as DuckDB evaluates a CASE arm.
    /// A scalar that cannot trap stays as it is, read on every row: the
    /// boundary NULLs it where the struct is NULL, and a CASE per field
    /// would cost a branch per field.
    fn guarded(self, guard: &SExpr) -> OutVal {
        match self {
            OutVal::Scalar(e) if !can_trap(&e) => OutVal::Scalar(e),
            OutVal::Scalar(e) => {
                let ty = e.ty;
                OutVal::Scalar(SExpr {
                    kind: SKind::Case {
                        arms: vec![(guard.clone(), e)],
                        default: None,
                    },
                    ty,
                    nullable: true,
                })
            }
            OutVal::Struct { valid, fields } => OutVal::Struct {
                valid: and(guard.clone(), valid),
                fields: fields
                    .into_iter()
                    .map(|(n, v)| (n, v.guarded(guard)))
                    .collect(),
            },
        }
    }

    /// The value as out lanes named after `base`, and its shape.
    pub(super) fn into_lanes(self, base: &str) -> (Vec<(String, SExpr)>, super::super::WideShape) {
        let mut lanes = Vec::new();
        let shape = self.flatten(base, &mut lanes);
        (lanes, shape)
    }

    fn flatten(self, at: &str, lanes: &mut Vec<(String, SExpr)>) -> super::super::WideShape {
        use super::super::WideShape;
        match self {
            OutVal::Scalar(e) => {
                lanes.push((at.to_string(), e));
                WideShape::Scalar
            }
            OutVal::Struct { valid, fields } => {
                lanes.push((format!("{at}\u{1}valid"), valid));
                let shape = fields
                    .into_iter()
                    .enumerate()
                    .map(|(j, (n, v))| {
                        let s = v.flatten(&format!("{at}\u{1}{j}"), lanes);
                        (n, s)
                    })
                    .collect();
                WideShape::Struct(shape)
            }
        }
    }
}

/// The shape of a wide extern's lanes: a list when its fields are unnamed,
/// a struct of scalars when they are named.
pub(super) fn extern_shape(
    lanes: &[(String, SExpr)],
    names: Vec<String>,
) -> super::super::WideShape {
    use super::super::WideShape;
    if names.is_empty() {
        return WideShape::List(lanes.len() as u32 - 1);
    }
    WideShape::Struct(names.into_iter().map(|n| (n, WideShape::Scalar)).collect())
}

/// `a AND b` over two non-nullable booleans.
fn and(a: SExpr, b: SExpr) -> SExpr {
    if let SKind::Lit(Lit::I1(true)) = b.kind {
        return a;
    }
    SExpr {
        kind: SKind::And {
            a: Box::new(a),
            b: Box::new(b),
        },
        ty: Ty::I1,
        nullable: false,
    }
}

/// `e` without the parentheses around it.
fn unnest(mut e: &SqlExpr) -> &SqlExpr {
    while let SqlExpr::Nested(i) = e {
        e = i;
    }
    e
}

fn is_null_literal(e: &SqlExpr) -> bool {
    matches!(unnest(e), SqlExpr::Value(v) if matches!(v.value, SqlValue::Null))
}

/// A value's type: its field names and leaf types, nested as it is.
#[derive(PartialEq)]
enum Skel {
    Leaf(Ty),
    Struct(Vec<(String, Skel)>),
}

impl OutVal {
    fn skel(&self) -> Skel {
        match self {
            OutVal::Scalar(e) => Skel::Leaf(e.ty),
            OutVal::Struct { fields, .. } => {
                Skel::Struct(fields.iter().map(|(n, v)| (n.clone(), v.skel())).collect())
            }
        }
    }
}

/// The CASE over `conds` of `arms`, with `default` as its ELSE (`None`:
/// no ELSE), every value of type `like`. A `None` arm is a NULL struct:
/// not there, its fields NULL.
fn case_of(
    conds: &[SExpr],
    arms: Vec<Option<OutVal>>,
    default: Option<Option<OutVal>>,
    like: &Skel,
) -> OutVal {
    match like {
        Skel::Leaf(ty) => {
            let leaf = |v: Option<OutVal>| match v {
                Some(OutVal::Scalar(e)) => e,
                _ => null_of(*ty),
            };
            OutVal::Scalar(fold(SExpr {
                kind: SKind::Case {
                    arms: conds.iter().cloned().zip(arms.into_iter().map(leaf)).collect(),
                    default: default.map(|d| Box::new(leaf(d))),
                },
                ty: *ty,
                nullable: true,
            }))
        }
        Skel::Struct(skel) => {
            // Each value's validity, and its fields in field order.
            let split = |v: Option<OutVal>| match v {
                Some(OutVal::Struct { valid, fields }) => {
                    (valid, fields.into_iter().map(|(_, f)| Some(f)).collect())
                }
                _ => (always(false), skel.iter().map(|_| None).collect()),
            };
            let (valid, mut columns): (Vec<SExpr>, Vec<Vec<Option<OutVal>>>) =
                arms.into_iter().map(split).unzip();
            let (default_valid, mut default_fields) = match default {
                Some(d) => {
                    let (v, f) = split(d);
                    (v, Some(f))
                }
                None => (always(false), None),
            };
            let valid = fold(SExpr {
                kind: SKind::Case {
                    arms: conds.iter().cloned().zip(valid).collect(),
                    default: Some(Box::new(default_valid)),
                },
                ty: Ty::I1,
                nullable: false,
            });
            let mut fields = Vec::with_capacity(skel.len());
            for (j, (name, s)) in skel.iter().enumerate().rev() {
                let arms = columns.iter_mut().map(|c| c.pop().expect("one value per field")).collect();
                let default = default_fields.as_mut().map(|f| f.pop().expect("one value per field"));
                debug_assert!(columns.iter().all(|c| c.len() == j));
                fields.push((name.clone(), case_of(conds, arms, default, s)));
            }
            fields.reverse();
            OutVal::Struct { valid, fields }
        }
    }
}

impl Binder<'_> {
    /// The value of struct node `node`: its presence as the validity, and
    /// each field as a lane or, for a nested struct, a value of its own.
    pub(super) fn node_value(&self, node: &NodeRef) -> Result<OutVal, PrepareError> {
        let structs: &[StructCol] = match node.src {
            NodeSrc::Row => self.structs,
            NodeSrc::Static(j) => &self.joins[j].table.structs,
        };
        let internal = || {
            PrepareError::Internal(format!("struct node '{}' not found", node.path.join(".")))
        };
        let sc = structs
            .iter()
            .find(|s| s.name == node.path[0])
            .ok_or_else(internal)?;
        let mut fields = sc.fields.as_slice();
        for seg in &node.path[1..] {
            match fields.iter().find(|f| &f.name == seg).map(|f| &f.node) {
                Some(StructNode::Nested(n)) => fields = n,
                _ => return Err(internal()),
            }
        }
        if let NodeSrc::Static(j) = node.src {
            // A USING/NATURAL struct key is merged into the request side;
            // its own whole value is not served.
            if key_struct(&self.joins[j], &node.path[0]) {
                return Err(self.merged_key(j, &node.path[0]));
            }
        }
        self.node_fields(node, &node.path, fields)
    }

    /// Whether the struct at `path` under `node`'s column is there: false
    /// where it is NULL (a NULL ancestor included), never because its
    /// fields are.
    fn node_valid(&self, node: &NodeRef, path: &[String]) -> Result<SExpr, PrepareError> {
        Ok(match node.src {
            NodeSrc::Row => self.present_lane(path),
            NodeSrc::Static(j) => {
                let sj = &self.joins[j];
                // The node's presence rides as a value lane of the join.
                let pos = sj
                    .table
                    .presence_lane(path)
                    .and_then(|ci| sj.val_cols.iter().position(|&v| v == ci))
                    .ok_or_else(|| unsup(node.refusal.clone()))?;
                is_not_null(self.static_lane(j, pos))
            }
        })
    }

    fn node_fields(
        &self,
        node: &NodeRef,
        path: &[String],
        fields: &[StructField],
    ) -> Result<OutVal, PrepareError> {
        if fields.is_empty() {
            // DuckDB has no STRUCT type without fields (it refuses one at
            // input), so there is no value to match.
            return Err(unsup(format!(
                "struct '{}' as a whole value: it has no fields",
                path.join(".")
            )));
        }
        let valid = self.node_valid(node, path)?;
        let mut out = Vec::with_capacity(fields.len());
        for f in fields {
            let v = match &f.node {
                StructNode::Leaf(l) => OutVal::Scalar(match node.src {
                    NodeSrc::Row => {
                        let c = &self.in_cols[*l as usize];
                        SExpr {
                            kind: SKind::Col(*l),
                            ty: c.ty.ty,
                            nullable: c.ty.nullable,
                        }
                    }
                    NodeSrc::Static(j) => {
                        let table = self.joins[j].name.clone();
                        self.static_leaf(j, *l, &table)?
                    }
                }),
                StructNode::Opaque => {
                    return Err(unsup(format!(
                        "struct '{}' as a whole value: its field '{}' has a non-scalar type",
                        path.join("."),
                        f.name
                    )))
                }
                StructNode::Nested(n) => {
                    let mut below = path.to_vec();
                    below.push(f.name.clone());
                    self.node_fields(node, &below, n)?
                }
            };
            out.push((f.name.clone(), v));
        }
        Ok(OutVal::Struct { valid, fields: out })
    }

    /// `e` as a whole struct value, when it is one: a struct column or a
    /// nested struct field in any spelling, a relation's row struct, a
    /// `struct_pack` or a struct literal (each field a value again), a
    /// NAMED extern's output, or a CASE whose results are such values or
    /// NULL. `Ok(None)` when it is
    /// not one, and the caller binds `e` as a scalar.
    pub(super) fn struct_value(&self, e: &SqlExpr) -> Result<Option<OutVal>, PrepareError> {
        let inner = unnest(e);
        if let Some(v) = self.packed_value(inner)? {
            return Ok(Some(v));
        }
        if let Some(rel) = self.row_struct(inner) {
            // Its columns as `rel.*` lists them, never NULL: a LEFT miss is
            // a struct of NULLs (measured).
            let fields = self.expand_star(Some(rel), &Default::default())?;
            return Ok(Some(OutVal::Struct {
                valid: always(true),
                fields,
            }));
        }
        match self.struct_ref(inner) {
            Some(n) => self.node_value(&n).map(Some),
            None => self.extern_value(inner),
        }
    }

    /// The relation whose whole row struct a bare name reads: a relation
    /// in scope that no column shares the name of (a column binds first,
    /// as on DuckDB).
    fn row_struct<'e>(&self, e: &'e SqlExpr) -> Option<&'e str> {
        match e {
            SqlExpr::Identifier(id)
                if self.is_relation(&id.value) && !self.binds_as_column(&id.value) =>
            {
                Some(&id.value)
            }
            _ => None,
        }
    }

    /// The struct node `e` reads, when it reads one: a struct column or a
    /// nested struct field, in any spelling.
    fn struct_ref(&self, e: &SqlExpr) -> Option<NodeRef> {
        let resolved = match e {
            SqlExpr::Identifier(id) => self.column_res(&id.value),
            SqlExpr::CompoundIdentifier(parts) => self.compound_res(parts),
            SqlExpr::CompoundFieldAccess { .. } | SqlExpr::Function(_) => {
                self.compound_res(&self.struct_access_path(e)?)
            }
            _ => return None,
        };
        // An error, or a lane, is the scalar path's to report and bind.
        match resolved {
            Ok(Resolved::Node(n)) => Some(n),
            _ => None,
        }
    }

    /// `e IS NOT NULL` when `e` reads a whole struct node or a relation's
    /// row struct, or calls a struct-valued UDF: the struct's own validity.
    /// `None` for anything else. A struct the query builds is not read here:
    /// DuckDB builds every field before it tests the struct, so a field's
    /// trap fires there. A UDF's validity is its call, which runs.
    pub(super) fn struct_valid(&self, e: &SqlExpr) -> Result<Option<SExpr>, PrepareError> {
        // A relation's row struct is never NULL.
        if self.row_struct(unnest(e)).is_some() {
            return Ok(Some(always(true)));
        }
        let Some(n) = self.struct_ref(unnest(e)) else {
            return Ok(match self.extern_value(unnest(e))? {
                Some(OutVal::Struct { valid, .. }) => Some(valid),
                _ => None,
            });
        };
        if let NodeSrc::Static(j) = n.src {
            if key_struct(&self.joins[j], &n.path[0]) {
                return Err(self.merged_key(j, &n.path[0]));
            }
        }
        self.node_valid(&n, &n.path).map(Some)
    }

    /// The refusal for a static struct key that USING or NATURAL merged,
    /// read whole on its own side.
    pub(super) fn merged_key(&self, j: usize, column: &str) -> PrepareError {
        unsup(format!(
            "static table '{}' column '{column}' as a whole value: USING or \
             NATURAL merges this struct key (read it unqualified)",
            self.joins[j].name
        ))
    }

    /// A field value inside a struct: a struct value, or a scalar.
    fn field_value(&self, e: &SqlExpr) -> Result<OutVal, PrepareError> {
        if let Some(v) = self.struct_value(e)? {
            return Ok(v);
        }
        // struct_pack(a := NULL) is STRUCT(a INTEGER) on DuckDB —
        // SQLNULL's int32 home.
        Ok(OutVal::Scalar(match self.expr_or_null(e)? {
            None => null_of(Ty::I32),
            Some(x) => fold(x),
        }))
    }

    /// A NAMED extern's output struct: its whole-call validity, then one
    /// lane per declared field. An unnamed one is a list, which a struct
    /// field cannot hold, so it stays on the scalar path and refuses there.
    fn extern_value(&self, e: &SqlExpr) -> Result<Option<OutVal>, PrepareError> {
        let SqlExpr::Function(f) = e else {
            return Ok(None);
        };
        // Named, before anything binds: the list path binds it itself.
        if !self
            .find_udf(&f.name.to_string())
            .is_some_and(|(_, spec)| !spec.ret_names.is_empty())
        {
            return Ok(None);
        }
        let Some((mut lanes, names)) = self.wide_extern_lanes(e, "")? else {
            return Ok(None);
        };
        let fields = lanes.split_off(1);
        let (_, valid) = lanes.pop().expect("the validity lane");
        Ok(Some(OutVal::Struct {
            valid,
            fields: names
                .into_iter()
                .zip(fields)
                .map(|(n, (_, lane))| (n, OutVal::Scalar(lane)))
                .collect(),
        }))
    }

    /// `struct_pack(n := e, ...)` or the struct literal `{'n': e, ...}`,
    /// or a CASE over struct values (see [`Self::case_value`]). `None` when
    /// `e` is none of these.
    ///
    /// Stricter than DuckDB, deliberately: an unnamed `struct_pack(i)`
    /// infers the field name there; here it refuses. Field ACCESS over
    /// struct_pack is `desugar_struct_field`'s bind-time desugar and never
    /// reaches this recognizer.
    fn packed_value(&self, e: &SqlExpr) -> Result<Option<OutVal>, PrepareError> {
        if let SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } = e
        {
            if let Some(v) = self.guard_value(operand.as_deref(), conditions, else_result.as_deref())? {
                return Ok(Some(v));
            }
            return self.case_value(operand.as_deref(), conditions, else_result.as_deref());
        }
        let Some(pairs) = self.pack_fields(e)? else {
            return Ok(None);
        };
        let mut fields = Vec::with_capacity(pairs.len());
        for (n, v) in pairs {
            fields.push((n, self.field_value(v)?));
        }
        Ok(Some(OutVal::Struct {
            valid: always(true),
            fields,
        }))
    }

    /// `CASE WHEN g IS NULL THEN NULL ELSE <struct_pack or literal> END`,
    /// the shape the θ rewrite and a SQL function's `null_when` emit: the
    /// struct is there where `g` is. `None` for any other CASE.
    fn guard_value(
        &self,
        operand: Option<&SqlExpr>,
        conditions: &[sqlparser::ast::CaseWhen],
        else_result: Option<&SqlExpr>,
    ) -> Result<Option<OutVal>, PrepareError> {
        let ([arm], None, Some(alt)) = (conditions, operand, else_result) else {
            return Ok(None);
        };
        // The oracle's own serialization parenthesizes both arms.
        let SqlExpr::IsNull(target) = unnest(&arm.condition) else {
            return Ok(None);
        };
        let packed = unnest(alt);
        if !is_null_literal(&arm.result) || self.pack_fields(packed)?.is_none() {
            return Ok(None);
        }
        let guard = fold(self.is_null(target, true)?);
        // DuckDB evaluates the ELSE arm only on the rows that reach it, so
        // a field that would trap on a NULL-guarded row does not.
        let value = self.packed_value(packed)?.expect("a struct literal");
        Ok(Some(value.guarded(&guard)))
    }

    /// A CASE whose results are struct values of one type, or NULL: each
    /// lane is the CASE of the arms' lanes, so a row computes only the arm
    /// it takes, as on DuckDB. `None` when no result is a struct value. A
    /// struct beside a result of another type refuses, and so do structs
    /// of different types (DuckDB would cast them to a common one).
    fn case_value(
        &self,
        operand: Option<&SqlExpr>,
        conditions: &[sqlparser::ast::CaseWhen],
        else_result: Option<&SqlExpr>,
    ) -> Result<Option<OutVal>, PrepareError> {
        // CASE arms are guarded, as in the scalar CASE (`in_guarded`).
        self.in_guarded.set(self.in_guarded.get() + 1);
        let _guard = GuardScope(&self.in_guarded);
        let results = conditions.iter().map(|w| &w.result).chain(else_result);
        let (mut vals, mut other) = (Vec::with_capacity(conditions.len() + 1), false);
        for r in results {
            if is_null_literal(r) {
                vals.push(None);
                continue;
            }
            match self.struct_value(r)? {
                Some(v @ OutVal::Struct { .. }) => vals.push(Some(v)),
                _ => {
                    other = true;
                    vals.push(None);
                }
            }
        }
        let Some(like) = vals.iter().flatten().next().map(OutVal::skel) else {
            return Ok(None);
        };
        if other || vals.iter().flatten().any(|v| v.skel() != like) {
            return Err(unsup(
                "a CASE whose results are structs of different types, or a struct and \
                 another value",
            ));
        }
        let conds = self.case_conditions(operand, conditions)?;
        let default = else_result.map(|_| vals.pop().expect("the ELSE result"));
        Ok(Some(case_of(&conds, vals, default, &like)))
    }

    /// The named fields of a `struct_pack(...)` call or a struct literal,
    /// in order; `None` when `e` is neither.
    fn pack_fields<'e>(
        &self,
        e: &'e SqlExpr,
    ) -> Result<Option<Vec<(String, &'e SqlExpr)>>, PrepareError> {
        use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
        let pairs: Vec<(String, &SqlExpr)> = match e {
            SqlExpr::Dictionary(fields) => fields
                .iter()
                .map(|f| (f.key.value.clone(), &*f.value))
                .collect(),
            SqlExpr::Function(f) if f.name.to_string().eq_ignore_ascii_case("struct_pack") => {
                if f.uses_odbc_syntax
                    || !matches!(f.parameters, FunctionArguments::None)
                    || f.filter.is_some()
                    || f.null_treatment.is_some()
                    || f.over.is_some()
                    || !f.within_group.is_empty()
                {
                    return Ok(None);
                }
                let FunctionArguments::List(list) = &f.args else {
                    return Ok(None);
                };
                if list.duplicate_treatment.is_some() || !list.clauses.is_empty() {
                    return Ok(None);
                }
                // Every field must be NAMED here.
                let mut pairs = Vec::with_capacity(list.args.len());
                for a in &list.args {
                    match a {
                        FunctionArg::Named {
                            name,
                            arg: FunctionArgExpr::Expr(v),
                            ..
                        } => pairs.push((name.value.clone(), v)),
                        _ => return Ok(None),
                    }
                }
                pairs
            }
            _ => return Ok(None),
        };
        if pairs.is_empty() {
            return Ok(None);
        }
        for (i, (n, _)) in pairs.iter().enumerate() {
            // Measured: DuckDB's binder rejects a duplicate struct entry
            // name, case-insensitively — never serve what batch cannot.
            if pairs[..i].iter().any(|(m, _)| m.eq_ignore_ascii_case(n)) {
                return Err(PrepareError::Bind(format!(
                    "duplicate struct entry name \"{n}\""
                )));
            }
        }
        Ok(Some(pairs))
    }
}

impl Binder<'_> {
    /// Whether DuckDB's binder folds struct `e` to NULL, which makes a
    /// field read over it a bare NULL. A read is `struct_extract`, and
    /// DuckDB's default NULL handling replaces a call whose argument is
    /// foldable and evaluates to NULL by an untyped NULL, which a context
    /// adopts as it adopts a NULL literal (measured:
    /// `(CASE WHEN TRUE THEN NULL ELSE struct_pack(f := 'x') END).f` is
    /// INTEGER, and so is that `|| 'y'`). Foldable means that no part reads
    /// a column or calls an extern, an arm the CASE does not take included:
    /// with `g := x` beside `f`, the read stays VARCHAR. A part that does
    /// not bind answers no, and the read reports the error itself.
    pub(super) fn struct_folds_to_null(&self, e: &SqlExpr) -> bool {
        // As in a CASE arm: a constant that traps is a trap of its row,
        // which DuckDB's fold skips, not a refusal.
        self.in_guarded.set(self.in_guarded.get() + 1);
        let _guard = GuardScope(&self.in_guarded);
        // Cheapest first: most structs are not NULL on the path taken, and
        // that binds only the conditions on it.
        matches!(self.closed_null(e), Ok(Some(true)))
            && matches!(self.foldable(e), Ok(true))
            && matches!(DuckNulls::new(self).closed(e), Ok(true))
    }

    /// Whether `e` binds to a NULL constant.
    fn binds_to_null(&self, e: &SqlExpr) -> bool {
        match self.expr_or_null(e) {
            Ok(None) => true,
            Ok(Some(v)) => matches!(v.kind, SKind::NullOf),
            Err(_) => false,
        }
    }

    /// DuckDB's `IsFoldable` over `e`, a struct value or a scalar.
    fn foldable(&self, e: &SqlExpr) -> Result<bool, PrepareError> {
        let e = unnest(e);
        if let Some((id, calls)) = calls::marker_call(e) {
            return self.foldable(&calls[id].1);
        }
        if let SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } = e
        {
            if !self
                .case_conditions(operand.as_deref(), conditions)?
                .iter()
                .all(|c| self.duck_foldable(c))
            {
                return Ok(false);
            }
            for r in conditions.iter().map(|w| &w.result).chain(else_result.as_deref()) {
                if !self.foldable(r)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        if let Some(pairs) = self.pack_fields(e)? {
            for (_, v) in pairs {
                if !self.foldable(v)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        // A struct column, a relation's row or an extern's output.
        if self.struct_value(e)?.is_some() {
            return Ok(false);
        }
        Ok(self.expr_or_null(e)?.as_ref().is_none_or(|x| self.duck_foldable(x)))
    }

    /// DuckDB's `IsFoldable` of a bound value: [`bind_foldable`], except
    /// that a call of a pure extern or of a tree model is foldable when its
    /// arguments are, since DuckDB registers both without side effects, and
    /// a let is foldable when its value is.
    fn duck_foldable(&self, e: &SExpr) -> bool {
        match &e.kind {
            SKind::Col(_)
            | SKind::Slot(_)
            | SKind::StaticCol { .. }
            | SKind::JoinHit(_)
            | SKind::Shared(_)
            | SKind::Raise(_) => false,
            SKind::ExternCall { ext, .. }
                if self.udfs.get(*ext as usize).is_none_or(|u| u.side_effects) =>
            {
                false
            }
            SKind::Let(i) => {
                let value = self.lets.borrow().get(*i as usize).cloned();
                value.is_some_and(|v| self.duck_foldable(&v))
            }
            _ => e.clone().children_mut().into_iter().all(|c| self.duck_foldable(c)),
        }
    }

    /// Whether struct `e` evaluates to NULL, following the arms DuckDB's
    /// evaluation takes. `None` when a condition on that path is not
    /// foldable, traps or is not evaluable here: DuckDB's binder then leaves
    /// the read to run per row.
    fn closed_null(&self, e: &SqlExpr) -> Result<Option<bool>, PrepareError> {
        let e = unnest(e);
        if is_null_literal(e) {
            return Ok(Some(true));
        }
        if let Some((id, calls)) = calls::marker_call(e) {
            return self.closed_null(&calls[id].1);
        }
        let SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } = e
        else {
            // A struct_pack or a struct literal is never NULL.
            return Ok(Some(false));
        };
        let conds = self.case_conditions(operand.as_deref(), conditions)?;
        for (c, w) in conds.iter().zip(conditions) {
            if !self.duck_foldable(c) {
                return Ok(None);
            }
            let Some(c) = self.baked(c) else {
                return Ok(None);
            };
            match self.eval_closed(&c) {
                None => return Ok(None),
                Some(Some(ScalarVal::I1(true))) => return self.closed_null(&w.result),
                // FALSE or NULL: the next arm.
                Some(_) => {}
            }
        }
        match else_result {
            Some(r) => self.closed_null(r),
            None => Ok(Some(true)),
        }
    }

    /// `e` with each let spelled out and each pure extern call replaced by
    /// the value DuckDB's binder folds it to, which [`eval_closed`] can run:
    /// it runs neither (a UDF in the condition of a SQL function's struct
    /// panicked it; nightly seed 4704064). `None` when a call does not fold
    /// here: it raises, or an argument is not constant.
    fn baked(&self, e: &SExpr) -> Option<SExpr> {
        let mut e = e.clone();
        self.bake(&mut e).then_some(e)
    }

    fn bake(&self, e: &mut SExpr) -> bool {
        let rep = match &e.kind {
            SKind::ExternCall {
                site,
                ext,
                args,
                ret,
                whole,
            } => {
                let Some(spec) = self.udfs.get(*ext as usize).filter(|_| !*whole) else {
                    return false;
                };
                match self.site_bind_fold(*site, *ext as usize, spec, args) {
                    Some(Ok(Some(lanes))) => match lanes.into_iter().nth(*ret as usize).flatten() {
                        Some(v) => scalar_lit(v, e.ty),
                        None => null_of(e.ty),
                    },
                    Some(Ok(None)) => null_of(e.ty),
                    _ => return false,
                }
            }
            SKind::Let(i) => match self.lets.borrow().get(*i as usize).cloned() {
                Some(v) => v,
                None => return false,
            },
            SKind::TreePredict { .. } => return false,
            _ => return e.children_mut().into_iter().all(|c| self.bake(c)),
        };
        *e = rep;
        // A let's value may read calls and lets of its own.
        self.bake(e)
    }
}

/// DuckDB's bind-time NULL folding, over the text of a struct value: which
/// calls its binder makes NULL constants, and so whether a part of the text
/// is foldable. Each call's answer is kept for one
/// [`Binder::struct_folds_to_null`], since a chain of calls asks about each
/// link twice and would otherwise take time exponential in its length.
pub(super) struct DuckNulls<'b, 'a> {
    b: &'b Binder<'a>,
    memo: std::cell::RefCell<std::collections::HashMap<*const SqlExpr, bool>>,
}

impl<'b, 'a> DuckNulls<'b, 'a> {
    pub(super) fn new(b: &'b Binder<'a>) -> Self {
        DuckNulls {
            b,
            memo: Default::default(),
        }
    }

    /// Whether `e` is foldable to DuckDB: no name in its text reads a
    /// column. Each name counts, also where our binder drops it for a value
    /// that cannot matter (`coalesce(2.0, c)` binds as 2.0), and a lateral
    /// alias stands for its expression. Only a call that DuckDB's binder
    /// makes a NULL constant (`c + NULL`) drops its own.
    pub(super) fn closed(&self, e: &SqlExpr) -> Result<bool, PrepareError> {
        use super::resolve::Walk;
        self.b.walk_expr(e, &mut |x| {
            Ok(match x {
                SqlExpr::Identifier(_) | SqlExpr::CompoundIdentifier(_) => match self.b.ref_res(x) {
                    Ok(Resolved::Lane(v)) if self.b.duck_foldable(&v) => Walk::Over,
                    _ => Walk::Stop,
                },
                _ if self.null_call(x) => Walk::Over,
                _ => Walk::Into,
            })
        })
    }

    /// Whether DuckDB's binder makes call `x` a NULL constant. A call with
    /// the default NULL handling becomes one when an argument has type
    /// SQLNULL, or is foldable and evaluates to NULL (measured, 1.5.5:
    /// `levenshtein('a', CASE WHEN c THEN NULL ELSE NULL END)` and
    /// `levenshtein(c, CAST(NULL AS VARCHAR))` fold, while
    /// `levenshtein('a', CAST(CASE WHEN c THEN NULL ELSE NULL END AS
    /// VARCHAR))` stays a call: that cast is VARCHAR and reads `c`; nightly
    /// seed 4848122). Our binder folds all three, so that `x` binds to NULL
    /// here says only that its NULL handling is the default.
    fn null_call(&self, x: &SqlExpr) -> bool {
        let key = x as *const SqlExpr;
        if let Some(&hit) = self.memo.borrow().get(&key) {
            return hit;
        }
        let args: Vec<&SqlExpr> = match x {
            SqlExpr::Function(_) if calls::marker_call(x).is_some() || lets::marker_let(x).is_some() => {
                Vec::new()
            }
            // Their NULL handling is their own: `nullif(NULL, a)` reads `a`
            // (measured: a field read beside it stays VARCHAR).
            SqlExpr::Function(f)
                if ["nullif", "coalesce", "ifnull", "concat", "concat_ws", "greatest", "least"]
                    .iter()
                    .any(|n| f.name.to_string().eq_ignore_ascii_case(n)) =>
            {
                Vec::new()
            }
            SqlExpr::Function(f) => call_args(f),
            SqlExpr::BinaryOp { left, right, .. } => vec![left, right],
            SqlExpr::UnaryOp { expr, .. } => vec![expr],
            _ => Vec::new(),
        };
        let hit = !args.is_empty()
            && self.b.binds_to_null(x)
            && args.into_iter().any(|a| {
                self.sqlnull(a) || (self.b.binds_to_null(a) && matches!(self.closed(a), Ok(true)))
            });
        self.memo.borrow_mut().insert(key, hit);
        hit
    }

    /// Whether DuckDB types `e` SQLNULL: a NULL literal, a bare NULL column
    /// of the level below, a CASE whose every result is one, or a call that
    /// its binder makes a NULL constant.
    fn sqlnull(&self, e: &SqlExpr) -> bool {
        let e = unnest(e);
        if is_null_literal(e) || self.b.null_col_ref(e).is_some() {
            return true;
        }
        if let Some((id, lets)) = lets::marker_let(e) {
            return self.sqlnull(&lets[id].ast);
        }
        if let Some((id, calls)) = calls::marker_call(e) {
            return self.sqlnull(&calls[id].1);
        }
        match e {
            SqlExpr::Case {
                conditions,
                else_result,
                ..
            } => {
                conditions.iter().all(|w| self.sqlnull(&w.result))
                    && else_result.as_deref().is_none_or(|r| self.sqlnull(r))
            }
            _ => self.null_call(e),
        }
    }
}

/// A call's argument expressions, in order.
fn call_args(f: &sqlparser::ast::Function) -> Vec<&SqlExpr> {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
    let FunctionArguments::List(list) = &f.args else {
        return Vec::new();
    };
    list.args
        .iter()
        .filter_map(|a| match a {
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
        })
        .collect()
}
