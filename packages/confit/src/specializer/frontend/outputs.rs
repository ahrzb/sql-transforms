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
    /// This value read only where `guard` holds: a scalar becomes
    /// `CASE WHEN guard THEN value END`, which the engine evaluates only on
    /// the rows the guard selects, as DuckDB evaluates a CASE arm.
    fn guarded(self, guard: &SExpr) -> OutVal {
        match self {
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
