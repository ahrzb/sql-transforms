//! The relational IR: what the frontend produces, what BTA annotates, what
//! lowering consumes. Deliberately skinny — the v0 shape is the
//! scan/filter/project ribbon over the dynamic table; joins and static
//! subtrees grow here at the BTA stretch.

use super::ir::{CmpPred, Lit, Ty};

/// A relational operator tree. `Scan` is always the dynamic table for now;
/// static relations appear with BTA.
pub enum Rel {
    Scan,
    Filter { input: Box<Rel>, pred: SExpr },
    Project { input: Box<Rel>, exprs: Vec<(String, SExpr)> },
}

/// A bound, typed scalar expression. `nullable` is the frontend's
/// conservative derivation ("cannot prove non-NULL"), and it is a contract
/// with lowering: an expression lowers to a flag lane IFF `nullable` — the
/// out-column nullability, the CASE join shape, and the store form all key
/// off it.
#[derive(Clone)]
pub struct SExpr {
    pub kind: SKind,
    pub ty: Ty,
    pub nullable: bool,
}

#[derive(Clone)]
pub enum SKind {
    /// Input column, by index into the dynamic table's schema.
    Col(u32),
    Lit(Lit),
    /// Typed NULL constant (`ty` is on the SExpr): flag=false, payload
    /// default. Produced where context gives the bare NULL literal a type.
    NullOf,
    /// Arithmetic after type promotion: both sides already the same `Ty`
    /// (the frontend inserts `IntToFloat` where DuckDB promotes).
    Arith { op: ArithOp, a: Box<SExpr>, b: Box<SExpr> },
    /// Comparison after promotion; result i1, NULL-propagating.
    Cmp { pred: CmpPred, a: Box<SExpr>, b: Box<SExpr> },
    /// i64 -> f64 promotion node, inserted by the frontend.
    IntToFloat(Box<SExpr>),
    /// 3VL NOT: value negates, NULL stays NULL.
    Not(Box<SExpr>),
    /// Kleene AND/OR over i1 operands.
    And { a: Box<SExpr>, b: Box<SExpr> },
    Or { a: Box<SExpr>, b: Box<SExpr> },
    /// IS NULL / IS NOT NULL — result i1, never NULL.
    IsNull { negated: bool, inner: Box<SExpr> },
    /// Searched CASE (the simple form is desugared to `operand = value`
    /// conditions at bind). First TRUE condition wins; NULL conditions do
    /// not match; missing ELSE yields NULL.
    Case { arms: Vec<(SExpr, SExpr)>, default: Option<Box<SExpr>> },
    /// CAST / TRY_CAST; source is `inner.ty`, target is the SExpr's `ty`.
    /// CAST traps on conversion failure (NULL input never traps); TRY_CAST
    /// yields NULL instead.
    Cast { inner: Box<SExpr>, trying: bool },
}

/// SQL-level arithmetic. `Div` is DuckDB's `/` — ALWAYS float division
/// (measured: `5/2 = 2.5 DOUBLE`); the frontend promotes both sides to f64.
/// Integer `%` stays integral (measured: `5%2 -> INTEGER`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}
