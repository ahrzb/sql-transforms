//! The relational IR: what the frontend produces, what BTA annotates, what
//! lowering consumes. Deliberately skinny — v0 stretch 1 is the
//! scan/filter/project ribbon over the dynamic table; joins and static
//! subtrees grow here at stretch 3 (BTA's home is this layer).

use super::ir::{CmpPred, Lit, Ty};

/// A relational operator tree. `Scan` is always the dynamic table in
/// stretch 1; static relations appear with BTA.
pub enum Rel {
    Scan,
    Filter { input: Box<Rel>, pred: SExpr },
    Project { input: Box<Rel>, exprs: Vec<(String, SExpr)> },
}

/// A bound, typed scalar expression. `nullable` is the frontend's
/// conservative derivation ("cannot prove non-NULL"), which becomes the
/// null-lane shape at lowering: `nullable == false` means no flag register
/// exists at all.
pub struct SExpr {
    pub kind: SKind,
    pub ty: Ty,
    pub nullable: bool,
}

pub enum SKind {
    /// Input column, by index into the dynamic table's schema.
    Col(u32),
    Lit(Lit),
    /// Arithmetic after type promotion: both sides already the same `Ty`
    /// (the frontend inserts `IntToFloat` where DuckDB promotes).
    Arith { op: ArithOp, a: Box<SExpr>, b: Box<SExpr> },
    /// Comparison after promotion; result i1.
    Cmp { pred: CmpPred, a: Box<SExpr>, b: Box<SExpr> },
    /// i64 -> f64 promotion node, inserted by the frontend.
    IntToFloat(Box<SExpr>),
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
