//! The relational IR: what the frontend produces, what BTA annotates, what
//! lowering consumes. Deliberately skinny — the v0 shape is the
//! scan/filter/project ribbon over the dynamic table; joins and static
//! subtrees grow here at the BTA stretch.

use super::ir::{CmpPred, Col, Lit, TrimSide, Ty};

/// A relational operator tree over the dynamic table. Joins to static
/// tables are not tree nodes: the v0 shape is rigid
/// (project(filter?(join*(scan)))), so the frontend returns them as an
/// ordered [`JoinSpec`] list instead — the tree would only restate the
/// vec's order.
pub enum Rel {
    Scan,
    Filter {
        input: Box<Rel>,
        pred: SExpr,
    },
    Project {
        input: Box<Rel>,
        exprs: Vec<(String, SExpr)>,
    },
}

/// A static (prepare-time-known) table's schema, as given to `prepare`.
/// Value-column nullability is deliberately ignored here: arrow schemas
/// default to nullable, so the real check — no NULL in a value column —
/// happens against the data at materialization.
pub struct StaticTable {
    pub name: String,
    pub cols: Vec<Col>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum JoinKind {
    Inner,
    Left,
}

/// One equi-join to a static table, in FROM-clause order. Join `i` probes
/// map static `@i`; its map layout is `keys[..] -> value columns`, where the
/// key column split comes from the ON clause.
pub struct JoinSpec {
    /// Index into the static-table catalog handed to `prepare`.
    pub table: usize,
    pub kind: JoinKind,
    /// Dynamic-side key expressions, one per key column, already promoted
    /// to the map's key types.
    pub keys: Vec<SExpr>,
    /// Static-table columns acting as map keys (indices into `table.cols`),
    /// aligned with `keys`.
    pub key_cols: Vec<u32>,
    /// The remaining columns, in table order — the probe's value lanes.
    /// ponytail: all non-key columns become map values even if unreferenced;
    /// prune to referenced columns when codegen makes the width measurable.
    pub val_cols: Vec<u32>,
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
    /// Value column `col` (index into the join's `val_cols`) of join `join`.
    /// Lowered as a lane of that join's probe: non-nullable under INNER
    /// (misses were already skipped), hit-flagged under LEFT.
    StaticCol {
        join: u32,
        col: u32,
    },
    Lit(Lit),
    /// Typed NULL constant (`ty` is on the SExpr): flag=false, payload
    /// default. Produced where context gives the bare NULL literal a type.
    NullOf,
    /// Arithmetic after type promotion: both sides already the same `Ty`
    /// (the frontend inserts `IntToFloat` where DuckDB promotes).
    Arith {
        op: ArithOp,
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// Comparison after promotion; result i1, NULL-propagating.
    Cmp {
        pred: CmpPred,
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// i64 -> f64 promotion node, inserted by the frontend.
    IntToFloat(Box<SExpr>),
    /// 3VL NOT: value negates, NULL stays NULL.
    Not(Box<SExpr>),
    /// Kleene AND/OR over i1 operands.
    And {
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    Or {
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// IS NULL / IS NOT NULL — result i1, never NULL.
    IsNull {
        negated: bool,
        inner: Box<SExpr>,
    },
    /// Searched CASE (the simple form is desugared to `operand = value`
    /// conditions at bind). First TRUE condition wins; NULL conditions do
    /// not match; missing ELSE yields NULL.
    Case {
        arms: Vec<(SExpr, SExpr)>,
        default: Option<Box<SExpr>>,
    },
    /// CAST / TRY_CAST; source is `inner.ty`, target is the SExpr's `ty`.
    /// CAST traps on conversion failure (NULL input never traps); TRY_CAST
    /// yields NULL instead.
    Cast {
        inner: Box<SExpr>,
        trying: bool,
    },
    /// UPPER / LOWER — Str -> Str, NULL-propagating, simple case mapping.
    StrCase {
        upper: bool,
        a: Box<SExpr>,
    },
    /// trim/ltrim/rtrim and all TRIM(...) forms. `chars` is always present:
    /// the 1-arg SQL form gets a `' '` literal (DuckDB trims ONLY spaces).
    Trim {
        side: TrimSide,
        a: Box<SExpr>,
        chars: Box<SExpr>,
    },
    /// substr/substring. `len: None` is the 2-arg form ("rest of the
    /// string") — kept distinct because DuckDB range-guards an explicit
    /// length but never a missing one. All operands NULL-propagate.
    Substr {
        a: Box<SExpr>,
        start: Box<SExpr>,
        len: Option<Box<SExpr>>,
    },
    /// ABS — I64 or F64; result type = operand type. Traps on i64::MIN.
    Abs(Box<SExpr>),
    /// ROUND(x) on F64, half away from zero. Integer round is identity and
    /// never builds a node.
    Round(Box<SExpr>),
    /// String concatenation: `||` (always concat in DuckDB, any operands,
    /// NULL-propagating) and the NULL-skipping CONCAT() after its per-arg
    /// desugar. Both operands are Str by construction.
    Concat {
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// Wave-1 string search (haystack, needle) — total, NULL-propagating.
    Str2 {
        op: super::ir::StrOp2,
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// String length: codepoints (length) or UTF-8 bytes (strlen).
    SLen {
        bytes: bool,
        a: Box<SExpr>,
    },
    /// Wave-1 f64 unary math (operand promoted to F64 by the frontend);
    /// NULL-propagating; the trapping ops get safe-masked payloads in
    /// lowering so a NULL row can never fire the domain trap.
    MathF1 {
        op: super::ir::NumOp1,
        a: Box<SExpr>,
    },
    /// Wave-1 f64 binary math: Fpow (total) and Flogb(base, x) (trapping).
    MathF2 {
        op: super::ir::BinOp,
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
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
