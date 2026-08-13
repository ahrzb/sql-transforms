//! Plan nodes and type derivation.
//!
//! Single source of truth for types: only the leaves ([`Expr::Col`],
//! [`Expr::Lit`]) and [`Expr::Cast`] CARRY a type; every interior node
//! DERIVES its type through [`Expr::ty`], the same function at bind time
//! and verify time — a stored result type that could drift from the rule
//! that produced it is the `duck_width` side-channel the lattice spec
//! rejected, so it does not exist here.
//!
//! Derivation rules are DuckDB's, as measured (pins-dialect/); combinations
//! not yet pinned refuse by name. In particular decimal arithmetic result
//! scales are lattice-spec phase-5 territory and refuse until measured —
//! decimal COMPARISON and CASE-unification of equal decimal types are fine.
//!
//! Relations are multisets (design D4): no node here can observe input
//! order. Order-sensitive nodes (Window with its mandatory total ORDER and
//! explicit frame, per D3) arrive in phase 2 carrying [`SortKey`], whose
//! `nulls_first` is mandatory — the field exists now so no order-carrying
//! node can ever be added without it.

use super::ty::DTy;
use super::{unsup, DialectError};

/// What a plan binds against: table schemas, nothing else. Nullability is
/// a column fact (expressions don't track it in v0 — named coarseness in
/// the module doc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Catalog {
    pub tables: Vec<Table>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub name: String,
    pub cols: Vec<ColDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColDef {
    pub name: String,
    pub ty: DTy,
    pub nullable: bool,
}

impl Catalog {
    /// DuckDB identifier semantics, pinned in the specializer frontend:
    /// case-insensitive match, spelling preserved.
    pub fn table(&self, name: &str) -> Option<&Table> {
        self.tables
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    /// `/` — float division on integers too (pinned: typeof(1/2) = DOUBLE).
    FDiv,
    /// `//` — integer division (pinned: typeof(1//2) = INTEGER; truncating).
    IDiv,
    Rem,
    Concat,
    And,
    Or,
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

impl BinOp {
    pub fn name(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::FDiv => "fdiv",
            BinOp::IDiv => "idiv",
            BinOp::Rem => "rem",
            BinOp::Concat => "concat",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Eq => "eq",
            BinOp::Neq => "neq",
            BinOp::Lt => "lt",
            BinOp::Lte => "lte",
            BinOp::Gt => "gt",
            BinOp::Gte => "gte",
        }
    }

    pub fn parse(s: &str) -> Option<BinOp> {
        Some(match s {
            "add" => BinOp::Add,
            "sub" => BinOp::Sub,
            "mul" => BinOp::Mul,
            "fdiv" => BinOp::FDiv,
            "idiv" => BinOp::IDiv,
            "rem" => BinOp::Rem,
            "concat" => BinOp::Concat,
            "and" => BinOp::And,
            "or" => BinOp::Or,
            "eq" => BinOp::Eq,
            "neq" => BinOp::Neq,
            "lt" => BinOp::Lt,
            "lte" => BinOp::Lte,
            "gt" => BinOp::Gt,
            "gte" => BinOp::Gte,
            _ => return None,
        })
    }

    pub fn is_comparison(self) -> bool {
        matches!(
            self,
            BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Lte | BinOp::Gt | BinOp::Gte
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

/// The scalar functions the plan has bought — each with a pinned
/// signature and, per printer, either a measured-identical spelling, a
/// forced one, or a named refusal (pins-dialect/scalar-functions probes,
/// 2026-08-13). Growth is corpus-first: a function enters with its
/// DuckDB semantics measured (NULL propagation included) and each
/// printer buys it separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFn {
    Upper,
    Lower,
    Reverse,
    Length,
    BitLength,
    Trim,
    Ltrim,
    Rtrim,
    Contains,
    StartsWith,
    Instr,
    Replace,
    Repeat,
    Translate,
    Concat,
    ConcatWs,
    /// BYTE-based edit distances in DuckDB (measured: levenshtein('é','e')
    /// = 2); Spark counts characters — Spark printer refuses these two.
    Levenshtein,
    DamerauLevenshtein,
}

impl ScalarFn {
    /// The canonical (DuckDB) spelling, also the plan-text head.
    pub fn name(self) -> &'static str {
        match self {
            ScalarFn::Upper => "upper",
            ScalarFn::Lower => "lower",
            ScalarFn::Reverse => "reverse",
            ScalarFn::Length => "length",
            ScalarFn::BitLength => "bit_length",
            ScalarFn::Trim => "trim",
            ScalarFn::Ltrim => "ltrim",
            ScalarFn::Rtrim => "rtrim",
            ScalarFn::Contains => "contains",
            ScalarFn::StartsWith => "starts_with",
            ScalarFn::Instr => "instr",
            ScalarFn::Replace => "replace",
            ScalarFn::Repeat => "repeat",
            ScalarFn::Translate => "translate",
            ScalarFn::Concat => "concat",
            ScalarFn::ConcatWs => "concat_ws",
            ScalarFn::Levenshtein => "levenshtein",
            ScalarFn::DamerauLevenshtein => "damerau_levenshtein",
        }
    }

    pub fn parse(s: &str) -> Option<ScalarFn> {
        Some(match s {
            "upper" => ScalarFn::Upper,
            "lower" => ScalarFn::Lower,
            "reverse" => ScalarFn::Reverse,
            "length" => ScalarFn::Length,
            "bit_length" => ScalarFn::BitLength,
            "trim" => ScalarFn::Trim,
            "ltrim" => ScalarFn::Ltrim,
            "rtrim" => ScalarFn::Rtrim,
            "contains" => ScalarFn::Contains,
            "starts_with" => ScalarFn::StartsWith,
            "instr" => ScalarFn::Instr,
            "replace" => ScalarFn::Replace,
            "repeat" => ScalarFn::Repeat,
            "translate" => ScalarFn::Translate,
            "concat" => ScalarFn::Concat,
            "concat_ws" => ScalarFn::ConcatWs,
            "levenshtein" => ScalarFn::Levenshtein,
            "damerau_levenshtein" => ScalarFn::DamerauLevenshtein,
            _ => return None,
        })
    }

    /// Check the argument types and derive the return type — the one
    /// signature rule, used at bind and verify time.
    pub fn ret(self, args: &[DTy]) -> Result<DTy, DialectError> {
        use ScalarFn::*;
        let sig_err = || {
            Err(unsup(format!(
                "function signature: {}({})",
                self.name(),
                args.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")
            )))
        };
        let all_str = |ts: &[DTy]| ts.iter().all(|t| matches!(t, DTy::Str));
        match self {
            Upper | Lower | Reverse | Trim | Ltrim | Rtrim => match args {
                [DTy::Str] => Ok(DTy::Str),
                _ => sig_err(),
            },
            Length | BitLength => match args {
                [DTy::Str] => Ok(DTy::I64),
                _ => sig_err(),
            },
            Contains | StartsWith => match args {
                [DTy::Str, DTy::Str] => Ok(DTy::Bool),
                _ => sig_err(),
            },
            Instr | Levenshtein | DamerauLevenshtein => match args {
                [DTy::Str, DTy::Str] => Ok(DTy::I64),
                _ => sig_err(),
            },
            Replace | Translate => match args {
                [DTy::Str, DTy::Str, DTy::Str] => Ok(DTy::Str),
                _ => sig_err(),
            },
            Repeat => match args {
                [DTy::Str, n] if int_rank(n).is_some() => Ok(DTy::Str),
                _ => sig_err(),
            },
            Concat => {
                if !args.is_empty() && all_str(args) {
                    Ok(DTy::Str)
                } else {
                    sig_err()
                }
            }
            ConcatWs => {
                if args.len() >= 2 && all_str(args) {
                    Ok(DTy::Str)
                } else {
                    sig_err()
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    /// A bound input column: ordinal into the input relation's schema, plus
    /// the bound spelling and type. Name and type are checked against the
    /// ordinal by the verifier — carried for rewrite ergonomics and text
    /// legibility, never trusted alone.
    Col {
        ordinal: usize,
        name: String,
        ty: DTy,
    },
    /// A typed literal, carrying its source LEXEME verbatim — printers emit
    /// the lexeme, so a plan never re-formats a number (re-formatting is
    /// where silent value drift lives). For `Str` the lexeme is the
    /// unescaped value; printers add quoting.
    Lit {
        lexeme: String,
        ty: DTy,
    },
    Bin {
        op: BinOp,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    Un {
        op: UnOp,
        e: Box<Expr>,
    },
    /// strict=true is SQL CAST (errors on failure), strict=false TRY_CAST
    /// (NULL on failure) — the design's mandatory explicit failure
    /// semantics (D3).
    Cast {
        strict: bool,
        e: Box<Expr>,
        target: DTy,
    },
    /// Searched CASE. `else_` is optional in SQL (defaults to NULL); the
    /// plan keeps the option explicit rather than synthesizing a NULL
    /// literal it cannot type.
    Case {
        whens: Vec<(Expr, Expr)>,
        else_: Option<Box<Expr>>,
    },
    IsNull {
        negated: bool,
        e: Box<Expr>,
    },
    /// IS [NOT] DISTINCT FROM — null-safe comparison, a distinct node (not
    /// an Eq flag) because printers spell it per dialect (D3).
    IsDistinct {
        negated: bool,
        l: Box<Expr>,
        r: Box<Expr>,
    },
    /// A bought scalar function call — the signature lives on [`ScalarFn`].
    Call {
        func: ScalarFn,
        args: Vec<Expr>,
    },
}

/// A sort key for the order-carrying nodes of phase 2+. Both fields are
/// mandatory — there is no "default order" anywhere in the plan (D3;
/// measured: DuckDB is NULLS LAST both directions, Spark is NULLS FIRST on
/// ASC — pins-dialect/).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortKey {
    pub expr: Expr,
    pub desc: bool,
    pub nulls_first: bool,
}

impl Expr {
    /// Derive this expression's type — DuckDB's rules where pinned, named
    /// refusal where not. The one typing function: binder and verifier
    /// both call it.
    pub fn ty(&self) -> Result<DTy, DialectError> {
        match self {
            Expr::Col { ty, .. } => Ok(ty.clone()),
            Expr::Lit { ty, .. } => Ok(ty.clone()),
            Expr::Cast { target, .. } => Ok(target.clone()),
            Expr::Un { op: UnOp::Not, e } => match e.ty()? {
                DTy::Bool => Ok(DTy::Bool),
                t => Err(unsup(format!(
                    "implicit boolean coercion: NOT over {}",
                    t.name()
                ))),
            },
            Expr::Un { op: UnOp::Neg, e } => {
                let t = e.ty()?;
                if t.is_numeric() {
                    Ok(t)
                } else {
                    Err(unsup(format!(
                        "implicit coercion: unary - over {}",
                        t.name()
                    )))
                }
            }
            Expr::Bin { op, l, r } => derive_bin(*op, &l.ty()?, &r.ty()?),
            Expr::Case { whens, else_ } => {
                if whens.is_empty() {
                    return Err(DialectError::Internal("CASE with no WHEN arms".into()));
                }
                for (c, _) in whens {
                    if c.ty()? != DTy::Bool {
                        return Err(DialectError::Bind(
                            "CASE WHEN condition is not BOOLEAN".into(),
                        ));
                    }
                }
                let mut branches: Vec<DTy> = Vec::new();
                for (_, v) in whens {
                    branches.push(v.ty()?);
                }
                if let Some(e) = else_ {
                    branches.push(e.ty()?);
                }
                let mut it = branches.into_iter();
                let first = it
                    .next()
                    .ok_or(DialectError::Internal("CASE with no branches".into()))?;
                it.try_fold(first, |a, b| unify(&a, &b))
            }
            Expr::IsNull { e, .. } => {
                e.ty()?;
                Ok(DTy::Bool)
            }
            Expr::IsDistinct { l, r, .. } => {
                comparable(&l.ty()?, &r.ty()?)?;
                Ok(DTy::Bool)
            }
            Expr::Call { func, args } => {
                let tys: Vec<DTy> = args.iter().map(|a| a.ty()).collect::<Result<_, _>>()?;
                func.ret(&tys)
            }
        }
    }
}

fn int_rank(t: &DTy) -> Option<u8> {
    Some(match t {
        DTy::I8 => 1,
        DTy::I16 => 2,
        DTy::I32 => 3,
        DTy::I64 => 4,
        DTy::I128 => 5,
        _ => return None,
    })
}

/// Widest of two SIGNED integer types. Unsigned arithmetic is unpinned —
/// refuse rather than guess DuckDB's signed/unsigned promotion.
fn wider_int(a: &DTy, b: &DTy) -> Result<DTy, DialectError> {
    match (int_rank(a), int_rank(b)) {
        (Some(ra), Some(rb)) => Ok(if ra >= rb { a.clone() } else { b.clone() }),
        _ => Err(unsup(format!(
            "integer promotion over {} and {} (unsigned promotion not pinned)",
            a.name(),
            b.name()
        ))),
    }
}

fn derive_bin(op: BinOp, l: &DTy, r: &DTy) -> Result<DTy, DialectError> {
    use BinOp::*;
    match op {
        And | Or => match (l, r) {
            (DTy::Bool, DTy::Bool) => Ok(DTy::Bool),
            // DuckDB coerces (1 AND 1 is valid); the coercion is unpinned.
            _ => Err(unsup(format!(
                "implicit boolean coercion: {} over {} and {}",
                op.name(),
                l.name(),
                r.name()
            ))),
        },
        Eq | Neq | Lt | Lte | Gt | Gte => {
            comparable(l, r)?;
            Ok(DTy::Bool)
        }
        Concat => match (l, r) {
            (DTy::Str, DTy::Str) => Ok(DTy::Str),
            _ => Err(unsup(format!("|| over {} and {}", l.name(), r.name()))),
        },
        FDiv => {
            if matches!(l, DTy::F32) || matches!(r, DTy::F32) {
                // Measured: FLOAT/FLOAT computes at FLOAT in DuckDB, not
                // DOUBLE - refusing beats a silently wrong width.
                return Err(unsup(format!(
                    "/ over {} and {} (f32 division width not lowered)",
                    l.name(),
                    r.name()
                )));
            }
            if l.is_numeric() && r.is_numeric() {
                // Pinned: / is DOUBLE on integers/decimals/doubles
                // (strings-operators.json).
                Ok(DTy::F64)
            } else {
                Err(DialectError::Bind(format!(
                    "/ over {} and {}",
                    l.name(),
                    r.name()
                )))
            }
        }
        IDiv | Rem => wider_int(l, r)
            .map_err(|_| unsup(format!("{} over {} and {}", op.name(), l.name(), r.name()))),
        Add | Sub | Mul => match (l, r) {
            (DTy::F64, o) | (o, DTy::F64) if o.is_numeric() => Ok(DTy::F64),
            (DTy::Dec(..), _) | (_, DTy::Dec(..)) => Err(unsup(format!(
                "decimal arithmetic result scale for {} over {} and {} (lattice-spec phase 5)",
                op.name(),
                l.name(),
                r.name()
            ))),
            (DTy::F32, _) | (_, DTy::F32) => Err(unsup(format!(
                "f32 arithmetic for {} over {} and {}",
                op.name(),
                l.name(),
                r.name()
            ))),
            _ if l.is_integer() && r.is_integer() => wider_int(l, r),
            // DuckDB coerces further (bool, strings, temporals); unpinned.
            _ => Err(unsup(format!(
                "implicit coercion: {} over {} and {}",
                op.name(),
                l.name(),
                r.name()
            ))),
        },
    }
}

/// May these two types meet in a comparison / IS DISTINCT FROM?
/// Only pinned-safe combinations pass. The review-confirmed hazards:
/// integer-vs-DECIMAL casts the integer to the decimal type in DuckDB and
/// ERRORS when it does not fit (value-vs-error against printed engines);
/// F32 mixed comparisons compare at FLOAT in DuckDB but DOUBLE downstream.
fn comparable(l: &DTy, r: &DTy) -> Result<(), DialectError> {
    if l == r {
        return Ok(());
    }
    let both_signed_int = int_rank(l).is_some() && int_rank(r).is_some();
    let f64_vs_numeric = (matches!(l, DTy::F64) && r.is_numeric() && !matches!(r, DTy::F32))
        || (matches!(r, DTy::F64) && l.is_numeric() && !matches!(l, DTy::F32));
    let dec_vs_dec = matches!(l, DTy::Dec(..)) && matches!(r, DTy::Dec(..));
    if both_signed_int || f64_vs_numeric || dec_vs_dec {
        return Ok(());
    }
    Err(unsup(format!(
        "comparison coercion over {} and {} (cross-class comparison semantics unpinned)",
        l.name(),
        r.name()
    )))
}

/// CASE branch unification: equal types, or numeric promotion where pinned.
fn unify(a: &DTy, b: &DTy) -> Result<DTy, DialectError> {
    if a == b {
        return Ok(a.clone());
    }
    match (a, b) {
        (DTy::F64, o) | (o, DTy::F64) if o.is_numeric() => Ok(DTy::F64),
        _ if a.is_integer() && b.is_integer() => wider_int(a, b),
        _ => Err(unsup(format!(
            "CASE branch unification of {} and {}",
            a.name(),
            b.name()
        ))),
    }
}

/// Join kinds (2026-08-13-dialect-join-node-design.md). SEMI/ANTI/ASOF/
/// APPLY/positional refuse at the frontend by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

impl JoinKind {
    pub fn name(self) -> &'static str {
        match self {
            JoinKind::Inner => "inner",
            JoinKind::Left => "left",
            JoinKind::Right => "right",
            JoinKind::Full => "full",
            JoinKind::Cross => "cross",
        }
    }

    pub fn parse(s: &str) -> Option<JoinKind> {
        Some(match s {
            "inner" => JoinKind::Inner,
            "left" => JoinKind::Left,
            "right" => JoinKind::Right,
            "full" => JoinKind::Full,
            "cross" => JoinKind::Cross,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rel {
    /// A base table scan, by bound spelling.
    Scan {
        table: String,
    },
    Filter {
        input: Box<Rel>,
        pred: Expr,
    },
    /// Output columns, in order. Names are NOT required unique — DuckDB
    /// allows duplicate output names (pins-wave5 dup-names).
    Project {
        input: Box<Rel>,
        items: Vec<(String, Expr)>,
    },
    /// Join. Output schema is left ++ right for EVERY kind (USING's merged
    /// column is a binder concern that never reaches the plan). `on` is a
    /// BOOLEAN expression bound over the combined schema; None iff Cross.
    /// Null-safe equality stays explicit through the expression node kind
    /// (Eq vs IsDistinct) — design D3 without a separate key list.
    Join {
        left: Box<Rel>,
        right: Box<Rel>,
        kind: JoinKind,
        on: Option<Expr>,
    },
}

impl Rel {
    /// The relation's output schema against a catalog.
    pub fn schema(&self, cat: &Catalog) -> Result<Vec<(String, DTy)>, DialectError> {
        match self {
            Rel::Scan { table } => {
                let t = cat
                    .table(table)
                    .ok_or_else(|| DialectError::Bind(format!("unknown table: {table}")))?;
                Ok(t.cols
                    .iter()
                    .map(|c| (c.name.clone(), c.ty.clone()))
                    .collect())
            }
            Rel::Filter { input, .. } => input.schema(cat),
            Rel::Project { items, .. } => items
                .iter()
                .map(|(n, e)| Ok((n.clone(), e.ty()?)))
                .collect(),
            Rel::Join { left, right, .. } => {
                let mut s = left.schema(cat)?;
                s.extend(right.schema(cat)?);
                Ok(s)
            }
        }
    }
}
