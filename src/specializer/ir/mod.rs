//! The specializer's imperative IR: SSA over typed scalars, explicit null
//! lane, no allocation vocabulary. This module is the spec; `verify`, `print`
//! and `parse` implement its three mandatory properties (airtight boundary,
//! canonical text, round-trip).
//!
//! # Execution model
//!
//! A [`Program`] describes one specialized function `run(in, out, scratch)`
//! executed once **per input row** (the produce/consume row loop is implicit —
//! there is no row-index value in the IR). The function reads the current
//! input row via `load`/`load.opt`, computes, writes the current output row
//! via `store`/`store.opt`, and finishes with a terminator:
//!
//! * `emit` — the output row is complete (every out column stored exactly
//!   once on this path); advance both cursors.
//! * `skip` — no output row for this input row (filters); nothing may have
//!   been stored on this path.
//! * `trap "msg"` — abort the whole call with a runtime error (CAST failures,
//!   division guards — whatever the lowered dialect defines as an error).
//!
//! No path may store the same column twice, whatever its terminator — a
//! double store is always a lowering bug, so the verifier rejects it even on
//! paths that end in `trap`.
//!
//! `|out| == |in|` therefore holds exactly when `skip` is unreachable, which
//! is statically known — the design doc's "filter is the one allowed
//! divergence and must declare it".
//!
//! # The null lane
//!
//! SSA values are always bare scalars — there is no nullable SSA type, so a
//! nullable value cannot reach an arithmetic instruction *by construction*
//! (the type system has no way to express it; this is how three-valued-logic
//! bugs are made unrepresentable rather than merely checked). Nullability
//! exists only at the edges: a nullable column or static is accessed through
//! the `.opt` instruction forms, which split it into an `i1` validity flag
//! plus a payload of the bare type; NULL logic is then ordinary `i1` algebra
//! (`and`, `or`, `select`), and `store.opt` reassembles flag + payload into a
//! nullable output column. On a false flag the payload is the type's default
//! value (0 / 0.0 / "" / false) — defined, deterministic, never poison.
//!
//! # SSA discipline
//!
//! Strict block-param form: a value may be used only in the block that
//! defines it (after its definition) or received as a block parameter.
//! Cross-block direct uses are illegal — everything flowing between blocks
//! rides on branch arguments. This removes the need for dominance analysis in
//! the verifier; the CFG must also be acyclic in v0 (no operator we lower
//! needs a loop yet; lift when one does, e.g. QuickScorer).
//!
//! # Statics
//!
//! Prepare-time structures are referenced through opaque `@N` handles typed
//! by the program header — `scalar<T>`/`scalar<T?>` read with
//! `sload`/`sload.opt`, and `map(K..) -> (V..)` probed with `probe` (returns
//! a hit flag plus one value per declared column; a LEFT-join miss is just
//! `hit = false`). How a map is materialized (perfect hash, dense array,
//! inline chain) is a backend decision invisible here. Nullable static map
//! columns are flattened to an `i1` column + payload column at prepare time,
//! so map key/value types are bare [`Ty`].
//!
//! # Allocation
//!
//! No instruction allocates on a heap. The only instructions that produce new
//! varlen data — `itos`, `ftos`, `sconcat` — write into the caller's bump
//! arena (`scratch` in the ABI), reset per call. Everything else moves
//! scalars between registers.
//!
//! # Text format
//!
//! Canonical, round-trippable; `parse(print(p)) == p` for every program whose
//! value ids are dense and in definition order (the parser and [`gen`] both
//! produce that form, so the property is closed under round-trip). A program
//! with sparse or out-of-order ids still verifies and still prints; the
//! round-trip then performs a bijective renumbering into canonical form, so
//! structural equality is only meaningful between canonical programs. Names
//! in the text are presentation only — they are not stored in the IR;
//! printing uses canonical `%vN` / `bN` names.
//!
//! ```text
//! program   := static* func
//! comment   := "#" ... end-of-line     // allowed anywhere whitespace is
//! static    := "static" "@" INT ":" static_ty
//! static_ty := "scalar" "<" col_ty ">"
//!            | "map" "(" ty ("," ty)* ")" "->" "(" ty ("," ty)* ")"
//! func      := "fn" IDENT "(" "in" ":" batch "," "out" ":" batch ")" "{" block+ "}"
//! batch     := "batch" "{" [col ("," col)*] "}"
//! col       := (IDENT | STRING) ":" col_ty        // STRING for non-ident names
//! col_ty    := ty ["?"]
//! ty        := "i1" | "i64" | "f64" | "str"
//! block     := IDENT ["(" VALUE ":" ty ("," VALUE ":" ty)* ")"] ":" inst* term
//! inst      := [VALUE ("," VALUE)* "="] OPCODE operands
//!              // only store/store.opt omit the dests; everything else
//!              // defines at least one value
//! term      := "jump" target
//!            | "brif" VALUE "," target "," target
//!            | "emit" | "skip" | "trap" STRING
//! target    := IDENT ["(" VALUE ("," VALUE)* ")"]
//! VALUE     := "%" NAME               // NAME: any run of [A-Za-z0-9_];
//!                                     // the printer only emits %vN
//! ```
//!
//! Instruction surface (`%d` result, `%f` an `i1` flag result):
//!
//! ```text
//! %d = const.i1 true|false     %d = const.i64 INT
//! %d = const.f64 FLOAT         %d = const.str STRING
//! %d = iadd|isub|imul|idiv|irem %a, %b        // i64; idiv/irem trap on 0 or overflow
//! %d = fadd|fsub|fmul|fdiv %a, %b             // f64, IEEE (inf/nan flow, no trap)
//! %d = and|or|xor %a, %b                      // i1
//! %d = not %a                                 // i1
//! %d = icmp.P|fcmp.P|scmp.P %a, %b            // P in eq ne lt le gt ge; -> i1
//! %d = select %c, %a, %b                      // %c: i1; %a, %b same type
//! %d = itof %a                                // i64 -> f64
//! %d = ftoi.trunc|ftoi.round %a               // f64 -> i64; traps out of range
//! %d = itos %a | ftos %a                      // -> str (arena)
//! %f, %d = stoi.opt %a | stof.opt %a          // parse; %f=false on failure
//! %d = sconcat %a, %b                         // str (arena)
//! %d = load in.COL                            // COL not nullable
//! %f, %d = load.opt in.COL                    // COL nullable
//! store out.COL, %v                           // COL not nullable
//! store.opt out.COL, %f, %v                   // COL nullable
//! %hit, %v1, .. = probe @N, %k1, ..           // map static
//! %d = sload @N                               // scalar<T> static
//! %f, %d = sload.opt @N                       // scalar<T?> static
//! ```
//!
//! Numeric-semantics pins deferred to M-interp (settled against the DuckDB
//! oracle there): `ftoi.round` tie behavior, `fcmp` NaN ordering. The IR
//! names the operations; the interpreter pins their edge cases with tests.

pub mod fixtures;
pub mod gen;
pub mod parse;
pub mod print;
pub mod verify;

#[cfg(test)]
mod tests;

/// Bare scalar type of an SSA value. There is deliberately no nullable
/// variant — see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Ty {
    I1,
    I64,
    F64,
    Str,
}

impl Ty {
    pub fn name(self) -> &'static str {
        match self {
            Ty::I1 => "i1",
            Ty::I64 => "i64",
            Ty::F64 => "f64",
            Ty::Str => "str",
        }
    }
}

/// Column type: a bare type plus nullability. Only batch columns and scalar
/// statics carry nullability; SSA values never do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ColTy {
    pub ty: Ty,
    pub nullable: bool,
}

/// A named batch column.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Col {
    pub name: String,
    pub ty: ColTy,
}

/// Type of a prepare-time static structure, addressed as `@N`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StaticTy {
    Scalar(ColTy),
    Map { keys: Vec<Ty>, values: Vec<Ty> },
}

/// SSA value id. Presentation names are not stored; the printer derives
/// `%vN` from the id.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Value(pub u32);

/// Block id = index into `Program::blocks`. Block 0 is the entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockId(pub u32);

/// Literal for `const.*`. `F64` equality is bitwise — so `-0.0 != 0.0`
/// survives a round-trip — EXCEPT that all NaNs compare equal: the text form
/// canonicalizes every NaN payload to the one `nan` token, and equality must
/// match what the text format can distinguish or `parse(print(p)) == p`
/// would fail for non-canonical payloads (found by adversarial fuzzing).
#[derive(Clone, Debug)]
pub enum Lit {
    I1(bool),
    I64(i64),
    F64(f64),
    Str(String),
}

impl PartialEq for Lit {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Lit::I1(a), Lit::I1(b)) => a == b,
            (Lit::I64(a), Lit::I64(b)) => a == b,
            (Lit::F64(a), Lit::F64(b)) => a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
            (Lit::Str(a), Lit::Str(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for Lit {}

impl Lit {
    pub fn ty(&self) -> Ty {
        match self {
            Lit::I1(_) => Ty::I1,
            Lit::I64(_) => Ty::I64,
            Lit::F64(_) => Ty::F64,
            Lit::Str(_) => Ty::Str,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Iadd,
    Isub,
    Imul,
    Idiv,
    Irem,
    Fadd,
    Fsub,
    Fmul,
    Fdiv,
    And,
    Or,
    Xor,
}

impl BinOp {
    /// (operand type, result type). Uniform for all v0 binary ops.
    pub fn sig(self) -> (Ty, Ty) {
        match self {
            BinOp::Iadd | BinOp::Isub | BinOp::Imul | BinOp::Idiv | BinOp::Irem => {
                (Ty::I64, Ty::I64)
            }
            BinOp::Fadd | BinOp::Fsub | BinOp::Fmul | BinOp::Fdiv => (Ty::F64, Ty::F64),
            BinOp::And | BinOp::Or | BinOp::Xor => (Ty::I1, Ty::I1),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BinOp::Iadd => "iadd",
            BinOp::Isub => "isub",
            BinOp::Imul => "imul",
            BinOp::Idiv => "idiv",
            BinOp::Irem => "irem",
            BinOp::Fadd => "fadd",
            BinOp::Fsub => "fsub",
            BinOp::Fmul => "fmul",
            BinOp::Fdiv => "fdiv",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpPred {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpPred {
    pub fn name(self) -> &'static str {
        match self {
            CmpPred::Eq => "eq",
            CmpPred::Ne => "ne",
            CmpPred::Lt => "lt",
            CmpPred::Le => "le",
            CmpPred::Gt => "gt",
            CmpPred::Ge => "ge",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RoundMode {
    Trunc,
    Round,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Inst {
    Const {
        dst: Value,
        lit: Lit,
    },
    Bin {
        op: BinOp,
        dst: Value,
        a: Value,
        b: Value,
    },
    /// `icmp.P` / `fcmp.P` / `scmp.P` — `ty` is the operand type
    /// (I64/F64/Str), result is I1.
    Cmp {
        pred: CmpPred,
        ty: Ty,
        dst: Value,
        a: Value,
        b: Value,
    },
    Not {
        dst: Value,
        a: Value,
    },
    /// Branchless pick; `a`/`b` any one type, `cond` I1.
    Select {
        dst: Value,
        cond: Value,
        a: Value,
        b: Value,
    },
    Itof {
        dst: Value,
        a: Value,
    },
    Ftoi {
        mode: RoundMode,
        dst: Value,
        a: Value,
    },
    Itos {
        dst: Value,
        a: Value,
    },
    Ftos {
        dst: Value,
        a: Value,
    },
    StoiOpt {
        flag: Value,
        dst: Value,
        a: Value,
    },
    StofOpt {
        flag: Value,
        dst: Value,
        a: Value,
    },
    Sconcat {
        dst: Value,
        a: Value,
        b: Value,
    },
    /// Read a NOT NULL input column at the current row.
    Load {
        dst: Value,
        col: u32,
    },
    /// Read a nullable input column: validity flag + payload.
    LoadOpt {
        flag: Value,
        dst: Value,
        col: u32,
    },
    /// Write a NOT NULL output column at the current row.
    Store {
        col: u32,
        val: Value,
    },
    /// Write a nullable output column: flag=false stores NULL.
    StoreOpt {
        col: u32,
        flag: Value,
        val: Value,
    },
    /// Probe a map static: hit flag + one value per declared value column.
    Probe {
        static_id: u32,
        hit: Value,
        dsts: Vec<Value>,
        keys: Vec<Value>,
    },
    /// Read a `scalar<T>` static.
    Sload {
        static_id: u32,
        dst: Value,
    },
    /// Read a `scalar<T?>` static: validity flag + payload.
    SloadOpt {
        static_id: u32,
        flag: Value,
        dst: Value,
    },
}

#[derive(Clone, PartialEq, Debug)]
pub enum Term {
    Jump {
        to: BlockId,
        args: Vec<Value>,
    },
    Brif {
        cond: Value,
        then_to: BlockId,
        then_args: Vec<Value>,
        else_to: BlockId,
        else_args: Vec<Value>,
    },
    /// Output row complete; advance both cursors.
    Emit,
    /// Drop this input row; nothing may have been stored on this path.
    Skip,
    /// Abort the whole call with a runtime error.
    Trap {
        msg: String,
    },
}

#[derive(Clone, PartialEq, Debug)]
pub struct Block {
    pub params: Vec<(Value, Ty)>,
    pub insts: Vec<Inst>,
    pub term: Term,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Program {
    pub statics: Vec<StaticTy>,
    pub name: String,
    pub in_cols: Vec<Col>,
    pub out_cols: Vec<Col>,
    pub blocks: Vec<Block>,
}

impl Inst {
    /// Values this instruction defines, in definition order.
    pub fn dsts(&self) -> Vec<Value> {
        match self {
            Inst::Const { dst, .. }
            | Inst::Bin { dst, .. }
            | Inst::Cmp { dst, .. }
            | Inst::Not { dst, .. }
            | Inst::Select { dst, .. }
            | Inst::Itof { dst, .. }
            | Inst::Ftoi { dst, .. }
            | Inst::Itos { dst, .. }
            | Inst::Ftos { dst, .. }
            | Inst::Sconcat { dst, .. }
            | Inst::Load { dst, .. }
            | Inst::Sload { dst, .. } => vec![*dst],
            Inst::StoiOpt { flag, dst, .. }
            | Inst::StofOpt { flag, dst, .. }
            | Inst::LoadOpt { flag, dst, .. }
            | Inst::SloadOpt { flag, dst, .. } => vec![*flag, *dst],
            Inst::Probe { hit, dsts, .. } => {
                let mut all = vec![*hit];
                all.extend(dsts.iter().copied());
                all
            }
            Inst::Store { .. } | Inst::StoreOpt { .. } => vec![],
        }
    }
}

impl Term {
    /// Successor blocks with their branch arguments.
    pub fn successors(&self) -> Vec<(BlockId, &[Value])> {
        match self {
            Term::Jump { to, args } => vec![(*to, args.as_slice())],
            Term::Brif {
                then_to,
                then_args,
                else_to,
                else_args,
                ..
            } => vec![
                (*then_to, then_args.as_slice()),
                (*else_to, else_args.as_slice()),
            ],
            Term::Emit | Term::Skip | Term::Trap { .. } => vec![],
        }
    }
}

impl Inst {
    /// Rewrite every value reference (defs and uses) through `m`.
    pub fn map_values(&mut self, m: &impl Fn(Value) -> Value) {
        match self {
            Inst::Const { dst, .. } | Inst::Load { dst, .. } | Inst::Sload { dst, .. } => {
                *dst = m(*dst)
            }
            Inst::Itos { dst, a }
            | Inst::Ftos { dst, a }
            | Inst::Itof { dst, a }
            | Inst::Ftoi { dst, a, .. }
            | Inst::Not { dst, a } => {
                *dst = m(*dst);
                *a = m(*a);
            }
            Inst::Bin { dst, a, b, .. }
            | Inst::Cmp { dst, a, b, .. }
            | Inst::Sconcat { dst, a, b } => {
                *dst = m(*dst);
                *a = m(*a);
                *b = m(*b);
            }
            Inst::Select { dst, cond, a, b } => {
                *dst = m(*dst);
                *cond = m(*cond);
                *a = m(*a);
                *b = m(*b);
            }
            Inst::StoiOpt { flag, dst, a } | Inst::StofOpt { flag, dst, a } => {
                *flag = m(*flag);
                *dst = m(*dst);
                *a = m(*a);
            }
            Inst::LoadOpt { flag, dst, .. } | Inst::SloadOpt { flag, dst, .. } => {
                *flag = m(*flag);
                *dst = m(*dst);
            }
            Inst::Store { val, .. } => *val = m(*val),
            Inst::StoreOpt { flag, val, .. } => {
                *flag = m(*flag);
                *val = m(*val);
            }
            Inst::Probe {
                hit, dsts, keys, ..
            } => {
                *hit = m(*hit);
                for d in dsts {
                    *d = m(*d);
                }
                for k in keys {
                    *k = m(*k);
                }
            }
        }
    }
}

impl Term {
    /// Rewrite every value reference through `m`.
    pub fn map_values(&mut self, m: &impl Fn(Value) -> Value) {
        match self {
            Term::Jump { args, .. } => {
                for a in args {
                    *a = m(*a);
                }
            }
            Term::Brif {
                cond,
                then_args,
                else_args,
                ..
            } => {
                *cond = m(*cond);
                for a in then_args.iter_mut().chain(else_args.iter_mut()) {
                    *a = m(*a);
                }
            }
            Term::Emit | Term::Skip | Term::Trap { .. } => {}
        }
    }
}

/// Renumber value ids into canonical form: dense, in definition order as the
/// text format reads (block params, then instruction defs, per block in
/// order). A bijective rename — semantics and verification are unaffected —
/// after which `parse(print(p)) == p` holds exactly. Lowering runs this so
/// every prepared program is canonical even when block-splitting minted ids
/// out of text order.
pub fn canonicalize(p: &mut Program) {
    let mut map: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut next = 0u32;
    for b in &p.blocks {
        for (v, _) in &b.params {
            map.entry(v.0).or_insert_with(|| {
                let n = next;
                next += 1;
                n
            });
        }
        for inst in &b.insts {
            for d in inst.dsts() {
                map.entry(d.0).or_insert_with(|| {
                    let n = next;
                    next += 1;
                    n
                });
            }
        }
    }
    let f = |v: Value| Value(map[&v.0]);
    for b in &mut p.blocks {
        for (v, _) in &mut b.params {
            *v = f(*v);
        }
        for inst in &mut b.insts {
            inst.map_values(&f);
        }
        b.term.map_values(&f);
    }
}

/// Builds programs with dense, definition-ordered value ids — the canonical
/// form for which `parse(print(p)) == p` holds exactly. Lowering (M-lower)
/// and the fuzz generator both construct through this.
pub struct Builder {
    next: u32,
}

impl Builder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Builder {
        Builder { next: 0 }
    }

    pub fn fresh(&mut self) -> Value {
        let v = Value(self.next);
        self.next += 1;
        v
    }
}
