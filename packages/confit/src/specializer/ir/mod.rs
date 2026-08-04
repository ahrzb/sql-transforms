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
//! program   := static* regex* extern* func
//! comment   := "#" ... end-of-line     // allowed anywhere whitespace is
//! static    := "static" "@" INT ":" static_ty
//! extern    := "extern" "@" INT ":" STRING "(" [ty ("," ty)*] ")" "->" "(" ret ("," ret)* ")"
//! ret       := ty | STRING ":" ty     // named returns: all named or none
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
//! %w, %f1, %v1, .. = ecall @N, %af1, %av1, .. // extern (UDF) call
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
    /// Stage-B (shape='many'): a map whose keys may REPEAT — probed as a
    /// flat row range (ProbeRange -> [start, end), ProbeRead per index).
    /// Zero keys = the keyless one-bucket join (cross/inequality).
    MultiMap { keys: Vec<Ty>, values: Vec<Ty> },
    /// Stage-B self-join: the BATCH as build side, assembled per call by
    /// the executor (always keyless; the ON is the join's residual).
    /// `values` = the batch's columns flattened like multimap values
    /// (nullable -> validity+payload pairs).
    BatchMap { values: Vec<Ty> },
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
    /// f64 `%`, IEEE remainder-of-truncated-division (Rust `%`): sign of the
    /// dividend, `x % 0.0` is NaN. Never traps (measured DuckDB 1.5.5).
    Frem,
    /// f64 pow — TOTAL, pure IEEE (pow(NaN,0)=1, pow(0,-1)=inf; wave-1 pins).
    Fpow,
    /// log(base, x) == log10(x)/log10(base) bit-exactly (wave-1 pins).
    /// Traps: base checked FIRST (zero / negative / base==1 each with their
    /// own DuckDB message), then x (zero / negative).
    Flogb,
    /// SQL fdiv(x, y) = floor(x / y) — TOTAL, ±inf on zero divisor
    /// (wave-3 pins). NOT the `//` operator, which is plain division.
    Ffloordiv,
    /// SQL fmod(x, y) = x − floor(x/y)·y — FLOORED mod, divisor's sign;
    /// NaN on zero or infinite divisor (wave-3 pins). NOT C fmod — that
    /// is `Frem` (SQL `%`/mod on doubles).
    Ffloormod,
    /// C nextafter, bit-exact, TOTAL; x == y returns y (wave-3 pins).
    Fnextafter,
    /// i64 `<<` — DuckDB's guarded shift (wave-5 pins): traps on negative
    /// value (even << 0), then negative count; value 0 short-circuits to 0
    /// BEFORE the count-range check; count >= 64 and value >= 2^(63-count)
    /// each trap with their own DuckDB message.
    Ishl,
    /// i64 `>>` — TOTAL (wave-5 pins): count < 0 or >= 64 gives 0
    /// (silently, even for negative values); else arithmetic shift.
    Ishr,
    /// i64 `&` / `|` / xor() — plain two's-complement, total (wave-5 pins).
    Iand,
    Ior,
    Ixor,
    And,
    Or,
    Xor,
}

impl BinOp {
    /// (operand type, result type). Uniform for all v0 binary ops.
    pub fn sig(self) -> (Ty, Ty) {
        match self {
            BinOp::Iadd
            | BinOp::Isub
            | BinOp::Imul
            | BinOp::Idiv
            | BinOp::Irem
            | BinOp::Ishl
            | BinOp::Ishr
            | BinOp::Iand
            | BinOp::Ior
            | BinOp::Ixor => (Ty::I64, Ty::I64),
            BinOp::Fadd
            | BinOp::Fsub
            | BinOp::Fmul
            | BinOp::Fdiv
            | BinOp::Frem
            | BinOp::Fpow
            | BinOp::Flogb
            | BinOp::Ffloordiv
            | BinOp::Ffloormod
            | BinOp::Fnextafter => (Ty::F64, Ty::F64),
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
            BinOp::Frem => "frem",
            BinOp::Fpow => "fpow",
            BinOp::Flogb => "flogb",
            BinOp::Ffloordiv => "ffloordiv",
            BinOp::Ffloormod => "ffloormod",
            BinOp::Fnextafter => "fnextafter",
            BinOp::Ishl => "ishl",
            BinOp::Ishr => "ishr",
            BinOp::Iand => "iand",
            BinOp::Ior => "ior",
            BinOp::Ixor => "ixor",
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

/// One-operand string ops (Str -> Str). Case mapping is SIMPLE (per-codepoint
/// 1:1) to track DuckDB/utf8proc — see the 2026-07-26 builtin-pins spec.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrOp1 {
    Upper,
    Lower,
    /// strip_accents: oracle-extracted per-codepoint map + Hangul jamo
    /// composition + the measured NUL quirk (wave-3 pins). TOTAL.
    StripAccents,
    /// reverse: DuckDB's two paths — all-ASCII input BYTE-reverses
    /// (splitting CRLF!), anything else reverses UAX-29 EXTENDED grapheme
    /// clusters byte-preserving (pins-waveA/reverse-graphemes.json). TOTAL.
    Reverse,
}

impl StrOp1 {
    pub fn name(self) -> &'static str {
        match self {
            StrOp1::Upper => "supper",
            StrOp1::Lower => "slower",
            StrOp1::StripAccents => "sstrip",
            StrOp1::Reverse => "srev",
        }
    }
}

/// Two-string ops, NULL-strict via lanes. The wave-1 search ops are TOTAL
/// with 1-based CODEPOINT positions and empty-needle-matches-everything;
/// the wave-3 similarity ops are raw UTF-8 BYTE-based (all of them —
/// measured), and `Jaccard`/`Hamming` TRAP (empty inputs / byte-length
/// mismatch, DuckDB messages verbatim).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrOp2 {
    /// instr/strpos/position: 1-based codepoint index, 0 = not found.
    Find,
    Contains,
    Starts,
    Ends,
    /// levenshtein == editdist3: plain byte-level edit distance.
    Levenshtein,
    /// damerau_levenshtein: the UNRESTRICTED DL variant (NOT OSA —
    /// witness ('ca','abc') = 2), transposition cost 1, bytes.
    Damerau,
    /// jaccard: |A∩B|/|A∪B| over single-BYTE sets, duplicates ignored;
    /// traps on an empty string either side.
    Jaccard,
    /// hamming == mismatches: byte-wise; traps on byte-length mismatch
    /// and on ANY empty input (('','') is an error, not 0).
    Hamming,
    /// GLOB: byte-level matcher (wave-5 pins) — `?` is one BYTE, classes
    /// are byte-sets with `!` negation, `\` escapes outside classes only;
    /// malformed patterns match nothing, never error.
    Glob,
}

impl StrOp2 {
    pub fn result_ty(self) -> Ty {
        match self {
            StrOp2::Find | StrOp2::Levenshtein | StrOp2::Damerau | StrOp2::Hamming => Ty::I64,
            StrOp2::Jaccard => Ty::F64,
            _ => Ty::I1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            StrOp2::Find => "sfind",
            StrOp2::Contains => "scontains",
            StrOp2::Starts => "sstarts",
            StrOp2::Ends => "sends",
            StrOp2::Levenshtein => "slevenshtein",
            StrOp2::Damerau => "sdamerau",
            StrOp2::Jaccard => "sjaccard",
            StrOp2::Hamming => "shamming",
            StrOp2::Glob => "sglob",
        }
    }
}

/// Three-string ops (Str × Str × Str -> Str), TOTAL, NULL-strict via lanes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrOp3 {
    /// replace(s, from, to): leftmost non-overlapping single pass, output
    /// never rescanned; empty needle is a strict no-op (wave-3 pins).
    Replace,
    /// translate(s, from, to): per-CODEPOINT map; from-chars beyond |to|
    /// are deleted; duplicate in `from` -> FIRST wins (wave-3 pins).
    Translate,
}

impl StrOp3 {
    pub fn name(self) -> &'static str {
        match self {
            StrOp3::Replace => "sreplace",
            StrOp3::Translate => "stranslate",
        }
    }
}

/// (Str × I64 -> Str) ops, TOTAL, NULL-strict via lanes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrOp2i {
    /// repeat(s, n): n <= 0 -> '' silently (wave-3 pins).
    Repeat,
    /// array_extract/list_extract/s[i] on VARCHAR: 1-based CODEPOINT,
    /// negative from the end (len+1+i), out-of-range/0 -> '' — NOT NULL
    /// (the LIST overload differs; wave-3 pins).
    Extract,
}

impl StrOp2i {
    pub fn name(self) -> &'static str {
        match self {
            StrOp2i::Repeat => "srepeat",
            StrOp2i::Extract => "sextract",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrimSide {
    Both,
    Lead,
    Trail,
}

impl TrimSide {
    pub fn name(self) -> &'static str {
        match self {
            TrimSide::Both => "both",
            TrimSide::Lead => "lead",
            TrimSide::Trail => "trail",
        }
    }
}

/// One-operand numeric ops. `Iabs` traps on i64::MIN (DuckDB: Out of Range);
/// `Fabs` clears the sign bit (abs(-0.0) = +0.0); `Fround` is half away from
/// zero (Rust `f64::round`), total on NaN/inf/huge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NumOp1 {
    Iabs,
    Fabs,
    Fround,
    // Wave-1 math unaries (pins: docs/superpowers/specs/2026-07-26-wave1-
    // builtin-pins.md). Ln/Log2/Log10 trap on x <= 0; Fsqrt traps on
    // negatives; Fsin/Fcos/Ftan trap on +-inf (NaN passes through
    // bit-exactly); Fexp/Fcbrt/Ffloor/Fceil/Ftrunc are total.
    Ln,
    Log2,
    Log10,
    Fexp,
    Fsqrt,
    Fcbrt,
    Fsin,
    Fcos,
    Ftan,
    Ffloor,
    Fceil,
    Ftrunc,
}

impl NumOp1 {
    /// Operand type == result type for all v0 numeric unaries.
    pub fn sig(self) -> Ty {
        match self {
            NumOp1::Iabs => Ty::I64,
            _ => Ty::F64,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            NumOp1::Iabs => "iabs",
            NumOp1::Fabs => "fabs",
            NumOp1::Fround => "fround",
            NumOp1::Ln => "ln",
            NumOp1::Log2 => "log2",
            NumOp1::Log10 => "log10",
            NumOp1::Fexp => "fexp",
            NumOp1::Fsqrt => "fsqrt",
            NumOp1::Fcbrt => "fcbrt",
            NumOp1::Fsin => "fsin",
            NumOp1::Fcos => "fcos",
            NumOp1::Ftan => "ftan",
            NumOp1::Ffloor => "ffloor",
            NumOp1::Fceil => "fceil",
            NumOp1::Ftrunc => "ftrunc",
        }
    }
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
    /// Wave-1 string search / wave-3 similarity — `a` is the haystack
    /// (resp. first argument), `b` the needle (second argument).
    Str2 {
        op: StrOp2,
        dst: Value,
        a: Value,
        b: Value,
    },
    /// `sreplace` / `stranslate` — replace(a, b, c) / translate(a, b, c).
    Str3 {
        op: StrOp3,
        dst: Value,
        a: Value,
        b: Value,
        c: Value,
    },
    /// `srepeat` / `sextract` — (a: Str, n: I64) -> Str.
    Str2i {
        op: StrOp2i,
        dst: Value,
        a: Value,
        n: Value,
    },
    /// `spad.left` / `spad.right` — lpad/rpad(a, len, pad): len counts
    /// CODEPOINTS; truncation keeps the FIRST len codepoints for BOTH
    /// sides; TRAPS ("Insufficient padding in LPAD/RPAD.") only when
    /// pad is empty AND growth is needed (data-dependent; wave-3 pins).
    Spad {
        left: bool,
        dst: Value,
        a: Value,
        len: Value,
        pad: Value,
    },
    /// `sslice` — array_slice/list_slice/s[a:b] on VARCHAR: 1-based,
    /// both-ends-INCLUSIVE codepoint slice, negative from-end, clamping,
    /// out-of-range/reversed -> ''. TOTAL (wave-3 pins).
    Sslice {
        dst: Value,
        a: Value,
        lo: Value,
        hi: Value,
    },
    /// `sord` / `sord.ascii` — first codepoint as i64; '' -> -1 (unicode/
    /// ord) or 0 (ascii — the measured sole divergence). TOTAL.
    Sord {
        empty_zero: bool,
        dst: Value,
        a: Value,
    },
    /// String length: codepoints (`slenc`) or UTF-8 bytes (`slenb`).
    SLen {
        bytes: bool,
        dst: Value,
        a: Value,
    },
    /// LIKE/ILIKE: `a LIKE p [ESCAPE esc]`. Byte-based matcher with
    /// codepoint `_`; TRAPS on dangling-escape (data-dependent) and on a
    /// multi-byte ESCAPE operand. `ci` folds both sides with the measured
    /// simple casemap first (ILIKE's generic path).
    Slike {
        ci: bool,
        dst: Value,
        a: Value,
        p: Value,
        esc: Option<Value>,
    },
    /// round/trunc with digits on f64 — DuckDB's scale-then-round with the
    /// oracle-extracted pow10 table; TOTAL (non-finite fallbacks differ
    /// between round and trunc by measurement). `n` is an i64 register.
    Round2f {
        trunc: bool,
        dst: Value,
        a: Value,
        n: Value,
    },
    /// Integer round/trunc with digits: identity for n >= 0, WRAPPING
    /// half-add at i64 for round with n < 0 (measured — never traps).
    Round2i {
        trunc: bool,
        dst: Value,
        a: Value,
        n: Value,
    },
    /// `supper` / `slower` — Str -> Str, simple case mapping.
    Str1 {
        op: StrOp1,
        dst: Value,
        a: Value,
    },
    /// `strim.side a, chars` — remove characters in the set `chars` from the
    /// given side(s) of `a`. Both operands Str; the empty set is a no-op.
    Strim {
        side: TrimSide,
        dst: Value,
        a: Value,
        chars: Value,
    },
    /// `ssubstr a, start, len` / `ssubstr.rest a, start` — codepoint-window
    /// substring with DuckDB's virtual-position arithmetic: negative start
    /// counts from the end, start <= 0 consumes length before character 1,
    /// a negative length slices BACKWARDS from the resolved start, and
    /// offsets/lengths outside ±2^32 trap. `len: None` is the 2-arg SQL
    /// form ("rest of the string"), which never length-traps — that is why
    /// it is not a sentinel value.
    Ssubstr {
        dst: Value,
        a: Value,
        start: Value,
        len: Option<Value>,
    },
    /// `iabs` / `fabs` / `fround` — numeric unaries, operand ty == result ty.
    Num1 {
        op: NumOp1,
        dst: Value,
        a: Value,
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
    /// Probe a MultiMap static: the flat row range matching `keys`
    /// (`[start, end)`, both I64; empty range on a miss). Zero keys =
    /// the whole table (keyless join). TOTAL.
    ProbeRange {
        static_id: u32,
        start: Value,
        end: Value,
        keys: Vec<Value>,
    },
    /// Read row `idx` of a MultiMap's flat value store — one dst per
    /// declared value lane. `idx` MUST come from a ProbeRange of the same
    /// static (verifier-checked bounds are the range's; out-of-range is a
    /// program bug, trapped at run).
    ProbeRead {
        static_id: u32,
        idx: Value,
        dsts: Vec<Value>,
    },
    /// Call opaque extern (UDF) `ext` — DRAFT-22 step 2. Everything at this
    /// boundary is nullable: `args` are a (validity i1, payload) pair per
    /// declared param (payload unread under a false flag); `dsts` are one
    /// whole-call validity i1 (false iff the callable returned NULL — at
    /// the width-k output boundary that is the NULL-list case, distinct
    /// from a list of NULLs) followed by a (validity i1, payload) pair per
    /// declared return. Component flags are all false when the whole flag
    /// is. Traps on a raised exception or a result violating the declared
    /// returns.
    ExternCall {
        ext: u32,
        dsts: Vec<Value>,
        args: Vec<Value>,
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
    /// Match `a` against program regex `re` (full-match forms are
    /// pre-anchored in the ReSpec pattern at bind) -> I1. TOTAL.
    ReMatch {
        re: u32,
        dst: Value,
        a: Value,
    },
    /// Leftmost-search extract of capture `group` (0 = whole match); no
    /// match / non-participating group -> '' (never NULL; wave-B pins).
    ReExtract {
        re: u32,
        group: u32,
        dst: Value,
        a: Value,
    },
    /// First-match (or `global`) replace using the ReSpec's rewrite
    /// template. TOTAL — invalid-rewrite quirks were resolved at bind.
    ReReplace {
        re: u32,
        global: bool,
        dst: Value,
        a: Value,
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
    /// Stage-B: emit the completed output row AND continue at `to` (the
    /// multiplicity loop's back-edge). The stored-output state resets —
    /// blocks after an EmitTo store the NEXT output row.
    EmitTo {
        to: BlockId,
        args: Vec<Value>,
    },
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
    /// Prepare-time-compiled regexes (wave-B): patterns already translated
    /// to rust-regex syntax (retrans.rs), full-match forms pre-anchored.
    pub regexes: Vec<ReSpec>,
    /// Declared opaque extern (UDF) signatures, addressed by `ecall @N`.
    /// The implementations arrive at compile, one per entry, name-checked.
    pub externs: Vec<ExternSpec>,
    pub name: String,
    pub in_cols: Vec<Col>,
    pub out_cols: Vec<Col>,
    pub blocks: Vec<Block>,
}

/// One declared extern: a scalar UDF in Confit's type vocabulary. Every
/// param and return is nullable by contract (DRAFT-22), so nullability is
/// not spelled here — the `ecall` operand layout carries the flags.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ExternSpec {
    pub name: String,
    pub params: Vec<Ty>,
    pub rets: Vec<Ty>,
    /// Declared output field names, parallel to `rets` (TASK-63): either
    /// empty (unnamed — field access over the call refuses by name) or
    /// exactly one name per return.
    pub ret_names: Vec<String>,
}

/// One entry of [`Program::regexes`]; `rewrite` is a rust replacement
/// template, present only for replace ops.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ReSpec {
    pub pattern: String,
    pub ci: bool,
    pub dotall: bool,
    pub rewrite: Option<String>,
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
            | Inst::Str1 { dst, .. }
            | Inst::Str2 { dst, .. }
            | Inst::Str3 { dst, .. }
            | Inst::Str2i { dst, .. }
            | Inst::Spad { dst, .. }
            | Inst::Sslice { dst, .. }
            | Inst::Sord { dst, .. }
            | Inst::SLen { dst, .. }
            | Inst::Round2f { dst, .. }
            | Inst::Round2i { dst, .. }
            | Inst::Slike { dst, .. }
            | Inst::Strim { dst, .. }
            | Inst::Ssubstr { dst, .. }
            | Inst::Num1 { dst, .. }
            | Inst::Load { dst, .. }
            | Inst::ReMatch { dst, .. }
            | Inst::ReExtract { dst, .. }
            | Inst::ReReplace { dst, .. }
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
            Inst::ExternCall { dsts, .. } => dsts.clone(),
            Inst::ProbeRange { start, end, .. } => vec![*start, *end],
            Inst::ProbeRead { dsts, .. } => dsts.clone(),
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
            Term::EmitTo { to, args } => vec![(*to, args.as_slice())],
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
            | Inst::Str1 { dst, a, .. }
            | Inst::SLen { dst, a, .. }
            | Inst::Sord { dst, a, .. }
            | Inst::Num1 { dst, a, .. }
            | Inst::ReMatch { dst, a, .. }
            | Inst::ReExtract { dst, a, .. }
            | Inst::ReReplace { dst, a, .. }
            | Inst::Not { dst, a } => {
                *dst = m(*dst);
                *a = m(*a);
            }
            Inst::Bin { dst, a, b, .. }
            | Inst::Cmp { dst, a, b, .. }
            | Inst::Strim {
                dst, a, chars: b, ..
            }
            | Inst::Str2 { dst, a, b, .. }
            | Inst::Str2i { dst, a, n: b, .. }
            | Inst::Round2f { dst, a, n: b, .. }
            | Inst::Round2i { dst, a, n: b, .. }
            | Inst::Sconcat { dst, a, b } => {
                *dst = m(*dst);
                *a = m(*a);
                *b = m(*b);
            }
            Inst::Str3 { dst, a, b, c, .. }
            | Inst::Spad {
                dst,
                a,
                len: b,
                pad: c,
                ..
            }
            | Inst::Sslice {
                dst,
                a,
                lo: b,
                hi: c,
            } => {
                *dst = m(*dst);
                *a = m(*a);
                *b = m(*b);
                *c = m(*c);
            }
            Inst::Slike { dst, a, p, esc, .. } => {
                *dst = m(*dst);
                *a = m(*a);
                *p = m(*p);
                if let Some(e) = esc {
                    *e = m(*e);
                }
            }
            Inst::Ssubstr { dst, a, start, len } => {
                *dst = m(*dst);
                *a = m(*a);
                *start = m(*start);
                if let Some(len) = len {
                    *len = m(*len);
                }
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
            Inst::ProbeRange {
                start, end, keys, ..
            } => {
                *start = m(*start);
                *end = m(*end);
                for k in keys {
                    *k = m(*k);
                }
            }
            Inst::ProbeRead { idx, dsts, .. } => {
                *idx = m(*idx);
                for d in dsts {
                    *d = m(*d);
                }
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
            Inst::ExternCall { dsts, args, .. } => {
                for d in dsts {
                    *d = m(*d);
                }
                for a in args {
                    *a = m(*a);
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
            Term::EmitTo { args, .. } => {
                for a in args.iter_mut() {
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
