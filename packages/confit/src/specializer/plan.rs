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
#[derive(Clone)]
pub struct StaticTable {
    pub name: String,
    pub cols: Vec<Col>,
}

/// A fitted model set's schema, as given to `prepare` — like [`StaticTable`],
/// this holds no data, only what the binder needs to check a call site.
/// Feature NAMES live here and nowhere else: the frontend resolves a struct
/// call site's fields to lane positions at build, so the lowered `predict`
/// is positional and the IR never carries a name.
#[derive(Clone, Debug)]
pub struct ModelTable {
    pub name: String,
    pub features: Vec<String>,
    pub grid: CompareGrid,
}

/// Which floating-point grid a model set's thresholds were fitted on, and
/// therefore how an INTEGER feature must reach the comparison.
///
/// sklearn narrows a feature array to float32 before walking the tree, so its
/// real test at each node is `float32(x) <= t`. That is a property of the
/// library that packed the model, not of the engine — hence a declared field
/// rather than a hardcoded assumption. `pack_trees` rewrites its thresholds
/// onto the float32 grid (TASK-65) and declares `F32`; a packer for a library
/// that compares in float64 declares `F64` and its integer features reach the
/// compare exactly.
///
/// It belongs to the SET, not to a model within it: the model id is a runtime
/// value (`tree_predict('m', id, ..)`), while the narrowing is a lowering
/// decision made once at build. A per-model flag could only be honoured with
/// a per-row branch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CompareGrid {
    F32,
    F64,
}

/// A struct row column flattened to scalar LANES at build time (TASK-56,
/// pins-waveA/struct-star.json + struct-nested.json): the binder resolves
/// `col.field...` paths and `col.*` expansion to the leaf lanes; no struct
/// value exists at runtime. Leaf lanes sit in `in_cols` AFTER every plain
/// scalar column, named by their dotted path.
#[derive(Debug, Clone)]
pub struct StructCol {
    /// Position among the row MODEL's columns (star order, rename guard).
    pub pos: usize,
    pub name: String,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub node: StructNode,
}

#[derive(Debug, Clone)]
pub enum StructNode {
    /// Scalar leaf: an input lane (index into `in_cols`).
    Leaf(u32),
    /// Unmappable leaf type — exists for resolution, rejects on reference.
    Opaque,
    Nested(Vec<StructField>),
}

impl StructCol {
    pub fn leaf_count(&self) -> usize {
        fn walk(fs: &[StructField]) -> usize {
            fs.iter()
                .map(|f| match &f.node {
                    StructNode::Leaf(_) => 1,
                    StructNode::Opaque => 0,
                    StructNode::Nested(n) => walk(n),
                })
                .sum()
        }
        walk(&self.fields)
    }
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
    /// MEANINGLESS when `batch` is true.
    pub table: usize,
    /// Stage-B self-join: the build side is the BATCH itself (a keyless
    /// batchmap built per call; the whole ON rides in `residual`).
    pub batch: bool,
    pub kind: JoinKind,
    /// Dynamic-side key expressions, one per key column, already promoted
    /// to the map's key types.
    pub keys: Vec<SExpr>,
    /// Static-table columns acting as map keys (indices into `table.cols`),
    /// aligned with `keys`.
    pub key_cols: Vec<u32>,
    /// Per key: true when the ON conjunct was IS NOT DISTINCT FROM. NULL is
    /// then an ordinary key value: the key flattens to TWO map-key lanes,
    /// (validity i1, payload masked to the type default under NULL), on
    /// both the probe and build sides — so NULL joins NULL, one bucket.
    pub key_indf: Vec<bool>,
    /// The remaining columns, in table order — the probe's value lanes.
    /// May be EMPTY (all-key/semi joins — wave-4 pins).
    /// ponytail: all non-key columns become map values even if unreferenced;
    /// prune to referenced columns when codegen makes the width measurable.
    pub val_cols: Vec<u32>,
    /// Non-key ON conjuncts, ANDed: `match = key_hit AND residual` with
    /// 3VL collapse (NULL => non-match). Evaluated HIT-GUARDED — exactly
    /// DuckDB's per-candidate-pair laziness for both-sides residuals;
    /// single-side residuals are restricted to trap-free shapes at bind
    /// (wave-4 pins: DuckDB scan-pushes those, different error timing).
    /// May reference this join's own columns via StaticCol.
    pub residual: Option<SExpr>,
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
    /// i64 -> f64 VIA f32 — `n as f32 as f64`, one rounding, not two.
    /// Only ever wraps a `tree_predict` feature: sklearn narrows an integer
    /// feature array to float32 in a single step, and above 2**53 that is a
    /// different number from `float32(float64(n))` (TASK-77). Below 2**53
    /// it is identical to [`SKind::IntToFloat`], which is what makes it safe
    /// to apply unconditionally.
    IntToFloat32(Box<SExpr>),
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
    /// Score model set `model` (an index into the hoisted model table).
    /// `feats` is already in the model's declared feature order — the
    /// struct call site's names were resolved away at bind.
    ///
    /// Nullability is deliberately asymmetric: the RESULT is NULL exactly
    /// when `id` is, because an unseen group has no model. A NULL FEATURE
    /// does not propagate — the model has a defined answer for missing, and
    /// lowering hands it NaN.
    TreePredict {
        model: u32,
        id: Box<SExpr>,
        feats: Vec<SExpr>,
    },
    /// Regex match against program regex `re` (wave-B; full-match forms
    /// pre-anchored in the ReSpec pattern at bind) -> I1.
    ReMatch {
        re: u32,
        a: Box<SExpr>,
    },
    /// Leftmost-search capture extract; no match -> '' (wave-B pins).
    ReExtract {
        re: u32,
        group: u32,
        a: Box<SExpr>,
    },
    /// Regex replace using the ReSpec's rewrite template.
    ReReplace {
        re: u32,
        global: bool,
        a: Box<SExpr>,
    },
    /// LIKE/ILIKE (negation handled by a Not wrapper at bind).
    Like {
        ci: bool,
        a: Box<SExpr>,
        p: Box<SExpr>,
        esc: Option<Box<SExpr>>,
    },
    /// round/trunc with digits — result type == subject type (I64 or F64);
    /// total, NULL-propagating.
    Round2 {
        trunc: bool,
        a: Box<SExpr>,
        n: Box<SExpr>,
    },
    /// Wave-1 f64 unary math (operand promoted to F64 by the frontend);
    /// NULL-propagating; the trapping ops get safe-masked payloads in
    /// lowering so a NULL row can never fire the domain trap.
    MathF1 {
        op: super::ir::NumOp1,
        a: Box<SExpr>,
    },
    /// Wave-1 f64 binary math: Fpow (total) and Flogb(base, x) (trapping);
    /// wave-3 adds Ffloordiv/Ffloormod/Fnextafter (all total).
    MathF2 {
        op: super::ir::BinOp,
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// replace/translate — Str × Str × Str -> Str, total, NULL-propagating.
    Str3 {
        op: super::ir::StrOp3,
        a: Box<SExpr>,
        b: Box<SExpr>,
        c: Box<SExpr>,
    },
    /// repeat / VARCHAR array_extract — (Str, I64) -> Str, total.
    Str2i {
        op: super::ir::StrOp2i,
        a: Box<SExpr>,
        n: Box<SExpr>,
    },
    /// lpad/rpad — traps only on empty pad + needed growth, so lowering
    /// masks ALL operands under a combined flag (NULL pre-empts the trap).
    Spad {
        left: bool,
        a: Box<SExpr>,
        len: Box<SExpr>,
        pad: Box<SExpr>,
    },
    /// VARCHAR array_slice/list_slice — total, NULL-propagating (a NULL
    /// bound is NULL, never an open bound).
    Sslice {
        a: Box<SExpr>,
        lo: Box<SExpr>,
        hi: Box<SExpr>,
    },
    /// unicode/ord ('' -> -1) / ascii ('' -> 0) — Str -> I64, total.
    Sord {
        empty_zero: bool,
        a: Box<SExpr>,
    },
    /// strip_accents — Str -> Str, total (oracle table + Hangul compose).
    StripAccents(Box<SExpr>),
    /// reverse — Str -> Str, total (ASCII byte path / UAX-29 grapheme
    /// path, pins-waveA).
    Reverse(Box<SExpr>),
    /// TRUE iff join `join` MATCHED this row (key hit AND residual) —
    /// i1, never NULL. The building block for key-column reconstruction
    /// (`r.id` ≡ CASE JoinHit THEN dyn-key ELSE NULL) and semi joins.
    JoinHit(u32),
    /// One lane of a declared-UDF extern call (DRAFT-22 step 2). The k+1
    /// lanes of one syntactic width-k call share `site` — lowering executes
    /// each site once per block (probe-style cache) so the callable runs
    /// once per row. `whole` reads the call-level validity (i1, never
    /// NULL); otherwise the lane is output `ret` (declared type, always
    /// nullable). Width-1 calls are ordinary scalar expressions; width-k
    /// lanes exist only as bare projection items.
    ExternCall {
        site: u32,
        ext: u32,
        args: Vec<SExpr>,
        ret: u32,
        whole: bool,
    },
}

/// Can evaluating this expression trap — overflow, division by zero, a
/// failed CAST, an unknown model id? Conservative in one direction only:
/// anything not on the trap-free allowlist counts as trapping.
///
/// One definition, two callers, and they must not drift apart:
///
/// * the JOIN ON residual rule — a single-side residual has to be trap-free
///   because DuckDB scan-pushes it, so a trap would fire at a different time
///   than ours (`bind_residual`);
/// * Kleene AND/OR lowering, which stays branchless — and therefore
///   evaluates both operands on every row — only when the right operand
///   cannot trap (`FB::kleene`, TASK-75).
///
/// A CASE is trap-free exactly when all of its arms are: lowering branches,
/// so an arm that is not taken is never evaluated (TASK-74).
pub fn may_trap(e: &SExpr) -> bool {
    match &e.kind {
        SKind::Col(_)
        | SKind::StaticCol { .. }
        | SKind::JoinHit(_)
        | SKind::Lit(_)
        | SKind::NullOf => false,
        SKind::Cmp { a, b, .. } | SKind::And { a, b } | SKind::Or { a, b } => {
            may_trap(a) || may_trap(b)
        }
        SKind::Not(a)
        | SKind::IsNull { inner: a, .. }
        | SKind::IntToFloat(a)
        | SKind::IntToFloat32(a) => may_trap(a),
        SKind::Case { arms, default } => {
            arms.iter().any(|(c, r)| may_trap(c) || may_trap(r))
                || default.as_deref().is_some_and(may_trap)
        }
        // Arith overflows, CAST fails, ABS traps on i64::MIN, tree_predict
        // rejects an unknown model id — and anything not named above is
        // simply unclassified. All of it counts as trapping.
        _ => true,
    }
}

/// SQL-level arithmetic. `Div` is DuckDB's `/` — ALWAYS float division
/// (measured: `5/2 = 2.5 DOUBLE`); the frontend promotes both sides to f64.
/// Integer `%` stays integral (measured: `5%2 -> INTEGER`). `IDiv` is
/// DuckDB's `//` / divide(): truncating division on ints, PLAIN division
/// on doubles (NOT floor — measured -7.5//2.0 = -3.75), zero divisor ->
/// NULL on both (the frontend wraps the CASE guard).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    IDiv,
    Rem,
    // Bitwise (wave-5 pins): BIGINT-only, one flat left-assoc parse tier.
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
}
