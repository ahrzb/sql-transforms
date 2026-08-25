//! The relational IR: what the frontend produces, what BTA annotates, what
//! lowering consumes. Deliberately skinny — the v0 shape is the
//! scan/filter/project ribbon over the dynamic table; joins and static
//! subtrees grow here at the BTA stretch.

use super::ir::{ColTy, CmpPred, Col, Lit, TrimSide, Ty};

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
    /// Columns present in the arrow schema whose type this engine does not
    /// serve. Carried rather than dropped: a dropped column makes the binder
    /// say "does not exist" about a column that plainly does, and sends the
    /// reader hunting a typo in a correct query. Unreferenced they cost
    /// nothing; referenced they refuse by name. `(column, arrow type)`.
    pub opaque: Vec<(String, String)>,
    /// The DECLARED column list, in declaration order, as a star must see it
    /// (TASK-125): one entry per schema column. A struct column is ONE
    /// opaque entry here even though its flattened leaves live in `cols` —
    /// a star answers with the column, never with its leaves.
    pub star: Vec<StarCol>,
    /// Struct columns as TREES (TASK-132): resolution walks these; the
    /// leaf lanes interleaved in `cols` keep their dotted names for
    /// DISPLAY ONLY. `StructNode::Leaf` here indexes into `cols`.
    pub structs: Vec<StructCol>,
}

/// One declared column of a [`StaticTable`], as star expansion sees it.
#[derive(Clone)]
pub enum StarCol {
    /// Servable: an index into [`StaticTable::cols`].
    Real(u32),
    /// A struct or non-vocabulary column, by declared name. Under a star it
    /// must be EXCLUDEd or the query refuses naming it — expanding its
    /// leaves, or dropping it, would emit a column set DuckDB never
    /// produces.
    Opaque(String),
}

impl StaticTable {
    /// A table of only servable scalar columns (self-joins against the
    /// batch, tests): every column is Real, in order.
    pub fn all_scalar(name: String, cols: Vec<Col>) -> Self {
        let star = (0..cols.len() as u32).map(StarCol::Real).collect();
        StaticTable {
            name,
            cols,
            opaque: Vec::new(),
            star,
            structs: Vec::new(),
        }
    }

    /// Whether lane `ci` is a struct LEAF: reachable only through its
    /// path, never by name (TASK-132 — a quoted identifier that happens
    /// to spell the dotted display name is a different reference).
    pub fn is_leaf_lane(&self, ci: u32) -> bool {
        fn walk(fs: &[StructField], ci: u32) -> bool {
            fs.iter().any(|f| match &f.node {
                StructNode::Leaf(l) => *l == ci,
                StructNode::Opaque => false,
                StructNode::Nested(n) => walk(n, ci),
            })
        }
        self.structs.iter().any(|sc| walk(&sc.fields, ci))
    }
}

/// Per-lane SEGMENT paths for a lane set with struct trees over it: a
/// plain column's path is its own (whole) name — dots included, a name is
/// not a path — and a leaf lane's is `[struct, field, ...]` from its tree.
/// The DATA paths walk these; nothing splits a name string (TASK-132).
pub fn lane_paths(cols: &[Col], structs: &[StructCol]) -> Vec<Vec<String>> {
    let mut paths: Vec<Vec<String>> = cols.iter().map(|c| vec![c.name.clone()]).collect();
    fn walk(fs: &[StructField], prefix: &mut Vec<String>, paths: &mut [Vec<String>]) {
        for f in fs {
            prefix.push(f.name.clone());
            match &f.node {
                StructNode::Leaf(l) => paths[*l as usize] = prefix.clone(),
                StructNode::Opaque => {}
                StructNode::Nested(n) => walk(n, prefix, paths),
            }
            prefix.pop();
        }
    }
    for sc in structs {
        let mut prefix = vec![sc.name.clone()];
        walk(&sc.fields, &mut prefix, &mut paths);
    }
    paths
}

/// One column of the engine's row input, as the BOUNDARY sees it: where to
/// read it out of a row (or an arrow batch), and what reading it means.
///
/// The lane list is built ONCE, in `prepare_opaque`, and handed out on
/// [`super::Prepared`]. `Program::in_cols` is its projection, not a second
/// list: `Prepared::input_lanes()[i].col()` is `program.in_cols[i]` for
/// every `i`, and a debug assert ties them at construction.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InputLane {
    /// Display name — what a refusal calls this lane. For a struct leaf it
    /// is the dotted path (TASK-132: display, never data); for a minted
    /// presence lane it is `"<dotted path> (present)"`.
    pub name: String,
    /// SEGMENT path (TASK-132). A plain column is ONE segment, dots and all
    /// — a name is not a path. Never empty.
    pub path: Vec<String>,
    pub kind: LaneKind,
}

/// What walking an [`InputLane`]'s path yields, and therefore which
/// obligations the boundary owes it. Adding a variant here is a compile
/// error at every boundary site, which is the point of the type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneKind {
    /// A caller-supplied column: the path ends at a SCALAR. The boundary
    /// dtype-checks it (arrow) and refuses `None` when `!nullable`.
    Value(ColTy),
    /// Minted by the binder for a struct join key (TASK-133): the path ends
    /// at a struct NODE and the lane's VALUE is that node's validity.
    /// Carries no type — a presence lane is non-nullable `Ty::I1`, always —
    /// which is what lets `ColData::push_present`'s `unreachable!` rest on a
    /// type rather than on a threshold. It is unreachable given that every
    /// `ColData` for a lane comes from `duckdb::col_for_lane`.
    Present,
}

impl InputLane {
    /// This lane as an IR input column. `Present` synthesizes
    /// `ColTy { ty: Ty::I1, nullable: false }`.
    pub fn col(&self) -> Col {
        Col {
            name: self.name.clone(),
            ty: match self.kind {
                LaneKind::Value(ct) => ct,
                LaneKind::Present => ColTy {
                    ty: Ty::I1,
                    nullable: false,
                },
            },
        }
    }
}

/// Every input lane for a row model, in IR order: plain scalar columns in
/// declaration order, then each struct's scalar leaf lanes in struct order.
/// All `Value` — the minted lanes are appended by `prepare_opaque`, which is
/// the only place the two halves ever meet.
pub fn input_lanes(cols: &[Col], structs: &[StructCol]) -> Vec<InputLane> {
    lane_paths(cols, structs)
        .into_iter()
        .zip(cols)
        .map(|(path, c)| InputLane {
            name: c.name.clone(),
            path,
            kind: LaneKind::Value(c.ty),
        })
        .collect()
}

/// A fitted tree transform's schema, as given to `prepare` — like
/// [`StaticTable`], this holds no data, only what the binder needs to check a
/// call site. `name` is the UDF's own name: a tree transform is called
/// `name(id, feats...)` like any other declared transform, so the binder
/// resolves it in the same namespace and the arguments bind by position.
#[derive(Clone, Debug)]
pub struct ModelTable {
    pub name: String,
    /// The DECLARED feature types, in call order — the same `takes` every
    /// other UDF is bound against. Binding against the argument's own type
    /// instead would make the engine disagree with both DuckDB (which casts
    /// to the declaration) and the class's own `__call__` (which narrows an
    /// integer lane in one step): a declared DOUBLE handed a BIGINT column
    /// must reach the model as `float64(n)`, a declared BIGINT as `n`.
    pub takes: Vec<Ty>,
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
/// It belongs to the TRANSFORM, not to an instance within it: the instance id
/// is a runtime value (`score(id, ..)`), while the narrowing is a lowering
/// decision made once at build. A per-instance flag could only be honoured
/// with a per-row branch.
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

/// How a map key compares — and therefore its FLATTENED SLOT LAYOUT.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyCmp {
    /// `=`: NULL propagates. One slot. A NULL key never matches, and each
    /// site says so its own way (drop the build row / AND the probe flag /
    /// empty the fan-out range) — that rule is NOT part of the layout.
    Eq,
    /// `IS NOT DISTINCT FROM`: NULL is an ordinary key value. Two slots.
    NotDistinct,
}

/// The layout of ONE map key. `ty` is the COMPARISON lane type — the probe
/// expression's type after `promote_key`, which may be wider than the static
/// column's own (an INTEGER column keyed against an F64 probe compares in
/// F64; the column's real value then rides a shadow VALUE lane, TASK-120).
/// Deliberately NOT the column type; see [`MapVal`].
///
/// INVARIANT: `ty` is stored ALREADY LANE-ERASED ([`Ty::lane`]). Every
/// producer this type replaces erased at the point of construction, and
/// `StaticTy`'s type vector is what a gate whose claim is "nothing moves"
/// compares. `promote_key` can leave an `I32`, so an un-erased `ty` would
/// print `map(i32, ..)` where the tree prints `map(i64, ..)` — an IR-shape
/// change wearing a refactor's clothes. `Ty::lane()` is identity on `I1` /
/// `F64` / `Str` / `Dec`, so this bites only on narrow integer keys, which
/// is exactly why nothing else notices and why it is written down here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MapKey {
    pub ty: Ty,
    pub cmp: KeyCmp,
}

/// The layout of ONE map value. `ty` is the static COLUMN's lane type, and
/// LANE-ERASED on the same terms as [`MapKey::ty`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MapVal {
    pub ty: Ty,
    pub nullable: bool,
}

impl MapKey {
    /// The flattened slot types this key occupies, in order. THE rule —
    /// every `StaticTy::Map`/`MultiMap` key vector is a flat_map of this,
    /// and every encoder that fills those slots walks the same shape.
    pub fn slots(self) -> Vec<Ty> {
        match self.cmp {
            KeyCmp::Eq => vec![self.ty],
            KeyCmp::NotDistinct => vec![Ty::I1, self.ty],
        }
    }
}

impl MapVal {
    /// The flattened slot types, in order. Same rule, value side: a
    /// declared-nullable column rides as (validity i1, payload) so a NULL
    /// join value flows through as NULL rather than as an error (TASK-55).
    pub fn slots(self) -> Vec<Ty> {
        if self.nullable {
            vec![Ty::I1, self.ty]
        } else {
            vec![self.ty]
        }
    }
}

/// The key layouts of one join, in declaration order.
pub fn map_keys(keys: &[SExpr], key_cols: &[JoinKey]) -> Vec<MapKey> {
    keys.iter()
        .zip(key_cols)
        .map(|(k, jk)| MapKey {
            ty: k.ty.lane(),
            cmp: jk.cmp,
        })
        .collect()
}

/// The value layouts of one join, in declaration order.
///
/// A FREE FUNCTION over an explicit column source, not a [`JoinSpec`]
/// method, because the two callers read from two different places:
/// `StaticTy::Map` and `MultiMap` read the static catalog's columns, and
/// `StaticTy::BatchMap` reads the caller's own `in_cols` — [`JoinSpec::table`]
/// is MEANINGLESS when `batch` is true, so a method taking the catalog has
/// no route to the batch case. The source is chosen at the call site.
pub fn map_vals(cols: &[Col], val_cols: &[u32]) -> Vec<MapVal> {
    val_cols
        .iter()
        .map(|&c| {
            let ct = cols[c as usize].ty;
            MapVal {
                ty: ct.ty.lane(),
                nullable: ct.nullable,
            }
        })
        .collect()
}

/// `(validity_slot, payload_slot)` index pairs for a whole value vector, in
/// slot space — the probe-dst layout that mirrors [`MapVal::slots`].
pub fn slot_pairs(vals: &[MapVal]) -> Vec<(Option<usize>, usize)> {
    let mut out = Vec::with_capacity(vals.len());
    let mut i = 0usize;
    for v in vals {
        if v.nullable {
            out.push((Some(i), i + 1));
            i += 2;
        } else {
            out.push((None, i));
            i += 1;
        }
    }
    out
}

/// One map key on the STATIC side of a [`JoinSpec`]: where the build side
/// reads it, and how it compares.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JoinKey {
    pub src: KeySrc,
    pub cmp: KeyCmp,
}

/// Where a key's build-side value comes from.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum KeySrc {
    /// A lane of the static table (index into [`StaticTable::cols`]).
    Lane(u32),
    /// "the struct node at this SEGMENT path is non-NULL" (TASK-133). A
    /// STRUCT join key expands into leaf keys plus one PRESENCE key per
    /// node, because DuckDB's nested `=` carries each node's own validity
    /// as a VALUE (`row_matcher.cpp:379-382`: top-level Equals, every child
    /// NOT_DISTINCT_FROM): `{inner: NULL}` misses `{inner: {val: NULL}}`
    /// although the two flatten to the same leaf tuple. No lane exists for
    /// a node on either side, so both sides synthesize the boolean —
    /// NULL when the node is absent, TRUE when it is present — and the
    /// ordinary plain / IS-NOT-DISTINCT key machinery does the rest.
    Present(Vec<String>),
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
    /// Static-table map keys, aligned with `keys`. Each carries its own
    /// comparison, and therefore its own slot count — see [`MapKey::slots`].
    pub key_cols: Vec<JoinKey>,
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
    /// Dec(p,s) -> f64, DuckDB's div/mod algorithm (NOT a correctly-rounded
    /// conversion — see kernels::dec_to_f64). Inserted where a DECIMAL
    /// meets a DOUBLE, which is the direction DuckDB casts: only
    /// decimal->double is a legal implicit cast (cast_rules.cpp:196-204).
    DecToFloat(Box<SExpr>),
    /// integer lane -> the scaled i128 of Dec(_, s). The OTHER direction of
    /// the same rule: against an integer DuckDB casts the INTEGER up, so
    /// the comparison stays exact. `s` is the target scale.
    IntToDec {
        s: u8,
        a: Box<SExpr>,
    },
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
        | SKind::DecToFloat(a)
        | SKind::IntToDec { a, .. }
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

/// Could DuckDB's BINDER constant-fold this expression? True iff the
/// subtree references no input (`Col`/`StaticCol`/`JoinHit`) and runs no
/// user code (`ExternCall`/`TreePredict` — TASK-101 relaxes pure externs
/// with constant args). This is a SPELLING test, deliberately weaker than
/// our own `fold`: fold dead-arm-eliminates a CASE whose column sits in an
/// untaken arm, while DuckDB's binder refuses to fold anything holding a
/// column — and its bind-time typing rules (the TASK-102 || collapse) key
/// on ITS notion, so the gate must too.
pub fn bind_foldable(e: &SExpr) -> bool {
    match &e.kind {
        SKind::Col(_) | SKind::StaticCol { .. } | SKind::JoinHit(_) => false,
        SKind::ExternCall { .. } | SKind::TreePredict { .. } => false,
        SKind::Lit(_) | SKind::NullOf => true,
        SKind::Arith { a, b, .. }
        | SKind::Cmp { a, b, .. }
        | SKind::And { a, b }
        | SKind::Or { a, b }
        | SKind::Concat { a, b }
        | SKind::Str2 { a, b, .. }
        | SKind::MathF2 { a, b, .. }
        | SKind::Str2i { a, n: b, .. }
        | SKind::Round2 { a, n: b, .. }
        | SKind::Trim { a, chars: b, .. } => bind_foldable(a) && bind_foldable(b),
        SKind::Not(a)
        | SKind::IsNull { inner: a, .. }
        | SKind::IntToFloat(a)
        | SKind::DecToFloat(a)
        | SKind::IntToDec { a, .. }
        | SKind::IntToFloat32(a)
        | SKind::Cast { inner: a, .. }
        | SKind::StrCase { a, .. }
        | SKind::Abs(a)
        | SKind::Round(a)
        | SKind::SLen { a, .. }
        | SKind::ReMatch { a, .. }
        | SKind::ReExtract { a, .. }
        | SKind::ReReplace { a, .. }
        | SKind::MathF1 { a, .. }
        | SKind::Sord { a, .. }
        | SKind::StripAccents(a)
        | SKind::Reverse(a) => bind_foldable(a),
        SKind::Substr { a, start, len } => {
            bind_foldable(a)
                && bind_foldable(start)
                && len.as_deref().map_or(true, bind_foldable)
        }
        SKind::Like { a, p, esc, .. } => {
            bind_foldable(a) && bind_foldable(p) && esc.as_deref().map_or(true, bind_foldable)
        }
        SKind::Str3 { a, b, c, .. } => {
            bind_foldable(a) && bind_foldable(b) && bind_foldable(c)
        }
        SKind::Spad { a, len, pad, .. } => {
            bind_foldable(a) && bind_foldable(len) && bind_foldable(pad)
        }
        SKind::Sslice { a, lo, hi } => {
            bind_foldable(a) && bind_foldable(lo) && bind_foldable(hi)
        }
        SKind::Case { arms, default } => {
            arms.iter().all(|(c, r)| bind_foldable(c) && bind_foldable(r))
                && default.as_deref().map_or(true, bind_foldable)
        }
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
