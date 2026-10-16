//! The relational IR: what the frontend produces, what BTA annotates, what
//! lowering consumes. Deliberately skinny — the shape is the
//! scan/filter/project ribbon over the dynamic table, and joins to static
//! tables are not tree nodes at all (see [`Rel`]).

use super::ir::{ColTy, CmpPred, Col, DecOp, DecUnary as DecUnaryOp, Lit, TrimSide, Ty};

/// The bound query as an ordered pipeline of stages, innermost first: one
/// stage per query level (docs/specs/2026-09-26-row-local-subqueries-design.md).
///
/// A stage runs its joins in FROM order, then its WHERE, then evaluates
/// EVERY one of its SELECT columns; the next stage reads those results as
/// [`SKind::Slot`]s. Each slot is computed once per row that reaches its
/// stage, read or not, which is DuckDB's evaluation order for a subquery.
/// The last stage's projection is the query's output.
pub struct Plan {
    pub stages: Vec<Stage>,
}

pub struct Stage {
    /// The joins this stage runs, in FROM order: EXECUTION ownership. Each
    /// is an index into the query-wide [`JoinSpec`] list, whose position is
    /// the join's STORAGE identity (join `j` probes static `@j`).
    pub joins: Vec<u32>,
    pub pred: Option<SExpr>,
    pub project: Vec<(String, SExpr)>,
    /// The values the projection reads through [`SKind::Let`], each
    /// reading only earlier ones. The other clauses never read one.
    pub lets: Vec<SExpr>,
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
    /// The DECLARED column list, in declaration order, as a star must see
    /// it: one entry per schema column. A struct column is ONE opaque entry
    /// here even though its flattened leaves live in `cols` — a star
    /// answers with the column, never with its leaves.
    pub star: Vec<StarCol>,
    /// Struct columns as TREES: resolution walks these; the leaf lanes
    /// interleaved in `cols` keep their dotted names for DISPLAY ONLY.
    /// `StructNode::Leaf` here indexes into `cols`.
    pub structs: Vec<StructCol>,
    /// One lane per struct NODE (each struct column and each struct field
    /// below it): its SEGMENT path and its index into `cols`, which holds
    /// it after every other lane. Its value is TRUE when the node is
    /// non-NULL and NULL when it is NULL, so a projected struct can tell a
    /// NULL struct from a struct of NULLs. Like a leaf, it is reachable
    /// only through its path ([`StaticTable::is_leaf_lane`]).
    pub presence: Vec<(Vec<String>, u32)>,
    /// Its row count, when the caller knows it. The engine always does:
    /// the tables are frozen at construction.
    pub rows: Option<usize>,
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
            presence: Vec::new(),
            rows: None,
        }
    }

    /// Whether lane `ci` belongs to a struct — a LEAF, or a node's
    /// presence lane: reachable only through its path, never by name — a
    /// quoted identifier that happens to spell the dotted display name is a
    /// different reference.
    pub fn is_leaf_lane(&self, ci: u32) -> bool {
        fn walk(fs: &[StructField], ci: u32) -> bool {
            fs.iter().any(|f| match &f.node {
                StructNode::Leaf(l) => *l == ci,
                StructNode::Opaque => false,
                StructNode::Nested(n) => walk(n, ci),
            })
        }
        self.presence.iter().any(|(_, l)| *l == ci)
            || self.structs.iter().any(|sc| walk(&sc.fields, ci))
    }

    /// The presence lane of the struct node at `path` (see
    /// [`StaticTable::presence`]).
    pub fn presence_lane(&self, path: &[String]) -> Option<u32> {
        self.presence.iter().find(|(p, _)| p == path).map(|(_, l)| *l)
    }
}

/// A static table computed at build from a static table of the caller: one
/// row per group of the source rows that share their key values, with one
/// column per aggregate. It backs the hidden LEFT join of a correlated scalar
/// subquery (`(SELECT sum(s.v) FROM s WHERE s.k = t.k)`, see
/// `frontend/subquery.rs`): a request reads the row of the group it matches,
/// and a request that matches no group reads the aggregate of no rows.
#[derive(Clone)]
pub struct AggTable {
    /// The grouped table as the catalog sees it: one lane per key, in key
    /// order, then one lane per aggregate.
    pub table: StaticTable,
    /// The source table's index in the caller's catalog.
    pub source: usize,
    /// Per key lane: the source lane it groups by.
    pub keys: Vec<u32>,
    /// Per aggregate lane: what it computes over a group.
    pub aggs: Vec<Agg>,
    /// The subquery's WHERE terms that read only the source table: the
    /// build groups only the rows on which every one is TRUE.
    pub filter: Option<AggFilter>,
    /// Set when the subquery's SELECT item is an expression over its
    /// aggregates (`max(v) + 1`), or over the columns of one row
    /// ([`AggPost::row`]): the table then has one value lane, the
    /// expression over each group's aggregates, which the build computes.
    pub post: Option<AggPost>,
}

/// The SELECT item of a scalar subquery that is an expression over its
/// aggregates, bound as the item of a query of its own whose input is the
/// aggregates (`frontend/subquery.rs`): the program the build runs over
/// the groups once it has their aggregates.
#[derive(Clone)]
pub struct AggPost {
    /// Per aggregate of the [`AggTable`], in order: its value as the
    /// program reads it.
    pub in_cols: Vec<Col>,
    /// The item: one entry.
    pub project: Vec<(String, SExpr)>,
    pub out_cols: Vec<Col>,
    pub regexes: Vec<super::ir::ReSpec>,
    /// Set when the item reads the columns of a row, not aggregates
    /// (`(SELECT s.v FROM s WHERE s.k = t.k)`): each input is a column of
    /// the row, which the [`AggTable`] lists as the aggregate `min` of that
    /// column, its value over one row. The build evaluates the item over
    /// each row of a group, and the group's value is the item over the row
    /// that [`AggPost::order`] keeps. DuckDB traps when a request matches a
    /// group that keeps more than one row, so the build refuses such a
    /// group by name. A group that keeps no row answers NULL, as a request
    /// that matches no group does.
    pub row: bool,
    /// The ORDER BY, LIMIT and OFFSET of an item over one row; `None`
    /// without them, where a group keeps all its rows.
    pub order: Option<RowOrder>,
}

/// The ORDER BY, LIMIT and OFFSET of a scalar subquery whose item reads one
/// row ([`AggPost::row`]), `(SELECT s.v FROM s WHERE s.k = t.k ORDER BY
/// s.o DESC LIMIT 1)`. DuckDB evaluates the item over every row that a
/// request matches, sorts these rows, and keeps the rows from position
/// `offset` on, at most `limit` of them (measured 2026-10-15). The build
/// does the same over each group. DuckDB does not specify the order of the
/// rows that the ORDER BY ties, so a group whose kept row ties rows with
/// another item refuses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowOrder {
    /// The ORDER BY terms, in written order; empty without ORDER BY.
    pub keys: Vec<OrderKey>,
    /// `LIMIT n`, at most the greatest BIGINT; `None` without LIMIT.
    /// DuckDB reads `LIMIT NULL` as that greatest BIGINT: under a key, it
    /// keeps the rows numbered `offset < r <= offset + limit`, with an
    /// addition that wraps, so a sum past that BIGINT keeps no row
    /// (measured 2026-10-15).
    pub limit: Option<u64>,
    /// `OFFSET m`; 0 without OFFSET or with `OFFSET NULL`.
    pub offset: u64,
}

/// One ORDER BY term of a [`RowOrder`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderKey {
    /// The source lane it sorts by; `None` for the SELECT item, which the
    /// term names by its position 1 or by its name.
    pub lane: Option<u32>,
    pub desc: bool,
    /// Whether NULLs sort first. DuckDB sorts them last by default, under
    /// DESC too (measured 2026-10-15).
    pub nulls_first: bool,
}

/// The WHERE terms of a scalar subquery that read only its static table,
/// bound as the items of a query of their own over that table
/// (`frontend/subquery.rs`): the program the build runs over the table's
/// rows before it groups them.
#[derive(Clone)]
pub struct AggFilter {
    /// The table's scalar columns in declared order: the query's input.
    pub in_cols: Vec<Col>,
    /// Per term, in written order: the term as a BOOLEAN that cannot trap.
    pub project: Vec<(String, SExpr)>,
    pub out_cols: Vec<Col>,
    pub regexes: Vec<super::ir::ReSpec>,
}

/// One aggregate of an [`AggTable`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Agg {
    pub func: AggFunc,
    /// The source lane it reads; `None` for `count(*)`.
    pub arg: Option<u32>,
    /// Whether it reads each distinct value of its argument once, as in
    /// `count(DISTINCT v)`. DuckDB's equality decides which values are
    /// equal: -0.0 and 0.0 are one DOUBLE, as are all NaNs (measured
    /// 2026-10-15). Never set for `min` and `max`, whose answer DISTINCT
    /// does not change.
    pub distinct: bool,
    /// Whether `min` or `max` may answer either of two values that DuckDB's
    /// order calls equal but whose bits differ (-0.0 and 0.0, or two NaN
    /// payloads). Set where only a comparison reads the answer, under ANY
    /// and ALL; else the build refuses such a tie, as DuckDB answers the
    /// first one in the order of the rows.
    pub ties: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggFunc {
    CountStar,
    Count,
    Min,
    Max,
    Sum,
    /// A DOUBLE: the exact sum over the count, rounded the way DuckDB
    /// finalizes it for the argument's type ([`avg_in_long_double`],
    /// `duckdb/avg.rs`).
    Avg,
}

/// Whether DuckDB finalizes `avg` over a column of `src` in C `long double`
/// (DuckDB 1.5.5's `avg.cpp`, and measured): every integer type and DECIMAL
/// but SMALLINT and a DECIMAL of width 4 or less, which sum into a BIGINT and
/// finalize in `double`. TINYINT and the unsigned types up to UINTEGER reach
/// the BIGINT overload through a cast, UBIGINT the HUGEINT one.
pub fn avg_in_long_double(src: Ty) -> bool {
    !matches!(src, Ty::I16 | Ty::Dec(..=4, _))
}

/// Whether the platform's C `long double` is the x87 extended format that
/// `duckdb/avg.rs` computes: measured on x86_64 Linux only. Elsewhere it is
/// IEEE binary128 or plain `double`, and an `avg` that DuckDB finalizes in it
/// refuses.
pub const LONG_DOUBLE_IS_X87: bool = cfg!(all(target_arch = "x86_64", target_os = "linux"));

/// Per-lane SEGMENT paths for a lane set with struct trees over it: a
/// plain column's path is its own (whole) name — dots included, a name is
/// not a path — and a leaf lane's is `[struct, field, ...]` from its tree.
/// The DATA paths walk these; nothing splits a name string.
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
    /// is the dotted path (display, never data); for a minted presence lane
    /// it is `"<dotted path> (present)"`.
    pub name: String,
    /// SEGMENT path. A plain column is ONE segment, dots and all — a name
    /// is not a path. Never empty.
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
    /// Minted by the binder for a struct join key: the path ends at a
    /// struct NODE and the lane's VALUE is that node's validity.
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
/// onto the float32 grid and declares `F32`; a packer for a library that
/// compares in float64 declares `F64` and its integer features reach the
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

/// A struct row column flattened to scalar LANES at build time
/// (pins-waveA/struct-star.json + struct-nested.json): the binder resolves
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
    /// How many input LANES this column flattens to, nested fields
    /// included. An `Opaque` leaf has no lane and does not count.
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
/// F64; the column's real value then rides a shadow VALUE lane).
/// Deliberately NOT the column type; see [`MapVal`].
///
/// INVARIANT: `ty` is stored ALREADY LANE-ERASED ([`Ty::lane`]). Erasing at
/// construction, not at each use, is what keeps this `ty` and the
/// `StaticTy` type vector the IR declares in agreement. `promote_key` can
/// leave an `I32`, so an un-erased `ty` would print `map(i32, ..)` where
/// the tree prints `map(i64, ..)` — a silent IR-shape change. `Ty::lane()`
/// is identity on `I1` / `F64` / `Str` / `Dec`, so this bites only on
/// narrow integer keys, which is exactly why nothing else notices and why
/// it is written down here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MapKey {
    pub ty: Ty,
    pub cmp: KeyCmp,
}

/// The layout of ONE map value. `ty` is the BUILD-side column's lane type
/// (the static table's, or the batch's own under a batchmap), LANE-ERASED
/// on the same terms as [`MapKey::ty`].
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
    /// join value flows through as NULL rather than as an error.
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
    /// "the struct node at this SEGMENT path is non-NULL".
    /// A STRUCT join key expands into leaf keys plus one PRESENCE key per
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
    /// Multiplicity self-join: the build side is the BATCH itself (a keyless
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
    /// Set for the hidden join of a scalar subquery that the WHERE of its
    /// level does not read: the stage forces its probe after the WHERE, as
    /// DuckDB evaluates a subquery of the SELECT list over the rows that
    /// the WHERE passes (`frontend/subquery.rs`). Every other join probes
    /// before the WHERE.
    pub after_where: bool,
    /// Set for the hidden join of a scalar subquery that the program
    /// computes for each request by a scan ([`Fold`]). Its static holds no
    /// row.
    pub fold: Option<Fold>,
    /// Set for the join whose rows a [`Fold`] scans ([`ScanRows`]): no stage
    /// probes it.
    pub scan: Option<ScanRows>,
}

/// The most static rows that the scan of a [`Fold`] reads for one request:
/// the rows of one key, or of the whole table when the subquery has no key.
/// The build refuses a table where more rows share one key. The owner gave
/// the number (decisions thread, 2026-10-15: "iterate on small lists if the
/// subquery can't be turned into a lookup, say 100 or so items").
pub const SCAN_LIMIT: usize = 100;

/// The value of a scalar subquery whose WHERE compares its static table
/// with the request other than by key equality, `(SELECT sum(s.v) FROM s
/// WHERE s.lo <= t.x AND t.x < s.hi)` (`frontend/subquery.rs`). The program
/// computes it for each request by a scan: it reads the rows of the join
/// `rows` whose keys equal the request's, keeps each row on which `pred`
/// is TRUE, and folds each aggregate of `aggs` over the rows it keeps, into
/// one value lane of the join that holds this fold. That join matches when
/// the scan keeps a row.
#[derive(Clone)]
pub struct Fold {
    /// The join whose rows the scan reads ([`JoinSpec::scan`]).
    pub rows: u32,
    /// The WHERE terms that are not equality keys and do not read only the
    /// table, ANDed: a BOOLEAN that cannot trap. It reads the row through
    /// [`SKind::StaticCol`] of the join `rows`.
    pub pred: Option<SExpr>,
    /// Per value lane of the join that holds the fold: the aggregate, and
    /// the value lane of `rows` that it reads (`None` for `count(*)`).
    pub aggs: Vec<(AggFunc, Option<u32>)>,
}

/// The rows of a static table that a [`Fold`] scans, as the join that reads
/// them holds them: its static is a multimap of the rows, keyed by the
/// subquery's equality keys.
#[derive(Clone)]
pub struct ScanRows {
    /// The subquery's WHERE terms that read only the table: the build keeps
    /// only the rows on which each one is TRUE.
    pub filter: Option<AggFilter>,
}

/// A bound, typed scalar expression. `nullable` is the frontend's
/// conservative derivation ("cannot prove non-NULL"), and it is a contract
/// with lowering: an expression lowers to a flag lane IFF `nullable` — the
/// out-column nullability, the CASE join shape, and the store form all key
/// off it.
#[derive(Clone, PartialEq)]
pub struct SExpr {
    pub kind: SKind,
    pub ty: Ty,
    pub nullable: bool,
}


#[derive(Clone, PartialEq)]
pub enum SKind {
    /// Input column, by index into the dynamic table's schema.
    Col(u32),
    /// Column `i` of the PREVIOUS stage's projection (see [`Plan`]): already
    /// computed, range-checked and trapped where it was produced, so reading
    /// it evaluates nothing.
    Slot(u32),
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
    /// `error('msg')` as a CASE result: typed like a NULL there (DuckDB's
    /// SQLNULL), and trapping with the full message whenever evaluated.
    Raise(String),
    /// greatest/least over arguments of one I64, F64 or VARCHAR lane: the
    /// first argument that is not NULL and that no later one beats (NULL
    /// only when every argument is), DuckDB's order (NaN above +inf).
    /// Lowered as a running extreme, each argument evaluated once in order.
    Extreme {
        greatest: bool,
        args: Vec<SExpr>,
    },
    /// Evaluate every item in order, for its traps, and answer item `pick`:
    /// a field read over `struct_pack`, which builds every field on DuckDB
    /// (so a sibling's trap fires) and answers one. Items that cannot trap
    /// are dropped at bind, so every item but `pick` can trap.
    Seq {
        items: Vec<SExpr>,
        pick: usize,
    },
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
    /// DECIMAL `+ - * %` (docs/specs/decimal-expressions.md §3-§6). The
    /// operands already carry the types DuckDB's binder casts them to, so
    /// this is scaled-integer arithmetic; `check` is the width the binder
    /// capped the result at (18 or 38), whose overflow traps, 0 when the
    /// width cannot overflow. `%` by zero is NULL.
    DecArith {
        op: DecOp,
        check: u8,
        a: Box<SExpr>,
        b: Box<SExpr>,
    },
    /// A checked conversion with a DECIMAL on one side (§8): integer ->
    /// Dec, Dec -> Dec, Dec -> integer (half away from zero), and Dec ->
    /// VARCHAR. Source and target are the operand's and the node's types.
    DecCast(Box<SExpr>),
    /// TRY_CAST with a DECIMAL on one side: the checked conversion, NULL
    /// where [`SKind::DecCast`] would trap.
    DecTryCast(Box<SExpr>),
    /// A DECIMAL rounding builtin (abs, ceil, floor, round, trunc) on the
    /// scaled integer: divide out `10^k`, multiply back `10^m`
    /// (`kernels::dec_unary`). Total.
    DecUnary {
        op: DecUnaryOp,
        k: u8,
        m: u8,
        a: Box<SExpr>,
    },
    /// i64 -> f64 VIA f32 — `n as f32 as f64`, one rounding, not two.
    /// Only ever wraps a `tree_predict` feature: sklearn narrows an integer
    /// feature array to float32 in a single step, and above 2**53 that is a
    /// different number from `float32(float64(n))`. Below 2**53 it is
    /// identical to [`SKind::IntToFloat`], which is what makes it safe to
    /// apply unconditionally.
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
    /// Two strings in (haystack, needle), NULL-propagating. The searches and
    /// GLOB are total; the edit distances, jaccard and hamming can trap.
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
    /// Regex match against program regex `re` (full-match forms
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
    /// f64 unary math (operand promoted to F64 by the frontend);
    /// NULL-propagating; the trapping ops get safe-masked payloads in
    /// lowering so a NULL row can never fire the domain trap.
    MathF1 {
        op: super::ir::NumOp1,
        a: Box<SExpr>,
    },
    /// f64 binary math: Fpow (total), Flogb(base, x) (trapping), and
    /// Ffloordiv/Ffloormod/Fnextafter (all total).
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
    /// repeat / VARCHAR array_extract — (Str, I64) -> Str. Both can trap:
    /// repeat past the string-builder budget, the subscript outside ±2^32.
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
    /// One lane of a declared-UDF extern call. The k+1
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
    /// A subexpression of the stage's projection evaluated once per row,
    /// before the first item, and read here (`share.rs`). Exists only
    /// between that pass and lowering; its value cannot trap.
    Shared(u32),
    /// A SQL function's subexpression that its body reads more than once,
    /// bound once per call: entry `n` of the stage's [`Stage::lets`]. The
    /// binder makes one only for a value that cannot trap and has no
    /// effect, so where and how often it is evaluated is invisible
    /// (`frontend/lets.rs`); `share.rs` reads the entry in its place.
    Let(u32),
}

impl SExpr {
    /// Every direct sub-expression, mutably, in evaluation order. The one
    /// generic walk over the tree: a pass that only cares about some leaves
    /// (a rewrite, a reference scan) recurses through this instead of
    /// restating all forty-odd kinds, so a new kind is one arm here.
    pub fn children_mut(&mut self) -> Vec<&mut SExpr> {
        match &mut self.kind {
            SKind::Col(_)
            | SKind::Slot(_)
            | SKind::StaticCol { .. }
            | SKind::Lit(_)
            | SKind::NullOf
            | SKind::Raise(_)
            | SKind::JoinHit(_)
            | SKind::Shared(_)
            | SKind::Let(_) => Vec::new(),
            SKind::Seq { items, .. } => items.iter_mut().collect(),
            SKind::Extreme { args, .. } => args.iter_mut().collect(),
            SKind::Arith { a, b, .. }
            | SKind::Cmp { a, b, .. }
            | SKind::And { a, b }
            | SKind::Or { a, b }
            | SKind::Concat { a, b }
            | SKind::Str2 { a, b, .. }
            | SKind::MathF2 { a, b, .. }
            | SKind::Trim { a, chars: b, .. }
            | SKind::Round2 { a, n: b, .. }
            | SKind::DecArith { a, b, .. }
            | SKind::Str2i { a, n: b, .. } => vec![a.as_mut(), b.as_mut()],
            SKind::IntToFloat(a)
            | SKind::DecToFloat(a)
            | SKind::IntToDec { a, .. }
            | SKind::DecCast(a)
            | SKind::DecTryCast(a)
            | SKind::DecUnary { a, .. }
            | SKind::IntToFloat32(a)
            | SKind::Not(a)
            | SKind::IsNull { inner: a, .. }
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
            | SKind::Reverse(a) => vec![a.as_mut()],
            SKind::Case { arms, default } => {
                let mut out: Vec<&mut SExpr> = Vec::new();
                for (c, r) in arms.iter_mut() {
                    out.push(c);
                    out.push(r);
                }
                if let Some(d) = default {
                    out.push(d.as_mut());
                }
                out
            }
            SKind::Substr { a, start, len } => {
                let mut out = vec![a.as_mut(), start.as_mut()];
                if let Some(l) = len {
                    out.push(l.as_mut());
                }
                out
            }
            SKind::Like { a, p, esc, .. } => {
                let mut out = vec![a.as_mut(), p.as_mut()];
                if let Some(e) = esc {
                    out.push(e.as_mut());
                }
                out
            }
            SKind::Str3 { a, b, c, .. } => vec![a.as_mut(), b.as_mut(), c.as_mut()],
            SKind::Spad { a, len, pad, .. } => vec![a.as_mut(), len.as_mut(), pad.as_mut()],
            SKind::Sslice { a, lo, hi } => vec![a.as_mut(), lo.as_mut(), hi.as_mut()],
            SKind::TreePredict { id, feats, .. } => {
                let mut out = vec![id.as_mut()];
                out.extend(feats.iter_mut());
                out
            }
            SKind::ExternCall { args, .. } => args.iter_mut().collect(),
        }
    }
}

/// Can evaluating this expression trap — overflow, division by zero, a
/// failed CAST, an unknown model id? Conservative in one direction only:
/// anything not on the trap-free allowlist counts as trapping.
///
/// The JOIN ON residual rule is what consumes it (`bind_residual`): a
/// single-side residual has to be trap-free because DuckDB scan-pushes it,
/// so a trap would fire at a different time than ours.
///
/// A CASE is trap-free exactly when all of its arms are: lowering branches,
/// so an arm that is not taken is never evaluated.
pub fn may_trap(e: &SExpr) -> bool {
    match &e.kind {
        SKind::Col(_)
        | SKind::Slot(_)
        | SKind::StaticCol { .. }
        | SKind::JoinHit(_)
        | SKind::Shared(_)
        | SKind::Let(_)
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
        SKind::Extreme { args, .. } => args.iter().any(may_trap),
        // Arith overflows, CAST fails, ABS traps on i64::MIN, tree_predict
        // rejects an unknown model id — and anything not named above is
        // simply unclassified. All of it counts as trapping. Total ops land
        // here too: a sign-bit flip (`MathF1{Fneg}`) cannot trap, so naming
        // it would ACCEPT single-side residuals this refuses today. That is
        // a widening, with its own DuckDB timing question to measure, and
        // deliberately not part of teaching the scan about the node.
        _ => true,
    }
}

/// Whether evaluating `e` can raise, as precisely as the kinds named here
/// allow: [`may_trap`] stays deliberately coarse for the JOIN ON residual
/// policy, while this one decides which struct fields and list elements a
/// read must still evaluate (`SKind::Seq`), where every item kept costs a
/// lane per read. Double arithmetic never traps (a zero divisor answers
/// inf/NaN, or a NULL flag); an integer or DECIMAL one can overflow. Any
/// kind not named counts as trapping.
pub fn can_trap(e: &SExpr) -> bool {
    can_trap_under(e, &mut Vec::new())
}

/// [`can_trap`] where `facts` hold: each a CASE condition with whether it
/// was taken (TRUE) or passed (FALSE or NULL) on the way to `e`.
fn can_trap_under<'a>(e: &'a SExpr, facts: &mut Vec<(&'a SExpr, bool)>) -> bool {
    use super::ir::{NumOp1 as Op, StrOp2};
    let any = |xs: &[&'a SExpr], facts: &mut Vec<(&'a SExpr, bool)>| {
        xs.iter().any(|x| can_trap_under(x, facts))
    };
    match &e.kind {
        // Total on every DOUBLE, NaN and the infinities included (measured,
        // DuckDB 1.5.5: exp(1000) is inf, -NaN is NaN).
        SKind::MathF1 {
            op: Op::Fneg | Op::Fabs | Op::Fround | Op::Fexp | Op::Fcbrt | Op::Ffloor | Op::Fceil | Op::Ftrunc,
            a,
        } => any(&[a], facts),
        // ln/log2/log10 raise exactly when x <= 0 (-0.0 and -inf included;
        // NaN is NaN), sqrt when x < 0 (sqrt(-0.0) is -0.0). Under a CASE
        // guard that rules that out, they cannot.
        SKind::MathF1 {
            op: op @ (Op::Ln | Op::Log2 | Op::Log10 | Op::Fsqrt),
            a,
        } => {
            let strict = !matches!(op, Op::Fsqrt);
            any(&[a], facts) || !facts.iter().any(|&(c, taken)| guards(c, taken, a, strict))
        }
        // sin/cos/tan raise exactly on +-inf (measured, DuckDB 1.5.5: "input
        // value inf is out of range"; NaN is NaN). Under a CASE guard that
        // rules the infinities out, they cannot.
        SKind::MathF1 {
            op: Op::Fsin | Op::Fcos | Op::Ftan,
            a,
        } => any(&[a], facts) || !facts.iter().any(|&(c, taken)| finite_guard(c, taken, a)),
        // ROUND(DOUBLE): total, like `Fround`, which it lowers to.
        SKind::Round(a) => any(&[a], facts),
        SKind::Case { arms, default } => {
            let depth = facts.len();
            let mut trap = false;
            for (c, r) in arms {
                if can_trap_under(c, facts) {
                    trap = true;
                    break;
                }
                facts.push((c, true));
                let t = can_trap_under(r, facts);
                facts.pop();
                if t {
                    trap = true;
                    break;
                }
                facts.push((c, false));
            }
            if !trap {
                trap = default.as_deref().is_some_and(|d| can_trap_under(d, facts));
            }
            facts.truncate(depth);
            trap
        }
        SKind::Arith { a, b, .. } if e.ty.lane() == Ty::F64 => any(&[a, b], facts),
        // Only the integer abs traps (on its minimum); DuckDB's DOUBLE abs
        // is fabs (measured: `abs(-1e308)`, `abs(-inf)`, `abs(NaN)` answer).
        SKind::Abs(a) if e.ty.lane() == Ty::F64 => any(&[a], facts),
        SKind::Cmp { a, b, .. } | SKind::And { a, b } | SKind::Or { a, b } => any(&[a, b], facts),
        SKind::Not(a) | SKind::IsNull { inner: a, .. } | SKind::IntToFloat(a) => any(&[a], facts),
        SKind::Cast { inner, trying } if *trying || cast_is_total(inner.ty, e.ty) => {
            any(&[inner], facts)
        }
        // A DECIMAL to DOUBLE is total, as is a cast to a DECIMAL that holds
        // every value of the source.
        SKind::DecToFloat(a) => any(&[a], facts),
        SKind::DecCast(a) if dec_cast_is_total(a.ty, e.ty) => any(&[a], facts),
        SKind::Seq { items, .. } => items.iter().any(|x| can_trap_under(x, facts)),
        SKind::Extreme { args, .. } => args.iter().any(|x| can_trap_under(x, facts)),
        // String operations that raise on no input: their kernels return no
        // error, and DuckDB 1.5.5 raises none over the inputs of
        // `tests/test_string_ops_cannot_trap.py`. LIKE raises only through
        // its ESCAPE. substr, lpad/rpad, repeat, a VARCHAR subscript and the
        // edit distances can raise, so they stay unnamed.
        SKind::Like { a, p, esc: None, .. }
        | SKind::Concat { a, b: p }
        | SKind::Trim { a, chars: p, .. }
        | SKind::Str2 {
            op: StrOp2::Find | StrOp2::Contains | StrOp2::Starts | StrOp2::Ends | StrOp2::Glob,
            a,
            b: p,
        } => any(&[a, p], facts),
        SKind::Str3 { a, b, c, .. } => any(&[a, b, c], facts),
        SKind::Sslice { a, lo, hi } => any(&[a, lo, hi], facts),
        SKind::StrCase { a, .. }
        | SKind::SLen { a, .. }
        | SKind::Sord { a, .. }
        | SKind::StripAccents(a)
        | SKind::Reverse(a) => any(&[a], facts),
        _ => can_trap_here(e),
    }
}

/// Whether condition `c`, TRUE (`taken`) or else FALSE or NULL, leaves `a`
/// positive (`strict`) or non-negative, NaN or NULL: the values on which
/// ln (`strict`) or sqrt cannot raise. DuckDB orders NaN above every
/// number, so `x <= 0` is FALSE for NaN, and `x > 0` TRUE.
fn guards(c: &SExpr, taken: bool, a: &SExpr, strict: bool) -> bool {
    match (&c.kind, taken) {
        (SKind::Or { a: l, b: r }, false) | (SKind::And { a: l, b: r }, true) => {
            guards(l, taken, a, strict) || guards(r, taken, a, strict)
        }
        (SKind::Cmp { pred, a: l, b: r }, _) => {
            let (pred, x, k) = match (lit_f64(r), lit_f64(l)) {
                (Some(k), _) => (*pred, l, k),
                (None, Some(k)) => {
                    let mirrored = match pred {
                        CmpPred::Lt => CmpPred::Gt,
                        CmpPred::Le => CmpPred::Ge,
                        CmpPred::Gt => CmpPred::Lt,
                        CmpPred::Ge => CmpPred::Le,
                        p => *p,
                    };
                    (mirrored, r, k)
                }
                _ => return false,
            };
            if strip_float(x) != strip_float(a) {
                return false;
            }
            // What `x` is then known to be beyond (NULL or NaN aside):
            // `> k` (open) or `>= k` (closed).
            let bound = match (pred, taken) {
                (CmpPred::Le, false) | (CmpPred::Gt, true) => Some((k, true)),
                (CmpPred::Lt, false) | (CmpPred::Ge, true) => Some((k, false)),
                _ => None,
            };
            match bound {
                Some((k, open)) => {
                    if strict {
                        k > 0.0 || (k == 0.0 && open)
                    } else {
                        k >= 0.0
                    }
                }
                None => false,
            }
        }
        _ => false,
    }
}

/// Whether condition `c`, TRUE (`taken`) or else FALSE or NULL, leaves `a`
/// finite, NaN or NULL: the values on which sin/cos/tan cannot raise. The
/// guard reads `abs(a)` against a constant: `abs(a) = inf` passed,
/// `abs(a) <> inf` taken, `abs(a) < k` taken, `abs(a) <= k` taken for a
/// finite k (NaN compares FALSE with `=` and `<`, and TRUE with `<>`, all of
/// which leave it to an arm where it is harmless).
fn finite_guard(c: &SExpr, taken: bool, a: &SExpr) -> bool {
    match (&c.kind, taken) {
        (SKind::Or { a: l, b: r }, false) | (SKind::And { a: l, b: r }, true) => {
            finite_guard(l, taken, a) || finite_guard(r, taken, a)
        }
        (SKind::Cmp { pred, a: l, b: r }, _) => {
            let (pred, x, k) = match (lit_f64(r), lit_f64(l)) {
                (Some(k), _) => (*pred, l, k),
                (None, Some(k)) => {
                    let mirrored = match pred {
                        CmpPred::Lt => CmpPred::Gt,
                        CmpPred::Le => CmpPred::Ge,
                        CmpPred::Gt => CmpPred::Lt,
                        CmpPred::Ge => CmpPred::Le,
                        p => *p,
                    };
                    (mirrored, r, k)
                }
                _ => return false,
            };
            let inner = match &strip_float(x).kind {
                SKind::Abs(i) | SKind::MathF1 { op: super::ir::NumOp1::Fabs, a: i } => i,
                _ => return false,
            };
            if strip_float(inner) != strip_float(a) {
                return false;
            }
            let inf = k == f64::INFINITY;
            match (pred, taken) {
                (CmpPred::Eq, false) | (CmpPred::Ge, false) | (CmpPred::Ne, true) => inf,
                (CmpPred::Lt, true) => true,
                (CmpPred::Le, true) => k.is_finite(),
                _ => false,
            }
        }
        _ => false,
    }
}

fn strip_float(e: &SExpr) -> &SExpr {
    match &e.kind {
        SKind::IntToFloat(a) => strip_float(a),
        _ => e,
    }
}

fn lit_f64(e: &SExpr) -> Option<f64> {
    match &strip_float(e).kind {
        SKind::Lit(Lit::F64(v)) if !v.is_nan() => Some(*v),
        SKind::Lit(Lit::I64(v)) => Some(*v as f64),
        _ => None,
    }
}

fn can_trap_here(e: &SExpr) -> bool {
    match &e.kind {
        SKind::Col(_)
        | SKind::Slot(_)
        | SKind::StaticCol { .. }
        | SKind::JoinHit(_)
        | SKind::Shared(_)
        | SKind::Let(_)
        | SKind::Lit(_)
        | SKind::NullOf => false,
        SKind::Arith { a, b, .. } if e.ty.lane() == Ty::F64 => can_trap(a) || can_trap(b),
        SKind::Cmp { a, b, .. } | SKind::And { a, b } | SKind::Or { a, b } => {
            can_trap(a) || can_trap(b)
        }
        SKind::Not(a) | SKind::IsNull { inner: a, .. } | SKind::IntToFloat(a) => can_trap(a),
        // A cast that cannot fail: TRY_CAST (a failure is NULL), and the
        // conversions that are total -- an integer or a DOUBLE to DOUBLE,
        // anything numeric or BOOLEAN to VARCHAR, an integer to a width
        // that holds its whole range, BOOLEAN to an integer, an integer or
        // a DOUBLE to BOOLEAN.
        SKind::Cast { inner, trying } if *trying || cast_is_total(inner.ty, e.ty) => {
            can_trap(inner)
        }
        SKind::Case { arms, default } => {
            arms.iter().any(|(c, r)| can_trap(c) || can_trap(r))
                || default.as_deref().is_some_and(can_trap)
        }
        SKind::Seq { items, .. } => items.iter().any(can_trap),
        SKind::Extreme { args, .. } => args.iter().any(can_trap),
        // A closed constant that folds to a value (`CAST('0.0' AS DOUBLE)`,
        // how a typed constant is spelled) cannot trap: fold leaves a
        // failing cast in place, to trap at run time.
        _ if bind_foldable(e) => !matches!(
            super::fold::fold(e.clone()).kind,
            SKind::Lit(_) | SKind::NullOf
        ),
        _ => true,
    }
}

/// Whether every value of `from` converts to `to` through
/// [`SKind::DecCast`]: a DECIMAL to VARCHAR, a DECIMAL to one with no fewer
/// integer digits, and one more if the scale shrinks (rounding can carry
/// into a new digit: 9.99 to 10.0), and an integer whose whole range has
/// fewer digits than the integer part of `to`.
fn dec_cast_is_total(from: Ty, to: Ty) -> bool {
    match (from, to) {
        (Ty::Dec(..), Ty::Str) => true,
        (Ty::Dec(p1, s1), Ty::Dec(p2, s2)) => {
            let (i1, i2) = (p1 - s1, p2 - s2);
            if s2 >= s1 {
                i2 >= i1
            } else {
                i2 > i1
            }
        }
        (a, Ty::Dec(p, s)) if a.is_integer() => {
            let lim = 10i128.pow(u32::from(p - s));
            let (lo, hi) = a.int_range128().expect("an integer width");
            -lim < lo && hi < lim
        }
        _ => false,
    }
}

/// Whether every value of `from` converts to `to` (see `can_trap`).
fn cast_is_total(from: Ty, to: Ty) -> bool {
    match (from, to) {
        (a, b) if a == b => true,
        (a, Ty::F64) => a.is_integer() || a == Ty::I1,
        (a, Ty::Str) => a.is_integer() || a == Ty::I1 || a == Ty::F64 || a.dec().is_some(),
        (Ty::I1, b) => b.is_integer(),
        // Nonzero is TRUE, NaN and the infinities included (measured,
        // DuckDB 1.5.5): the cast of a WHERE term that is not a BOOLEAN.
        (a, Ty::I1) => a.is_integer() || a == Ty::F64,
        (a, b) if a.is_integer() && b.is_integer() => {
            let range = |t: Ty| t.int_range128().expect("an integer width");
            let ((alo, ahi), (blo, bhi)) = (range(a), range(b));
            blo <= alo && ahi <= bhi
        }
        _ => false,
    }
}

/// What a sibling kept only for its traps must still evaluate: `None` when
/// it cannot trap. A CASE keeps its conditions (they decide which arm runs,
/// and may trap) and the results that can trap; the rest answer NULL. The
/// value is never read, so this traps exactly when `e` does, with the same
/// message, first trap first. Siblings that differ only in their trap-free
/// values (a fitted lane per field, every one with the same
/// `ELSE error(..)`) come out equal, and a read keeps one of them.
pub fn trap_skeleton(e: &SExpr) -> Option<SExpr> {
    if !can_trap(e) {
        return None;
    }
    match &e.kind {
        SKind::Case { arms, default } => {
            let null = || SExpr {
                kind: SKind::NullOf,
                ty: e.ty,
                nullable: true,
            };
            Some(SExpr {
                kind: SKind::Case {
                    arms: arms
                        .iter()
                        .map(|(c, r)| (c.clone(), trap_skeleton(r).unwrap_or_else(null)))
                        .collect(),
                    default: default
                        .as_deref()
                        .and_then(trap_skeleton)
                        .map(Box::new),
                },
                ty: e.ty,
                nullable: true,
            })
        }
        // An operation that cannot trap itself traps where its operands do,
        // in operand order: keep only theirs. Lane j of a Normalizer,
        // `x_j / CASE WHEN norm < tiny THEN 1.0 ELSE norm END`, keeps the
        // CASE every lane shares, not the lane.
        _ if traps_only_in_operands(e) => {
            let mut c = e.clone();
            let mut parts: Vec<SExpr> = c
                .children_mut()
                .into_iter()
                .filter_map(|x| trap_skeleton(x))
                .collect();
            match parts.len() {
                0 => None,
                1 => parts.pop(),
                n => {
                    let (ty, nullable) = (parts[n - 1].ty, parts[n - 1].nullable);
                    Some(SExpr {
                        kind: SKind::Seq {
                            items: parts,
                            pick: n - 1,
                        },
                        ty,
                        nullable,
                    })
                }
            }
        }
        _ => Some(e.clone()),
    }
}

/// Whether `e`'s own operation is total, so it traps only where an operand
/// does: the kinds [`can_trap`] looks through, short of CASE (whose arms
/// are conditional) and AND/OR.
fn traps_only_in_operands(e: &SExpr) -> bool {
    match &e.kind {
        SKind::Arith { .. } | SKind::Abs(_) => e.ty.lane() == Ty::F64,
        SKind::Cmp { .. } | SKind::Not(_) | SKind::IsNull { .. } | SKind::IntToFloat(_) => true,
        SKind::Cast { inner, trying } => *trying || cast_is_total(inner.ty, e.ty),
        _ => false,
    }
}

/// Could DuckDB's BINDER constant-fold this expression? True iff the
/// subtree references no input (`Col`/`StaticCol`/`JoinHit`) and runs no
/// user code (`ExternCall`/`TreePredict`; a PURE extern over constant args
/// is folded separately, by the frontend's bind-time execution). This is a
/// SPELLING test, deliberately weaker than our own `fold`: fold
/// dead-arm-eliminates a CASE whose column sits in an untaken arm, while
/// DuckDB's binder refuses to fold anything holding a column — and its
/// bind-time typing rules (the `||`-to-SQLNULL collapse) key on ITS notion,
/// so the gate must too.
/// Whether `op` at result type `ty` answers NULL for a zero or NULL divisor
/// (DuckDB: integer `%`, and `//` on ints and doubles). The lowering applies
/// it as a flag, so the dividend is always evaluated; the binder reads it
/// for nullability.
pub fn zero_divisor_nulls(op: ArithOp, ty: Ty) -> bool {
    (op == ArithOp::Rem && ty.is_integer()) || op == ArithOp::IDiv
}

pub fn bind_foldable(e: &SExpr) -> bool {
    match &e.kind {
        // A slot never folds: constants do not fold across a query level
        // (DuckDB: `k + MAX` over `SELECT 1 AS k` errors per row, and not at
        // all on zero rows).
        SKind::Col(_)
        | SKind::Slot(_)
        | SKind::StaticCol { .. }
        | SKind::JoinHit(_)
        | SKind::Shared(_)
        | SKind::Let(_) => false,
        // `error()` is never folded at bind: it raises when a row reaches it.
        SKind::ExternCall { .. } | SKind::TreePredict { .. } | SKind::Raise(_) => false,
        SKind::Seq { items, .. } => items.iter().all(bind_foldable),
        SKind::Extreme { args, .. } => args.iter().all(bind_foldable),
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
        | SKind::DecArith { a, b, .. }
        | SKind::Trim { a, chars: b, .. } => bind_foldable(a) && bind_foldable(b),
        SKind::Not(a)
        | SKind::IsNull { inner: a, .. }
        | SKind::IntToFloat(a)
        | SKind::DecToFloat(a)
        | SKind::IntToDec { a, .. }
        | SKind::DecCast(a)
        | SKind::DecTryCast(a)
        | SKind::DecUnary { a, .. }
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
/// NULL on both (the lowering's result flag, `zero_divisor_nulls`).
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
