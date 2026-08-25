# Lane and slot seams become types instead of conventions

Origin: an Ousterhout-lens code review of packages/confit, 2026-08-25, plus
three grounding audits that traced each seam against every call site in the
tree at `debaf8c`. Every file:line below was re-verified against that tree
while writing this document; the handful of places where a grounding claim did
not survive re-verification are called out inline as DEVIATION notes.

Revised 2026-08-25 after an adversarial verification round -- a citation audit
and an implementer dry-run -- that failed the first draft on 20 fact errors and
9 decisions it left to the implementer. Every correction below was re-checked
against the tree before being applied; where a CORRECTION was itself wrong the
first draft's text stands with the reason recorded (the nine-vs-eleven count in
Finding 2, and the `*Spec:*`-footer half of the house-style note under Proposed
additions). The document is ASCII-only by house rule, so em-dashes are spelled
`--` throughout.

This is a refactor spec, not a feature spec. It ships no behavior. Its whole
value is that the two tickets queued behind it -- TASK-134 (a third input-lane
kind) and TASK-135 (the fan-out loop reading the slot rule) -- would otherwise
each have to re-learn, and re-write, a convention that is already written
between three and nine times.

## Context

### Finding 1: the input-lane kind is an index threshold

An input lane is one of two things, and which one is encoded as "is my index
at least `present_from`".

The threshold is built at `src/duckdb/mod.rs:1736` (`let present_from =
in_cols.len();`), immediately before the append loop at `:1737-1740` that
pushes `prepared.present_lanes` onto both `in_cols` and `in_paths`. It is then
stored twice more -- `Engine::Compiled.present_from` (`src/duckdb/mod.rs:1393`)
and `Marshaller.present_from` (`src/duckdb/mod.rs:1180`, set at `:1202`) -- and
branched on at three sites, each with the identical shape:

```
src/duckdb/arrow.rs:229      if i >= present_from { ... continue; }
src/duckdb/mod.rs:1283       if i >= self.present_from { col.push_present(!null); continue; }
src/duckdb/mod.rs:1978       if i >= present_from { col.push_present(!null); continue; }
```

Three facts make this a convention rather than a type:

- **It is maintained at three independent sites, and nothing checks that they
  agree.** `src/specializer/frontend.rs:3900` mints the lane's IR index as
  `SKind::Col((self.in_cols.len() + idx) as u32)`; `src/specializer/mod.rs:143-144`
  appends the same lanes to `all_in` for lowering; `src/duckdb/mod.rs:1737-1740`
  appends them a second time, independently, to the boundary's own `in_cols`.
  `all_in` becomes `Program::in_cols` verbatim (`src/specializer/lower.rs:446`,
  and `ir::canonicalize` at `src/specializer/ir/mod.rs:1336-1366` renumbers
  `Value` ids only), so the boundary vector and the program vector are two
  copies of one list. A divergence desyncs `Inst::Load{col}` from `Batch::cols`
  and produces WRONG ANSWERS, not a refusal.
- **It has an unstated temporal coupling.** `src/duckdb/mod.rs:1731` computes
  `in_paths = plan::lane_paths(&in_cols, &structs)` and MUST run before the
  append at `:1737`. `lane_paths` gives a plain lane the path `[c.name]`
  (`src/specializer/plan.rs:96`), so run after the append a presence lane would
  get the path `["w (present)"]` and the boundary would look up an attribute
  literally named `w (present)`. Nothing in the field declarations says this.
- **A threshold cannot express three kinds.** TASK-134 adds a key-only lane.
  Key-only and presence lanes both mint lazily from the binder, so they
  interleave in mint order inside the appended region and no single index
  separates them.

The two lane kinds also carry different obligations, and today those live in
the branch bodies rather than in a type: a value lane refuses `None` on a
non-nullable column (`src/duckdb/mod.rs:1287-1292`, `:1982-1987`; `:1988` is
the `push_input_cell` call that follows the refusal, not part of it) and is dtype-
checked at arrow ingest (`src/duckdb/arrow.rs:246-264`); a presence lane skips
both, is always non-nullable I1, and lands on `ColData::I1` with
`valid: vec![true; rows]` (`src/duckdb/arrow.rs:240-243`). The last of those is
enforced by an `unreachable!("a presence lane is always I1")` at
`src/specializer/exec/mod.rs:130`, reached through a `ColData` allocated from
the lane's `Ty` at two separate constructors (`src/duckdb/mod.rs:1217` via
`ColData::new`, and the hand-inlined capacity-hinted match at
`src/duckdb/mod.rs:1906-1930`).

### Finding 2: the map slot-pair convention, with its typed-default table copied four times

The rule, stated once:

> A flattened map-static slot vector is built by walking the logical key (or
> value) columns in declaration order and emitting TWO slots -- `Ty::I1`
> validity, then the payload masked to the type default -- for a key whose
> comparison is IS NOT DISTINCT FROM or for a value column declared nullable,
> and ONE payload slot otherwise. Probe side and build side must produce
> byte-identical encodings, and the NULL payload must be the SAME type default
> on both.

It is currently written in nine places, across eleven listed sites. The
counting rule, because the arithmetic below depends on it: a site COUNTS as a
writing of the rule when it emits or consumes the two-slot form. The two
plain-only sites -- `lower.rs:158-167` and `lower.rs:1627-1633` -- do not; they
are the fan-out GAP, listed here because they are where the rule is missing and
where TASK-135 writes it for the tenth and eleventh time. Producers:

```
src/specializer/lower.rs:255-269   StaticTy::Map keys      (INDF -> [I1, ty])
src/specializer/lower.rs:270-283   StaticTy::Map values    (nullable -> [I1, ty])
src/specializer/lower.rs:145-157   the `flat` closure      (a literal re-write of the line above)
src/specializer/lower.rs:158-167   StaticTy::MultiMap keys (plain-only: the fan-out gap, in declaration form)
src/specializer/lower.rs:1963-1982 emit_probe key encode   (the probe-side encoder)
src/specializer/lower.rs:1627-1633 lower_many_loop keys    (plain-only: the same gap, in encoder form)
```

Consumers:

```
src/duckdb/mod.rs:889-910          build-side key encode
src/duckdb/mod.rs:959-976          build-side value encode
src/specializer/lower.rs:1550-1569 val_slots     (index pairing)
src/specializer/lower.rs:1572-1590 val_flat_tys  (type vector; a duplicate of the producer above)
src/specializer/exec/interp.rs:464-497  build_batch_rows (a THIRD writer of the value pair)
```

The iterator-skip at `src/duckdb/mod.rs:893-894` IS the seam, written out:

```rust
let _validity_ty = kt.next();
let ty = *kt.next().expect("payload type follows validity");
```

and its value twin at `src/duckdb/mod.rs:962-963`. Nothing connects that skip
to the `flat_map` in `lower.rs` that produced the layout it is skipping over.

Four typed-default tables must agree and nothing checks that they do:

```
src/specializer/lower.rs:497-507        probe side, as Lit
src/duckdb/mod.rs:897-903               build key, as KeyBits
src/duckdb/mod.rs:966-972               build value, as ScalarVal
src/specializer/exec/interp.rs:485-491  batchmap value, as ScalarVal
```

(Two more of the same SHAPE live at a different seam --
`src/specializer/exec/kernels.rs:96-104` for extern returns and
`src/specializer/exec/cranelift.rs:853-859` for `h_sload`. Those are not this
seam and are deliberately not dragged in.)

Alongside it, the same concept rides as parallel vectors at three layers:
`plan::JoinSpec` carries `keys` / `key_cols` / `key_indf`
(`src/specializer/plan.rs:233-240`); `StaticSpec` carries `key_cols` /
`key_indf` / `key_present` / `val_cols` / `val_nullable`
(`src/specializer/mod.rs:48-72`, derived at `:175-202`); and the boundary zips
three of them at once at `src/duckdb/mod.rs:819-824`.

### Finding 3: the 10-argument / 7-tuple prepare surface

```rust
// src/specializer/frontend.rs:332-358
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn frontend(sql, this_name, in_cols, opaque, structs, statics, many, udfs, models, bind_eval)
    -> Result<(Rel, Vec<JoinSpec>, Vec<Col>, Vec<ReSpec>, Vec<WideOut>, Vec<u32>,
               Vec<(Col, Vec<String>)>), PrepareError>
```

The 7-tuple has exactly ONE consumer, `src/specializer/mod.rs:136-139`.
Elements 3, 4 and 5 (`out_cols`, `regexes`, `wide_outputs`) are pure conduits:
prepare never reads them, it forwards them to `lower` or straight onto
`Prepared`. They are in the tuple only because `frontend` and `lower` are two
functions.

`prepare_opaque` (`src/specializer/mod.rs:121-135`) takes ten arguments and has
14 callers: `src/duckdb/mod.rs:1633`, the 4-arg `prepare` adapter at
`src/specializer/mod.rs:111`, and 12 in `src/specializer/tests.rs`. Four of the
test sites spell a lone flag inside a run of `&[]` -- `tests.rs:1173-1184`,
`:3621-3632`, `:3705-3717`, `:3732-3743` -- which is exactly the shape where a
silent argument transposition hides. Only one test passes a non-empty `opaque`
at all (`tests.rs:3164`), and that is the one TASK-134 changes again.

The 4-arg `prepare` (`src/specializer/mod.rs:105-112`) has zero non-test
callers and 31 test call sites; it is a test convenience wearing a `pub` coat.

### Why now

Both queued tickets consume these seams, and both would deepen them.

- **TASK-134** adds a third input-lane kind (servable-as-key, not-as-value) for
  opaque scalar columns. `present_from: usize` cannot express three kinds, so
  all three `i >= present_from` branches have to become a per-lane match
  anyway. Doing it inside 134 means doing it while also writing per-arrow-type
  ingest conversions and an equality proof matrix -- two unrelated risks in one
  diff.
- **TASK-135** teaches the fan-out loop the pair encoding. Without `MapKey` it
  writes the pair rule a TENTH and ELEVENTH time (a second INDF arm at
  `lower.rs:164`, a second masked encode at `lower.rs:1627`), in the one path
  where build side and probe side sit in different layers and no test compares
  them. With `MapKey` it is two call-site substitutions and one `if` deleted in
  each of two files.

## Goals

- **Behavior-preserving, and that IS the gate.** Byte-identical outputs,
  identical refusal messages (including the synthetic lane name
  `format!("{} (present)", path.join("."))` at
  `src/specializer/frontend.rs:3888`, which is user-reachable), identical
  build/refuse decisions, full suite green with no test edited except where a
  Rust unit test names a changed internal type. No new pins, no changed pins.
- One home per rule: one place that says how many slots a key takes, one place
  that says what a NULL payload is on the build side, one place that builds the
  input-lane list.
- The three `i >= present_from` branches become exhaustive matches with no
  catch-all, so TASK-134's third kind is a compile error at every site that has
  to handle it rather than a silent fall-through.
- Fewer parallel vectors: `Engine::Compiled`'s three lane fields become one,
  and `StaticSpec`'s five become two.
- **Three debug asserts, and that is the whole assert budget.** They are
  release-invisible, so none of them changes behavior:
  1. `input_lanes()[i].col() == program.in_cols[i]` at construction in
     `prepare_opaque`. Documentation, not the guarantee -- see Section 1
     complication 11.
  2. `!k.present || k.map.ty == Ty::I1` at `StaticKey` construction (Section 2
     complication 5).
  3. `debug_assert_eq!` on FLATTENED slot lengths in `materialize_statics`
     (`src/duckdb/mod.rs:1055-1080`): the sum of `spec.keys[i].map.slots().len()`
     against `keys.len()` from the `StaticTy`, and the same for values. This one
     replaces a check that exists today by accident -- see Section 2
     complication 8 -- and it guards a silent wider join, not a panic.

## Non-goals

- **No Python API change.** `DuckDBInferFn`'s pyo3 signature
  (`src/duckdb/mod.rs:1425`), the `shape` / `output_schema` / `backend` /
  `boundary` getters, `infer_rows` and `infer_arrow` are untouched.
  `StaticSpec` is Rust-internal (no `#[pyclass]`), so nothing here crosses the
  boundary.
- **Not unifying dialect and specializer**, and not touching the IR text
  format. `StaticTy::Map` keeps `keys: Vec<Ty>` / `values: Vec<Ty>`: the
  printer emits a flat type list (`src/specializer/ir/print.rs:19-27`), the
  parser reads one (`src/specializer/ir/parse.rs:719-794`), `prepare`
  CANONICALIZES so `parse(print(p)) == p` holds (`ir::canonicalize` at
  `src/specializer/mod.rs:151`, then `ir::verify::verify` at `:152`) while the
  assertion itself lives only in tests
  (`src/specializer/exec/tests.rs:1270`, `src/specializer/ir/tests.rs:32`,
  `src/specializer/tests.rs:313`), and the IR fuzz
  generator builds them (`src/specializer/ir/gen.rs:214-233`). `MapKey`/`MapVal`
  live in `specializer::plan` and `slots()` is what FEEDS `StaticTy`.
- **Not the `frontend.rs` god-module split.** That file is 4500+ lines and the
  split is a real finding, but it is a different, larger change and it would
  make this one unreviewable. The only frontend edits here are the ones the
  seams force.
- **Not a `Shape` enum.** `DuckDBInferFn::new` represents one concept three
  ways (`many` at `src/duckdb/mod.rs:1440`, `shape_kind` at `:1441-1446`,
  `strict_map` at `:1447-1456`), and unifying them looks adjacent. It is not
  safe to do here: `strict_map` cannot ride along. Today a shape='map' query
  that fails prepare with `Unsupported`/`Parse` falls into the constant-emitter
  arm (`src/duckdb/mod.rs:1657-1689`), and the shape='map' refusal fires only
  if `eval_static_only` SUCCEEDS (`:1673-1679`). If `Shape::Map` made prepare
  itself refuse on `one_row_blocker`, that refusal would arrive as
  `PrepareError::Unsupported`, route into the same constant-emitter arm, and
  emit a different message with an `"unsupported: "` prefix. A two-thirds
  unification is worse than none. `many: bool` stays a `bool`.
- **`RowTy` is deliberately deferred to the decimal-widening opener.** A
  newtype for "a type that can be a ROW input lane" -- the row schema
  vocabulary minus `Dec` -- would delete the `Ty::Dec(..) => unreachable!("a
  decimal row column is opaque")` arms at `src/duckdb/arrow.rs:361`,
  `src/duckdb/mod.rs:1928`, `src/specializer/exec/cranelift.rs:1988` and
  `:2021`, `src/specializer/exec/mod.rs:99`, and
  `src/specializer/exec/tests.rs:912`. It belongs with the work that changes
  which types can be row lanes, not with a refactor whose gate is
  "nothing moves".

  DEVIATION from the dispatch brief, which said 16, and from this spec's own
  first draft, which said 12: the measured count is 15 decimal `unreachable!`
  arms in `packages/confit/src`, carrying FOUR distinct claim-kinds.

  - **Row lane** (6, the ones a `RowTy` would kill, listed above):
    `src/duckdb/arrow.rs:361`, `src/duckdb/mod.rs:1928`,
    `src/specializer/exec/cranelift.rs:1988`, `:2021`,
    `src/specializer/exec/mod.rs:99`, `src/specializer/exec/tests.rs:912`.
  - **Key / probe lane** (3): `"a probe expression is never a decimal"` at
    `src/duckdb/mod.rs:886`, `:902`, and
    `src/specializer/exec/cranelift.rs:2066`.
  - **Extern boundary** (5): `"a udf parameter / return is never a decimal"`
    at `src/specializer/exec/cranelift.rs:1003`, `:1022`, `:1482`, `:1527`,
    and `src/specializer/exec/interp.rs:1588`.
  - **Wide field child** (1): `"a wide field child is never a decimal"` at
    `src/duckdb/arrow.rs:670`.

  Four newtypes' worth, not one; all four are the same opener's business.

## Design

### Section 1: `InputLane`

#### Current contract

Three parallel vectors of equal length, index-aligned with
`prepared.program.in_cols` and with the executor's `Batch::cols`, in which

- every index `< present_from` means "walk `in_paths[i]` to a SCALAR leaf, push
  its value, refuse `None` on a non-nullable column";
- every index `>= present_from` means "walk `in_paths[i]` to a struct NODE,
  push `!node.is_none()` as a non-nullable I1";
- and the threshold is only meaningful because presence lanes are appended
  last.

Written at `src/duckdb/mod.rs:1731-1740`, mirrored at
`src/specializer/mod.rs:141-144`, forward-referenced at
`src/specializer/frontend.rs:3900`, asserted nowhere at runtime, and pinned
only by `src/specializer/tests.rs:126-186` -- two lane COUNTS (`:168`, `:173`)
and one minted PATH (`:174`). The lane's KIND is pinned nowhere, because today
there is no kind to pin.

#### The new interface

```rust
// src/specializer/plan.rs

/// One column of the engine's row input, as the BOUNDARY sees it: where to
/// read it out of a row (or an arrow batch), and what reading it means.
///
/// The lane list is built ONCE, in `prepare_opaque`, and handed out on
/// [`super::Prepared`]. `Program::in_cols` is its projection, not a second
/// list: `Prepared::input_lanes()[i].col()` is `program.in_cols[i]` for
/// every `i`, and a debug assert ties them at construction.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct InputLane {
    /// Display name -- what a refusal calls this lane. For a struct leaf it
    /// is the dotted path (TASK-132: display, never data); for a minted
    /// presence lane it is `"<dotted path> (present)"`.
    pub name: String,
    /// SEGMENT path (TASK-132). A plain column is ONE segment, dots and all
    /// -- a name is not a path. Never empty.
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
    /// Carries no type -- a presence lane is non-nullable `Ty::I1`, always
    /// -- which is what lets `ColData::push_present`'s `unreachable!` rest
    /// on a type rather than on a threshold. It is unreachable given that
    /// every `ColData` for a lane comes from `duckdb::col_for_lane`; see
    /// complication 10 for what that buys and what it does not.
    Present,
}

impl InputLane {
    /// This lane as an IR input column. `Present` synthesizes
    /// `ColTy { ty: Ty::I1, nullable: false }`.
    pub fn col(&self) -> Col;
    /// The lane's declared type: `Ty::I1` for `Present`.
    pub fn ty(&self) -> Ty;
}

/// Every input lane for a row model, in IR order: plain scalar columns in
/// declaration order, then each struct's scalar leaf lanes in struct order.
/// All `Value`. The minted lanes are appended by `prepare_opaque`.
pub fn input_lanes(cols: &[Col], structs: &[StructCol]) -> Vec<InputLane>;
```

```rust
// src/duckdb/mod.rs -- NOT exec/mod.rs; see complication 10

/// Empty `ColData` for one input lane, optionally with capacity. The ONE
/// place a lane's kind chooses a `ColData` variant. All THREE ingest paths
/// call it -- `Marshaller::call`, the row-boundary loop at `:1937-1989`, and
/// `arrow::ingest` (as `super::col_for_lane`) -- so `push_present` can only
/// ever meet an `I1`.
pub(crate) fn col_for_lane(lane: &InputLane, cap: usize) -> ColData;

/// The hot-path cell writer, re-signed so it takes no `&Col`. `name` and
/// `ty` are exactly what it reads today -- `c.name` for the two error
/// messages and `c.ty.ty` for `arrow_ty_name` / `int_range`.
///
/// CONSTRAINT, and the reason the signature is specified here rather than
/// left to the implementer: this runs ONCE PER CELL PER ROW on the engine's
/// headline path, against a ~200 ns/row floor. `name: &str` borrows out of
/// `InputLane.name`; `ty: ColTy` is `Copy` (`ir/mod.rs:270`). Writing
/// `push_input_cell(col, &lane.col(), ..)` instead would build a fresh
/// `String` per cell and is the one migration reading that passes the suite
/// and loses the benchmark.
fn push_input_cell(col: &mut ColData, name: &str, ty: ColTy,
                   attr: &Bound<'_, PyAny>, null: bool) -> PyResult<()>;
```

**Two erasure conventions, stated together because confusing them is the one
silent bug this section can produce.** `LaneKind::Value(ColTy)` carries the
DECLARED type, narrow widths and all -- `push_input_cell` needs `Ty::I32` to
call `int_range` (`ir/mod.rs:246-252`), and `Program::in_cols` carries the
un-erased type today (`check_input` compares `col.ty() != ty.lane()`,
`interp.rs:423`). Section 2's `MapKey.ty` / `MapVal.ty` are the opposite: they
are stored ALREADY `.lane()`-erased. Lane types never meet column types in the
same field.

DEVIATION from grounding A's sketch, which proposed
`enum InputLane { Value { col, path }, Present { col, path } }`. Rejected on
the grounding's own complication 8: those two variants are payload-identical,
so any site that destructures `col`/`path` generically still compiles when a
third variant lands, and the compiler enforces nothing beyond a tag match.
Hoisting `name` and `path` -- which EVERY reader needs unconditionally -- out
of the variants, and stripping `ColTy` off `Present`, is what removes the
generic-destructure escape hatch and makes `Present`'s "always non-nullable I1"
a fact of the type rather than a comment.

#### Migration list

Writers (the whole triple is produced once, at site W-new):

| site | change |
|---|---|
| `src/specializer/plan.rs:95-113` | `lane_paths` stays (statics still use it at `src/specializer/mod.rs:174`); `input_lanes` is added on top of it |
| `src/specializer/frontend.rs:3881-3898` | `Binder::minted_lanes` (renamed, see complication 12); `present_key` mints an `InputLane { name, path, kind: Present }` directly and dedups by `path` exactly as `:3883` does today. The `name` expression at `:3888` is copied verbatim. |
| `src/specializer/frontend.rs:3899-3903` | `SKind::Col(self.in_cols.len() + idx)` UNCHANGED (see complications) |
| `src/specializer/frontend.rs:352-355`, `:625-633` | return element 7 becomes `Vec<InputLane>`, named `minted_lanes` on `Bound` |
| `src/specializer/frontend.rs:2078` | the FIELD DECLARATION: `present_lanes: RefCell<Vec<(Col, Vec<String>)>>` becomes `minted_lanes: RefCell<Vec<InputLane>>`, carrying the invariant doc comment from Section 3 |
| `src/specializer/frontend.rs:749` | the `Binder` construction line. `RefCell::new(Vec::new())` is type-inferred, so this is a RENAME only -- no type edit. (Both were folded into a single `:722-750` row in the first draft; they are two different edits.) |
| `src/specializer/mod.rs:141-144` | **W-new**: `let mut lanes = plan::input_lanes(in_cols, structs); lanes.extend(minted);` -- and `all_in` becomes `lanes.iter().map(InputLane::col).collect()` |
| `src/specializer/mod.rs:93-98`, `:205-215` | `Prepared::present_lanes` is DELETED and replaced by `input_lanes: Vec<InputLane>` (accessor `input_lanes()`), plus a `debug_assert` that it projects to `program.in_cols` |
| `src/duckdb/mod.rs:1731-1740` | **DELETED**. `lane_paths` call, `present_from` capture, and the second append all go; the boundary reads `prepared.input_lanes()`. This is the whole point of finding 1. |
| `src/duckdb/mod.rs:1383-1393` | `Engine::Compiled`'s `in_cols` / `in_paths` / `present_from` become one `lanes: Vec<InputLane>` |
| `src/duckdb/mod.rs:1755-1762` | construction follows |
| `src/duckdb/mod.rs:1177-1180`, `:1191-1220` | `Marshaller` drops `present_from`; keeps `in_names: Vec<Vec<Py<PyString>>>` (interned, built once); `cols` allocate via `col_for_lane(lane, 0)` |
| `src/duckdb/mod.rs:62` (`push_input_cell`) | signature migration, per the interface block above: `(col, name: &str, ty: ColTy, attr, null)`. Body unchanged -- `c.name` becomes `name`, `c.ty.ty` becomes `ty.ty`, `c.ty.nullable` is already read by the caller. Call sites `:1293` and `:1988`. |
| `src/specializer/plan.rs` (`StructCol`) | unchanged |

Readers:

| site | change |
|---|---|
| `src/duckdb/mod.rs:1225-1294` (`Marshaller::call`) | the `in_cols: &[Col]` parameter becomes `lanes: &[InputLane]` (call site `:1901`); the zip at `:1239-1244` zips lanes with `in_names`; `:1283-1286` becomes `match lane.kind { Present => {..; continue}, Value(ct) => {..} }` with the nullable refusal at `:1287-1292` INSIDE the `Value` arm, and `:1293` calling `push_input_cell(col, &lane.name, ct, ..)` |
| `src/duckdb/mod.rs:1906-1930` | the hand-inlined capacity match is DELETED; `col_for_lane(lane, n)` replaces it. Removes one of the two `Ty::Dec(..) => unreachable!` copies. |
| `src/duckdb/mod.rs:1937-1989` | same match as the marshaller; reads `lane.path` as `&str` with NO interning (the generic path's cost model is the point -- see constraints). The refusal at `:1982-1987` moves inside the `Value` arm; `:1988` is the `push_input_cell` call, not part of the refusal. |
| `src/duckdb/mod.rs:1830-1847`, `:1857-1866` | destructure one `lanes` field instead of three |
| `src/duckdb/arrow.rs:213-245` (`ingest`) | signature `(py, batch, lanes: &[InputLane])`; `walk_lane(batch, &l.path, &l.name)` unchanged; `:229-245` becomes the `Present` arm of an exhaustive match, `:246-264` and `:363-368` become the `Value` arm; any `ColData` it allocates comes from `super::col_for_lane` |
| `src/duckdb/arrow.rs:174-209` (`walk_lane`) | unchanged -- shared by both kinds by design |
| `src/specializer/exec/mod.rs:126-134` (`push_present`) | unchanged; its `unreachable!` is now structurally unreachable |
| `src/specializer/exec/interp.rs:259`, `:1463`; `src/specializer/exec/cranelift.rs:1980`, `:2001` | unchanged -- below `Prepared` a presence lane is an ordinary non-nullable I1 column and `Vec<Col>` stays the IR's input type |
| `src/specializer/ir/verify.rs:120-135`, `:517-533` | unchanged |
| `src/specializer/ir/print.rs:367`, `:373-391` | unchanged -- `ident_or_quoted` still quotes `"w (present)"`, round-trip survives |
| `src/specializer/lower.rs:1550-1557`, `:1571-1578` | unchanged -- batch self-join value slots read `self.in_cols`, which is still the `Vec<Col>` projection |
| `src/specializer/tests.rs:126-186` (`presence_lanes_are_minted_lazily`) | THREE readers, and one of them changes shape -- see below. |

**The one test edit, spelled out**, because the first draft claimed "the
assertion is unchanged" and that is false for one of the three. The readers are
`tests.rs:168`, `:173` and `:174` (`:172` is `let keyed = prep(..)`, not a
reader). `input_lanes()` returns ALL lanes, so a reader of the MINTED SUBSET
has to index past the caller lanes:

```rust
// :168   assert!(plain.present_lanes.is_empty());
         assert_eq!(plain.input_lanes().len(), schema.len());
// :173   assert_eq!(keyed.present_lanes.len(), 1);
         assert_eq!(keyed.input_lanes().len(), schema.len() + 1);
// :174   assert_eq!(keyed.present_lanes[0].1, vec!["w".to_string()]);
         assert_eq!(keyed.input_lanes()[schema.len()].path, vec!["w".to_string()]);
         assert_eq!(keyed.input_lanes()[schema.len()].kind, LaneKind::Present);
```

DECISION: `input_lanes()` plus a position, NOT a minted-subset accessor. A
second accessor would exist for one test and would have to re-derive
`in_cols.len()` -- the same forward-reference arithmetic complication 4 keeps in
exactly one place. The cost is that `:168` becomes a length compare that reads
close to `:169`'s; they check two different vectors (the authority and its
projection), which is worth two lines. The gain is the `kind` assertion on the
last line: the minted lane's KIND was unpinnable before this change.

#### How this resolves the grounding's complications

1. **Two writers of the same append.** Resolved by deletion, not by the type:
   `src/duckdb/mod.rs:1731-1740` goes away and `prepare_opaque` is the only
   producer. `program.in_cols` is derived from the same vector in the same
   function, so the two lists cannot drift.
2. **`lane_paths` must run before the append.** Dissolved: there is one
   construction site, and `input_lanes` takes the caller's `in_cols` before
   anything is minted, by signature.
3. **`n_plain` is derived by subtraction** (`src/specializer/frontend.rs:656`,
   `in_cols.len() - sum(leaf_count)`). UNTOUCHED and load-bearing. It is
   computed inside `bind_from` from the caller's `in_cols`, which never
   contains a minted lane, and every scan that must not see minted lanes is
   bounded by it (`Binder::column` at `:4313-4315`, `this_col_with_fields` at
   `:4171-4176`, star expansion at `:2683-2684` with its
   `.expect("scalar count matches positions")` at `:2698`, the column-list-alias
   budget at `:688`, the batch self-join table at `:809`, `:1024`). This spec
   moves nothing into `in_cols` before `n_plain` is taken, so all six stay
   correct by the same argument they are correct by today.
4. **The forward reference stays.** `SKind::Col(self.in_cols.len() + idx)` at
   `frontend.rs:3900` writes into a lane space the caller constructs later, and
   an `InputLane` type does not fix that: the binder cannot own the lane vector
   because `Binder::in_cols` is a `Cow` that may be an owned RENAMED copy under
   `t AS u(x,y)` (`frontend.rs:709-713`, `:2015`), and because the lane is
   minted lazily on purpose (measured +22..26 ns/row/lane, TASK-133 spec). What
   changes is that the convention now has ONE partner instead of three: only
   `prepare_opaque`'s append has to agree with it.
5. **`" (present)"` is user-reachable and unpinned.** Preserved bit-for-bit --
   the `name` field carries the same string the `Col` carried, and lane ORDER
   is unchanged, so the messages at `src/duckdb/mod.rs:1247`, `:1272`, `:1953`,
   `:1968` and `src/duckdb/arrow.rs:198-202`, `:234` are byte-identical. It is
   still unpinned; see ASK 2.
6. **The `Present` arm does not check the node is a struct.** UNCHANGED, and
   deliberately so: adding an `is_struct` assertion would change behavior for a
   leafless struct (`pa.struct([])`, or a nested struct whose fields are all
   opaque), which the gate forbids. The type makes the gap addressable -- there
   is now one arm to put the check in -- but the check itself is ASK 1.
7. **`present_from` stored three ways.** All three die: `Engine::Compiled`'s
   three lane fields become one, and `Marshaller` loses its copy.
8. **Payload-identical variants buy nothing.** Addressed by the DEVIATION
   above: `Present` carries strictly less than `Value`, and the branch sites
   become exhaustive matches with no catch-all. That is the entire behavioral
   delta of section 1; everything else is bookkeeping.
9. **The static side made the opposite choice.** Section 2 is that sibling.
10. **Presence lanes are invisible below `Prepared`, and that constrains where
    `col_for_lane` can live.** `Vec<Col>` stays the IR's input-column type and
    no `plan` enum crosses into `lower` / `verify` / `exec`. The import graph
    today is one-directional and shallow: `plan.rs:6` imports only
    `super::ir::{..}`, and `exec/mod.rs:27` imports only `super::ir::Ty`.
    Neither module names the other.

    The first draft put `ColData::for_lane(lane: &InputLane, ..)` in
    `exec/mod.rs`, which would have made `exec` import `plan::InputLane` --
    inverting the only layering the two modules have, and contradicting this
    complication in the same document. Section 2's first draft did the same in
    the other direction (`MapKey::null_key_bits -> Vec<KeyBits>` and
    `MapVal::null_vals -> Vec<ScalarVal>` put `plan` on `exec`), so the two
    interfaces together described a cycle. It would compile -- both are in one
    crate -- but "it compiles" is not the property being defended.

    RESOLUTION, applied to both sections:

    - `plan.rs` holds the TYPES and the PURE slot logic only. `InputLane`,
      `LaneKind`, `MapKey`, `MapVal`, `KeyCmp`, `JoinKey`, `KeySrc`,
      `slots()`, `slot_pairs()`, `map_keys()`, `map_vals()`, `input_lanes()`.
      Every one of those mentions `Ty`, `Col` and `String` and nothing else.
    - The functions that need `KeyBits` / `ScalarVal` live at PREPARE TIME on
      the side that owns those types: two free functions in `exec/mod.rs`
      beside their own vocabulary, taking a plain `Ty` and returning plain
      vectors (Section 2's interface block). `exec` never learns a `plan`
      type; `plan` never learns an `exec` type.
    - `col_for_lane` moves to `duckdb/mod.rs` as ONE boundary helper called by
      all three ingest paths, instead of a constructor on `ColData`.

    What is KEPT: exactly one place chooses a `ColData` variant from a lane's
    kind, which is the whole argument that `push_present`'s `unreachable!`
    rests on the type rather than on `present_from`. What is WEAKENED, stated
    plainly: the chooser lives at the BOUNDARY, not in `exec`, so `exec` alone
    cannot prove it -- a future third constructor inside `exec` could still
    hand `push_present` a non-`I1`. Nothing in the tree does that today
    (`ColData::new` at `exec/mod.rs:99` is type-driven and lane-agnostic), and
    buying the stronger version costs the layering. The layering is worth more.
11. **The `input_lanes()[i].col() == program.in_cols[i]` assert is
    DOCUMENTATION, not the guarantee.** With W-new deriving `all_in` from the
    lane vector inside the same function, the assert is a tautology: it
    compares a vector against its own projection, taken three lines apart. It
    costs nothing in release and it tells a reader what the relationship is,
    which is why it stays. But the "two lists cannot drift" property is bought
    ENTIRELY by deleting `src/duckdb/mod.rs:1731-1740` -- the second,
    independent append. A reviewer who reads the assert as the guarantee has
    the causality backwards, and would then accept a future patch that
    re-introduces a second producer while keeping the assert green.
12. **`present_lanes` is renamed to `minted_lanes` everywhere,** including on
    the `Binder`. TASK-134 puts a second KIND in the same vector, so a name
    that says `present` becomes a lie at the moment the next ticket lands, and
    the first draft already used `minted_lanes` on `Bound` while keeping
    `present_lanes` on the `Binder` -- one list, two names, which is the exact
    defect Section 3 complication (f) is about. The write-side invariant that
    the type does NOT enforce is written into the field's doc comment instead;
    see Section 3.

### Section 2: `MapKey` / `MapVal`, and what happens to `StaticSpec`

#### Current contract

Stated in full under Finding 2. Two corollaries the code also relies on, and
which the new types must NOT absorb:

- **A plain (`=`) key's NULL rule is not a slot rule.** A plain key contributes
  one slot; NULL-ness is handled OUTSIDE the slot vector, and differently per
  site: the build side drops the row (`src/duckdb/mod.rs:913-915`,
  `continue 'row`), the scalar probe ANDs the flag into `keys_valid`
  (`src/specializer/lower.rs:1976-1980`, `:1993-1996`), and the fan-out loop
  forces an EMPTY range instead (`src/specializer/lower.rs:1642-1662`).
- **The runtime is layout-blind.** `cmp_key`
  (`src/specializer/exec/interp.rs:1709-1724`) and its cranelift twin
  (`src/specializer/exec/cranelift.rs:898-914`) iterate `KeyBits` positionally
  and know nothing about pairing; `ir::verify` length-checks the flat vectors
  AND positionally type-checks each key through `want(..)`
  (`src/specializer/ir/verify.rs:602-634`, esp. `:618-620`, and `:756-786`,
  esp. `:769-771`) -- what it has no notion of is PAIRING, not types. The pair
  convention never escapes prepare time, which is exactly why an owning type is
  viable AND why it must stop at the IR boundary.

#### The new interface

```rust
// src/specializer/plan.rs

/// How a map key compares -- and therefore its FLATTENED SLOT LAYOUT.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyCmp {
    /// `=`: NULL propagates. One slot. A NULL key never matches, and each
    /// site says so its own way (drop the build row / AND the probe flag /
    /// empty the fan-out range) -- that rule is NOT part of the layout.
    Eq,
    /// `IS NOT DISTINCT FROM`: NULL is an ordinary key value. Two slots.
    NotDistinct,
}

/// The layout of ONE map key. `ty` is the COMPARISON lane type -- the probe
/// expression's type after `promote_key`, which may be wider than the static
/// column's own (an INTEGER column keyed against an F64 probe compares in
/// F64; the column's real value then rides a shadow VALUE lane, TASK-120).
/// Deliberately NOT the column type; see `MapVal`.
///
/// INVARIANT: `ty` is stored ALREADY LANE-ERASED (`Ty::lane()`). Every
/// producer this type replaces erases at the point of construction
/// (`lower.rs:264`, `:266`, `:164`), and `StaticTy`'s type vector is what a
/// gate whose claim is "nothing moves" compares. `promote_key` can leave an
/// `I32`, so an un-erased `ty` would print `map(i32, ..)` where the tree
/// prints `map(i64, ..)` -- an IR-shape change wearing a refactor's clothes.
/// `Ty::lane()` is identity on `I1` / `F64` / `Str` / `Dec`, so this bites
/// only on narrow integer keys, which is exactly why nothing else in the
/// migration notices and why it has to be written down here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MapKey { pub ty: Ty, pub cmp: KeyCmp }

/// The layout of ONE map value. `ty` is the static COLUMN's lane type, and
/// LANE-ERASED on the same terms as `MapKey.ty` (`lower.rs:278`, `:280`,
/// `:151`, `:1584`, `:1586` all erase today).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MapVal { pub ty: Ty, pub nullable: bool }

/// The key layouts of one join, in declaration order. A free function
/// taking its sources EXPLICITLY, for the same reason `map_vals` does.
/// Erases: `MapKey { ty: keys[i].ty.lane(), cmp: key_cols[i].cmp }`.
pub fn map_keys(keys: &[SExpr], key_cols: &[JoinKey]) -> Vec<MapKey>;

/// The value layouts of one join, in declaration order.
///
/// A FREE FUNCTION over an explicit column source, not a `JoinSpec` method,
/// because the two callers read from two different places: `StaticTy::Map`
/// and `MultiMap` read `catalog[spec.table].cols`, and `StaticTy::BatchMap`
/// reads the caller's `in_cols` -- `JoinSpec.table` is documented MEANINGLESS
/// when `batch` is true (`plan.rs:225-227`), so a method taking the catalog
/// has no route to the batch case. `val_slots` / `val_flat_tys` already solve
/// it this way with an explicit `if spec.batch { self.in_cols } else { .. }`
/// at `lower.rs:1552-1556` and `:1574-1578`; this mirrors that, so ONE
/// function serves both and the source is chosen at the call site.
/// Erases: `MapVal { ty: cols[c].ty.ty.lane(), nullable: cols[c].ty.nullable }`.
pub fn map_vals(cols: &[Col], val_cols: &[u32]) -> Vec<MapVal>;

impl MapKey {
    /// The flattened slot types this key occupies, in order. THE rule --
    /// every `StaticTy::Map`/`MultiMap` key vector is a flat_map of this.
    pub fn slots(self) -> Vec<Ty>;
}

impl MapVal {
    /// The flattened slot types, in order. Same rule, value side.
    pub fn slots(self) -> Vec<Ty>;
}

/// `(validity_slot, payload_slot)` index pairs for a whole value vector, in
/// slot space. A FREE FUNCTION, not an `FB` method, so the borrow at
/// `lower.rs:124` cannot force a duplicate (grounding complication 6). It is
/// declared here rather than inside `impl MapVal`, where the first draft put
/// it while its own doc said "a free function" -- one of those was wrong.
pub fn slot_pairs(vals: &[MapVal]) -> Vec<(Option<usize>, usize)>;
```

The two NULL-payload tables cannot live on `MapKey` / `MapVal`: `KeyBits` and
`ScalarVal` are `exec` types and `plan` does not import `exec` (complication
10). They go beside their own vocabulary instead, as free functions over a
plain `Ty`:

```rust
// src/specializer/exec/mod.rs -- beside KeyBits (:305-327) and ScalarVal

/// The `(false, type default)` PAIR a NULL `IS NOT DISTINCT FROM` key stores
/// -- the same default the probe side masks to. Prepare-time / build-time
/// only. `KeyBits` has no `Dec` variant, so this genuinely is a different
/// table from `null_val_slots`, not a copy of it.
///
/// The `Eq`-key rule ("a NULL never matches, so drop the build row") is NOT
/// here: it is per-site (drop the row / AND the probe flag / empty the range)
/// and the caller branches on `k.cmp` before calling. The first draft folded
/// it in as an `Option` return, which put the per-site rule inside the layout
/// after the corollary above says it must stay out.
pub(crate) fn null_key_slots(ty: Ty) -> Vec<KeyBits>;

/// The `(false, type default)` pair for a NULL nullable map value. Called
/// only when `nullable` -- a non-nullable column with a NULL keeps its named
/// refusal at the call site (`duckdb/mod.rs:977-988`).
pub(crate) fn null_val_slots(ty: Ty) -> Vec<ScalarVal>;
```

```rust
// src/specializer/plan.rs -- JoinSpec's key record

/// One map key on the STATIC side of a [`JoinSpec`]: where the build side
/// reads it, and how it compares. Replaces the parallel
/// `key_cols: Vec<KeyCol>` + `key_indf: Vec<bool>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JoinKey { pub src: KeySrc, pub cmp: KeyCmp }

/// Where a key's build-side value comes from. (Today's `KeyCol`, renamed so
/// `JoinKey` can own the name.)
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum KeySrc {
    /// A lane of the static table (index into `StaticTable::cols`).
    Lane(u32),
    /// "the struct node at this SEGMENT path is non-NULL" (TASK-133).
    Present(Vec<String>),
}
```

```rust
// src/specializer/mod.rs -- the boundary recipe

/// One map key at the BOUNDARY: where to read it out of a build row, and
/// its slot layout. Replaces four parallel vectors -- `key_cols`,
/// `key_indf`, `key_present`, and the key types the materializer used to
/// walk out of `StaticTy::Map` with an iterator skip.
#[derive(Debug)]
pub struct StaticKey {
    /// The column's SEGMENT path. When `present`, it names a struct NODE.
    pub path: Vec<String>,
    /// TASK-133: the key is the node's presence, not its value.
    pub present: bool,
    pub map: plan::MapKey,
}

#[derive(Debug)]
pub struct StaticVal { pub path: Vec<String>, pub map: plan::MapVal }

#[derive(Debug)]
pub struct StaticSpec {
    pub batch: bool,
    pub table: String,
    pub keys: Vec<StaticKey>,
    pub vals: Vec<StaticVal>,
}
```

DEVIATION from grounding B, which said `key_present` "disappears entirely (it
is `KeyCol::Present` again)". It cannot. `StaticSpec` stores RESOLVED PATHS,
not `KeySrc`s -- `prepare_opaque` maps `KeySrc::Lane(c)` through
`plan::lane_paths` at `src/specializer/mod.rs:174-185`, after which a `Present`
path and a `Lane` path are both just `Vec<String>` and are indistinguishable.
The bit survives as one field on `StaticKey` rather than a parallel
`Vec<bool>`, which is the win that was actually available.

DEVIATION from grounding B's sketch of `MapKey { path, src, cmp }` as one
record spanning both layers. Split into `JoinKey` (identity: src + cmp, lives
on `JoinSpec`) and `MapKey` (layout: ty + cmp, `Copy`) because the two have
different lifetimes and different consumers, and because keeping `MapKey`
`Copy` dissolves grounding complication 7 outright: `emit_probe` reads
`spec.key_indf[i]` today as a `bool` before calling `&mut self` methods
(`lower.rs:1965-1974`), and a `MapKey` that carried a `Vec<String>` path would
need a clone or an index-and-refetch there. `Ty` is `Copy`
(`src/specializer/ir/mod.rs:207`) and `KeyCmp` is a fieldless enum, so
`MapKey`/`MapVal` are `Copy` and the encoder loop is unchanged.

#### Migration list

Producers:

| site | change |
|---|---|
| `src/specializer/lower.rs:255-269` | `map_keys(&spec.keys, &spec.key_cols).iter().flat_map(\|k\| k.slots())`. The `.lane()` at `:264` and `:266` moves INTO `map_keys`; it does not disappear. |
| `src/specializer/lower.rs:270-283` | `map_vals(&catalog[spec.table].cols, &spec.val_cols).iter().flat_map(\|v\| v.slots())`. The `.lane()` at `:278` / `:280` likewise. |
| `src/specializer/lower.rs:145-157` | the duplicate `flat` closure is DELETED. Its two call sites take the free function with their own source: `:160` `map_vals(in_cols, &joins[0].val_cols)` (BatchMap), `:165` `map_vals(&catalog[joins[0].table].cols, &joins[0].val_cols)` (MultiMap). The closure's `.lane()` at `:151`/`:153` is the same erasure `map_vals` now owns. |
| `src/specializer/lower.rs:158-167` | `StaticTy::MultiMap.keys` uses `map_keys` + `slots()` too; the `k.ty.lane()` at `:164` moves into `map_keys`. **Behavior-preserving TODAY**: `lower.rs:135-141` still refuses any INDF key under `many`, so every key reaching this line has `cmp == Eq` and `slots()` returns exactly `[ty.lane()]` -- the identical vector. TASK-135 deletes that refusal and the line starts carrying pairs with no further edit. |
| `src/specializer/lower.rs:1963-1982` | `emit_probe` calls `MapKey::encode_probe` (see below) |
| `src/specializer/lower.rs:1627-1633` | `lower_many_loop` calls the SAME encoder. Also behavior-preserving today, by the same argument. |

The probe encoder is a method on `FB`, not on `MapKey`, because it emits IR
through `&mut FB`:

```rust
// src/specializer/lower.rs, on FB
/// Encode one probe-side key into its flattened slots. Returns the slot
/// values plus, for an `Eq` key only, the validity flag the CALLER must
/// fold (into `keys_valid` for a map probe, into the range gate for the
/// fan-out loop) -- the plain-key NULL rule is per-site and stays out of
/// the layout.
fn encode_probe_key(&mut self, k: MapKey, lane: Lane, out: &mut Vec<Value>)
    -> Option<Value>;
```

Consumers:

| site | change |
|---|---|
| `src/duckdb/mod.rs:766-772` (`materialize_map`) | the `key_tys` / `val_tys` parameters are DELETED -- `spec.keys[i].map.ty` and `spec.vals[i].map.ty` carry them |
| `src/duckdb/mod.rs:1073-1079` (`materialize_statics`, the ONLY caller) | the call at `:1079` drops two arguments; the `keys, values` binding destructured out of `StaticTy` at `:1073` is then dead FOR THE CALL, but stays for the new arity assert below |
| `src/duckdb/mod.rs:817-824` | the 3-way nested zip becomes `for k in &spec.keys` |
| `src/duckdb/mod.rs:889-910` | `let _validity_ty = kt.next(); let ty = *kt.next()...` DELETED (`:893-894`). The `(false, default)` table at `:897-903` becomes `exec::null_key_slots(k.map.ty)`. |
| `src/duckdb/mod.rs:911-920` | the plain arm's `continue 'row` stays at the call site (per-site rule), driven by `k.map.cmp == KeyCmp::Eq`. Its `kt.next().expect("one type per plain key column")` at `:911` also goes -- see complication 8. |
| `src/duckdb/mod.rs:924` | value zip becomes `for v in &spec.vals` |
| `src/duckdb/mod.rs:959-976` | `let _validity_ty = vt.next()` DELETED (`:962-963`); the table at `:966-972` becomes `exec::null_val_slots(v.map.ty)` |
| `src/duckdb/mod.rs:977-988` | the non-nullable NULL refusal stays verbatim |
| `src/specializer/lower.rs:1550-1569` (`val_slots`) | becomes `plan::slot_pairs(&map_vals(src, &j.val_cols))`, where `src` is the existing `if spec.batch { self.in_cols } else { .. }` choice at `:1552-1556` |
| `src/specializer/lower.rs:1572-1590` (`val_flat_tys`) | becomes `map_vals(src, &j.val_cols).iter().flat_map(\|v\| v.slots())`, source chosen at `:1574-1578` as today |
| `src/specializer/lower.rs:830-848` (`SKind::StaticCol`) | unchanged |
| `src/specializer/exec/interp.rs:464-497` (`build_batch_rows`) | `in_decl` KEEPS its `&[(Ty, bool)]` type -- see below. Only the default table at `:485-491` changes, to `null_val_slots(ty)`. |
| `src/specializer/mod.rs:159-204` | the `StaticSpec` fold builds `StaticKey`/`StaticVal`. `StaticKey.map` is built from `j.keys[i].ty.lane()`, NOT `j.keys[i].ty` -- `map_keys` is the one constructor and it erases. |
| `src/specializer/plan.rs:203-240` | `KeyCol` -> `KeySrc`; `JoinSpec.key_cols: Vec<JoinKey>`, `key_indf` deleted |
| `src/specializer/tests.rs:1149-1150` | the `indf_*` trio reading `p.statics[0].key_indf` |
| `src/specializer/tests.rs:179-184` | `spec.key_cols` position lookup, `spec.key_present[kp]`, `spec.key_indf[kp]` all become `spec.keys[kp].path` / `.present` / `.map.cmp` |
| `src/specializer/tests.rs:818`, `:864`, `:874` | compare `StaticSpec::key_cols` as `Vec<Vec<String>>`; they become `spec.keys.iter().map(\|k\| &k.path)`. Not listed in the first draft. |

**The `frontend.rs` half of Seam B is roughly 20 sites, not one row.** The
first draft gave `ScopeJoin` a single line naming four index sites. Every
reader below destructures a `KeyCol` and must read `.src`, and three of them
are SIGNATURE changes the first draft never named. All compiler-caught -- that
is the point of the seam -- but the half-day-per-commit estimate rests on the
real number.

| site | change |
|---|---|
| `src/specializer/frontend.rs:1996-1998` (`ScopeJoin`) | `key_cols: Vec<JoinKey>`; the doc comment at `:1998` still says the dynamic keys are aligned with it |
| `:812`, `:826-827` | the empty-key `ScopeJoin` / `JoinSpec` literals: `key_cols: Vec::new()`, `key_indf: Vec::new()` collapse to one |
| `:847-850`, `:858-859`, `:895-896`, `:899`, `:906-907`, `:923-924`, `:934` | the `(keys, key_cols, key_indf, ..)` tuple threading in `bind_from`: two parallel vectors become one `Vec<JoinKey>` |
| `:938`, `:947`, `:960-961` | `val_cols_for` call, the `ScopeJoin` literal, the `JoinSpec` literal |
| `:1027`, `:1037-1038`, `:1064`, `:1099`, `:1102`, `:1110`, `:1115`, `:1121-1122` | the self-join path's second copy of all of the above; `:1122`'s `key_indf: vec![false; n]` becomes `cmp: KeyCmp::Eq` on each pushed `JoinKey` at `:1099` |
| `:1188-1211` (`head_hits_in_join`) | `KeyCol::Lane(ci) => Some(ci)` / `Present(_) => None` at `:1205-1206` read `k.src` |
| `:1230-1236` (`key_struct`) | same, `:1233-1234` |
| `:1242-1252` (`binds_in_join`) | same, `:1246-1250` |
| `:1384-1394`, `:1468-1471` (`bind_on`) | **SIGNATURE**: `-> Result<(Vec<SExpr>, Vec<KeyCol>, Vec<bool>, Vec<&'e SqlExpr>), _>` becomes `-> Result<(Vec<SExpr>, Vec<JoinKey>, Vec<&'e SqlExpr>), _>` |
| `:1604-1609`, `:1699` (`shared_key`) | **SIGNATURE**: `-> Result<Vec<(SExpr, KeyCol, bool)>, _>` becomes `-> Result<Vec<(SExpr, JoinKey)>, _>` |
| `:1706-1719` (`struct_keys`) | **SIGNATURE**: same change; `:1719`'s `KeyCol::Present(vec![st_sc.name.clone()])` becomes a `JoinKey` |
| `:1738-1746`, `:1784`, `:1793` (`walk_key_fields`) | **SIGNATURE**: `out: &mut Vec<(SExpr, KeyCol, bool)>` becomes `&mut Vec<(SExpr, JoinKey)>`; the `true` at `:1784` and the `Present` push at `:1793` fold into `JoinKey.cmp` |
| `:1832-1841` (`val_cols_for`) | **SIGNATURE**: `key_cols: &[KeyCol]` becomes `&[JoinKey]`; `:1838`'s `contains(&KeyCol::Lane(*c))` and `:1841`'s `if let KeyCol::Lane(c) = k` read `.src` |
| `:1853-1854` (`is_shadow_lane`) | `sj.key_cols.contains(&KeyCol::Lane(sj.val_cols[pos]))` reads `.src` |
| `:2742` | `position(\|k\| *k == KeyCol::Lane(ci))` reads `.src` |
| `:3924-3926` | `let KeyCol::Lane(ci) = sj.key_cols[key_pos] else` reads `.src` |
| `:4128-4130` | the same `position(..)` shape as `:2742`. Not listed in the first draft. |
| `:4365-4375`, `:4537-4538` | `for (kp, k) in sj.key_cols.iter().enumerate()` + `let KeyCol::Lane(ci) = k else { continue }` read `.src` |
| `src/specializer/mod.rs:180-190` | the two `plan::KeyCol::` matches in the `StaticSpec` fold read `.src`; the `key_indf: j.key_indf.clone()` at `:186` becomes `cmp` on each `StaticKey` |

**`in_decl` stays a `Vec<(Ty, bool)>`, and that is a DESIGN DECISION, not an
omission.** The first draft said it "already IS a `MapVal`, spelled as a
tuple". It is not, and the mistake is a role mistake rather than a shape one:
`in_decl` is the PROGRAM'S ROW SCHEMA, built from `p.in_cols` at
`interp.rs:259` and read by `check_input` on EVERY `run` (`:415`, `:419`,
`:422`, `:439`) for the arity, lane-type and validity-length checks of every
program, not only the batchmap ones. Retyping the field would drag a
prepare-time layout type into the runtime type-checker and contradict this
section's own corollary that the pair convention never escapes prepare time --
and complication 10 forbids `exec` importing `plan` at all, which is what
`&[MapVal]` on `build_batch_rows`'s parameter would require.

So the scope is narrower than the first draft's, and narrower than the obvious
correction: nothing about `in_decl` changes, at the field OR the parameter.
What changes is only the fourth default TABLE at `:485-491`, which becomes a
call to the shared `null_val_slots(ty)`. The honest residue: the value-pair
SHAPE (`if nullable { validity, payload } else { payload }`) is still written
twice -- once at `duckdb/mod.rs:959-976`, once here -- because the two live on
opposite sides of a layering boundary. What collapses is the four typed-default
tables, which is the half that had to agree BY HAND and the half that fails
silently. The cost of the alternative (a `Vec<MapVal>` rebuilt per `run` at
`:303`) buys only the two-line branch, and it buys it by breaking the import
graph.

Unchanged, and load-bearing that they are: `StaticTy` keeps flat `Vec<Ty>`;
`KeyBits` (`src/specializer/exec/mod.rs:305-327`) and `canon_f64_bits`
(`:344-352`) are untouched; `cranelift.rs:1104-1118` still routes every
multiplicity program to the interpreter; `ProbeDesc`
(`src/specializer/exec/cranelift.rs:2078-2084`) still clones the flat
`Vec<Ty>`.

#### How this resolves the grounding's complications

1. **Two `Ty` roles conflated.** Made explicit by two types: `MapKey.ty` is the
   COMPARISON lane type (post-`promote_key`, possibly widened -- TASK-120's
   `key_is_lossy` at `frontend.rs:1826-1828` then mints a shadow VALUE lane for
   the same physical column), `MapVal.ty` is the COLUMN's. One physical column
   can appear in both vectors with different `Ty` and now says so in its type
   name, with the rule written once in each doc comment.
2. **Cranelift's scratch-slot sizing sums flat slots.** Unaffected, BECAUSE
   `StaticTy::Map` keeps flat `Vec<Ty>`: `max_keys` at
   `src/specializer/exec/cranelift.rs:1174-1184` is still `keys.len()` and
   still means "one 16-byte cell per SLOT". This is the concrete reason
   Non-goal 2 is a non-goal rather than a nicety -- a design where `StaticTy`
   held logical keys would have to change that arithmetic, and it fails
   silently (`stack_store` at `base = 16*i`, `:2057`), not loudly.
3. **`ProbeDesc` clones the flat vector per probe site.** `slots()` returns a
   fresh `Vec` and is a prepare-time call; it is never called inside the
   per-site loop that indexes `key_tys[i]` (`cranelift.rs:2060`).
4. **Four typed-default tables.** Three of the four -- the ones that must agree
   BY HAND across files -- collapse to two owned tables:
   `exec::null_key_slots` (as `KeyBits`; note `KeyBits` has no `Dec` variant,
   `exec/mod.rs:311-316`, so it genuinely is a different table) and
   `exec::null_val_slots` (as `ScalarVal`, shared by `materialize_map` and
   `build_batch_rows`). The fourth, `lower.rs:497-507`'s `default_of`, stays
   where it is: it is the general masking default, used by `masked` for every
   trapping instruction, not only for keys. It is reached from the key path
   through exactly one encoder, so probe-vs-build agreement becomes a
   two-methods-on-one-concept inspection instead of a cross-file comment.
   `kernels.rs:96-104` and `cranelift.rs:853-859` are a DIFFERENT seam and are
   not touched.
5. **`key_present` short-circuits the convert closure, not the slot walk.**
   `src/duckdb/mod.rs:837-839` returns `KeyBits::I1(true)` before the type
   dispatch, and that is only consistent because `present_key` independently
   builds an `SExpr { ty: Ty::I1, nullable: true }`
   (`src/specializer/frontend.rs:3899-3918`) so the declared key type happens
   to be `I1`. With `present` and `map.ty` on ONE record, the boundary can
   `debug_assert!(!k.present || k.map.ty == Ty::I1)` at construction -- a debug
   assert, not a refusal, so release behavior is byte-identical.
6. **`lower.rs:145-157` cannot call `val_flat_tys`** because `fb` is
   moved-borrowed at `:124` and `lower_many_loop` takes `&mut fb` at `:143`.
   Dissolved by making the slot walk a FREE function over `&[MapVal]` rather
   than an `FB` method; the duplicate closure is deleted, not moved.
7. **`emit_probe`'s `Copy` read.** Dissolved by `MapKey: Copy` (see the
   DEVIATION above).
8. **Deleting the iterator-skips deletes the only runtime key-arity
   cross-check, so the design puts one back deliberately.** Today the three
   `kt.next().expect(..)` / `vt.next().expect(..)` calls
   (`duckdb/mod.rs:894`, `:911`, `:963`) are the accidental guard that
   `StaticSpec` and the lowered `StaticTy` agree on how many slots the key
   tuple has: if the type vector runs short, the build panics on
   `"payload type follows validity"`. After the migration the types come from
   `spec.keys[i].map.ty` and NOTHING compares the two vectors.

   That matters because the failure is silent, not loud. `cmp_key`
   (`exec/interp.rs:1709-1724`) zips `stored` against `key_regs` and stops at
   the shorter of the two, returning `Ordering::Equal` when every compared
   position matched -- so a build tuple one slot short compares EQUAL on its
   prefix, and the join silently widens. That is Finding 1's failure class
   (WRONG ANSWERS, not a refusal) reappearing on the static side.

   The fix is one line, and it is in the Goals' assert budget:

   ```rust
   // src/duckdb/mod.rs, in materialize_statics, before materialize_map
   debug_assert_eq!(
       spec.keys.iter().map(|k| k.map.slots().len()).sum::<usize>(),
       keys.len(),
       "StaticSpec and StaticTy disagree on key slot arity"
   );
   ```

   plus the same for `spec.vals` against `values`. Debug-only, so release
   behavior stays byte-identical, and it is a strictly better guard than the
   `expect` it replaces: the `expect` only fired when the type vector was
   SHORT, this fires on either direction.

Two more, stated because the design deliberately does NOT change them:

- The refusal at `src/specializer/lower.rs:130-134` (one join per query under
  `many`) is a separate gate and stays.
- The two refusals TASK-135 will delete stay here, untouched:
  `src/specializer/lower.rs:135-141` and
  `src/specializer/frontend.rs:1674-1684`.

### Section 3: `PrepareRequest` and `Prepared::input_lanes()`

#### Current contract

`prepare_opaque(sql, this_name, in_cols, opaque, structs, statics, many, udfs,
models, bind_eval) -> Result<Prepared, PrepareError>`, forwarding nine of ten
arguments verbatim to `frontend`, which returns them refracted through a
7-tuple whose only consumer is the caller. `prepare` is the 4-arg adapter with
31 test callers and no production caller. Lane assembly happens in three places
(`prepare_opaque` for the IR, `DuckDBInferFn::new` for the boundary,
`Binder::present_key` for the index), described in Section 1.

#### The new interface

```rust
// src/specializer/mod.rs

/// Everything stage 1 binds against. All slices, never `Vec`s: `Binder<'a>`
/// holds `Cow<'a, [Col]>` and five `&'a [T]` (frontend.rs:2008-2079), and
/// `&'_ PrepareRequest<'a>` can only reborrow at `'a` for fields that are
/// themselves `&'a [T]`.
pub struct PrepareRequest<'a> {
    pub sql: &'a str,
    /// The driving table's name, as the SQL spells it.
    pub this_name: &'a str,
    /// Caller-supplied input columns: plain scalars in declaration order,
    /// then struct leaf lanes. NOT the full lane list -- minted lanes are
    /// appended by `prepare_opaque` and come back on `Prepared`.
    pub in_cols: &'a [ir::Col],
    /// Row-model columns with no scalar lane, as (model position, name).
    /// Refuse on REFERENCE, not on construction.
    pub opaque: &'a [(usize, String)],
    pub structs: &'a [plan::StructCol],
    pub statics: &'a [plan::StaticTable],
    /// shape='many': joins lower as multiplicity loops. Stays a `bool` --
    /// see Non-goals on why `Shape` does not belong here.
    pub many: bool,
    pub udfs: &'a [ir::ExternSpec],
    pub models: &'a [plan::ModelTable],
    /// TASK-101: the udf callables, decl-order-aligned with `udfs`, for the
    /// bind-time fold of pure externs. Empty disables the fold.
    pub bind_eval: &'a [exec::ExternImpl],
}

impl<'a> PrepareRequest<'a> {
    /// The no-statics, no-opaque, no-udfs case -- what `prepare` spells.
    pub fn new(sql: &'a str, this_name: &'a str, in_cols: &'a [ir::Col]) -> Self;
    pub fn with_statics(self, statics: &'a [plan::StaticTable]) -> Self;
}

pub fn prepare_opaque(req: &PrepareRequest<'_>) -> Result<Prepared, PrepareError>;

/// Unchanged 4-argument shape: 31 test call sites keep working verbatim.
pub fn prepare(sql, this_name, in_cols, statics) -> Result<Prepared, PrepareError>;

impl Prepared {
    /// Every input lane, in IR order: caller lanes first, then the lanes the
    /// binder minted. `input_lanes()[i].col() == program.in_cols[i]` for
    /// every `i` -- the program's list is the projection, this one is the
    /// authority.
    pub fn input_lanes(&self) -> &[plan::InputLane];
}
```

```rust
// src/specializer/frontend.rs -- the tuple becomes a named record
/// What binding produced. Three of these (`out_cols`, `regexes`,
/// `wide_outputs`) are pure conduits: prepare never reads them, it forwards
/// them to `lower` or straight onto `Prepared`.
pub struct Bound {
    pub rel: Rel,
    pub joins: Vec<plan::JoinSpec>,
    pub out_cols: Vec<Col>,
    pub regexes: Vec<super::ir::ReSpec>,
    pub wide_outputs: Vec<super::WideOut>,
    pub model_refs: Vec<u32>,
    /// Lanes the binder minted (TASK-133: struct-key presence; TASK-134: a
    /// second kind). The caller APPENDS these to the lane list before
    /// lowering. Same name as `Binder::minted_lanes`, which is the same list
    /// moved out of its `RefCell` -- one list, one name, in both places.
    ///
    /// WRITE-SIDE INVARIANT, which `LaneKind` does NOT enforce and which is
    /// therefore written here: ONE vector, APPEND-ONLY, DEDUPED BY PATH
    /// ACROSS KINDS, INDEXED BY POSITION, and OFFSET BY `in_cols.len()`.
    /// `present_key` mints `SKind::Col((self.in_cols.len() + idx) as u32)`
    /// where `idx` is a position in this vector (`frontend.rs:3882-3900`), so
    /// a second minter that pushed into a SECOND vector would collide on
    /// `idx`, and a second minter that deduped only within its own kind would
    /// mint two lanes for one path. `LaneKind` makes the three READERS
    /// exhaustive; it says nothing about the writer. That asymmetry is the
    /// honest limit of Seam A, and TASK-134's minting sibling has to respect
    /// this paragraph rather than a type.
    pub minted_lanes: Vec<plan::InputLane>,
}

pub fn frontend(req: &PrepareRequest<'_>) -> Result<Bound, PrepareError>;
```

Both `#[allow]`s at `frontend.rs:332` go away.

#### Migration list

| # | site | change |
|---|---|---|
| 1 | `src/specializer/frontend.rs:332-358` | signature and return type; construction at `:625-633` becomes a `Bound` literal |
| 2 | `src/specializer/frontend.rs:426-428` | the `bind_from(...)` call, 10 positional args |
| 3 | `src/specializer/frontend.rs:640-651` | `bind_from` takes the request (its `'a` already unifies every borrow -- the struct is a rename of an existing constraint, not a new one) |
| 4 | `src/specializer/frontend.rs:722-750` | `Binder` construction reads from the request; `:749` is the `minted_lanes` rename (Section 1) and `:2078` the field declaration |
| 5 | `src/specializer/mod.rs:105-112` | `prepare` builds a `PrepareRequest` and calls through |
| 6 | `src/specializer/mod.rs:121-139` | `prepare_opaque(req)`; the `frontend` call destructures `Bound` |
| 7 | `src/specializer/mod.rs:141-148` | lane assembly (Section 1's W-new); `lower` still takes `&[Col]` |
| 8 | `src/specializer/mod.rs:76-99`, `:205-215` | `Prepared`: `present_lanes` out, `input_lanes` in |
| 9 | `src/duckdb/mod.rs:1633-1644` | request construction instead of ten positional arguments |
| 10 | `src/duckdb/mod.rs:1731-1740` | deleted (Section 1) |
| 11 | `src/duckdb/mod.rs:1744-1763` | `Marshaller::build` and `Engine::Compiled` take lanes |
| 12 | `src/duckdb/mod.rs:1830-1847`, `:1857-1866`, `:1897-1989` | one field instead of three (Section 1) |
| 13 | `src/duckdb/arrow.rs:213-232` | `ingest(py, batch, lanes)` (Section 1) |
| 14-25 | `src/specializer/tests.rs:161`, `:1173`, `:1245`, `:1361`, `:1450`, `:3164`, `:3476`, `:3586`, `:3621`, `:3705`, `:3732`, `:3798` | the 12 `prepare_opaque` sites become request literals. The four with a lone flag in a run of `&[]` (`:1173`, `:3621`, `:3705`, `:3732`) are the ones this exists for. |
| 26 | `src/specializer/tests.rs:168`, `:173`, `:174` | `.present_lanes` becomes `.input_lanes()`; `:174` changes SHAPE, not just name -- the exact three replacements are in Section 1. (`:172` is `let keyed = prep(..)` and is not a reader; the first draft listed it and omitted `:174`.) |

**Untouched:** the 31 `prepare(..)` sites in `tests.rs` (lines 26, 62, 107,
196, 620, 630, 641, 806, 842, 861, 872, 887, 954, 977, 987, 1137, 1198, 1954,
1965, 1992, 1998, 2909, 2919, 2956, 2967, 3242, 3690, 3885, 3911, 3924, 3944);
all 40 pytest files under `packages/confit/tests/` (43 `.py` including
`conftest.py`, `_native_guard.py` and `known_divergences/_helpers.py`);
`packages/confit/fuzz/*.py`
(Python, entering through `DuckDBInferFn`). There is no `benches/`, no
`tests/*.rs`, and no `[[bench]]`/`[[test]]` in `Cargo.toml`, so no other entry
point touches this seam.

#### How this resolves the grounding's complications

- **(a) Lifetimes.** All fields are slices under one `'a`, which is what
  `bind_from<'a>` already requires (`frontend.rs:640-651`). `select` / AST
  borrows are on an independent anonymous lifetime and do not interact. Nothing
  in the request is `Py<PyAny>`, so it is GIL-agnostic.
- **(b) The discarded arrow type.** `src/duckdb/mod.rs:1488` is
  `RowField::Opaque(_) => opaque.push((pos, name))`, throwing away the arrow
  spelling that `schema.rs:171-175` produced -- while the static side keeps it
  (`StaticTable::opaque: Vec<(String, String)>`, `plan.rs:38`) and `shared_key`
  uses it to name the refusal (`frontend.rs:1620-1624`). Re-verified: the row
  arm of `shared_key` can only say `"a non-scalar type on row table '{}'"`.
  This spec does NOT fix it -- carrying a field nobody reads is speculative.
  TASK-134 needs it, and with the request struct that is ONE field's type
  changing plus its 11 use sites, instead of the same edit plus a 10-argument
  signature churn at 14 call sites.
- **(c) Positional test callers.** Migrated; that is item 14-25 and the
  concrete payoff.
- **(d) `Prepared` derives `Debug`, not `Clone`.** `ir::Col` is
  `Clone + PartialEq + Eq + Debug` (`ir/mod.rs:277`), so `Vec<InputLane>` is
  free. `Prepared` stays `Debug`-only.
- **(e) `Shape` would change behavior.** Resolved by not doing it -- see
  Non-goals. `one_row_blocker` stays data on `Prepared` and `strict_map` stays
  in `DuckDBInferFn::new`.
- **(f) Two names for one list.** Accepted and made explicit:
  `Prepared::input_lanes()` is the authority, `Program::in_cols` is its
  projection (kept because `ir::print`/`parse` round-trip and `ir::verify`
  need it), and a `debug_assert` in `prepare_opaque` ties them -- as
  DOCUMENTATION; Section 1 complication 11 says why it is not the guarantee.
  Deleting `Program::in_cols` is out of scope for a behavior-preserving
  change. The minted sublist does NOT get a third name: `minted_lanes` on the
  `Binder` and on `Bound`, both, per Section 1 complication 12.
- **(g) The minted name is user-reachable and unpinned.** Preserved verbatim;
  see ASK 2.
- **(h) Self-joins and non-scalar row columns are mutually exclusive**
  (`frontend.rs:776-780`, `:1001-1005`), which is why present lanes never
  coexist with the `n_plain`-slicing self-join path. Unchanged here; TASK-134
  inherits the refusal for free as long as that gate stays keyed on
  `!opaque.is_empty()`.

## Consumers

### TASK-134 delta

What 134 adds ON TOP of these seams. Everything below is 134's diff, not this
spec's.

1. **One `LaneKind` variant -- plus a derive, plus a borrow at every reader.**
   ```rust
   /// Servable as a join KEY, never as a value (TASK-134): the path ends at
   /// a scalar the row vocabulary refuses to serve, ingested at its exact
   /// physical width into a comparison lane. `src` is the ARROW type, which
   /// `ColTy` does not carry and which both the ingest conversion and the
   /// per-type refusal are functions of.
   KeyOnly { cmp_ty: ColTy, src: ArrowKind },
   ```
   Because Section 1 made the boundary branches exhaustive matches, adding
   this variant is a compile error at `src/duckdb/arrow.rs:229`,
   `src/duckdb/mod.rs:1283` and `:1978` -- the three sites that must grow an
   arm -- instead of silently taking the `Value` path.

   Stated honestly, because the first draft called this "one variant" and it
   is not: TASK-134's ticket also says timezone-carrying types need their own
   answer or a named refusal, and a timezone is a `String`. The moment
   `KeyOnly` carries one, `LaneKind` loses its `Copy` derive, and the three
   sites Seam A just specified as by-value `match lane.kind` become
   `match &lane.kind`. So the delta is: **add a variant, drop the `Copy`
   derive when it carries a `String`, re-edit the borrow at every reader the
   compiler flags.** All three steps are compiler-caught and mechanical, and
   the payoff below is still real -- but it is a smaller payoff than "one
   variant" implied. If `ArrowKind` can stay a fieldless enum (a timezone
   refused by name rather than carried), the derive survives and the delta
   really is one variant; that is 134's call, not this spec's.
2. **The row-side arrow type.** `PrepareRequest.opaque` becomes
   `&'a [(usize, String, String)]` (or a named record), fed from
   `src/duckdb/mod.rs:1488`. One field on the request; its 15 references move.
   The first draft said 11 and listed 12; the tree has 15:
   `frontend.rs:337` (the `frontend` parameter), `:427` (the `bind_from` call
   -- the exact 10-positional-argument site this seam exists to kill), `:644`
   (the `bind_from` parameter), `:688`, `:698`, `:729` (the `Binder` field
   init), `:776`, `:1001`, `:1643`, `:2025` (the `Binder` field declaration),
   `:2684` and `:2685` (two statements in the star-expansion loop, not one),
   `:4193` (`this_col_with_fields`), `:4325` (`Binder::column`), and `:4479`
   (`Binder::qualified` -- a THIRD copy of the "row column has a non-scalar
   type" refusal, listed in no earlier draft or audit).
3. **The landing site is one arm.** `frontend.rs:1673`,
   `(Side::Opaque(w), _) | (_, Side::Opaque(w)) => cannot(w)`, splits into
   "admitted arrow type -> mint a `KeyOnly` lane" and "everything else ->
   the existing named refusal", which is the `cannot` closure at
   `:1667-1671` (AC #4). (`:1671` is that closure's `};` and `:1666` is the
   shared-name early return's; the first draft cited both off by one.)
4. **A minting sibling of `present_key`** (`frontend.rs:3881-3903`), pushing
   into the SAME `minted_lanes` vector, with the same dedup-by-path and the
   same `in_cols.len() + idx` forward index. Key-only and presence lanes then
   INTERLEAVE in the appended region in mint order -- which is precisely the
   event `present_from` cannot survive and `LaneKind` does. It must mint
   LAZILY, for the same +22..26 ns/row/lane reason (`frontend.rs:2068-2077`);
   minting eagerly at schema parse would break
   `n_plain = len - sum(leaf_count)` (`:656`) and all five `in_cols[..n_plain]`
   scans with it.

   The four write-side obligations -- one vector, append-only, deduped by path
   ACROSS kinds, indexed by position, offset by `in_cols.len()` -- are the
   invariant paragraph on `Bound::minted_lanes` in Section 3, not something
   the type checks. Two vectors would collide on `idx`; per-kind dedup would
   mint two lanes for one path. 134 owes this a reading, not a compile.
5. **Star invisibility (AC #3)** is inherited, not re-argued: a key-only lane
   is minted after `n_plain` is computed, so `frontend.rs:2683-2698` cannot see
   it, and the column stays in `binder.opaque` so `Binder::column`
   (`:4192-4199`, `:4324-4331`) keeps producing `"row column '{n}' has a
   non-scalar type"`. The lane and the refusal coexist on the same column.
6. **Ingest and row conversion, per admitted type.** `src/duckdb/arrow.rs`'s
   `KeyOnly` arm reuses `walk_lane` and `valid_at` (`:174-209`, `:275`) and
   adds the buffer read (timestamp[us] -> i64 direct, date32 -> i32 widened,
   float32 -> f64, uint64 -> bit-reinterpreted), plus the row-count check and
   the `null_seen && !nullable` check (`:363-368`) that the `Present` arm
   skips. Both row boundaries grow the matching Python conversion.
   `src/duckdb/mod.rs:843-887`'s static-side `convert` grows the mirror.
7. **`StructNode::KeyOnly(u32)`** beside `StructNode::Opaque`
   (`plan.rs:177`), so `walk_key_fields`'s Opaque arm (`frontend.rs:1798-1805`)
   can pair struct fields while struct-star (`:2790-2802`) and
   `this_col_with_fields` (`:4171-4176`) keep refusing.
8. **Nothing below `Prepared`.** `lower.rs`, `ir/verify.rs`, `ir/print.rs`,
   `exec/*` are untouched: a key-only lane is an ordinary typed input column
   in the IR, exactly as a presence lane is.
9. **Test flips.** `packages/confit/tests/test_join_keys.py:401-434`
   (`_OPAQUE_SHARED`) flips per admitted type; `pa.list_(pa.int64())` at `:406`
   stays a refusal (AC #4); `:456-465`
   (`test_a_struct_key_with_an_unlaneable_field_refuses_by_name`, a struct
   field with a TIMESTAMP leaf) flips too, same code path. The first draft
   said `:455-470`, which reaches into
   `test_a_struct_key_with_a_dotted_field_name_refuses_by_name` at `:468-475`
   -- and that one must NOT flip: a dotted field name is a path-encoding
   limit, which item 7 above keeps refusing, not a type-vocabulary one.

Checkable payoff: without this spec, item 1 is instead "replace `present_from`
with a per-lane tag at three sites, in the same diff as the per-type equality
proofs"; item 2 is instead "change one of ten positional arguments at 14 call
sites"; and item 4 has to re-derive that the threshold encoding is untenable.
Item 1's derive-and-borrow churn is owed either way, so it is not part of the
payoff.

### TASK-135 delta

What 135 adds on top. Three call sites, all in `lower.rs`:

1. **`MapKey::slots()` at `lower.rs:164`** so `StaticTy::MultiMap.keys` carries
   the same pairs `StaticTy::Map.keys` does. Section 2 already routes this line
   through `MapKey::slots`, so 135 changes NOTHING here -- it starts carrying
   pairs the moment step 3 admits INDF keys.
2. **`encode_probe_key` at `lower.rs:1627-1633`**, replacing the plain-only
   loop -- the same encoder `emit_probe` uses at `:1963-1982`. Section 2
   already substitutes it. What 135 must add is the range-gate exception: the
   empty-range forcing at `lower.rs:1642-1662` must fire for `Eq` keys and
   must NOT fire for `NotDistinct` keys (a NULL INDF key searches the
   `(false, default)` bucket instead of yielding an empty range). The encoder's
   return type is already `Option<Value>` = "the `Eq`-key flag the caller must
   fold", so the loop folds only what it is handed. Today it folds EVERY flag
   unconditionally (`:1630-1632`); that is the one real behavioral change in
   the loop, and it is a deletion.
3. **Two refusals deleted, plus one dead parameter.**
   - `src/specializer/lower.rs:135-141` -- `"IS NOT DISTINCT FROM join keys
     under shape='many' (params joins are the map/filter shapes)"`. Closes
     AC #3.
   - `src/specializer/frontend.rs:1674-1684` -- the `(Side::Struct,
     Side::Struct)` guard naming the struct column. Closes AC #2. Its pin at
     `packages/confit/tests/test_join_keys.py:523-532` inverts from refuse to
     serve.
   - With that guard gone, `shared_key`'s `many: bool` parameter
     (`frontend.rs:1604-1609`) has no reader; both call sites (`:883`, `:916`)
     drop the argument.

What 135 does NOT need, and this spec is why it does not:

- **No new comparison form.** `cmp_key` (`interp.rs:1709-1724`) and the
  `partition_point` pair (`interp.rs:1624-1631`) are already positional over
  flat `KeyBits` and are correct for the pair encoding as-is. The ticket title
  says the loop "learns NOT-DISTINCT comparison"; what it learns is the
  ENCODING plus the range-gate exception.
- **No build-side change.** `materialize_statics` already routes
  `StaticTy::MultiMap` through `materialize_map` with the identical key/value
  type vectors (`src/duckdb/mod.rs:1073-1079`), and `materialize_map`'s INDF
  arm is already exercised by the map shape. It starts working for `many` the
  moment the key vector carries pairs.
- **No presence-lane change.** Minted lanes are appended input columns before
  `lower` runs, so they reach the loop as ordinary `SKind::Col`.
- **No cranelift change.** `cranelift.rs:1104-1118` keeps every multiplicity
  program on the interpreter.
- **No `MapVal` change.** The loop's value half already handles nullable pairs.

Checkable payoff: with `MapKey`, TASK-135 is two `if`s deleted (one per file)
plus one fold narrowed, and the pair rule is written ZERO more times. Without
it, 135 writes the rule a tenth and eleventh time, in the one path where build
side and probe side are in different layers and no test compares them.

## Alternatives considered

### Seam A: the input-lane kind

**Context.** Two lane kinds today, three after TASK-134, distinguished by an
index threshold stored three times and maintained at three independent sites.

**A0. Do nothing.**
- Pro: zero diff, zero regression risk, the gate stays trivially green.
- Con: TASK-134 must replace the threshold anyway (it cannot express three
  kinds), so the work is not avoided, only merged into a diff that also carries
  per-arrow-type ingest conversions and an equality-proof matrix. The
  temporal coupling at `mod.rs:1731`/`:1736`/`:1737` and the unchecked
  duplicate append survive into that diff as live hazards.
- Verdict: rejected. Not doing it is not cheaper, only later and riskier.

**A1. Keep `present_from`, add a second threshold `key_only_from`.**
- Pro: smallest possible diff for TASK-134; no new type; the branch shape
  (`i >= X`) is already understood everywhere.
- Con: it is only correct if BOTH minted kinds are contiguous suffixes, and
  they are not -- both mint lazily from the binder, so they interleave in mint
  order. Making them contiguous means sorting the minted region, which breaks
  the `in_cols.len() + idx` index `present_key` already baked into
  `SKind::Col`. A fourth kind needs a third threshold.
- Verdict: rejected. It is wrong, not merely ugly.

**A2. `struct InputLane { name, path, kind }` with `enum LaneKind` (CHOSEN).**
- Pro: one construction site kills the duplicate append and the temporal
  coupling by deletion; three `Engine::Compiled` fields become one; the
  boundary branches become exhaustive matches, so TASK-134's variant is a
  compile error at exactly the three sites that must handle it; `Present`
  carrying no `ColTy` makes `push_present`'s `unreachable!` structural.
- Con: ~26 call sites move (the migration tables above are the count that
  matters, not this line); `Prepared` and `Program::in_cols` are two views of
  one list, tied by DELETING the second producer -- the debug assert only
  documents that (complication 11); the `in_cols.len() + idx` forward
  reference in the binder survives, and so does the whole write-side
  minting convention (complication 12).
- Verdict: chosen.

**A3. Generic `InputLane<S>` over the segment type** (`S = String` for the
engine and arrow, `S = Py<PyString>` for the marshaller).
- Pro: one type for both, no interned mirror.
- Con: generic plumbing through `arrow::ingest`, `Marshaller`, `Engine`, and
  `Prepared` to save one `Vec` per query at build time -- a compile-once engine
  paying in type complexity for a prepare-time allocation nobody measured. The
  path is already duplicated three ways today and nobody has complained.
- Verdict: rejected as over-engineering. Factoring the kind OUT of the path
  (A2) gets the invariant for free: the marshaller keeps `in_names` and reads
  `lane.kind` from the borrowed lane list.

**Recommendation: A2.**

### Seam B: the map slot pair

**Context.** One rule written nine times, four typed-default tables that must
agree by hand, and an iterator-skip (`let _validity_ty = kt.next()`) as the
only thing tying the build side to the layout the lowering declared.

**B0. Do nothing.**
- Pro: zero diff.
- Con: TASK-135 writes the rule a tenth and eleventh time, in the path where no
  test compares the two sides. The four default tables stay unchecked.
- Verdict: rejected.

**B1. Put `MapKey` in `StaticTy` (make the IR carry logical keys).**
- Pro: the pair rule would exist in exactly one place, and no `slots()` call
  would be needed at all.
- Con: `StaticTy` is a serialized TEXT format with a round-trip law --
  `print` emits `map(i1, i64) -> (...)` (`ir/print.rs:19-27`), `parse` reads a
  flat type list (`ir/parse.rs:719-794`), and `prepare` canonicalizes so the
  law holds (`mod.rs:151`) while three tests assert it
  (`exec/tests.rs:1270`, `ir/tests.rs:32`, `tests.rs:313`). This breaks every
  `.ir` fixture,
  the IR fuzz generator (`ir/gen.rs:214-233`, `:642-655`), and cranelift's
  scratch-slot arithmetic (`cranelift.rs:1174-1206`, which sizes `16 *
  keys.len()` and would silently undersize). It is not behavior-preserving.
- Verdict: rejected, and the reason is written into Non-goal 2 so the next
  reader does not re-derive it.

**B2. `MapKey`/`MapVal` as layout types in `plan`, feeding `StaticTy` (CHOSEN).**
- Pro: `slots()` is the one home; `StaticTy` and therefore the IR text form,
  the fuzz generator and the cranelift arithmetic are all untouched; the three
  iterator-skips delete; `materialize_map` loses two parameters; three of four
  default tables collapse to two owned ones; `StaticSpec`'s five parallel
  vectors become two records; both `Copy`, so no encoder-loop borrow churn.
- Con, the second one: deleting those skips deletes an accidental arity
  cross-check, so the design owes a `debug_assert_eq!` back (complication 8),
  and the value-pair SHAPE stays written twice because `build_batch_rows` is
  on the far side of the `exec` / `plan` layering boundary.
- Con: `encode` is genuinely two methods, not one -- the build side produces
  `KeyBits`/`ScalarVal` inside a pyo3 `PyResult` with `continue 'row` control
  flow, the probe side emits `Inst::Select` through `&mut FB`. They are
  provably parallel by inspection, not by construction.
- Verdict: chosen.

**B3. One `encode()` over a trait abstracting "value sink".**
- Pro: probe and build could not drift, by construction.
- Con: an interface with two implementations that share no error channel, no
  value type and no control flow, introduced to unify eleven lines with eleven
  other lines. The trait would have to abstract `PyResult` + `continue 'row`
  against `&mut FB` + SSA. This is the abstraction that reads well in a design
  doc and is decoded at 3am.
- Verdict: rejected.

**Recommendation: B2.**

### Seam C: the prepare surface

**Context.** Ten positional arguments with 14 callers, a 7-tuple with one
consumer, and lane assembly done three times.

**C0. Do nothing.**
- Pro: zero diff, and the surface is Rust-internal -- no user ever sees it.
- Con: the four test sites with a lone `true` in a run of `&[]` stay a
  transposition waiting to happen, and TASK-134 must change one of ten
  positional arguments at 14 call sites.
- Verdict: rejected, but honestly the weakest of the three rejections -- see
  ASK 3.

**C1. Split `frontend` into smaller functions so no tuple is needed.**
- Pro: attacks the real cause (a god function returning seven things because it
  does seven things).
- Con: that is the `frontend.rs` split, explicitly a non-goal, and it cannot be
  behavior-preserving-by-inspection at 4500 lines. Doing it in the same change
  as A and B would make all three unreviewable.
- Verdict: rejected for this spec; the split remains a real, separate finding.

**C2. `PrepareRequest<'a>` of slices + a named `Bound` return (CHOSEN).**
- Pro: named fields at 12 test sites; both `#[allow]`s at `frontend.rs:332`
  delete; `Prepared::input_lanes()` gives Section 1 somewhere to live; TASK-134's
  `opaque` widening becomes a one-field edit; the 4-arg `prepare` keeps its
  shape so 31 test sites do not move.
- Con: 26 sites move for zero behavior change; the request is a rename of an
  existing lifetime constraint, so it buys clarity, not safety; `Shape` stays
  un-unified, which some readers will find half-done.
- Verdict: chosen, with the smallest scope that carries Section 1.

**Recommendation: C2, landed last, and droppable (ASK 3).**

## Proposed additions to docs/properties.md

Drafted to match `docs/properties.md`'s house style for the Engine (Confit)
section, numbered after P20. These are PROPOSALS inside this spec;
`docs/properties.md` is not edited by this change.

House-style notes, so adoption is a copy and not a rewrite. P18/P19/P20 are
each 3-5 lines with the pin named INLINE, not in a footer; the file's `*Spec:*`
footers are real house style but belong to the P1-P17 sections, so P21/P22
carry one `*Spec:*` line and no `*Pinned (proposed):*` line. No `---` rule
separates laws -- the file's only rules are at `properties.md:11` and `:246`,
and neither is a law separator. Every law separates its label from its title
with an EM-DASH; this spec file is ASCII-only, so the two below use `--`, and
the em-dash is restored when the text is adopted into `properties.md` (which
is itself unicode).

**P21 -- A masked payload is a SAFE constant, and the same one on both sides.**
Wherever a value rides beside a validity flag -- a nullable map value, an IS
NOT DISTINCT FROM key, a NULL extern return, a static load under a false flag
-- the payload under `valid = false` is a constant the op accepts: the type
default (`false` / `0` / `0.0` / `""` / a zero at the column's declared scale),
or the op-specific safe constant `masked_to` names, and never the
un-normalized source register. That is what makes "evaluate then discard" safe
(computed garbage is unbounded), and it is what makes a NULL key ONE bucket:
probe and build store the identical `(false, constant)` pair, so `cmp_key` can
be positional over flat `KeyBits` and know nothing about NULLs. Owned per value
type, not per site -- `exec::null_key_slots`, `exec::null_val_slots`,
`FB::default_of`, `FB::masked_to`; a default written a fifth time is the bug
this law names. (`indf_*` in `specializer/tests.rs` pins the end-to-end
reading; the three tables agreeing type-for-type is not yet pinned.)

*Spec:* 2026-08-25-lane-and-slot-seams-design (Seam B).

**P22 -- Every IR instruction is reachable by the generator.** The
interpreter-vs-cranelift differential (P19, 500 seeds) guarantees only what the
generator can emit, so coverage is itself the invariant: every `Inst` variant is
either produced by `ir::gen::gen_program` or listed, by name and with a reason,
in a totality test that fails when a new variant is neither. Measured
2026-08-25 at `debaf8c`: 42 variants, 32 reachable, 10 not -- `Dtof`, `Itod`,
`StoiOpt`, `StofOpt`, `ReMatch`, `ReExtract`, `ReReplace`, `ExternCall`,
`ProbeRange`, `ProbeRead` -- four of them the narrowing/parsing conversions
where the backends are likeliest to disagree, so the hole is not uniformly
cheap. A claim about coverage that nothing counts is a hope. (Not yet pinned:
the totality test goes beside the differential in `exec/tests.rs`.)

*Spec:* 2026-08-25-lane-and-slot-seams-design (proposed properties).

**Why P21's wording changed from the first draft, since the change is the
whole point of it.** The draft said "the payload under `valid = false` is the
TYPE DEFAULT ... never whatever the source happened to hold". That is FALSE in
the tree. `FB::masked_to` (`src/specializer/lower.rs:481`, doc at `:478-480`)
exists precisely to mask to something the type default is not, and it is called
seven times: `:1156` `F64(1.0)` for `Ln|Log2|Log10`, `:1158` `F64(0.0)`,
`:1188` `F64(10.0)` and `:1189` `F64(1.0)` for the log pair, `:1226` and
`:1227` `Str("a")`, `:1298` `I64(0)`. Five of those seven name a constant that
is NOT `default_of`'s -- and the law's own rationale is why: `0.0` is itself in
`ln`'s trap domain, so "the type default" would be the unsafe choice at exactly
the sites the law exists to protect. The clause is not an exception bolted on;
it is the law stated correctly. (Checked and clean on the other side: `h_sload`
at `cranelift.rs:850-859`, `h_probe`'s miss path at `:934-945` and `LoadOpt` at
`:2005-2030` all write type defaults, so there is no cranelift-specific
violation.)

## ASK blocks

Only the decisions this design does not settle.

1. **The `Present` arm still does not check the node is a struct.** For a
   top-level presence lane the path has length 1, so `walk_lane`'s loop body
   never runs and `raw_array` is taken on whatever column the batch supplies;
   the lane then reads that column's validity bitmap. It is shadowed by the
   leaf lanes' "is {ty}, the schema declares a struct -- cast first" refusal
   except for a LEAFLESS struct (`pa.struct([])`, or a nested struct whose
   fields are all opaque). Adding the assertion is a new refusal, so it is
   outside this spec's gate. Ticket it as a follow-up, or fold it in and accept
   that "identical refusal messages" gains one exception?
   Trade-off: folding it in is three lines and closes a real hole now; ticketing
   it keeps this change's gate absolute, which is the only thing making a
   1,000-line refactor reviewable.

2. **`" (present)"` is observable and unpinned.** It reaches users through
   `"Row for table '{}' is missing attribute '{}'"` and
   `"infer_arrow: column '{}' has {} rows"`, and `grep "(present)"` finds it in
   exactly one place -- `frontend.rs:3888` -- with no test asserting it. Pin it
   as-is in this change (cheap, and it makes the refactor's "byte-identical
   messages" claim checkable), or leave it unpinned and let TASK-134 decide the
   minted-lane naming scheme for all three kinds at once?
   Trade-off: pinning now costs one test and locks a string we may want to
   change in 134; not pinning leaves the refactor's central claim untested at
   its most fragile point.

3. **Is Seam C worth 26 moved call sites for zero behavior?** Seams A and B
   each pay for themselves in a queued ticket. Seam C's payoff is thinner:
   `Prepared::input_lanes()` is needed by A and could live on today's
   `prepare_opaque` signature unchanged, and TASK-134's `opaque` widening is
   the same edit either way -- one field's type versus one of ten arguments,
   at 14 sites. What C buys is the four test call sites where a lone `true`
   sits in a run of `&[]`, and deleting two `#[allow]`s.
   Trade-off: landing it makes the surface honest and the next ticket's diff
   smaller; dropping it cuts roughly a third of this change's line count and
   all of its non-lane risk. If it lands, it lands LAST and alone.

4. **Adopt P21 and P22 into `docs/properties.md` now, or when their pins
   land?** They are not in the same state, and the first draft's "both are
   true today" was wrong on both halves.
   - **P21, with its safe-constant clause, is TRUE today** -- as an unenforced
     convention, held by inspection across the four default tables plus
     `masked_to`'s seven call sites. P21 as the draft FIRST worded it ("the
     type default ... never whatever the source held") was false, falsified by
     five of those seven; the redraft above is what is true.
   - **P22 is FALSE today**, with a measured 10-of-42 hole. It would enter as a
     law the code does not satisfy.
   Trade-off: writing a law before its pin is how "agreed direction, not yet
   law" (`properties.md:246-259`) exists as a section, and P22 belongs there
   rather than in the numbered list until the exclusion list exists. Adopting
   P22 as a numbered law immediately would mean either generating ten
   instruction kinds or writing the exclusion list first, which is a separate
   ticket's worth of work. P21 has no such blocker -- the only question for it
   is whether an inspection-held law is a law.

## Staging

Three commits, in this order, each landable ALONE with the full suite green
after it. Each is behavior-preserving on its own, so any one can be reverted
without touching the others.

1. **`InputLane` (Seam A).** The heaviest, the one both tickets need, and the
   one that pays for itself immediately by deleting
   `src/duckdb/mod.rs:1731-1740`. Land first so TASK-134 is unblocked even if
   the rest stalls. Suite green; the only test edit is
   `tests.rs:126-186` -- three lines (`:168`, `:173`, `:174`), of which `:174`
   changes shape rather than name, spelled out in Section 1.
2. **`MapKey`/`MapVal` (Seam B).** Independent of Seam A -- it touches
   `lower.rs`, `plan.rs`, `frontend.rs`'s key plumbing, `materialize_map` and
   `build_batch_rows`, none of which Seam A moves. Unblocks TASK-135. Suite
   green; the test edits are Rust unit tests naming `key_indf`
   (`tests.rs:1149-1150`, `:179-184`) and the three comparing `key_cols` as
   `Vec<Vec<String>>` (`:818`, `:864`, `:874`).
3. **`PrepareRequest` (Seam C).** Last, because it is the widest mechanical
   diff and the least load-bearing, and because it is the one a reviewer may
   decide to drop (ASK 3). It depends on Seam A only in that
   `Prepared::input_lanes()` already exists by then.

**Gate for each commit, in order:** `cargo test` for the crate; `uv run maturin
develop` then the full pytest suite from the repo ROOT (the maturin rebuild is
not optional -- `uv sync` does not rebuild the `.pyd` and pytest will silently
run the stale engine); then the same in a DEBUG build so `debug_asserts` runs,
because all three of the invariants this change installs are debug asserts --
and the third (key slot arity, Section 2 complication 8) replaces a check that
fires TODAY in release, so the debug run is the only thing that would catch its
regression. No fuzz
campaign is required: this change moves no accept/refuse boundary, and a
campaign that finds anything means the "behavior-preserving" claim was false,
which the suite should already have said.

Half a day per commit, plus the migration typing -- with the caveat that the
estimate rests on the migration lists being complete, and the first draft's
were not. Seam B in particular is ~20 `frontend.rs` sites and three signature
changes, not the one table row it originally had. Every one of them is
compiler-caught, so the risk is schedule, not correctness.
