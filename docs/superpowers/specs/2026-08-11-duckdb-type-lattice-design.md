# DuckDB's type lattice — Design (milestone m-8)

**Decided with AmirHossein 2026-08-11.** Three fuzz campaigns (55k seeds) put
the entire remaining finding family in one gap: the engine types in four
primitives while DuckDB types in a lattice. The dogma-shaped question ("keep
the 4-type limit?") resolved to a rule instead of a number:

> **Semantic typing is unconditional** — bit-for-bit means typing exactly as
> DuckDB types, for every query we serve. **Compute machinery is bought per
> type, by reachability** — and an entry point we haven't bought refuses by
> name, never approximates.

## The lattice and the strategies

```rust
enum Ty { I1, I32, I64, I128, Dec(u8, u8), F64, Str, Blob }
```

| type | compute strategy | why it is enough |
|---|---|---|
| `I32` | **erase**: bounded i64 | DuckDB's INT32 is checked-never-wrapping. `a op b` then `trap if result ∉ [-2^31, 2^31)` is bit-identical to real int32 for every operator it has — the width is observable ONLY as the trap threshold and the Arrow schema, both reproduced. Falsifiable; the fuzzer patrols it. |
| `I128` | **one new lane width** | existing integer instructions at a second width, not a new op set. Entry points: `sum`/`product` aggregates (build-time only — aggregation at inference is out of scope by founding decision), out-of-i64 literals, `::HUGEINT`. Hard dependency to verify before the phase: cranelift i128 legalization on x64. |
| `Dec(p,s)` | **scaled integers over i64/i128** | DuckDB stores DECIMAL as scaled ints with STATIC `(p,s)` — so every rescale factor is a build-time constant and decimal arithmetic lowers to integer arithmetic we have. It is NOT a second arithmetic tower; that earlier framing was wrong and was what justified refusing. Frontend carries the per-operator result-scale rules (measure first). |
| `Blob` | **byte string over the existing arena** | own compare/display; entry points are narrow (`repeat(NULL, n)` overload et al.). Last, demand-driven. |

`Ty::I32` (and each new member) is a REAL frontend type — every `match` on
`Ty` is forced by rustc to answer for it. That is the TASK-69 doctrine
("break the build, not the answers") applied to typing: a `duck_width`
side-channel field was considered and rejected as a second source of truth
that fails silently.

## Phase 0 measurements (2026-08-11, all reproduced by probe)

```text
fit-time aggregate Arrow types:
  avg/stddev/max(DOUBLE)   -> double        (safe today)
  count/min(BIGINT)        -> int64         (safe today)
  sum(BIGINT)              -> decimal128(38,0)   <-- the live entry point

confit today, decimal128 static column (ordinary fit-path output!):
  value 9007199254740993 (2^53+1)  ->  SERVES 9007199254740992.0
```

**That last line is a silent wrong answer on master** — the decimal static
narrows through f64 at ingest with no refusal, while known-limitations
claims out-of-range payloads reject. It violates "no third mode" from the
ordinary `sum`-in-fit path. Consequence: the decimal *ingest boundary* is
phase 1 of this epic, not phase 5 — as a REFUSAL first (exact-or-refuse),
upgraded to exact serving when the Dec lanes land.

## Phases (each certified by a 20k+ fuzz campaign before the next)

1. **Decimal static ingest: exact-or-refuse.** A decimal value that
   round-trips f64 exactly keeps today's conversion; one that doesn't
   refuses by name. Closes the phase-0 hole immediately, no lattice needed.
2. **`Ty::I32` typing + output schema.** Inference (literal→I32,
   `I32 op I32→I32`, function return widths per the measured catalogue,
   NULL typing, CASE unification), erased at lowering; `infer_arrow` emits
   int32 from the type. Closes TASK-79; un-refuses TASK-86's nullif face.
3. **INT32 bounds traps.** The erase strategy's trap half: DuckDB's named
   overflow errors at ±2^31 on I32-typed lanes. Closes TASK-84's residual.
4. **i128 lane + exact aggregates.** Verify cranelift i128 first; then the
   build-time `sum`/`product` accumulator and decimal128(38,0) boundary
   output; `::HUGEINT` and wide literals ride along. Upgrades phase 1's
   refusal to exact serving.
5. **`Dec(p,s)` arithmetic.** Measure DuckDB's per-operator result-scale
   rules; constant rescales over integer lanes; bare decimal literals stop
   being a documented divergence (`0.1 + 0.2 = 0.3`, DuckDB's answer).
6. **Blob.** When an entry point demands it.

## Gates

- **UDF vocabulary**: `takes`/`returns` today speak `bool/int64/double/
  string`. Growing that is user-facing API — its own approval round with
  concrete before/after cases, never folded into an engine phase.
- **known-limitations §3** rewrites when phase 2 lands: "computes in two
  numeric machine widths at row time; types in DuckDB's lattice".
- Refusals that survive the whole epic are exactly the non-type classes:
  non-constant regex (specialization), aggregation at inference (scope),
  builder budget (resource, TASK-88 — revisit once typed).

## Out of scope

Physical narrow storage (int32 arrays, SIMD lanes) — DuckDB needs it for
columnar bandwidth; sub-5k row serving is boundary-bound (measured, the
two-engine decision), and we ceded large-batch columnar to DuckDB.

## Phase 2 groundwork: the measured width catalogue (2026-08-11)

DuckDB's integer-width inference, probed directly — these become `Ty::I32`'s
inference rules, and the fuzzer patrols the catalogue for drift:

```text
int32:  integer literals, literal-only arithmetic, ascii, unicode, ord,
        abs(int32-typed arg), nullif(NULL, ..), coalesce(NULL, 1),
        greatest(1, 2), CASE unifying int32-only arms
int64:  BIGINT columns and anything they touch (k + 1), length, len,
        strpos, instr, position, levenshtein, bit_length,
        CASE unifying int32 with int64 (standard promotion)
int8:   sign()  <-- a THIRD narrow width; same erase strategy, the enum
        carries I8 (and I16 for completeness) from the start so the
        catalogue never forces a second enum change
double: round, floor (not integer-typed at all — no width question)
```

Notables: `abs` is width-POLYMORPHIC (follows its argument); the
length-family is fixed int64 despite string inputs; CASE promotes across
widths exactly like arithmetic.

## Known divergences are scaffolding, not contract

Each phase's definition of done includes DELETING that phase's markers, in
all three homes, in the same PR as the fix:

1. the xfail-strict pin flips to a real parity test (enforced — strict
   xfail turns XPASS-loud the moment the fix lands);
2. the known-limitations row goes (enforced — its executable twin breaks
   until doc and code agree);
3. the fuzzer's suppression tag (`KNOWN-TASK-79`, `decimal-literal` in
   `fuzz/oracle.py`) is REMOVED (unenforced — stated here because a tag
   that outlives its phase would silently swallow regressions in exactly
   the code the phase just changed; the certification campaign after the
   tag removal is what proves the class is gone rather than hidden).

What survives the whole epic is not a divergence list: refusals (named,
tested), multiset row order under `shape='many'` (order is not part of the
contract), and oracle-self-inconsistency corners (DuckDB disagreeing with
its own constant fold) where refusal is the only well-defined answer.
