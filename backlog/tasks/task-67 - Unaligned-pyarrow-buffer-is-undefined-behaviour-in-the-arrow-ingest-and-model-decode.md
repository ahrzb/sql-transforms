---
id: TASK-67
title: >-
  Unaligned pyarrow buffer is undefined behaviour in the arrow ingest and model decode
status: Done
assignee: []
created_date: '2026-08-07 22:40'
labels:
  - bug
  - safety
dependencies: []
documentation: []
type: bug
ordinal: 60000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`RawArray::data` (`packages/confit/src/duckdb/arrow.rs`) casts a raw pyarrow
buffer address straight to `*const T` and dereferences it with no alignment
check:

```rust
unsafe fn data<T: Copy>(&self, slot: usize) -> *const T {
    let (addr, _) = self.bufs[slot].expect("data buffer present");
    unsafe { (addr as *const T).add(self.offset) }
}
```

Arrow does not guarantee 8-byte-aligned value buffers. A numpy view starting at
a non-multiple-of-8 offset into a byte blob — `np.frombuffer(blob,
np.float64, offset=4)`, which is what you get parsing a packed binary record or
an mmap with a header — is zero-copied by `pa.array()`, and pyarrow itself
reads and computes over it perfectly.

Reachable on both paths:

- `infer_arrow`, per request (`arrow.rs` ingest arms);
- the `models=` decode at `DuckDBInferFn` construction (`ints`, `doubles`,
  `strings`).

**Severity is build-profile dependent, and the sweep corrected itself on this
point.** The debug build's `debug_assertions` misalignment check turns it into
an uncatchable abort — `thread caused non-unwinding panic. aborting.`, not a
Python exception, process gone. The **release build does not fault today**. But
the load is still UB: LLVM is free to autovectorise the `(0..rows)` loop into
aligned SSE/AVX moves, which would fault, so "release is fine" is a property of
today's codegen, not a guarantee.

Not independently reproduced by me — relayed from the sweep, which did
reproduce it on both backends and did stage a release build to check.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 An unaligned input buffer either works or raises a named Python error;
      it never aborts the process
- [x] #2 Covers the request path (`infer_arrow`) and the construction path
      (`models=` decode)
- [x] #3 A test builds a deliberately unaligned buffer and asserts the outcome
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two options: check `addr % align_of::<T>() == 0` and refuse by name, or use
`ptr::read_unaligned` throughout. The second serves more inputs; the first is
cheaper on the hot path. A per-request input batch must never be able to kill a
serving process, so whichever is chosen, the refusal path has to be an
exception rather than a panic.

Same pattern in every arm: ingest `I64`/`F64`/`Str` offsets, plus `ints`,
`doubles` and `strings` in the model decode.

## Resolution (2026-08-07): `read_unaligned` everywhere, behind a newtype

**Independently reproduced first** (this was the finding I had only relayed).
`np.ndarray(buffer=blob, offset=4)` through `pa.array()` is zero-copied, and a
debug build aborts on it — verified at three distinct sites: `arrow.rs:154`
(i64 ingest), `:205` (i32 string offsets), and the `models=` decode. Exit code
0xC0000409, non-unwinding, process gone. Release read them correctly.

Refusal was rejected as the remedy for exactly that reason: release serves
these inputs correctly today, so refusing them would regress working callers.

`RawArray::data` now returns `Unaligned<T>` — a newtype whose only reader is
`get(i)` doing `read_unaligned`, so the unchecked deref is not spellable at a
future call site. All typed reads were already `*p.add(i)` derefs rather than
slices, so nothing needed restructuring; `from_raw_parts` appears only on
`*const u8` (align 1) and on Rust-owned output `Vec`s.

**Measured cost on the ALIGNED path** (the one that must not regress),
`infer_arrow`, 8 f64 + 2 i64 columns, median of 3 runs:

| rows | plain deref | read_unaligned | delta |
|---|---|---|---|
| 1024 | 71.8 ns/row | 74.4 | +3.6% |
| 8192 | 41.7 ns/row | 43.5 | +4.3% |
| 65536 | 69.5 ns/row | 74.4 | +7.0% |

Real, small, and largest where it matters least (large-batch columnar is
ceded to DuckDB; the sub-5k band pays 3.6%). Accepted rather than optimised
away. If it ever needs recovering, the move is to bulk
`copy_nonoverlapping` the value buffer once per column — which has no
alignment requirement — and apply validity in a second pass, rather than to
branch on alignment and keep two loop bodies.

Folded in as free hardening, same "a raw address is not a Rust reference"
class: the two string-bytes reads used `raw.bufs[2].unwrap_or((0, 0))` and
handed a NULL pointer to `from_raw_parts`, which demands a non-null pointer
even at len 0. Now one `str_bytes` helper returning `&[]`. pyarrow refuses to
build a string array without a data buffer (`ArrowInvalid: Value data buffer
is null`), so this is unreachable today and has no red test — flagged rather
than claimed as a live bug.

Tests live in `packages/confit/tests/test_infer_arrow.py` and run the probe in
a SUBPROCESS: an abort kills the interpreter rather than raising, so it has to
be observed from outside — and "a request batch must never end the process" is
the actual contract. They pass trivially in release and are only red in debug.
<!-- SECTION:NOTES:END -->
