---
id: TASK-67
title: >-
  Unaligned pyarrow buffer is undefined behaviour in the arrow ingest and model decode
status: To Do
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
- [ ] #1 An unaligned input buffer either works or raises a named Python error;
      it never aborts the process
- [ ] #2 Covers the request path (`infer_arrow`) and the construction path
      (`models=` decode)
- [ ] #3 A test builds a deliberately unaligned buffer and asserts the outcome
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
<!-- SECTION:NOTES:END -->
