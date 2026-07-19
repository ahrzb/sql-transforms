---
id: DRAFT-3
title: Rust-optimized serving inference path (fast at n=1)
status: Draft
assignee: []
created_date: '2026-07-19 01:09'
labels:
  - perf
  - serving
milestone: m-3
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the preprocessing fast at n=1: keep the dict/DataFrame off the request path entirely, parse request JSON in Rust into typed values, run native (non-fallback) transforms, and hand model.predict a single contiguous feature buffer (near-zero-copy numpy view) with no per-feature Python objects on either boundary. The payoff behind the serving thesis -- the functionality/parity work proves the vector is RIGHT; this makes it FAST. Depends on: the sklearn functionality & parity work (needs the parity harness as a correctness net) and the benchmark task (measure before optimizing). Why separate: correctness and representation-performance are different risks and sequence differently.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 native hot-path transformer impls on the Rust value representation -- no Python/pandas intermediate on the request path
- [ ] #2 contiguous feature-buffer output via the buffer protocol; contiguous typed input parsed in Rust; both boundaries object-free
- [ ] #3 low-level tactics (thread-local arena, GIL release/threshold, JSON parser) gated by the benchmark -- DataFrame deletion is the primary win, the rest only if a profile justifies it
<!-- AC:END -->
