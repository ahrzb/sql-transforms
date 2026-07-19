---
id: DRAFT-4
title: Benchmark inference-path optimizations before building them
status: Draft
assignee: []
created_date: '2026-07-19 01:09'
labels:
  - perf
  - benchmark
milestone: m-2
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Candidate online-inference optimizations are currently HUNCHES, not measured wins: a thread-local bump arena for per-row scratch; extracting Python values into an owned Rust type and releasing the GIL (allow_threads) during compute; and -- the load-bearing one -- parsing request JSON in Rust so the dict/DataFrame never touches the request path. Building all three and attributing wins afterward is backwards; two are probably aimed at the wrong path (arena + GIL-release mostly help batch/throughput, not the single-object latency path we optimize for). Measure first. Stand up a baseline harness capturing the four corners: single-row latency DISTRIBUTION (p50/p99, not mean) and batch throughput (rows/sec), each single- and multi-threaded. Traps: (1) GIL-reacquire contention only appears with CONCURRENT callers -- a single-threaded microbench reports the handoff as free and misleads. (2) For the latency path, profile one infer(row) first -- expected top costs are the JSON/dict boundary + per-call setup not amortized to construction, NOT allocation; confirm/kill before touching the arena. (3) Releasing the GIL for one tiny object is expected to HURT p99 (handoff > compute) -- gate behind a batch-size threshold, prefer process-level parallelism / predict-side GIL release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 baseline harness: single-row latency distribution (p50/p99) + batch throughput, single- and multi-threaded
- [ ] #2 one infer(row) profiled to locate the real bottleneck before any optimization is built
<!-- AC:END -->
