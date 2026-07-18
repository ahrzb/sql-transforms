---
id: m-6
title: "Epic: Multi-language inference runtimes"
---

## Description

UNSCOPED EPIC — not prioritized (AmirHossein 2026-07-18). Multi-quarter. Goal: serve trained SQL-transforms from any backend language (Go/Java/C#/Node) with NO Python/DataFusion/FFI at inference time. Interchange = a serialized logical plan (Substrait or homegrown IR) + Parquet fitted tables; runtimes are thin pre-resolved tree-walk interpreters. Substrait feasibility validated with real artifacts. **Entry point is scoping/planning, not implementation** — no B-tickets until scoped. Full validated design brief: backlog/docs/doc-4. Rests on the two-engine framing (native = one-of-N), an open question not being ratified short-term.
