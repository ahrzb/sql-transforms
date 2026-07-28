# Building the SQL Specializer with a Loop / Dynamic Workflow

**Status:** draft for review — the "how we build it" companion to
`2026-07-25-sql-specializer-design.md`. Covers (a) what must be prepared before
any unattended execution starts, and (b) the design of the loop itself.

## 0. The shape of the problem

The specializer build is a long, mostly-sequential grind (IR → interpreter →
lowering → codegen) with two properties that decide the tooling:

- **Between milestones it is sequential and judgment-heavy** → human review
  gates, one milestone at a time. A loop's job here is *pacing and persistence*,
  not parallelism.
- **Within a milestone it is wide and mechanical** (one lowering rule per
  operator × one differential test per rule; hundreds of corpus cases; N
  verifier rules) → fan-out via dynamic workflows *inside* a loop iteration.

So: **/loop is the spine, Workflow is the muscle.** Not either/or.

The design doc's verifier + text format + differential suite are what make this
safe: an unattended iteration cannot self-certify with prose — it either turns
a machine-checkable gate green or it doesn't.

## 1. Preparation checklist (all before the first unattended iteration)

Everything here is small, and every item removes a way for the loop to stall or
silently go wrong.

### 1.1 Decisions locked
- [ ] Design doc reviewed and amended by AmirHossein (M-restate gate). Partially
      done in conversation (row-major ABI amendment, 2026-07-25); final go still
      pending.
- [x] Pending diff landed on this branch (2026-07-25): the `packages/confit/src/duckdb/` stub +
      shared-module refactor (`error.rs`/`value.rs`/`types.rs` split) — the
      shared substrate `specializer/` sits on; the stub is the future pyclass
      shell.

### 1.2 Oracle and frontend availability (validate, don't assume)
- [x] `uv add --dev duckdb` — the differential oracle. duckdb 1.5.5 installed.
- [x] **Spike (2026-07-25): substrait is UNAVAILABLE** — HTTP 404 for
      duckdb 1.5.5 / windows_amd64 from community, core, and nightly repos.
      Frontend flag flipped to the sqlparser fallback (design doc §4 updated).
      Bonus finding: `json_serialize_sql` (core, extension-free) exposes
      DuckDB's own AST as JSON — usable as a differential check on our parser.
- [x] **Spike (2026-07-25): `cranelift-jit` 0.126 works on
      x86_64-pc-windows-msvc** — built and called `f(x) = x*2+42` at runtime,
      correct results. Version pin recorded for M-cranelift.

### 1.3 The gate command (the loop's definition of done)
One command, exit-code-honest, that every iteration must leave green:

```toml
# mise.toml — wired 2026-07-25 and green (cargo test + 574 pytest)
[tasks.gate-specializer]
run = "uv run python scripts/gate.py"   # cargo test + pytest, one exit code
```

`scripts/gate.py` also handles the Windows wrinkle: tests link libpython
(extension-module moved to a maturin-only feature), so the runner puts the
uv-managed CPython's `python3.dll` on PATH. Per-milestone suites get appended
there (e.g. M-lower adds the corpus replay). The loop never reports progress
that `mise gate-specializer` can't confirm — "validate, don't assume" made
mechanical.

### 1.4 Task ledger
The loop needs durable, machine-readable state that survives context loss.
Use `backlog.md` (already wired via MCP): one milestone note + one task per unit
of work, each with acceptance = "gate green + which new tests exist". The loop's
first act each iteration is `task_list`, its last is `task_edit`. No progress
lives only in conversation memory.

- [x] Milestone `sql-specializer` (m-7) seeded 2026-07-25 with TASK-41 (M-ir) →
      TASK-42 (M-interp) → TASK-43 (M-lower) → TASK-44 (M-cranelift) →
      TASK-45 (M-boundary), dependency-chained. Working the chain (dispatch)
      still needs AmirHossein's go — PM dispatch rule.

### 1.5 Corpus extraction (pre-mined, not mined mid-loop)
- [x] `scripts/mine_duckdb_corpus.py` (2026-07-25): 678 cases from 2758 queries
      across 250 files → `packages/confit/tests/corpus/duckdb_mined.jsonl` (262 KB, checked in;
      setup statements + sql + duckdb-computed expected rows). Replay contract
      is three-outcome — match / clean-unsupported / FAIL — so the corpus
      includes SQL beyond the v0 builtin list on purpose: each case the engine
      learns flips from clean-unsupported to must-match. See the script
      docstring.
- [x] The `duckdb/` clone stays untracked; `.gitignore`d. `testpaths` pinned in
      pyproject so pytest never collects the clone's own test_*.py scripts.

### 1.6 Hygiene rails (mechanical, from memory/feedback)
- [ ] Branch per milestone: `git checkout -b specializer-m2-ir origin/master`
      as the *first* act of a milestone (branch-first rule).
- [ ] Land via PR to `origin`, never ref-push (land-via-PR rule). One PR per
      milestone, opened at the review gate.
- [ ] Bug protocol: specializer disagrees with DuckDB → xfail-strict test +
      ledger ticket, never an inline semantics patch (adapted decision-1 with
      DuckDB as this engine's oracle).

## 2. Loop design

### 2.1 State machine

```
         ┌────────────────────────────────────────────┐
         ▼                                            │
   read ledger ─► milestone done? ──yes──► open PR, notify user, STOP (review gate)
         │no
         ▼
   pick next task ─► TDD it (red → green → gate) ─► commit ─► ledger update
         │                                                        │
         │ blocked/ambiguous? ─► write blocker note, notify, STOP │
         └────────────── ScheduleWakeup ◄─────────────────────────┘
```

Hard rules:
- **STOP at milestone boundaries.** The design doc's "stopping for review at
  each boundary" is a contract; the loop opens the PR and does not start the
  next milestone until told to. No self-granted approvals.
- **STOP on ambiguity** (surface-confusion-early rule). A blocker note in the
  ledger + a message beats an unattended guess.
- **Every commit passes the gate.** An iteration that can't get green either
  reverts to last green or stops with a red-state note — it never commits red.

### 2.2 Iteration budget and pacing
- One task per iteration (a task is sized ≤ ~1h of work: one IR op family, one
  lowering rule, one verifier rule + its tests).
- Dynamic pacing: `ScheduleWakeup` long (1200s+) after a STOP; short only when
  a background build/test run is the wait.
- Kill switch: the loop halts if the same task fails the gate in two consecutive
  iterations → blocker note + notify (prevents grinding a wall).

### 2.3 Where dynamic workflows plug in

Inside a single loop iteration, when the task is wide-and-mechanical:

| task shape | workflow pattern |
|---|---|
| implement lowering for N operators (M-lower) | `pipeline(ops, implement-in-worktree, differential-verify)` — worktree isolation per op, verify stage runs the op's mined corpus slice |
| corpus triage (M-lower) | fan-out readers over `duckdb_mined.jsonl` failures → cluster by root cause → one ledger ticket per cluster, not per case |
| verifier adversarial pass (M-ir) | N agents each try to construct an IR program that passes the verifier but breaks the interpreter; loop-until-dry |
| codegen parity sweep (M-cranelift) | random-IR generator fan-out, interpreter-vs-cranelift adversarial verify |

Sketch of the M-lower fan-out an iteration would launch:

```js
const results = await pipeline(
  OPS,                                      // e.g. ["fdiv", "case", "probe", ...]
  op => agent(`Implement lowering for ${op}; TDD; run mise gate-specializer`,
              {isolation: "worktree", phase: "Implement"}),
  (r, op) => agent(`Run the ${op} slice of packages/confit/tests/corpus/duckdb_mined.jsonl
                    against the interpreter backend; report mismatches as
                    structured findings`, {phase: "Verify", schema: FINDINGS}),
)
```

The loop (not the workflow) merges surviving worktrees and runs the full gate —
one integrator, many implementers.

### 2.4 What the loop prompt contains (draft)

The recurring prompt must be self-contained (survives compaction):

> Work the `sql-specializer` milestone ledger in backlog.md. Read
> `docs/superpowers/specs/2026-07-25-sql-specializer-design.md` and this doc
> first. One task per iteration; TDD; `mise gate-specializer` must be green
> before any commit; update the ledger after. STOP and notify at milestone
> boundaries (open a PR) or on any ambiguity/blocked task. Never start the next
> milestone without explicit approval.

### 2.5 Loop vs. plain sessions — honest tradeoff

M-ir and M-interp are compact enough that an attended session each might beat a
loop (tight feedback, no pacing overhead). The loop earns its keep from M-lower
onward, where the work is a long tail of similar units against mechanical gates.
A defensible plan: attended sessions through M-interp, loop for M-lower and
M-cranelift. Decide at the M-interp review gate.

## 3. Open questions for the loop conversation

1. Land the pending refactor+stub PR first, or fold it into M-ir's branch?
2. Attended vs loop for M-ir/M-interp (§2.5)?
3. Token/notification budget per unattended iteration, and quiet hours?
4. Ledger seeding needs your go (PM-dispatch rule) — seed all milestones now or
   only M-ir?
5. Workflow size guideline is "medium" (≤15 agents) in this session — fine for
   the sweeps above, or raise it for the corpus triage?
