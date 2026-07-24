# Struct-Qualifier Folding (TASK-38) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an unquoted struct-column qualifier fold to lowercase like every other identifier (`SELECT S.x` resolves against struct column `s`), matching the DataFusion oracle, while a quoted qualifier stays case-exact.

**Architecture:** `expr_build` cannot decide whether a qualifier names a relation or a struct column — only `plan.rs` knows, once `resolved.get(t)` has hit or missed. So the qualifier is carried from parse to that decision point as a `QualifiedName { value, quoted }` instead of a bare `String`, and each consumer asks for the form it needs: `.value` on the relation branch (raw, because our own generated SQL qualifies with `__THIS__`/`__STATE__`), `.folded()` on the struct-column branch and in `unnest_display_name`.

**Tech Stack:** Rust (pyo3 extension module `sql_transform._interpreter`), sqlparser 0.62, maturin, pytest + the differential harness (`tests/differential.py`).

## Global Constraints

- Design spec of record: `docs/superpowers/specs/2026-07-24-struct-qualifier-folding-design.md`. Read it before starting.
- **DataFusion is the parity oracle (decision-1).** Native must match it bug-for-bug.
- **The invariant this whole plan turns on: fold ONLY where the qualifier names data, NEVER where it names a relation.** Folding on the relation branch was measured at **96 test failures**, because our own rewrite emits `SELECT __THIS__.age / __STATE__.avg_age … FROM __THIS__ LEFT JOIN __STATE__`.
- Rust changes require `uv run maturin develop`. `uv sync` does **NOT** recompile Rust.
- **NEVER run `cargo test`** — it fails in this environment with an unrelated pyo3 `STATUS_DLL_NOT_FOUND`.
- Run tests with `uv run pytest`.
- Baseline before starting: **572 passed, 12 xfailed**. After this plan: **574 passed, 11 xfailed** (2 new tests, 1 xfail flipped).
- `xfail(strict=True)`: flip the marker in the SAME commit as the fix, or the suite goes red on success.
- To undo a temporary mutation, **RE-EDIT the file back. NEVER `git checkout`** — that destroys uncommitted work (it did exactly that earlier in this project).
- Do NOT edit `sql_transform/_codegen/plan.py` or `tests/test_codegen_coverage.py` — another developer owns them. Codegen already folds correctly; this is native catching up.
- Out of scope, do not fix: DRAFT-22 (relation-qualifier folding inversion), DRAFT-21 (`array(...)` dispatch).
- Land as a PR: `git push origin <branch>` then `gh pr create --body-file -`. Never `git push . <branch>:master`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/expr.rs` | `Expr` enum, `Value`, evaluation | Add `QualifiedName` type; change `Column.table` to `Option<QualifiedName>`; update `resolve_column` call site |
| `src/expr_build.rs` | sqlparser AST → `Expr` | Build `QualifiedName` from the qualifier `Ident`; replace the stale TASK-28 ceiling note |
| `src/plan.rs` | Planning, column validation, unnest naming | 3 consumers: relation branch (`.value`), struct-column fallback (`.folded()`), `unnest_display_name` (`.folded()`) |
| `src/types.rs` | Static type inference | One call site — passes the qualifier to `resolve_column_type` (relation branch → `.value`) |
| `tests/test_diff_types.py` | Differential type/container tests | Flip 1 xfail; add 2 tests |

### Acceptance criteria → task map

| AC | requirement | task |
|---|---|---|
| #1 | `Expr::Column` carries the qualifier's quote info end-to-end | Task 1 |
| #2 | unquoted folds (`S` → `s`), quoted stays exact (`"S"`) | Task 2 (both tests) |
| #3 | relation qualifiers unaffected — full suite green | Task 1 Step 7 + Task 2 Step 6 |
| #4 | xfail flipped, passes both engines, same commit as the fix | Task 2 |
| #5 | stale TASK-28 ceiling note updated | Task 3 |
| #6 | measure the CamelCase-table-alias case | **already discharged** — produced DRAFT-22 |

### The five `.table` consumers, classified

Every consumer must be deliberately assigned. Getting one wrong reintroduces the 96-failure signature.

| site | what it does with the qualifier | branch | takes |
|---|---|---|---|
| `expr.rs:339` → `resolve_column` | `row.get(t)` — `row` is keyed by relation name | relation | `.value` |
| `types.rs:39` → `resolve_column_type` | `schemas.get(t)` — keyed by relation name | relation | `.value` |
| `plan.rs:412` `column_qualifier` | compared against `static_table` (a relation) | relation | `.value` |
| `plan.rs:1062` `validate_expr` | `resolved.get(t)` — relation alias lookup | relation | `.value` |
| `plan.rs:1077` struct-column fallback | reinterprets `t` as a **column** after the lookup missed | **data** | **`.folded()`** |
| `plan.rs:1017` `unnest_display_name` | renders the qualifier into an output **column name** | **data** | **`.folded()`** |

---

### Task 1: Introduce `QualifiedName` and thread it through

**Files:**
- Modify: `src/expr.rs` (add the type; change `Column.table`; update `resolve_column` call at `:339`)
- Modify: `src/expr_build.rs:43-46` (build a `QualifiedName`)
- Modify: `src/plan.rs` (`:412`, `:1015`, `:1059`, `:1080`)
- Modify: `src/types.rs:39`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pub struct QualifiedName { pub value: String, pub quoted: bool }` with `pub fn folded(&self) -> String`. `Expr::Column`'s `table` field becomes `Option<QualifiedName>`. Task 2 relies on `.folded()` existing and on every consumer already compiling.

**Background:** This task is a pure refactor — **no behaviour change**. Every consumer takes `.value`, which is byte-for-byte what the old `String` held. The suite must be identically green at the end (572 passed, 12 xfailed, the TASK-38 test still xfailing). Behaviour changes land in Task 2. Splitting it this way means that if the suite moves in this task, the refactor is wrong and you know immediately.

- [ ] **Step 1: Add the type to `src/expr.rs`**

Add above the `Expr` enum:

```rust
/// A column's qualifier (`t` in `t.col`) with its SQL quoting preserved.
///
/// Carried unfolded on purpose: at parse time we cannot tell whether a
/// qualifier names a RELATION (`__THIS__.age`) or a STRUCT COLUMN (`s.x`) --
/// only `plan.rs` knows, once the relation-alias lookup has hit or missed.
/// Relation consumers take `.value` (our generated SQL qualifies with the raw
/// `__THIS__`/`__STATE__`); the struct-column branch takes `.folded()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedName {
    /// The identifier as written: `S`, `__THIS__`, `t`.
    pub value: String,
    /// Was it double-quoted in the SQL?
    pub quoted: bool,
}

impl QualifiedName {
    /// DataFusion's identifier rule, identical to `expr_build::fold_ident`:
    /// an unquoted identifier folds to lowercase, a quoted one stays exact.
    pub fn folded(&self) -> String {
        if self.quoted {
            self.value.clone()
        } else {
            self.value.to_lowercase()
        }
    }
}
```

- [ ] **Step 2: Change the `Column` variant in `src/expr.rs`**

```rust
    Column {
        table: Option<QualifiedName>,
        name: String,
    },
```

- [ ] **Step 3: Update the evaluator call site (`src/expr.rs:339`)**

```rust
        Expr::Column { table, name } => {
            resolve_column(row, table.as_ref().map(|t| t.value.as_str()), name)
        }
```

`resolve_column` looks the qualifier up in `row`, which is keyed by relation name — the relation branch, so `.value`.

- [ ] **Step 4: Build a `QualifiedName` in `src/expr_build.rs`**

Replace the `table:` line inside the `SqlExpr::CompoundIdentifier` arm (currently `table: Some(parts[0].value.clone()),`) with:

```rust
                table: Some(crate::expr::QualifiedName {
                    value: parts[0].value.clone(),
                    quoted: parts[0].quote_style.is_some(),
                }),
```

Leave `name: fold_ident(&parts[1])` and the `parts[2..]` `FieldAccess` loop exactly as they are.

- [ ] **Step 5: Update the four relation-branch consumers**

`src/types.rs:39`:

```rust
        Expr::Column { table, name } => {
            resolve_column_type(table.as_ref().map(|t| t.value.as_str()), name, schemas)
        }
```

`src/plan.rs:412` (inside `column_qualifier`):

```rust
        Expr::Column { table: Some(t), .. } => Some(t.value.as_str()),
```

`src/plan.rs:1062` — the `resolved.get(...)` lookup inside `validate_expr`'s `Expr::Column { table: Some(t), name }` arm:

```rust
            if let Some((real, is_row)) = resolved.get(t.value.as_str()) {
```

`src/plan.rs:1017` — `unnest_display_name`'s qualified arm. Keep `.value` **for now**; Task 2 changes it to `.folded()`:

```rust
        Expr::Column {
            table: Some(t),
            name,
        } => Ok(format!("{}.{name}", t.value)),
```

- [ ] **Step 6: Update the struct-column fallback to compile (`src/plan.rs:1077-1082`)**

Keep `.value` **for now** — Task 2 makes it `.folded()`:

```rust
            let base_name = t.value.clone();
            let field = name.clone();
            *e = Expr::FieldAccess {
                base: Box::new(Expr::Column {
                    table: None,
                    name: base_name,
                }),
                field,
            };
```

- [ ] **Step 7: Build and run the full suite**

Run: `uv run maturin develop && uv run pytest -q`
Expected: `572 passed, 12 xfailed` — **identical to baseline**. `test_uppercase_qualifier_field_access` must still xfail; this task changes no behaviour.

If any count differs, the refactor changed behaviour somewhere — find it before continuing. A jump to ~96 failures means a consumer got `.folded()` instead of `.value`.

- [ ] **Step 8: Commit**

```bash
git add src/expr.rs src/expr_build.rs src/plan.rs src/types.rs
git commit -m "refactor(native): carry a column qualifier as QualifiedName — TASK-38

A qualifier is either a relation alias or a struct column, and expr_build
cannot tell which -- only plan.rs knows, once the relation lookup hits or
misses. Carrying it as QualifiedName { value, quoted } preserves the SQL
quoting to that decision point, so each consumer can ask for the form it
needs.

Pure refactor: every consumer takes .value, which is exactly what the old
String held. No behaviour change; suite identical."
```

---

### Task 2: Fold on the struct-column branch

**Files:**
- Modify: `src/plan.rs:1077` (struct-column fallback), `src/plan.rs:1017` (`unnest_display_name`)
- Modify: `tests/test_diff_types.py` (flip 1 xfail, add 2 tests)

**Interfaces:**
- Consumes: `QualifiedName::folded()` and `Expr::Column { table: Option<QualifiedName> }` from Task 1.
- Produces: nothing.

**Background:** This is the behaviour change. Both edited sites are **data** consumers — the struct-column fallback runs only after the relation lookup has already missed, and `unnest_display_name` renders the qualifier into an output column name. Measured on the oracle:

```
SELECT unnest(t.s) FROM t   ->  cols ['t.s.x', 't.s.y']
SELECT unnest(T.s) FROM t   ->  cols ['t.s.x', 't.s.y']     <- qualifier normalised
```

- [ ] **Step 1: Write the failing tests**

In `tests/test_diff_types.py`, replace the existing `test_uppercase_qualifier_field_access` (currently takes an `xfail_on_native` fixture) with these three tests:

```python
def test_uppercase_qualifier_field_access():
    # An unquoted struct-column qualifier folds like any identifier: `S.x` ->
    # `s.x`, matching DataFusion (TASK-38). The fold happens in plan.rs's
    # struct-column fallback -- the branch reached only after `S` fails to
    # resolve as a relation alias -- so relation qualifiers are untouched.
    check(
        "SELECT S.x AS v FROM t",
        {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 7}}])},
    )


def test_quoted_qualifier_stays_case_exact():
    # The other half of TASK-28's folding rule, and the guard against
    # over-folding: a QUOTED qualifier keeps its case, so `"S"` does NOT match
    # struct column `s` and both engines reject the query. Without this, a fix
    # that folded unconditionally would still look correct.
    check_both_raise(
        'SELECT "S".x AS v FROM t',
        {"t": rows({"s": "struct{x:int}"}, [{"s": {"x": 7}}])},
    )


def test_unnest_uppercase_qualifier_output_names():
    # unnest() renders the qualifier into the OUTPUT COLUMN NAMES, and the
    # oracle normalises it there too: `unnest(T.s)` yields t.s.x / t.s.y, not
    # T.s.x. Measured -- this site is not obvious from the field-access tests.
    check(
        "SELECT unnest(T.s) FROM t",
        {"t": rows({"s": "struct{x:int,y:int}"}, [{"s": {"x": 5, "y": 9}}])},
    )
```

`check`, `check_both_raise` and `rows` are already imported at the top of the file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_diff_types.py::test_uppercase_qualifier_field_access tests/test_diff_types.py::test_quoted_qualifier_stays_case_exact tests/test_diff_types.py::test_unnest_uppercase_qualifier_output_names -q`

Expected:
- `test_uppercase_qualifier_field_access[native]` FAILS with `Unknown column: S` (the xfail marker is gone, so the pre-existing gap now shows as a real failure).
- `test_unnest_uppercase_qualifier_output_names[native]` FAILS — native emits `T.s.x`, the oracle `t.s.x`.
- `test_quoted_qualifier_stays_case_exact` PASSES already (both engines reject it today). It is a lock-in against over-folding, and Step 5's mutation check is what proves it earns its place.
- All `[codegen]` variants PASS — codegen already folds.

- [ ] **Step 3: Fold at the struct-column fallback (`src/plan.rs:1077`)**

```rust
            // `t` isn't a relation alias -- the "table.column" parse was wrong;
            // reinterpret it as struct field access. `t` now names DATA, not a
            // relation, so it folds like any other identifier (TASK-38):
            // unquoted `S` -> `s`, quoted `"S"` stays exact. Folding earlier
            // (in expr_build) is what breaks relation qualifiers -- our own
            // rewrite emits `__THIS__.age`, and folding that misses.
            let base_name = t.folded();
```

Leave the rest of the arm unchanged.

- [ ] **Step 4: Fold in `unnest_display_name` (`src/plan.rs:1017`)**

```rust
        // The qualifier is rendered into the OUTPUT COLUMN NAME here, and the
        // oracle normalises it: `unnest(T.s)` names its columns `t.s.x`, not
        // `T.s.x`. Same fold as the struct-column branch.
        Expr::Column {
            table: Some(t),
            name,
        } => Ok(format!("{}.{name}", t.folded())),
```

- [ ] **Step 5: Build, run the tests, then mutation-check**

Run: `uv run maturin develop && uv run pytest tests/test_diff_types.py -q`
Expected: all pass on both engines.

Mutation A — make `folded()` always fold (delete the `quoted` branch), in `src/expr.rs`:

```rust
    pub fn folded(&self) -> String {
        self.value.to_lowercase()
    }
```

Run: `uv run maturin develop && uv run pytest tests/test_diff_types.py::test_quoted_qualifier_stays_case_exact -q`
Expected: **FAIL** — `"S".x` now wrongly resolves on native while the oracle still rejects it. This proves the AC#2 test catches over-folding. **Restore by re-editing** the `if self.quoted` branch back. Never `git checkout`.

Mutation B — make `folded()` never fold:

```rust
    pub fn folded(&self) -> String {
        self.value.clone()
    }
```

Run: `uv run maturin develop && uv run pytest tests/test_diff_types.py::test_uppercase_qualifier_field_access -q`
Expected: **FAIL** with `Unknown column: S`. **Restore by re-editing.**

After restoring, run `uv run maturin develop && uv run pytest tests/test_diff_types.py -q` and confirm green again.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: `574 passed, 11 xfailed`.

The count is the AC#3 guard: **572 → 574** means the 2 new tests passed and the flipped xfail now passes. Anything near **96 failures** means a fold leaked onto the relation branch — revert Step 3/4 and recheck which consumer you edited.

- [ ] **Step 7: Commit**

```bash
git add src/plan.rs tests/test_diff_types.py
git commit -m "fix(native): fold an unquoted struct-column qualifier — TASK-38

SELECT S.x against struct column s raised 'Unknown column: S' on native while
DataFusion folded the qualifier like any identifier and returned the value. So
TASK-28's folding rule silently did not apply in the struct-qualifier position.

The fold happens in plan.rs's struct-column fallback -- the branch reached only
once the qualifier has FAILED to resolve as a relation alias -- so relation
qualifiers are untouched. Folding earlier, in expr_build, breaks them: our own
rewrite emits __THIS__.age, and folding that misses (measured: 96 failures).

unnest_display_name folds too: it renders the qualifier into output column
names, and the oracle normalises it there (unnest(T.s) -> t.s.x).

A quoted qualifier still stays case-exact, pinned by a new test."
```

---

### Task 3: Replace the stale ceiling note

**Files:**
- Modify: `src/expr_build.rs:37-42`

**Interfaces:**
- Consumes: the behaviour from Task 2.
- Produces: nothing.

**Background (AC#5):** The comment in the `CompoundIdentifier` arm currently reads:

> ponytail: a real CamelCase table or struct-column qualifier won't fold like DataFusion here; not reachable today (qualifiers are always library-internal). Widen to fold parts[0] too if user-named CamelCase relations ever appear.

Both claims are now false. The struct-column case **is** reachable and is fixed, and "widen to fold parts[0]" is precisely the change measured at 96 failures. Leaving it would send the next reader at the exact wrong fix.

- [ ] **Step 1: Replace the comment**

Replace the comment block above the `let mut expr = Expr::Column {` line with:

```rust
            // Fold the column/field parts (`parts[1..]`) here; carry the
            // leading qualifier (`parts[0]`) UNFOLDED as a QualifiedName.
            //
            // We cannot fold it here: at parse time a qualifier may name a
            // RELATION (`__THIS__.age`, which our own rewrite emits) or a
            // STRUCT COLUMN (`S.x`). Only plan.rs knows which, once the
            // relation-alias lookup hits or misses. Folding at this point
            // breaks every internally-qualified query -- measured at 96 test
            // failures -- so the fold lives on plan.rs's struct-column branch
            // instead (TASK-38).
            //
            // Relation qualifiers themselves still diverge from the oracle,
            // which registers relations under a FOLDED name: see DRAFT-22.
```

- [ ] **Step 2: Verify the comment is accurate**

Run: `grep -n "not reachable today" src/expr_build.rs`
Expected: no output — the stale claim is gone.

- [ ] **Step 3: Build and run the full suite**

Run: `uv run maturin develop && uv run pytest -q`
Expected: `574 passed, 11 xfailed` — unchanged; this is a comment-only change.

- [ ] **Step 4: Commit**

```bash
git add src/expr_build.rs
git commit -m "docs(native): correct the qualifier-folding note — TASK-38 AC#5

The TASK-28 ceiling note said a struct-column qualifier 'won't fold ... not
reachable today' and advised widening the fold to parts[0]. Both are now
false: the case is reachable and fixed, and folding parts[0] is exactly the
change that breaks relation qualifiers (96 failures). Replaced with why the
qualifier is carried unfolded, and a pointer to DRAFT-22 for the relation
half that still diverges."
```

---

### Task 4: Land as a PR

**Files:** none.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a PR URL.

- [ ] **Step 1: Confirm the suite and the scope**

```bash
uv run pytest -q
git diff --name-only origin/master
```
Expected: `574 passed, 11 xfailed`; exactly five files — `src/expr.rs`, `src/expr_build.rs`, `src/plan.rs`, `src/types.rs`, `tests/test_diff_types.py` (plus the spec/plan docs).

- [ ] **Step 2: Confirm no forbidden file was touched**

Run: `git diff --name-only origin/master -- sql_transform/_codegen/plan.py tests/test_codegen_coverage.py`
Expected: no output.

- [ ] **Step 3: Rebase onto master**

```bash
git fetch origin master
git rebase origin/master
uv run maturin develop && uv run pytest -q
```
Expected: still green after rebase. Rebuild is required — a rebase can change `src/*.rs`.

- [ ] **Step 4: Push and open the PR**

```bash
git push origin task-38-struct-qualifier-fold
gh pr create --base master --head task-38-struct-qualifier-fold \
  --title "TASK-38: fold an unquoted struct-column qualifier" --body-file - <<'PRBODY'
`SELECT S.x` against struct column `s` raised `Unknown column: S` on native, while DataFusion folded the qualifier like any other identifier and returned the value — so TASK-28's folding rule silently did not apply in the struct-qualifier position.

## Why the one-line fix doesn't work

The ticket framed this as folding the qualifier in `expr_build.rs`. Measured:

```
expr_build.rs   Some(parts[0].value.clone())  ->  Some(fold_ident(&parts[0]))
result:         96 failed, 477 passed
```

Our own rewrite emits `SELECT __THIS__.age / __STATE__.avg_age … FROM __THIS__ LEFT JOIN __STATE__`, so folding at parse turns `__THIS__` into `__this__` and every windowed transform on native dies.

`expr_build` structurally cannot make the call: a qualifier is either a relation alias or a struct column, and only `plan.rs` knows which — once `resolved.get(t)` hits or misses.

## The fix

The qualifier is carried to that decision point as a type instead of a bare string:

```rust
pub struct QualifiedName { pub value: String, pub quoted: bool }
impl QualifiedName { pub fn folded(&self) -> String { … } }   // unquoted -> lowercase
```

Each consumer then asks for the form it needs — **fold only where the qualifier names data, never where it names a relation**:

| site | branch | takes |
|---|---|---|
| `resolve_column`, `resolve_column_type`, `column_qualifier`, `validate_expr`'s alias lookup | relation | `.value` |
| struct-column fallback (`plan.rs`) | data | `.folded()` |
| `unnest_display_name` | data | `.folded()` |

`unnest_display_name` is not in the ticket; it surfaced by measurement — it renders the qualifier into output column *names*, and the oracle normalises it there too (`unnest(T.s)` → `t.s.x`, not `T.s.x`).

## Verification

- Suite **574 passed, 11 xfailed** (572 baseline + 2 new tests, 1 xfail flipped).
- Mutation-checked both directions: forcing `folded()` to always fold breaks the quoted test; never folding breaks the unquoted test.
- Task 1 is a pure refactor committed separately, with the suite proven byte-identical (572/12) before any behaviour changed — so a regression is attributable to one commit.
- Codegen untouched; it already folds correctly. This is native catching up.

## Out of scope

**DRAFT-22** — relation qualifiers still diverge from the oracle (DataFusion registers relations under a *folded* name, so `"__THIS__".age` misses there and hits on native). Measured, unreachable through the public API, fails loudly. Different branch, second core change — deliberately not folded in.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
PRBODY
```

- [ ] **Step 5: Verify the PR body landed**

Run: `gh pr view --json body --jq .body | head -5`
Expected: the real body text, not the literal string `@-`.

- [ ] **Step 6: Report to the PM**

Message Iris with the PR URL, the five ACs' status (AC#6 already discharged), the suite delta, and the `unnest_display_name` finding — a site the ticket did not name, found by measurement.
