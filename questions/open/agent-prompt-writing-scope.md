# Should loop prompts use the maintainer writing rules?

**Asked by:** native catalog, 2026-10-17. **Holds:** rewriting the mandates' briefs and READMEs outside the simple-English rules of `AGENTS.md`.
**Against:** `40f6ab17`.

## Context

Should AGENTS.md rule 1 exempt loop prompts from the simple-English skill?
Rule 1 requires that skill for text the maintainer reviews or maintains.
The skill sets sentence rules and requires a reader who has not seen the session.

**Interim:** Keep AGENTS.md unchanged. Apply its current rule to maintainer texts, including this README and decision records.
Agent-only working files still need clear instructions.

## Read first

- [Native catalog brief](../../mandates/native/brief.md)
- [The native loop review](../../research/native-loop-review/README.md)

## Options

### Option 1

Keep the present scope, including loop READMEs.

### Option 2

Exempt agent prompts. Keep simple English for reports, decision records, goals, package specs and large PR summaries.

### Option 3

Apply simple English only to maintainer-facing sections within mixed documents.

Option 2 covers README cycles, briefs, workers.md, reporting.md and traps.md.
The last file supplies the checklist of repeatable lessons to each native worker.

## Recommendation

Option 2: the briefs and trackers are read by agents every session, and the reviews in [the native loop review](../../research/native-loop-review/README.md) found the rewrite removed reasons that agents need. It would change if the maintainer reads the briefs himself.

## Evidence

The writing-rule changes in PR #401 (`e2720646`) and PR #408 (`c9245858`) increased seven documents by 15–33%.
These are whitespace word counts from each commit against its parent:

| document and purpose | before | after |
|---|---|---|
| AGENTS.md#helpers-in-cloud-sessions: worker procedure | 1,148 | 1,443 |
| mandates/README.md: reporting policy | 649 | 810 |
| AGENTS.md#commands-repo-root: shared opening prompt | 313 | 360 |
| mandates/README.md: shared loop rules | 407 | 541 |
| mandates/native/README.md: native cycle | 1,145 | 1,380 |
| mandates/native/README.md#objective: native goal | 402 | 518 |
| mandates/native/README.md#reporting-contract: report shape | 921 | 1,072 |

The counts establish length changes, not worse execution.
PR #408 also removed examples from the bound and fixture instructions.
The removed bound examples were “BLAS sums, NumPy pairwise reductions”.
BLAS is the numerical library that sums products; pairwise reduction sums a tree of pairs.
The fixture examples were “strings, categories, a narrower domain”. They name inputs the tests might otherwise omit.
The old Done step explained `--no-sync`: it keeps the installed extension when a command starts.
A readability check alone does not check whether an instruction preserves its meaning.

What would close it: The maintainer chooses the scope of rule 1.
This record does not change that rule or restore an automatic prose check.

## Answer

