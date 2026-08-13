# RFCs

An RFC is how this repo decides something that has real alternatives — an
architecture fork, a representation choice, a boundary between components.
Specs (`docs/superpowers/specs/`) describe a design already chosen; an RFC
exists *before* the choice, to make the alternatives and their costs
explicit. If there is only one sensible way, write a spec, not an RFC.

## Format

One file: `docs/rfcs/rfc-N-<slug>.md`, numbered in creation order. Three
mandatory sections, in this order:

1. **Context** — what exists today, with file paths and measured numbers.
   No aspiration; the reader must be able to verify every sentence.
2. **Problem(s)** — what the current state cannot do, or forces us to do
   badly. Numbered, so choices can reference them.
3. **Choices** — the genuinely viable options, do-nothing included when
   viable. Each choice is one block, in this exact shape:

   ```markdown
   ### A. <name>

   <code: a signature, a node form, a pseudo-code lowering, a before/after pair>

   **Pros**
   - one claim per bullet, tied to a problem number or a KPI

   **Cons**
   - one claim per bullet; include the cost you would regret ignoring
   ```

   Two rules, both learned the hard way. **Show code, not prose** — prose
   descriptions of designs hide small nuances; a reader must be able to
   disagree with a specific line. **Keep Pros/Cons attached to their
   choice** — a trade-off in a collected section at the end gets skimmed
   past. Bullets, not paragraphs: one claim each, scannable side by side.
   Tie claims to the problems by number and to the project's KPIs
   (docs/kpis.md: never trade a control for a drive gain).

Decisions the author cannot make alone appear inline as ASK blocks:

> **ASK(name):** the question, with the author's recommendation and why.

## Lifecycle

Header carries `Status: proposed | decided | withdrawn`. An RFC is
*decided* when AmirHossein answers every ASK block; record each answer
inline under its ASK (`**DECIDED:** ...`), then write or amend the spec
that carries the chosen design and link it from the RFC header. RFCs are
never deleted and never edited after decision except to fix links — the
trade-off record is the point.

Keep it short. Two pages is a long RFC; the format is four sections, not
a template to fill.
