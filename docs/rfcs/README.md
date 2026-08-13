# RFCs

An RFC is how this repo decides something that has real alternatives — an
architecture fork, a representation choice, a boundary between components.
Specs (`docs/superpowers/specs/`) describe a design already chosen; an RFC
exists *before* the choice, to make the alternatives and their costs
explicit. If there is only one sensible way, write a spec, not an RFC.

## Format

One file: `docs/rfcs/rfc-N-<slug>.md`, numbered in creation order. Four
mandatory sections, in this order:

1. **Context** — what exists today, with file paths and measured numbers.
   No aspiration; the reader must be able to verify every sentence.
2. **Problem(s)** — what the current state cannot do, or forces us to do
   badly. Numbered, so choices can reference them.
3. **Choices** — the genuinely viable options, each with enough concrete
   shape (signatures, node forms, gate names) to be criticized. Include
   the do-nothing option when it is viable.
4. **Trade-offs** — what each choice costs and buys, against the problems
   by number and against the project's KPIs (docs/kpis.md: never trade a
   control for a drive gain).

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
