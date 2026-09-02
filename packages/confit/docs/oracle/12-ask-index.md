## 12. ASK index

### 12.1 Ruled

An ASK leaves this section only by being answered. The ruling text lives at the point in
the document where it binds, next to the claim it created.

| ask | ruling | where it landed | implemented by |
|---|---|---|---|
| **ask: unshipped-never-compared** | normalization is out of the oracle's answer and out of the verdict: an unshipped feature FAILS or is CLASSIFIED, never absorbed by weakening a comparison, and a deviation from raw equality happens only via a **named bound in a reviewed draft** (precedent: cbrt 1-ulp, sklearn `1e-9`, DRAFT-23). The 1-ulp decimal deltas in past campaigns were artifacts of the harness's own cast, manufactured by neither engine | claim: unshipped-verdict, chapter 5; divergence: decimal-cast-artifact's ruling cell | the decimal-to-float64 cast **deleted** from `fuzz/oracle.py`; the `UNSHIPPED` verdict kind and its report section in `fuzz/runner.py`; gated by `tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared` |

### 12.2 Open

Fifteen open, in document order, with what each binds.

| ask | question | at | binds |
|---|---|---|---|
| **ask: engine-fold-reading** | does the engine's build-time fold move to the oracle's reading? It folds optimizer-ON in production while the oracle is optimizer-OFF. Whether that is observable is **unmeasured** — take the measurement before ruling | 1.3 | claim: oracle-identity, claim: contract-surface-gap, claim: no-raw-connections, claim: row-limit-refusal, claim: build-vs-build-repeatability, claim: duckdb-three-roles' role (b), claim: one-door-bypass |
| **ask: version-pin** | pin `==1.5.5` or floor-plus-assert; and 1.5.5 or the LTS line. The assert's landing spot is one line in `Oracle.__init__` | 1.3 | claim: oracle-identity, claim: oracle-version-constant, claim: capture-outside-the-oracle, every pin |
| **ask: frozen-row-order** | build-vs-build repeatability: sort-at-freeze, out of contract, or tentative | 3.7 | claim: build-vs-build-repeatability |
| **ask: threads-and-value-order** | does `threads` join the oracle constant, and what disposition covers order *inside* a value. Landing spot: one line in `Oracle.__init__` | 3.7 | claim: oracle-identity, claim: disposition-table, claim: build-vs-build-repeatability, claim: threads-setting |
| **ask: refusal-cost-counting** | make the accepted severity-4 refusal cost countable, or amend the RFC to say it is unmeasured | 4.3 | claim: refusal-absorb, claim: countable-rung-four, divergence: bind-time-constant-refusals |
| **ask: opt-emulated-branch** | is `OPT_EMULATED`'s `AGREE` treatment deliberate? `UNSHIPPED` now shares that branch **with** a stated ground; `OPT_EMULATED` still has none | 4.3 | claim: opt-emulated-classification, claim: unshipped-verdict |
| **ask: reason-code-visibility** | do reason codes stay internal, or become user-visible refusal text | 4.3 | claim: refusal-message-prefixes |
| **ask: float-tolerance-list** | bit-for-bit floats and what governs the exceptions already in force; and whether `_type_delta`'s unshipped arm earns a gate of its own | 5 | claim: float-bit-equality, claim: cbrt-ulp-tolerance, claim: feature-in-flight, claim: standing-rejections, claim: unshipped-verdict, divergence: decimal-literal-typing, divergence: decimal-cast-rounding |
| **ask: doc-twin-overstatement** | close TASK-95 or downgrade the five doc-twin totality sites | 7.2 | claim: doc-twin-totality |
| **ask: unlisted-divergence** | adopt "an unlisted divergence is a bug by definition" | 7.4 | claim: ledger-adjudication |
| **ask: tentative-bucket** | admit a `tentative` (measured, not ruled) bucket | 7.4 | claim: build-vs-build-repeatability, claim: threads-setting, claim: one-door-bypass, divergence: phase-two-width-residuals, divergence: snapshot-baseline |
| **ask: baseline-as-evidence** | is a campaign baseline evidence, or a snapshot | 7.4 | claim: regexp-fuzz-gate, divergence: trap-elision, divergence: decimal-cast-artifact, divergence: snapshot-baseline, ask: width-residual-classification |
| **ask: width-residual-classification** | classify the width residuals before treating them as a defect count | 10 | divergence: phase-two-width-residuals, divergence: snapshot-baseline |
| **ask: match-count-ratchet** | corpus match count: dated generated number, or a ratchet | 10 | claim: zero-fails-gate, claim: dialect-gate-oracle |
| **ask: proposed-rules-adoption** | do this document's own eight rules (across nine claims) become normative. Covers nine of the fifteen `[PROPOSED]` claims; the other eight route through their own tickets | 10 | claim: target-status-vocabulary, claim: countable-cost, claim: multi-answer-sets, claim: unadopted-mechanisms, claim: divergence-placement, claim: directional-rungs, claim: countable-rung-four, claim: unspecified-residuals, claim: standing-rejections |

Plus the section 7.3 ledger: **16 rows**, of which **divergence: decimal-cast-artifact
is now ruled** (the ask: unshipped-never-compared ruling, transcribed into its cell).
divergence: phase-two-width-residuals and divergence: snapshot-baseline are marked
*unruled*, divergence: decimal-literal-typing and divergence: decimal-cast-rounding are
held together under ask: float-tolerance-list, and the remainder carry a proposed status
that is not in force until you fill the column.

### 12.3 What changed in this revision

The abstractions this document described have shipped as code, and the document is now
written against them rather than against the arrangement that preceded them. The changes
a reader needs to know about:

- **Every code became a slug.** Claims, ASK blocks, ledger rows and proposed tickets are
  named by what they are about, and cited as `claim:` / `ask:` / `divergence:` /
  `ticket:` — the front matter states the rule, including that a slug is assigned once.
  The numeric codes are kept as pointers in [13-old-ids.md](13-old-ids.md), because they
  are cited from merged PRs, backlog tickets and at least one code comment.
- **Non-circularity is stated once, in the README front matter.** The claims are the
  authority and the modules are their enforcement; every claim that has an
  implementation now carries `Enforced-by:` (the module function) and `Verified-by:`
  (the test), and the prose that used to explain where a behavior lived and how its
  copies were kept in sync is deleted — the pointers are that.
- **ask: unshipped-never-compared is ruled**, and the ruling created
  claim: unshipped-verdict. The consequences are applied throughout:
  claim: schema-comparison no longer authorizes a cast, claim: optimizer-bracket gains
  the ranking rule, claim: coverage-accounting gains the report section, the
  normalization blind spot in claim: blind-spots is replaced by the honest one, and
  divergence: decimal-literal-typing and divergence: decimal-cast-artifact are
  corrected.
- **claim: multiset-default is retired** — one comparison vocabulary made it a
  restatement of claim: compare-modes plus a default. Its slug is kept, with a tombstone
  naming the old code (ORC-36); nothing is reassigned.
- **Four claims are new.** claim: repr-equality (what equal means, and the single
  declared axis), claim: one-door-bypass (the comparison path's single known bypass —
  the engine's own build-time fold, optimizer-on), claim: unshipped-verdict (the
  ask: unshipped-never-compared ruling as a rule), and claim: native-tables (the
  oracle's tables are native tables, and a bare `register` is a different oracle).
- **ask: engine-fold-reading is new**, and it is the decision claim: one-door-bypass
  does not make.
- **Chapters 1, 6 and 9 re-anchor on `Oracle.VERSION`**, which is the object a bump
  moves. It is recorded and not asserted, which is still ask: version-pin — but it is
  now a name in code rather than only a sentence in the 36 markdown files
  claim: oracle-version-constant counts.
- **Two scope corrections against the shipped code**, both measured 2026-09-02 and both
  narrowing a claim rather than weakening it: claim: schema-comparison and
  claim: unshipped-verdict hold on the campaign's **row path** only
  (ticket: static-only-schema-check), and `fuzz.runner`'s verdict tuples have no test at
  all, so claim: contract-surface-gap, claim: optimizer-bracket,
  claim: opt-emulated-classification, claim: abstention-reporting and
  claim: coverage-accounting read `Unverified` for that half
  (ticket: verdict-tuple-test).
