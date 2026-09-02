## 12. ASK index

### 12.1 Ruled

An ASK leaves this section only by being answered. The ruling text lives at the point in
the document where it binds, next to the claim it created.

| ask | ruling | where it landed | implemented by |
|---|---|---|---|
| **ASK-12** | normalization is out of the oracle's answer and out of the verdict: an unshipped feature FAILS or is CLASSIFIED, never absorbed by weakening a comparison, and a deviation from raw equality happens only via a **named bound in a reviewed draft** (precedent: cbrt 1-ulp, sklearn `1e-9`, DRAFT-23). The 1-ulp decimal deltas in past campaigns were artifacts of the harness's own cast, manufactured by neither engine | ORC-92, chapter 5; ledger row D12's ruling cell | the decimal-to-float64 cast **deleted** from `fuzz/oracle.py`; the `UNSHIPPED` verdict kind and its report section in `fuzz/runner.py`; gated by `tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared` |

### 12.2 Open

Fifteen open, in document order, with what each binds.

| ask | question | at | binds |
|---|---|---|---|
| **ASK-1** | pin `==1.5.5` or floor-plus-assert; and 1.5.5 or the LTS line. The assert's landing spot is one line in `Oracle.__init__` | 1.3 | ORC-02, ORC-09, ORC-86, every pin |
| **ASK-16** | does the engine's build-time fold move to the oracle's reading? It folds optimizer-ON in production while the oracle is optimizer-OFF — latent today, measured | 1.3 | ORC-02, ORC-06, ORC-07, ORC-17, ORC-22, ORC-91 |
| **ASK-2** | build-vs-build repeatability: sort-at-freeze, out of contract, or tentative | 3.7 | ORC-22 |
| **ASK-13** | does `threads` join the oracle constant, and what disposition covers order *inside* a value. Landing spot: one line in `Oracle.__init__` | 3.7 | ORC-02, ORC-14, ORC-22, ORC-75 |
| **ASK-3** | make the accepted severity-4 refusal cost countable, or amend the RFC to say it is unmeasured | 4.3 | ORC-30, ORC-59, D9 |
| **ASK-4** | is `OPT_EMULATED`'s `AGREE` treatment deliberate? `UNSHIPPED` now shares that branch **with** a stated ground; `OPT_EMULATED` still has none | 4.3 | ORC-25, ORC-92 |
| **ASK-5** | do reason codes stay internal, or become user-visible refusal text | 4.3 | ORC-29 |
| **ASK-6** | bit-for-bit floats and what governs the exceptions already in force; and whether `_type_delta`'s unshipped arm earns a gate of its own | 5 | ORC-32, ORC-76, ORC-80, ORC-84, ORC-92, D7, D8 |
| **ASK-7** | close TASK-95 or downgrade the five doc-twin totality sites | 7.2 | ORC-54 |
| **ASK-8** | adopt "an unlisted divergence is a bug by definition" | 7.4 | ORC-55 |
| **ASK-9** | admit a `tentative` (measured, not ruled) bucket | 7.4 | ORC-22, ORC-75, ORC-91, D13, D16 |
| **ASK-14** | is a campaign baseline evidence, or a snapshot | 7.4 | ORC-66, D4, D12, D16, ASK-10 |
| **ASK-10** | classify the width residuals before treating them as a defect count | 10 | D13, D16 |
| **ASK-11** | corpus match count: dated generated number, or a ratchet | 10 | ORC-67, ORC-73 |
| **ASK-15** | do this document's own eight `[PROPOSED]` rules (across nine claim ids) become normative | 10 | ORC-15, 31, 35, 41, 56, 58, 59, 65, 84 |

Plus the section 7.3 ledger: **16 rows**, of which **D12 is now ruled** (the ASK-12
ruling, transcribed into its cell). D13 and D16 are marked *unruled*, D7 and D8 are held
together under ASK-6, and the remainder carry a proposed status that is not in force
until you fill the column.

### 12.3 What changed in this revision

The abstractions this document described have shipped as code, and the document is now
written against them rather than against the arrangement that preceded them. The changes
a reader needs to know about:

- **Non-circularity is stated once, in the README front matter.** The claims are the
  authority and the modules are their enforcement; every claim that has an implementation
  now carries `Enforced-by:` (the module function) and `Verified-by:` (the test), and the
  prose that used to explain where a behavior lived and how its copies were kept in sync
  is deleted — the pointers are that.
- **ASK-12 is ruled**, and the ruling created ORC-92. The consequences are applied
  throughout: ORC-38 no longer authorizes a cast, ORC-24 gains the ranking rule, ORC-28
  gains the report section, ORC-68's normalization blind spot is replaced by the honest
  one, and ledger rows D7 and D12 are corrected.
- **ORC-36 is retired** — one comparison vocabulary made it a restatement of ORC-18 plus a
  default. Its number is kept with a tombstone; nothing is renumbered.
- **Four claims are new.** ORC-90 (what equal means, and the single declared axis),
  ORC-91 (the one-door property's single known bypass — the engine's own build-time fold,
  optimizer-on, latent), ORC-92 (the ASK-12 ruling as a rule), and ORC-93 (the oracle's
  tables are native tables, and a bare `register` is a different oracle).
- **ASK-16 is new**, and it is the decision ORC-91 does not make.
- **Chapters 1, 6 and 9 re-anchor on `Oracle.VERSION`**, which is the object a bump moves.
  It is recorded and not asserted, which is still ASK-1 — but it is now a name in code
  rather than a sentence in thirty-six markdown files.
