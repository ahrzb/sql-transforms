## 12. ASK index

Every ASK, in document order, with what it binds. All fifteen are open.

| ask | question | at | binds |
|---|---|---|---|
| **ASK-1** | pin `==1.5.5` or floor-plus-assert; and 1.5.5 or the LTS line | 1.3 | ORC-02, ORC-09, ORC-86, every pin |
| **ASK-2** | build-vs-build repeatability: sort-at-freeze, out of contract, or tentative | 3.7 | ORC-22 |
| **ASK-13** | does `threads` join the oracle constant, and what disposition covers order *inside* a value | 3.7 | ORC-02, ORC-14, ORC-22, ORC-75 |
| **ASK-3** | make the accepted severity-4 refusal cost countable, or amend the RFC to say it is unmeasured | 4.3 | ORC-30, ORC-59, D9 |
| **ASK-4** | is `OPT_EMULATED`'s `AGREE` treatment at `oracle.py:740` deliberate | 4.3 | ORC-25 |
| **ASK-5** | do reason codes stay internal, or become user-visible refusal text | 4.3 | ORC-29 |
| **ASK-12** | is the comparison harness's own normalization part of the oracle's answer | 5 | ORC-32, ORC-38, ORC-26, D7, D12 |
| **ASK-6** | bit-for-bit floats and what governs the three exceptions already in force; and the unenforced third home | 5 | ORC-32, ORC-76, ORC-80, ORC-84, D7, D8 |
| **ASK-7** | close TASK-95 or downgrade the five doc-twin totality sites | 7.2 | ORC-54 |
| **ASK-8** | adopt "an unlisted divergence is a bug by definition" | 7.4 | ORC-55 |
| **ASK-9** | admit a `tentative` (measured, not ruled) bucket | 7.4 | ORC-22, ORC-75, D13, D16 |
| **ASK-14** | is a campaign baseline evidence, or a snapshot | 7.4 | ORC-66, D4, D12, D16, ASK-10 |
| **ASK-10** | classify the width residuals before treating them as a defect count | 10 | D13, D16 |
| **ASK-11** | corpus match count: dated generated number, or a ratchet | 10 | ORC-67, ORC-73 |
| **ASK-15** | do this document's own nine `[PROPOSED]` rules become normative | 10 | ORC-15, 31, 35, 41, 56, 58, 59, 65, 84 |

Plus the section 7.3 ledger: **16 rows awaiting a ruling** (D1-D16), of which D13 and
D16 are marked *unruled*, D7 and D8 are ruled together under ASK-6, D12 is ruled by
attribution to D7, and the remainder carry a proposed status that is not in force until
you fill the column.

**What changed since the first draft, in one place.** This revision was written against
four independent audits of the first draft. Every substantive correction is marked in
place with the word *Correction* and the measurement that forced it; the ones that change
what a reader should do are: ORC-05 (four sites -> twelve, two of them under
`src/planner/binder/`, and constant folding *is* removed), ORC-38 / ASK-12 (the
`decimals` tag does not suppress the value comparison, and the normalization cast is not
value-preserving), ledger D12 (four of five "unattributed" residuals were attributed all
along), ORC-32 (three float tolerances are already in force, so ASK-6's question was
mis-posed), ORC-66 (the campaign fuzzer is not a standing gate), and the eight rules this
document had been stating as decided that nobody ever decided (ASK-15). Two ledger rows
were added from a re-sweep of `known_divergences/` (D14, D15) and one from the audit of
the baseline file itself (D16).
