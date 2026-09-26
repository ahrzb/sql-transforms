# C1 run depth

**Question.** kpi: training-round-trip (C1) specifies 1,500–2,000-case runs; the
standing gate runs the seeded differential at 25 cases (`MARGINALIZE_FUZZ_N`
default in `packages/sql-transform/sql_transform/_projection_test.py`). Which
one is the control?

**Options.**

- **Raise the default to 1,500.** Measured 2026-09-26 on 4 cores: the differential
  passes at 1,500 in 58.8 s, against 7.6 s for the whole file at 25, so about 50 s
  more per gate run.
- **Amend C1's text to 25.** A change to a correctness control, so it goes through
  review per the [success measures](../../specs/success-measures.md#standing-rule).

**Ruling.** None yet.
