# Pin governance metadata

**Question.** Should the pin corpus carry corpus-wide governance metadata (owners,
review state), generic re-record tooling, and evidence mutability classes?

**Ruling.** Not now. Each pin carries the uniform `_pin` header
(`scripts/pin_corpus.py`), and `docs/specs/pins-drift.json` reports drift.

**Reopens when** drift review needs more than the header and the drift report
can give, such as re-recording pins at scale after a DuckDB version bump.
