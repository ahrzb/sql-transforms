# Research behind the open native decisions (2026-10-06)

This folder backs the **Methodology** and **Recommendation** sections of the five
records in [`../../open/`](../../open/). Each record names the notes it rests on.
Every claim in the notes is labelled as one of:

- **measured:** names the script, the sample size and the numbers;
- **sourced:** cites a file and line in the installed packages, or a URL;
- **derived:** gives the derivation.

## How the notes were produced

1. **Research.** One agent per question, eight in all. Each read the record, the
   catalog code and the upstream sources (sklearn, scipy, numpy, OpenBLAS, glibc,
   DuckDB). Each then measured what the record did not already contain, and
   re-measured the record's own numbers where that was cheap.
2. **Adversarial verification.** A second agent per note tried to refute its key
   claims. It wrote its own scripts on different data and seeds, re-derived each
   bound line by line, and opened every cited source. Its verdicts (confirmed,
   partly, refuted) and corrections are at the end of each note. **Where they
   disagree with the report, the verdicts win.**
3. **Critique.** One agent read every note and verification. It settled the
   contradictions between them with its own experiments and listed what none of
   them answered: [critique.md](critique.md).
4. **Downstream inference.** One experiment asks the owner's question directly: does
   replacing the twin change a prediction? See [inference.md](inference.md).

| note | records it serves | verified |
|---|---|---|
| [matvec.md](matvec.md) | matvec | yes |
| [distances.md](distances.md) | matvec (KMeans and kin, kernel samplers) | yes |
| [framework.md](framework.md) | matvec, power, additive chi2 (the general form) | yes |
| [power.md](power.md) | power | yes |
| [chi2.md](chi2.md) | additive chi2 | yes |
| [sparse.md](sparse.md) | sparse outputs | yes |
| [tolerated.md](tolerated.md) | tolerated differences | yes |
| [inference.md](inference.md) | all three bounds | **pending** |
| [critique.md](critique.md) | all | n/a |

## Environment

- **Machine:** one x86-64 host with AVX-512 (numpy reports `X86_V4`, `AVX512_ICL`
  and `AVX512_SPR`), 4 CPUs.
- **Software:** numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, OpenBLAS 0.3.33
  (scipy-openblas, `DYNAMIC_ARCH`), glibc 2.39, DuckDB 1.5.5. Everything ran in the
  repo's `uv` venv on master 113fba7.
- **Emulating other CPUs:**
  - Other BLAS kernels were selected with `OPENBLAS_CORETYPE` (SkylakeX, Haswell,
    Sandybridge, Nehalem, Prescott).
  - numpy's non-AVX-512 kernels were selected with
    `NPY_DISABLE_CPU_FEATURES=X86_V4`.
  - glibc's non-FMA variants were selected with `GLIBC_TUNABLES`.
  - Each setting ran in its own process, against one pickled fitted model.
- **What emulation does not cover:** these switches reproduce the kernels other
  CPUs would pick. They do not reproduce an ARM, macOS or Windows host, which none
  of this measured.
- **`sql_transform` does not import in this venv:** pydantic 2.13.4 passes
  `typing._eval_type(prefer_fwd_module=...)`, and CPython 3.14.0rc2 does not have
  it. The agents worked around this in two ways:
  - a two-line shim that drops the keyword;
  - loading `_udf.py` standalone, or copying the fixture generator verbatim.

  The repo's own pytest does not collect here, and **no bounded family went
  through `native.check` end to end.**

## Scripts

`scripts/<topic>/` holds the research agent's scripts and small outputs, and
`scripts/verify-<topic>/` holds the verifier's. In the notes, `$R` and similar
variables pointed at the session scratch folder `.../scratchpad/research/<topic>/`,
which maps to `scripts/<topic>/` here. The scripts still contain those absolute
paths.

They are scratch code, kept as evidence rather than as tools, and kept
verbatim: `.pre-commit-config.yaml` excludes this folder from ruff. To rerun one:

1. Fix its path.
2. Run it with `uv run --frozen python` from the repo root.
3. Set any environment variable before import, in a subprocess.

Some content was left out:

- Large intermediates (pickles, `.npz`, `.npy`) are not committed; the `build_*`
  and `*_gen` scripts regenerate them.
- Third-party sources the agents downloaded are not committed; the notes cite them
  by URL and line.

## Limits

- **Some primary sources were cited from memory or search snippets.** The session's
  network proxy blocked SIAM, Intel, Oracle docs, duckdb.org and arxiv. This affects
  the equation numbers in Higham's *Accuracy and Stability*, Higham & Mary (2019),
  Jeannerod & Rump (2013), Intel's SVML accuracy tiers and MKL CNR, and IEEE 754
  §11. Each note says which of its claims rest on such a source.
- **Measured maxima are sample maxima.** Every verifier exceeded some reported
  maximum on its own data. That is why the recommendations declare derived
  constants and keep measurements only as evidence.
