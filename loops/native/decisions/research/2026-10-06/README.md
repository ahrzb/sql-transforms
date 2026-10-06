# Research behind the open native decisions (2026-10-06)

This folder holds the evidence for the **Methodology** and **Recommendation**
sections of the five decision records in [`../../closed/`](../../closed/).
This README and its table name each record by a short name:

- **matvec:** the parity bound for a matvec (matrix-vector product) entry, in
  [`matvec-parity-bound.md`](../../closed/matvec-parity-bound.md).
- **power:** the parity bound for a power transformer, in
  [`power-parity-bound.md`](../../closed/power-parity-bound.md).
- **additive chi2:** the parity bound for an additive chi2 (chi-squared)
  sampler, in
  [`additive-chi2-parity-bound.md`](../../closed/additive-chi2-parity-bound.md).
- **sparse outputs:** whether the twin should make a sparse output dense, in
  [`sparse-outputs.md`](../../closed/sparse-outputs.md).
- **tolerated differences:** where a catalog entry may differ from its twin,
  in [`tolerated-differences.md`](../../closed/tolerated-differences.md).

Each record names the notes that it uses as evidence. Each claim in the notes
has one of these three labels:

- **measured:** The claim names the script, the sample size and the numbers.
- **sourced:** The claim cites a file and a line in the installed packages, or
  a web address (URL).
- **derived:** The claim gives the derivation.

## How the notes were produced

Four steps produced the notes:

1. **Research.** Eight agents did the research, one agent for each question.
   Each agent read the record of its question, the code of the native catalog
   and the upstream sources. The sources were sklearn (scikit-learn), scipy,
   numpy, OpenBLAS, glibc (the GNU C library) and DuckDB. Then each agent
   measured what the record did not contain yet. Where that was cheap, it also
   measured the record's own numbers again.
2. **Adversarial verification.** For each note, a second agent tried to refute
   the key claims of the note. This agent is the verifier. The verifier did
   these three things:
   - It wrote its own scripts, with different data and different random seeds.
   - It derived each bound again, line by line.
   - It opened each source that the note cites.

   The verifier rated each claim as confirmed, partly or refuted. It also wrote
   corrections. The ratings and the corrections are at the end of each note.
   **Where a rating disagrees with the note, the rating wins.**
3. **Critique.** One agent read each note and each verification. It used its
   own experiments to settle the contradictions between them. It also listed
   what none of them answered. Its note is [critique.md](critique.md).
4. **Downstream inference.** One experiment asks the owner's question
   directly. If the catalog entry replaces the twin, does a prediction change?
   The experiment is in [inference.md](inference.md).

The table below lists each note and the records that it supports.

| note | records that it supports | verified |
|---|---|---|
| [matvec.md](matvec.md) | matvec | yes |
| [distances.md](distances.md) | matvec (KMeans and similar models, kernel samplers) | yes |
| [framework.md](framework.md) | matvec, power, additive chi2 (the general form of a parity bound) | yes |
| [power.md](power.md) | power | yes |
| [chi2.md](chi2.md) | additive chi2 | yes |
| [sparse.md](sparse.md) | sparse outputs | yes |
| [tolerated.md](tolerated.md) | tolerated differences | yes |
| [inference.md](inference.md) | all three bounds, and the owner's amendment | yes |
| [critique.md](critique.md) | all | not applicable |

## Environment

- **Machine.** The research ran on one x86-64 host with 4 CPUs (central
  processing units). The host has AVX-512 (Advanced Vector Extensions,
  512-bit). On this host, numpy reports the CPU features `X86_V4`,
  `AVX512_ICL` and `AVX512_SPR`.
- **Software.** The research used these versions:
  - numpy 2.5.1, scipy 1.18.0 and scikit-learn 1.9.0
  - OpenBLAS 0.3.33 and glibc 2.39
  - DuckDB 1.5.5
- **OpenBLAS.** OpenBLAS is a library of BLAS (Basic Linear Algebra
  Subprograms) kernels. The research used the scipy-openblas build of
  OpenBLAS, with the build option `DYNAMIC_ARCH`. With this option, OpenBLAS
  selects a kernel for the CPU at run time.
- **Repository.** All the work ran on master at commit 113fba7. It ran in the
  virtual environment (venv) of the repository. The package manager `uv`
  manages this venv.
- **Emulation of other CPUs.** The agents used these environment variables to
  select the kernels of other CPUs:
  - `OPENBLAS_CORETYPE` selected other BLAS kernels. OpenBLAS names its
    kernels after CPU generations. The agents used SkylakeX, Haswell,
    Sandybridge, Nehalem and Prescott.
  - `NPY_DISABLE_CPU_FEATURES=X86_V4` selected the numpy kernels without
    AVX-512.
  - `GLIBC_TUNABLES` selected the glibc variants without FMA (fused
    multiply-add).

  Each setting ran in its own process. All the processes loaded the same
  fitted model from one file that Python's `pickle` module wrote.
- **What the emulation does not cover.** These settings reproduce the kernels
  that other CPUs would select. They do not reproduce an ARM, macOS or Windows
  host. None of this research measured such a host.
- **The authoring package `sql_transform` does not import in this venv.** The
  data validation library pydantic 2.13.4 calls
  `typing._eval_type(prefer_fwd_module=...)`. CPython 3.14.0rc2 (the second
  release candidate of Python 3.14.0) does not have this keyword. The agents
  used two workarounds:
  - A shim of two lines removes the keyword.
  - The agents loaded `_udf.py` on its own, or they copied the generator of
    the catalog fixtures verbatim. `_udf.py` is the module that holds the twin
    (`PythonTransform`).

  In this venv, the test runner pytest does not collect the repository's own
  tests. The check (`native.check`) is the test that serves the same query
  with the twin and with the catalog entry, and compares the results. **No
  family with a parity bound that is not bit-exact went through
  `native.check` end to end.**

## Scripts

The folder `scripts/<topic>/` holds the scripts and small outputs of the
research agent for a topic. The folder `scripts/verify-<topic>/` holds the
scripts and small outputs of the verifier. In the notes, `$R` and similar
variables pointed to the scratch folder of the session,
`.../scratchpad/research/<topic>/`. That scratch folder corresponds to
`scripts/<topic>/` in this folder. The scripts still contain the absolute
paths of the scratch folder.

The scripts are scratch code. The repository keeps them as evidence, not as
tools. It also keeps them verbatim. The configuration of the pre-commit hooks
(`.pre-commit-config.yaml`) excludes this folder from ruff, the Python linter
and formatter. To run a script again, do these steps:

1. Correct the path in the script.
2. From the root of the repository, run the script with
   `uv run --frozen python`.
3. If the script needs an environment variable, set it in a subprocess before
   the import.

Some content is not in this folder:

- The repository does not hold the large intermediate files. These are
  `pickle` files and numpy array files (`.npz`, `.npy`). The scripts whose
  names match `build_*` and `*_gen` make these files again.
- The repository does not hold the third-party sources that the agents
  downloaded. The notes cite these sources by URL and line.

## Limits

- **The agents cited some primary sources from memory or from search
  snippets.** The network proxy of the session blocked SIAM (the Society for
  Industrial and Applied Mathematics), Intel, the Oracle documentation,
  duckdb.org and arxiv. This affects these citations:
  - the equation numbers in Higham's *Accuracy and Stability*
  - Higham & Mary (2019)
  - Jeannerod & Rump (2013)
  - the accuracy tiers of Intel's SVML (Short Vector Math Library)
  - the CNR (Conditional Numerical Reproducibility) mode of Intel's MKL
    (Math Kernel Library)
  - §11 of IEEE 754, the standard of the IEEE (Institute of Electrical and
    Electronics Engineers) for floating-point arithmetic

  Each note says which of its claims rest on such a source.
- **Measured maxima are sample maxima.** On its own data, each verifier
  exceeded some maximum that its note reported. For this reason, the
  recommendations declare derived constants. They keep the measurements only
  as evidence.
