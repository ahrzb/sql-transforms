# Research behind the native decisions of 2026-10-06

This folder holds the evidence for the **Methodology** and **Recommendation**
sections of the five decision records in [`../../closed/`](../../closed/).
All five records are about the native catalog. The catalog translates fitted
transformers of sklearn (scikit-learn, a Python library for machine learning).
This README and its table name each record by a short name:

- **matvec:** the parity bound for a matvec entry, in
  [`matvec-parity-bound.md`](../../closed/matvec-parity-bound.md). A matvec
  entry is a catalog entry that computes a matrix-vector product (matvec). The
  record covers three groups of families:
  - the linear projections, for example `PCA`
  - the distances to the fitted centres of `KMeans` (the sklearn clustering
    model) and similar models
  - the kernel samplers, for example `RBFSampler`. A kernel sampler is an
    sklearn transformer that computes an approximate feature map of a kernel.
- **power:** the parity bound for a power transformer, in
  [`power-parity-bound.md`](../../closed/power-parity-bound.md). This is
  sklearn's `PowerTransformer`. It applies a fitted power function to each
  feature, so that the data is more like a normal distribution.
- **additive chi2:** the parity bound for an additive chi2 sampler, in
  [`additive-chi2-parity-bound.md`](../../closed/additive-chi2-parity-bound.md).
  This is sklearn's `AdditiveChi2Sampler`, the kernel sampler for the additive
  chi-squared (chi2) kernel.
- **sparse outputs:** whether the twin should make a sparse output dense, in
  [`sparse-outputs.md`](../../closed/sparse-outputs.md). A sparse output is a
  sparse matrix of scipy, a Python library for scientific computing. A sparse
  matrix stores only its nonzero values.
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
   Each agent read the decision record that its question came from. It also
   read the code of the native catalog and these upstream sources:
   - sklearn and scipy
   - numpy, the Python library for arrays
   - OpenBLAS, a library of linear algebra kernels
   - glibc, the GNU C library, which holds math functions such as `log` and
     `exp`
   - DuckDB, the database of the oracle

   Then each agent measured what the record did not contain yet. Where that
   was cheap, it also measured the record's own numbers again.
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
| [distances.md](distances.md) | matvec (`KMeans` and similar models, kernel samplers) | yes |
| [framework.md](framework.md) | matvec, power, additive chi2 (the general form of a parity bound) | yes |
| [power.md](power.md) | power | yes |
| [chi2.md](chi2.md) | additive chi2 | yes |
| [sparse.md](sparse.md) | sparse outputs | yes |
| [tolerated.md](tolerated.md) | tolerated differences | yes |
| [inference.md](inference.md) | all three bounds, and the owner's amendment | yes |
| [critique.md](critique.md) | all | not applicable |

## Environment

- **Machine.** The research ran on one x86-64 host with 4 CPUs (central
  processing units). x86-64 is the 64-bit instruction set of Intel and AMD
  CPUs. The host has AVX-512 (Advanced Vector Extensions, 512-bit), the
  512-bit vector instructions of x86-64. On this host, numpy reports these
  groups of CPU features:
  - `X86_V4`, the level of x86-64 that includes AVX-512
  - `AVX512_ICL`, the AVX-512 extensions of the Intel Ice Lake CPUs
  - `AVX512_SPR`, the AVX-512 extensions of the Intel Sapphire Rapids CPUs
- **Software.** The research used these versions:
  - numpy 2.5.1, scipy 1.18.0 and scikit-learn 1.9.0
  - OpenBLAS 0.3.33 and glibc 2.39
  - DuckDB 1.5.5
- **OpenBLAS.** OpenBLAS is a library of BLAS (Basic Linear Algebra
  Subprograms) kernels. The research used the scipy-openblas build of
  OpenBLAS, with the build option `DYNAMIC_ARCH`. With this option, OpenBLAS
  selects a kernel for the CPU at run time.
- **Repository.** All the work ran on master at the git commit 113fba7. It ran
  in the virtual environment (venv) of the repository. The package manager
  `uv` manages this venv.
- **Emulation of other CPUs.** The agents used these environment variables to
  select the kernels of other CPUs:
  - `OPENBLAS_CORETYPE` selected other BLAS kernels. OpenBLAS names its
    kernels after Intel CPU generations. The agents used SkylakeX, Haswell,
    Sandybridge, Nehalem and Prescott, from the newest generation to the
    oldest.
  - `NPY_DISABLE_CPU_FEATURES=X86_V4` selected the numpy kernels without
    AVX-512.
  - `GLIBC_TUNABLES` selected the glibc variants without FMA (fused
    multiply-add). An FMA instruction computes a·b + c with one rounding.

  Each setting ran in its own process. All the processes loaded the same
  fitted model from one file. The Python module `pickle` wrote the model to
  this file.
- **What the emulation does not cover.** These settings reproduce the kernels
  that other CPUs would select. They do not reproduce an ARM, macOS or Windows
  host. None of this research measured such a host.
- **The authoring package `sql_transform` does not import in this venv.** The
  package uses the data validation library pydantic 2.13.4. pydantic calls
  `typing._eval_type(prefer_fwd_module=...)`, a private function of Python.
  CPython 3.14.0rc2 does not have this keyword. CPython is the standard Python
  interpreter. The version 3.14.0rc2 is the second release candidate of Python
  3.14.0. The agents used two workarounds:
  - A shim of two lines removes the keyword.
  - The agents loaded `_udf.py` on its own, or they copied the generator of
    the catalog fixtures verbatim. `_udf.py` is the module that holds the twin
    (`PythonTransform`). The catalog fixtures are the test data of the native
    catalog.

  In this venv, pytest (the Python test runner) does not collect the
  repository's own tests. The check (`native.check`) is the test that serves
  the same query with the twin and with the catalog entry. Then it compares
  the results. **Some families have a parity bound that is not bit-exact. None
  of these families went through `native.check` end to end.**

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
(`.pre-commit-config.yaml`) excludes this folder from ruff (the Python linter
and formatter). To run a script again, do these steps:

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
  snippets.** The network proxy of the session blocked these sites:
  - the site of SIAM (the Society for Industrial and Applied Mathematics)
  - the site of Intel
  - the documentation site of the company Oracle
  - duckdb.org
  - arXiv, a site for the preprints of scientific papers

  This affects these citations:
  - the equation numbers in Higham's book *Accuracy and Stability of
    Numerical Algorithms*
  - the paper of Higham and Mary (2019) on a probabilistic bound for rounding
    errors
  - the paper of Jeannerod and Rump (2013) on the rounding error of a dot
    product in any order
  - the accuracy tiers of Intel's SVML (Short Vector Math Library), a library
    of vector math functions such as `log` and `exp`
  - the CNR (Conditional Numerical Reproducibility) mode of Intel's MKL (Math
    Kernel Library). MKL is a math library that includes BLAS kernels. The
    CNR mode makes the results of MKL reproducible under some conditions.
  - §11 of IEEE 754, the standard of the IEEE (Institute of Electrical and
    Electronics Engineers) for floating-point arithmetic

  Each note says which of its claims rest on such a source.
- **Measured maxima are sample maxima.** On its own data, each verifier
  exceeded some maximum that its note reported. For this reason, the
  recommendations declare derived constants. They keep the measurements only
  as evidence.
