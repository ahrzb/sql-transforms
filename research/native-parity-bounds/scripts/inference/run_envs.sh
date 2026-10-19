#!/bin/bash
# Each config in its own process, env set before python starts.
R=/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/inference
cd /home/user/sql-transforms
NOAVX512="X86_V4,AVX512_ICL,AVX512_SPR"
NOAVX2="X86_V3,X86_V4,AVX512_ICL,AVX512_SPR"
run() { name=$1; shift; env "$@" uv run --frozen python $R/feats.py $name > $R/logs/$name.log 2>&1; }
mkdir -p $R/logs
run default &
run threads1 OPENBLAS_NUM_THREADS=1 &
run haswell OPENBLAS_CORETYPE=Haswell &
run sandybridge OPENBLAS_CORETYPE=Sandybridge &
wait
run npy_noavx512 NPY_DISABLE_CPU_FEATURES=$NOAVX512 &
run avx2_box OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES=$NOAVX512 &
run sse_box OPENBLAS_CORETYPE=Nehalem NPY_DISABLE_CPU_FEATURES=$NOAVX2 &
(env uv run --frozen python $R/feats.py default_rerun pca_all kmeans8 > $R/logs/default_rerun.log 2>&1) &
wait
echo ALLDONE
