#!/bin/bash
set -eu
: "${CUTLASS_ROOT:?CUTLASS directory with include/cutlass}"
: "${CUDA_HOME:?CUDA toolkit directory}"
src=$(cd "$(dirname "$0")/../.." && pwd)/kernels/cuda/nvfp4
out=$1
mkdir -p "$(dirname "$out")"
"$CUDA_HOME/bin/nvcc" -std=c++17 -arch=sm_110a --expt-relaxed-constexpr -O3 --shared --cudart=static -Xcompiler=-fPIC,-fvisibility=hidden -I"$CUTLASS_ROOT/include" "$src/ffn.cu" -o "$out"
