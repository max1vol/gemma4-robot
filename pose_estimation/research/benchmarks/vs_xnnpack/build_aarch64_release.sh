#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BENCH_DIR="$ROOT/pose_estimation/research/benchmarks/vs_xnnpack"
XNNPACK_DIR="$BENCH_DIR/XNNPACK"
IMAGE="gemma4-xnnpack-build:bookworm-arm64"
BUILD_DIR="build/container-aarch64-release"

if [[ ! -d "$XNNPACK_DIR" ]]; then
  echo "Missing $XNNPACK_DIR" >&2
  echo "Clone XNNPACK with: git clone --depth 1 https://github.com/google/XNNPACK.git '$XNNPACK_DIR'" >&2
  exit 1
fi

if ! container image list | awk '{print $1 ":" $2}' | grep -qx "$IMAGE"; then
  container build \
    --arch arm64 \
    --tag "$IMAGE" \
    --file "$BENCH_DIR/Containerfile" \
    "$BENCH_DIR"
fi

container run --rm --arch arm64 \
  -v "$ROOT:/work" \
  -w "/work/pose_estimation/research/benchmarks/vs_xnnpack/XNNPACK" \
  "$IMAGE" \
  /bin/bash -lc "
    set -euo pipefail
    cmake -S . -B '$BUILD_DIR' -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DXNNPACK_LIBRARY_TYPE=static \
      -DXNNPACK_BUILD_BENCHMARKS=ON \
      -DXNNPACK_BUILD_TESTS=OFF
    cmake --build '$BUILD_DIR' --target \
      f32-gemm-bench \
      f32-gemm-minmax-bench \
      f32-igemm-bench \
      f32-dwconv-bench \
      f32-conv-hwc-bench \
      subgraph-mobilenet-bench \
      operator-convolution-bench \
      -j\"\$(nproc)\"
    file \
      '$BUILD_DIR/bench/f32-gemm-bench' \
      '$BUILD_DIR/bench/f32-igemm-bench' \
      '$BUILD_DIR/bench/f32-dwconv-bench' \
      '$BUILD_DIR/bench/f32-conv-hwc-bench' \
      '$BUILD_DIR/bench/subgraph/subgraph-mobilenet-bench' \
      '$BUILD_DIR/bench/operators/operator-convolution-bench'
  "
