#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="$ROOT/research/xnnpack/XNNPACK/build/container-aarch64-release"
REMOTE="max@pi3"
REMOTE_DIR="~/gemma4-robot/xnnpack-release-bench"

ssh_cmd=(tailscale ssh "$REMOTE")

"${ssh_cmd[@]}" "mkdir -p $REMOTE_DIR/bin $REMOTE_DIR/results"

for bin in \
  bench/f32-gemm-bench \
  bench/f32-gemm-minmax-bench \
  bench/f32-igemm-bench \
  bench/f32-dwconv-bench \
  bench/f32-conv-hwc-bench \
  bench/subgraph/subgraph-mobilenet-bench \
  bench/operators/operator-convolution-bench
do
  src="$BUILD_DIR/$bin"
  test -x "$src"
  tar -C "$BUILD_DIR" -cf - "$bin" | "${ssh_cmd[@]}" "tar -C $REMOTE_DIR/bin -xf -"
done

run_remote() {
  local name="$1"
  shift
  "${ssh_cmd[@]}" "cd $REMOTE_DIR && ./bin/$name --benchmark_min_time=1 --benchmark_repetitions=3 $*"
}

run_remote bench/f32-conv-hwc-bench 256 256 3 3 1 1 2 1 3 24 \
  | tee "$ROOT/research/xnnpack/pi_f32_conv_hwc_op3.txt"

run_remote bench/f32-dwconv-bench 128 128 3 3 1 1 1 1 24 \
  | tee "$ROOT/research/xnnpack/pi_f32_dwconv_op6.txt"

run_remote bench/f32-gemm-bench 16384 8 24 16384 32 8 16384 8 8 4096 16 32 \
  | tee "$ROOT/research/xnnpack/pi_f32_gemm_pose_pointwise.txt"

run_remote bench/subgraph/subgraph-mobilenet-bench --benchmark_filter='MobileNetV2/f32' \
  | tee "$ROOT/research/xnnpack/pi_subgraph_mobilenet_v2.txt"
