# XNNPACK Comparison Benchmarks

This directory is a benchmark and implementation-reference area for comparing
the custom Raspberry Pi 3B+ pose runtime against selected upstream XNNPACK
kernels.

It is not part of the production pose runtime. The production runtime remains
the small custom C/NEON code under `pose_estimation/` and does not link
TensorFlow Lite, LiteRT, MediaPipe, OpenCV, NumPy, or XNNPACK.

## Contents

- `HANDOFF.md`: notes on the XNNPACK design details that mattered for the pose
  runtime: packed 1x1 weights, 6x8/4x8/1x8 Cortex-A53 FP32 kernels, fused
  min/max activation, and persistent worker scheduling.
- `pose_prefix_shapes.txt`: pose-model shape notes used when mapping model
  operators onto XNNPACK microbenchmarks.
- `Containerfile`: Debian arm64 build environment for XNNPACK benchmarks.
- `build_aarch64_release.sh`: cross-builds selected upstream XNNPACK benchmark
  binaries on the Mac using Apple `container`.
- `run_pose_shape_benchmarks_on_pi.sh`: copies those benchmark binaries to the
  Pi and runs pose-shape microbenchmarks.
- `XNNPACK/`: local shallow clone of `https://github.com/google/XNNPACK.git`.
  This directory is intentionally git-ignored because it is large upstream
  source, not repo-owned runtime code.

## Setup

From the repository root, clone the upstream source if `XNNPACK/` is not already
present:

```sh
git clone --depth 1 https://github.com/google/XNNPACK.git \
  pose_estimation/research/benchmarks/vs_xnnpack/XNNPACK
```

The clone inspected during the optimization work was:

```text
1c292bfc98d0bc412721c335e72f7a188e436c8c
```

To reproduce that exact snapshot:

```sh
cd pose_estimation/research/benchmarks/vs_xnnpack/XNNPACK
git fetch --depth 1 origin 1c292bfc98d0bc412721c335e72f7a188e436c8c
git checkout 1c292bfc98d0bc412721c335e72f7a188e436c8c
```

## Build Benchmarks

Use the laptop for cross-builds. Do not build XNNPACK on the Pi during live pose
runtime benchmarking.

```sh
pose_estimation/research/benchmarks/vs_xnnpack/build_aarch64_release.sh
```

That script builds an arm64 Release tree under:

```text
pose_estimation/research/benchmarks/vs_xnnpack/XNNPACK/build/container-aarch64-release
```

It builds the benchmark binaries that were useful for calibration, including
`f32-gemm-bench`, `f32-gemm-minmax-bench`, `f32-igemm-bench`,
`f32-dwconv-bench`, `f32-conv-hwc-bench`, and MobileNet/subgraph benchmark
targets.

## Run On Pi

The Pi path used by this project is `max@pi3` through Tailscale. Run:

```sh
pose_estimation/research/benchmarks/vs_xnnpack/run_pose_shape_benchmarks_on_pi.sh
```

The script copies the benchmark binaries to:

```text
~/gemma4-robot/xnnpack-release-bench
```

and writes captured benchmark logs back into this directory as:

```text
pi_f32_conv_hwc_op3.txt
pi_f32_dwconv_op6.txt
pi_f32_gemm_pose_pointwise.txt
pi_subgraph_mobilenet_v2.txt
```

Those output files are small enough to track when they are intentionally
captured, but they should be regenerated when the Pi power state or build
configuration changes.

## How This Was Used

The XNNPACK benchmarks were used to calibrate what the Cortex-A53 can do for
the model's expensive kernels. They showed that the important pattern is not a
general framework dependency, but a concrete schedule:

1. Pack pointwise weights once in output-channel blocks of 8.
2. Tile work over output pixels and output-channel blocks.
3. Use Cortex-A53 FP32 NEON FMA microkernels, especially 6 output pixels by 8
   output channels.
4. Fuse activation clamps into the kernel.
5. Keep a persistent worker pool alive across operators.

The retained runtime implements these ideas directly in `pose_estimation/`.
The local `a53_pw6x8.S` file is a narrow, renamed Cortex-A53 microkernel derived
from the XNNPACK source under its license, but the runtime does not link the
XNNPACK operator library.
