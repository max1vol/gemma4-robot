# Pose Estimation Runtime

This directory contains the low-level custom pose runtime for Raspberry Pi 3B+.
It intentionally does not link TensorFlow Lite, LiteRT, MediaPipe, OpenCV,
NumPy, or XNNPACK as an operator library.

- `pose_runtime.c`: fixed-graph C/NEON pose detector and landmarker executor,
  preprocessing, tracking, and JSON export.
- `a53_pw6x8.S`: local Cortex-A53 FP32 NEON FMA pointwise microkernel adapted
  from XNNPACK source under its BSD-style license.

Build the Pi binary from the repo root with:

```sh
scripts/build_pose_runtime_aarch64.sh out/pose_neon_runtime_aarch64_ofast
```

The current detailed optimization report, measurements, charts, and validation
overlay are in [`REPORT.md`](REPORT.md). XNNPACK comparison material lives under
`research/benchmarks/vs_xnnpack/` as calibration/reference data only; it is not
linked into the runtime.
