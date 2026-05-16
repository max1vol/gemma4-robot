# Pose Runtime Optimization Progress

## Goal Amendment

Optimize the same computation as the official MediaPipe Pose Landmarker Lite model on Raspberry Pi 3B+ for real-time camera use. Allowed low-level paths are:

1. GPU-only using GLES2 fragment-shader compute under `vc4-kms-v3d`.
2. Pure NEON on 2 CPU cores.
3. Pure NEON on 3 CPU cores.
4. Pure NEON on 4 CPU cores.

Do not use raw QPU, VC4CL, or other low-level GPU paths because Chromium kiosk must run at the same time.
Combined GPU+NEON runtime mode was explicitly dropped on 2026-05-16. GPU readback remains acceptable for validation only, not as a candidate runtime split.

The priority order was revised on 2026-05-16:

1. First get pure NEON working end-to-end, including pose estimation outputs, not only isolated network prefixes.
2. Validate the NEON path on `out/human-for-pose.png` against official MediaPipe output run on this laptop.
3. Push the validated NEON code to GitHub.
4. Then port to GPU-only GLES2, measure it, run it on the same image, compare/fix against the official output, and push again.

Progress, plans, experiments, and results should be kept here. A fresh-context subagent should be launched at least every 10 minutes during long runs to suggest new ideas and check whether the main work is cycling without progress.

## Current Model And Environment

- Model: official MediaPipe `pose_landmarker_lite.task`, landmarker model `pose_landmarks_detector.tflite`.
- Pi: Raspberry Pi 3 Model B Plus Rev 1.3, aarch64 Raspberry Pi OS, reachable as `tailscale ssh max@pi3`.
- Official MediaPipe CPU/XNNPACK benchmark on Pi:
  - Synthetic 256x256: about 1.99 FPS.
  - Camera 256x256 YUV420: about 2.05 FPS inference-only, about 1.97 FPS pipeline.
  - Camera 640x480 YUV420: about 1.99 FPS inference-only, about 1.76 FPS pipeline.
- GLES2 works under Mesa VC4:
  - Renderer: `VC4 V3D 2.1`.
  - API: OpenGL ES 2.0.
  - No float render targets observed, so the GPU path is RGBA8/fixed-point oriented.

## Extracted Prefix

The current reproducible prefix covers:

- `op3`: `256x256x3 -> 128x128x24`, 3x3 stride-2 conv, ReLU6, 10.62M MACs.
- `op6`: `128x128x24 -> 128x128x24`, 3x3 depthwise, ReLU6, 3.54M MACs.
- `op9`: `128x128x24 -> 128x128x8`, 1x1 pointwise, no activation, 3.15M MACs.
- `op12`: `128x128x8 -> 128x128x32`, 1x1 pointwise, ReLU6, 4.19M MACs.
- Total prefix: 21.50M MACs.

Extraction files:

- `scripts/extract_pose_prefix_data.py`
- Pi data directory: `~/gemma4-robot/pose-opt/pose_prefix`

The extraction now also emits the next block:

- `op15`: `128x128x8 -> 128x128x8`, 3x3 depthwise, ReLU6, 1.18M MACs.
- `op18`: `128x128x8 -> 128x128x8`, 1x1 pointwise, no activation, 1.05M MACs.
- `op19`: residual add `op9 + op18`, ReLU6.
- `op23`: `128x128x32 -> 64x64x32`, 3x3 stride-2 depthwise, ReLU6, 1.18M MACs.
- `op26`: `64x64x32 -> 64x64x16`, 1x1 pointwise, no activation, 2.10M MACs.
- Total added block: 5.51M MACs.

## Results So Far

### Pure NEON

End-to-end graph executor status:

- Files:
  - `scripts/pose_tflite_inventory.py`
  - `scripts/extract_pose_runtime_data.py`
  - `scripts/pose_litert_reference.py`
  - `scripts/pose_neon_runtime.c`
  - `scripts/pose_official_image_ref.py`
  - `scripts/pose_official_debug_dump.py`
  - `scripts/pose_runtime_pipeline.py`
- Offline extraction now folds constant `DEQUANTIZE` and `DENSIFY` ops out of the target runtime and writes flat C model plans plus raw constant blobs in `out/pose_runtime_data`.
- XNNPACK reference notes were added in `research/xnnpack/HANDOFF.md` from a shallow clone of official Google XNNPACK. The important borrowed design points are weight packing, output-channel tiles of 8, Cortex-A53-specific pointwise GEMM kernels, fused min/max activation, and persistent worker threads. We are not linking XNNPACK; this is being used as a microkernel design reference.
- Target runtime does not link MediaPipe, TensorFlow Lite, LiteRT, OpenCV, or NumPy. It consumes generated model plans and constants and executes supported graph ops directly.
- Supported runtime ops currently cover both official models:
  - detector: `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `PAD`, `RESIZE_BILINEAR`, `DEPTH_TO_SPACE`, `RESHAPE`, `CONCATENATION`.
  - landmarker: `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `PAD`, `RESIZE_BILINEAR`, `MAX_POOL_2D`, `RESHAPE`, `LOGISTIC`.
- Official laptop reference for `out/human-for-pose.png`:
  - `out/human-for-pose-official-mediapipe.json`
  - `out/human-for-pose-official-debug.json`
  - official result has `pose_count=1`.
- Raw model correctness against LiteRT reference tensors:
  - detector outputs: max abs `5.26e-4` on `1x2254x12`, max abs `3.20e-4` on `1x2254x1`.
  - landmarker outputs after pruning segmentation: max abs `8.28e-3` on `1x195`, `2.86e-14` on presence, `5.19e-4` on heatmap, `1.49e-5` on world landmarks.
  - These are float-order differences from the direct executor; output-level agreement is good enough to proceed to postprocessing.
- Landmarker graph pruning:
  - Official comparison disables segmentation masks.
  - MediaPipe only decodes segmentation conditionally, after the model output split.
  - The generated no-segmentation landmarker plan keeps outputs `310`, `315`, `283`, and `312`, and drops tensor `282`.
  - Runtime op count dropped from 129 to 108.
- Laptop image-level pipeline through the C executor:
  - output: `out/human-for-pose-neon-pipeline.json`
  - preprocess uses the model metadata range `[-1, 1]` with zero border.
  - final normalized landmark comparison against official MediaPipe after the current pruned/persistent-pool runtime:
    - max abs x `0.0146`, mean abs x `0.00526`.
    - max abs y `0.00819`, mean abs y `0.00366`.
    - max abs z `0.237`, mean abs z `0.06896`.
  - final world-landmark comparison:
    - max abs x `0.0214`, mean abs x `0.0104`.
    - max abs y `0.0146`, mean abs y `0.00463`.
    - max abs z `0.0661`, mean abs z `0.0171`.
  - Remaining mismatch is dominated by exact MediaPipe `ImageToTensor` resize/crop conventions and detector ROI drift, not raw model inference.
- Pi 3B+ raw full-model timings with initial per-op row threading before pruning:
  - landmarker:
    - 1 core: 1775.6 ms.
    - 2 cores: 1013.1 ms.
    - 3 cores: 798.1 ms.
    - 4 cores: 670.2 ms.
  - detector:
    - 1 core: 3271.0 ms.
    - 2 cores: 1806.2 ms.
    - 3 cores: 1406.0 ms.
    - 4 cores: 1195.2 ms.
  - Combined detector+landmarker 4-core raw inference was about 1.87 s, so this was correct but not optimized enough.
- Pi 3B+ raw full-model timings after segmentation pruning, selective XNNPACK-style pointwise packing for the landmarker, and a persistent worker pool:
  - landmarker:
    - 2 cores: 633.3 ms.
    - 3 cores: 505.5 ms.
    - 4 cores: 442.2 ms.
  - detector:
    - 2 cores: 1763.1 ms.
    - 3 cores: 1349.1 ms.
    - 4 cores: 1105.3 ms.
  - Combined detector+landmarker 4-core raw inference is now about 1.55 s.
  - A first attempt to pack all detector convs in input-major/output-contiguous layout made detector slower on Pi; keep detector on the original layout until a true XNNPACK-style 6x8/4x8 packed GEMM microkernel is implemented.
- 2026-05-16 integrated NEON runtime updates after re-reading `research/xnnpack/HANDOFF.md`:
  - Fixed the persistent-pool lifecycle so initialized `pthread_mutex_t` / `pthread_cond_t` objects are no longer returned by value.
  - Default packing now covers detector and landmarker 1x1 convs, plus the fixed first RGB 3x3 stride-2 conv; non-1x1 convs are no longer packed except that first-conv specialization.
  - Added an AArch64 packed 6-output-pixel x 8-output-channel pointwise path with fused activation.
  - Added a specialized AArch64 first-conv path for `3x3 stride2, RGB -> 24 channels`.
  - Added a no-inner-boundary-check AArch64 depthwise path for `3x3 stride1 SAME`, with checked border pixels and fused activation.
  - Changed PAD to use row-tiled persistent workers instead of a single-thread full-output memset plus row copies.
  - Cross-build command used on laptop:
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -O3 -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64 -lm -pthread'`
  - Pi copy/benchmark pattern:
    - `tar -cf - out/pose_neon_runtime_aarch64 | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64 && ...'`
  - Pi 3B+ timings after packed pointwise + first conv + fast SAME depthwise + parallel PAD:
    - detector 2 cores: 885.1 ms; 3 cores: 646.6 ms; 4 cores: 533.8 ms.
    - landmarker 2 cores: 311.3 ms; 3 cores: 252.3 ms; 4 cores: 237.0 ms.
    - Combined detector+landmarker 4-core raw inference is now about 770.8 ms, versus the previous 1.55 s baseline in this file.
  - Correctness on Pi against LiteRT raw tensor references after these changes:
    - detector: tensor 441 max abs `5.30e-4`, tensor 429 max abs `3.05e-4`.
    - landmarker: tensor 310 max abs `8.28e-3`, tensor 315 max abs `2.86e-14`, tensor 283 max abs `5.19e-4`, tensor 312 max abs `1.49e-5`.
  - Profiles after the first-conv/depthwise changes show detector is now mostly pointwise CONV2D (`~372 ms`) plus DEPTHWISE (`~156 ms`), while landmarker is balanced across CONV2D (`~129 ms`), DEPTHWISE (`~124 ms`), and PAD (`~26 ms`).
  - Added packed 1x1 tiled scheduling over `6 pixel x 8 channel` work items instead of only output rows. This lets small-spatial / large-channel pointwise layers use all workers.
  - Tried parallelizing ADD. It was correct but whole-model timings got worse from worker-dispatch overhead, so the change was reverted.
  - `-Ofast -mcpu=cortex-a53` was tested against raw tensor references and kept errors in the same range. It is currently the fastest measured build flag set.
  - Final 5-repetition Pi timings for the current `-Ofast` AArch64 binary:
    - detector 2 cores: 742.9 ms; 3 cores: 507.3 ms; 4 cores: 452.3 ms.
    - landmarker 2 cores: 281.5 ms; 3 cores: 225.0 ms; 4 cores: 199.9 ms.
    - Combined detector+landmarker raw inference:
      - 2 cores: 1024.4 ms, about 0.98 FPS.
      - 3 cores: 732.3 ms, about 1.37 FPS.
      - 4 cores: 652.1 ms, about 1.53 FPS.
  - The best current raw-model speedup over the previous persistent/padded baseline is about `1.55 s / 0.652 s = 2.38x`.
  - Added a no-boundary-check VALID depthwise path for explicitly padded depthwise layers. A 7-repetition Pi repeat with the same `-Ofast` binary measured:
    - detector 4 cores: 444.0 ms.
    - landmarker 4 cores: 197.5 ms.
    - Combined 4-core raw inference: 641.5 ms, about 1.56 FPS.
  - Re-ran the laptop image-level pipeline on `out/human-for-pose.png` through the current C executor and it still produced `pose_count=1` with the same final landmark comparison as before: max abs x `0.01459`, y `0.00819`, z `0.23707`.
  - Added a guarded first-conv specialization condition for padding/dilation so a regenerated plan cannot accidentally use the RGB stride-2 fast path with different border semantics.
  - Added safe PAD->DEPTHWISE fusion only when the padded tensor has exactly one downstream consumer. Multi-consumer detector head pads are intentionally left materialized.
  - Pi 3B+ 5-repetition timings after safe PAD->DEPTHWISE fusion:
    - detector 3 cores: 542.8 ms; 4 cores: 416.4 ms.
    - landmarker 3 cores: 200.7 ms; 4 cores: 186.3 ms.
    - Combined 4-core raw inference: 602.7 ms, about 1.66 FPS.
  - Current 4-core profile after PAD->DEPTHWISE fusion:
    - detector: CONV2D `315.5 ms`, DEPTHWISE `135.8 ms`, PAD `4.8 ms`.
    - landmarker: CONV2D `99.4 ms`, DEPTHWISE `80.8 ms`, PAD `2.7 ms`.
  - Fresh subagent review agreed the recent work is not cycling. Next high-leverage items are detector reuse for camera mode, real XNNPACK-style A53 pointwise assembly/packing, and further depthwise/PAD fusion.
  - Added a compile-time `POSE_PW_TILE` switch and a 4x8 pointwise tile to compare against the 6x8 tile. The 4x8 version was slower on Pi:
    - default 6x8, 4 cores: detector `425.0 ms`, landmarker `171.9 ms`.
    - `-DPOSE_PW_TILE=4`, 4 cores: detector `503.0 ms`, landmarker `194.5 ms`.
    - Keep 6x8 as the default for this GCC/intrinsics implementation.
  - Added `pipeline-rgb` mode to `scripts/pose_neon_runtime.c`:
    - input: raw RGB24 frame, width, height.
    - does C-side detector letterbox resize, detector decode/anchor scan, rect creation, rotated ROI sampling, landmarker decode, heatmap refinement, projection, world-landmark rotation, and JSON output.
    - This moves the end-to-end pose math out of `scripts/pose_runtime_pipeline.py` for raw RGB frames. PNG loading is still done outside the runtime for the sample image; camera input should provide raw RGB/YUV-converted frames directly.
  - Validation command on laptop:
    - `python3 - <<'PY' ... Image.open('out/human-for-pose.png').convert('RGB').tobytes() ... PY`
    - `/tmp/pose_neon_runtime pipeline-rgb out/pose_runtime_data out/human-for-pose.rgb 514 994 out/human-for-pose-neon-c-pipeline.json 4`
  - C pipeline validation:
    - laptop C pipeline produced `pose_count=1`.
    - C pipeline vs Python validation harness: max abs x `0.02771`, y `0.00452`, z `0.06447`; mean abs x `0.00322`, y `0.00204`, z `0.04305`.
    - C pipeline vs official MediaPipe: max abs x `0.04230`, y `0.01100`, z `0.27116`; mean abs x `0.00646`, y `0.00258`, z `0.10072`.
    - Pi C pipeline output matched Mac C pipeline to float-order noise: max abs x about `1.2e-6`, y about `7.2e-7`, z about `1.1e-5`.
  - Pi `pipeline-rgb` frame timings after initializing packed runtimes before the measured frame section:
    - 2 cores: detector `989.9 ms`, landmarker `309.4 ms`, measured frame section `1324.5 ms`.
    - 3 cores: detector `513.3 ms`, landmarker `250.6 ms`, measured frame section `789.5 ms`.
    - 4 cores: detector `471.3 ms`, landmarker `242.5 ms`, measured frame section `739.9 ms`.
    - These one-shot frame timings were collected while Chromium/Tailscale and other Pi processes were active, so raw 5-rep model timings above remain the cleaner kernel benchmark.
  - Added `pipeline-rgb-rect` mode to measure the detector-reuse path for camera use:
    - input: raw RGB24 frame plus a normalized rect (`x_center`, `y_center`, `width`, `height`, `rotation`).
    - skips detector, runs rotated ROI sampling, landmarker, heatmap refinement, projection, world decode, JSON output.
    - When given the rect from the full C pipeline, laptop `pipeline-rgb-rect` matched the full C pipeline landmarks exactly.
  - Pi `pipeline-rgb-rect` timings with the same rect from `out/human-for-pose-neon-c-pipeline.json`:
    - first run: 2 cores `319.9 ms` frame section, 3 cores noisy at `418.8 ms`, 4 cores `201.2 ms`.
    - repeat: 3 cores `250.5 ms`, 4 cores noisy at `262.7 ms`.
    - Interpretation: ROI-reuse landmarker-only frames are around `200-260 ms` on this loaded Pi, so detector reuse is now the practical route to several FPS before further pointwise assembly work.
  - Updated `pipeline-rgb` and `pipeline-rgb-rect` to accept an optional `reps` argument so timing happens inside one process with packed weights and persistent worker pools reused across frames.
  - Repeated Pi `pipeline-rgb-rect` timings, same rect and RGB frame, `reps=10`:
    - 2 cores: sample/crop `14.36 ms`, landmarker `251.80 ms`, frame `267.83 ms`, `3.73 FPS`.
    - 3 cores: sample/crop `14.19 ms`, landmarker `196.50 ms`, frame `212.25 ms`, `4.71 FPS`.
    - 4 cores: sample/crop `14.02 ms`, landmarker `181.87 ms`, frame `197.46 ms`, `5.06 FPS`.
  - Repeated Pi full detector+landmarker `pipeline-rgb` timings, `reps=5`:
    - 2 cores: detector `712.28 ms`, landmarker `251.57 ms`, frame `986.98 ms`, `1.01 FPS`.
    - 3 cores: detector `480.68 ms`, landmarker `197.44 ms`, frame `701.36 ms`, `1.43 FPS`.
    - 4 cores: detector `429.52 ms`, landmarker `186.30 ms`, frame `639.38 ms`, `1.56 FPS`.
  - Fetched the Pi `pipeline-rgb-rect` 4-core JSON and compared it with the Mac C full-pipeline JSON for the same rect. Max differences were float-order only: x `4.18e-7`, y `4.17e-7`, z `2.09e-6`.
  - This proves the current NEON camera strategy should be: run detector to acquire/reacquire, then run ROI-reuse landmarker frames at about 5 FPS on 4 cores or about 4.7 FPS on 3 cores. The missing production piece is still rect update/tracking across frames, not another one-shot detector benchmark.

Persistent thread-pool prefix runner:

- File: `scripts/pose_neon_prefix_bench.c`
- 1 core: 51.48 ms, 19.43 FPS, 0.418 GMAC/s.
- 2 cores: 28.69 ms, 34.86 FPS, 0.749 GMAC/s.
- 3 cores: 19.01 ms, 52.60 FPS, 1.131 GMAC/s.
- 4 cores: best measured about 17.78 ms, 56.25 FPS, 1.209 GMAC/s.
- Correctness against extracted float references:
  - `op3` max abs about `1.22e-5`.
  - `op6` max abs about `4.58e-5`.
  - `op9` max abs about `1.72e-5`.
  - `op12` max abs about `5.85e-5`.

Single primitive notes:

- `op3` first conv, 4-core NEON: about 6.6 ms.
- `op6` depthwise, optimized 4-core NEON: about 6.31 ms.
- `op9` pointwise, per-op 4-core with thread creation: about 3.36 ms.
- `op12` pointwise, per-op 4-core with thread creation: about 4.37 ms.

Second block persistent runner:

- File: `scripts/pose_neon_block2_bench.c`
- Unfused:
  - 1 core: 21.84 ms, 45.78 FPS, 0.252 GMAC/s.
  - 2 cores: 11.84 ms, 84.47 FPS, 0.465 GMAC/s.
  - 3 cores: 9.31 ms, 107.38 FPS, 0.591 GMAC/s.
  - 4 cores: 9.04 ms, 110.67 FPS, 0.609 GMAC/s.
- Fused `op18+op19`, avoiding the intermediate `op18` tensor write/read and one barrier:
  - 1 core: 21.63 ms, 46.24 FPS, 0.255 GMAC/s.
  - 2 cores: 11.55 ms, 86.60 FPS, 0.477 GMAC/s.
  - 3 cores: 8.72 ms, 114.72 FPS, 0.632 GMAC/s.
  - 4 cores: 8.36 ms, 119.63 FPS, 0.659 GMAC/s.
- Correctness against extracted float references:
  - `op15` max abs about `4.05e-6`.
  - `op19` max abs about `1.91e-6`.
  - `op23` max abs about `4.81e-6`.
  - `op26` max abs about `7.63e-5`.
- The block is memory/barrier-heavy; 3 cores are almost as fast as 4 cores.

### GLES2 GPU

Files:

- `scripts/pi3_gles2_texture_probe.c`
- `scripts/pose_first_conv_gles2.c`

Findings:

- Texture coordinate issue fixed: on this VC4 path, use `gl_FragCoord.x/y` directly, not `gl_FragCoord - 0.5`, when deriving output pixel coordinates for this full-screen draw.
- Texture probe now has exact RGBA8 sampling: `mae_u8=0`.
- `op3` GLES2: 12.05 ms, 82.98 FPS for this one op, 0.881 GMAC/s.
- `op3` correctness: mean abs 0.00527 on `0..6` activation scale; most error is expected RGBA8 input/output quantization.
- `op6` GLES2: 7.17 ms, 139.44 FPS for this one op, 0.493 GMAC/s.
- `op6` correctness update: the apparent max error of 6.0 was from comparing against a CPU-generated quantized op3 reference instead of the actual GLES op3 RGBA8 output. Recomputing op6 reference from the actual GLES op3 bytes gives max abs 0.0176 against float output and 0.0235 against quantized output, with no large outliers.
- `op9` GLES2: 1.44 ms, 2.19 GMAC/s, using affine RGBA8 for signed output range `[-28.797268, 34.145924]`.
- `op12` GLES2: 2.41 ms, 1.74 GMAC/s.
- GPU-only prefix `op3+op6+op9+op12`: 18.52 ms, 54.01 FPS, 1.161 GMAC/s for 21.50M MACs.
- Longer 200-300 rep GPU-only runs are stable around 18.53-18.55 ms for the prefix.
- Actual GPU-byte cascade correctness:
  - `op6`: max abs 0.0176, mean abs 0.00263.
  - `op9`: max abs 0.1847, mean abs 0.0687 in signed affine range; no errors over 1.0.
  - `op12`: max abs 0.0176, mean abs 0.00401.

### Dropped GPU+NEON Mixed

Measured seam-inclusive GPU-to-CPU transfer/conversion in `scripts/pose_first_conv_gles2.c`:

- GPU op3 plus readback/decode of `128x128x24`: 30.58 ms.
- GPU op3+op6 plus readback/decode of `128x128x24`: 36.45 ms.

Measured NEON tails in `scripts/pose_neon_tail_bench.c`:

- Tail after op3, `op6+op9+op12`, 4 cores: 11.33 ms.
- Tail after op6, `op9+op12`, 4 cores: 5.50 ms.
- Tail after op3, `op6+op9+op12`, 3 cores: 11.95 ms.
- Tail after op6, `op9+op12`, 3 cores: 6.20 ms.

Conclusion for this prefix: mixed GPU+NEON with a synchronous readback seam is not competitive. GPU op3 + readback + NEON tail is about 41.9 ms, and GPU op3+op6 + readback + NEON tail is also about 42.0 ms. This mode is now out of scope by user request; keep the whole runtime on GPU-only or pure NEON.

## Active Questions

- Port the current Python image/postprocessing harness into the target C runtime so the pure NEON path is fully standalone from camera frame to pose landmarks.
- Reduce Pi runtime latency. The first complete direct executor is correct but far slower than the official XNNPACK path; detector inference is the largest component.
- Check Chromium kiosk contention for GPU-only and NEON 3/4-core modes.
- Decide whether NEON 3-core should be the practical default because it leaves one core available for Chromium.
- Match MediaPipe `ImageToTensor` resize/crop semantics more exactly. Current image-level landmark x/y agreement is useful, but remaining differences are mostly ROI/preprocess drift.

## Fresh-Context Review Notes

Subagent review agreed that repeatedly revisiting texture offsets would be low-value after the exact sampler probe. The useful diagnostic ladder is raw bytes, coordinate clustering, sentinel clear, scalar single-channel shader, then precision/tap tests. It also flagged that 3-core NEON is likely a practical kiosk mode because it is close to 4-core latency while leaving one core more available for Chromium.

Second subagent review recommended the current priority order: fuse NEON `op18+op19`, then do one clean GPU-only block2 experiment, and avoid more synchronous GPU/CPU seam experiments because readback has already dominated. The user subsequently removed combined GPU+NEON from the goal, so this is now an explicit scope rule.

## Next Experiment

End-to-end NEON reference path:

1. Replace per-op pthread creation in `scripts/pose_neon_runtime.c` with a persistent worker pool.
2. Add per-op timing so optimization is guided by the actual full detector/landmarker bottlenecks on Pi 3B+.
3. Port the Python postprocessing/preprocessing from `scripts/pose_runtime_pipeline.py` into C.
4. Re-run `out/human-for-pose.png` end-to-end on the Pi and compare against `out/human-for-pose-official-mediapipe.json`.
5. Push the validated NEON implementation to GitHub, then resume GPU-only GLES2.

- 2026-05-16 continuation after XNNPACK handoff refresh:
  - Re-read `research/xnnpack/HANDOFF.md`, the current `POSE_PROGRESS.md`, and `scripts/pose_neon_runtime.c` before editing.
  - Rebuilt the current runtime locally and for AArch64 from the laptop:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm`
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
  - Copied the rebuilt binary to Pi with:
    - `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast'`
  - Reproduced the current raw NEON baseline on Pi 3B+ (`reps=3`; first command used missing `/tmp` output directories, so the writes failed after timing, but the timings printed before the write failure):
    - `tailscale ssh max@pi3 'cd ~/gemma4-robot && for t in 2 3 4; do echo RAW-BASE threads=$t; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-t$t $t 3 out/pose_runtime_test_detector/ref; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-t$t $t 3 out/pose_runtime_test_noseg/ref; done'`
    - 2 cores: detector `670.999 ms`, landmarker `258.389 ms`, combined raw `929.388 ms` (`1.08 FPS`).
    - 3 cores: detector `480.146 ms`, landmarker `200.330 ms`, combined raw `680.476 ms` (`1.47 FPS`).
    - 4 cores: detector `435.414 ms`, landmarker `187.877 ms`, combined raw `623.291 ms` (`1.60 FPS`).
  - Re-ran correctness/profile with explicit output directories:
    - `tailscale ssh max@pi3 'cd ~/gemma4-robot && rm -rf /tmp/pose-det-profile /tmp/pose-lm-profile && mkdir -p /tmp/pose-det-profile /tmp/pose-lm-profile && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-profile 4 1 out/pose_runtime_test_detector/ref && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-profile 4 1 out/pose_runtime_test_noseg/ref'`
    - Detector profile totals, 4 cores: `CONV2D 308.283 ms`, `DEPTHWISE 137.404 ms`, `ADD 13.324 ms`, `PAD 4.986 ms`, `RESIZE_BILINEAR 4.102 ms`; total `472.346 ms` for the one profiled rep.
    - Detector correctness: tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.204e-4`.
    - Landmarker profile totals, 4 cores: `CONV2D 121.258 ms`, `DEPTHWISE 111.523 ms`, `ADD 6.601 ms`, `PAD 2.564 ms`, `RESIZE_BILINEAR 2.016 ms`; total `244.977 ms` for the one profiled rep.
    - Landmarker correctness: tensor 310 max abs `8.278e-3`, tensor 315 max abs `2.86e-14`, tensor 283 max abs `5.188e-4`, tensor 312 max abs `1.36e-5`.
  - Fresh-context subagent review said this is not cycling yet, but full detector-every-frame is now the wrong camera benchmark. The high-leverage camera path is detector acquisition/reacquire plus rect-tracked landmarker frames.
  - Implemented MediaPipe-style next-frame rect tracking in `scripts/pose_neon_runtime.c`:
    - Added `rect_from_aux_landmarks_c`, matching the graph path `auxiliary landmarks -> LandmarksToDetectionCalculator -> AlignmentPointsRectsCalculator -> RectTransformationCalculator`.
    - Uses auxiliary landmarks 33 and 34, projects them through the current rect, computes target-angle-90 rotation, square-long rect, and 1.25 scale.
    - Added `pipeline-rgb-track`: detector acquires once, repeated frames run the landmarker using the current rect, and each result updates the rect from the auxiliary landmarks when pose presence is above `0.5`.
    - JSON output now includes `next_rect` when available.
  - Laptop validation of next-rect math:
    - `/tmp/pose_neon_runtime pipeline-rgb out/pose_runtime_data out/human-for-pose.rgb 514 994 out/human-for-pose-neon-c-pipeline.json 4 1`
    - Full C pipeline `next_rect` vs `out/human-for-pose-official-debug.json` `pose_rects_next_frame[0]` differed by about: x `0.00150`, y `0.00017`, width `0.02062`, height `0.01066`, rotation `0.01094 rad`.
    - This is within the current C-vs-official image/preprocess drift and confirms the C tracking formula matches the official graph shape.
  - Pi 3B+ tracked camera-path benchmark after copying the new binary:
    - Command: `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast && for t in 2 3 4; do echo TRACK-REPS threads=$t; ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-neon-c-pipeline-track-t$t.json $t 10; done'`
    - 2 cores: acquisition `975.122 ms`; tracked frame `266.032 ms`; tracked `3.759 FPS`; amortized over only 10 frames `363.544 ms` (`2.751 FPS`).
    - 3 cores: acquisition `532.963 ms`; tracked frame `213.228 ms`; tracked `4.690 FPS`; amortized over only 10 frames `266.524 ms` (`3.752 FPS`).
    - 4 cores: acquisition `479.500 ms`; tracked frame `193.457 ms`; tracked `5.169 FPS`; amortized over only 10 frames `241.407 ms` (`4.142 FPS`).
    - Final presence after the 10 repeated same-image tracking frames was `0.997458`.
  - Fetched the Pi 4-core tracking JSON with:
    - `tailscale ssh max@pi3 'cat /tmp/human-for-pose-neon-c-pipeline-track-t4.json' > out/human-for-pose-neon-c-pipeline-track-pi-t4.json`
    - Pi output had `pose_count=1`, `pose_presence=0.997457564`, selected rect `{x_center:0.549986243,y_center:0.458562106,width:2.26596928,height:1.17173862,rotation:-0.0218091011}`, next rect `{x_center:0.548440397,y_center:0.459108859,width:2.27232337,height:1.17502439,rotation:-0.0295673609}`.
  - Interpretation:
    - The current pure NEON camera strategy is now implemented, not just estimated: run detector for acquisition/reacquire, then use auxiliary-landmark rect tracking for landmarker-only frames.
    - On this loaded Pi 3B+, this reaches `5.17 FPS` on 4 cores and `4.69 FPS` on 3 cores for tracked frames from the sample image. The remaining gap to the earlier theoretical `~6 FPS` target is now mostly landmarker CONV2D/DEPTHWISE kernel speed and RGB ROI sampling cost.

- 2026-05-16 NEON tracked-frame optimization pass:
  - Targeted the measured tracked-camera frame bottleneck: landmarker inference plus rotated RGB ROI sampling.
  - Runtime changes in `scripts/pose_neon_runtime.c`:
    - Added AArch64 helper intrinsics for 8-channel depthwise FMA tiles (`fmaq8_at`) while retaining 4-channel/scalar tails.
    - Updated depthwise paths to process channels in blocks of 8 when possible: checked 3x3 border pixels, 3x3 stride-1 SAME interior, generic VALID depthwise, fused PAD->DEPTHWISE, and the generic SAME fallback.
    - Reworked `sample_rotated_rect_rgb` to use affine stepping across each row instead of recomputing the rotated transform for every output pixel.
  - Build commands:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm`
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
  - Local pipeline sanity:
    - `/tmp/pose_neon_runtime pipeline-rgb out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-neon-c-pipeline-new.json 4 1`
    - `/tmp/pose_neon_runtime pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-neon-c-pipeline-track-new.json 4 10`
    - The full C pipeline still produced `pose_count=1`.
    - New full-pipeline landmarks vs the previous committed C full-pipeline output changed only at sampler float-order scale: max x `1.10e-5`, max y `2.22e-5`, max z `2.63e-4`.
  - Copied to Pi and ran raw correctness plus tracked benchmark:
    - `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast && rm -rf /tmp/pose-det-new /tmp/pose-lm-new && mkdir -p /tmp/pose-det-new /tmp/pose-lm-new && ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-new 4 3 out/pose_runtime_test_detector/ref && ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-new 4 3 out/pose_runtime_test_noseg/ref && for t in 2 3 4; do ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-neon-c-pipeline-track-new-t$t.json $t 10; done'`
    - Correctness remained unchanged against LiteRT raw tensor references:
      - detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.204e-4`.
      - landmarker tensor 310 max abs `8.278e-3`, tensor 315 max abs `2.86e-14`, tensor 283 max abs `5.188e-4`, tensor 312 max abs `1.36e-5`.
    - Tracked camera-path timings, `reps=10`:
      - 2 cores: acquisition `669.704 ms`; sample `9.013 ms`; landmarker `221.527 ms`; tracked frame `232.100 ms`; `4.308 FPS`.
      - 3 cores: acquisition `497.076 ms`; sample `9.036 ms`; landmarker `175.280 ms`; tracked frame `185.872 ms`; `5.380 FPS`.
      - 4 cores: acquisition `453.584 ms`; sample `9.196 ms`; landmarker `159.281 ms`; tracked frame `170.087 ms`; `5.879 FPS`.
    - Compared with the previous tracked benchmark from the same sample:
      - 4-core tracked frame improved from `193.457 ms` / `5.169 FPS` to `170.087 ms` / `5.879 FPS`.
      - 3-core tracked frame improved from `213.228 ms` / `4.690 FPS` to `185.872 ms` / `5.380 FPS`.
      - ROI sampling improved from about `14 ms` to about `9 ms` on Pi.
  - Re-ran raw model timings on Pi with `reps=5` for all requested NEON worker counts:
    - 2 cores: detector `668.715 ms`, landmarker `226.484 ms`, combined raw `895.199 ms` (`1.12 FPS`).
    - 3 cores: detector `441.926 ms`, landmarker `180.705 ms`, combined raw `622.631 ms` (`1.61 FPS`).
    - 4 cores: detector `402.781 ms`, landmarker `163.532 ms`, combined raw `566.313 ms` (`1.77 FPS`).
  - Re-ran full detector-every-frame pipeline on Pi with `reps=5`:
    - 2 cores: detector `621.294 ms`, landmarker `226.207 ms`, frame `866.904 ms`, `1.154 FPS`.
    - 3 cores: detector `461.814 ms`, landmarker `177.183 ms`, frame `657.880 ms`, `1.520 FPS`.
    - 4 cores: detector `406.368 ms`, landmarker `174.890 ms`, frame `600.974 ms`, `1.664 FPS`.
  - Fetched the new Pi 4-core tracking JSON and compared against official MediaPipe output:
    - `pose_count=1`, `pose_presence=0.995636642`.
    - selected rect `{x_center:0.549493849,y_center:0.460266829,width:2.27443814,height:1.1761179,rotation:-0.0179902315}`.
    - next rect `{x_center:0.551523387,y_center:0.459532052,width:2.28196335,height:1.18000913,rotation:-0.0342178345}`.
    - final pose landmarks vs official MediaPipe: max abs x `0.02142`, y `0.00680`, z `0.22151`; mean abs x `0.00522`, y `0.00283`, z `0.07442`.
  - Interpretation:
    - The pure NEON tracked camera path is now within about 2% of the earlier theoretical `~6 FPS` target on 4 cores and exceeds 5 FPS on 3 cores.
    - The next model-compute target is still pointwise CONV2D. Depthwise improved enough that the remaining large raw buckets are detector/landmarker pointwise convolutions and detector acquisition latency.

- 2026-05-16 pointwise 8-channel block packing pass:
  - Fresh-context subagent review independently recommended changing pointwise packed weight layout before tile-size or assembly work.
  - Runtime changes in `scripts/pose_neon_runtime.c`:
    - Added `PosePackedType` and `PackedPointwiseX8`.
    - Kept the first RGB `3x3 stride2` conv on the previous `[kh][kw][input_channel][output_channel]` packed layout.
    - Repacked 1x1 pointwise weights by output-channel blocks of 8 with bias interleaved per block:
      - block layout is `bias[8]` followed by `input_channel * weight_lanes[8]`.
      - tail output channels are padded in the packed data but stores still write only real channels.
    - Updated the 6x8/4x8/1x8 pointwise tile paths to read each 8-output-channel block contiguously instead of stepping by full `out_c` for every input channel.
  - Build commands:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm -pthread`
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
  - Local correctness check before Pi run:
    - `rm -rf /tmp/pose-det-pw-mac /tmp/pose-lm-pw-mac && mkdir -p /tmp/pose-det-pw-mac /tmp/pose-lm-pw-mac`
    - `/tmp/pose_neon_runtime out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-pw-mac 4 1 out/pose_runtime_test_detector/ref`
    - `/tmp/pose_neon_runtime out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-pw-mac 4 1 out/pose_runtime_test_noseg/ref`
    - `/tmp/pose_neon_runtime pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-pw-track-mac.json 4 3`
    - Local tensor diffs stayed in the same range: detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`; landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.49e-5`.
  - Copied to Pi and ran profile/raw/tracked validation:
    - `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast && rm -rf /tmp/pose-det-pw /tmp/pose-lm-pw && mkdir -p /tmp/pose-det-pw /tmp/pose-lm-pw && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-pw 4 1 out/pose_runtime_test_detector/ref && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-pw 4 1 out/pose_runtime_test_noseg/ref && for t in 2 3 4; do rm -rf /tmp/pose-det-pw-t$t /tmp/pose-lm-pw-t$t && mkdir -p /tmp/pose-det-pw-t$t /tmp/pose-lm-pw-t$t; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-pw-t$t $t 5 out/pose_runtime_test_detector/ref; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-pw-t$t $t 5 out/pose_runtime_test_noseg/ref; done && for t in 2 3 4; do ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-pw-track-t$t.json $t 10; done'`
    - Pi correctness remained in-range against LiteRT raw tensor references:
      - detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`.
      - landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.36e-5`.
    - Raw model timings, `reps=5`:
      - 2 cores: detector `398.498 ms`, landmarker `201.264 ms`, combined `599.762 ms` (`1.67 FPS`).
      - 3 cores: detector `300.626 ms`, landmarker `163.721 ms`, combined `464.347 ms` (`2.15 FPS`).
      - 4 cores: detector `299.480 ms`, landmarker `152.849 ms`, combined `452.329 ms` (`2.21 FPS`).
    - Compared to the previous raw-model checkpoint:
      - 4-core detector improved from `402.781 ms` to `299.480 ms`.
      - 4-core landmarker improved from `163.532 ms` to `152.849 ms`.
      - 4-core combined raw inference improved from `566.313 ms` to `452.329 ms`.
    - Tracked camera-path timings, `reps=10`:
      - 2 cores: acquisition `454.202 ms`; sample `9.062 ms`; landmarker `194.039 ms`; tracked frame `204.790 ms`; `4.883 FPS`.
      - 3 cores: acquisition `345.951 ms`; sample `9.266 ms`; landmarker `155.049 ms`; tracked frame `165.836 ms`; `6.030 FPS`.
      - 4 cores: acquisition `366.797 ms`; sample `9.221 ms`; landmarker `143.371 ms`; tracked frame `154.296 ms`; `6.481 FPS`.
    - Compared to the previous tracked checkpoint:
      - 4-core tracked frame improved from `170.087 ms` / `5.879 FPS` to `154.296 ms` / `6.481 FPS`.
      - 3-core tracked frame improved from `185.872 ms` / `5.380 FPS` to `165.836 ms` / `6.030 FPS`.
  - Full detector-every-frame pipeline after pointwise packing, `reps=5`:
    - `tailscale ssh max@pi3 'cd ~/gemma4-robot && for t in 2 3 4; do ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-pw-full-t$t.json $t 5; done'`
    - 2 cores: detector `426.982 ms`, landmarker `223.877 ms`, frame `673.812 ms`, `1.484 FPS`.
    - 3 cores: detector `302.036 ms`, landmarker `157.508 ms`, frame `478.665 ms`, `2.089 FPS`.
    - 4 cores: detector `269.124 ms`, landmarker `168.155 ms`, frame `456.944 ms`, `2.188 FPS`.
  - Fetched Pi JSONs and compared final landmarks to official MediaPipe:
    - tracked output: `pose_count=1`, `pose_presence=0.99358058`, selected rect `{x_center:0.548704624,y_center:0.460995644,width:2.31046009,height:1.19474483,rotation:-0.0186568499}`.
    - tracked final pose landmarks vs official: max abs x `0.02175`, y `0.00626`, z `0.12991`; mean abs x `0.00591`, y `0.00223`, z `0.03345`.
    - full detector-every-frame output: `pose_count=1`, `pose_presence=0.99475795`; final landmarks remain at the same C-vs-official detector/preprocess drift as before, max abs x `0.04231`, y `0.01098`, z `0.27118`.
  - Interpretation:
    - This is the first pointwise packing change that clearly improves both detector acquisition and tracked camera frames.
    - The pure NEON path now exceeds the earlier `~6 FPS` tracked-frame target on both 3 and 4 cores for the provided sample, while preserving raw model correctness.

- 2026-05-16 graph-pattern fusion pass after pointwise packing:
  - Fresh-context review said the main loop is not cycling. It recommended avoiding low-value tile-size churn and targeting concrete residual-block fusions plus A53-specific pointwise improvements next.
  - Runtime changes in `scripts/pose_neon_runtime.c`:
    - Fused the initial `PAD -> 3x3 stride-2 RGB CONV2D` pattern used by both models. The fused path reads the unpadded RGB input directly, handles only the zero-border pixels with checks, and skips the padded tensor write/read.
    - Added fused packed `1x1 CONV2D -> ADD` for residual blocks when the packed pointwise conv output has one consumer, the conv is linear, and the residual/add output shapes match. The fused tile computes the packed x8 pointwise conv, adds the residual tensor, then applies the ADD activation in the same store.
  - Local correctness/build commands:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm -pthread`
    - `/tmp/pose_neon_runtime out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-fusions-mac 4 1 out/pose_runtime_test_detector/ref`
    - `/tmp/pose_neon_runtime out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-fusions-mac 4 1 out/pose_runtime_test_noseg/ref`
    - `/tmp/pose_neon_runtime pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-fusions-track-mac.json 4 3`
    - Tensor diffs stayed in the same range: detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`; landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.49e-5`.
  - Pi build/copy/benchmark command:
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
    - `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast && rm -rf /tmp/pose-det-fusions /tmp/pose-lm-fusions && mkdir -p /tmp/pose-det-fusions /tmp/pose-lm-fusions && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-fusions 4 1 out/pose_runtime_test_detector/ref && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-fusions 4 1 out/pose_runtime_test_noseg/ref && for t in 2 3 4; do rm -rf /tmp/pose-det-fusions-t$t /tmp/pose-lm-fusions-t$t && mkdir -p /tmp/pose-det-fusions-t$t /tmp/pose-lm-fusions-t$t; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-fusions-t$t $t 5 out/pose_runtime_test_detector/ref; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-fusions-t$t $t 5 out/pose_runtime_test_noseg/ref; done && for t in 2 3 4; do ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-fusions-track-t$t.json $t 10; done'`
  - Pi correctness stayed unchanged against LiteRT raw tensor references:
    - detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`.
    - landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.36e-5`.
  - Raw model timings on Pi, `reps=5`:
    - 2 cores: detector `396.620 ms`, landmarker `196.627 ms`, combined raw `593.247 ms` (`1.69 FPS`).
    - 3 cores: detector `298.784 ms`, landmarker `148.924 ms`, combined raw `447.708 ms` (`2.23 FPS`).
    - 4 cores: detector `291.904 ms`, landmarker `136.323 ms`, combined raw `428.227 ms` (`2.34 FPS`).
  - Tracked camera-path timings on Pi, `reps=10`:
    - 2 cores: acquisition `463.899 ms`; sample `9.097 ms`; landmarker `194.802 ms`; tracked frame `205.507 ms`; `4.866 FPS`.
    - 3 cores: acquisition `359.946 ms`; sample `9.013 ms`; landmarker `147.788 ms`; tracked frame `158.468 ms`; `6.310 FPS`.
    - 4 cores: acquisition `319.376 ms`; sample `9.285 ms`; landmarker `138.355 ms`; tracked frame `149.200 ms`; `6.702 FPS`.
  - Full detector-every-frame pipeline on Pi, `reps=5`:
    - 2 cores: detector `414.269 ms`, landmarker `224.533 ms`, frame `658.407 ms`, `1.519 FPS`.
    - 3 cores: detector `315.452 ms`, landmarker `146.772 ms`, frame `481.121 ms`, `2.078 FPS`.
    - 4 cores: detector `310.710 ms`, landmarker `140.000 ms`, frame `470.370 ms`, `2.126 FPS`.
  - End-to-end tracked output on `out/human-for-pose.png` from Pi 4-core run:
    - `pose_count=1`, `pose_presence=0.99358058`.
    - final pose landmarks vs official MediaPipe: max abs x `0.02175`, y `0.00626`, z `0.12991`; mean abs x `0.00591`, y `0.00223`, z `0.03345`.
  - Interpretation:
    - Compared with the pointwise-packing checkpoint, 4-core raw combined inference improved from `452.329 ms` to `428.227 ms`; 3-core improved from `464.347 ms` to `447.708 ms`.
    - Tracked 4-core frames improved from `154.296 ms` / `6.481 FPS` to `149.200 ms` / `6.702 FPS`; 3-core improved from `165.836 ms` / `6.030 FPS` to `158.468 ms` / `6.310 FPS`.
    - The fused path preserved raw tensor correctness and final landmark agreement. Next highest-value work is A53-specific pointwise kernel/K-padding or a fixed `3x3 stride2` depthwise path, not more generic scheduler work.

- 2026-05-16 3x3 stride-1 SAME depthwise adjacent-pixel pass:
  - Target:
    - The retained change only touches the hot `3x3 stride-1 SAME` depthwise path. It computes adjacent interior x pixels together, reusing the same 3 depthwise weight vectors across both outputs and reusing the overlapping source vectors.
    - This is deliberately narrower than a general depthwise rewrite, because raw tensor correctness is already solid and the current tracked path is sensitive to small kernel regressions.
  - Runtime changes in `scripts/pose_neon_runtime.c`:
    - Added AArch64 helpers `fmaq8_depthwise_pair_row` and `fmaq4_depthwise_pair_row`.
    - Added `depthwise_3x3s1_same_pair`.
    - Updated `depthwise_3x3s1_same_rows` to process interior columns two at a time, with the existing single-pixel path retained for left/right borders and odd tails.
  - Failed idea:
    - Also tried hoisting the interior/border branch out of each channel block in fused `PAD -> DEPTHWISE`.
    - It was correct, but the Pi numbers were mixed: 4-core tracked improved slightly, while 3-core tracked and raw 3/4-core model timings regressed. Dropped this change and kept only the adjacent-pixel `3x3 stride-1 SAME` path.
  - Local correctness command:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm -pthread && rm -rf /tmp/pose-det-dw2-final-mac /tmp/pose-lm-dw2-final-mac && mkdir -p /tmp/pose-det-dw2-final-mac /tmp/pose-lm-dw2-final-mac && /tmp/pose_neon_runtime out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-dw2-final-mac 4 1 out/pose_runtime_test_detector/ref && /tmp/pose_neon_runtime out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-dw2-final-mac 4 1 out/pose_runtime_test_noseg/ref`
    - Detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`.
    - Landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.49e-5`.
  - Pi build/copy/final retained benchmark command:
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
    - `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast && for t in 2 3 4; do rm -rf /tmp/pose-det-dw2-final-t$t /tmp/pose-lm-dw2-final-t$t && mkdir -p /tmp/pose-det-dw2-final-t$t /tmp/pose-lm-dw2-final-t$t; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-dw2-final-t$t $t 5 out/pose_runtime_test_detector/ref; ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-dw2-final-t$t $t 5 out/pose_runtime_test_noseg/ref; done && for t in 2 3 4; do ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-dw2-final-track-t$t.json $t 10; done'`
  - Pi environment note:
    - Chromium kiosk and the local voice bot were running during the final retained benchmark.
    - `vcgencmd get_throttled` reported `0x50005`; the governor was `ondemand` with max `1400000`, and `scaling_cur_freq` later returned `1400000`. Treat small differences as noisy.
  - Pi final retained benchmark, raw model timings, `reps=5`:
    - 2 cores: detector `444.669 ms`, landmarker `193.094 ms`, combined raw `637.763 ms` (`1.57 FPS`).
    - 3 cores: detector `291.847 ms`, landmarker `154.795 ms`, combined raw `446.642 ms` (`2.24 FPS`).
    - 4 cores: detector `288.676 ms`, landmarker `148.944 ms`, combined raw `437.620 ms` (`2.29 FPS`).
  - Pi final retained benchmark, tracked camera-path timings, `reps=10`:
    - 2 cores: acquisition `458.544 ms`; sample `9.054 ms`; landmarker `191.681 ms`; tracked frame `202.280 ms`; `4.944 FPS`.
    - 3 cores: acquisition `325.825 ms`; sample `8.965 ms`; landmarker `146.995 ms`; tracked frame `157.608 ms`; `6.345 FPS`.
    - 4 cores: acquisition `340.193 ms`; sample `9.331 ms`; landmarker `136.777 ms`; tracked frame `148.304 ms`; `6.743 FPS`.
  - Interpretation:
    - Against the previous fused-graph checkpoint, the tracked path improved modestly: 4-core tracked frame `149.200 ms -> 148.304 ms`, 3-core `158.468 ms -> 157.608 ms`, and 2-core `205.507 ms -> 202.280 ms`.
    - Raw timings were noisy because of current Pi load/throttling, but the repeated tensor comparisons remained unchanged. The retained change is small, model-local, and directionally useful for the real tracked camera path.
    - Fresh-context review agreed the paired-pixel tap order and bounds are correct, and warned that more small depthwise variants would become low-value unless tied to a clearly profiled stride-2 target.
    - Next higher-leverage work is still pointwise A53-specific tuning or a dedicated fixed-shape depthwise implementation for the remaining named hot ops. Avoid more fused `PAD -> DEPTHWISE` branch rearrangement unless measured in isolation.

- 2026-05-16 pointwise A53 experiments that were rejected:
  - Tried a source-level `6x8` pointwise K-loop unroll by 4 for the hot packed pointwise kernels, including the fused `1x1 CONV2D -> ADD` path.
    - Local tensor correctness was unchanged: detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`; landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.49e-5`.
    - Pi tracked path regressed: 2 cores `211.723 ms` / `4.723 FPS`, 3 cores `164.656 ms` / `6.073 FPS`, 4 cores `158.720 ms` / `6.300 FPS`.
    - Pi raw timings also regressed versus the retained depthwise checkpoint: 2-core combined raw `621.731 ms`, 3-core `467.448 ms`, 4-core `445.277 ms`.
    - Reverted. The compiler/A53 path appears register-pressure sensitive; this simple unroll is worse than the compact loop.
  - Tried forcing `POSE_PW_TILE=4` at build time, so the runtime used the existing `4x8` pointwise path instead of the normal `6x8` tile.
    - Pi tracked path regressed: 3 cores `172.295 ms` / `5.804 FPS`, 4 cores `152.894 ms` / `6.540 FPS`.
    - Pi raw timings were also worse for detector: 3-core detector `411.131 ms`, 4-core detector `319.274 ms`.
    - Rejected. The default `6x8` tile remains better on the current 64-bit Pi 3B+ runtime.
  - Tried changing pointwise work-item order from pixel-tile-major to output-channel-block-major to improve packed-weight reuse.
    - Local tensor correctness was unchanged, but Pi performance regressed heavily.
    - Pi raw timings: 2-core detector `484.724 ms`, landmarker `217.486 ms`; 3-core detector `336.449 ms`, landmarker `172.702 ms`; 4-core detector `318.008 ms`, landmarker `169.824 ms`.
    - Pi tracked path: 2 cores `219.705 ms` / `4.552 FPS`, 3 cores `177.667 ms` / `5.629 FPS`, 4 cores `167.755 ms` / `5.961 FPS`.
    - Reverted. For this model and worker pool, keeping the input pixel tile hot across all output-channel blocks is more important than sweeping one weight block across the whole image.
  - Current conclusion:
    - The retained pointwise design is still the best measured C/intrinsics implementation so far: `6x8`, pixel-tile-major scheduling, packed output-channel blocks of 8, and graph-level residual fusion.
    - Next pointwise work should be either a true A53 assembly microkernel based on the XNNPACK 6x8/4x8 kernels, or a dedicated microbenchmark harness that compares variants outside the full model before touching the integrated runtime again.

- 2026-05-16 single-op benchmark instrumentation:
  - Added a production-path `bench-op` mode to `scripts/pose_neon_runtime.c`:
    - Usage: `pose_neon_runtime bench-op <data_dir> <detector|landmarker> <input_f32.bin> <src_op> [threads] [reps] [warmup]`.
    - It runs the real graph up to the requested source op, preserving the same fusions used by `run_model()` (`PAD -> RGB stride2 CONV`, `1x1 CONV2D -> ADD`, and `PAD -> DEPTHWISE`).
    - It snapshots non-constant benchmark inputs after the prefix and restores them before each warmup/timed rep, so repeated timings do not depend on stale mutated intermediates.
    - It prints the selected fused/standalone kind, output shape, MAC count, GMAC/s, and FP32 FMA-equivalent GFLOP/s.
    - Set `POSE_BENCH_FUSED=0` to force standalone `run_op()` dispatch for comparison.
  - Local build and correctness commands:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm -pthread`
    - `rm -rf /tmp/pose-det-bench-check /tmp/pose-lm-bench-check && mkdir -p /tmp/pose-det-bench-check /tmp/pose-lm-bench-check && /tmp/pose_neon_runtime out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-bench-check 4 1 out/pose_runtime_test_detector/ref && /tmp/pose_neon_runtime out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-bench-check 4 1 out/pose_runtime_test_noseg/ref`
    - Local tensor diffs remained unchanged: detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`; landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.49e-5`.
    - Local smoke examples:
      - `/tmp/pose_neon_runtime bench-op out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin 204 4 10 1`
      - `/tmp/pose_neon_runtime bench-op out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin 012 4 10 1`
  - Pi cross-build/copy command:
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
    - `tar -cf - out/pose_neon_runtime_aarch64_ofast | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_ofast ...'`
  - Pi correctness check after the refactor:
    - Command inside the copy run: `./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-benchop 4 1 out/pose_runtime_test_detector/ref && ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-benchop 4 1 out/pose_runtime_test_noseg/ref`
    - Detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`.
    - Landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.36e-5`.
  - Pi op-level benchmark command pattern, `reps=20`, `warmup=2`:
    - Detector ops: `204` pointwise CONV2D, `053` depthwise, `003` fused initial `PAD+CONV2D`, `151` fused `PAD+DEPTHWISE`.
    - Landmarker ops: `012` pointwise CONV2D, `058` depthwise, `003` fused initial `PAD+CONV2D`, `023` fused `PAD+DEPTHWISE`.
  - Pi op-level timings:
    - Detector `204` pointwise CONV2D, out `[1,14,14,192]`, `43,352,064` MACs:
      - 2 threads: `37.440 ms`, `1.158 GMAC/s`, `2.316 GFLOP/s`.
      - 3 threads: `18.697 ms`, `2.319 GMAC/s`, `4.637 GFLOP/s`.
      - 4 threads: `20.896 ms`, `2.075 GMAC/s`, `4.149 GFLOP/s`.
    - Detector `053` depthwise, out `[1,28,28,240]`, `4,704,000` MACs:
      - 2 threads: `18.237 ms`, `0.258 GMAC/s`, `0.516 GFLOP/s`.
      - 3 threads: `13.709 ms`, `0.343 GMAC/s`, `0.686 GFLOP/s`.
      - 4 threads: `14.146 ms`, `0.333 GMAC/s`, `0.665 GFLOP/s`.
    - Detector `003` fused initial `PAD+CONV2D`, out `[1,112,112,24]`, `8,128,512` MACs:
      - 2 threads: `8.497 ms`, `0.957 GMAC/s`, `1.913 GFLOP/s`.
      - 3 threads: `6.655 ms`, `1.221 GMAC/s`, `2.443 GFLOP/s`.
      - 4 threads: `4.248 ms`, `1.913 GMAC/s`, `3.827 GFLOP/s`.
    - Detector `151` fused `PAD+DEPTHWISE`, out `[1,7,7,672]`, `823,200` MACs:
      - 2 threads: `4.911 ms`, `0.168 GMAC/s`, `0.335 GFLOP/s`.
      - 3 threads: `4.684 ms`, `0.176 GMAC/s`, `0.351 GFLOP/s`.
      - 4 threads: `2.618 ms`, `0.314 GMAC/s`, `0.629 GFLOP/s`.
    - Landmarker `012` pointwise CONV2D, out `[1,128,128,32]`, `4,194,304` MACs:
      - 2 threads: `3.606 ms`, `1.163 GMAC/s`, `2.326 GFLOP/s`.
      - 3 threads: `2.955 ms`, `1.419 GMAC/s`, `2.839 GFLOP/s`.
      - 4 threads: `3.552 ms`, `1.181 GMAC/s`, `2.362 GFLOP/s`.
    - Landmarker `058` depthwise, out `[1,32,32,144]`, `3,686,400` MACs:
      - 2 threads: `12.539 ms`, `0.294 GMAC/s`, `0.588 GFLOP/s`.
      - 3 threads: `9.627 ms`, `0.383 GMAC/s`, `0.766 GFLOP/s`.
      - 4 threads: `6.850 ms`, `0.538 GMAC/s`, `1.076 GFLOP/s`.
    - Landmarker `003` fused initial `PAD+CONV2D`, out `[1,128,128,24]`, `10,616,832` MACs:
      - 2 threads: `10.868 ms`, `0.977 GMAC/s`, `1.954 GFLOP/s`.
      - 3 threads: `7.065 ms`, `1.503 GMAC/s`, `3.005 GFLOP/s`.
      - 4 threads: `8.592 ms`, `1.236 GMAC/s`, `2.471 GFLOP/s`.
    - Landmarker `023` fused `PAD+DEPTHWISE`, out `[1,64,64,32]`, `1,179,648` MACs:
      - 2 threads: `5.476 ms`, `0.215 GMAC/s`, `0.431 GFLOP/s`.
      - 3 threads: `5.097 ms`, `0.231 GMAC/s`, `0.463 GFLOP/s`.
      - 4 threads: `3.278 ms`, `0.360 GMAC/s`, `0.720 GFLOP/s`.
  - Pi tracked-frame sanity after the refactor, `reps=5`:
    - Command: `tailscale ssh max@pi3 'cd ~/gemma4-robot && for t in 2 3 4; do ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-benchop-track-t$t.json $t 5; done'`
    - 2 cores: acquisition `443.214 ms`; sample `9.229 ms`; landmarker `192.192 ms`; tracked frame `203.157 ms`; `4.922 FPS`.
    - 3 cores: acquisition `363.503 ms`; sample `9.221 ms`; landmarker `150.839 ms`; tracked frame `161.902 ms`; `6.177 FPS`.
    - 4 cores: acquisition `346.428 ms`; sample `9.364 ms`; landmarker `135.055 ms`; tracked frame `146.214 ms`; `6.839 FPS`.
  - Failed idea from the new harness:
    - Tried capping packed pointwise work to 3 compute threads in 4-thread mode, while leaving depthwise and other ops on the requested worker count.
    - Pi A/B command used detector op `204` and tracked 4-core frames, with `POSE_POINTWISE_MAX_THREADS=4` forcing the original behavior for comparison.
    - The cap regressed detector op `204` from `22.169 ms` to `33.301 ms` and tracked 4-core frames from `152.059 ms` to `160.336 ms` in the same run.
    - Reverted. The earlier 3-thread-vs-4-thread differences are too load-sensitive to justify an integrated scheduler heuristic.
  - Interpretation:
    - This commit is instrumentation, not a new kernel-speed claim. It preserves the current end-to-end runtime and gives a lower-noise way to test A53 pointwise/depthwise microkernel changes.
    - The op-level results show why blind full-model pointwise changes were hard to judge: several 4-thread samples are slower than 3-thread samples under current Pi load, while fused stride/depthwise ops scale differently from large pointwise ops.
    - Next implementation work should use `bench-op` to compare a true A53 assembly or carefully constrained intrinsics pointwise microkernel against source op `204` and a few smaller landmarker pointwise ops before touching the integrated path.

- 2026-05-16 post-`bench-op` kernel selection pass:
  - Refreshed the current Pi profile with the retained runtime:
    - `tailscale ssh max@pi3 'cd ~/gemma4-robot && rm -rf /tmp/pose-det-profile-now /tmp/pose-lm-profile-now && mkdir -p /tmp/pose-det-profile-now /tmp/pose-lm-profile-now && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-profile-now 4 1 out/pose_runtime_test_detector/ref && POSE_PROFILE=1 ./out/pose_neon_runtime_aarch64_ofast out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-profile-now 4 1 out/pose_runtime_test_noseg/ref && ./out/pose_neon_runtime_aarch64_ofast pipeline-rgb-track out/pose_runtime_data out/human-for-pose.rgb 514 994 /tmp/human-for-pose-profile-now-track-t4.json 4 8'`
    - Detector profile totals: `CONV2D 278.759 ms`, `DEPTHWISE 166.745 ms`, `ADD 2.109 ms`, `RESIZE_BILINEAR 5.573 ms`, `PAD 5.017 ms`.
    - Detector top ops: `053 DEPTHWISE 27.113 ms`, `204 CONV2D 26.307 ms`, `029 DEPTHWISE 21.689 ms`, `042 DEPTHWISE 14.499 ms`, `119 DEPTHWISE 13.576 ms`, `014 CONV2D 13.177 ms`.
    - Landmarker profile totals: `CONV2D 174.804 ms`, `DEPTHWISE 154.485 ms`, `ADD 3.741 ms`, `RESIZE_BILINEAR 3.944 ms`.
    - Landmarker top ops: `058 DEPTHWISE 28.218 ms`, `003 CONV2D 25.298 ms`, `023 DEPTHWISE 18.307 ms`, `006 DEPTHWISE 18.248 ms`, `032 DEPTHWISE 16.775 ms`, `049 DEPTHWISE 14.293 ms`, `029 CONV2D 14.281 ms`.
    - Tracked 4-core sanity: acquisition `296.682 ms`; sample `9.357 ms`; landmarker `141.748 ms`; tracked frame `152.756 ms`; `6.546 FPS`.
  - Failed depthwise idea:
    - Tried a 4-adjacent-pixel `3x3 stride1 SAME` depthwise interior kernel to reuse weights across four neighboring output columns.
    - Local tensor diffs stayed unchanged, but Pi op results were mixed and not strong enough to retain.
    - Pi `bench-op` after the quad path, `reps=30`, `warmup=3`:
      - detector `053`: 3 threads `24.362 ms`, 4 threads `13.333 ms`.
      - detector `029`: 3 threads `7.954 ms`, 4 threads `7.859 ms`.
      - landmarker `058`: 3 threads `9.210 ms`, 4 threads `8.394 ms`.
      - landmarker `032`: 3 threads `6.126 ms`, 4 threads `6.146 ms`.
    - The tracked 4-core run was essentially flat at `146.644 ms` / `6.819 FPS`, while landmarker `058` regressed against the prior `6.850 ms` checkpoint. Reverted the quad path.
  - XNNPACK A53 6x8 assembly calibration:
    - Built an experimental binary that called XNNPACK's generated Cortex-A53 `f32-gemm-6x8-minmax` assembly only for standalone packed `p_count=6, oc_count=8` pointwise tiles. This was a calibration experiment only; the retained/default runtime remains XNNPACK-free.
    - Build command:
      - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -DPOSE_USE_XNNPACK_A53_GEMM -std=c11 -Wall -Wextra -I out/pose_runtime_data -I research/xnnpack/XNNPACK scripts/pose_neon_runtime.c research/xnnpack/XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S -o out/pose_neon_runtime_aarch64_xnn6x8_exp -lm -pthread'`
    - Correctness stayed in the same range against raw tensor references.
    - Pi `bench-op` result for detector `204` improved materially:
      - 3 threads: `13.119 ms`, `3.305 GMAC/s`, `6.609 GFLOP/s`.
      - 4 threads: `16.607 ms`, `2.611 GMAC/s`, `5.221 GFLOP/s`.
    - Landmarker `012` was roughly flat: 3 threads `2.990 ms`, 4 threads `2.896 ms`.
    - Tracked 4-core frames stayed roughly flat at `147.558 ms` / `6.777 FPS` because the hot tracked path still spends most pointwise time in fused residual `CONV2D -> ADD` tiles that the experiment did not replace.
    - Restored and recopied the default XNNPACK-free binary afterward:
      - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_ofast -lm -pthread'`
      - Default 4-core tracked sanity after restore: acquisition `330.347 ms`; sample `10.419 ms`; landmarker `134.875 ms`; tracked frame `147.004 ms`; `6.803 FPS`.
  - Interpretation:
    - Do not keep exploring C-level depthwise adjacent-column variants unless a very specific hot op is isolated first; the 4-column variant did not translate into end-to-end gain.
    - The true next implementation target is now clearer: reimplement the XNNPACK-style A53 6x8 pointwise schedule inside our custom runtime, then apply it to both standalone and fused `1x1 CONV2D -> ADD` tiles. Detector `204` proves the schedule can be faster; tracked-frame gains require the fused-add variant too.

- 2026-05-16 local A53 `6x8` inline-asm pointwise attempt:
  - Implemented a compile-flagged local AArch64 inline-asm microkernel for full `p_count=6`, `oc_count=8` standalone packed pointwise tiles, then used the existing C/intrinsics path as the oracle.
  - Local correctness command:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm -pthread && rm -rf /tmp/pose-det-a53local /tmp/pose-lm-a53local && mkdir -p /tmp/pose-det-a53local /tmp/pose-lm-a53local && /tmp/pose_neon_runtime out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-a53local 4 1 out/pose_runtime_test_detector/ref && /tmp/pose_neon_runtime out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-a53local 4 1 out/pose_runtime_test_noseg/ref`
  - Pi cross-build command for the experimental binary:
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -DPOSE_USE_A53_PW6X8_LOCAL -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_a53local -lm -pthread'`
  - First local asm schedule:
    - Detector `204`: 3 threads `19.207 ms`, 4 threads `22.563 ms`.
    - Landmarker `012`: 3 threads `3.056 ms`, 4 threads `3.003 ms`.
    - Tracked 4-core frame: `152.432 ms`, `6.560 FPS`.
  - K-by-2 unrolled local asm schedule:
    - Detector `204`: 3 threads `30.951 ms`, 4 threads `22.235 ms`.
    - Landmarker `012`: 3 threads `3.550 ms`, 4 threads `4.371 ms`.
    - Tracked 4-core frame: `146.533 ms`, `6.824 FPS`.
  - Rejected and removed before commit:
    - The local asm did not reproduce the XNNPACK A53 schedule. It was correct, but it was slower than the retained C/intrinsics path on the important standalone op and did not produce a reliable tracked-frame gain.
    - No runtime code from this experiment is retained. The next pointwise attempt should either mirror XNNPACK's instruction scheduling much more faithfully in a separate `.S`/microkernel-style file, or first target the fused `1x1 CONV2D -> ADD` path where tracked-frame time is actually exposed.
  - Additional small-code experiments that were rejected in the same pass:
    - Tried replacing `floorf` with integer truncation in model `RESIZE_BILINEAR` after clamping, and also tried a negative-safe integer floor helper in the camera/ROI samplers. Local raw tensor diffs stayed unchanged.
    - Pi A/B for `RESIZE_BILINEAR` source ops `186`, `188`, `190` was not stable enough to keep. One run improved `188` from `0.813 ms` to `0.238 ms`, but a repeated run had the retained binary at `0.109/0.238/0.771 ms` and the truncation binary slightly slower at `0.116/0.247/0.775 ms`.
    - The ROI sampler helper also moved tracked sample time the wrong way in repeated `pipeline-rgb-track` runs, so all sampler changes were reverted.
    - Tried adding `restrict` qualifiers to the packed pointwise tile input/residual/output pointers. Correctness stayed unchanged, but tracked 4-core performance regressed in the measured run from `149.389 ms` / `6.694 FPS` to `152.696 ms` / `6.549 FPS`. Reverted.
    - Lesson: do not keep micro-edits that only improve isolated noisy op samples. For this runtime, retain changes only when they improve the tracked landmarker frame or a consistently hot fused op with repeated A/B evidence.

- 2026-05-16 retained landmarker `RESIZE_BILINEAR -> ADD` fusion:
  - Targeted the three landmarker decoder skip merges at source ops `186->187`, `188->189`, and `190->191`.
  - Runtime changes in `scripts/pose_neon_runtime.c`:
    - Added `can_fuse_resize_add` and `op_resize_bilinear_add`, guarded by single-consumer and shape checks.
    - The fused path writes the ADD output directly, skips the intermediate resized tensor, and vectorizes the resize-plus-residual-add channel loop on AArch64.
    - Added `bench-op` support for `RESIZE_BILINEAR+ADD`, so either the resize source op or the following add source op can time the fused pair.
  - Local correctness command:
    - `cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm -pthread && rm -rf /tmp/pose-det-resizeadd /tmp/pose-lm-resizeadd && mkdir -p /tmp/pose-det-resizeadd /tmp/pose-lm-resizeadd && /tmp/pose_neon_runtime out/pose_runtime_data detector out/pose_runtime_test_detector/detector_input.bin /tmp/pose-det-resizeadd 4 1 out/pose_runtime_test_detector/ref && /tmp/pose_neon_runtime out/pose_runtime_data landmarker out/pose_runtime_test_noseg/landmarker_input.bin /tmp/pose-lm-resizeadd 4 1 out/pose_runtime_test_noseg/ref`
    - Detector tensor 441 max abs `5.264e-4`, tensor 429 max abs `3.357e-4`.
    - Landmarker tensor 310 max abs `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.49e-5`.
  - Pi cross-build/copy command:
    - `container run --rm --arch arm64 -v "$PWD:/work" -w /work gemma4-xnnpack-build:bookworm-arm64 /bin/bash -lc 'gcc -Ofast -mcpu=cortex-a53 -std=c11 -Wall -Wextra -I out/pose_runtime_data scripts/pose_neon_runtime.c -o out/pose_neon_runtime_aarch64_resizeadd -lm -pthread'`
    - `tar -cf - out/pose_neon_runtime_aarch64_resizeadd | tailscale ssh max@pi3 'cd ~/gemma4-robot && tar -xf - && chmod +x out/pose_neon_runtime_aarch64_resizeadd ...'`
  - Pi fused-op benchmark, 4 threads, `reps=30`, `warmup=3`:
    - Old op `186` + `187`: `0.111296 + 0.048967 = 0.160263 ms`; fused `186->187`: `0.118091 ms`.
    - Old op `188` + `189`: `0.226710 + 0.219992 = 0.446702 ms`; fused `188->189`: `0.277648 ms`.
    - Old op `190` + `191`: `0.720188 + 1.106153 = 1.826341 ms`; fused `190->191`: `1.466114 ms`.
  - Pi tracked-frame A/B:
    - Three 4-core alternating runs, `reps=12`, averaged old `152.257 ms` and new `150.636 ms`; landmarker averaged old `141.011 ms` and new `139.338 ms`.
    - 2-core one-shot, `reps=10`: old `206.006 ms` / `4.854 FPS`; new `200.170 ms` / `4.996 FPS`.
    - 3-core one-shot, `reps=10`: old `161.258 ms` / `6.201 FPS`; new `156.260 ms` / `6.400 FPS`.
    - 4-core one-shot, `reps=10`: old `148.867 ms` / `6.717 FPS`; new `150.105 ms` / `6.662 FPS`.
    - Longer 4-core run, `reps=30`: old `146.070 ms` / `6.846 FPS`; new `145.557 ms` / `6.870 FPS`.
  - Final default binary refresh:
    - Rebuilt `out/pose_neon_runtime_aarch64_ofast` with the retained fusion and copied it to the Pi.
    - Final Pi raw tensor check with the default binary stayed unchanged: detector tensor 441 max abs `5.264e-4`, tensor 429 `3.357e-4`; landmarker tensor 310 `8.278e-3`, tensor 315 `2.86e-14`, tensor 283 `5.188e-4`, tensor 312 `1.36e-5`.
    - Final 4-core tracked sanity, `reps=20`: acquisition `309.567 ms`; sample `9.330 ms`; landmarker `135.364 ms`; tracked frame `146.331 ms`; `6.834 FPS`; amortized over 20 frames `161.809 ms` / `6.180 FPS`.
  - Interpretation:
    - This is a retained graph-level fusion rather than a low-level pointwise replacement. It removes most standalone ADD time in the landmarker decoder and gives a small but measurable tracked-frame improvement under repeated A/B, especially on 2 and 3 cores.
    - The 4-core tracked path remains noisy under current Pi load, so future claimed gains should continue using alternating old/new runs plus at least one longer run.
    - Fresh-context review found no blocking correctness issue with the fusion guards, noted that AArch64 FMA can be non-bit-identical to separate resize then add, and recommended the Pi raw-output, fused-op, and repeated tracked A/B validations recorded above.
    - Next high-leverage runtime work is still the fused residual pointwise microkernel or a targeted depthwise kernel for the current profile's named hot ops.
