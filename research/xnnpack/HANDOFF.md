# XNNPACK 1x1 Convolution Notes For Pose Runtime

This directory contains a shallow clone of the official Google XNNPACK source:

- Source: `research/xnnpack/XNNPACK`
- Remote: `https://github.com/google/XNNPACK.git`
- Inspected commit: `1c292bfc98d0bc412721c335e72f7a188e436c8c`

Purpose: provide a concrete implementation reference for making the custom pose
runtime on Raspberry Pi 3B+ closer to XNNPACK/MediaPipe performance without
pulling in TensorFlow Lite or a large runtime.

## Executive Summary

The important thing to copy from XNNPACK is not "use NEON somewhere". The
important pieces are:

1. Pack weights once into a microkernel-friendly layout.
2. Use small fixed GEMM/IGEMM tiles, especially output-channel tiles of 8.
3. Use Cortex-A53-specific FP32 NEON FMA kernels.
4. Fuse min/max activation into the microkernel.
5. Use persistent worker threads and tile the work, not pthread creation per op.

For the pose model, the first high-value target is 1x1 pointwise convolution.
Most of the model's dense arithmetic should be there, and XNNPACK's design maps
that operation to packed GEMM/IGEMM-style compute rather than raw convolution
loops.

## Relevant XNNPACK Source Files

- `research/xnnpack/XNNPACK/src/configs/gemm-config.c`
- `research/xnnpack/XNNPACK/src/operators/convolution-nhwc.c`
- `research/xnnpack/XNNPACK/src/operator-run.c`
- `research/xnnpack/XNNPACK/src/indirection.c`
- `research/xnnpack/XNNPACK/src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S`
- `research/xnnpack/XNNPACK/src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S`
- `research/xnnpack/XNNPACK/src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S`

## Cortex-A53 FP32 GEMM Configuration

In `src/configs/gemm-config.c`, the AArch64 Cortex-A53 path selects:

- `xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm`
- `xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm`
- `xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm`
- `mr = 6`
- `nr = 8`
- `pack_gemm_gio = xnn_x32_packw_gemm_gio_ukernel_x8__neon_u2`
- `pack_gemm_goi = xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm`

For Raspberry Pi 3B+ running 64-bit userspace, the shape to imitate first is
therefore 6 output pixels by 8 output channels. A 4x8 or 1x8 kernel can handle
tails and is useful as a simpler first implementation.

For 32-bit ARM userspace, XNNPACK uses 4x8 for Cortex-A53. If the Pi runtime is
32-bit, prefer 4x8 rather than 6x8.

## How XNNPACK Packs Convolution Weights

In `src/operators/convolution-nhwc.c`, `create_igemm` computes:

- `nr = gemm_config->nr`
- `kr = 1 << gemm_config->log2_kr`
- `sr = 1 << gemm_config->log2_sr`
- `n_stride = round_up(group_output_channels, nr)`
- `k_stride = round_up_po2(group_input_channels, kr * sr)`

Then packed weights size is:

```text
packed_group_weights_size =
  ((kernel_size * k_stride << log2_filter_element_size)
   + bias_element_size
   + extra_weights_bytes)
  * n_stride
```

For normal convolution it calls `pack_conv_goki_w(...)`; for depthwise-flagged
convolution it calls `pack_conv_kgo_w(...)`.

The key consequence: the runtime inner loop should never walk raw TFLite/OHWI
weights for every output pixel. The weights must be converted once into output
channel tiles of 8 with padded input-channel stride and bias interleaved in the
format expected by the microkernel.

## How IGEMM Is Scheduled

In `src/operators/convolution-nhwc.c`, the IGEMM context stores:

- `ks`: kernel size
- `kc`: input channel bytes
- `w_stride`: packed weight stride
- `indirect_a`: pointer table for input pixels
- `zero`: zero buffer for padding
- `packed_w`: packed weights
- `mr`, `nr`, `kr`, `sr`
- `cm_stride`, `cn_stride`, group/batch strides

It schedules `xnn_compute_igemm` with
`xnn_parallelization_type_4d_tile_2d_dynamic`, ranging over:

- batch
- group
- output-channel tile
- output-pixel tile

In `src/operator-run.c`, `xnn_compute_igemm` eventually calls the chosen
microkernel with:

- `mr_step`
- `nr_block_size`
- `kc`
- `ks_scaled`
- `indirect_a + mr_block_start * ks`
- `packed_w + nr_block_start * w_stride + group_index * gw_stride`
- output pointer and output strides
- activation params

This confirms the structure we should use: split by output-channel tiles and
pixel tiles, then call a fixed microkernel repeatedly.

## 1x1 Conv Recommendation For Our Runtime

For 1x1 stride-1 pointwise conv in NHWC, do not implement full XNNPACK
indirection first. A direct packed GEMM path is simpler:

```text
M = output_height * output_width
K = input_channels
N = output_channels
C[M,N] = A[M,K] * W[K,N] + bias[N]
```

Implementation target:

1. Add model-load/offline packing for every 1x1 conv.
2. Pack output channels in blocks of 8.
3. Pad input channels to the kernel stride/alignment.
4. Interleave bias with each output-channel block.
5. Implement `f32_pointwise_6x8_a53` for AArch64.
6. Add `4x8` or `1x8` tail handling.
7. Fuse min/max activation in the kernel.
8. Dispatch tiles through a persistent worker pool.

The 6x8 kernel should hold accumulators for 6 rows and 8 output channels. Since
8 FP32 channels are two NEON vectors, this is 12 accumulator vectors. On
AArch64 that is realistic. The inner loop loads or broadcasts 6 input scalars
for one input channel and loads two weight vectors for 8 output channels, then
uses FMLA into the accumulators.

Compile flags should include something like:

```sh
-O3 -mcpu=cortex-a53
```

For AArch64, the compiler should emit `fmla` for intrinsics. If the intrinsics
kernel is not close enough, use XNNPACK's assembly as the reference.

## Depthwise Conv Comes Second

After 1x1 is correct and fast, optimize depthwise conv. XNNPACK also has
generated depthwise kernels, but for our fixed model a smaller hand-written
NHWC depthwise path is probably easier:

- channel tiles of 4 or 8
- precomputed row pointers
- zero row/buffer for padding
- no boundary checks inside the inner loop
- fused activation

Do not spend time on a fully general convolution engine before pointwise conv is
fast.

## Threading Recommendation

XNNPACK uses an external/persistent pthreadpool. The custom runtime should do
the same:

- create worker threads once
- keep them alive for the whole inference
- split each op into tiles
- use a barrier or simple work queue per op

Avoid `pthread_create` per operator. On Pi 3B+, that overhead is significant and
will hide gains from NEON kernels.

Measure these variants separately:

- 1 worker
- 2 workers
- 3 workers
- 4 workers

The requested project targets include NEON 2-core, 3-core, and 4-core modes, so
the threadpool should have a runtime worker-count setting.

## Suggested Implementation Order

1. Add per-op timing to the current custom executor if not already present.
2. Confirm top expensive ops are 1x1 pointwise conv.
3. Add packed weight storage for 1x1 conv only.
4. Implement a scalar packed 1x1 path first to validate packing.
5. Implement 1x8 NEON kernel.
6. Implement 4x8 or 6x8 NEON kernel.
7. Compare every replaced layer against the current scalar executor with strict
   max-abs and RMS error checks.
8. Run the full pose image pipeline on `out/human-for-pose.png`.
9. Only then optimize depthwise conv and postprocess.

## What To Tell The Main Coding Agent

The main agent should read this file and then start by replacing the generic
1x1 convolution path with a packed pointwise-GEMM path. The goal is not to link
XNNPACK; the goal is to reproduce the relevant low-level design in the small
custom runtime.

The immediate code shape should be:

```c
typedef struct {
  int in_channels;
  int out_channels;
  int out_channels_padded;
  int in_channels_padded;
  float* packed_bias_weights;
  float activation_min;
  float activation_max;
} packed_pointwise_f32;

void pack_pointwise_f32_x8(...);
void pointwise_f32_1x8_a53(...);
void pointwise_f32_4x8_a53(...);
void pointwise_f32_6x8_a53(...);
void run_pointwise_f32_packed_threaded(...);
```

Keep the existing scalar implementation as the oracle until the full image-level
result matches the official reference.
