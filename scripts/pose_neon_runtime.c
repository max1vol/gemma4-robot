/*
 * Low-level Pose model executor.
 *
 * Build after generating out/pose_runtime_data with extract_pose_runtime_data.py:
 *   cc -O3 -std=c11 -Wall -Wextra -I out/pose_runtime_data \
 *      scripts/pose_neon_runtime.c -o /tmp/pose_neon_runtime -lm
 *
 * This file intentionally does not link MediaPipe, TensorFlow Lite, or LiteRT.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifndef POSE_PW_TILE
#define POSE_PW_TILE 6
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum {
  POSE_TENSOR_UNSUPPORTED = 0,
  POSE_TENSOR_FLOAT32 = 1,
  POSE_TENSOR_INT32 = 2,
} PoseTensorType;

typedef enum {
  POSE_OP_ADD = 1,
  POSE_OP_CONCAT = 2,
  POSE_OP_CONV2D = 3,
  POSE_OP_DEPTHWISE = 4,
  POSE_OP_DEPTH_TO_SPACE = 5,
  POSE_OP_LOGISTIC = 6,
  POSE_OP_MAX_POOL2D = 7,
  POSE_OP_PAD = 8,
  POSE_OP_RESHAPE = 9,
  POSE_OP_RESIZE_BILINEAR = 10,
} PoseOpCode;

typedef struct {
  int index;
  int type;
  int rank;
  int dims[4];
  int is_const;
  uint32_t const_offset;
  uint32_t byte_count;
} PoseTensorDef;

typedef struct {
  int index;
  int op;
  int input_count;
  int inputs[8];
  int output_count;
  int outputs[4];
  int activation;
  int padding;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int filter_h;
  int filter_w;
  int axis;
  int block_size;
  int half_pixel_centers;
  int align_corners;
  int depth_multiplier;
} PoseOpDef;

typedef struct {
  const char* name;
  const char* const_file;
  int input_count;
  int inputs[4];
  int output_count;
  int outputs[8];
  const PoseTensorDef* tensors;
  int tensor_count;
  const PoseOpDef* ops;
  int op_count;
} PoseModelDef;

#ifndef POSE_MODEL_PLAN
#define POSE_MODEL_PLAN "pose_models_plan.inc"
#endif
#include POSE_MODEL_PLAN

typedef struct PoseRuntime PoseRuntime;
typedef void (*PoseRangeFn)(PoseRuntime* rt, const PoseOpDef* op, int y0, int y1);

typedef struct {
  PoseRuntime* rt;
  int id;
} PoolWorker;

typedef enum {
  POSE_PACK_NONE = 0,
  POSE_PACK_ICOC = 1,
  POSE_PACK_POINTWISE_X8 = 2,
} PosePackedType;

typedef struct {
  int in_c;
  int out_c;
  int oc_blocks;
  int block_stride;
  float data[];
} PackedPointwiseX8;

struct PoseRuntime {
  const PoseModelDef* def;
  uint8_t* constants;
  size_t constants_bytes;
  void** tensor;
  void** packed;
  uint8_t* packed_type;
  uint8_t* owned;
  int threads;
  int pool_workers;
  pthread_t worker_tids[7];
  PoolWorker worker_args[7];
  pthread_mutex_t pool_mutex;
  pthread_cond_t pool_start;
  pthread_cond_t pool_done;
  int pool_generation;
  int pool_done_count;
  int pool_stop;
  int task_threads;
  int task_rows;
  const PoseOpDef* task_op;
  const PoseOpDef* task_aux_op;
  PoseRangeFn task_fn;
};

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void* xaligned_alloc(size_t alignment, size_t bytes) {
  if (bytes == 0) bytes = 1;
  size_t padded = (bytes + alignment - 1) & ~(alignment - 1);
  void* ptr = aligned_alloc(alignment, padded);
  if (!ptr) {
    fprintf(stderr, "allocation failed for %zu bytes\n", bytes);
    exit(2);
  }
  return ptr;
}

static size_t file_size(FILE* f) {
  if (fseek(f, 0, SEEK_END) != 0) return 0;
  long n = ftell(f);
  if (n < 0) return 0;
  rewind(f);
  return (size_t)n;
}

static void read_file_exact(const char* path, void* dst, size_t bytes) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (fread(dst, 1, bytes, f) != bytes) {
    fprintf(stderr, "short read: %s\n", path);
    exit(2);
  }
  fclose(f);
}

static void write_file_exact(const char* path, const void* src, size_t bytes) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (fwrite(src, 1, bytes, f) != bytes) {
    fprintf(stderr, "short write: %s\n", path);
    exit(2);
  }
  fclose(f);
}

static size_t tensor_elements(const PoseTensorDef* t) {
  size_t n = 1;
  for (int i = 0; i < t->rank; ++i) n *= (size_t)t->dims[i];
  return n;
}

static size_t tensor_bytes(const PoseTensorDef* t) {
  switch (t->type) {
    case POSE_TENSOR_FLOAT32:
    case POSE_TENSOR_INT32:
      return tensor_elements(t) * sizeof(float);
    default:
      return 0;
  }
}

static inline float relu6f(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 6.0f) return 6.0f;
  return x;
}

static inline float activate(float x, int activation) {
  return activation == 3 ? relu6f(x) : x;
}

static void* pool_worker_main(void* ptr) {
  PoolWorker* arg = (PoolWorker*)ptr;
  PoseRuntime* rt = arg->rt;
  int id = arg->id;
  int seen_generation = 0;
  pthread_mutex_lock(&rt->pool_mutex);
  for (;;) {
    while (!rt->pool_stop && rt->pool_generation == seen_generation) {
      pthread_cond_wait(&rt->pool_start, &rt->pool_mutex);
    }
    if (rt->pool_stop) break;
    seen_generation = rt->pool_generation;
    int task_threads = rt->task_threads;
    int rows = rt->task_rows;
    const PoseOpDef* op = rt->task_op;
    PoseRangeFn fn = rt->task_fn;
    pthread_mutex_unlock(&rt->pool_mutex);

    if (id < task_threads) {
      int y0 = (rows * id) / task_threads;
      int y1 = (rows * (id + 1)) / task_threads;
      fn(rt, op, y0, y1);
    }

    pthread_mutex_lock(&rt->pool_mutex);
    rt->pool_done_count++;
    if (rt->pool_done_count == rt->pool_workers) {
      pthread_cond_signal(&rt->pool_done);
    }
  }
  pthread_mutex_unlock(&rt->pool_mutex);
  return NULL;
}

static void parallel_rows_aux(PoseRuntime* rt, const PoseOpDef* op, const PoseOpDef* aux_op, int rows, PoseRangeFn fn) {
  int threads = rt->threads;
  if (threads <= 1 || rows < 2) {
    rt->task_aux_op = aux_op;
    fn(rt, op, 0, rows);
    rt->task_aux_op = NULL;
    return;
  }
  if (threads > rows) threads = rows;
  if (threads > 8) threads = 8;
  if (threads <= 1 || rt->pool_workers <= 0) {
    rt->task_aux_op = aux_op;
    fn(rt, op, 0, rows);
    rt->task_aux_op = NULL;
    return;
  }

  pthread_mutex_lock(&rt->pool_mutex);
  rt->task_threads = threads;
  rt->task_rows = rows;
  rt->task_op = op;
  rt->task_aux_op = aux_op;
  rt->task_fn = fn;
  rt->pool_done_count = 0;
  rt->pool_generation++;
  pthread_cond_broadcast(&rt->pool_start);
  pthread_mutex_unlock(&rt->pool_mutex);

  int y0 = 0;
  int y1 = rows / threads;
  fn(rt, op, y0, y1);

  pthread_mutex_lock(&rt->pool_mutex);
  while (rt->pool_done_count < rt->pool_workers) {
    pthread_cond_wait(&rt->pool_done, &rt->pool_mutex);
  }
  rt->task_aux_op = NULL;
  pthread_mutex_unlock(&rt->pool_mutex);
}

static void parallel_rows(PoseRuntime* rt, const PoseOpDef* op, int rows, PoseRangeFn fn) {
  parallel_rows_aux(rt, op, NULL, rows, fn);
}

#if defined(__aarch64__)
static inline float32x4_t relu6q(float32x4_t v) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  return vminq_f32(vmaxq_f32(v, zero), six);
}

static inline float32x4_t activateq(float32x4_t v, int activation) {
  return activation == 3 ? relu6q(v) : v;
}

static inline void fmaq4_at(float32x4_t* acc, const float* src, const float* w) {
  *acc = vfmaq_f32(*acc, vld1q_f32(src), vld1q_f32(w));
}

static inline void fmaq8_at(
    float32x4_t* acc0, float32x4_t* acc1, const float* src, const float* w) {
  *acc0 = vfmaq_f32(*acc0, vld1q_f32(src), vld1q_f32(w));
  *acc1 = vfmaq_f32(*acc1, vld1q_f32(src + 4), vld1q_f32(w + 4));
}

static inline void fmaq8_depthwise_pair_row(
    float32x4_t* a0, float32x4_t* a1, float32x4_t* b0, float32x4_t* b1,
    const float* row, const float* wrow, int c) {
  float32x4_t p00 = vld1q_f32(row);
  float32x4_t p01 = vld1q_f32(row + 4);
  float32x4_t p10 = vld1q_f32(row + c);
  float32x4_t p11 = vld1q_f32(row + c + 4);
  float32x4_t p20 = vld1q_f32(row + 2 * c);
  float32x4_t p21 = vld1q_f32(row + 2 * c + 4);
  float32x4_t p30 = vld1q_f32(row + 3 * c);
  float32x4_t p31 = vld1q_f32(row + 3 * c + 4);
  float32x4_t w00 = vld1q_f32(wrow);
  float32x4_t w01 = vld1q_f32(wrow + 4);
  float32x4_t w10 = vld1q_f32(wrow + c);
  float32x4_t w11 = vld1q_f32(wrow + c + 4);
  float32x4_t w20 = vld1q_f32(wrow + 2 * c);
  float32x4_t w21 = vld1q_f32(wrow + 2 * c + 4);
  *a0 = vfmaq_f32(*a0, p00, w00);
  *a1 = vfmaq_f32(*a1, p01, w01);
  *b0 = vfmaq_f32(*b0, p10, w00);
  *b1 = vfmaq_f32(*b1, p11, w01);
  *a0 = vfmaq_f32(*a0, p10, w10);
  *a1 = vfmaq_f32(*a1, p11, w11);
  *b0 = vfmaq_f32(*b0, p20, w10);
  *b1 = vfmaq_f32(*b1, p21, w11);
  *a0 = vfmaq_f32(*a0, p20, w20);
  *a1 = vfmaq_f32(*a1, p21, w21);
  *b0 = vfmaq_f32(*b0, p30, w20);
  *b1 = vfmaq_f32(*b1, p31, w21);
}

static inline void fmaq4_depthwise_pair_row(
    float32x4_t* a, float32x4_t* b, const float* row, const float* wrow, int c) {
  float32x4_t p0 = vld1q_f32(row);
  float32x4_t p1 = vld1q_f32(row + c);
  float32x4_t p2 = vld1q_f32(row + 2 * c);
  float32x4_t p3 = vld1q_f32(row + 3 * c);
  float32x4_t w0 = vld1q_f32(wrow);
  float32x4_t w1 = vld1q_f32(wrow + c);
  float32x4_t w2 = vld1q_f32(wrow + 2 * c);
  *a = vfmaq_f32(*a, p0, w0);
  *b = vfmaq_f32(*b, p1, w0);
  *a = vfmaq_f32(*a, p1, w1);
  *b = vfmaq_f32(*b, p2, w1);
  *a = vfmaq_f32(*a, p2, w2);
  *b = vfmaq_f32(*b, p3, w2);
}
#endif

static int same_pad_before(int in, int out, int stride, int kernel, int dilation) {
  int effective = (kernel - 1) * dilation + 1;
  int total = (out - 1) * stride + effective - in;
  if (total < 0) total = 0;
  return total / 2;
}

static float* pack_conv_weights_icoc(const PoseTensorDef* wt, const float* weights) {
  int out_c = wt->dims[0];
  int kh = wt->dims[1];
  int kw = wt->dims[2];
  int in_c = wt->dims[3];
  size_t count = (size_t)kh * kw * in_c * out_c;
  float* packed = (float*)xaligned_alloc(64, count * sizeof(float));
  for (int ky = 0; ky < kh; ++ky) {
    for (int kx = 0; kx < kw; ++kx) {
      for (int ci = 0; ci < in_c; ++ci) {
        for (int oc = 0; oc < out_c; ++oc) {
          packed[(((size_t)ky * kw + kx) * in_c + ci) * out_c + oc] =
              weights[(((size_t)oc * kh + ky) * kw + kx) * in_c + ci];
        }
      }
    }
  }
  return packed;
}

static PackedPointwiseX8* pack_pointwise_x8(
    const PoseTensorDef* wt, const float* weights, const float* bias) {
  int out_c = wt->dims[0];
  int in_c = wt->dims[3];
  int oc_blocks = (out_c + 7) / 8;
  int block_stride = 8 + in_c * 8;
  size_t bytes = sizeof(PackedPointwiseX8) + (size_t)oc_blocks * block_stride * sizeof(float);
  PackedPointwiseX8* packed = (PackedPointwiseX8*)xaligned_alloc(64, bytes);
  packed->in_c = in_c;
  packed->out_c = out_c;
  packed->oc_blocks = oc_blocks;
  packed->block_stride = block_stride;
  for (int ob = 0; ob < oc_blocks; ++ob) {
    int oc0 = ob * 8;
    float* block = packed->data + (size_t)ob * block_stride;
    for (int lane = 0; lane < 8; ++lane) {
      int oc = oc0 + lane;
      block[lane] = oc < out_c ? bias[oc] : 0.0f;
    }
    float* wblock = block + 8;
    for (int ci = 0; ci < in_c; ++ci) {
      for (int lane = 0; lane < 8; ++lane) {
        int oc = oc0 + lane;
        wblock[(size_t)ci * 8 + lane] =
            oc < out_c ? weights[(size_t)oc * in_c + ci] : 0.0f;
      }
    }
  }
  return packed;
}

static void op_pad_rows(PoseRuntime* rt, const PoseOpDef* op, int y0, int y1) {
  const float* src = (const float*)rt->tensor[op->inputs[0]];
  const int32_t* pads = (const int32_t*)rt->tensor[op->inputs[1]];
  float* dst = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* in = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  int in_h = in->dims[1], in_w = in->dims[2], c = in->dims[3];
  int out_w = out->dims[2];
  int top = pads[2], left = pads[4];
  size_t out_row_bytes = (size_t)out_w * c * sizeof(float);
  size_t copy_bytes = (size_t)in_w * c * sizeof(float);
  for (int oy = y0; oy < y1; ++oy) {
    float* drow = dst + (size_t)oy * out_w * c;
    memset(drow, 0, out_row_bytes);
    int sy = oy - top;
    if ((unsigned)sy < (unsigned)in_h) {
      const float* srow = src + (size_t)sy * in_w * c;
      memcpy(drow + (size_t)left * c, srow, copy_bytes);
    }
  }
}

static void op_pad(PoseRuntime* rt, const PoseOpDef* op) {
  const PoseTensorDef* in = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  if (in->rank != 4 || out->rank != 4) {
    fprintf(stderr, "PAD only supports rank4 tensors\n");
    exit(2);
  }
  parallel_rows(rt, op, out->dims[1], op_pad_rows);
}

static void conv2d_1x1(
    const float* input, const float* weights, const float* bias, float* output,
    int y0, int y1, int w, int in_c, int out_c, int activation, int packed_layout) {
#if defined(__aarch64__)
  if (!packed_layout) {
    for (int y = y0; y < y1; ++y) {
      for (int x = 0; x < w; ++x) {
        size_t p = (size_t)y * w + x;
        const float* src = input + p * in_c;
        float* dst = output + p * out_c;
        int oc = 0;
        for (; oc + 4 <= out_c; oc += 4) {
          float32x4_t acc = vld1q_f32(bias + oc);
          for (int ci = 0; ci < in_c; ++ci) {
            const float* wv = weights + ((size_t)oc * in_c + ci);
            float lane[4] = {wv[0], wv[in_c], wv[2 * in_c], wv[3 * in_c]};
            acc = vfmaq_n_f32(acc, vld1q_f32(lane), src[ci]);
          }
          acc = activateq(acc, activation);
          vst1q_f32(dst + oc, acc);
        }
        for (; oc < out_c; ++oc) {
          float acc = bias[oc];
          const float* wv = weights + (size_t)oc * in_c;
          for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * wv[ci];
          dst[oc] = activate(acc, activation);
        }
      }
    }
    return;
  }

  size_t p_begin = (size_t)y0 * w;
  size_t p_end = (size_t)y1 * w;
  size_t p = p_begin;
  for (; p + 6 <= p_end; p += 6) {
    const float* s0 = input + (p + 0) * in_c;
    const float* s1 = input + (p + 1) * in_c;
    const float* s2 = input + (p + 2) * in_c;
    const float* s3 = input + (p + 3) * in_c;
    const float* s4 = input + (p + 4) * in_c;
    const float* s5 = input + (p + 5) * in_c;
    float* d0 = output + (p + 0) * out_c;
    float* d1 = output + (p + 1) * out_c;
    float* d2 = output + (p + 2) * out_c;
    float* d3 = output + (p + 3) * out_c;
    float* d4 = output + (p + 4) * out_c;
    float* d5 = output + (p + 5) * out_c;
    int oc = 0;
    for (; oc + 8 <= out_c; oc += 8) {
      float32x4_t a00 = vld1q_f32(bias + oc + 0), a01 = vld1q_f32(bias + oc + 4);
      float32x4_t a10 = a00, a11 = a01;
      float32x4_t a20 = a00, a21 = a01;
      float32x4_t a30 = a00, a31 = a01;
      float32x4_t a40 = a00, a41 = a01;
      float32x4_t a50 = a00, a51 = a01;
      for (int ci = 0; ci < in_c; ++ci) {
        const float* wv = weights + (size_t)ci * out_c + oc;
        float32x4_t w0v = vld1q_f32(wv + 0);
        float32x4_t w1v = vld1q_f32(wv + 4);
        a00 = vfmaq_n_f32(a00, w0v, s0[ci]); a01 = vfmaq_n_f32(a01, w1v, s0[ci]);
        a10 = vfmaq_n_f32(a10, w0v, s1[ci]); a11 = vfmaq_n_f32(a11, w1v, s1[ci]);
        a20 = vfmaq_n_f32(a20, w0v, s2[ci]); a21 = vfmaq_n_f32(a21, w1v, s2[ci]);
        a30 = vfmaq_n_f32(a30, w0v, s3[ci]); a31 = vfmaq_n_f32(a31, w1v, s3[ci]);
        a40 = vfmaq_n_f32(a40, w0v, s4[ci]); a41 = vfmaq_n_f32(a41, w1v, s4[ci]);
        a50 = vfmaq_n_f32(a50, w0v, s5[ci]); a51 = vfmaq_n_f32(a51, w1v, s5[ci]);
      }
      a00 = activateq(a00, activation); a01 = activateq(a01, activation);
      a10 = activateq(a10, activation); a11 = activateq(a11, activation);
      a20 = activateq(a20, activation); a21 = activateq(a21, activation);
      a30 = activateq(a30, activation); a31 = activateq(a31, activation);
      a40 = activateq(a40, activation); a41 = activateq(a41, activation);
      a50 = activateq(a50, activation); a51 = activateq(a51, activation);
      vst1q_f32(d0 + oc + 0, a00); vst1q_f32(d0 + oc + 4, a01);
      vst1q_f32(d1 + oc + 0, a10); vst1q_f32(d1 + oc + 4, a11);
      vst1q_f32(d2 + oc + 0, a20); vst1q_f32(d2 + oc + 4, a21);
      vst1q_f32(d3 + oc + 0, a30); vst1q_f32(d3 + oc + 4, a31);
      vst1q_f32(d4 + oc + 0, a40); vst1q_f32(d4 + oc + 4, a41);
      vst1q_f32(d5 + oc + 0, a50); vst1q_f32(d5 + oc + 4, a51);
    }
    for (; oc < out_c; ++oc) {
      float a0 = bias[oc], a1 = bias[oc], a2 = bias[oc];
      float a3 = bias[oc], a4 = bias[oc], a5 = bias[oc];
      for (int ci = 0; ci < in_c; ++ci) {
        float wv = weights[(size_t)ci * out_c + oc];
        a0 += s0[ci] * wv; a1 += s1[ci] * wv; a2 += s2[ci] * wv;
        a3 += s3[ci] * wv; a4 += s4[ci] * wv; a5 += s5[ci] * wv;
      }
      d0[oc] = activate(a0, activation); d1[oc] = activate(a1, activation);
      d2[oc] = activate(a2, activation); d3[oc] = activate(a3, activation);
      d4[oc] = activate(a4, activation); d5[oc] = activate(a5, activation);
    }
  }
  for (; p < p_end; ++p) {
    const float* src = input + p * in_c;
    float* dst = output + p * out_c;
    int oc = 0;
    for (; oc + 8 <= out_c; oc += 8) {
      float32x4_t acc0 = vld1q_f32(bias + oc);
      float32x4_t acc1 = vld1q_f32(bias + oc + 4);
      for (int ci = 0; ci < in_c; ++ci) {
        const float* wv = weights + (size_t)ci * out_c + oc;
        float v = src[ci];
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(wv), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(wv + 4), v);
      }
      acc0 = activateq(acc0, activation);
      acc1 = activateq(acc1, activation);
      vst1q_f32(dst + oc, acc0);
      vst1q_f32(dst + oc + 4, acc1);
    }
    for (; oc < out_c; ++oc) {
      float acc = bias[oc];
      for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * weights[(size_t)ci * out_c + oc];
      dst[oc] = activate(acc, activation);
    }
  }
#else
  if (!packed_layout) {
    for (int y = y0; y < y1; ++y) {
      for (int x = 0; x < w; ++x) {
        size_t p = (size_t)y * w + x;
        const float* src = input + p * in_c;
        float* dst = output + p * out_c;
        for (int oc = 0; oc < out_c; ++oc) {
          float acc = bias[oc];
          const float* wv = weights + (size_t)oc * in_c;
          for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * wv[ci];
          dst[oc] = activate(acc, activation);
        }
      }
    }
    return;
  }
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < w; ++x) {
    size_t p = (size_t)y * w + x;
    const float* src = input + p * in_c;
    float* dst = output + p * out_c;
    for (int oc = 0; oc < out_c; ++oc) {
      float acc = bias[oc];
      for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * weights[(size_t)ci * out_c + oc];
      dst[oc] = activate(acc, activation);
    }
    }
  }
#endif
}

static void conv2d_3x3s2_rgb24_packed(
    const float* input, const float* weights, const float* bias, float* output,
    int y0, int y1, int in_w, int out_w, int activation) {
#if defined(__aarch64__)
  for (int oy = y0; oy < y1; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      const float* base = input + ((size_t)oy * 2 * in_w + ox * 2) * 3;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      float32x4_t acc4 = vld1q_f32(bias + 16);
      float32x4_t acc5 = vld1q_f32(bias + 20);
      for (int ky = 0; ky < 3; ++ky) {
        const float* row = base + (size_t)ky * in_w * 3;
        for (int kx = 0; kx < 3; ++kx) {
          const float* src = row + kx * 3;
          for (int ci = 0; ci < 3; ++ci) {
            const float* wv = weights + ((ky * 3 + kx) * 3 + ci) * 24;
            float v = src[ci];
            acc0 = vfmaq_n_f32(acc0, vld1q_f32(wv + 0), v);
            acc1 = vfmaq_n_f32(acc1, vld1q_f32(wv + 4), v);
            acc2 = vfmaq_n_f32(acc2, vld1q_f32(wv + 8), v);
            acc3 = vfmaq_n_f32(acc3, vld1q_f32(wv + 12), v);
            acc4 = vfmaq_n_f32(acc4, vld1q_f32(wv + 16), v);
            acc5 = vfmaq_n_f32(acc5, vld1q_f32(wv + 20), v);
          }
        }
      }
      acc0 = activateq(acc0, activation);
      acc1 = activateq(acc1, activation);
      acc2 = activateq(acc2, activation);
      acc3 = activateq(acc3, activation);
      acc4 = activateq(acc4, activation);
      acc5 = activateq(acc5, activation);
      float* dst = output + ((size_t)oy * out_w + ox) * 24;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
      vst1q_f32(dst + 8, acc2);
      vst1q_f32(dst + 12, acc3);
      vst1q_f32(dst + 16, acc4);
      vst1q_f32(dst + 20, acc5);
    }
  }
#else
  for (int oy = y0; oy < y1; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      const float* base = input + ((size_t)oy * 2 * in_w + ox * 2) * 3;
      float* dst = output + ((size_t)oy * out_w + ox) * 24;
      for (int oc = 0; oc < 24; ++oc) {
        float acc = bias[oc];
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = base + (size_t)ky * in_w * 3;
          for (int kx = 0; kx < 3; ++kx) {
            const float* src = row + kx * 3;
            const float* wv = weights + ((ky * 3 + kx) * 3) * 24 + oc;
            acc += src[0] * wv[0 * 24];
            acc += src[1] * wv[1 * 24];
            acc += src[2] * wv[2 * 24];
          }
        }
        dst[oc] = activate(acc, activation);
      }
    }
  }
#endif
}

static void conv2d_3x3s2_rgb24_from_pad_packed(
    const float* input, const int32_t* pads, const float* weights, const float* bias, float* output,
    int y0, int y1, int in_h, int in_w, int out_w, int activation) {
  int top = pads[2];
  int left = pads[4];
#if defined(__aarch64__)
  for (int oy = y0; oy < y1; ++oy) {
    int py0 = oy * 2;
    int interior_y = py0 >= top && py0 + 3 <= top + in_h;
    for (int ox = 0; ox < out_w; ++ox) {
      int px0 = ox * 2;
      int interior = interior_y && px0 >= left && px0 + 3 <= left + in_w;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      float32x4_t acc4 = vld1q_f32(bias + 16);
      float32x4_t acc5 = vld1q_f32(bias + 20);
      if (interior) {
        const float* base = input + ((size_t)(py0 - top) * in_w + (px0 - left)) * 3;
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = base + (size_t)ky * in_w * 3;
          for (int kx = 0; kx < 3; ++kx) {
            const float* src = row + kx * 3;
            for (int ci = 0; ci < 3; ++ci) {
              const float* wv = weights + ((ky * 3 + kx) * 3 + ci) * 24;
              float v = src[ci];
              acc0 = vfmaq_n_f32(acc0, vld1q_f32(wv + 0), v);
              acc1 = vfmaq_n_f32(acc1, vld1q_f32(wv + 4), v);
              acc2 = vfmaq_n_f32(acc2, vld1q_f32(wv + 8), v);
              acc3 = vfmaq_n_f32(acc3, vld1q_f32(wv + 12), v);
              acc4 = vfmaq_n_f32(acc4, vld1q_f32(wv + 16), v);
              acc5 = vfmaq_n_f32(acc5, vld1q_f32(wv + 20), v);
            }
          }
        }
      } else {
        for (int ky = 0; ky < 3; ++ky) {
          int iy = py0 + ky - top;
          if ((unsigned)iy >= (unsigned)in_h) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int ix = px0 + kx - left;
            if ((unsigned)ix >= (unsigned)in_w) continue;
            const float* src = input + ((size_t)iy * in_w + ix) * 3;
            for (int ci = 0; ci < 3; ++ci) {
              const float* wv = weights + ((ky * 3 + kx) * 3 + ci) * 24;
              float v = src[ci];
              acc0 = vfmaq_n_f32(acc0, vld1q_f32(wv + 0), v);
              acc1 = vfmaq_n_f32(acc1, vld1q_f32(wv + 4), v);
              acc2 = vfmaq_n_f32(acc2, vld1q_f32(wv + 8), v);
              acc3 = vfmaq_n_f32(acc3, vld1q_f32(wv + 12), v);
              acc4 = vfmaq_n_f32(acc4, vld1q_f32(wv + 16), v);
              acc5 = vfmaq_n_f32(acc5, vld1q_f32(wv + 20), v);
            }
          }
        }
      }
      acc0 = activateq(acc0, activation);
      acc1 = activateq(acc1, activation);
      acc2 = activateq(acc2, activation);
      acc3 = activateq(acc3, activation);
      acc4 = activateq(acc4, activation);
      acc5 = activateq(acc5, activation);
      float* dst = output + ((size_t)oy * out_w + ox) * 24;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
      vst1q_f32(dst + 8, acc2);
      vst1q_f32(dst + 12, acc3);
      vst1q_f32(dst + 16, acc4);
      vst1q_f32(dst + 20, acc5);
    }
  }
#else
  for (int oy = y0; oy < y1; ++oy) {
    int py0 = oy * 2;
    for (int ox = 0; ox < out_w; ++ox) {
      int px0 = ox * 2;
      float* dst = output + ((size_t)oy * out_w + ox) * 24;
      for (int oc = 0; oc < 24; ++oc) {
        float acc = bias[oc];
        for (int ky = 0; ky < 3; ++ky) {
          int iy = py0 + ky - top;
          if ((unsigned)iy >= (unsigned)in_h) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int ix = px0 + kx - left;
            if ((unsigned)ix >= (unsigned)in_w) continue;
            const float* src = input + ((size_t)iy * in_w + ix) * 3;
            const float* wv = weights + ((ky * 3 + kx) * 3) * 24 + oc;
            acc += src[0] * wv[0 * 24];
            acc += src[1] * wv[1 * 24];
            acc += src[2] * wv[2 * 24];
          }
        }
        dst[oc] = activate(acc, activation);
      }
    }
  }
#endif
}

static void conv2d_1x1_packed_tile(
    const float* input, const PackedPointwiseX8* packed, float* output,
    size_t p0, int p_count, int oc_block, int oc_count, int activation) {
  int in_c = packed->in_c;
  int out_c = packed->out_c;
  int oc0 = oc_block * 8;
  const float* block = packed->data + (size_t)oc_block * packed->block_stride;
  const float* bias = block;
  const float* weights = block + 8;
#if defined(__aarch64__)
  if (oc_count == 8 && p_count == 4) {
    const float* s0 = input + (p0 + 0) * in_c;
    const float* s1 = input + (p0 + 1) * in_c;
    const float* s2 = input + (p0 + 2) * in_c;
    const float* s3 = input + (p0 + 3) * in_c;
    float* d0 = output + (p0 + 0) * out_c + oc0;
    float* d1 = output + (p0 + 1) * out_c + oc0;
    float* d2 = output + (p0 + 2) * out_c + oc0;
    float* d3 = output + (p0 + 3) * out_c + oc0;
    float32x4_t a00 = vld1q_f32(bias + 0), a01 = vld1q_f32(bias + 4);
    float32x4_t a10 = a00, a11 = a01;
    float32x4_t a20 = a00, a21 = a01;
    float32x4_t a30 = a00, a31 = a01;
    for (int ci = 0; ci < in_c; ++ci) {
      const float* wv = weights + (size_t)ci * 8;
      float32x4_t w0v = vld1q_f32(wv + 0);
      float32x4_t w1v = vld1q_f32(wv + 4);
      a00 = vfmaq_n_f32(a00, w0v, s0[ci]); a01 = vfmaq_n_f32(a01, w1v, s0[ci]);
      a10 = vfmaq_n_f32(a10, w0v, s1[ci]); a11 = vfmaq_n_f32(a11, w1v, s1[ci]);
      a20 = vfmaq_n_f32(a20, w0v, s2[ci]); a21 = vfmaq_n_f32(a21, w1v, s2[ci]);
      a30 = vfmaq_n_f32(a30, w0v, s3[ci]); a31 = vfmaq_n_f32(a31, w1v, s3[ci]);
    }
    a00 = activateq(a00, activation); a01 = activateq(a01, activation);
    a10 = activateq(a10, activation); a11 = activateq(a11, activation);
    a20 = activateq(a20, activation); a21 = activateq(a21, activation);
    a30 = activateq(a30, activation); a31 = activateq(a31, activation);
    vst1q_f32(d0 + 0, a00); vst1q_f32(d0 + 4, a01);
    vst1q_f32(d1 + 0, a10); vst1q_f32(d1 + 4, a11);
    vst1q_f32(d2 + 0, a20); vst1q_f32(d2 + 4, a21);
    vst1q_f32(d3 + 0, a30); vst1q_f32(d3 + 4, a31);
    return;
  }
  if (oc_count == 8 && p_count == 6) {
    const float* s0 = input + (p0 + 0) * in_c;
    const float* s1 = input + (p0 + 1) * in_c;
    const float* s2 = input + (p0 + 2) * in_c;
    const float* s3 = input + (p0 + 3) * in_c;
    const float* s4 = input + (p0 + 4) * in_c;
    const float* s5 = input + (p0 + 5) * in_c;
    float* d0 = output + (p0 + 0) * out_c + oc0;
    float* d1 = output + (p0 + 1) * out_c + oc0;
    float* d2 = output + (p0 + 2) * out_c + oc0;
    float* d3 = output + (p0 + 3) * out_c + oc0;
    float* d4 = output + (p0 + 4) * out_c + oc0;
    float* d5 = output + (p0 + 5) * out_c + oc0;
    float32x4_t a00 = vld1q_f32(bias + 0), a01 = vld1q_f32(bias + 4);
    float32x4_t a10 = a00, a11 = a01;
    float32x4_t a20 = a00, a21 = a01;
    float32x4_t a30 = a00, a31 = a01;
    float32x4_t a40 = a00, a41 = a01;
    float32x4_t a50 = a00, a51 = a01;
    for (int ci = 0; ci < in_c; ++ci) {
      const float* wv = weights + (size_t)ci * 8;
      float32x4_t w0v = vld1q_f32(wv + 0);
      float32x4_t w1v = vld1q_f32(wv + 4);
      a00 = vfmaq_n_f32(a00, w0v, s0[ci]); a01 = vfmaq_n_f32(a01, w1v, s0[ci]);
      a10 = vfmaq_n_f32(a10, w0v, s1[ci]); a11 = vfmaq_n_f32(a11, w1v, s1[ci]);
      a20 = vfmaq_n_f32(a20, w0v, s2[ci]); a21 = vfmaq_n_f32(a21, w1v, s2[ci]);
      a30 = vfmaq_n_f32(a30, w0v, s3[ci]); a31 = vfmaq_n_f32(a31, w1v, s3[ci]);
      a40 = vfmaq_n_f32(a40, w0v, s4[ci]); a41 = vfmaq_n_f32(a41, w1v, s4[ci]);
      a50 = vfmaq_n_f32(a50, w0v, s5[ci]); a51 = vfmaq_n_f32(a51, w1v, s5[ci]);
    }
    a00 = activateq(a00, activation); a01 = activateq(a01, activation);
    a10 = activateq(a10, activation); a11 = activateq(a11, activation);
    a20 = activateq(a20, activation); a21 = activateq(a21, activation);
    a30 = activateq(a30, activation); a31 = activateq(a31, activation);
    a40 = activateq(a40, activation); a41 = activateq(a41, activation);
    a50 = activateq(a50, activation); a51 = activateq(a51, activation);
    vst1q_f32(d0 + 0, a00); vst1q_f32(d0 + 4, a01);
    vst1q_f32(d1 + 0, a10); vst1q_f32(d1 + 4, a11);
    vst1q_f32(d2 + 0, a20); vst1q_f32(d2 + 4, a21);
    vst1q_f32(d3 + 0, a30); vst1q_f32(d3 + 4, a31);
    vst1q_f32(d4 + 0, a40); vst1q_f32(d4 + 4, a41);
    vst1q_f32(d5 + 0, a50); vst1q_f32(d5 + 4, a51);
    return;
  }
  if (oc_count == 8) {
    for (int pi = 0; pi < p_count; ++pi) {
      const float* src = input + (p0 + (size_t)pi) * in_c;
      float* dst = output + (p0 + (size_t)pi) * out_c + oc0;
      float32x4_t acc0 = vld1q_f32(bias);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      for (int ci = 0; ci < in_c; ++ci) {
        const float* wv = weights + (size_t)ci * 8;
        float v = src[ci];
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(wv), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(wv + 4), v);
      }
      acc0 = activateq(acc0, activation);
      acc1 = activateq(acc1, activation);
      vst1q_f32(dst, acc0);
      vst1q_f32(dst + 4, acc1);
    }
    return;
  }
#endif
  for (int pi = 0; pi < p_count; ++pi) {
    const float* src = input + (p0 + (size_t)pi) * in_c;
    float* dst = output + (p0 + (size_t)pi) * out_c + oc0;
    for (int o = 0; o < oc_count; ++o) {
      float acc = bias[o];
      for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * weights[(size_t)ci * 8 + o];
      dst[o] = activate(acc, activation);
    }
  }
}

static void conv2d_1x1_packed_add_tile(
    const float* input, const PackedPointwiseX8* packed, const float* residual, float* output,
    size_t p0, int p_count, int oc_block, int oc_count, int activation) {
  int in_c = packed->in_c;
  int out_c = packed->out_c;
  int oc0 = oc_block * 8;
  const float* block = packed->data + (size_t)oc_block * packed->block_stride;
  const float* bias = block;
  const float* weights = block + 8;
#if defined(__aarch64__)
  if (oc_count == 8 && p_count == 4) {
    const float* s0 = input + (p0 + 0) * in_c;
    const float* s1 = input + (p0 + 1) * in_c;
    const float* s2 = input + (p0 + 2) * in_c;
    const float* s3 = input + (p0 + 3) * in_c;
    const float* r0 = residual + (p0 + 0) * out_c + oc0;
    const float* r1 = residual + (p0 + 1) * out_c + oc0;
    const float* r2 = residual + (p0 + 2) * out_c + oc0;
    const float* r3 = residual + (p0 + 3) * out_c + oc0;
    float* d0 = output + (p0 + 0) * out_c + oc0;
    float* d1 = output + (p0 + 1) * out_c + oc0;
    float* d2 = output + (p0 + 2) * out_c + oc0;
    float* d3 = output + (p0 + 3) * out_c + oc0;
    float32x4_t a00 = vld1q_f32(bias + 0), a01 = vld1q_f32(bias + 4);
    float32x4_t a10 = a00, a11 = a01;
    float32x4_t a20 = a00, a21 = a01;
    float32x4_t a30 = a00, a31 = a01;
    for (int ci = 0; ci < in_c; ++ci) {
      const float* wv = weights + (size_t)ci * 8;
      float32x4_t w0v = vld1q_f32(wv + 0);
      float32x4_t w1v = vld1q_f32(wv + 4);
      a00 = vfmaq_n_f32(a00, w0v, s0[ci]); a01 = vfmaq_n_f32(a01, w1v, s0[ci]);
      a10 = vfmaq_n_f32(a10, w0v, s1[ci]); a11 = vfmaq_n_f32(a11, w1v, s1[ci]);
      a20 = vfmaq_n_f32(a20, w0v, s2[ci]); a21 = vfmaq_n_f32(a21, w1v, s2[ci]);
      a30 = vfmaq_n_f32(a30, w0v, s3[ci]); a31 = vfmaq_n_f32(a31, w1v, s3[ci]);
    }
    a00 = vaddq_f32(a00, vld1q_f32(r0 + 0)); a01 = vaddq_f32(a01, vld1q_f32(r0 + 4));
    a10 = vaddq_f32(a10, vld1q_f32(r1 + 0)); a11 = vaddq_f32(a11, vld1q_f32(r1 + 4));
    a20 = vaddq_f32(a20, vld1q_f32(r2 + 0)); a21 = vaddq_f32(a21, vld1q_f32(r2 + 4));
    a30 = vaddq_f32(a30, vld1q_f32(r3 + 0)); a31 = vaddq_f32(a31, vld1q_f32(r3 + 4));
    a00 = activateq(a00, activation); a01 = activateq(a01, activation);
    a10 = activateq(a10, activation); a11 = activateq(a11, activation);
    a20 = activateq(a20, activation); a21 = activateq(a21, activation);
    a30 = activateq(a30, activation); a31 = activateq(a31, activation);
    vst1q_f32(d0 + 0, a00); vst1q_f32(d0 + 4, a01);
    vst1q_f32(d1 + 0, a10); vst1q_f32(d1 + 4, a11);
    vst1q_f32(d2 + 0, a20); vst1q_f32(d2 + 4, a21);
    vst1q_f32(d3 + 0, a30); vst1q_f32(d3 + 4, a31);
    return;
  }
  if (oc_count == 8 && p_count == 6) {
    const float* s0 = input + (p0 + 0) * in_c;
    const float* s1 = input + (p0 + 1) * in_c;
    const float* s2 = input + (p0 + 2) * in_c;
    const float* s3 = input + (p0 + 3) * in_c;
    const float* s4 = input + (p0 + 4) * in_c;
    const float* s5 = input + (p0 + 5) * in_c;
    const float* r0 = residual + (p0 + 0) * out_c + oc0;
    const float* r1 = residual + (p0 + 1) * out_c + oc0;
    const float* r2 = residual + (p0 + 2) * out_c + oc0;
    const float* r3 = residual + (p0 + 3) * out_c + oc0;
    const float* r4 = residual + (p0 + 4) * out_c + oc0;
    const float* r5 = residual + (p0 + 5) * out_c + oc0;
    float* d0 = output + (p0 + 0) * out_c + oc0;
    float* d1 = output + (p0 + 1) * out_c + oc0;
    float* d2 = output + (p0 + 2) * out_c + oc0;
    float* d3 = output + (p0 + 3) * out_c + oc0;
    float* d4 = output + (p0 + 4) * out_c + oc0;
    float* d5 = output + (p0 + 5) * out_c + oc0;
    float32x4_t a00 = vld1q_f32(bias + 0), a01 = vld1q_f32(bias + 4);
    float32x4_t a10 = a00, a11 = a01;
    float32x4_t a20 = a00, a21 = a01;
    float32x4_t a30 = a00, a31 = a01;
    float32x4_t a40 = a00, a41 = a01;
    float32x4_t a50 = a00, a51 = a01;
    for (int ci = 0; ci < in_c; ++ci) {
      const float* wv = weights + (size_t)ci * 8;
      float32x4_t w0v = vld1q_f32(wv + 0);
      float32x4_t w1v = vld1q_f32(wv + 4);
      a00 = vfmaq_n_f32(a00, w0v, s0[ci]); a01 = vfmaq_n_f32(a01, w1v, s0[ci]);
      a10 = vfmaq_n_f32(a10, w0v, s1[ci]); a11 = vfmaq_n_f32(a11, w1v, s1[ci]);
      a20 = vfmaq_n_f32(a20, w0v, s2[ci]); a21 = vfmaq_n_f32(a21, w1v, s2[ci]);
      a30 = vfmaq_n_f32(a30, w0v, s3[ci]); a31 = vfmaq_n_f32(a31, w1v, s3[ci]);
      a40 = vfmaq_n_f32(a40, w0v, s4[ci]); a41 = vfmaq_n_f32(a41, w1v, s4[ci]);
      a50 = vfmaq_n_f32(a50, w0v, s5[ci]); a51 = vfmaq_n_f32(a51, w1v, s5[ci]);
    }
    a00 = vaddq_f32(a00, vld1q_f32(r0 + 0)); a01 = vaddq_f32(a01, vld1q_f32(r0 + 4));
    a10 = vaddq_f32(a10, vld1q_f32(r1 + 0)); a11 = vaddq_f32(a11, vld1q_f32(r1 + 4));
    a20 = vaddq_f32(a20, vld1q_f32(r2 + 0)); a21 = vaddq_f32(a21, vld1q_f32(r2 + 4));
    a30 = vaddq_f32(a30, vld1q_f32(r3 + 0)); a31 = vaddq_f32(a31, vld1q_f32(r3 + 4));
    a40 = vaddq_f32(a40, vld1q_f32(r4 + 0)); a41 = vaddq_f32(a41, vld1q_f32(r4 + 4));
    a50 = vaddq_f32(a50, vld1q_f32(r5 + 0)); a51 = vaddq_f32(a51, vld1q_f32(r5 + 4));
    a00 = activateq(a00, activation); a01 = activateq(a01, activation);
    a10 = activateq(a10, activation); a11 = activateq(a11, activation);
    a20 = activateq(a20, activation); a21 = activateq(a21, activation);
    a30 = activateq(a30, activation); a31 = activateq(a31, activation);
    a40 = activateq(a40, activation); a41 = activateq(a41, activation);
    a50 = activateq(a50, activation); a51 = activateq(a51, activation);
    vst1q_f32(d0 + 0, a00); vst1q_f32(d0 + 4, a01);
    vst1q_f32(d1 + 0, a10); vst1q_f32(d1 + 4, a11);
    vst1q_f32(d2 + 0, a20); vst1q_f32(d2 + 4, a21);
    vst1q_f32(d3 + 0, a30); vst1q_f32(d3 + 4, a31);
    vst1q_f32(d4 + 0, a40); vst1q_f32(d4 + 4, a41);
    vst1q_f32(d5 + 0, a50); vst1q_f32(d5 + 4, a51);
    return;
  }
  if (oc_count == 8) {
    for (int pi = 0; pi < p_count; ++pi) {
      const float* src = input + (p0 + (size_t)pi) * in_c;
      const float* res = residual + (p0 + (size_t)pi) * out_c + oc0;
      float* dst = output + (p0 + (size_t)pi) * out_c + oc0;
      float32x4_t acc0 = vld1q_f32(bias);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      for (int ci = 0; ci < in_c; ++ci) {
        const float* wv = weights + (size_t)ci * 8;
        float v = src[ci];
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(wv), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(wv + 4), v);
      }
      acc0 = activateq(vaddq_f32(acc0, vld1q_f32(res)), activation);
      acc1 = activateq(vaddq_f32(acc1, vld1q_f32(res + 4)), activation);
      vst1q_f32(dst, acc0);
      vst1q_f32(dst + 4, acc1);
    }
    return;
  }
#endif
  for (int pi = 0; pi < p_count; ++pi) {
    const float* src = input + (p0 + (size_t)pi) * in_c;
    const float* res = residual + (p0 + (size_t)pi) * out_c + oc0;
    float* dst = output + (p0 + (size_t)pi) * out_c + oc0;
    for (int o = 0; o < oc_count; ++o) {
      float acc = bias[o];
      for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * weights[(size_t)ci * 8 + o];
      dst[o] = activate(acc + res[o], activation);
    }
  }
}

static void op_conv2d_1x1_packed_tiles(PoseRuntime* rt, const PoseOpDef* op, int item0, int item1) {
  const float* input = (const float*)rt->tensor[op->inputs[0]];
  const PackedPointwiseX8* packed = (const PackedPointwiseX8*)rt->packed[op->inputs[1]];
  float* output = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  int out_c = out->dims[3];
  size_t pixels = (size_t)out->dims[1] * out->dims[2];
  int oc_blocks = (out_c + 7) / 8;
  for (int item = item0; item < item1; ++item) {
    size_t p0 = (size_t)(item / oc_blocks) * POSE_PW_TILE;
    int oc_block = item % oc_blocks;
    int oc0 = oc_block * 8;
    int p_count = (int)(pixels - p0 < POSE_PW_TILE ? pixels - p0 : POSE_PW_TILE);
    int oc_count = out_c - oc0 < 8 ? out_c - oc0 : 8;
    conv2d_1x1_packed_tile(input, packed, output, p0, p_count, oc_block, oc_count, op->activation);
  }
}

static void op_conv2d_1x1_add_packed_tiles(PoseRuntime* rt, const PoseOpDef* conv_op, int item0, int item1) {
  const PoseOpDef* add_op = rt->task_aux_op;
  const float* input = (const float*)rt->tensor[conv_op->inputs[0]];
  const PackedPointwiseX8* packed = (const PackedPointwiseX8*)rt->packed[conv_op->inputs[1]];
  int residual_idx = add_op->inputs[0] == conv_op->outputs[0] ? add_op->inputs[1] : add_op->inputs[0];
  const float* residual = (const float*)rt->tensor[residual_idx];
  float* output = (float*)rt->tensor[add_op->outputs[0]];
  const PoseTensorDef* out = &rt->def->tensors[add_op->outputs[0]];
  int out_c = out->dims[3];
  size_t pixels = (size_t)out->dims[1] * out->dims[2];
  int oc_blocks = (out_c + 7) / 8;
  for (int item = item0; item < item1; ++item) {
    size_t p0 = (size_t)(item / oc_blocks) * POSE_PW_TILE;
    int oc_block = item % oc_blocks;
    int oc0 = oc_block * 8;
    int p_count = (int)(pixels - p0 < POSE_PW_TILE ? pixels - p0 : POSE_PW_TILE);
    int oc_count = out_c - oc0 < 8 ? out_c - oc0 : 8;
    conv2d_1x1_packed_add_tile(
        input, packed, residual, output,
        p0, p_count, oc_block, oc_count, add_op->activation);
  }
}

static void op_conv2d_1x1_add_packed(PoseRuntime* rt, const PoseOpDef* conv_op, const PoseOpDef* add_op) {
  const PoseTensorDef* out = &rt->def->tensors[add_op->outputs[0]];
  size_t pixels = (size_t)out->dims[1] * out->dims[2];
  int pixel_tiles = (int)((pixels + POSE_PW_TILE - 1) / POSE_PW_TILE);
  int oc_blocks = (out->dims[3] + 7) / 8;
  parallel_rows_aux(rt, conv_op, add_op, pixel_tiles * oc_blocks, op_conv2d_1x1_add_packed_tiles);
}

static void op_conv2d_rows(PoseRuntime* rt, const PoseOpDef* op, int y0, int y1) {
  const float* input = (const float*)rt->tensor[op->inputs[0]];
  const float* weights_ohwi = (const float*)rt->tensor[op->inputs[1]];
  const int packed_layout = rt->packed_type[op->inputs[1]] == POSE_PACK_ICOC;
  const float* weights = packed_layout ? (const float*)rt->packed[op->inputs[1]] : weights_ohwi;
  const float* bias = (const float*)rt->tensor[op->inputs[2]];
  float* output = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* in = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* wt = &rt->def->tensors[op->inputs[1]];
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  int in_h = in->dims[1], in_w = in->dims[2], in_c = in->dims[3];
  int out_h = out->dims[1], out_w = out->dims[2], out_c = out->dims[3];
  int kh = wt->dims[1], kw = wt->dims[2];
  if (packed_layout && kh == 3 && kw == 3 && op->stride_h == 2 && op->stride_w == 2 &&
      op->dilation_h == 1 && op->dilation_w == 1 && op->padding == 1 &&
      in_c == 3 && out_c == 24) {
    conv2d_3x3s2_rgb24_packed(input, weights, bias, output, y0, y1, in_w, out_w, op->activation);
    return;
  }
  if (kh == 1 && kw == 1 && op->stride_h == 1 && op->stride_w == 1) {
    conv2d_1x1(input, weights, bias, output, y0, y1, out_w, in_c, out_c, op->activation, packed_layout);
    return;
  }
  int pad_top = op->padding == 0 ? same_pad_before(in_h, out_h, op->stride_h, kh, op->dilation_h) : 0;
  int pad_left = op->padding == 0 ? same_pad_before(in_w, out_w, op->stride_w, kw, op->dilation_w) : 0;
  for (int oy = y0; oy < y1; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      for (int oc = 0; oc < out_c; ++oc) {
        float acc = bias[oc];
        for (int ky = 0; ky < kh; ++ky) {
          int iy = oy * op->stride_h + ky * op->dilation_h - pad_top;
          if ((unsigned)iy >= (unsigned)in_h) continue;
          for (int kx = 0; kx < kw; ++kx) {
            int ix = ox * op->stride_w + kx * op->dilation_w - pad_left;
            if ((unsigned)ix >= (unsigned)in_w) continue;
            const float* src = input + ((size_t)iy * in_w + ix) * in_c;
            if (packed_layout) {
              const float* wv = weights + (((size_t)ky * kw + kx) * in_c) * out_c + oc;
              for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * wv[(size_t)ci * out_c];
            } else {
              const float* wv = weights + (((size_t)oc * kh + ky) * kw + kx) * in_c;
              for (int ci = 0; ci < in_c; ++ci) acc += src[ci] * wv[ci];
            }
          }
        }
        output[((size_t)oy * out_w + ox) * out_c + oc] = activate(acc, op->activation);
      }
    }
  }
}

static void op_conv2d(PoseRuntime* rt, const PoseOpDef* op) {
  if (rt->packed_type[op->inputs[1]] == POSE_PACK_POINTWISE_X8) {
    const PoseTensorDef* wt = &rt->def->tensors[op->inputs[1]];
    const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
    if (wt->dims[1] == 1 && wt->dims[2] == 1 && op->stride_h == 1 && op->stride_w == 1) {
      size_t pixels = (size_t)out->dims[1] * out->dims[2];
      int pixel_tiles = (int)((pixels + POSE_PW_TILE - 1) / POSE_PW_TILE);
      int oc_blocks = (out->dims[3] + 7) / 8;
      parallel_rows(rt, op, pixel_tiles * oc_blocks, op_conv2d_1x1_packed_tiles);
      return;
    }
  }
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  parallel_rows(rt, op, out->dims[1], op_conv2d_rows);
}

static void op_conv2d_3x3s2_rgb24_from_pad_rows(PoseRuntime* rt, const PoseOpDef* pad_op, int y0, int y1) {
  const PoseOpDef* conv_op = rt->task_aux_op;
  const float* input = (const float*)rt->tensor[pad_op->inputs[0]];
  const int32_t* pads = (const int32_t*)rt->tensor[pad_op->inputs[1]];
  const float* weights = (const float*)rt->packed[conv_op->inputs[1]];
  const float* bias = (const float*)rt->tensor[conv_op->inputs[2]];
  float* output = (float*)rt->tensor[conv_op->outputs[0]];
  const PoseTensorDef* in = &rt->def->tensors[pad_op->inputs[0]];
  const PoseTensorDef* out = &rt->def->tensors[conv_op->outputs[0]];
  conv2d_3x3s2_rgb24_from_pad_packed(
      input, pads, weights, bias, output,
      y0, y1, in->dims[1], in->dims[2], out->dims[2], conv_op->activation);
}

static void op_conv2d_3x3s2_rgb24_from_pad(PoseRuntime* rt, const PoseOpDef* pad_op, const PoseOpDef* conv_op) {
  const PoseTensorDef* out = &rt->def->tensors[conv_op->outputs[0]];
  parallel_rows_aux(rt, pad_op, conv_op, out->dims[1], op_conv2d_3x3s2_rgb24_from_pad_rows);
}

static void depthwise_3x3_checked_pixel(
    const float* input, const float* weights, const float* bias, float* output,
    int y, int x, int h, int w, int c, int activation) {
  float* dst = output + ((size_t)y * w + x) * c;
  int ch = 0;
#if defined(__aarch64__)
  for (; ch + 8 <= c; ch += 8) {
    float32x4_t acc0 = vld1q_f32(bias + ch);
    float32x4_t acc1 = vld1q_f32(bias + ch + 4);
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y + ky - 1;
      if ((unsigned)iy >= (unsigned)h) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x + kx - 1;
        if ((unsigned)ix >= (unsigned)w) continue;
        fmaq8_at(
            &acc0, &acc1,
            input + ((size_t)iy * w + ix) * c + ch,
            weights + (ky * 3 + kx) * c + ch);
      }
    }
    acc0 = activateq(acc0, activation);
    acc1 = activateq(acc1, activation);
    vst1q_f32(dst + ch, acc0);
    vst1q_f32(dst + ch + 4, acc1);
  }
  for (; ch + 4 <= c; ch += 4) {
    float32x4_t acc = vld1q_f32(bias + ch);
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y + ky - 1;
      if ((unsigned)iy >= (unsigned)h) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x + kx - 1;
        if ((unsigned)ix >= (unsigned)w) continue;
        fmaq4_at(
            &acc,
            input + ((size_t)iy * w + ix) * c + ch,
            weights + (ky * 3 + kx) * c + ch);
      }
    }
    acc = activateq(acc, activation);
    vst1q_f32(dst + ch, acc);
  }
#endif
  for (; ch < c; ++ch) {
    float acc = bias[ch];
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y + ky - 1;
      if ((unsigned)iy >= (unsigned)h) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x + kx - 1;
        if ((unsigned)ix >= (unsigned)w) continue;
        acc += input[((size_t)iy * w + ix) * c + ch] * weights[(ky * 3 + kx) * c + ch];
      }
    }
    dst[ch] = activate(acc, activation);
  }
}

static inline void depthwise_3x3s1_same_pair(
    const float* row0, const float* row1, const float* row2,
    const float* weights, const float* bias, float* dst0, float* dst1,
    int c, int activation) {
  int ch = 0;
#if defined(__aarch64__)
  for (; ch + 8 <= c; ch += 8) {
    float32x4_t a0 = vld1q_f32(bias + ch);
    float32x4_t a1 = vld1q_f32(bias + ch + 4);
    float32x4_t b0 = a0;
    float32x4_t b1 = a1;
    fmaq8_depthwise_pair_row(&a0, &a1, &b0, &b1, row0 + ch, weights + 0 * c + ch, c);
    fmaq8_depthwise_pair_row(&a0, &a1, &b0, &b1, row1 + ch, weights + 3 * c + ch, c);
    fmaq8_depthwise_pair_row(&a0, &a1, &b0, &b1, row2 + ch, weights + 6 * c + ch, c);
    a0 = activateq(a0, activation);
    a1 = activateq(a1, activation);
    b0 = activateq(b0, activation);
    b1 = activateq(b1, activation);
    vst1q_f32(dst0 + ch, a0);
    vst1q_f32(dst0 + ch + 4, a1);
    vst1q_f32(dst1 + ch, b0);
    vst1q_f32(dst1 + ch + 4, b1);
  }
  for (; ch + 4 <= c; ch += 4) {
    float32x4_t a = vld1q_f32(bias + ch);
    float32x4_t b = a;
    fmaq4_depthwise_pair_row(&a, &b, row0 + ch, weights + 0 * c + ch, c);
    fmaq4_depthwise_pair_row(&a, &b, row1 + ch, weights + 3 * c + ch, c);
    fmaq4_depthwise_pair_row(&a, &b, row2 + ch, weights + 6 * c + ch, c);
    a = activateq(a, activation);
    b = activateq(b, activation);
    vst1q_f32(dst0 + ch, a);
    vst1q_f32(dst1 + ch, b);
  }
#endif
  for (; ch < c; ++ch) {
    float a = bias[ch];
    float b = bias[ch];
    a += row0[ch] * weights[0 * c + ch];
    b += row0[c + ch] * weights[0 * c + ch];
    a += row0[c + ch] * weights[1 * c + ch];
    b += row0[2 * c + ch] * weights[1 * c + ch];
    a += row0[2 * c + ch] * weights[2 * c + ch];
    b += row0[3 * c + ch] * weights[2 * c + ch];
    a += row1[ch] * weights[3 * c + ch];
    b += row1[c + ch] * weights[3 * c + ch];
    a += row1[c + ch] * weights[4 * c + ch];
    b += row1[2 * c + ch] * weights[4 * c + ch];
    a += row1[2 * c + ch] * weights[5 * c + ch];
    b += row1[3 * c + ch] * weights[5 * c + ch];
    a += row2[ch] * weights[6 * c + ch];
    b += row2[c + ch] * weights[6 * c + ch];
    a += row2[c + ch] * weights[7 * c + ch];
    b += row2[2 * c + ch] * weights[7 * c + ch];
    a += row2[2 * c + ch] * weights[8 * c + ch];
    b += row2[3 * c + ch] * weights[8 * c + ch];
    dst0[ch] = activate(a, activation);
    dst1[ch] = activate(b, activation);
  }
}

static void depthwise_3x3s1_same_rows(
    const float* input, const float* weights, const float* bias, float* output,
    int y0, int y1, int h, int w, int c, int activation) {
  if (h < 3 || w < 3) {
    for (int y = y0; y < y1; ++y) {
      for (int x = 0; x < w; ++x) {
        depthwise_3x3_checked_pixel(input, weights, bias, output, y, x, h, w, c, activation);
      }
    }
    return;
  }
  if (y0 == 0) {
    for (int x = 0; x < w; ++x) depthwise_3x3_checked_pixel(input, weights, bias, output, 0, x, h, w, c, activation);
    y0 = 1;
  }
  if (y1 == h) {
    for (int x = 0; x < w; ++x) depthwise_3x3_checked_pixel(input, weights, bias, output, h - 1, x, h, w, c, activation);
    y1 = h - 1;
  }
  for (int y = y0; y < y1; ++y) {
    depthwise_3x3_checked_pixel(input, weights, bias, output, y, 0, h, w, c, activation);
    depthwise_3x3_checked_pixel(input, weights, bias, output, y, w - 1, h, w, c, activation);
    int x = 1;
    for (; x + 1 < w - 1; x += 2) {
      float* dst0 = output + ((size_t)y * w + x) * c;
      float* dst1 = dst0 + c;
      const float* row0 = input + ((size_t)(y - 1) * w + x - 1) * c;
      const float* row1 = input + ((size_t)y * w + x - 1) * c;
      const float* row2 = input + ((size_t)(y + 1) * w + x - 1) * c;
      depthwise_3x3s1_same_pair(row0, row1, row2, weights, bias, dst0, dst1, c, activation);
    }
    for (; x < w - 1; ++x) {
      float* dst = output + ((size_t)y * w + x) * c;
      const float* row0 = input + ((size_t)(y - 1) * w + x - 1) * c;
      const float* row1 = input + ((size_t)y * w + x - 1) * c;
      const float* row2 = input + ((size_t)(y + 1) * w + x - 1) * c;
      int ch = 0;
#if defined(__aarch64__)
      for (; ch + 8 <= c; ch += 8) {
        float32x4_t acc0 = vld1q_f32(bias + ch);
        float32x4_t acc1 = vld1q_f32(bias + ch + 4);
        fmaq8_at(&acc0, &acc1, row0 + ch, weights + 0 * c + ch);
        fmaq8_at(&acc0, &acc1, row0 + c + ch, weights + 1 * c + ch);
        fmaq8_at(&acc0, &acc1, row0 + 2 * c + ch, weights + 2 * c + ch);
        fmaq8_at(&acc0, &acc1, row1 + ch, weights + 3 * c + ch);
        fmaq8_at(&acc0, &acc1, row1 + c + ch, weights + 4 * c + ch);
        fmaq8_at(&acc0, &acc1, row1 + 2 * c + ch, weights + 5 * c + ch);
        fmaq8_at(&acc0, &acc1, row2 + ch, weights + 6 * c + ch);
        fmaq8_at(&acc0, &acc1, row2 + c + ch, weights + 7 * c + ch);
        fmaq8_at(&acc0, &acc1, row2 + 2 * c + ch, weights + 8 * c + ch);
        acc0 = activateq(acc0, activation);
        acc1 = activateq(acc1, activation);
        vst1q_f32(dst + ch, acc0);
        vst1q_f32(dst + ch + 4, acc1);
      }
      for (; ch + 4 <= c; ch += 4) {
        float32x4_t acc = vld1q_f32(bias + ch);
        fmaq4_at(&acc, row0 + ch, weights + 0 * c + ch);
        fmaq4_at(&acc, row0 + c + ch, weights + 1 * c + ch);
        fmaq4_at(&acc, row0 + 2 * c + ch, weights + 2 * c + ch);
        fmaq4_at(&acc, row1 + ch, weights + 3 * c + ch);
        fmaq4_at(&acc, row1 + c + ch, weights + 4 * c + ch);
        fmaq4_at(&acc, row1 + 2 * c + ch, weights + 5 * c + ch);
        fmaq4_at(&acc, row2 + ch, weights + 6 * c + ch);
        fmaq4_at(&acc, row2 + c + ch, weights + 7 * c + ch);
        fmaq4_at(&acc, row2 + 2 * c + ch, weights + 8 * c + ch);
        acc = activateq(acc, activation);
        vst1q_f32(dst + ch, acc);
      }
#endif
      for (; ch < c; ++ch) {
        float acc = bias[ch];
        acc += row0[ch] * weights[0 * c + ch];
        acc += row0[c + ch] * weights[1 * c + ch];
        acc += row0[2 * c + ch] * weights[2 * c + ch];
        acc += row1[ch] * weights[3 * c + ch];
        acc += row1[c + ch] * weights[4 * c + ch];
        acc += row1[2 * c + ch] * weights[5 * c + ch];
        acc += row2[ch] * weights[6 * c + ch];
        acc += row2[c + ch] * weights[7 * c + ch];
        acc += row2[2 * c + ch] * weights[8 * c + ch];
        dst[ch] = activate(acc, activation);
      }
    }
  }
}

static void depthwise_valid_rows(
    const float* input, const float* weights, const float* bias, float* output,
    int y0, int y1, int in_w, int out_w, int c, int kh, int kw,
    int stride_h, int stride_w, int activation) {
  for (int oy = y0; oy < y1; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      const float* base = input + ((size_t)oy * stride_h * in_w + ox * stride_w) * c;
      float* dst = output + ((size_t)oy * out_w + ox) * c;
      int ch = 0;
#if defined(__aarch64__)
      for (; ch + 8 <= c; ch += 8) {
        float32x4_t acc0 = vld1q_f32(bias + ch);
        float32x4_t acc1 = vld1q_f32(bias + ch + 4);
        for (int ky = 0; ky < kh; ++ky) {
          const float* row = base + (size_t)ky * in_w * c;
          const float* wrow = weights + (size_t)ky * kw * c;
          for (int kx = 0; kx < kw; ++kx) {
            fmaq8_at(&acc0, &acc1, row + (size_t)kx * c + ch, wrow + (size_t)kx * c + ch);
          }
        }
        acc0 = activateq(acc0, activation);
        acc1 = activateq(acc1, activation);
        vst1q_f32(dst + ch, acc0);
        vst1q_f32(dst + ch + 4, acc1);
      }
      for (; ch + 4 <= c; ch += 4) {
        float32x4_t acc = vld1q_f32(bias + ch);
        for (int ky = 0; ky < kh; ++ky) {
          const float* row = base + (size_t)ky * in_w * c;
          const float* wrow = weights + (size_t)ky * kw * c;
          for (int kx = 0; kx < kw; ++kx) {
            fmaq4_at(&acc, row + (size_t)kx * c + ch, wrow + (size_t)kx * c + ch);
          }
        }
        acc = activateq(acc, activation);
        vst1q_f32(dst + ch, acc);
      }
#endif
      for (; ch < c; ++ch) {
        float acc = bias[ch];
        for (int ky = 0; ky < kh; ++ky) {
          const float* row = base + (size_t)ky * in_w * c;
          const float* wrow = weights + (size_t)ky * kw * c;
          for (int kx = 0; kx < kw; ++kx) {
            acc += row[(size_t)kx * c + ch] * wrow[(size_t)kx * c + ch];
          }
        }
        dst[ch] = activate(acc, activation);
      }
    }
  }
}

static void depthwise_from_padded_input_rows(PoseRuntime* rt, const PoseOpDef* pad_op, int y0, int y1) {
  const PoseOpDef* dw_op = rt->task_aux_op;
  const float* input = (const float*)rt->tensor[pad_op->inputs[0]];
  const int32_t* pads = (const int32_t*)rt->tensor[pad_op->inputs[1]];
  const float* weights = (const float*)rt->tensor[dw_op->inputs[1]];
  const float* bias = (const float*)rt->tensor[dw_op->inputs[2]];
  float* output = (float*)rt->tensor[dw_op->outputs[0]];
  const PoseTensorDef* in = &rt->def->tensors[pad_op->inputs[0]];
  const PoseTensorDef* wt = &rt->def->tensors[dw_op->inputs[1]];
  const PoseTensorDef* out = &rt->def->tensors[dw_op->outputs[0]];
  int in_h = in->dims[1], in_w = in->dims[2], c = in->dims[3];
  int out_w = out->dims[2];
  int kh = wt->dims[1], kw = wt->dims[2];
  int top = pads[2], left = pads[4];
  for (int oy = y0; oy < y1; ++oy) {
    int py0 = oy * dw_op->stride_h;
    for (int ox = 0; ox < out_w; ++ox) {
      int px0 = ox * dw_op->stride_w;
      float* dst = output + ((size_t)oy * out_w + ox) * c;
      int interior =
          py0 >= top && px0 >= left &&
          py0 + kh <= top + in_h &&
          px0 + kw <= left + in_w;
      int ch = 0;
#if defined(__aarch64__)
      for (; ch + 8 <= c; ch += 8) {
        float32x4_t acc0 = vld1q_f32(bias + ch);
        float32x4_t acc1 = vld1q_f32(bias + ch + 4);
        if (interior) {
          const float* base = input + ((size_t)(py0 - top) * in_w + (px0 - left)) * c;
          for (int ky = 0; ky < kh; ++ky) {
            const float* row = base + (size_t)ky * in_w * c;
            const float* wrow = weights + (size_t)ky * kw * c;
            for (int kx = 0; kx < kw; ++kx) {
              fmaq8_at(&acc0, &acc1, row + (size_t)kx * c + ch, wrow + (size_t)kx * c + ch);
            }
          }
        } else {
          for (int ky = 0; ky < kh; ++ky) {
            int iy = py0 + ky - top;
            if ((unsigned)iy >= (unsigned)in_h) continue;
            for (int kx = 0; kx < kw; ++kx) {
              int ix = px0 + kx - left;
              if ((unsigned)ix >= (unsigned)in_w) continue;
              fmaq8_at(
                  &acc0, &acc1,
                  input + ((size_t)iy * in_w + ix) * c + ch,
                  weights + ((size_t)ky * kw + kx) * c + ch);
            }
          }
        }
        acc0 = activateq(acc0, dw_op->activation);
        acc1 = activateq(acc1, dw_op->activation);
        vst1q_f32(dst + ch, acc0);
        vst1q_f32(dst + ch + 4, acc1);
      }
      for (; ch + 4 <= c; ch += 4) {
        float32x4_t acc = vld1q_f32(bias + ch);
        if (interior) {
          const float* base = input + ((size_t)(py0 - top) * in_w + (px0 - left)) * c;
          for (int ky = 0; ky < kh; ++ky) {
            const float* row = base + (size_t)ky * in_w * c;
            const float* wrow = weights + (size_t)ky * kw * c;
            for (int kx = 0; kx < kw; ++kx) {
              fmaq4_at(&acc, row + (size_t)kx * c + ch, wrow + (size_t)kx * c + ch);
            }
          }
        } else {
          for (int ky = 0; ky < kh; ++ky) {
            int iy = py0 + ky - top;
            if ((unsigned)iy >= (unsigned)in_h) continue;
            for (int kx = 0; kx < kw; ++kx) {
              int ix = px0 + kx - left;
              if ((unsigned)ix >= (unsigned)in_w) continue;
              fmaq4_at(
                  &acc,
                  input + ((size_t)iy * in_w + ix) * c + ch,
                  weights + ((size_t)ky * kw + kx) * c + ch);
            }
          }
        }
        acc = activateq(acc, dw_op->activation);
        vst1q_f32(dst + ch, acc);
      }
#endif
      for (; ch < c; ++ch) {
        float acc = bias[ch];
        if (interior) {
          const float* base = input + ((size_t)(py0 - top) * in_w + (px0 - left)) * c;
          for (int ky = 0; ky < kh; ++ky) {
            const float* row = base + (size_t)ky * in_w * c;
            const float* wrow = weights + (size_t)ky * kw * c;
            for (int kx = 0; kx < kw; ++kx) {
              acc += row[(size_t)kx * c + ch] * wrow[(size_t)kx * c + ch];
            }
          }
        } else {
          for (int ky = 0; ky < kh; ++ky) {
            int iy = py0 + ky - top;
            if ((unsigned)iy >= (unsigned)in_h) continue;
            for (int kx = 0; kx < kw; ++kx) {
              int ix = px0 + kx - left;
              if ((unsigned)ix >= (unsigned)in_w) continue;
              acc += input[((size_t)iy * in_w + ix) * c + ch] * weights[((size_t)ky * kw + kx) * c + ch];
            }
          }
        }
        dst[ch] = activate(acc, dw_op->activation);
      }
    }
  }
}

static void op_depthwise_from_pad(PoseRuntime* rt, const PoseOpDef* pad_op, const PoseOpDef* dw_op) {
  const PoseTensorDef* in = &rt->def->tensors[pad_op->inputs[0]];
  const PoseTensorDef* out = &rt->def->tensors[dw_op->outputs[0]];
  if (in->rank != 4 || out->rank != 4) {
    fprintf(stderr, "fused PAD->DEPTHWISE only supports rank4 tensors\n");
    exit(2);
  }
  parallel_rows_aux(rt, pad_op, dw_op, out->dims[1], depthwise_from_padded_input_rows);
}

static void op_depthwise_rows(PoseRuntime* rt, const PoseOpDef* op, int y0, int y1) {
  const float* input = (const float*)rt->tensor[op->inputs[0]];
  const float* weights = (const float*)rt->tensor[op->inputs[1]];
  const float* bias = (const float*)rt->tensor[op->inputs[2]];
  float* output = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* in = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* wt = &rt->def->tensors[op->inputs[1]];
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  int in_h = in->dims[1], in_w = in->dims[2], c = in->dims[3];
  int out_h = out->dims[1], out_w = out->dims[2];
  int kh = wt->dims[1], kw = wt->dims[2];
  if (op->depth_multiplier != 1 || out->dims[3] != c) {
    fprintf(stderr, "only depth_multiplier=1 is supported\n");
    exit(2);
  }
  if (kh == 3 && kw == 3 && op->stride_h == 1 && op->stride_w == 1 &&
      op->dilation_h == 1 && op->dilation_w == 1 && op->padding == 0 &&
      in_h == out_h && in_w == out_w) {
    depthwise_3x3s1_same_rows(input, weights, bias, output, y0, y1, out_h, out_w, c, op->activation);
    return;
  }
  if (op->padding != 0 && op->dilation_h == 1 && op->dilation_w == 1) {
    depthwise_valid_rows(
        input, weights, bias, output,
        y0, y1, in_w, out_w, c, kh, kw, op->stride_h, op->stride_w, op->activation);
    return;
  }
  int pad_top = op->padding == 0 ? same_pad_before(in_h, out_h, op->stride_h, kh, op->dilation_h) : 0;
  int pad_left = op->padding == 0 ? same_pad_before(in_w, out_w, op->stride_w, kw, op->dilation_w) : 0;
  for (int oy = y0; oy < y1; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      float* dst = output + ((size_t)oy * out_w + ox) * c;
      int ch = 0;
#if defined(__aarch64__)
      for (; ch + 8 <= c; ch += 8) {
        float32x4_t acc0 = vld1q_f32(bias + ch);
        float32x4_t acc1 = vld1q_f32(bias + ch + 4);
        for (int ky = 0; ky < kh; ++ky) {
          int iy = oy * op->stride_h + ky * op->dilation_h - pad_top;
          if ((unsigned)iy >= (unsigned)in_h) continue;
          for (int kx = 0; kx < kw; ++kx) {
            int ix = ox * op->stride_w + kx * op->dilation_w - pad_left;
            if ((unsigned)ix >= (unsigned)in_w) continue;
            fmaq8_at(
                &acc0, &acc1,
                input + ((size_t)iy * in_w + ix) * c + ch,
                weights + ((size_t)ky * kw + kx) * c + ch);
          }
        }
        acc0 = activateq(acc0, op->activation);
        acc1 = activateq(acc1, op->activation);
        vst1q_f32(dst + ch, acc0);
        vst1q_f32(dst + ch + 4, acc1);
      }
      for (; ch + 4 <= c; ch += 4) {
        float32x4_t acc = vld1q_f32(bias + ch);
        for (int ky = 0; ky < kh; ++ky) {
          int iy = oy * op->stride_h + ky * op->dilation_h - pad_top;
          if ((unsigned)iy >= (unsigned)in_h) continue;
          for (int kx = 0; kx < kw; ++kx) {
            int ix = ox * op->stride_w + kx * op->dilation_w - pad_left;
            if ((unsigned)ix >= (unsigned)in_w) continue;
            const float* src = input + ((size_t)iy * in_w + ix) * c + ch;
            const float* wv = weights + ((size_t)ky * kw + kx) * c + ch;
            fmaq4_at(&acc, src, wv);
          }
        }
        acc = activateq(acc, op->activation);
        vst1q_f32(dst + ch, acc);
      }
#endif
      for (; ch < c; ++ch) {
        float acc = bias[ch];
        for (int ky = 0; ky < kh; ++ky) {
          int iy = oy * op->stride_h + ky * op->dilation_h - pad_top;
          if ((unsigned)iy >= (unsigned)in_h) continue;
          for (int kx = 0; kx < kw; ++kx) {
            int ix = ox * op->stride_w + kx * op->dilation_w - pad_left;
            if ((unsigned)ix >= (unsigned)in_w) continue;
            acc += input[((size_t)iy * in_w + ix) * c + ch] *
                   weights[((size_t)ky * kw + kx) * c + ch];
          }
        }
        dst[ch] = activate(acc, op->activation);
      }
    }
  }
}

static void op_depthwise(PoseRuntime* rt, const PoseOpDef* op) {
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  parallel_rows(rt, op, out->dims[1], op_depthwise_rows);
}

static void op_add(PoseRuntime* rt, const PoseOpDef* op) {
  const float* a = (const float*)rt->tensor[op->inputs[0]];
  const float* b = (const float*)rt->tensor[op->inputs[1]];
  float* out = (float*)rt->tensor[op->outputs[0]];
  size_t n = tensor_elements(&rt->def->tensors[op->outputs[0]]);
  size_t i = 0;
#if defined(__aarch64__)
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    v = activateq(v, op->activation);
    vst1q_f32(out + i, v);
  }
#endif
  for (; i < n; ++i) out[i] = activate(a[i] + b[i], op->activation);
}

static void op_logistic(PoseRuntime* rt, const PoseOpDef* op) {
  const float* in = (const float*)rt->tensor[op->inputs[0]];
  float* out = (float*)rt->tensor[op->outputs[0]];
  size_t n = tensor_elements(&rt->def->tensors[op->outputs[0]]);
  for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

static void op_reshape(PoseRuntime* rt, const PoseOpDef* op) {
  const PoseTensorDef* out_def = &rt->def->tensors[op->outputs[0]];
  memcpy(rt->tensor[op->outputs[0]], rt->tensor[op->inputs[0]], tensor_bytes(out_def));
}

static void op_concat(PoseRuntime* rt, const PoseOpDef* op) {
  float* out = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* out_def = &rt->def->tensors[op->outputs[0]];
  int axis = op->axis < 0 ? op->axis + out_def->rank : op->axis;
  if (out_def->rank != 3 || axis != 1) {
    fprintf(stderr, "CONCAT currently supports rank3 axis1 only\n");
    exit(2);
  }
  size_t offset = 0;
  int trailing = out_def->dims[2];
  for (int i = 0; i < op->input_count; ++i) {
    const PoseTensorDef* in_def = &rt->def->tensors[op->inputs[i]];
    size_t count = (size_t)in_def->dims[1] * trailing;
    memcpy(out + offset, rt->tensor[op->inputs[i]], count * sizeof(float));
    offset += count;
  }
}

static void op_depth_to_space(PoseRuntime* rt, const PoseOpDef* op) {
  const float* in = (const float*)rt->tensor[op->inputs[0]];
  float* out = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* in_def = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* out_def = &rt->def->tensors[op->outputs[0]];
  int block = op->block_size;
  int in_h = in_def->dims[1], in_w = in_def->dims[2], in_c = in_def->dims[3];
  int out_w = out_def->dims[2], out_c = out_def->dims[3];
  for (int y = 0; y < in_h; ++y) {
    for (int x = 0; x < in_w; ++x) {
      for (int by = 0; by < block; ++by) {
        for (int bx = 0; bx < block; ++bx) {
          int oy = y * block + by;
          int ox = x * block + bx;
          int c_base = (by * block + bx) * out_c;
          memcpy(
              out + ((size_t)oy * out_w + ox) * out_c,
              in + ((size_t)y * in_w + x) * in_c + c_base,
              (size_t)out_c * sizeof(float));
        }
      }
    }
  }
}

static void op_resize_bilinear_rows(PoseRuntime* rt, const PoseOpDef* op, int y0_out, int y1_out) {
  const float* in = (const float*)rt->tensor[op->inputs[0]];
  float* out = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* in_def = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* out_def = &rt->def->tensors[op->outputs[0]];
  int in_h = in_def->dims[1], in_w = in_def->dims[2], c = in_def->dims[3];
  int out_h = out_def->dims[1], out_w = out_def->dims[2];
  for (int oy = y0_out; oy < y1_out; ++oy) {
    float fy;
    if (op->align_corners && out_h > 1) {
      fy = (float)oy * (float)(in_h - 1) / (float)(out_h - 1);
    } else if (op->half_pixel_centers) {
      fy = ((float)oy + 0.5f) * (float)in_h / (float)out_h - 0.5f;
    } else {
      fy = (float)oy * (float)in_h / (float)out_h;
    }
    if (fy < 0.0f) fy = 0.0f;
    int y0 = (int)floorf(fy);
    if (y0 >= in_h - 1) y0 = in_h - 1;
    int y1 = y0 + 1 < in_h ? y0 + 1 : y0;
    float wy = fy - (float)y0;
    for (int ox = 0; ox < out_w; ++ox) {
      float fx;
      if (op->align_corners && out_w > 1) {
        fx = (float)ox * (float)(in_w - 1) / (float)(out_w - 1);
      } else if (op->half_pixel_centers) {
        fx = ((float)ox + 0.5f) * (float)in_w / (float)out_w - 0.5f;
      } else {
        fx = (float)ox * (float)in_w / (float)out_w;
      }
      if (fx < 0.0f) fx = 0.0f;
      int x0 = (int)floorf(fx);
      if (x0 >= in_w - 1) x0 = in_w - 1;
      int x1 = x0 + 1 < in_w ? x0 + 1 : x0;
      float wx = fx - (float)x0;
      const float* p00 = in + ((size_t)y0 * in_w + x0) * c;
      const float* p01 = in + ((size_t)y0 * in_w + x1) * c;
      const float* p10 = in + ((size_t)y1 * in_w + x0) * c;
      const float* p11 = in + ((size_t)y1 * in_w + x1) * c;
      float* dst = out + ((size_t)oy * out_w + ox) * c;
      for (int ch = 0; ch < c; ++ch) {
        float top = p00[ch] + (p01[ch] - p00[ch]) * wx;
        float bot = p10[ch] + (p11[ch] - p10[ch]) * wx;
        dst[ch] = top + (bot - top) * wy;
      }
    }
  }
}

static void op_resize_bilinear(PoseRuntime* rt, const PoseOpDef* op) {
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  parallel_rows(rt, op, out->dims[1], op_resize_bilinear_rows);
}

static void op_resize_bilinear_add_rows(PoseRuntime* rt, const PoseOpDef* resize_op, int y0_out, int y1_out) {
  const PoseOpDef* add_op = rt->task_aux_op;
  const float* in = (const float*)rt->tensor[resize_op->inputs[0]];
  int resize_out_idx = resize_op->outputs[0];
  int residual_idx = add_op->inputs[0] == resize_out_idx ? add_op->inputs[1] : add_op->inputs[0];
  const float* residual = (const float*)rt->tensor[residual_idx];
  float* out = (float*)rt->tensor[add_op->outputs[0]];
  const PoseTensorDef* in_def = &rt->def->tensors[resize_op->inputs[0]];
  const PoseTensorDef* out_def = &rt->def->tensors[add_op->outputs[0]];
  int in_h = in_def->dims[1], in_w = in_def->dims[2], c = in_def->dims[3];
  int out_h = out_def->dims[1], out_w = out_def->dims[2];
  for (int oy = y0_out; oy < y1_out; ++oy) {
    float fy;
    if (resize_op->align_corners && out_h > 1) {
      fy = (float)oy * (float)(in_h - 1) / (float)(out_h - 1);
    } else if (resize_op->half_pixel_centers) {
      fy = ((float)oy + 0.5f) * (float)in_h / (float)out_h - 0.5f;
    } else {
      fy = (float)oy * (float)in_h / (float)out_h;
    }
    if (fy < 0.0f) fy = 0.0f;
    int y0 = (int)floorf(fy);
    if (y0 >= in_h - 1) y0 = in_h - 1;
    int y1 = y0 + 1 < in_h ? y0 + 1 : y0;
    float wy = fy - (float)y0;
    for (int ox = 0; ox < out_w; ++ox) {
      float fx;
      if (resize_op->align_corners && out_w > 1) {
        fx = (float)ox * (float)(in_w - 1) / (float)(out_w - 1);
      } else if (resize_op->half_pixel_centers) {
        fx = ((float)ox + 0.5f) * (float)in_w / (float)out_w - 0.5f;
      } else {
        fx = (float)ox * (float)in_w / (float)out_w;
      }
      if (fx < 0.0f) fx = 0.0f;
      int x0 = (int)floorf(fx);
      if (x0 >= in_w - 1) x0 = in_w - 1;
      int x1 = x0 + 1 < in_w ? x0 + 1 : x0;
      float wx = fx - (float)x0;
      const float* p00 = in + ((size_t)y0 * in_w + x0) * c;
      const float* p01 = in + ((size_t)y0 * in_w + x1) * c;
      const float* p10 = in + ((size_t)y1 * in_w + x0) * c;
      const float* p11 = in + ((size_t)y1 * in_w + x1) * c;
      size_t out_offset = ((size_t)oy * out_w + ox) * c;
      const float* res = residual + out_offset;
      float* dst = out + out_offset;
      int ch = 0;
#if defined(__aarch64__)
      for (; ch + 4 <= c; ch += 4) {
        float32x4_t p00v = vld1q_f32(p00 + ch);
        float32x4_t p01v = vld1q_f32(p01 + ch);
        float32x4_t p10v = vld1q_f32(p10 + ch);
        float32x4_t p11v = vld1q_f32(p11 + ch);
        float32x4_t top = vfmaq_n_f32(p00v, vsubq_f32(p01v, p00v), wx);
        float32x4_t bot = vfmaq_n_f32(p10v, vsubq_f32(p11v, p10v), wx);
        float32x4_t v = vfmaq_n_f32(top, vsubq_f32(bot, top), wy);
        v = vaddq_f32(v, vld1q_f32(res + ch));
        v = activateq(v, add_op->activation);
        vst1q_f32(dst + ch, v);
      }
#endif
      for (; ch < c; ++ch) {
        float top = p00[ch] + (p01[ch] - p00[ch]) * wx;
        float bot = p10[ch] + (p11[ch] - p10[ch]) * wx;
        dst[ch] = activate(top + (bot - top) * wy + res[ch], add_op->activation);
      }
    }
  }
}

static void op_resize_bilinear_add(PoseRuntime* rt, const PoseOpDef* resize_op, const PoseOpDef* add_op) {
  const PoseTensorDef* out = &rt->def->tensors[add_op->outputs[0]];
  parallel_rows_aux(rt, resize_op, add_op, out->dims[1], op_resize_bilinear_add_rows);
}

static void op_max_pool_rows(PoseRuntime* rt, const PoseOpDef* op, int y0, int y1) {
  const float* in = (const float*)rt->tensor[op->inputs[0]];
  float* out = (float*)rt->tensor[op->outputs[0]];
  const PoseTensorDef* in_def = &rt->def->tensors[op->inputs[0]];
  const PoseTensorDef* out_def = &rt->def->tensors[op->outputs[0]];
  int in_h = in_def->dims[1], in_w = in_def->dims[2], c = in_def->dims[3];
  int out_h = out_def->dims[1], out_w = out_def->dims[2];
  int pad_top = op->padding == 0 ? same_pad_before(in_h, out_h, op->stride_h, op->filter_h, 1) : 0;
  int pad_left = op->padding == 0 ? same_pad_before(in_w, out_w, op->stride_w, op->filter_w, 1) : 0;
  for (int oy = y0; oy < y1; ++oy) {
    for (int ox = 0; ox < out_w; ++ox) {
      float* dst = out + ((size_t)oy * out_w + ox) * c;
      for (int ch = 0; ch < c; ++ch) dst[ch] = -INFINITY;
      for (int ky = 0; ky < op->filter_h; ++ky) {
        int iy = oy * op->stride_h + ky - pad_top;
        if ((unsigned)iy >= (unsigned)in_h) continue;
        for (int kx = 0; kx < op->filter_w; ++kx) {
          int ix = ox * op->stride_w + kx - pad_left;
          if ((unsigned)ix >= (unsigned)in_w) continue;
          const float* src = in + ((size_t)iy * in_w + ix) * c;
          for (int ch = 0; ch < c; ++ch) {
            if (src[ch] > dst[ch]) dst[ch] = src[ch];
          }
        }
      }
      for (int ch = 0; ch < c; ++ch) dst[ch] = activate(dst[ch], op->activation);
    }
  }
}

static void op_max_pool(PoseRuntime* rt, const PoseOpDef* op) {
  const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
  parallel_rows(rt, op, out->dims[1], op_max_pool_rows);
}

static void run_op(PoseRuntime* rt, const PoseOpDef* op) {
  switch (op->op) {
    case POSE_OP_ADD: op_add(rt, op); break;
    case POSE_OP_CONCAT: op_concat(rt, op); break;
    case POSE_OP_CONV2D: op_conv2d(rt, op); break;
    case POSE_OP_DEPTHWISE: op_depthwise(rt, op); break;
    case POSE_OP_DEPTH_TO_SPACE: op_depth_to_space(rt, op); break;
    case POSE_OP_LOGISTIC: op_logistic(rt, op); break;
    case POSE_OP_MAX_POOL2D: op_max_pool(rt, op); break;
    case POSE_OP_PAD: op_pad(rt, op); break;
    case POSE_OP_RESHAPE: op_reshape(rt, op); break;
    case POSE_OP_RESIZE_BILINEAR: op_resize_bilinear(rt, op); break;
    default:
      fprintf(stderr, "unsupported op code %d at source op %d\n", op->op, op->index);
      exit(2);
  }
}

static const char* op_name(int op) {
  switch (op) {
    case POSE_OP_ADD: return "ADD";
    case POSE_OP_CONCAT: return "CONCAT";
    case POSE_OP_CONV2D: return "CONV2D";
    case POSE_OP_DEPTHWISE: return "DEPTHWISE";
    case POSE_OP_DEPTH_TO_SPACE: return "DEPTH_TO_SPACE";
    case POSE_OP_LOGISTIC: return "LOGISTIC";
    case POSE_OP_MAX_POOL2D: return "MAX_POOL2D";
    case POSE_OP_PAD: return "PAD";
    case POSE_OP_RESHAPE: return "RESHAPE";
    case POSE_OP_RESIZE_BILINEAR: return "RESIZE_BILINEAR";
    default: return "UNKNOWN";
  }
}

static uint8_t* load_constants(const char* data_dir, const char* file, size_t* out_bytes) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", data_dir, file);
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  size_t bytes = file_size(f);
  uint8_t* data = (uint8_t*)xaligned_alloc(64, bytes);
  if (fread(data, 1, bytes, f) != bytes) {
    fprintf(stderr, "short read: %s\n", path);
    exit(2);
  }
  fclose(f);
  *out_bytes = bytes;
  return data;
}

static void runtime_init(PoseRuntime* rt, const PoseModelDef* model, const char* data_dir, int threads) {
  memset(rt, 0, sizeof(*rt));
  rt->def = model;
  rt->threads = threads < 1 ? 1 : threads;
  if (rt->threads > 8) rt->threads = 8;
  rt->pool_workers = rt->threads > 1 ? rt->threads - 1 : 0;
  pthread_mutex_init(&rt->pool_mutex, NULL);
  pthread_cond_init(&rt->pool_start, NULL);
  pthread_cond_init(&rt->pool_done, NULL);
  rt->constants = load_constants(data_dir, model->const_file, &rt->constants_bytes);
  rt->tensor = (void**)calloc((size_t)model->tensor_count, sizeof(void*));
  rt->packed = (void**)calloc((size_t)model->tensor_count, sizeof(void*));
  rt->packed_type = (uint8_t*)calloc((size_t)model->tensor_count, sizeof(uint8_t));
  rt->owned = (uint8_t*)calloc((size_t)model->tensor_count, sizeof(uint8_t));
  if (!rt->tensor || !rt->packed || !rt->packed_type || !rt->owned) {
    fprintf(stderr, "runtime metadata allocation failed\n");
    exit(2);
  }
  for (int i = 0; i < model->tensor_count; ++i) {
    const PoseTensorDef* t = &model->tensors[i];
    if (t->is_const) {
      if ((size_t)t->const_offset + t->byte_count > rt->constants_bytes) {
        fprintf(stderr, "constant tensor %d outside blob\n", i);
        exit(2);
      }
      rt->tensor[i] = rt->constants + t->const_offset;
    } else if (t->type == POSE_TENSOR_FLOAT32 || t->type == POSE_TENSOR_INT32) {
      size_t bytes = tensor_bytes(t);
      if (bytes) {
        rt->tensor[i] = xaligned_alloc(64, bytes);
        rt->owned[i] = 1;
      }
    }
  }
  int pack_conv_weights = 1;
  const char* pack_env = getenv("POSE_PACK_CONV");
  if (pack_env) pack_conv_weights = strcmp(pack_env, "0") != 0;
  if (pack_conv_weights) {
    for (int i = 0; i < model->op_count; ++i) {
      const PoseOpDef* op = &model->ops[i];
      if (op->op != POSE_OP_CONV2D) continue;
      const PoseTensorDef* wt = &model->tensors[op->inputs[1]];
      int is_pointwise = op->filter_h == 1 && op->filter_w == 1 && op->stride_h == 1 && op->stride_w == 1;
      int is_rgb_stride2 =
          wt->rank == 4 && wt->dims[0] == 24 && wt->dims[1] == 3 && wt->dims[2] == 3 && wt->dims[3] == 3 &&
          op->stride_h == 2 && op->stride_w == 2;
      if (!is_pointwise && !is_rgb_stride2) continue;
      int weight_idx = op->inputs[1];
      if (weight_idx < 0 || rt->packed[weight_idx]) continue;
      if (is_pointwise) {
        rt->packed[weight_idx] = pack_pointwise_x8(
            wt, (const float*)rt->tensor[weight_idx], (const float*)rt->tensor[op->inputs[2]]);
        rt->packed_type[weight_idx] = POSE_PACK_POINTWISE_X8;
      } else {
        rt->packed[weight_idx] = pack_conv_weights_icoc(
            wt, (const float*)rt->tensor[weight_idx]);
        rt->packed_type[weight_idx] = POSE_PACK_ICOC;
      }
    }
  }
}

static void runtime_start_pool(PoseRuntime* rt) {
  for (int i = 0; i < rt->pool_workers; ++i) {
    rt->worker_args[i] = (PoolWorker){rt, i + 1};
    int err = pthread_create(&rt->worker_tids[i], NULL, pool_worker_main, &rt->worker_args[i]);
    if (err) {
      fprintf(stderr, "pthread_create failed: %s\n", strerror(err));
      exit(2);
    }
  }
}

static void runtime_destroy(PoseRuntime* rt) {
  if (rt->pool_workers > 0) {
    pthread_mutex_lock(&rt->pool_mutex);
    rt->pool_stop = 1;
    pthread_cond_broadcast(&rt->pool_start);
    pthread_mutex_unlock(&rt->pool_mutex);
    for (int i = 0; i < rt->pool_workers; ++i) pthread_join(rt->worker_tids[i], NULL);
  }
  pthread_cond_destroy(&rt->pool_done);
  pthread_cond_destroy(&rt->pool_start);
  pthread_mutex_destroy(&rt->pool_mutex);
  for (int i = 0; i < rt->def->tensor_count; ++i) {
    if (rt->owned[i]) free(rt->tensor[i]);
    if (rt->packed[i]) free(rt->packed[i]);
  }
  free(rt->tensor);
  free(rt->packed);
  free(rt->packed_type);
  free(rt->owned);
  free(rt->constants);
}

static void print_profile(PoseRuntime* rt, const double* times_ms) {
  int top[24];
  int top_count = rt->def->op_count < 24 ? rt->def->op_count : 24;
  for (int i = 0; i < top_count; ++i) top[i] = -1;
  double totals[16] = {0};
  for (int i = 0; i < rt->def->op_count; ++i) {
    const PoseOpDef* op = &rt->def->ops[i];
    if (op->op >= 0 && op->op < 16) totals[op->op] += times_ms[i];
    for (int slot = 0; slot < top_count; ++slot) {
      if (top[slot] < 0 || times_ms[i] > times_ms[top[slot]]) {
        for (int j = top_count - 1; j > slot; --j) top[j] = top[j - 1];
        top[slot] = i;
        break;
      }
    }
  }
  printf("profile totals for %s:\n", rt->def->name);
  for (int op = 0; op < 16; ++op) {
    if (totals[op] > 0.001) printf("  %-16s %.3f ms\n", op_name(op), totals[op]);
  }
  printf("profile top ops:\n");
  for (int slot = 0; slot < top_count; ++slot) {
    int i = top[slot];
    if (i < 0 || times_ms[i] <= 0.001) continue;
    const PoseOpDef* op = &rt->def->ops[i];
    const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
    printf("  src_op=%03d %-16s %.3f ms out=[", op->index, op_name(op->op), times_ms[i]);
    for (int d = 0; d < out->rank; ++d) {
      printf("%s%d", d ? "," : "", out->dims[d]);
    }
    printf("]\n");
  }
}

static int tensor_consumer_count_after(const PoseModelDef* def, int op_start, int tensor_idx) {
  int count = 0;
  for (int i = op_start; i < def->op_count; ++i) {
    const PoseOpDef* op = &def->ops[i];
    for (int j = 0; j < op->input_count; ++j) {
      if (op->inputs[j] == tensor_idx) count++;
    }
  }
  return count;
}

static int same_tensor_shape(const PoseTensorDef* a, const PoseTensorDef* b) {
  if (a->rank != b->rank) return 0;
  for (int i = 0; i < a->rank; ++i) {
    if (a->dims[i] != b->dims[i]) return 0;
  }
  return 1;
}

static int can_fuse_pad_depthwise(const PoseRuntime* rt, int op_index) {
  if (op_index + 1 >= rt->def->op_count) return 0;
  const PoseOpDef* pad_op = &rt->def->ops[op_index];
  const PoseOpDef* dw_op = &rt->def->ops[op_index + 1];
  if (pad_op->op != POSE_OP_PAD || dw_op->op != POSE_OP_DEPTHWISE) return 0;
  if (dw_op->inputs[0] != pad_op->outputs[0]) return 0;
  if (tensor_consumer_count_after(rt->def, op_index + 1, pad_op->outputs[0]) != 1) return 0;
  if (dw_op->depth_multiplier != 1 || dw_op->dilation_h != 1 || dw_op->dilation_w != 1) return 0;
  const PoseTensorDef* in = &rt->def->tensors[pad_op->inputs[0]];
  const PoseTensorDef* padded = &rt->def->tensors[pad_op->outputs[0]];
  const PoseTensorDef* wt = &rt->def->tensors[dw_op->inputs[1]];
  const PoseTensorDef* out = &rt->def->tensors[dw_op->outputs[0]];
  if (in->rank != 4 || padded->rank != 4 || wt->rank != 4 || out->rank != 4) return 0;
  if (in->dims[3] != out->dims[3] || padded->dims[3] != in->dims[3]) return 0;
  if (wt->dims[0] != 1 || wt->dims[3] != in->dims[3]) return 0;
  return 1;
}

static int can_fuse_pointwise_conv_add(const PoseRuntime* rt, int op_index) {
  if (op_index + 1 >= rt->def->op_count) return 0;
  const PoseOpDef* conv_op = &rt->def->ops[op_index];
  const PoseOpDef* add_op = &rt->def->ops[op_index + 1];
  if (conv_op->op != POSE_OP_CONV2D || add_op->op != POSE_OP_ADD) return 0;
  if (conv_op->activation != 0) return 0;
  if (add_op->input_count != 2 || add_op->output_count != 1) return 0;
  int conv_out = conv_op->outputs[0];
  int residual_idx = -1;
  if (add_op->inputs[0] == conv_out) residual_idx = add_op->inputs[1];
  if (add_op->inputs[1] == conv_out) residual_idx = add_op->inputs[0];
  if (residual_idx < 0) return 0;
  if (tensor_consumer_count_after(rt->def, op_index + 1, conv_out) != 1) return 0;
  if (rt->packed_type[conv_op->inputs[1]] != POSE_PACK_POINTWISE_X8) return 0;
  const PoseTensorDef* wt = &rt->def->tensors[conv_op->inputs[1]];
  if (wt->dims[1] != 1 || wt->dims[2] != 1 || conv_op->stride_h != 1 || conv_op->stride_w != 1) return 0;
  const PoseTensorDef* conv_out_def = &rt->def->tensors[conv_out];
  const PoseTensorDef* residual = &rt->def->tensors[residual_idx];
  const PoseTensorDef* add_out = &rt->def->tensors[add_op->outputs[0]];
  if (conv_out_def->rank != 4 || residual->rank != 4 || add_out->rank != 4) return 0;
  if (!same_tensor_shape(conv_out_def, residual) || !same_tensor_shape(conv_out_def, add_out)) return 0;
  return 1;
}

static int can_fuse_pad_rgb_stride2_conv(const PoseRuntime* rt, int op_index) {
  if (op_index + 1 >= rt->def->op_count) return 0;
  const PoseOpDef* pad_op = &rt->def->ops[op_index];
  const PoseOpDef* conv_op = &rt->def->ops[op_index + 1];
  if (pad_op->op != POSE_OP_PAD || conv_op->op != POSE_OP_CONV2D) return 0;
  if (conv_op->inputs[0] != pad_op->outputs[0]) return 0;
  if (tensor_consumer_count_after(rt->def, op_index + 1, pad_op->outputs[0]) != 1) return 0;
  if (conv_op->stride_h != 2 || conv_op->stride_w != 2 ||
      conv_op->dilation_h != 1 || conv_op->dilation_w != 1 ||
      conv_op->padding != 1) {
    return 0;
  }
  if (rt->packed_type[conv_op->inputs[1]] != POSE_PACK_ICOC) return 0;
  const PoseTensorDef* in = &rt->def->tensors[pad_op->inputs[0]];
  const PoseTensorDef* padded = &rt->def->tensors[pad_op->outputs[0]];
  const PoseTensorDef* wt = &rt->def->tensors[conv_op->inputs[1]];
  const PoseTensorDef* out = &rt->def->tensors[conv_op->outputs[0]];
  if (in->rank != 4 || padded->rank != 4 || wt->rank != 4 || out->rank != 4) return 0;
  if (in->dims[3] != 3 || padded->dims[3] != 3) return 0;
  if (wt->dims[0] != 24 || wt->dims[1] != 3 || wt->dims[2] != 3 || wt->dims[3] != 3) return 0;
  if (out->dims[3] != 24) return 0;
  return 1;
}

static int can_fuse_resize_add(const PoseRuntime* rt, int op_index) {
  if (op_index + 1 >= rt->def->op_count) return 0;
  const PoseOpDef* resize_op = &rt->def->ops[op_index];
  const PoseOpDef* add_op = &rt->def->ops[op_index + 1];
  if (resize_op->op != POSE_OP_RESIZE_BILINEAR || add_op->op != POSE_OP_ADD) return 0;
  if (add_op->input_count != 2 || add_op->output_count != 1) return 0;
  int resize_out = resize_op->outputs[0];
  int residual_idx = -1;
  if (add_op->inputs[0] == resize_out) residual_idx = add_op->inputs[1];
  if (add_op->inputs[1] == resize_out) residual_idx = add_op->inputs[0];
  if (residual_idx < 0) return 0;
  if (tensor_consumer_count_after(rt->def, op_index + 1, resize_out) != 1) return 0;
  const PoseTensorDef* resize_out_def = &rt->def->tensors[resize_out];
  const PoseTensorDef* residual = &rt->def->tensors[residual_idx];
  const PoseTensorDef* add_out = &rt->def->tensors[add_op->outputs[0]];
  if (resize_out_def->rank != 4 || residual->rank != 4 || add_out->rank != 4) return 0;
  if (!same_tensor_shape(resize_out_def, residual) || !same_tensor_shape(resize_out_def, add_out)) return 0;
  return 1;
}

static int run_model_step(PoseRuntime* rt, int i, int end_exclusive, double* times_ms) {
  int profile = times_ms != NULL;
  if (i + 1 < end_exclusive && can_fuse_pad_rgb_stride2_conv(rt, i)) {
    double t0 = profile ? now_s() : 0.0;
    op_conv2d_3x3s2_rgb24_from_pad(rt, &rt->def->ops[i], &rt->def->ops[i + 1]);
    if (profile) times_ms[i + 1] += (now_s() - t0) * 1000.0;
    return i + 2;
  }
  if (i + 1 < end_exclusive && can_fuse_pointwise_conv_add(rt, i)) {
    double t0 = profile ? now_s() : 0.0;
    op_conv2d_1x1_add_packed(rt, &rt->def->ops[i], &rt->def->ops[i + 1]);
    if (profile) times_ms[i] += (now_s() - t0) * 1000.0;
    return i + 2;
  }
  if (i + 1 < end_exclusive && can_fuse_pad_depthwise(rt, i)) {
    double t0 = profile ? now_s() : 0.0;
    op_depthwise_from_pad(rt, &rt->def->ops[i], &rt->def->ops[i + 1]);
    if (profile) times_ms[i + 1] += (now_s() - t0) * 1000.0;
    return i + 2;
  }
  if (i + 1 < end_exclusive && can_fuse_resize_add(rt, i)) {
    double t0 = profile ? now_s() : 0.0;
    op_resize_bilinear_add(rt, &rt->def->ops[i], &rt->def->ops[i + 1]);
    if (profile) times_ms[i] += (now_s() - t0) * 1000.0;
    return i + 2;
  }
  double t0 = profile ? now_s() : 0.0;
  run_op(rt, &rt->def->ops[i]);
  if (profile) times_ms[i] += (now_s() - t0) * 1000.0;
  return i + 1;
}

static void run_model_until(PoseRuntime* rt, int end_exclusive) {
  if (end_exclusive < 0) end_exclusive = 0;
  if (end_exclusive > rt->def->op_count) end_exclusive = rt->def->op_count;
  for (int i = 0; i < end_exclusive;) {
    i = run_model_step(rt, i, end_exclusive, NULL);
  }
}

static void run_model(PoseRuntime* rt, int profile) {
  double* times_ms = NULL;
  if (profile) times_ms = (double*)calloc((size_t)rt->def->op_count, sizeof(double));
  for (int i = 0; i < rt->def->op_count;) {
    i = run_model_step(rt, i, rt->def->op_count, times_ms);
  }
  if (profile) {
    print_profile(rt, times_ms);
    free(times_ms);
  }
}

typedef enum {
  BENCH_SINGLE = 0,
  BENCH_PAD_RGB_CONV = 1,
  BENCH_POINTWISE_CONV_ADD = 2,
  BENCH_PAD_DEPTHWISE = 3,
  BENCH_RESIZE_ADD = 4,
} BenchKind;

typedef struct {
  BenchKind kind;
  int requested_pos;
  int start_pos;
  int timed_pos;
  int aux_pos;
} BenchTarget;

typedef struct {
  int count;
  int tensor_idx[16];
  void* copy[16];
  size_t bytes[16];
} BenchSnapshots;

static const PoseModelDef* select_model(const char* name);

static int find_op_pos_by_source_index(const PoseModelDef* def, int source_op_index) {
  for (int i = 0; i < def->op_count; ++i) {
    if (def->ops[i].index == source_op_index) return i;
  }
  return -1;
}

static const char* bench_kind_name(BenchKind kind) {
  switch (kind) {
    case BENCH_PAD_RGB_CONV: return "PAD+CONV2D";
    case BENCH_POINTWISE_CONV_ADD: return "CONV2D+ADD";
    case BENCH_PAD_DEPTHWISE: return "PAD+DEPTHWISE";
    case BENCH_RESIZE_ADD: return "RESIZE_BILINEAR+ADD";
    case BENCH_SINGLE:
    default: return "SINGLE";
  }
}

static BenchTarget resolve_bench_target(PoseRuntime* rt, int source_op_index, int prefer_fused) {
  BenchTarget target = {BENCH_SINGLE, -1, -1, -1, -1};
  int pos = find_op_pos_by_source_index(rt->def, source_op_index);
  if (pos < 0) {
    fprintf(stderr, "source op %d not found in model %s\n", source_op_index, rt->def->name);
    exit(2);
  }
  target.requested_pos = pos;
  target.start_pos = pos;
  target.timed_pos = pos;
  if (prefer_fused) {
    if (pos > 0 && can_fuse_pad_rgb_stride2_conv(rt, pos - 1)) {
      target.kind = BENCH_PAD_RGB_CONV;
      target.start_pos = pos - 1;
      target.timed_pos = pos;
      target.aux_pos = pos;
      return target;
    }
    if (pos > 0 && can_fuse_pointwise_conv_add(rt, pos - 1)) {
      target.kind = BENCH_POINTWISE_CONV_ADD;
      target.start_pos = pos - 1;
      target.timed_pos = pos - 1;
      target.aux_pos = pos;
      return target;
    }
    if (pos > 0 && can_fuse_pad_depthwise(rt, pos - 1)) {
      target.kind = BENCH_PAD_DEPTHWISE;
      target.start_pos = pos - 1;
      target.timed_pos = pos;
      target.aux_pos = pos;
      return target;
    }
    if (can_fuse_pad_rgb_stride2_conv(rt, pos)) {
      target.kind = BENCH_PAD_RGB_CONV;
      target.start_pos = pos;
      target.timed_pos = pos + 1;
      target.aux_pos = pos + 1;
      return target;
    }
    if (can_fuse_pointwise_conv_add(rt, pos)) {
      target.kind = BENCH_POINTWISE_CONV_ADD;
      target.start_pos = pos;
      target.timed_pos = pos;
      target.aux_pos = pos + 1;
      return target;
    }
    if (can_fuse_pad_depthwise(rt, pos)) {
      target.kind = BENCH_PAD_DEPTHWISE;
      target.start_pos = pos;
      target.timed_pos = pos + 1;
      target.aux_pos = pos + 1;
      return target;
    }
    if (pos > 0 && can_fuse_resize_add(rt, pos - 1)) {
      target.kind = BENCH_RESIZE_ADD;
      target.start_pos = pos - 1;
      target.timed_pos = pos - 1;
      target.aux_pos = pos;
      return target;
    }
    if (can_fuse_resize_add(rt, pos)) {
      target.kind = BENCH_RESIZE_ADD;
      target.start_pos = pos;
      target.timed_pos = pos;
      target.aux_pos = pos + 1;
      return target;
    }
  }
  return target;
}

static const PoseOpDef* bench_output_op(const PoseRuntime* rt, const BenchTarget* target) {
  if (target->kind == BENCH_PAD_RGB_CONV ||
      target->kind == BENCH_POINTWISE_CONV_ADD ||
      target->kind == BENCH_PAD_DEPTHWISE ||
      target->kind == BENCH_RESIZE_ADD) {
    return &rt->def->ops[target->aux_pos];
  }
  return &rt->def->ops[target->timed_pos];
}

static uint64_t op_mac_count(const PoseRuntime* rt, const PoseOpDef* op) {
  if (op->op == POSE_OP_CONV2D) {
    const PoseTensorDef* wt = &rt->def->tensors[op->inputs[1]];
    const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
    if (wt->rank != 4 || out->rank != 4) return 0;
    return (uint64_t)out->dims[1] * (uint64_t)out->dims[2] * (uint64_t)out->dims[3] *
        (uint64_t)wt->dims[1] * (uint64_t)wt->dims[2] * (uint64_t)wt->dims[3];
  }
  if (op->op == POSE_OP_DEPTHWISE) {
    const PoseTensorDef* wt = &rt->def->tensors[op->inputs[1]];
    const PoseTensorDef* out = &rt->def->tensors[op->outputs[0]];
    if (wt->rank != 4 || out->rank != 4) return 0;
    return (uint64_t)out->dims[1] * (uint64_t)out->dims[2] * (uint64_t)out->dims[3] *
        (uint64_t)wt->dims[1] * (uint64_t)wt->dims[2];
  }
  return 0;
}

static uint64_t bench_target_mac_count(const PoseRuntime* rt, const BenchTarget* target) {
  switch (target->kind) {
    case BENCH_PAD_RGB_CONV:
    case BENCH_PAD_DEPTHWISE:
      return op_mac_count(rt, &rt->def->ops[target->aux_pos]);
    case BENCH_POINTWISE_CONV_ADD:
      return op_mac_count(rt, &rt->def->ops[target->timed_pos]);
    case BENCH_RESIZE_ADD:
      return 0;
    case BENCH_SINGLE:
    default:
      return op_mac_count(rt, &rt->def->ops[target->timed_pos]);
  }
}

static void run_bench_target_once(PoseRuntime* rt, const BenchTarget* target) {
  switch (target->kind) {
    case BENCH_PAD_RGB_CONV:
      op_conv2d_3x3s2_rgb24_from_pad(
          rt, &rt->def->ops[target->start_pos], &rt->def->ops[target->aux_pos]);
      break;
    case BENCH_POINTWISE_CONV_ADD:
      op_conv2d_1x1_add_packed(
          rt, &rt->def->ops[target->timed_pos], &rt->def->ops[target->aux_pos]);
      break;
    case BENCH_PAD_DEPTHWISE:
      op_depthwise_from_pad(
          rt, &rt->def->ops[target->start_pos], &rt->def->ops[target->aux_pos]);
      break;
    case BENCH_RESIZE_ADD:
      op_resize_bilinear_add(
          rt, &rt->def->ops[target->start_pos], &rt->def->ops[target->aux_pos]);
      break;
    case BENCH_SINGLE:
    default:
      run_op(rt, &rt->def->ops[target->timed_pos]);
      break;
  }
}

static void bench_snapshot_add_tensor(PoseRuntime* rt, BenchSnapshots* snaps, int tensor_idx) {
  if (tensor_idx < 0) return;
  const PoseTensorDef* t = &rt->def->tensors[tensor_idx];
  if (t->is_const) return;
  size_t bytes = tensor_bytes(t);
  if (!bytes) return;
  for (int i = 0; i < snaps->count; ++i) {
    if (snaps->tensor_idx[i] == tensor_idx) return;
  }
  if (snaps->count >= 16) {
    fprintf(stderr, "too many benchmark input tensors to snapshot\n");
    exit(2);
  }
  int slot = snaps->count++;
  snaps->tensor_idx[slot] = tensor_idx;
  snaps->bytes[slot] = bytes;
  snaps->copy[slot] = xaligned_alloc(64, bytes);
  memcpy(snaps->copy[slot], rt->tensor[tensor_idx], bytes);
}

static void bench_snapshot_op_inputs(PoseRuntime* rt, BenchSnapshots* snaps, const PoseOpDef* op, int skip_tensor) {
  for (int i = 0; i < op->input_count; ++i) {
    if (op->inputs[i] == skip_tensor) continue;
    bench_snapshot_add_tensor(rt, snaps, op->inputs[i]);
  }
}

static BenchSnapshots bench_snapshot_inputs(PoseRuntime* rt, const BenchTarget* target) {
  BenchSnapshots snaps;
  memset(&snaps, 0, sizeof(snaps));
  switch (target->kind) {
    case BENCH_PAD_RGB_CONV:
    case BENCH_PAD_DEPTHWISE:
      bench_snapshot_op_inputs(rt, &snaps, &rt->def->ops[target->start_pos], -1);
      bench_snapshot_op_inputs(
          rt, &snaps, &rt->def->ops[target->aux_pos],
          rt->def->ops[target->start_pos].outputs[0]);
      break;
    case BENCH_POINTWISE_CONV_ADD:
      bench_snapshot_op_inputs(rt, &snaps, &rt->def->ops[target->timed_pos], -1);
      bench_snapshot_op_inputs(
          rt, &snaps, &rt->def->ops[target->aux_pos],
          rt->def->ops[target->timed_pos].outputs[0]);
      break;
    case BENCH_RESIZE_ADD:
      bench_snapshot_op_inputs(rt, &snaps, &rt->def->ops[target->start_pos], -1);
      bench_snapshot_op_inputs(
          rt, &snaps, &rt->def->ops[target->aux_pos],
          rt->def->ops[target->start_pos].outputs[0]);
      break;
    case BENCH_SINGLE:
    default:
      bench_snapshot_op_inputs(rt, &snaps, &rt->def->ops[target->timed_pos], -1);
      break;
  }
  return snaps;
}

static void bench_restore_inputs(PoseRuntime* rt, const BenchSnapshots* snaps) {
  for (int i = 0; i < snaps->count; ++i) {
    memcpy(rt->tensor[snaps->tensor_idx[i]], snaps->copy[i], snaps->bytes[i]);
  }
}

static void bench_free_snapshots(BenchSnapshots* snaps) {
  for (int i = 0; i < snaps->count; ++i) free(snaps->copy[i]);
  memset(snaps, 0, sizeof(*snaps));
}

static void print_tensor_shape(const PoseTensorDef* t) {
  printf("[");
  for (int d = 0; d < t->rank; ++d) {
    printf("%s%d", d ? "," : "", t->dims[d]);
  }
  printf("]");
}

static int run_bench_op(
    const char* data_dir, const char* model_name, const char* input_path,
    int source_op_index, int threads, int reps, int warmup) {
  if (threads < 1) threads = 1;
  if (reps < 1) reps = 1;
  if (warmup < 0) warmup = 0;
  const PoseModelDef* model = select_model(model_name);
  PoseRuntime rt;
  runtime_init(&rt, model, data_dir, threads);
  runtime_start_pool(&rt);
  int input_idx = model->inputs[0];
  read_file_exact(input_path, rt.tensor[input_idx], tensor_bytes(&model->tensors[input_idx]));

  int prefer_fused = 1;
  const char* fused_env = getenv("POSE_BENCH_FUSED");
  if (fused_env && strcmp(fused_env, "0") == 0) prefer_fused = 0;
  BenchTarget target = resolve_bench_target(&rt, source_op_index, prefer_fused);
  run_model_until(&rt, target.start_pos);

  BenchSnapshots snaps = bench_snapshot_inputs(&rt, &target);
  for (int i = 0; i < warmup; ++i) {
    bench_restore_inputs(&rt, &snaps);
    run_bench_target_once(&rt, &target);
  }
  double total_ms = 0.0;
  for (int i = 0; i < reps; ++i) {
    bench_restore_inputs(&rt, &snaps);
    double t0 = now_s();
    run_bench_target_once(&rt, &target);
    total_ms += (now_s() - t0) * 1000.0;
  }
  double avg_ms = total_ms / (double)reps;
  uint64_t macs = bench_target_mac_count(&rt, &target);
  const PoseOpDef* req_op = &rt.def->ops[target.requested_pos];
  const PoseOpDef* timed_op = &rt.def->ops[target.timed_pos];
  const PoseOpDef* out_op = bench_output_op(&rt, &target);
  const PoseTensorDef* out = &rt.def->tensors[out_op->outputs[0]];
  printf("bench_op model=%s request_src_op=%03d request_op=%s kind=%s timed_src_op=%03d timed_op=%s",
      model->name, req_op->index, op_name(req_op->op), bench_kind_name(target.kind),
      timed_op->index, op_name(timed_op->op));
  if (target.aux_pos >= 0) {
    const PoseOpDef* aux_op = &rt.def->ops[target.aux_pos];
    printf(" aux_src_op=%03d aux_op=%s", aux_op->index, op_name(aux_op->op));
  }
  printf(" threads=%d reps=%d warmup=%d avg_ms=%.6f", rt.threads, reps, warmup, avg_ms);
  if (macs) {
    printf(" macs=%llu gmac_s=%.6f gflop_s_fma2=%.6f",
        (unsigned long long)macs, (double)macs / (avg_ms * 1.0e6),
        2.0 * (double)macs / (avg_ms * 1.0e6));
  }
  printf(" out=");
  print_tensor_shape(out);
  printf("\n");

  bench_free_snapshots(&snaps);
  runtime_destroy(&rt);
  return 0;
}

static const PoseModelDef* select_model(const char* name) {
  if (strcmp(name, "detector") == 0 || strcmp(name, "pose_detector") == 0) {
    return &pose_detector_model;
  }
  if (strcmp(name, "landmarker") == 0 || strcmp(name, "pose_landmarker") == 0) {
    return &pose_landmarker_model;
  }
  fprintf(stderr, "unknown model '%s' (use detector or landmarker)\n", name);
  exit(2);
}

static void dump_outputs(PoseRuntime* rt, const char* out_dir) {
  char path[1024];
  for (int i = 0; i < rt->def->output_count; ++i) {
    int idx = rt->def->outputs[i];
    const PoseTensorDef* t = &rt->def->tensors[idx];
    snprintf(path, sizeof(path), "%s/%s_tensor_%d.bin", out_dir, rt->def->name, idx);
    write_file_exact(path, rt->tensor[idx], tensor_bytes(t));
  }
}

static void compare_outputs(PoseRuntime* rt, const char* ref_dir) {
  char path[1024];
  for (int i = 0; i < rt->def->output_count; ++i) {
    int idx = rt->def->outputs[i];
    const PoseTensorDef* t = &rt->def->tensors[idx];
    size_t bytes = tensor_bytes(t);
    float* ref = (float*)xaligned_alloc(64, bytes);
    snprintf(path, sizeof(path), "%s/%s_tensor_%d.bin", ref_dir, rt->def->name, idx);
    read_file_exact(path, ref, bytes);
    const float* out = (const float*)rt->tensor[idx];
    size_t n = tensor_elements(t);
    double mae = 0.0;
    float max_abs = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      float d = fabsf(out[j] - ref[j]);
      mae += d;
      if (d > max_abs) max_abs = d;
    }
    printf("compare tensor_%d max_abs=%.9g mean_abs=%.9g\n", idx, max_abs, mae / (double)n);
    free(ref);
  }
}

typedef struct {
  float left;
  float top;
  float right;
  float bottom;
} Letterbox;

typedef struct {
  float score;
  float box_xmin;
  float box_ymin;
  float box_width;
  float box_height;
  float keypoints[4][2];
} PoseDetection;

typedef struct {
  float x_center;
  float y_center;
  float width;
  float height;
  float rotation;
} PoseRect;

typedef struct {
  float x;
  float y;
  float z;
  float visibility;
  float presence;
} PoseLandmark;

static inline float clampf_local(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline float normalize_radians_c(float angle) {
  while (angle > (float)M_PI) angle -= 2.0f * (float)M_PI;
  while (angle <= -(float)M_PI) angle += 2.0f * (float)M_PI;
  return angle;
}

static float sigmoid_scalar(float x) {
  if (x > 100.0f) return 1.0f;
  if (x < -100.0f) return 0.0f;
  return 1.0f / (1.0f + expf(-x));
}

static uint8_t* read_rgb24(const char* path, int w, int h) {
  size_t bytes = (size_t)w * h * 3;
  uint8_t* rgb = (uint8_t*)xaligned_alloc(64, bytes);
  read_file_exact(path, rgb, bytes);
  return rgb;
}

static void sample_rgb_norm(
    const uint8_t* rgb, int w, int h, float x, float y, float out[3]) {
  x = clampf_local(x, 0.0f, (float)(w - 1));
  y = clampf_local(y, 0.0f, (float)(h - 1));
  int x0 = (int)floorf(x);
  int y0 = (int)floorf(y);
  int x1 = x0 + 1 < w ? x0 + 1 : x0;
  int y1 = y0 + 1 < h ? y0 + 1 : y0;
  float wx = x - (float)x0;
  float wy = y - (float)y0;
  const uint8_t* p00 = rgb + ((size_t)y0 * w + x0) * 3;
  const uint8_t* p01 = rgb + ((size_t)y0 * w + x1) * 3;
  const uint8_t* p10 = rgb + ((size_t)y1 * w + x0) * 3;
  const uint8_t* p11 = rgb + ((size_t)y1 * w + x1) * 3;
  for (int c = 0; c < 3; ++c) {
    float top = (float)p00[c] + ((float)p01[c] - (float)p00[c]) * wx;
    float bot = (float)p10[c] + ((float)p11[c] - (float)p10[c]) * wx;
    out[c] = ((top + (bot - top) * wy) / 127.5f) - 1.0f;
  }
}

static Letterbox resize_letterbox_rgb(
    const uint8_t* rgb, int w, int h, int size, float* out) {
  memset(out, 0, (size_t)size * size * 3 * sizeof(float));
  float scale = (float)size / (float)(w > h ? w : h);
  int resized_w = (int)lroundf((float)w * scale);
  int resized_h = (int)lroundf((float)h * scale);
  if (resized_w < 1) resized_w = 1;
  if (resized_h < 1) resized_h = 1;
  int left = (size - resized_w) / 2;
  int top = (size - resized_h) / 2;
  for (int oy = 0; oy < resized_h; ++oy) {
    float sy = ((float)oy + 0.5f) * (float)h / (float)resized_h - 0.5f;
    for (int ox = 0; ox < resized_w; ++ox) {
      float sx = ((float)ox + 0.5f) * (float)w / (float)resized_w - 0.5f;
      float pix[3];
      sample_rgb_norm(rgb, w, h, sx, sy, pix);
      float* dst = out + ((size_t)(oy + top) * size + (ox + left)) * 3;
      dst[0] = pix[0];
      dst[1] = pix[1];
      dst[2] = pix[2];
    }
  }
  Letterbox lb = {
      (float)left / (float)size,
      (float)top / (float)size,
      (float)(size - left - resized_w) / (float)size,
      (float)(size - top - resized_h) / (float)size};
  return lb;
}

static int decode_best_detection(
    const float* raw_boxes, const float* raw_scores, Letterbox lb, PoseDetection* best) {
  const int strides[5] = {8, 16, 32, 32, 32};
  const float min_scale = 0.1484375f;
  const float max_scale = 0.75f;
  const int input_size = 224;
  const float x_scale = 1.0f - lb.left - lb.right;
  const float y_scale = 1.0f - lb.top - lb.bottom;
  int index = 0;
  int best_index = -1;
  float best_score = 0.5f;
  float best_ax = 0.0f, best_ay = 0.0f;
  int layer = 0;
  while (layer < 5) {
    int same_stride = layer;
    int anchors_per_location = 0;
    while (same_stride < 5 && strides[same_stride] == strides[layer]) {
      float scale = min_scale + (max_scale - min_scale) * (float)same_stride / 4.0f;
      float next_scale = same_stride == 4 ? 1.0f :
          min_scale + (max_scale - min_scale) * (float)(same_stride + 1) / 4.0f;
      (void)scale;
      (void)next_scale;
      anchors_per_location += 2;
      same_stride++;
    }
    int feature = (input_size + strides[layer] - 1) / strides[layer];
    for (int y = 0; y < feature; ++y) {
      for (int x = 0; x < feature; ++x) {
        float ax = ((float)x + 0.5f) / (float)feature;
        float ay = ((float)y + 0.5f) / (float)feature;
        for (int a = 0; a < anchors_per_location; ++a, ++index) {
          float score = sigmoid_scalar(raw_scores[index]);
          if (score > best_score) {
            best_score = score;
            best_index = index;
            best_ax = ax;
            best_ay = ay;
          }
        }
      }
    }
    layer = same_stride;
  }
  if (index != 2254 || best_index < 0) return 0;
  const float* raw = raw_boxes + (size_t)best_index * 12;
  float x_center = raw[0] / 224.0f + best_ax;
  float y_center = raw[1] / 224.0f + best_ay;
  float box_w = raw[2] / 224.0f;
  float box_h = raw[3] / 224.0f;
  best->score = best_score;
  best->box_xmin = (x_center - box_w * 0.5f - lb.left) / x_scale;
  best->box_ymin = (y_center - box_h * 0.5f - lb.top) / y_scale;
  best->box_width = box_w / x_scale;
  best->box_height = box_h / y_scale;
  for (int k = 0; k < 4; ++k) {
    float kx = raw[4 + 2 * k] / 224.0f + best_ax;
    float ky = raw[5 + 2 * k] / 224.0f + best_ay;
    best->keypoints[k][0] = (kx - lb.left) / x_scale;
    best->keypoints[k][1] = (ky - lb.top) / y_scale;
  }
  return 1;
}

static PoseRect rect_from_detection_c(const PoseDetection* det, int image_w, int image_h) {
  float x0 = det->keypoints[0][0] * (float)image_w;
  float y0 = det->keypoints[0][1] * (float)image_h;
  float x1 = det->keypoints[1][0] * (float)image_w;
  float y1 = det->keypoints[1][1] * (float)image_h;
  float dx = x1 - x0;
  float dy = y1 - y0;
  float box_size = hypotf(dx, dy) * 2.0f;
  float rotation = normalize_radians_c((float)M_PI * 0.5f - atan2f(-dy, dx));
  float width = box_size / (float)image_w;
  float height = box_size / (float)image_h;
  float long_side = width * (float)image_w > height * (float)image_h ?
      width * (float)image_w : height * (float)image_h;
  PoseRect rect = {
      det->keypoints[0][0],
      det->keypoints[0][1],
      (long_side / (float)image_w) * 1.25f,
      (long_side / (float)image_h) * 1.25f,
      rotation};
  return rect;
}

static void sample_rotated_rect_rgb(
    const uint8_t* rgb, int w, int h, const PoseRect* rect, int size, float* out) {
  memset(out, 0, (size_t)size * size * 3 * sizeof(float));
  float cs = cosf(rect->rotation);
  float sn = sinf(rect->rotation);
  float inv_size = 1.0f / (float)size;
  float u0 = 0.5f * inv_size - 0.5f;
  float dx = cs * rect->width * (float)w * inv_size;
  float dy = sn * rect->height * (float)h * inv_size;
  for (int oy = 0; oy < size; ++oy) {
    float v = ((float)oy + 0.5f) * inv_size - 0.5f;
    float x = (rect->x_center + (cs * u0 - sn * v) * rect->width) * (float)w - 0.5f;
    float y = (rect->y_center + (sn * u0 + cs * v) * rect->height) * (float)h - 0.5f;
    for (int ox = 0; ox < size; ++ox) {
      int x0 = (int)floorf(x);
      int y0 = (int)floorf(y);
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) {
        x += dx;
        y += dy;
        continue;
      }
      float wx = x - (float)x0;
      float wy = y - (float)y0;
      const uint8_t* p00 = rgb + ((size_t)y0 * w + x0) * 3;
      const uint8_t* p01 = rgb + ((size_t)y0 * w + x0 + 1) * 3;
      const uint8_t* p10 = rgb + ((size_t)(y0 + 1) * w + x0) * 3;
      const uint8_t* p11 = rgb + ((size_t)(y0 + 1) * w + x0 + 1) * 3;
      float* dst = out + ((size_t)oy * size + ox) * 3;
      for (int c = 0; c < 3; ++c) {
        float top = (float)p00[c] + ((float)p01[c] - (float)p00[c]) * wx;
        float bot = (float)p10[c] + ((float)p11[c] - (float)p10[c]) * wx;
        dst[c] = ((top + (bot - top) * wy) / 127.5f) - 1.0f;
      }
      x += dx;
      y += dy;
    }
  }
}

static void refine_landmarks_c(const float* raw, const float* heatmap, PoseLandmark out[39]) {
  for (int lm = 0; lm < 39; ++lm) {
    out[lm].x = raw[lm * 5 + 0] / 256.0f;
    out[lm].y = raw[lm * 5 + 1] / 256.0f;
    out[lm].z = raw[lm * 5 + 2] / 256.0f;
    out[lm].visibility = sigmoid_scalar(raw[lm * 5 + 3]);
    out[lm].presence = sigmoid_scalar(raw[lm * 5 + 4]);
  }
  for (int lm = 0; lm < 39; ++lm) {
    int center_col = (int)(out[lm].x * 64.0f);
    int center_row = (int)(out[lm].y * 64.0f);
    if (center_col < 0 || center_col >= 64 || center_row < 0 || center_row >= 64) continue;
    int begin_col = center_col - 3 > 0 ? center_col - 3 : 0;
    int end_col = center_col + 4 < 64 ? center_col + 4 : 64;
    int begin_row = center_row - 3 > 0 ? center_row - 3 : 0;
    int end_row = center_row + 4 < 64 ? center_row + 4 : 64;
    float total = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int row = begin_row; row < end_row; ++row) {
      for (int col = begin_col; col < end_col; ++col) {
        float weight = sigmoid_scalar(heatmap[((size_t)row * 64 + col) * 39 + lm]);
        total += weight;
        sum_x += weight * (float)col;
        sum_y += weight * (float)row;
      }
    }
    if (total > 0.0f) {
      out[lm].x = sum_x / 64.0f / total;
      out[lm].y = sum_y / 64.0f / total;
    }
  }
}

static PoseLandmark project_one_landmark_c(const PoseLandmark* lm, const PoseRect* rect) {
  float cs = cosf(rect->rotation);
  float sn = sinf(rect->rotation);
  float x = lm->x - 0.5f;
  float y = lm->y - 0.5f;
  PoseLandmark out = {
      (cs * x - sn * y) * rect->width + rect->x_center,
      (sn * x + cs * y) * rect->height + rect->y_center,
      lm->z * rect->width,
      lm->visibility,
      lm->presence};
  return out;
}

static void project_landmarks_c(const PoseLandmark internal[39], const PoseRect* rect, PoseLandmark out[33]) {
  for (int i = 0; i < 33; ++i) {
    out[i] = project_one_landmark_c(&internal[i], rect);
  }
}

static int rect_is_usable(const PoseRect* rect) {
  return isfinite(rect->x_center) && isfinite(rect->y_center) &&
      isfinite(rect->width) && isfinite(rect->height) && isfinite(rect->rotation) &&
      rect->width > 1.0e-6f && rect->height > 1.0e-6f;
}

static PoseRect rect_from_aux_landmarks_c(
    const PoseLandmark internal[39], const PoseRect* current_rect,
    int image_w, int image_h) {
  PoseLandmark center = project_one_landmark_c(&internal[33], current_rect);
  PoseLandmark scale = project_one_landmark_c(&internal[34], current_rect);
  float x0 = center.x * (float)image_w;
  float y0 = center.y * (float)image_h;
  float x1 = scale.x * (float)image_w;
  float y1 = scale.y * (float)image_h;
  float dx = x1 - x0;
  float dy = y1 - y0;
  float box_size = hypotf(dx, dy) * 2.0f;
  float rotation = normalize_radians_c((float)M_PI * 0.5f - atan2f(-dy, dx));
  float width = box_size / (float)image_w;
  float height = box_size / (float)image_h;
  float long_side = width * (float)image_w > height * (float)image_h ?
      width * (float)image_w : height * (float)image_h;
  PoseRect next = {
      center.x,
      center.y,
      (long_side / (float)image_w) * 1.25f,
      (long_side / (float)image_h) * 1.25f,
      rotation};
  return rect_is_usable(&next) ? next : *current_rect;
}

static void decode_world_c(const float* world, const PoseLandmark internal[39], const PoseRect* rect, PoseLandmark out[33]) {
  float cs = cosf(rect->rotation);
  float sn = sinf(rect->rotation);
  for (int i = 0; i < 33; ++i) {
    float x = world[i * 3 + 0];
    float y = world[i * 3 + 1];
    out[i].x = cs * x - sn * y;
    out[i].y = sn * x + cs * y;
    out[i].z = world[i * 3 + 2];
    out[i].visibility = internal[i].visibility;
    out[i].presence = internal[i].presence;
  }
}

static void write_landmark_array(FILE* f, const PoseLandmark* lm, int n, int indent) {
  const char* sp = "                                ";
  for (int i = 0; i < n; ++i) {
    fprintf(f,
        "%.*s{\"x\":%.9g,\"y\":%.9g,\"z\":%.9g,\"visibility\":%.9g,\"presence\":%.9g}%s\n",
        indent, sp, lm[i].x, lm[i].y, lm[i].z, lm[i].visibility, lm[i].presence,
        i + 1 == n ? "" : ",");
  }
}

static void write_rect_field(FILE* f, const char* name, const PoseRect* rect, const char* suffix) {
  fprintf(f, "  \"%s\":{\"x_center\":%.9g,\"y_center\":%.9g,\"width\":%.9g,\"height\":%.9g,\"rotation\":%.9g}%s\n",
      name, rect->x_center, rect->y_center, rect->width, rect->height, rect->rotation, suffix);
}

static void write_pipeline_json(
    const char* path, int pose_count, const PoseDetection* det, const PoseRect* rect,
    const PoseRect* next_rect, float pose_presence,
    const PoseLandmark* pose_lm, const PoseLandmark* world_lm) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (!pose_count) {
    fprintf(f, "{\"pose_count\":0,\"detections\":[]}\n");
    fclose(f);
    return;
  }
  fprintf(f, "{\n");
  fprintf(f, "  \"pose_count\":1,\n");
  fprintf(f, "  \"detections\":[{\"score\":%.9g,\"box\":{\"xmin\":%.9g,\"ymin\":%.9g,\"width\":%.9g,\"height\":%.9g},\"keypoints\":[",
      det->score, det->box_xmin, det->box_ymin, det->box_width, det->box_height);
  for (int k = 0; k < 4; ++k) {
    fprintf(f, "%s{\"x\":%.9g,\"y\":%.9g}", k ? "," : "", det->keypoints[k][0], det->keypoints[k][1]);
  }
  fprintf(f, "]}],\n");
  write_rect_field(f, "selected_rect", rect, ",");
  if (next_rect) write_rect_field(f, "next_rect", next_rect, ",");
  fprintf(f, "  \"pose_presence\":%.9g,\n", pose_presence);
  fprintf(f, "  \"pose_landmarks\":[[\n");
  write_landmark_array(f, pose_lm, 33, 4);
  fprintf(f, "  ]],\n");
  fprintf(f, "  \"pose_world_landmarks\":[[\n");
  write_landmark_array(f, world_lm, 33, 4);
  fprintf(f, "  ]]\n");
  fprintf(f, "}\n");
  fclose(f);
}

static int run_pipeline_rgb(
    const char* data_dir, const char* rgb_path, int image_w, int image_h,
    const char* out_json, int threads, int reps) {
  if (reps < 1) reps = 1;
  uint8_t* rgb = read_rgb24(rgb_path, image_w, image_h);
  float* detector_input = (float*)xaligned_alloc(64, (size_t)224 * 224 * 3 * sizeof(float));
  float* landmarker_input = (float*)xaligned_alloc(64, (size_t)256 * 256 * 3 * sizeof(float));
  PoseRuntime detector_rt;
  runtime_init(&detector_rt, &pose_detector_model, data_dir, threads);
  runtime_start_pool(&detector_rt);
  PoseRuntime landmarker_rt;
  runtime_init(&landmarker_rt, &pose_landmarker_model, data_dir, threads);
  runtime_start_pool(&landmarker_rt);

  PoseDetection det;
  PoseRect rect;
  PoseRect next_rect;
  PoseLandmark internal[39];
  PoseLandmark pose_lm[33];
  PoseLandmark world_lm[33];
  float presence = 0.0f;
  double det_ms = 0.0, lm_ms = 0.0, frame_ms = 0.0;
  int pose_count = 0;
  for (int rep = 0; rep < reps; ++rep) {
    double frame_t0 = now_s();
    Letterbox lb = resize_letterbox_rgb(rgb, image_w, image_h, 224, detector_input);
    memcpy(detector_rt.tensor[pose_detector_model.inputs[0]], detector_input, (size_t)224 * 224 * 3 * sizeof(float));
    double det_t0 = now_s();
    run_model(&detector_rt, 0);
    double det_t1 = now_s();
    det_ms += (det_t1 - det_t0) * 1000.0;
    if (!decode_best_detection(
            (const float*)detector_rt.tensor[441],
            (const float*)detector_rt.tensor[429],
            lb,
            &det)) {
      frame_ms += (now_s() - frame_t0) * 1000.0;
      pose_count = 0;
      continue;
    }

    rect = rect_from_detection_c(&det, image_w, image_h);
    sample_rotated_rect_rgb(rgb, image_w, image_h, &rect, 256, landmarker_input);
    memcpy(landmarker_rt.tensor[pose_landmarker_model.inputs[0]], landmarker_input, (size_t)256 * 256 * 3 * sizeof(float));
    double lm_t0 = now_s();
    run_model(&landmarker_rt, 0);
    double lm_t1 = now_s();
    lm_ms += (lm_t1 - lm_t0) * 1000.0;

    refine_landmarks_c((const float*)landmarker_rt.tensor[310], (const float*)landmarker_rt.tensor[283], internal);
    project_landmarks_c(internal, &rect, pose_lm);
    decode_world_c((const float*)landmarker_rt.tensor[312], internal, &rect, world_lm);
    next_rect = rect_from_aux_landmarks_c(internal, &rect, image_w, image_h);
    presence = ((const float*)landmarker_rt.tensor[315])[0];
    frame_ms += (now_s() - frame_t0) * 1000.0;
    pose_count = 1;
  }
  if (!pose_count) {
    write_pipeline_json(out_json, 0, NULL, NULL, NULL, 0.0f, NULL, NULL);
  } else {
    write_pipeline_json(out_json, 1, &det, &rect, &next_rect, presence, pose_lm, world_lm);
  }

  printf("pipeline_rgb pose_count=%d threads=%d reps=%d detector_avg_ms=%.3f landmarker_avg_ms=%.3f frame_avg_ms=%.3f fps=%.3f\n",
      pose_count, threads, reps, det_ms / (double)reps, lm_ms / (double)reps,
      frame_ms / (double)reps, 1000.0 * (double)reps / frame_ms);

  runtime_destroy(&landmarker_rt);
  runtime_destroy(&detector_rt);
  free(detector_input);
  free(landmarker_input);
  free(rgb);
  return 0;
}

static int run_pipeline_rgb_rect(
    const char* data_dir, const char* rgb_path, int image_w, int image_h,
    PoseRect rect, const char* out_json, int threads, int reps) {
  if (reps < 1) reps = 1;
  uint8_t* rgb = read_rgb24(rgb_path, image_w, image_h);
  float* landmarker_input = (float*)xaligned_alloc(64, (size_t)256 * 256 * 3 * sizeof(float));
  PoseRuntime landmarker_rt;
  runtime_init(&landmarker_rt, &pose_landmarker_model, data_dir, threads);
  runtime_start_pool(&landmarker_rt);

  PoseLandmark internal[39];
  PoseLandmark pose_lm[33];
  PoseLandmark world_lm[33];
  PoseRect next_rect = rect;
  float presence = 0.0f;
  double sample_ms = 0.0, lm_ms = 0.0, frame_ms = 0.0;
  for (int rep = 0; rep < reps; ++rep) {
    double frame_t0 = now_s();
    double sample_t0 = frame_t0;
    sample_rotated_rect_rgb(rgb, image_w, image_h, &rect, 256, landmarker_input);
    double sample_t1 = now_s();
    sample_ms += (sample_t1 - sample_t0) * 1000.0;
    memcpy(landmarker_rt.tensor[pose_landmarker_model.inputs[0]], landmarker_input, (size_t)256 * 256 * 3 * sizeof(float));
    double lm_t0 = now_s();
    run_model(&landmarker_rt, 0);
    double lm_t1 = now_s();
    lm_ms += (lm_t1 - lm_t0) * 1000.0;

    refine_landmarks_c((const float*)landmarker_rt.tensor[310], (const float*)landmarker_rt.tensor[283], internal);
    project_landmarks_c(internal, &rect, pose_lm);
    decode_world_c((const float*)landmarker_rt.tensor[312], internal, &rect, world_lm);
    next_rect = rect_from_aux_landmarks_c(internal, &rect, image_w, image_h);
    presence = ((const float*)landmarker_rt.tensor[315])[0];
    frame_ms += (now_s() - frame_t0) * 1000.0;
  }

  PoseDetection dummy;
  memset(&dummy, 0, sizeof(dummy));
  write_pipeline_json(out_json, 1, &dummy, &rect, &next_rect, presence, pose_lm, world_lm);
  printf("pipeline_rgb_rect pose_count=1 threads=%d reps=%d sample_avg_ms=%.3f landmarker_avg_ms=%.3f frame_avg_ms=%.3f fps=%.3f\n",
      threads, reps, sample_ms / (double)reps, lm_ms / (double)reps,
      frame_ms / (double)reps, 1000.0 * (double)reps / frame_ms);

  runtime_destroy(&landmarker_rt);
  free(landmarker_input);
  free(rgb);
  return 0;
}

static int run_pipeline_rgb_track(
    const char* data_dir, const char* rgb_path, int image_w, int image_h,
    const char* out_json, int threads, int reps) {
  if (reps < 1) reps = 1;
  uint8_t* rgb = read_rgb24(rgb_path, image_w, image_h);
  float* detector_input = (float*)xaligned_alloc(64, (size_t)224 * 224 * 3 * sizeof(float));
  float* landmarker_input = (float*)xaligned_alloc(64, (size_t)256 * 256 * 3 * sizeof(float));
  PoseRuntime detector_rt;
  runtime_init(&detector_rt, &pose_detector_model, data_dir, threads);
  runtime_start_pool(&detector_rt);
  PoseRuntime landmarker_rt;
  runtime_init(&landmarker_rt, &pose_landmarker_model, data_dir, threads);
  runtime_start_pool(&landmarker_rt);

  PoseDetection det;
  PoseRect rect;
  PoseRect used_rect;
  PoseRect next_rect;
  PoseLandmark internal[39];
  PoseLandmark pose_lm[33];
  PoseLandmark world_lm[33];
  float presence = 0.0f;
  int pose_count = 0;
  double acquire_ms = 0.0, sample_ms = 0.0, lm_ms = 0.0, frame_ms = 0.0;

  double acquire_t0 = now_s();
  Letterbox lb = resize_letterbox_rgb(rgb, image_w, image_h, 224, detector_input);
  memcpy(detector_rt.tensor[pose_detector_model.inputs[0]], detector_input, (size_t)224 * 224 * 3 * sizeof(float));
  run_model(&detector_rt, 0);
  if (!decode_best_detection(
          (const float*)detector_rt.tensor[441],
          (const float*)detector_rt.tensor[429],
          lb,
          &det)) {
    write_pipeline_json(out_json, 0, NULL, NULL, NULL, 0.0f, NULL, NULL);
    printf("pipeline_rgb_track pose_count=0 threads=%d reps=%d acquire_ms=%.3f\n",
        threads, reps, (now_s() - acquire_t0) * 1000.0);
    runtime_destroy(&landmarker_rt);
    runtime_destroy(&detector_rt);
    free(detector_input);
    free(landmarker_input);
    free(rgb);
    return 0;
  }
  acquire_ms = (now_s() - acquire_t0) * 1000.0;
  rect = rect_from_detection_c(&det, image_w, image_h);
  next_rect = rect;

  for (int rep = 0; rep < reps; ++rep) {
    double frame_t0 = now_s();
    used_rect = rect;
    double sample_t0 = frame_t0;
    sample_rotated_rect_rgb(rgb, image_w, image_h, &used_rect, 256, landmarker_input);
    double sample_t1 = now_s();
    sample_ms += (sample_t1 - sample_t0) * 1000.0;
    memcpy(landmarker_rt.tensor[pose_landmarker_model.inputs[0]], landmarker_input, (size_t)256 * 256 * 3 * sizeof(float));
    double lm_t0 = now_s();
    run_model(&landmarker_rt, 0);
    double lm_t1 = now_s();
    lm_ms += (lm_t1 - lm_t0) * 1000.0;

    refine_landmarks_c((const float*)landmarker_rt.tensor[310], (const float*)landmarker_rt.tensor[283], internal);
    project_landmarks_c(internal, &used_rect, pose_lm);
    decode_world_c((const float*)landmarker_rt.tensor[312], internal, &used_rect, world_lm);
    presence = ((const float*)landmarker_rt.tensor[315])[0];
    next_rect = rect_from_aux_landmarks_c(internal, &used_rect, image_w, image_h);
    if (presence >= 0.5f && rect_is_usable(&next_rect)) rect = next_rect;
    frame_ms += (now_s() - frame_t0) * 1000.0;
    pose_count = 1;
  }

  write_pipeline_json(out_json, pose_count, &det, &used_rect, &next_rect, presence, pose_lm, world_lm);
  printf("pipeline_rgb_track pose_count=%d threads=%d reps=%d acquire_ms=%.3f sample_avg_ms=%.3f landmarker_avg_ms=%.3f tracked_frame_avg_ms=%.3f tracked_fps=%.3f amortized_frame_ms=%.3f amortized_fps=%.3f final_presence=%.6g\n",
      pose_count, threads, reps, acquire_ms, sample_ms / (double)reps, lm_ms / (double)reps,
      frame_ms / (double)reps, 1000.0 * (double)reps / frame_ms,
      (frame_ms + acquire_ms) / (double)reps, 1000.0 * (double)reps / (frame_ms + acquire_ms), presence);

  runtime_destroy(&landmarker_rt);
  runtime_destroy(&detector_rt);
  free(detector_input);
  free(landmarker_input);
  free(rgb);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && strcmp(argv[1], "bench-op") == 0) {
    if (argc < 6) {
      fprintf(stderr,
              "usage: %s bench-op <data_dir> <detector|landmarker> <input_f32.bin> <src_op> [threads] [reps] [warmup]\n",
              argv[0]);
      return 2;
    }
    int threads = argc > 6 ? atoi(argv[6]) : 1;
    int reps = argc > 7 ? atoi(argv[7]) : 50;
    int warmup = argc > 8 ? atoi(argv[8]) : 3;
    return run_bench_op(argv[2], argv[3], argv[4], atoi(argv[5]), threads, reps, warmup);
  }
  if (argc >= 2 && strcmp(argv[1], "pipeline-rgb") == 0) {
    if (argc < 7) {
      fprintf(stderr,
              "usage: %s pipeline-rgb <data_dir> <rgb24.bin> <width> <height> <out.json> [threads] [reps]\n",
              argv[0]);
      return 2;
    }
    int threads = argc > 7 ? atoi(argv[7]) : 1;
    int reps = argc > 8 ? atoi(argv[8]) : 1;
    if (threads < 1) threads = 1;
    return run_pipeline_rgb(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), argv[6], threads, reps);
  }
  if (argc >= 2 && strcmp(argv[1], "pipeline-rgb-rect") == 0) {
    if (argc < 12) {
      fprintf(stderr,
              "usage: %s pipeline-rgb-rect <data_dir> <rgb24.bin> <width> <height> <x_center> <y_center> <rect_w> <rect_h> <rotation> <out.json> [threads] [reps]\n",
              argv[0]);
      return 2;
    }
    PoseRect rect = {
        strtof(argv[6], NULL),
        strtof(argv[7], NULL),
        strtof(argv[8], NULL),
        strtof(argv[9], NULL),
        strtof(argv[10], NULL)};
    int threads = argc > 12 ? atoi(argv[12]) : 1;
    int reps = argc > 13 ? atoi(argv[13]) : 1;
    if (threads < 1) threads = 1;
    return run_pipeline_rgb_rect(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), rect, argv[11], threads, reps);
  }
  if (argc >= 2 && strcmp(argv[1], "pipeline-rgb-track") == 0) {
    if (argc < 7) {
      fprintf(stderr,
              "usage: %s pipeline-rgb-track <data_dir> <rgb24.bin> <width> <height> <out.json> [threads] [reps]\n",
              argv[0]);
      return 2;
    }
    int threads = argc > 7 ? atoi(argv[7]) : 1;
    int reps = argc > 8 ? atoi(argv[8]) : 1;
    if (threads < 1) threads = 1;
    return run_pipeline_rgb_track(argv[2], argv[3], atoi(argv[4]), atoi(argv[5]), argv[6], threads, reps);
  }
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s <data_dir> <detector|landmarker> <input_f32.bin> <out_dir> [threads] [reps] [ref_dir]\n"
            "       %s bench-op <data_dir> <detector|landmarker> <input_f32.bin> <src_op> [threads] [reps] [warmup]\n"
            "       %s pipeline-rgb <data_dir> <rgb24.bin> <width> <height> <out.json> [threads] [reps]\n",
            argv[0],
            argv[0],
            argv[0]);
    fprintf(stderr,
            "       %s pipeline-rgb-rect <data_dir> <rgb24.bin> <width> <height> <x_center> <y_center> <rect_w> <rect_h> <rotation> <out.json> [threads] [reps]\n",
            argv[0]);
    fprintf(stderr,
            "       %s pipeline-rgb-track <data_dir> <rgb24.bin> <width> <height> <out.json> [threads] [reps]\n",
            argv[0]);
    return 2;
  }
  const char* data_dir = argv[1];
  const PoseModelDef* model = select_model(argv[2]);
  const char* input_path = argv[3];
  const char* out_dir = argv[4];
  int threads = argc > 5 ? atoi(argv[5]) : 1;
  int reps = argc > 6 ? atoi(argv[6]) : 1;
  const char* ref_dir = argc > 7 ? argv[7] : NULL;
  if (reps < 1) reps = 1;
  int profile = getenv("POSE_PROFILE") && strcmp(getenv("POSE_PROFILE"), "0") != 0;

  PoseRuntime rt;
  runtime_init(&rt, model, data_dir, threads);
  runtime_start_pool(&rt);
  int input_idx = model->inputs[0];
  read_file_exact(input_path, rt.tensor[input_idx], tensor_bytes(&model->tensors[input_idx]));

  double t0 = now_s();
  for (int i = 0; i < reps; ++i) {
    if (i) {
      read_file_exact(input_path, rt.tensor[input_idx], tensor_bytes(&model->tensors[input_idx]));
    }
    run_model(&rt, profile && reps == 1);
  }
  double t1 = now_s();
  printf("%s threads=%d reps=%d avg_ms=%.3f\n", model->name, rt.threads, reps, (t1 - t0) * 1000.0 / (double)reps);
  dump_outputs(&rt, out_dir);
  if (ref_dir) compare_outputs(&rt, ref_dir);
  runtime_destroy(&rt);
  return 0;
}
