// CPU/NEON benchmarks for two high-resolution Pose Landmarker Lite pointwise ops:
// op 9:  128x128x24 -> 128x128x8,  1x1, activation NONE
// op 12: 128x128x8  -> 128x128x32, 1x1, activation ReLU6

#define _POSIX_C_SOURCE 200809L

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

enum {
  H = 128,
  W = 128,
};

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void read_exact(const char* path, void* ptr, size_t bytes) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  if (fread(ptr, 1, bytes, f) != bytes) {
    fprintf(stderr, "short read: %s\n", path);
    exit(2);
  }
  fclose(f);
}

static float relu6(float x) {
  if (x < 0.0f) return 0.0f;
  if (x > 6.0f) return 6.0f;
  return x;
}

static void transpose_ohwi_1x1(const float* w_ohwi, float* w_icoc, int ic, int oc) {
  for (int i = 0; i < ic; ++i) {
    for (int o = 0; o < oc; ++o) {
      w_icoc[i * oc + o] = w_ohwi[o * ic + i];
    }
  }
}

static void compare_ref(const char* label, const float* out, const float* ref, size_t n) {
  double mae = 0.0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float d = fabsf(out[i] - ref[i]);
    mae += d;
    if (d > max_abs) max_abs = d;
  }
  printf("%s max_abs=%.9g mean_abs=%.9g\n", label, max_abs, mae / (double)n);
}

static void op9_scalar_rows(
    const float* input, const float* w_icoc, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W; ++x) {
      const float* src = input + ((size_t)y * W + x) * 24;
      float* dst = out + ((size_t)y * W + x) * 8;
      float acc[8];
      memcpy(acc, bias, sizeof(acc));
      for (int ci = 0; ci < 24; ++ci) {
        float v = src[ci];
        const float* w = w_icoc + ci * 8;
        for (int co = 0; co < 8; ++co) acc[co] += v * w[co];
      }
      memcpy(dst, acc, sizeof(acc));
    }
  }
}

static void op12_scalar_rows(
    const float* input, const float* w_icoc, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W; ++x) {
      const float* src = input + ((size_t)y * W + x) * 8;
      float* dst = out + ((size_t)y * W + x) * 32;
      for (int co = 0; co < 32; ++co) dst[co] = bias[co];
      for (int ci = 0; ci < 8; ++ci) {
        float v = src[ci];
        const float* w = w_icoc + ci * 32;
        for (int co = 0; co < 32; ++co) dst[co] += v * w[co];
      }
      for (int co = 0; co < 32; ++co) dst[co] = relu6(dst[co]);
    }
  }
}

#if defined(__aarch64__)
static void op9_neon_rows(
    const float* input, const float* w_icoc, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W; ++x) {
      const float* src = input + ((size_t)y * W + x) * 24;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      for (int ci = 0; ci < 24; ++ci) {
        float v = src[ci];
        const float* w = w_icoc + ci * 8;
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(w + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(w + 4), v);
      }
      float* dst = out + ((size_t)y * W + x) * 8;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
    }
  }
}

static void op12_neon_rows(
    const float* input, const float* w_icoc, const float* bias, float* out,
    int y0, int y1) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W; ++x) {
      const float* src = input + ((size_t)y * W + x) * 8;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      float32x4_t acc4 = vld1q_f32(bias + 16);
      float32x4_t acc5 = vld1q_f32(bias + 20);
      float32x4_t acc6 = vld1q_f32(bias + 24);
      float32x4_t acc7 = vld1q_f32(bias + 28);
      for (int ci = 0; ci < 8; ++ci) {
        float v = src[ci];
        const float* w = w_icoc + ci * 32;
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(w + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(w + 4), v);
        acc2 = vfmaq_n_f32(acc2, vld1q_f32(w + 8), v);
        acc3 = vfmaq_n_f32(acc3, vld1q_f32(w + 12), v);
        acc4 = vfmaq_n_f32(acc4, vld1q_f32(w + 16), v);
        acc5 = vfmaq_n_f32(acc5, vld1q_f32(w + 20), v);
        acc6 = vfmaq_n_f32(acc6, vld1q_f32(w + 24), v);
        acc7 = vfmaq_n_f32(acc7, vld1q_f32(w + 28), v);
      }
      acc0 = vminq_f32(vmaxq_f32(acc0, zero), six);
      acc1 = vminq_f32(vmaxq_f32(acc1, zero), six);
      acc2 = vminq_f32(vmaxq_f32(acc2, zero), six);
      acc3 = vminq_f32(vmaxq_f32(acc3, zero), six);
      acc4 = vminq_f32(vmaxq_f32(acc4, zero), six);
      acc5 = vminq_f32(vmaxq_f32(acc5, zero), six);
      acc6 = vminq_f32(vmaxq_f32(acc6, zero), six);
      acc7 = vminq_f32(vmaxq_f32(acc7, zero), six);
      float* dst = out + ((size_t)y * W + x) * 32;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
      vst1q_f32(dst + 8, acc2);
      vst1q_f32(dst + 12, acc3);
      vst1q_f32(dst + 16, acc4);
      vst1q_f32(dst + 20, acc5);
      vst1q_f32(dst + 24, acc6);
      vst1q_f32(dst + 28, acc7);
    }
  }
}

typedef struct {
  int op;
  const float* input;
  const float* weights;
  const float* bias;
  float* out;
  int y0;
  int y1;
} Worker;

static void* worker_main(void* ptr) {
  Worker* w = (Worker*)ptr;
  if (w->op == 9) op9_neon_rows(w->input, w->weights, w->bias, w->out, w->y0, w->y1);
  else op12_neon_rows(w->input, w->weights, w->bias, w->out, w->y0, w->y1);
  return NULL;
}

static void run_threads(
    int op, const float* input, const float* weights, const float* bias,
    float* out, int threads) {
  if (threads <= 1) {
    if (op == 9) op9_neon_rows(input, weights, bias, out, 0, H);
    else op12_neon_rows(input, weights, bias, out, 0, H);
    return;
  }
  if (threads > 8) threads = 8;
  pthread_t tids[8];
  Worker workers[8];
  for (int t = 0; t < threads; ++t) {
    workers[t] = (Worker){
        op, input, weights, bias, out,
        (H * t) / threads,
        (H * (t + 1)) / threads};
    pthread_create(&tids[t], NULL, worker_main, &workers[t]);
  }
  for (int t = 0; t < threads; ++t) pthread_join(tids[t], NULL);
}
#endif

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 500;
  int threads = argc > 3 ? atoi(argv[3]) : 4;
  char path[512];

  float* op6 = aligned_alloc(64, (size_t)H * W * 24 * sizeof(float));
  float* op9_ref = aligned_alloc(64, (size_t)H * W * 8 * sizeof(float));
  float* op9_out = aligned_alloc(64, (size_t)H * W * 8 * sizeof(float));
  float* op9_w_ohwi = aligned_alloc(64, 8 * 24 * sizeof(float));
  float* op9_w = aligned_alloc(64, 8 * 24 * sizeof(float));
  float* op9_b = malloc(8 * sizeof(float));
  float* op12_ref = aligned_alloc(64, (size_t)H * W * 32 * sizeof(float));
  float* op12_out = aligned_alloc(64, (size_t)H * W * 32 * sizeof(float));
  float* op12_w_ohwi = aligned_alloc(64, 32 * 8 * sizeof(float));
  float* op12_w = aligned_alloc(64, 32 * 8 * sizeof(float));
  float* op12_b = aligned_alloc(64, 32 * sizeof(float));
  if (!op6 || !op9_ref || !op9_out || !op9_w_ohwi || !op9_w || !op9_b ||
      !op12_ref || !op12_out || !op12_w_ohwi || !op12_w || !op12_b) {
    return 2;
  }

  snprintf(path, sizeof(path), "%s/op6_ref_f32.bin", dir);
  read_exact(path, op6, (size_t)H * W * 24 * sizeof(float));
  snprintf(path, sizeof(path), "%s/op9_ref_f32.bin", dir);
  read_exact(path, op9_ref, (size_t)H * W * 8 * sizeof(float));
  snprintf(path, sizeof(path), "%s/op9_w_ohwi_f32.bin", dir);
  read_exact(path, op9_w_ohwi, 8 * 24 * sizeof(float));
  snprintf(path, sizeof(path), "%s/op9_b_f32.bin", dir);
  read_exact(path, op9_b, 8 * sizeof(float));
  transpose_ohwi_1x1(op9_w_ohwi, op9_w, 24, 8);

  snprintf(path, sizeof(path), "%s/op12_ref_f32.bin", dir);
  read_exact(path, op12_ref, (size_t)H * W * 32 * sizeof(float));
  snprintf(path, sizeof(path), "%s/op12_w_ohwi_f32.bin", dir);
  read_exact(path, op12_w_ohwi, 32 * 8 * sizeof(float));
  snprintf(path, sizeof(path), "%s/op12_b_f32.bin", dir);
  read_exact(path, op12_b, 32 * sizeof(float));
  transpose_ohwi_1x1(op12_w_ohwi, op12_w, 8, 32);

  double t0 = now_s();
  op9_scalar_rows(op6, op9_w, op9_b, op9_out, 0, H);
  double t1 = now_s();
  printf("op9_scalar_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref("op9_scalar", op9_out, op9_ref, (size_t)H * W * 8);

  t0 = now_s();
  op12_scalar_rows(op9_ref, op12_w, op12_b, op12_out, 0, H);
  t1 = now_s();
  printf("op12_scalar_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref("op12_scalar", op12_out, op12_ref, (size_t)H * W * 32);

#if defined(__aarch64__)
  t0 = now_s();
  op9_neon_rows(op6, op9_w, op9_b, op9_out, 0, H);
  t1 = now_s();
  printf("op9_neon_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref("op9_neon", op9_out, op9_ref, (size_t)H * W * 8);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) op9_neon_rows(op6, op9_w, op9_b, op9_out, 0, H);
  t1 = now_s();
  double op9_macs = (double)H * W * 24 * 8;
  printf("op9_neon_reps=%d avg_ms=%.3f gmac_s=%.3f checksum=%.6f\n",
      reps, (t1 - t0) * 1000.0 / reps, op9_macs * reps / (t1 - t0) / 1e9, op9_out[12345]);

  t0 = now_s();
  run_threads(9, op6, op9_w, op9_b, op9_out, threads);
  t1 = now_s();
  printf("op9_neon_threads=%d one_ms=%.3f ", threads, (t1 - t0) * 1000.0);
  compare_ref("op9_neon_threads", op9_out, op9_ref, (size_t)H * W * 8);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) run_threads(9, op6, op9_w, op9_b, op9_out, threads);
  t1 = now_s();
  printf("op9_neon_threads=%d reps=%d avg_ms=%.3f gmac_s=%.3f checksum=%.6f\n",
      threads, reps, (t1 - t0) * 1000.0 / reps, op9_macs * reps / (t1 - t0) / 1e9, op9_out[12345]);

  t0 = now_s();
  op12_neon_rows(op9_ref, op12_w, op12_b, op12_out, 0, H);
  t1 = now_s();
  printf("op12_neon_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref("op12_neon", op12_out, op12_ref, (size_t)H * W * 32);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) op12_neon_rows(op9_ref, op12_w, op12_b, op12_out, 0, H);
  t1 = now_s();
  double op12_macs = (double)H * W * 8 * 32;
  printf("op12_neon_reps=%d avg_ms=%.3f gmac_s=%.3f checksum=%.6f\n",
      reps, (t1 - t0) * 1000.0 / reps, op12_macs * reps / (t1 - t0) / 1e9, op12_out[12345]);

  t0 = now_s();
  run_threads(12, op9_ref, op12_w, op12_b, op12_out, threads);
  t1 = now_s();
  printf("op12_neon_threads=%d one_ms=%.3f ", threads, (t1 - t0) * 1000.0);
  compare_ref("op12_neon_threads", op12_out, op12_ref, (size_t)H * W * 32);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) run_threads(12, op9_ref, op12_w, op12_b, op12_out, threads);
  t1 = now_s();
  printf("op12_neon_threads=%d reps=%d avg_ms=%.3f gmac_s=%.3f checksum=%.6f\n",
      threads, reps, (t1 - t0) * 1000.0 / reps, op12_macs * reps / (t1 - t0) / 1e9, op12_out[12345]);
#else
  (void)reps;
  (void)threads;
#endif

  return 0;
}
