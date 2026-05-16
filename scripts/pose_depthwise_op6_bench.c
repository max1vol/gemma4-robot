// CPU/NEON benchmark for Pose Landmarker Lite op 6:
// depthwise 3x3, 128x128x24 -> 128x128x24, SAME padding, bias, ReLU6.

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
  C = 24,
  GROUPS = 6,
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

static void depthwise_scalar_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W; ++x) {
      float* dst = out + ((size_t)y * W + x) * C;
      for (int c = 0; c < C; ++c) {
        float acc = bias[c];
        for (int ky = 0; ky < 3; ++ky) {
          int iy = y + ky - 1;
          if ((unsigned)iy >= H) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int ix = x + kx - 1;
            if ((unsigned)ix >= W) continue;
            acc += input[((size_t)iy * W + ix) * C + c] * weights[(ky * 3 + kx) * C + c];
          }
        }
        dst[c] = relu6(acc);
      }
    }
  }
}

#if defined(__aarch64__)
static void depthwise_neon_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W; ++x) {
      float* dst = out + ((size_t)y * W + x) * C;
      for (int g = 0; g < GROUPS; ++g) {
        int c = g * 4;
        float32x4_t acc = vld1q_f32(bias + c);
        for (int ky = 0; ky < 3; ++ky) {
          int iy = y + ky - 1;
          if ((unsigned)iy >= H) continue;
          for (int kx = 0; kx < 3; ++kx) {
            int ix = x + kx - 1;
            if ((unsigned)ix >= W) continue;
            const float* src = input + ((size_t)iy * W + ix) * C + c;
            const float* w = weights + (ky * 3 + kx) * C + c;
            acc = vfmaq_f32(acc, vld1q_f32(src), vld1q_f32(w));
          }
        }
        acc = vminq_f32(vmaxq_f32(acc, zero), six);
        vst1q_f32(dst + c, acc);
      }
    }
  }
}

static void depthwise_neon_pixel_checked(
    const float* input, const float* weights, const float* bias, float* out,
    int y, int x) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  float* dst = out + ((size_t)y * W + x) * C;
  for (int g = 0; g < GROUPS; ++g) {
    int c = g * 4;
    float32x4_t acc = vld1q_f32(bias + c);
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y + ky - 1;
      if ((unsigned)iy >= H) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x + kx - 1;
        if ((unsigned)ix >= W) continue;
        acc = vfmaq_f32(
            acc,
            vld1q_f32(input + ((size_t)iy * W + ix) * C + c),
            vld1q_f32(weights + (ky * 3 + kx) * C + c));
      }
    }
    acc = vminq_f32(vmaxq_f32(acc, zero), six);
    vst1q_f32(dst + c, acc);
  }
}

static void depthwise_neon_rows_fast(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);

  if (y0 == 0) {
    depthwise_neon_rows(input, weights, bias, out, 0, 1);
    y0 = 1;
  }
  if (y1 == H) {
    depthwise_neon_rows(input, weights, bias, out, H - 1, H);
    y1 = H - 1;
  }
  if (y0 >= y1) return;

  for (int y = y0; y < y1; ++y) {
    depthwise_neon_pixel_checked(input, weights, bias, out, y, 0);
    depthwise_neon_pixel_checked(input, weights, bias, out, y, W - 1);
    for (int x = 1; x < W - 1; ++x) {
      float* dst = out + ((size_t)y * W + x) * C;
      const float* row0 = input + ((size_t)(y - 1) * W + x - 1) * C;
      const float* row1 = input + ((size_t)y * W + x - 1) * C;
      const float* row2 = input + ((size_t)(y + 1) * W + x - 1) * C;
      for (int g = 0; g < GROUPS; ++g) {
        int c = g * 4;
        float32x4_t acc = vld1q_f32(bias + c);
        acc = vfmaq_f32(acc, vld1q_f32(row0 + c), vld1q_f32(weights + 0 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + C + c), vld1q_f32(weights + 1 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + 2 * C + c), vld1q_f32(weights + 2 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + c), vld1q_f32(weights + 3 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + C + c), vld1q_f32(weights + 4 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + 2 * C + c), vld1q_f32(weights + 5 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + c), vld1q_f32(weights + 6 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + C + c), vld1q_f32(weights + 7 * C + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + 2 * C + c), vld1q_f32(weights + 8 * C + c));
        acc = vminq_f32(vmaxq_f32(acc, zero), six);
        vst1q_f32(dst + c, acc);
      }
    }
  }
}

typedef struct {
  const float* input;
  const float* weights;
  const float* bias;
  float* out;
  int y0;
  int y1;
} Worker;

static void* worker_main(void* ptr) {
  Worker* w = (Worker*)ptr;
  depthwise_neon_rows_fast(w->input, w->weights, w->bias, w->out, w->y0, w->y1);
  return NULL;
}

static void depthwise_neon_threads(
    const float* input, const float* weights, const float* bias, float* out, int threads) {
  if (threads <= 1) {
    depthwise_neon_rows_fast(input, weights, bias, out, 0, H);
    return;
  }
  if (threads > 8) threads = 8;
  pthread_t tids[8];
  Worker workers[8];
  for (int t = 0; t < threads; ++t) {
    workers[t] = (Worker){
        input, weights, bias, out,
        (H * t) / threads,
        (H * (t + 1)) / threads};
    pthread_create(&tids[t], NULL, worker_main, &workers[t]);
  }
  for (int t = 0; t < threads; ++t) pthread_join(tids[t], NULL);
}
#endif

static void compare_ref(const float* out, const float* ref) {
  double mae = 0.0;
  float max_abs = 0.0f;
  size_t n = (size_t)H * W * C;
  for (size_t i = 0; i < n; ++i) {
    float d = fabsf(out[i] - ref[i]);
    mae += d;
    if (d > max_abs) max_abs = d;
  }
  printf("max_abs=%.9g mean_abs=%.9g\n", max_abs, mae / (double)n);
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 200;
  int threads = argc > 3 ? atoi(argv[3]) : 4;
  char path[512];

  float* input = aligned_alloc(64, (size_t)H * W * C * sizeof(float));
  float* weights = aligned_alloc(64, 3 * 3 * C * sizeof(float));
  float* bias = aligned_alloc(64, C * sizeof(float));
  float* out = aligned_alloc(64, (size_t)H * W * C * sizeof(float));
  float* ref = aligned_alloc(64, (size_t)H * W * C * sizeof(float));
  if (!input || !weights || !bias || !out || !ref) return 2;

  snprintf(path, sizeof(path), "%s/op3_ref_f32.bin", dir);
  read_exact(path, input, (size_t)H * W * C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op6_w_1hwc_f32.bin", dir);
  read_exact(path, weights, 3 * 3 * C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op6_b_f32.bin", dir);
  read_exact(path, bias, C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op6_ref_f32.bin", dir);
  read_exact(path, ref, (size_t)H * W * C * sizeof(float));

  double t0 = now_s();
  depthwise_scalar_rows(input, weights, bias, out, 0, H);
  double t1 = now_s();
  printf("scalar_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref(out, ref);

#if defined(__aarch64__)
  t0 = now_s();
  depthwise_neon_rows(input, weights, bias, out, 0, H);
  t1 = now_s();
  printf("neon_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref(out, ref);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) depthwise_neon_rows(input, weights, bias, out, 0, H);
  t1 = now_s();
  double macs = (double)H * W * C * 9.0;
  printf("neon_reps=%d avg_ms=%.3f depthwise_gmac_s=%.3f checksum=%.6f\n",
      reps, (t1 - t0) * 1000.0 / reps, macs * reps / (t1 - t0) / 1e9, out[12345]);

  t0 = now_s();
  depthwise_neon_threads(input, weights, bias, out, threads);
  t1 = now_s();
  printf("neon_threads=%d one_ms=%.3f ", threads, (t1 - t0) * 1000.0);
  compare_ref(out, ref);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) depthwise_neon_threads(input, weights, bias, out, threads);
  t1 = now_s();
  printf("neon_threads=%d reps=%d avg_ms=%.3f depthwise_gmac_s=%.3f checksum=%.6f\n",
      threads, reps, (t1 - t0) * 1000.0 / reps, macs * reps / (t1 - t0) / 1e9, out[12345]);
#else
  (void)reps;
  (void)threads;
#endif

  return 0;
}
