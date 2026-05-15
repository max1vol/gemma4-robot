// Benchmark the first Pose Landmarker Lite landmarker convolution:
// input 256x256x3, pad 1, 3x3 stride 2, 24 output channels, bias, ReLU6.
// Uses actual TFLite weights exported by the companion extraction step.

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

enum {
  IN_H = 256,
  IN_W = 256,
  IN_C = 3,
  OUT_H = 128,
  OUT_W = 128,
  OUT_C = 24,
  K_H = 3,
  K_W = 3,
  K_SIZE = K_H * K_W * IN_C,
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

static void transpose_weights(const float* w_ohwi, float* w_kco) {
  for (int co = 0; co < OUT_C; ++co) {
    for (int ky = 0; ky < K_H; ++ky) {
      for (int kx = 0; kx < K_W; ++kx) {
        for (int ci = 0; ci < IN_C; ++ci) {
          int k = (ky * K_W + kx) * IN_C + ci;
          int src = ((co * K_H + ky) * K_W + kx) * IN_C + ci;
          w_kco[k * OUT_C + co] = w_ohwi[src];
        }
      }
    }
  }
}

static void conv_scalar(const float* input, const float* w_kco, const float* bias, float* out) {
  for (int oy = 0; oy < OUT_H; ++oy) {
    for (int ox = 0; ox < OUT_W; ++ox) {
      float acc[OUT_C];
      memcpy(acc, bias, sizeof(acc));
      for (int ky = 0; ky < K_H; ++ky) {
        int iy = oy * 2 + ky - 1;
        if ((unsigned)iy >= IN_H) continue;
        for (int kx = 0; kx < K_W; ++kx) {
          int ix = ox * 2 + kx - 1;
          if ((unsigned)ix >= IN_W) continue;
          for (int ci = 0; ci < IN_C; ++ci) {
            int k = (ky * K_W + kx) * IN_C + ci;
            float x = input[(iy * IN_W + ix) * IN_C + ci];
            const float* wk = w_kco + k * OUT_C;
            for (int co = 0; co < OUT_C; ++co) acc[co] += x * wk[co];
          }
        }
      }
      float* dst = out + (oy * OUT_W + ox) * OUT_C;
      for (int co = 0; co < OUT_C; ++co) dst[co] = relu6(acc[co]);
    }
  }
}

#if defined(__aarch64__)
static void conv_neon_rows(
    const float* input, const float* w_kco, const float* bias, float* out,
    int oy_begin, int oy_end) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  for (int oy = oy_begin; oy < oy_end; ++oy) {
    for (int ox = 0; ox < OUT_W; ++ox) {
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      float32x4_t acc4 = vld1q_f32(bias + 16);
      float32x4_t acc5 = vld1q_f32(bias + 20);
      for (int ky = 0; ky < K_H; ++ky) {
        int iy = oy * 2 + ky - 1;
        if ((unsigned)iy >= IN_H) continue;
        for (int kx = 0; kx < K_W; ++kx) {
          int ix = ox * 2 + kx - 1;
          if ((unsigned)ix >= IN_W) continue;
          const float* src = input + (iy * IN_W + ix) * IN_C;
          for (int ci = 0; ci < IN_C; ++ci) {
            int k = (ky * K_W + kx) * IN_C + ci;
            float x = src[ci];
            const float* wk = w_kco + k * OUT_C;
            acc0 = vfmaq_n_f32(acc0, vld1q_f32(wk + 0), x);
            acc1 = vfmaq_n_f32(acc1, vld1q_f32(wk + 4), x);
            acc2 = vfmaq_n_f32(acc2, vld1q_f32(wk + 8), x);
            acc3 = vfmaq_n_f32(acc3, vld1q_f32(wk + 12), x);
            acc4 = vfmaq_n_f32(acc4, vld1q_f32(wk + 16), x);
            acc5 = vfmaq_n_f32(acc5, vld1q_f32(wk + 20), x);
          }
        }
      }
      acc0 = vminq_f32(vmaxq_f32(acc0, zero), six);
      acc1 = vminq_f32(vmaxq_f32(acc1, zero), six);
      acc2 = vminq_f32(vmaxq_f32(acc2, zero), six);
      acc3 = vminq_f32(vmaxq_f32(acc3, zero), six);
      acc4 = vminq_f32(vmaxq_f32(acc4, zero), six);
      acc5 = vminq_f32(vmaxq_f32(acc5, zero), six);
      float* dst = out + (oy * OUT_W + ox) * OUT_C;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
      vst1q_f32(dst + 8, acc2);
      vst1q_f32(dst + 12, acc3);
      vst1q_f32(dst + 16, acc4);
      vst1q_f32(dst + 20, acc5);
    }
  }
}

static void conv_neon(const float* input, const float* w_kco, const float* bias, float* out) {
  conv_neon_rows(input, w_kco, bias, out, 0, OUT_H);
}

typedef struct {
  const float* input;
  const float* w_kco;
  const float* bias;
  float* out;
  int oy_begin;
  int oy_end;
} Worker;

static void* worker_main(void* ptr) {
  Worker* w = (Worker*)ptr;
  conv_neon_rows(w->input, w->w_kco, w->bias, w->out, w->oy_begin, w->oy_end);
  return NULL;
}

static void conv_neon_threads(
    const float* input, const float* w_kco, const float* bias, float* out, int threads) {
  if (threads <= 1) {
    conv_neon(input, w_kco, bias, out);
    return;
  }
  if (threads > 8) threads = 8;
  pthread_t tids[8];
  Worker workers[8];
  for (int t = 0; t < threads; ++t) {
    int y0 = (OUT_H * t) / threads;
    int y1 = (OUT_H * (t + 1)) / threads;
    workers[t] = (Worker){input, w_kco, bias, out, y0, y1};
    pthread_create(&tids[t], NULL, worker_main, &workers[t]);
  }
  for (int t = 0; t < threads; ++t) pthread_join(tids[t], NULL);
}
#endif

static void compare_ref(const float* out, const float* ref) {
  double mae = 0.0;
  float max_abs = 0.0f;
  size_t n = (size_t)OUT_H * OUT_W * OUT_C;
  for (size_t i = 0; i < n; ++i) {
    float d = fabsf(out[i] - ref[i]);
    mae += d;
    if (d > max_abs) max_abs = d;
  }
  printf("max_abs=%.9g mean_abs=%.9g\n", max_abs, mae / (double)n);
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 50;
  int threads = argc > 3 ? atoi(argv[3]) : 4;
  char path[512];

  float* input = aligned_alloc(64, (size_t)IN_H * IN_W * IN_C * sizeof(float));
  float* w_ohwi = aligned_alloc(64, (size_t)OUT_C * K_SIZE * sizeof(float));
  float* w_kco = aligned_alloc(64, (size_t)OUT_C * K_SIZE * sizeof(float));
  float* bias = aligned_alloc(64, OUT_C * sizeof(float));
  float* out = aligned_alloc(64, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  float* ref = aligned_alloc(64, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  if (!input || !w_ohwi || !w_kco || !bias || !out || !ref) {
    fprintf(stderr, "allocation failed\n");
    return 2;
  }

  snprintf(path, sizeof(path), "%s/input_f32.bin", dir);
  read_exact(path, input, (size_t)IN_H * IN_W * IN_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/w_ohwi_f32.bin", dir);
  read_exact(path, w_ohwi, (size_t)OUT_C * K_SIZE * sizeof(float));
  snprintf(path, sizeof(path), "%s/b_f32.bin", dir);
  read_exact(path, bias, OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/ref_out_f32.bin", dir);
  read_exact(path, ref, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  transpose_weights(w_ohwi, w_kco);

  memset(out, 0, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  double t0 = now_s();
  conv_scalar(input, w_kco, bias, out);
  double t1 = now_s();
  printf("scalar_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref(out, ref);

#if defined(__aarch64__)
  memset(out, 0, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  t0 = now_s();
  conv_neon(input, w_kco, bias, out);
  t1 = now_s();
  printf("neon_one_ms=%.3f ", (t1 - t0) * 1000.0);
  compare_ref(out, ref);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) conv_neon(input, w_kco, bias, out);
  t1 = now_s();
  double ms = (t1 - t0) * 1000.0 / (double)reps;
  double macs = (double)OUT_H * OUT_W * OUT_C * K_SIZE;
  printf("neon_reps=%d avg_ms=%.3f conv_gmac_s=%.3f checksum=%.6f\n",
         reps, ms, macs / (ms * 1e6), out[12345]);

  memset(out, 0, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  t0 = now_s();
  conv_neon_threads(input, w_kco, bias, out, threads);
  t1 = now_s();
  printf("neon_threads=%d one_ms=%.3f ", threads, (t1 - t0) * 1000.0);
  compare_ref(out, ref);

  t0 = now_s();
  for (int i = 0; i < reps; ++i) conv_neon_threads(input, w_kco, bias, out, threads);
  t1 = now_s();
  ms = (t1 - t0) * 1000.0 / (double)reps;
  printf("neon_threads=%d reps=%d avg_ms=%.3f conv_gmac_s=%.3f checksum=%.6f\n",
         threads, reps, ms, macs / (ms * 1e6), out[12345]);
#endif
  return 0;
}
