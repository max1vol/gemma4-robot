// Persistent 4-core NEON prefix benchmark for Pose Landmarker Lite.
//
// Runs the first high-resolution prefix with a fixed pthread team:
//   op3  256x256x3  -> 128x128x24 conv 3x3 stride 2 ReLU6
//   op6  128x128x24 -> 128x128x24 depthwise 3x3 ReLU6
//   op9  128x128x24 -> 128x128x8  pointwise 1x1
//   op12 128x128x8  -> 128x128x32 pointwise 1x1 ReLU6
//
// The goal is to measure a realistic no-GPU NEON path without per-op pthread
// creation overhead. Threads synchronize between dependent ops with barriers.

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
  IN_H = 256,
  IN_W = 256,
  OP_H = 128,
  OP_W = 128,
  C3 = 24,
  C9 = 8,
  C12 = 32,
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

static void transpose_conv3(const float* w_ohwi, float* w_kco) {
  for (int co = 0; co < C3; ++co) {
    for (int ky = 0; ky < 3; ++ky) {
      for (int kx = 0; kx < 3; ++kx) {
        for (int ci = 0; ci < 3; ++ci) {
          int k = (ky * 3 + kx) * 3 + ci;
          w_kco[k * C3 + co] = w_ohwi[((co * 3 + ky) * 3 + kx) * 3 + ci];
        }
      }
    }
  }
}

static void transpose_1x1(const float* w_ohwi, float* w_icoc, int ic, int oc) {
  for (int i = 0; i < ic; ++i) {
    for (int o = 0; o < oc; ++o) {
      w_icoc[i * oc + o] = w_ohwi[o * ic + i];
    }
  }
}

#if defined(__aarch64__)
static void op3_pixel_checked(
    const float* input, const float* w_kco, const float* bias, float* out,
    int oy, int ox) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  float32x4_t acc0 = vld1q_f32(bias + 0);
  float32x4_t acc1 = vld1q_f32(bias + 4);
  float32x4_t acc2 = vld1q_f32(bias + 8);
  float32x4_t acc3 = vld1q_f32(bias + 12);
  float32x4_t acc4 = vld1q_f32(bias + 16);
  float32x4_t acc5 = vld1q_f32(bias + 20);
  for (int ky = 0; ky < 3; ++ky) {
    int iy = oy * 2 + ky - 1;
    if ((unsigned)iy >= IN_H) continue;
    for (int kx = 0; kx < 3; ++kx) {
      int ix = ox * 2 + kx - 1;
      if ((unsigned)ix >= IN_W) continue;
      const float* src = input + ((size_t)iy * IN_W + ix) * 3;
      for (int ci = 0; ci < 3; ++ci) {
        const float* wk = w_kco + ((ky * 3 + kx) * 3 + ci) * C3;
        float v = src[ci];
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(wk + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(wk + 4), v);
        acc2 = vfmaq_n_f32(acc2, vld1q_f32(wk + 8), v);
        acc3 = vfmaq_n_f32(acc3, vld1q_f32(wk + 12), v);
        acc4 = vfmaq_n_f32(acc4, vld1q_f32(wk + 16), v);
        acc5 = vfmaq_n_f32(acc5, vld1q_f32(wk + 20), v);
      }
    }
  }
  acc0 = vminq_f32(vmaxq_f32(acc0, zero), six);
  acc1 = vminq_f32(vmaxq_f32(acc1, zero), six);
  acc2 = vminq_f32(vmaxq_f32(acc2, zero), six);
  acc3 = vminq_f32(vmaxq_f32(acc3, zero), six);
  acc4 = vminq_f32(vmaxq_f32(acc4, zero), six);
  acc5 = vminq_f32(vmaxq_f32(acc5, zero), six);
  float* dst = out + ((size_t)oy * OP_W + ox) * C3;
  vst1q_f32(dst + 0, acc0);
  vst1q_f32(dst + 4, acc1);
  vst1q_f32(dst + 8, acc2);
  vst1q_f32(dst + 12, acc3);
  vst1q_f32(dst + 16, acc4);
  vst1q_f32(dst + 20, acc5);
}

static void op3_rows(
    const float* input, const float* w_kco, const float* bias, float* out,
    int y0, int y1) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  if (y0 == 0) {
    for (int ox = 0; ox < OP_W; ++ox) op3_pixel_checked(input, w_kco, bias, out, 0, ox);
    y0 = 1;
  }
  for (int oy = y0; oy < y1; ++oy) {
    op3_pixel_checked(input, w_kco, bias, out, oy, 0);
    for (int ox = 1; ox < OP_W; ++ox) {
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      float32x4_t acc4 = vld1q_f32(bias + 16);
      float32x4_t acc5 = vld1q_f32(bias + 20);
      for (int ky = 0; ky < 3; ++ky) {
        int iy = oy * 2 + ky - 1;
        for (int kx = 0; kx < 3; ++kx) {
          int ix = ox * 2 + kx - 1;
          const float* src = input + ((size_t)iy * IN_W + ix) * 3;
          for (int ci = 0; ci < 3; ++ci) {
            const float* wk = w_kco + ((ky * 3 + kx) * 3 + ci) * C3;
            float v = src[ci];
            acc0 = vfmaq_n_f32(acc0, vld1q_f32(wk + 0), v);
            acc1 = vfmaq_n_f32(acc1, vld1q_f32(wk + 4), v);
            acc2 = vfmaq_n_f32(acc2, vld1q_f32(wk + 8), v);
            acc3 = vfmaq_n_f32(acc3, vld1q_f32(wk + 12), v);
            acc4 = vfmaq_n_f32(acc4, vld1q_f32(wk + 16), v);
            acc5 = vfmaq_n_f32(acc5, vld1q_f32(wk + 20), v);
          }
        }
      }
      acc0 = vminq_f32(vmaxq_f32(acc0, zero), six);
      acc1 = vminq_f32(vmaxq_f32(acc1, zero), six);
      acc2 = vminq_f32(vmaxq_f32(acc2, zero), six);
      acc3 = vminq_f32(vmaxq_f32(acc3, zero), six);
      acc4 = vminq_f32(vmaxq_f32(acc4, zero), six);
      acc5 = vminq_f32(vmaxq_f32(acc5, zero), six);
      float* dst = out + ((size_t)oy * OP_W + ox) * C3;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
      vst1q_f32(dst + 8, acc2);
      vst1q_f32(dst + 12, acc3);
      vst1q_f32(dst + 16, acc4);
      vst1q_f32(dst + 20, acc5);
    }
  }
}

static void op6_pixel_checked(
    const float* input, const float* weights, const float* bias, float* out,
    int y, int x) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  float* dst = out + ((size_t)y * OP_W + x) * C3;
  for (int g = 0; g < 6; ++g) {
    int c = g * 4;
    float32x4_t acc = vld1q_f32(bias + c);
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y + ky - 1;
      if ((unsigned)iy >= OP_H) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x + kx - 1;
        if ((unsigned)ix >= OP_W) continue;
        acc = vfmaq_f32(
            acc,
            vld1q_f32(input + ((size_t)iy * OP_W + ix) * C3 + c),
            vld1q_f32(weights + (ky * 3 + kx) * C3 + c));
      }
    }
    acc = vminq_f32(vmaxq_f32(acc, zero), six);
    vst1q_f32(dst + c, acc);
  }
}

static void op6_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  if (y0 == 0) {
    for (int x = 0; x < OP_W; ++x) op6_pixel_checked(input, weights, bias, out, 0, x);
    y0 = 1;
  }
  if (y1 == OP_H) {
    for (int x = 0; x < OP_W; ++x) op6_pixel_checked(input, weights, bias, out, OP_H - 1, x);
    y1 = OP_H - 1;
  }
  for (int y = y0; y < y1; ++y) {
    op6_pixel_checked(input, weights, bias, out, y, 0);
    op6_pixel_checked(input, weights, bias, out, y, OP_W - 1);
    for (int x = 1; x < OP_W - 1; ++x) {
      float* dst = out + ((size_t)y * OP_W + x) * C3;
      const float* row0 = input + ((size_t)(y - 1) * OP_W + x - 1) * C3;
      const float* row1 = input + ((size_t)y * OP_W + x - 1) * C3;
      const float* row2 = input + ((size_t)(y + 1) * OP_W + x - 1) * C3;
      for (int g = 0; g < 6; ++g) {
        int c = g * 4;
        float32x4_t acc = vld1q_f32(bias + c);
        acc = vfmaq_f32(acc, vld1q_f32(row0 + c), vld1q_f32(weights + 0 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + C3 + c), vld1q_f32(weights + 1 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + 2 * C3 + c), vld1q_f32(weights + 2 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + c), vld1q_f32(weights + 3 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + C3 + c), vld1q_f32(weights + 4 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + 2 * C3 + c), vld1q_f32(weights + 5 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + c), vld1q_f32(weights + 6 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + C3 + c), vld1q_f32(weights + 7 * C3 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + 2 * C3 + c), vld1q_f32(weights + 8 * C3 + c));
        acc = vminq_f32(vmaxq_f32(acc, zero), six);
        vst1q_f32(dst + c, acc);
      }
    }
  }
}

static void op9_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < OP_W; ++x) {
      const float* src = input + ((size_t)y * OP_W + x) * C3;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      for (int ci = 0; ci < C3; ++ci) {
        float v = src[ci];
        const float* w = weights + ci * C9;
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(w + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(w + 4), v);
      }
      float* dst = out + ((size_t)y * OP_W + x) * C9;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
    }
  }
}

static void op12_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < OP_W; ++x) {
      const float* src = input + ((size_t)y * OP_W + x) * C9;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      float32x4_t acc4 = vld1q_f32(bias + 16);
      float32x4_t acc5 = vld1q_f32(bias + 20);
      float32x4_t acc6 = vld1q_f32(bias + 24);
      float32x4_t acc7 = vld1q_f32(bias + 28);
      for (int ci = 0; ci < C9; ++ci) {
        float v = src[ci];
        const float* w = weights + ci * C12;
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
      float* dst = out + ((size_t)y * OP_W + x) * C12;
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
  int id;
  int threads;
  int reps;
  pthread_barrier_t* barrier;
  const float* input;
  const float* op3_w;
  const float* op3_b;
  const float* op6_w;
  const float* op6_b;
  const float* op9_w;
  const float* op9_b;
  const float* op12_w;
  const float* op12_b;
  float* op3_out;
  float* op6_out;
  float* op9_out;
  float* op12_out;
} Worker;

static void* worker_main(void* ptr) {
  Worker* w = (Worker*)ptr;
  int y0 = (OP_H * w->id) / w->threads;
  int y1 = (OP_H * (w->id + 1)) / w->threads;
  for (int r = 0; r < w->reps; ++r) {
    op3_rows(w->input, w->op3_w, w->op3_b, w->op3_out, y0, y1);
    pthread_barrier_wait(w->barrier);
    op6_rows(w->op3_out, w->op6_w, w->op6_b, w->op6_out, y0, y1);
    pthread_barrier_wait(w->barrier);
    op9_rows(w->op6_out, w->op9_w, w->op9_b, w->op9_out, y0, y1);
    pthread_barrier_wait(w->barrier);
    op12_rows(w->op9_out, w->op12_w, w->op12_b, w->op12_out, y0, y1);
    pthread_barrier_wait(w->barrier);
  }
  return NULL;
}
#endif

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 100;
  int threads = argc > 3 ? atoi(argv[3]) : 4;
  if (threads < 1) threads = 1;
  if (threads > 8) threads = 8;
  char path[512];

  float* input = aligned_alloc(64, (size_t)IN_H * IN_W * 3 * sizeof(float));
  float* op3_w_ohwi = aligned_alloc(64, C3 * 3 * 3 * 3 * sizeof(float));
  float* op3_w = aligned_alloc(64, C3 * 3 * 3 * 3 * sizeof(float));
  float* op3_b = malloc(C3 * sizeof(float));
  float* op6_w = aligned_alloc(64, 3 * 3 * C3 * sizeof(float));
  float* op6_b = malloc(C3 * sizeof(float));
  float* op9_w_ohwi = aligned_alloc(64, C9 * C3 * sizeof(float));
  float* op9_w = aligned_alloc(64, C9 * C3 * sizeof(float));
  float* op9_b = malloc(C9 * sizeof(float));
  float* op12_w_ohwi = aligned_alloc(64, C12 * C9 * sizeof(float));
  float* op12_w = aligned_alloc(64, C12 * C9 * sizeof(float));
  float* op12_b = aligned_alloc(64, C12 * sizeof(float));
  float* op3 = aligned_alloc(64, (size_t)OP_H * OP_W * C3 * sizeof(float));
  float* op6 = aligned_alloc(64, (size_t)OP_H * OP_W * C3 * sizeof(float));
  float* op9 = aligned_alloc(64, (size_t)OP_H * OP_W * C9 * sizeof(float));
  float* op12 = aligned_alloc(64, (size_t)OP_H * OP_W * C12 * sizeof(float));
  float* ref3 = aligned_alloc(64, (size_t)OP_H * OP_W * C3 * sizeof(float));
  float* ref6 = aligned_alloc(64, (size_t)OP_H * OP_W * C3 * sizeof(float));
  float* ref9 = aligned_alloc(64, (size_t)OP_H * OP_W * C9 * sizeof(float));
  float* ref12 = aligned_alloc(64, (size_t)OP_H * OP_W * C12 * sizeof(float));
  if (!input || !op3_w_ohwi || !op3_w || !op3_b || !op6_w || !op6_b ||
      !op9_w_ohwi || !op9_w || !op9_b || !op12_w_ohwi || !op12_w || !op12_b ||
      !op3 || !op6 || !op9 || !op12 || !ref3 || !ref6 || !ref9 || !ref12) {
    fprintf(stderr, "allocation failed\n");
    return 2;
  }

#define LOAD_FILE(name, ptr, bytes) do { \
  snprintf(path, sizeof(path), "%s/%s", dir, name); \
  read_exact(path, ptr, bytes); \
} while (0)

  LOAD_FILE("input_f32.bin", input, (size_t)IN_H * IN_W * 3 * sizeof(float));
  LOAD_FILE("op3_w_ohwi_f32.bin", op3_w_ohwi, C3 * 3 * 3 * 3 * sizeof(float));
  LOAD_FILE("op3_b_f32.bin", op3_b, C3 * sizeof(float));
  LOAD_FILE("op6_w_1hwc_f32.bin", op6_w, 3 * 3 * C3 * sizeof(float));
  LOAD_FILE("op6_b_f32.bin", op6_b, C3 * sizeof(float));
  LOAD_FILE("op9_w_ohwi_f32.bin", op9_w_ohwi, C9 * C3 * sizeof(float));
  LOAD_FILE("op9_b_f32.bin", op9_b, C9 * sizeof(float));
  LOAD_FILE("op12_w_ohwi_f32.bin", op12_w_ohwi, C12 * C9 * sizeof(float));
  LOAD_FILE("op12_b_f32.bin", op12_b, C12 * sizeof(float));
  LOAD_FILE("op3_ref_f32.bin", ref3, (size_t)OP_H * OP_W * C3 * sizeof(float));
  LOAD_FILE("op6_ref_f32.bin", ref6, (size_t)OP_H * OP_W * C3 * sizeof(float));
  LOAD_FILE("op9_ref_f32.bin", ref9, (size_t)OP_H * OP_W * C9 * sizeof(float));
  LOAD_FILE("op12_ref_f32.bin", ref12, (size_t)OP_H * OP_W * C12 * sizeof(float));

  transpose_conv3(op3_w_ohwi, op3_w);
  transpose_1x1(op9_w_ohwi, op9_w, C3, C9);
  transpose_1x1(op12_w_ohwi, op12_w, C9, C12);

#if defined(__aarch64__)
  pthread_t tids[8];
  Worker workers[8];
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, (unsigned)threads);

  double t0 = now_s();
  for (int t = 0; t < threads; ++t) {
    workers[t] = (Worker){
        t, threads, reps, &barrier,
        input, op3_w, op3_b, op6_w, op6_b, op9_w, op9_b, op12_w, op12_b,
        op3, op6, op9, op12};
    pthread_create(&tids[t], NULL, worker_main, &workers[t]);
  }
  for (int t = 0; t < threads; ++t) pthread_join(tids[t], NULL);
  double t1 = now_s();
  pthread_barrier_destroy(&barrier);

  const double macs =
      (double)OP_H * OP_W * C3 * 27.0 +
      (double)OP_H * OP_W * C3 * 9.0 +
      (double)OP_H * OP_W * C3 * C9 +
      (double)OP_H * OP_W * C9 * C12;
  printf("neon_prefix threads=%d reps=%d avg_ms=%.3f fps=%.2f gmac_s=%.3f macs=%.0f\n",
      threads, reps, (t1 - t0) * 1000.0 / reps, reps / (t1 - t0),
      macs * reps / (t1 - t0) / 1e9, macs);
  compare_ref("op3", op3, ref3, (size_t)OP_H * OP_W * C3);
  compare_ref("op6", op6, ref6, (size_t)OP_H * OP_W * C3);
  compare_ref("op9", op9, ref9, (size_t)OP_H * OP_W * C9);
  compare_ref("op12", op12, ref12, (size_t)OP_H * OP_W * C12);
  printf("checksums %.6f %.6f %.6f %.6f\n", op3[12345], op6[12345], op9[12345], op12[12345]);
#else
  (void)reps;
  (void)threads;
  fprintf(stderr, "NEON prefix benchmark requires aarch64\n");
  return 2;
#endif

  return 0;
}
