// Persistent-thread NEON benchmark for the next Pose Landmarker Lite block.
//
// Inputs are the extracted float outputs from the previous prefix:
//   op9_ref_f32  : 128x128x8
//   op12_ref_f32 : 128x128x32
//
// Computes:
//   op15: 128x128x8  -> 128x128x8  depthwise 3x3 ReLU6
//   op18: 128x128x8  -> 128x128x8  pointwise 1x1
//   op19: residual add op9 + op18, ReLU6
//   op23: 128x128x32 -> 64x64x32 depthwise 3x3 stride 2 ReLU6
//   op26: 64x64x32   -> 64x64x16 pointwise 1x1

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
  H128 = 128,
  W128 = 128,
  H64 = 64,
  W64 = 64,
  C8 = 8,
  C16 = 16,
  C32 = 32,
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

static void transpose_1x1(const float* w_ohwi, float* w_icoc, int ic, int oc) {
  for (int i = 0; i < ic; ++i) {
    for (int o = 0; o < oc; ++o) {
      w_icoc[i * oc + o] = w_ohwi[o * ic + i];
    }
  }
}

#if defined(__aarch64__)
static inline float32x4_t relu6q(float32x4_t v) {
  const float32x4_t zero = vdupq_n_f32(0.0f);
  const float32x4_t six = vdupq_n_f32(6.0f);
  return vminq_f32(vmaxq_f32(v, zero), six);
}

static void op15_pixel_checked(
    const float* input, const float* weights, const float* bias, float* out,
    int y, int x) {
  float* dst = out + ((size_t)y * W128 + x) * C8;
  for (int g = 0; g < 2; ++g) {
    int c = g * 4;
    float32x4_t acc = vld1q_f32(bias + c);
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y + ky - 1;
      if ((unsigned)iy >= H128) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x + kx - 1;
        if ((unsigned)ix >= W128) continue;
        acc = vfmaq_f32(
            acc,
            vld1q_f32(input + ((size_t)iy * W128 + ix) * C8 + c),
            vld1q_f32(weights + (ky * 3 + kx) * C8 + c));
      }
    }
    vst1q_f32(dst + c, relu6q(acc));
  }
}

static void op15_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  if (y0 == 0) {
    for (int x = 0; x < W128; ++x) op15_pixel_checked(input, weights, bias, out, 0, x);
    y0 = 1;
  }
  if (y1 == H128) {
    for (int x = 0; x < W128; ++x) op15_pixel_checked(input, weights, bias, out, H128 - 1, x);
    y1 = H128 - 1;
  }
  for (int y = y0; y < y1; ++y) {
    op15_pixel_checked(input, weights, bias, out, y, 0);
    op15_pixel_checked(input, weights, bias, out, y, W128 - 1);
    for (int x = 1; x < W128 - 1; ++x) {
      float* dst = out + ((size_t)y * W128 + x) * C8;
      const float* row0 = input + ((size_t)(y - 1) * W128 + x - 1) * C8;
      const float* row1 = input + ((size_t)y * W128 + x - 1) * C8;
      const float* row2 = input + ((size_t)(y + 1) * W128 + x - 1) * C8;
      for (int g = 0; g < 2; ++g) {
        int c = g * 4;
        float32x4_t acc = vld1q_f32(bias + c);
        acc = vfmaq_f32(acc, vld1q_f32(row0 + c), vld1q_f32(weights + 0 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + C8 + c), vld1q_f32(weights + 1 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + 2 * C8 + c), vld1q_f32(weights + 2 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + c), vld1q_f32(weights + 3 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + C8 + c), vld1q_f32(weights + 4 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + 2 * C8 + c), vld1q_f32(weights + 5 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + c), vld1q_f32(weights + 6 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + C8 + c), vld1q_f32(weights + 7 * C8 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + 2 * C8 + c), vld1q_f32(weights + 8 * C8 + c));
        vst1q_f32(dst + c, relu6q(acc));
      }
    }
  }
}

static void op18_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W128; ++x) {
      const float* src = input + ((size_t)y * W128 + x) * C8;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      for (int ci = 0; ci < C8; ++ci) {
        float v = src[ci];
        const float* w = weights + ci * C8;
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(w + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(w + 4), v);
      }
      float* dst = out + ((size_t)y * W128 + x) * C8;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
    }
  }
}

static void op18_add_rows(
    const float* input, const float* skip, const float* weights, const float* bias,
    float* out, int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W128; ++x) {
      const float* src = input + ((size_t)y * W128 + x) * C8;
      const float* residual = skip + ((size_t)y * W128 + x) * C8;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      for (int ci = 0; ci < C8; ++ci) {
        float v = src[ci];
        const float* w = weights + ci * C8;
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(w + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(w + 4), v);
      }
      float* dst = out + ((size_t)y * W128 + x) * C8;
      vst1q_f32(dst + 0, relu6q(vaddq_f32(vld1q_f32(residual + 0), acc0)));
      vst1q_f32(dst + 4, relu6q(vaddq_f32(vld1q_f32(residual + 4), acc1)));
    }
  }
}

static void op23_pixel_checked(
    const float* input, const float* weights, const float* bias, float* out,
    int y, int x) {
  float* dst = out + ((size_t)y * W64 + x) * C32;
  for (int g = 0; g < 8; ++g) {
    int c = g * 4;
    float32x4_t acc = vld1q_f32(bias + c);
    for (int ky = 0; ky < 3; ++ky) {
      int iy = y * 2 + ky - 1;
      if ((unsigned)iy >= H128) continue;
      for (int kx = 0; kx < 3; ++kx) {
        int ix = x * 2 + kx - 1;
        if ((unsigned)ix >= W128) continue;
        acc = vfmaq_f32(
            acc,
            vld1q_f32(input + ((size_t)iy * W128 + ix) * C32 + c),
            vld1q_f32(weights + (ky * 3 + kx) * C32 + c));
      }
    }
    vst1q_f32(dst + c, relu6q(acc));
  }
}

static void op23_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  if (y0 == 0) {
    for (int x = 0; x < W64; ++x) op23_pixel_checked(input, weights, bias, out, 0, x);
    y0 = 1;
  }
  for (int y = y0; y < y1; ++y) {
    op23_pixel_checked(input, weights, bias, out, y, 0);
    for (int x = 1; x < W64; ++x) {
      float* dst = out + ((size_t)y * W64 + x) * C32;
      const int iy0 = y * 2 - 1;
      const int ix0 = x * 2 - 1;
      const float* row0 = input + ((size_t)iy0 * W128 + ix0) * C32;
      const float* row1 = input + ((size_t)(iy0 + 1) * W128 + ix0) * C32;
      const float* row2 = input + ((size_t)(iy0 + 2) * W128 + ix0) * C32;
      for (int g = 0; g < 8; ++g) {
        int c = g * 4;
        float32x4_t acc = vld1q_f32(bias + c);
        acc = vfmaq_f32(acc, vld1q_f32(row0 + c), vld1q_f32(weights + 0 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + C32 + c), vld1q_f32(weights + 1 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row0 + 2 * C32 + c), vld1q_f32(weights + 2 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + c), vld1q_f32(weights + 3 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + C32 + c), vld1q_f32(weights + 4 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row1 + 2 * C32 + c), vld1q_f32(weights + 5 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + c), vld1q_f32(weights + 6 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + C32 + c), vld1q_f32(weights + 7 * C32 + c));
        acc = vfmaq_f32(acc, vld1q_f32(row2 + 2 * C32 + c), vld1q_f32(weights + 8 * C32 + c));
        vst1q_f32(dst + c, relu6q(acc));
      }
    }
  }
}

static void op26_rows(
    const float* input, const float* weights, const float* bias, float* out,
    int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    for (int x = 0; x < W64; ++x) {
      const float* src = input + ((size_t)y * W64 + x) * C32;
      float32x4_t acc0 = vld1q_f32(bias + 0);
      float32x4_t acc1 = vld1q_f32(bias + 4);
      float32x4_t acc2 = vld1q_f32(bias + 8);
      float32x4_t acc3 = vld1q_f32(bias + 12);
      for (int ci = 0; ci < C32; ++ci) {
        float v = src[ci];
        const float* w = weights + ci * C16;
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(w + 0), v);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(w + 4), v);
        acc2 = vfmaq_n_f32(acc2, vld1q_f32(w + 8), v);
        acc3 = vfmaq_n_f32(acc3, vld1q_f32(w + 12), v);
      }
      float* dst = out + ((size_t)y * W64 + x) * C16;
      vst1q_f32(dst + 0, acc0);
      vst1q_f32(dst + 4, acc1);
      vst1q_f32(dst + 8, acc2);
      vst1q_f32(dst + 12, acc3);
    }
  }
}

typedef struct {
  int id;
  int threads;
  int reps;
  pthread_barrier_t* barrier;
  const float* op9;
  const float* op12;
  const float* op15_w;
  const float* op15_b;
  const float* op18_w;
  const float* op18_b;
  const float* op23_w;
  const float* op23_b;
  const float* op26_w;
  const float* op26_b;
  float* op15;
  float* op18;
  float* op19;
  float* op23;
  float* op26;
} Worker;

static void* worker_main(void* ptr) {
  Worker* w = (Worker*)ptr;
  int y1280 = (H128 * w->id) / w->threads;
  int y1281 = (H128 * (w->id + 1)) / w->threads;
  int y640 = (H64 * w->id) / w->threads;
  int y641 = (H64 * (w->id + 1)) / w->threads;
  for (int r = 0; r < w->reps; ++r) {
    op15_rows(w->op9, w->op15_w, w->op15_b, w->op15, y1280, y1281);
    op23_rows(w->op12, w->op23_w, w->op23_b, w->op23, y640, y641);
    pthread_barrier_wait(w->barrier);
    op18_add_rows(w->op15, w->op9, w->op18_w, w->op18_b, w->op19, y1280, y1281);
    op26_rows(w->op23, w->op26_w, w->op26_b, w->op26, y640, y641);
    pthread_barrier_wait(w->barrier);
  }
  return NULL;
}
#endif

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 500;
  int threads = argc > 3 ? atoi(argv[3]) : 4;
  if (threads < 1) threads = 1;
  if (threads > 8) threads = 8;
  char path[512];

#define ALLOC_F32(count) aligned_alloc(64, (count) * sizeof(float))
  float* op9 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* op12 = ALLOC_F32((size_t)H128 * W128 * C32);
  float* op15 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* op18 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* op19 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* op23 = ALLOC_F32((size_t)H64 * W64 * C32);
  float* op26 = ALLOC_F32((size_t)H64 * W64 * C16);
  float* op15_w = ALLOC_F32(3 * 3 * C8);
  float* op15_b = ALLOC_F32(C8);
  float* op18_w_ohwi = ALLOC_F32(C8 * C8);
  float* op18_w = ALLOC_F32(C8 * C8);
  float* op18_b = ALLOC_F32(C8);
  float* op23_w = ALLOC_F32(3 * 3 * C32);
  float* op23_b = ALLOC_F32(C32);
  float* op26_w_ohwi = ALLOC_F32(C16 * C32);
  float* op26_w = ALLOC_F32(C16 * C32);
  float* op26_b = ALLOC_F32(C16);
  float* ref15 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* ref18 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* ref19 = ALLOC_F32((size_t)H128 * W128 * C8);
  float* ref23 = ALLOC_F32((size_t)H64 * W64 * C32);
  float* ref26 = ALLOC_F32((size_t)H64 * W64 * C16);
  if (!op9 || !op12 || !op15 || !op18 || !op19 || !op23 || !op26 ||
      !op15_w || !op15_b || !op18_w_ohwi || !op18_w || !op18_b ||
      !op23_w || !op23_b || !op26_w_ohwi || !op26_w || !op26_b ||
      !ref15 || !ref18 || !ref19 || !ref23 || !ref26) {
    fprintf(stderr, "allocation failed\n");
    return 2;
  }

#define LOAD(name, ptr, bytes) do { \
  snprintf(path, sizeof(path), "%s/%s", dir, name); \
  read_exact(path, ptr, bytes); \
} while (0)
  LOAD("op9_ref_f32.bin", op9, (size_t)H128 * W128 * C8 * sizeof(float));
  LOAD("op12_ref_f32.bin", op12, (size_t)H128 * W128 * C32 * sizeof(float));
  LOAD("op15_w_1hwc_f32.bin", op15_w, 3 * 3 * C8 * sizeof(float));
  LOAD("op15_b_f32.bin", op15_b, C8 * sizeof(float));
  LOAD("op18_w_ohwi_f32.bin", op18_w_ohwi, C8 * C8 * sizeof(float));
  LOAD("op18_b_f32.bin", op18_b, C8 * sizeof(float));
  LOAD("op23_w_1hwc_f32.bin", op23_w, 3 * 3 * C32 * sizeof(float));
  LOAD("op23_b_f32.bin", op23_b, C32 * sizeof(float));
  LOAD("op26_w_ohwi_f32.bin", op26_w_ohwi, C16 * C32 * sizeof(float));
  LOAD("op26_b_f32.bin", op26_b, C16 * sizeof(float));
  LOAD("op15_ref_f32.bin", ref15, (size_t)H128 * W128 * C8 * sizeof(float));
  LOAD("op18_ref_f32.bin", ref18, (size_t)H128 * W128 * C8 * sizeof(float));
  LOAD("op19_ref_f32.bin", ref19, (size_t)H128 * W128 * C8 * sizeof(float));
  LOAD("op23_ref_f32.bin", ref23, (size_t)H64 * W64 * C32 * sizeof(float));
  LOAD("op26_ref_f32.bin", ref26, (size_t)H64 * W64 * C16 * sizeof(float));
  transpose_1x1(op18_w_ohwi, op18_w, C8, C8);
  transpose_1x1(op26_w_ohwi, op26_w, C32, C16);

#if defined(__aarch64__)
  pthread_t tids[8];
  Worker workers[8];
  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, (unsigned)threads);
  double t0 = now_s();
  for (int t = 0; t < threads; ++t) {
    workers[t] = (Worker){
        t, threads, reps, &barrier,
        op9, op12, op15_w, op15_b, op18_w, op18_b, op23_w, op23_b, op26_w, op26_b,
        op15, op18, op19, op23, op26};
    pthread_create(&tids[t], NULL, worker_main, &workers[t]);
  }
  for (int t = 0; t < threads; ++t) pthread_join(tids[t], NULL);
  double t1 = now_s();
  pthread_barrier_destroy(&barrier);

  double macs =
      (double)H128 * W128 * C8 * 9.0 +
      (double)H128 * W128 * C8 * C8 +
      (double)H64 * W64 * C32 * 9.0 +
      (double)H64 * W64 * C32 * C16;
  printf("neon_block2_fused threads=%d reps=%d avg_ms=%.3f fps=%.2f gmac_s=%.3f macs=%.0f\n",
      threads, reps, (t1 - t0) * 1000.0 / reps, reps / (t1 - t0),
      macs * reps / (t1 - t0) / 1e9, macs);
  compare_ref("op15", op15, ref15, (size_t)H128 * W128 * C8);
  compare_ref("op19", op19, ref19, (size_t)H128 * W128 * C8);
  compare_ref("op23", op23, ref23, (size_t)H64 * W64 * C32);
  compare_ref("op26", op26, ref26, (size_t)H64 * W64 * C16);
  printf("checksums %.6f %.6f %.6f %.6f %.6f\n",
      op15[12345], op18[12345], op19[12345], op23[12345], op26[12345]);
#else
  (void)reps;
  (void)threads;
  fprintf(stderr, "requires aarch64\n");
  return 2;
#endif

  return 0;
}
