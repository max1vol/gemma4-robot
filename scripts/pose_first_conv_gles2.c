// KMS-safe Raspberry Pi 3B+ GPU probe for the first Pose Landmarker Lite
// landmarker convolution. Uses OpenGL ES 2.0 fragment shaders through Mesa VC4.
//
// This is intentionally not a full runtime yet. It tests the practical fixed
// point/texture path available under vc4-kms-v3d:
//   input: 256x256 RGB, quantized to RGBA8 texture
//   op:    3x3 stride-2 conv, 24 outputs, bias, ReLU6
//   output: six 128x128 RGBA8 textures, storing activation / 6

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void* EGLDisplay;
typedef void* EGLConfig;
typedef void* EGLContext;
typedef void* EGLSurface;
typedef void* EGLNativeDisplayType;
typedef int32_t EGLint;
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef char GLchar;
typedef float GLfloat;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;

enum {
  IN_H = 256,
  IN_W = 256,
  IN_C = 3,
  OUT_H = 128,
  OUT_W = 128,
  OUT_C = 24,
  GROUPS = 6,
  OP9_C = 8,
  OP9_GROUPS = 2,
  OP12_C = 32,
  OP12_GROUPS = 8,
  K_H = 3,
  K_W = 3,
  K_SIZE = K_H * K_W * IN_C,
};

#define EGL_DEFAULT_DISPLAY ((EGLNativeDisplayType)0)
#define EGL_NO_DISPLAY ((EGLDisplay)0)
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_NONE 0x3038
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_OPENGL_ES2_BIT 0x0004
#define EGL_SURFACE_TYPE 0x3033
#define EGL_PBUFFER_BIT 0x0001
#define EGL_RED_SIZE 0x3024
#define EGL_GREEN_SIZE 0x3023
#define EGL_BLUE_SIZE 0x3022
#define EGL_ALPHA_SIZE 0x3021
#define EGL_DEPTH_SIZE 0x3025
#define EGL_WIDTH 0x3057
#define EGL_HEIGHT 0x3056
#define EGL_CONTEXT_CLIENT_VERSION 0x3098
#define EGL_OPENGL_ES_API 0x30A0
#define EGL_VENDOR 0x3053
#define EGL_VERSION 0x3054
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD

#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_ARRAY_BUFFER 0x8892
#define GL_STATIC_DRAW 0x88E4
#define GL_FLOAT 0x1406
#define GL_FALSE 0
#define GL_TRIANGLE_STRIP 0x0005
#define GL_TEXTURE_2D 0x0DE1
#define GL_TEXTURE0 0x84C0
#define GL_TEXTURE1 0x84C1
#define GL_TEXTURE2 0x84C2
#define GL_TEXTURE3 0x84C3
#define GL_TEXTURE4 0x84C4
#define GL_TEXTURE5 0x84C5
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_TEXTURE_WRAP_S 0x2802
#define GL_TEXTURE_WRAP_T 0x2803
#define GL_NEAREST 0x2600
#define GL_CLAMP_TO_EDGE 0x812F
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02

#define PROC(ret, name, args) static ret (*name) args
PROC(EGLDisplay, eglGetDisplay, (EGLNativeDisplayType));
PROC(EGLint, eglInitialize, (EGLDisplay, EGLint*, EGLint*));
PROC(const char*, eglQueryString, (EGLDisplay, EGLint));
PROC(EGLint, eglBindAPI, (EGLint));
PROC(EGLint, eglChooseConfig, (EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*));
PROC(EGLSurface, eglCreatePbufferSurface, (EGLDisplay, EGLConfig, const EGLint*));
PROC(EGLContext, eglCreateContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint*));
PROC(EGLint, eglMakeCurrent, (EGLDisplay, EGLSurface, EGLSurface, EGLContext));
PROC(EGLint, eglDestroyContext, (EGLDisplay, EGLContext));
PROC(EGLint, eglDestroySurface, (EGLDisplay, EGLSurface));
PROC(EGLint, eglTerminate, (EGLDisplay));
PROC(EGLint, eglGetError, (void));
PROC(void*, eglGetProcAddress, (const char*));
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLint, void*, const EGLint*);

PROC(const unsigned char*, glGetString, (GLenum));
PROC(void, glViewport, (GLint, GLint, GLsizei, GLsizei));
PROC(GLuint, glCreateShader, (GLenum));
PROC(void, glShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*));
PROC(void, glCompileShader, (GLuint));
PROC(void, glGetShaderiv, (GLuint, GLenum, GLint*));
PROC(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*));
PROC(GLuint, glCreateProgram, (void));
PROC(void, glAttachShader, (GLuint, GLuint));
PROC(void, glLinkProgram, (GLuint));
PROC(void, glGetProgramiv, (GLuint, GLenum, GLint*));
PROC(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*));
PROC(void, glUseProgram, (GLuint));
PROC(void, glGenBuffers, (GLsizei, GLuint*));
PROC(void, glBindBuffer, (GLenum, GLuint));
PROC(void, glBufferData, (GLenum, intptr_t, const void*, GLenum));
PROC(GLint, glGetAttribLocation, (GLuint, const GLchar*));
PROC(GLint, glGetUniformLocation, (GLuint, const GLchar*));
PROC(void, glUniform1i, (GLint, GLint));
PROC(void, glEnableVertexAttribArray, (GLuint));
PROC(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
PROC(void, glDrawArrays, (GLenum, GLint, GLsizei));
PROC(void, glFinish, (void));
PROC(void, glGenTextures, (GLsizei, GLuint*));
PROC(void, glActiveTexture, (GLenum));
PROC(void, glBindTexture, (GLenum, GLuint));
PROC(void, glTexParameteri, (GLenum, GLenum, GLint));
PROC(void, glTexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
PROC(void, glGenFramebuffers, (GLsizei, GLuint*));
PROC(void, glBindFramebuffer, (GLenum, GLuint));
PROC(void, glFramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint));
PROC(GLenum, glCheckFramebufferStatus, (GLenum));
PROC(void, glReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*));
PROC(void, glDeleteProgram, (GLuint));
PROC(void, glDeleteShader, (GLuint));

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void* sym(void* lib, const char* name) {
  void* ptr = dlsym(lib, name);
  if (!ptr) {
    fprintf(stderr, "missing symbol %s\n", name);
    exit(2);
  }
  return ptr;
}

#define LOAD(lib, name) name = sym(lib, #name)
static void load_symbols(void) {
  void* egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
  void* gl = dlopen("libGLESv2.so.2", RTLD_NOW | RTLD_LOCAL);
  if (!egl || !gl) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    exit(2);
  }
  LOAD(egl, eglGetDisplay); LOAD(egl, eglInitialize); LOAD(egl, eglQueryString);
  LOAD(egl, eglBindAPI); LOAD(egl, eglChooseConfig); LOAD(egl, eglCreatePbufferSurface);
  LOAD(egl, eglCreateContext); LOAD(egl, eglMakeCurrent); LOAD(egl, eglDestroyContext);
  LOAD(egl, eglDestroySurface); LOAD(egl, eglTerminate); LOAD(egl, eglGetError);
  LOAD(egl, eglGetProcAddress);
  LOAD(gl, glGetString); LOAD(gl, glViewport); LOAD(gl, glCreateShader);
  LOAD(gl, glShaderSource); LOAD(gl, glCompileShader); LOAD(gl, glGetShaderiv);
  LOAD(gl, glGetShaderInfoLog); LOAD(gl, glCreateProgram); LOAD(gl, glAttachShader);
  LOAD(gl, glLinkProgram); LOAD(gl, glGetProgramiv); LOAD(gl, glGetProgramInfoLog);
  LOAD(gl, glUseProgram); LOAD(gl, glGenBuffers); LOAD(gl, glBindBuffer);
  LOAD(gl, glBufferData); LOAD(gl, glGetAttribLocation); LOAD(gl, glGetUniformLocation);
  LOAD(gl, glUniform1i); LOAD(gl, glEnableVertexAttribArray); LOAD(gl, glVertexAttribPointer);
  LOAD(gl, glDrawArrays); LOAD(gl, glFinish); LOAD(gl, glGenTextures);
  LOAD(gl, glActiveTexture); LOAD(gl, glBindTexture); LOAD(gl, glTexParameteri);
  LOAD(gl, glTexImage2D); LOAD(gl, glGenFramebuffers); LOAD(gl, glBindFramebuffer);
  LOAD(gl, glFramebufferTexture2D); LOAD(gl, glCheckFramebufferStatus); LOAD(gl, glReadPixels);
  LOAD(gl, glDeleteProgram); LOAD(gl, glDeleteShader);
}

typedef struct {
  char* data;
  size_t len;
  size_t cap;
} Str;

static void appendf(Str* s, const char* fmt, ...) {
  for (;;) {
    if (s->cap - s->len < 1024) {
      s->cap = s->cap ? s->cap * 2 : 8192;
      s->data = realloc(s->data, s->cap);
      if (!s->data) exit(2);
    }
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(s->data + s->len, s->cap - s->len, fmt, ap);
    va_end(ap);
    if (n < 0) exit(2);
    if ((size_t)n < s->cap - s->len) {
      s->len += (size_t)n;
      return;
    }
    s->cap *= 2;
    s->data = realloc(s->data, s->cap);
    if (!s->data) exit(2);
  }
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

static GLuint compile_shader(GLenum type, const char* src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[8192];
    GLsizei n = 0;
    glGetShaderInfoLog(shader, sizeof(log), &n, log);
    fprintf(stderr, "shader compile failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  return shader;
}

static GLuint make_program(const float* w_ohwi, const float* bias, int group) {
  const char* vs =
      "attribute vec2 pos;\n"
      "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D src;\n"
      "vec3 sample_at(float ix, float iy) {\n"
      "  if (ix < 0.0 || ix >= 256.0 || iy < 0.0 || iy >= 256.0) return vec3(0.0);\n"
      "  return texture2D(src, vec2((ix + 0.5) * %.12g, (iy + 0.5) * %.12g)).rgb;\n"
      "}\n"
      "void main() {\n"
      "  float ox = gl_FragCoord.x;\n"
      "  float oy = gl_FragCoord.y;\n"
      "  vec4 acc = vec4(%.9g, %.9g, %.9g, %.9g);\n",
      1.0 / 256.0, 1.0 / 256.0,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ky = 0; ky < K_H; ++ky) {
    for (int kx = 0; kx < K_W; ++kx) {
      appendf(&fs, "  { vec3 s = sample_at(ox * 2.0 + %.1f, oy * 2.0 + %.1f);\n",
          (double)kx - 1.0, (double)ky - 1.0);
      for (int ci = 0; ci < IN_C; ++ci) {
        const char chan = ci == 0 ? 'r' : (ci == 1 ? 'g' : 'b');
        int base_co = group * 4;
        float ww[4];
        for (int lane = 0; lane < 4; ++lane) {
          int co = base_co + lane;
          int idx = ((co * K_H + ky) * K_W + kx) * IN_C + ci;
          ww[lane] = w_ohwi[idx];
        }
        appendf(&fs,
            "    acc += vec4(%.9g, %.9g, %.9g, %.9g) * s.%c;\n",
            ww[0], ww[1], ww[2], ww[3], chan);
      }
      appendf(&fs, "  }\n");
    }
  }
  appendf(&fs, "  gl_FragColor = clamp(acc, 0.0, 6.0) * %.12g;\n}\n", 1.0 / 6.0);

  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[8192];
    GLsizei n = 0;
    glGetProgramInfoLog(p, sizeof(log), &n, log);
    fprintf(stderr, "link failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  free(fs.data);
  return p;
}

static GLuint make_depthwise_program(const float* w_1hwc, const float* bias, int group) {
  const char* vs =
      "attribute vec2 pos;\n"
      "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D src;\n"
      "vec4 sample_at(float ix, float iy) {\n"
      "  if (ix < 0.0 || ix >= 128.0 || iy < 0.0 || iy >= 128.0) return vec4(0.0);\n"
      "  return texture2D(src, vec2((ix + 0.5) * %.12g, (iy + 0.5) * %.12g)) * 6.0;\n"
      "}\n"
      "void main() {\n"
      "  float ox = gl_FragCoord.x;\n"
      "  float oy = gl_FragCoord.y;\n"
      "  vec4 acc = vec4(%.9g, %.9g, %.9g, %.9g);\n",
      1.0 / 128.0, 1.0 / 128.0,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ky = 0; ky < K_H; ++ky) {
    for (int kx = 0; kx < K_W; ++kx) {
      int base = (ky * K_W + kx) * OUT_C + group * 4;
      appendf(&fs,
          "  acc += sample_at(ox + %.1f, oy + %.1f) * vec4(%.9g, %.9g, %.9g, %.9g);\n",
          (double)kx - 1.0, (double)ky - 1.0,
          w_1hwc[base + 0], w_1hwc[base + 1], w_1hwc[base + 2], w_1hwc[base + 3]);
    }
  }
  appendf(&fs, "  gl_FragColor = clamp(acc, 0.0, 6.0) * %.12g;\n}\n", 1.0 / 6.0);

  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[8192];
    GLsizei n = 0;
    glGetProgramInfoLog(p, sizeof(log), &n, log);
    fprintf(stderr, "depthwise link failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  free(fs.data);
  return p;
}

static GLuint make_op9_program(const float* w_ohwi, const float* bias, int group, float q_min, float q_range) {
  const char* vs =
      "attribute vec2 pos;\n"
      "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D s0;\n"
      "uniform sampler2D s1;\n"
      "uniform sampler2D s2;\n"
      "uniform sampler2D s3;\n"
      "uniform sampler2D s4;\n"
      "uniform sampler2D s5;\n"
      "void main() {\n"
      "  vec2 uv = vec2((gl_FragCoord.x + 0.5) * %.12g, (gl_FragCoord.y + 0.5) * %.12g);\n"
      "  vec4 acc = vec4(%.9g, %.9g, %.9g, %.9g);\n",
      1.0 / 128.0, 1.0 / 128.0,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ig = 0; ig < GROUPS; ++ig) {
    appendf(&fs, "  { vec4 src = texture2D(s%d, uv) * 6.0;\n", ig);
    for (int lane = 0; lane < 4; ++lane) {
      int ci = ig * 4 + lane;
      const char chan = lane == 0 ? 'r' : lane == 1 ? 'g' : lane == 2 ? 'b' : 'a';
      float ww[4];
      for (int out_lane = 0; out_lane < 4; ++out_lane) {
        int co = group * 4 + out_lane;
        ww[out_lane] = w_ohwi[co * OUT_C + ci];
      }
      appendf(&fs,
          "    acc += vec4(%.9g, %.9g, %.9g, %.9g) * src.%c;\n",
          ww[0], ww[1], ww[2], ww[3], chan);
    }
    appendf(&fs, "  }\n");
  }
  appendf(&fs,
      "  gl_FragColor = clamp((acc - %.9g) * %.12g, 0.0, 1.0);\n"
      "}\n",
      q_min, 1.0 / q_range);

  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[8192];
    GLsizei n = 0;
    glGetProgramInfoLog(p, sizeof(log), &n, log);
    fprintf(stderr, "op9 link failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  free(fs.data);
  return p;
}

static GLuint make_op12_program(const float* w_ohwi, const float* bias, int group, float q_min, float q_range) {
  const char* vs =
      "attribute vec2 pos;\n"
      "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D s0;\n"
      "uniform sampler2D s1;\n"
      "void main() {\n"
      "  vec2 uv = vec2((gl_FragCoord.x + 0.5) * %.12g, (gl_FragCoord.y + 0.5) * %.12g);\n"
      "  vec4 v0 = texture2D(s0, uv) * %.9g + %.9g;\n"
      "  vec4 v1 = texture2D(s1, uv) * %.9g + %.9g;\n"
      "  vec4 acc = vec4(%.9g, %.9g, %.9g, %.9g);\n",
      1.0 / 128.0, 1.0 / 128.0,
      q_range, q_min, q_range, q_min,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ci = 0; ci < OP9_C; ++ci) {
    const char* src = ci < 4 ? "v0" : "v1";
    int lane = ci & 3;
    const char chan = lane == 0 ? 'r' : lane == 1 ? 'g' : lane == 2 ? 'b' : 'a';
    float ww[4];
    for (int out_lane = 0; out_lane < 4; ++out_lane) {
      int co = group * 4 + out_lane;
      ww[out_lane] = w_ohwi[co * OP9_C + ci];
    }
    appendf(&fs,
        "  acc += vec4(%.9g, %.9g, %.9g, %.9g) * %s.%c;\n",
        ww[0], ww[1], ww[2], ww[3], src, chan);
  }
  appendf(&fs, "  gl_FragColor = clamp(acc, 0.0, 6.0) * %.12g;\n}\n", 1.0 / 6.0);

  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[8192];
    GLsizei n = 0;
    glGetProgramInfoLog(p, sizeof(log), &n, log);
    fprintf(stderr, "op12 link failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  free(fs.data);
  return p;
}

static void setup_egl(EGLDisplay* display, EGLSurface* surface, EGLContext* ctx) {
  *display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLint major = 0, minor = 0;
  if (*display == EGL_NO_DISPLAY || !eglInitialize(*display, &major, &minor)) {
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!get_platform) {
      fprintf(stderr, "no EGL display\n");
      exit(2);
    }
    *display = get_platform(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
    if (*display == EGL_NO_DISPLAY || !eglInitialize(*display, &major, &minor)) {
      fprintf(stderr, "eglInitialize failed: 0x%x\n", eglGetError());
      exit(2);
    }
  }
  printf("EGL %d.%d vendor=%s version=%s\n", major, minor,
      eglQueryString(*display, EGL_VENDOR), eglQueryString(*display, EGL_VERSION));
  eglBindAPI(EGL_OPENGL_ES_API);
  EGLint cfg_attr[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_DEPTH_SIZE, 0,
      EGL_NONE};
  EGLConfig cfg = NULL;
  EGLint num = 0;
  if (!eglChooseConfig(*display, cfg_attr, &cfg, 1, &num) || num < 1) {
    fprintf(stderr, "eglChooseConfig failed: 0x%x\n", eglGetError());
    exit(2);
  }
  EGLint pb_attr[] = {EGL_WIDTH, OUT_W, EGL_HEIGHT, OUT_H, EGL_NONE};
  *surface = eglCreatePbufferSurface(*display, cfg, pb_attr);
  EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  *ctx = eglCreateContext(*display, cfg, EGL_NO_CONTEXT, ctx_attr);
  if (*surface == EGL_NO_SURFACE || *ctx == EGL_NO_CONTEXT ||
      !eglMakeCurrent(*display, *surface, *surface, *ctx)) {
    fprintf(stderr, "context creation failed: 0x%x\n", eglGetError());
    exit(2);
  }
  printf("GL vendor=%s\nGL renderer=%s\nGL version=%s\n",
      glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION));
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 200;
  char path[512];

  float* input = malloc((size_t)IN_H * IN_W * IN_C * sizeof(float));
  float* w = malloc((size_t)OUT_C * K_SIZE * sizeof(float));
  float* bias = malloc(OUT_C * sizeof(float));
  float* ref = malloc((size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  float* dw_w = malloc((size_t)K_H * K_W * OUT_C * sizeof(float));
  float* dw_bias = malloc(OUT_C * sizeof(float));
  float* dw_ref = malloc((size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  float* op9_w = malloc((size_t)OP9_C * OUT_C * sizeof(float));
  float* op9_bias = malloc(OP9_C * sizeof(float));
  float* op9_ref = malloc((size_t)OUT_H * OUT_W * OP9_C * sizeof(float));
  float* op12_w = malloc((size_t)OP12_C * OP9_C * sizeof(float));
  float* op12_bias = malloc(OP12_C * sizeof(float));
  float* op12_ref = malloc((size_t)OUT_H * OUT_W * OP12_C * sizeof(float));
  if (!input || !w || !bias || !ref || !dw_w || !dw_bias || !dw_ref ||
      !op9_w || !op9_bias || !op9_ref || !op12_w || !op12_bias || !op12_ref) return 2;
  snprintf(path, sizeof(path), "%s/input_f32.bin", dir);
  read_exact(path, input, (size_t)IN_H * IN_W * IN_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/w_ohwi_f32.bin", dir);
  read_exact(path, w, (size_t)OUT_C * K_SIZE * sizeof(float));
  snprintf(path, sizeof(path), "%s/b_f32.bin", dir);
  read_exact(path, bias, OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/ref_out_f32.bin", dir);
  read_exact(path, ref, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op6_w_1hwc_f32.bin", dir);
  read_exact(path, dw_w, (size_t)K_H * K_W * OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op6_b_f32.bin", dir);
  read_exact(path, dw_bias, OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op6_ref_from_qop3_f32.bin", dir);
  read_exact(path, dw_ref, (size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op9_w_ohwi_f32.bin", dir);
  read_exact(path, op9_w, (size_t)OP9_C * OUT_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op9_b_f32.bin", dir);
  read_exact(path, op9_bias, OP9_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op9_ref_from_qop6_f32.bin", dir);
  read_exact(path, op9_ref, (size_t)OUT_H * OUT_W * OP9_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op12_w_ohwi_f32.bin", dir);
  read_exact(path, op12_w, (size_t)OP12_C * OP9_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op12_b_f32.bin", dir);
  read_exact(path, op12_bias, OP12_C * sizeof(float));
  snprintf(path, sizeof(path), "%s/op12_ref_from_qop9_f32.bin", dir);
  read_exact(path, op12_ref, (size_t)OUT_H * OUT_W * OP12_C * sizeof(float));

  float op9_min = op9_ref[0], op9_max = op9_ref[0];
  for (size_t i = 1; i < (size_t)OUT_H * OUT_W * OP9_C; ++i) {
    if (op9_ref[i] < op9_min) op9_min = op9_ref[i];
    if (op9_ref[i] > op9_max) op9_max = op9_ref[i];
  }
  float op9_range = op9_max - op9_min;
  if (op9_range <= 0.0f) op9_range = 1.0f;

  uint8_t* rgba = malloc((size_t)IN_H * IN_W * 4);
  for (int y = 0; y < IN_H; ++y) {
    for (int x = 0; x < IN_W; ++x) {
      size_t pix = (size_t)y * IN_W + x;
      rgba[pix * 4 + 0] = (uint8_t)lrintf(fminf(fmaxf(input[pix * 3 + 0], 0.0f), 1.0f) * 255.0f);
      rgba[pix * 4 + 1] = (uint8_t)lrintf(fminf(fmaxf(input[pix * 3 + 1], 0.0f), 1.0f) * 255.0f);
      rgba[pix * 4 + 2] = (uint8_t)lrintf(fminf(fmaxf(input[pix * 3 + 2], 0.0f), 1.0f) * 255.0f);
      rgba[pix * 4 + 3] = 255;
    }
  }

  load_symbols();
  EGLDisplay display;
  EGLSurface surface;
  EGLContext ctx;
  setup_egl(&display, &surface, &ctx);

  GLuint programs[GROUPS];
  for (int g = 0; g < GROUPS; ++g) programs[g] = make_program(w, bias, g);
  GLuint dw_programs[GROUPS];
  for (int g = 0; g < GROUPS; ++g) dw_programs[g] = make_depthwise_program(dw_w, dw_bias, g);
  GLuint op9_programs[OP9_GROUPS];
  for (int g = 0; g < OP9_GROUPS; ++g) op9_programs[g] = make_op9_program(op9_w, op9_bias, g, op9_min, op9_range);
  GLuint op12_programs[OP12_GROUPS];
  for (int g = 0; g < OP12_GROUPS; ++g) op12_programs[g] = make_op12_program(op12_w, op12_bias, g, op9_min, op9_range);

  GLuint input_tex = 0;
  glGenTextures(1, &input_tex);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, input_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, IN_W, IN_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);

  GLuint out_tex[GROUPS], fbo[GROUPS];
  glGenTextures(GROUPS, out_tex);
  glGenFramebuffers(GROUPS, fbo);
  for (int g = 0; g < GROUPS; ++g) {
    glBindTexture(GL_TEXTURE_2D, out_tex[g]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OUT_W, OUT_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex[g], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete fbo %d\n", g);
      return 2;
    }
  }

  GLuint dw_tex[GROUPS], dw_fbo[GROUPS];
  glGenTextures(GROUPS, dw_tex);
  glGenFramebuffers(GROUPS, dw_fbo);
  for (int g = 0; g < GROUPS; ++g) {
    glBindTexture(GL_TEXTURE_2D, dw_tex[g]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OUT_W, OUT_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, dw_fbo[g]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, dw_tex[g], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete depthwise fbo %d\n", g);
      return 2;
    }
  }

  GLuint op9_tex[OP9_GROUPS], op9_fbo[OP9_GROUPS];
  glGenTextures(OP9_GROUPS, op9_tex);
  glGenFramebuffers(OP9_GROUPS, op9_fbo);
  for (int g = 0; g < OP9_GROUPS; ++g) {
    glBindTexture(GL_TEXTURE_2D, op9_tex[g]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OUT_W, OUT_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, op9_fbo[g]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, op9_tex[g], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete op9 fbo %d\n", g);
      return 2;
    }
  }

  GLuint op12_tex[OP12_GROUPS], op12_fbo[OP12_GROUPS];
  glGenTextures(OP12_GROUPS, op12_tex);
  glGenFramebuffers(OP12_GROUPS, op12_fbo);
  for (int g = 0; g < OP12_GROUPS; ++g) {
    glBindTexture(GL_TEXTURE_2D, op12_tex[g]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, OUT_W, OUT_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, op12_fbo[g]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, op12_tex[g], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete op12 fbo %d\n", g);
      return 2;
    }
  }

  GLfloat verts[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
  GLuint vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glViewport(0, 0, OUT_W, OUT_H);

  double t0 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(programs[g]);
      GLint loc = glGetAttribLocation(programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, input_tex);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
  }
  glFinish();
  double t1 = now_s();

  double t2 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(dw_programs[g]);
      GLint loc = glGetAttribLocation(dw_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(dw_programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, dw_fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, out_tex[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
  }
  glFinish();
  double t3 = now_s();

  double t4 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < OP9_GROUPS; ++g) {
      glUseProgram(op9_programs[g]);
      GLint loc = glGetAttribLocation(op9_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      for (int ig = 0; ig < GROUPS; ++ig) {
        char uname[4];
        snprintf(uname, sizeof(uname), "s%d", ig);
        glUniform1i(glGetUniformLocation(op9_programs[g], uname), ig);
        glActiveTexture(GL_TEXTURE0 + (GLenum)ig);
        glBindTexture(GL_TEXTURE_2D, dw_tex[ig]);
      }
      glBindFramebuffer(GL_FRAMEBUFFER, op9_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
  }
  glFinish();
  double t5 = now_s();

  double t6 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < OP12_GROUPS; ++g) {
      glUseProgram(op12_programs[g]);
      GLint loc = glGetAttribLocation(op12_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(op12_programs[g], "s0"), 0);
      glUniform1i(glGetUniformLocation(op12_programs[g], "s1"), 1);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, op9_tex[0]);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, op9_tex[1]);
      glBindFramebuffer(GL_FRAMEBUFFER, op12_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
  }
  glFinish();
  double t7 = now_s();

  double t8 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(programs[g]);
      GLint loc = glGetAttribLocation(programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, input_tex);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(dw_programs[g]);
      GLint loc = glGetAttribLocation(dw_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(dw_programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, dw_fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, out_tex[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    for (int g = 0; g < OP9_GROUPS; ++g) {
      glUseProgram(op9_programs[g]);
      GLint loc = glGetAttribLocation(op9_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      for (int ig = 0; ig < GROUPS; ++ig) {
        char uname[4];
        snprintf(uname, sizeof(uname), "s%d", ig);
        glUniform1i(glGetUniformLocation(op9_programs[g], uname), ig);
        glActiveTexture(GL_TEXTURE0 + (GLenum)ig);
        glBindTexture(GL_TEXTURE_2D, dw_tex[ig]);
      }
      glBindFramebuffer(GL_FRAMEBUFFER, op9_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    for (int g = 0; g < OP12_GROUPS; ++g) {
      glUseProgram(op12_programs[g]);
      GLint loc = glGetAttribLocation(op12_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(op12_programs[g], "s0"), 0);
      glUniform1i(glGetUniformLocation(op12_programs[g], "s1"), 1);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, op9_tex[0]);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, op9_tex[1]);
      glBindFramebuffer(GL_FRAMEBUFFER, op12_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
  }
  glFinish();
  double t9 = now_s();

  uint8_t* out = calloc((size_t)OUT_H * OUT_W * OUT_C, 1);
  uint8_t* dw_out = calloc((size_t)OUT_H * OUT_W * OUT_C, 1);
  uint8_t* op9_out = calloc((size_t)OUT_H * OUT_W * OP9_C, 1);
  uint8_t* op12_out = calloc((size_t)OUT_H * OUT_W * OP12_C, 1);
  uint8_t* group_pixels = malloc((size_t)OUT_H * OUT_W * 4);
  for (int g = 0; g < GROUPS; ++g) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
    glReadPixels(0, 0, OUT_W, OUT_H, GL_RGBA, GL_UNSIGNED_BYTE, group_pixels);
    for (int y = 0; y < OUT_H; ++y) {
      for (int x = 0; x < OUT_W; ++x) {
        size_t p = (size_t)y * OUT_W + x;
        for (int c = 0; c < 4; ++c) {
          out[p * OUT_C + g * 4 + c] = group_pixels[p * 4 + c];
        }
      }
    }
  }
  for (int g = 0; g < GROUPS; ++g) {
    glBindFramebuffer(GL_FRAMEBUFFER, dw_fbo[g]);
    glReadPixels(0, 0, OUT_W, OUT_H, GL_RGBA, GL_UNSIGNED_BYTE, group_pixels);
    for (int y = 0; y < OUT_H; ++y) {
      for (int x = 0; x < OUT_W; ++x) {
        size_t p = (size_t)y * OUT_W + x;
        for (int c = 0; c < 4; ++c) {
          dw_out[p * OUT_C + g * 4 + c] = group_pixels[p * 4 + c];
        }
      }
    }
  }
  for (int g = 0; g < OP9_GROUPS; ++g) {
    glBindFramebuffer(GL_FRAMEBUFFER, op9_fbo[g]);
    glReadPixels(0, 0, OUT_W, OUT_H, GL_RGBA, GL_UNSIGNED_BYTE, group_pixels);
    for (int y = 0; y < OUT_H; ++y) {
      for (int x = 0; x < OUT_W; ++x) {
        size_t p = (size_t)y * OUT_W + x;
        for (int c = 0; c < 4; ++c) {
          op9_out[p * OP9_C + g * 4 + c] = group_pixels[p * 4 + c];
        }
      }
    }
  }
  for (int g = 0; g < OP12_GROUPS; ++g) {
    glBindFramebuffer(GL_FRAMEBUFFER, op12_fbo[g]);
    glReadPixels(0, 0, OUT_W, OUT_H, GL_RGBA, GL_UNSIGNED_BYTE, group_pixels);
    for (int y = 0; y < OUT_H; ++y) {
      for (int x = 0; x < OUT_W; ++x) {
        size_t p = (size_t)y * OUT_W + x;
        for (int c = 0; c < 4; ++c) {
          op12_out[p * OP12_C + g * 4 + c] = group_pixels[p * 4 + c];
        }
      }
    }
  }
  glFinish();
  snprintf(path, sizeof(path), "%s/gles_first_conv_out_u8.bin", dir);
  FILE* dump = fopen(path, "wb");
  if (dump) {
    fwrite(out, 1, (size_t)OUT_H * OUT_W * OUT_C, dump);
    fclose(dump);
  }
  snprintf(path, sizeof(path), "%s/gles_op6_depthwise_out_u8.bin", dir);
  dump = fopen(path, "wb");
  if (dump) {
    fwrite(dw_out, 1, (size_t)OUT_H * OUT_W * OUT_C, dump);
    fclose(dump);
  }
  snprintf(path, sizeof(path), "%s/gles_op9_pointwise_out_u8.bin", dir);
  dump = fopen(path, "wb");
  if (dump) {
    fwrite(op9_out, 1, (size_t)OUT_H * OUT_W * OP9_C, dump);
    fclose(dump);
  }
  snprintf(path, sizeof(path), "%s/gles_op12_pointwise_out_u8.bin", dir);
  dump = fopen(path, "wb");
  if (dump) {
    fwrite(op12_out, 1, (size_t)OUT_H * OUT_W * OP12_C, dump);
    fclose(dump);
  }

  float* seam_float = malloc((size_t)OUT_H * OUT_W * OUT_C * sizeof(float));
  double t10 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(programs[g]);
      GLint loc = glGetAttribLocation(programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, input_tex);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    for (int g = 0; g < GROUPS; ++g) {
      glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
      glReadPixels(0, 0, OUT_W, OUT_H, GL_RGBA, GL_UNSIGNED_BYTE, group_pixels);
      for (int p = 0; p < OUT_H * OUT_W; ++p) {
        for (int c = 0; c < 4; ++c) {
          seam_float[p * OUT_C + g * 4 + c] = (float)group_pixels[p * 4 + c] * (6.0f / 255.0f);
        }
      }
    }
  }
  glFinish();
  double t11 = now_s();

  double t12 = now_s();
  for (int r = 0; r < reps; ++r) {
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(programs[g]);
      GLint loc = glGetAttribLocation(programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, input_tex);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    for (int g = 0; g < GROUPS; ++g) {
      glUseProgram(dw_programs[g]);
      GLint loc = glGetAttribLocation(dw_programs[g], "pos");
      glEnableVertexAttribArray((GLuint)loc);
      glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
      glUniform1i(glGetUniformLocation(dw_programs[g], "src"), 0);
      glBindFramebuffer(GL_FRAMEBUFFER, dw_fbo[g]);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, out_tex[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    for (int g = 0; g < GROUPS; ++g) {
      glBindFramebuffer(GL_FRAMEBUFFER, dw_fbo[g]);
      glReadPixels(0, 0, OUT_W, OUT_H, GL_RGBA, GL_UNSIGNED_BYTE, group_pixels);
      for (int p = 0; p < OUT_H * OUT_W; ++p) {
        for (int c = 0; c < 4; ++c) {
          seam_float[p * OUT_C + g * 4 + c] = (float)group_pixels[p * 4 + c] * (6.0f / 255.0f);
        }
      }
    }
  }
  glFinish();
  double t13 = now_s();

  double mae = 0.0, qmae = 0.0;
  double mae_xflip = 0.0, mae_yflip = 0.0, mae_xyflip = 0.0;
  float max_abs = 0.0f, qmax_abs = 0.0f;
  float decoded_min = 999.0f, decoded_max = -999.0f, decoded_sum = 0.0f;
  float ref_min = 999.0f, ref_max = -999.0f, ref_sum = 0.0f;
  unsigned nonzero = 0;
  size_t n = (size_t)OUT_H * OUT_W * OUT_C;
  for (size_t i = 0; i < n; ++i) {
    float decoded = ((float)out[i]) * (6.0f / 255.0f);
    float qref = roundf(fminf(fmaxf(ref[i], 0.0f), 6.0f) * (255.0f / 6.0f)) * (6.0f / 255.0f);
    float d = fabsf(decoded - ref[i]);
    float qd = fabsf(decoded - qref);
    if (decoded < decoded_min) decoded_min = decoded;
    if (decoded > decoded_max) decoded_max = decoded;
    if (ref[i] < ref_min) ref_min = ref[i];
    if (ref[i] > ref_max) ref_max = ref[i];
    decoded_sum += decoded;
    ref_sum += ref[i];
    if (out[i]) ++nonzero;
    mae += d;
    qmae += qd;
    if (d > max_abs) max_abs = d;
    if (qd > qmax_abs) qmax_abs = qd;
  }
  for (int y = 0; y < OUT_H; ++y) {
    for (int x = 0; x < OUT_W; ++x) {
      for (int c = 0; c < OUT_C; ++c) {
        size_t oi = ((size_t)y * OUT_W + x) * OUT_C + c;
        float decoded = ((float)out[oi]) * (6.0f / 255.0f);
        size_t rx = ((size_t)y * OUT_W + (OUT_W - 1 - x)) * OUT_C + c;
        size_t ry = ((size_t)(OUT_H - 1 - y) * OUT_W + x) * OUT_C + c;
        size_t rxy = ((size_t)(OUT_H - 1 - y) * OUT_W + (OUT_W - 1 - x)) * OUT_C + c;
        mae_xflip += fabsf(decoded - ref[rx]);
        mae_yflip += fabsf(decoded - ref[ry]);
        mae_xyflip += fabsf(decoded - ref[rxy]);
      }
    }
  }
  double seconds = t1 - t0;
  double macs = (double)OUT_H * OUT_W * OUT_C * K_SIZE;
  printf("first_conv_gles2 reps=%d avg_ms=%.3f fps=%.2f effective_gmac_s=%.3f\n",
      reps, seconds * 1000.0 / (double)reps, (double)reps / seconds, macs * reps / seconds / 1e9);
  double dw_seconds = t3 - t2;
  double dw_macs = (double)OUT_H * OUT_W * OUT_C * K_H * K_W;
  printf("op6_depthwise_gles2 reps=%d avg_ms=%.3f fps=%.2f effective_gmac_s=%.3f\n",
      reps, dw_seconds * 1000.0 / (double)reps, (double)reps / dw_seconds,
      dw_macs * reps / dw_seconds / 1e9);
  double op9_seconds = t5 - t4;
  double op9_macs = (double)OUT_H * OUT_W * OUT_C * OP9_C;
  printf("op9_pointwise_gles2 reps=%d avg_ms=%.3f fps=%.2f effective_gmac_s=%.3f range=[%.6f, %.6f]\n",
      reps, op9_seconds * 1000.0 / (double)reps, (double)reps / op9_seconds,
      op9_macs * reps / op9_seconds / 1e9, op9_min, op9_max);
  double op12_seconds = t7 - t6;
  double op12_macs = (double)OUT_H * OUT_W * OP9_C * OP12_C;
  printf("op12_pointwise_gles2 reps=%d avg_ms=%.3f fps=%.2f effective_gmac_s=%.3f\n",
      reps, op12_seconds * 1000.0 / (double)reps, (double)reps / op12_seconds,
      op12_macs * reps / op12_seconds / 1e9);
  double prefix_seconds = t9 - t8;
  double prefix_macs = macs + dw_macs + op9_macs + op12_macs;
  printf("prefix_gles2 reps=%d avg_ms=%.3f fps=%.2f effective_gmac_s=%.3f macs=%.0f\n",
      reps, prefix_seconds * 1000.0 / (double)reps, (double)reps / prefix_seconds,
      prefix_macs * reps / prefix_seconds / 1e9, prefix_macs);
  printf("mixed_seam_gpu_op3_read_decode reps=%d avg_ms=%.3f fps=%.2f\n",
      reps, (t11 - t10) * 1000.0 / (double)reps, (double)reps / (t11 - t10));
  printf("mixed_seam_gpu_op3_op6_read_decode reps=%d avg_ms=%.3f fps=%.2f\n",
      reps, (t13 - t12) * 1000.0 / (double)reps, (double)reps / (t13 - t12));
  printf("decoded_vs_float_ref max_abs=%.6f mean_abs=%.6f\n", max_abs, mae / (double)n);
  printf("decoded_vs_quantized_ref max_abs=%.6f mean_abs=%.6f checksum=%u\n",
      qmax_abs, qmae / (double)n, out[12345]);
  printf("decoded_stats min=%.6f max=%.6f mean=%.6f nonzero=%u/%zu ref_min=%.6f ref_max=%.6f ref_mean=%.6f\n",
      decoded_min, decoded_max, decoded_sum / (float)n, nonzero, n,
      ref_min, ref_max, ref_sum / (float)n);
  printf("layout_probe mean_abs xflip=%.6f yflip=%.6f xyflip=%.6f\n",
      mae_xflip / (double)n, mae_yflip / (double)n, mae_xyflip / (double)n);

  double dw_mae = 0.0, dw_qmae = 0.0;
  float dw_max_abs = 0.0f, dw_qmax_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float decoded = ((float)dw_out[i]) * (6.0f / 255.0f);
    float qref = roundf(fminf(fmaxf(dw_ref[i], 0.0f), 6.0f) * (255.0f / 6.0f)) * (6.0f / 255.0f);
    float d = fabsf(decoded - dw_ref[i]);
    float qd = fabsf(decoded - qref);
    dw_mae += d;
    dw_qmae += qd;
    if (d > dw_max_abs) dw_max_abs = d;
    if (qd > dw_qmax_abs) dw_qmax_abs = qd;
  }
  printf("op6_decoded_vs_qop3_float_ref max_abs=%.6f mean_abs=%.6f\n",
      dw_max_abs, dw_mae / (double)n);
  printf("op6_decoded_vs_qop3_quantized_ref max_abs=%.6f mean_abs=%.6f checksum=%u\n",
      dw_qmax_abs, dw_qmae / (double)n, dw_out[12345]);

  double op9_mae = 0.0, op9_qmae = 0.0;
  float op9_max_abs = 0.0f, op9_qmax_abs = 0.0f;
  size_t op9_n = (size_t)OUT_H * OUT_W * OP9_C;
  for (size_t i = 0; i < op9_n; ++i) {
    float decoded = ((float)op9_out[i]) * (op9_range / 255.0f) + op9_min;
    float q = roundf(fminf(fmaxf((op9_ref[i] - op9_min) / op9_range, 0.0f), 1.0f) * 255.0f);
    float qref = q * (op9_range / 255.0f) + op9_min;
    float d = fabsf(decoded - op9_ref[i]);
    float qd = fabsf(decoded - qref);
    op9_mae += d;
    op9_qmae += qd;
    if (d > op9_max_abs) op9_max_abs = d;
    if (qd > op9_qmax_abs) op9_qmax_abs = qd;
  }
  printf("op9_decoded_vs_qop6_float_ref max_abs=%.6f mean_abs=%.6f\n",
      op9_max_abs, op9_mae / (double)op9_n);
  printf("op9_decoded_vs_qop6_quantized_ref max_abs=%.6f mean_abs=%.6f checksum=%u\n",
      op9_qmax_abs, op9_qmae / (double)op9_n, op9_out[12345]);

  double op12_mae = 0.0, op12_qmae = 0.0;
  float op12_max_abs = 0.0f, op12_qmax_abs = 0.0f;
  size_t op12_n = (size_t)OUT_H * OUT_W * OP12_C;
  for (size_t i = 0; i < op12_n; ++i) {
    float decoded = ((float)op12_out[i]) * (6.0f / 255.0f);
    float qref = roundf(fminf(fmaxf(op12_ref[i], 0.0f), 6.0f) * (255.0f / 6.0f)) * (6.0f / 255.0f);
    float d = fabsf(decoded - op12_ref[i]);
    float qd = fabsf(decoded - qref);
    op12_mae += d;
    op12_qmae += qd;
    if (d > op12_max_abs) op12_max_abs = d;
    if (qd > op12_qmax_abs) op12_qmax_abs = qd;
  }
  printf("op12_decoded_vs_qop9_float_ref max_abs=%.6f mean_abs=%.6f\n",
      op12_max_abs, op12_mae / (double)op12_n);
  printf("op12_decoded_vs_qop9_quantized_ref max_abs=%.6f mean_abs=%.6f checksum=%u\n",
      op12_qmax_abs, op12_qmae / (double)op12_n, op12_out[12345]);

  eglDestroyContext(display, ctx);
  eglDestroySurface(display, surface);
  eglTerminate(display);
  return 0;
}
