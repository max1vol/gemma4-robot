// Minimal GLES2 texture sampling correctness probe for Raspberry Pi 3B+ VC4.
// Uploads the first-conv random input as RGBA8 and renders a 128x128 output
// sampling input pixel (2*x, 2*y). Reports error against the uploaded bytes.

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
#define GL_NO_ERROR 0

#define PROC(ret, name, args) static ret (*name) args
PROC(EGLDisplay, eglGetDisplay, (EGLNativeDisplayType));
PROC(EGLint, eglInitialize, (EGLDisplay, EGLint*, EGLint*));
PROC(EGLint, eglBindAPI, (EGLint));
PROC(EGLint, eglChooseConfig, (EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*));
PROC(EGLSurface, eglCreatePbufferSurface, (EGLDisplay, EGLConfig, const EGLint*));
PROC(EGLContext, eglCreateContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint*));
PROC(EGLint, eglMakeCurrent, (EGLDisplay, EGLSurface, EGLSurface, EGLContext));
PROC(EGLint, eglGetError, (void));
PROC(void*, eglGetProcAddress, (const char*));
PROC(EGLint, eglTerminate, (EGLDisplay));
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLint, void*, const EGLint*);

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
PROC(void, glViewport, (GLint, GLint, GLsizei, GLsizei));
PROC(GLenum, glGetError, (void));

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
  LOAD(egl, eglGetDisplay); LOAD(egl, eglInitialize); LOAD(egl, eglBindAPI);
  LOAD(egl, eglChooseConfig); LOAD(egl, eglCreatePbufferSurface);
  LOAD(egl, eglCreateContext); LOAD(egl, eglMakeCurrent); LOAD(egl, eglGetError);
  LOAD(egl, eglGetProcAddress); LOAD(egl, eglTerminate);
  LOAD(gl, glCreateShader); LOAD(gl, glShaderSource); LOAD(gl, glCompileShader);
  LOAD(gl, glGetShaderiv); LOAD(gl, glGetShaderInfoLog); LOAD(gl, glCreateProgram);
  LOAD(gl, glAttachShader); LOAD(gl, glLinkProgram); LOAD(gl, glGetProgramiv);
  LOAD(gl, glGetProgramInfoLog); LOAD(gl, glUseProgram); LOAD(gl, glGenBuffers);
  LOAD(gl, glBindBuffer); LOAD(gl, glBufferData); LOAD(gl, glGetAttribLocation);
  LOAD(gl, glGetUniformLocation); LOAD(gl, glUniform1i); LOAD(gl, glEnableVertexAttribArray);
  LOAD(gl, glVertexAttribPointer); LOAD(gl, glDrawArrays); LOAD(gl, glFinish);
  LOAD(gl, glGenTextures); LOAD(gl, glActiveTexture); LOAD(gl, glBindTexture);
  LOAD(gl, glTexParameteri); LOAD(gl, glTexImage2D); LOAD(gl, glGenFramebuffers);
  LOAD(gl, glBindFramebuffer); LOAD(gl, glFramebufferTexture2D);
  LOAD(gl, glCheckFramebufferStatus); LOAD(gl, glReadPixels); LOAD(gl, glViewport);
  LOAD(gl, glGetError);
}

static GLuint compile_shader(GLenum type, const char* src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096];
    GLsizei n = 0;
    glGetShaderInfoLog(shader, sizeof(log), &n, log);
    fprintf(stderr, "shader compile failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  return shader;
}

static GLuint make_program(void) {
  const char* vs =
      "attribute vec2 pos;\n"
      "void main() { gl_Position = vec4(pos, 0.0, 1.0); }\n";
  const char* fs =
      "precision highp float;\n"
      "uniform sampler2D src;\n"
      "void main() {\n"
      "  float ox = gl_FragCoord.x;\n"
      "  float oy = gl_FragCoord.y;\n"
      "  vec2 uv = vec2((2.0 * ox + 0.5) / 256.0, (2.0 * oy + 0.5) / 256.0);\n"
      "  gl_FragColor = texture2D(src, uv);\n"
      "}\n";
  GLuint program = glCreateProgram();
  glAttachShader(program, compile_shader(GL_VERTEX_SHADER, vs));
  glAttachShader(program, compile_shader(GL_FRAGMENT_SHADER, fs));
  glLinkProgram(program);
  GLint ok = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096];
    GLsizei n = 0;
    glGetProgramInfoLog(program, sizeof(log), &n, log);
    fprintf(stderr, "link failed:\n%.*s\n", (int)n, log);
    exit(3);
  }
  return program;
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

static void check_gl(const char* label) {
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    fprintf(stderr, "GL error after %s: 0x%x\n", label, err);
    exit(3);
  }
}

static void setup_egl(EGLDisplay* d) {
  *d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLint major = 0, minor = 0;
  if (*d == EGL_NO_DISPLAY || !eglInitialize(*d, &major, &minor)) {
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    *d = get_platform(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
    if (*d == EGL_NO_DISPLAY || !eglInitialize(*d, &major, &minor)) {
      fprintf(stderr, "eglInitialize failed: 0x%x\n", eglGetError());
      exit(2);
    }
  }
  eglBindAPI(EGL_OPENGL_ES_API);

  EGLint cfg_attr[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
      EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
      EGL_DEPTH_SIZE, 0,
      EGL_NONE};
  EGLConfig cfg = NULL;
  EGLint num = 0;
  if (!eglChooseConfig(*d, cfg_attr, &cfg, 1, &num) || num < 1) {
    fprintf(stderr, "eglChooseConfig failed: 0x%x\n", eglGetError());
    exit(2);
  }

  EGLint pb_attr[] = {EGL_WIDTH, 128, EGL_HEIGHT, 128, EGL_NONE};
  EGLSurface surface = eglCreatePbufferSurface(*d, cfg, pb_attr);
  EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  EGLContext ctx = eglCreateContext(*d, cfg, EGL_NO_CONTEXT, ctx_attr);
  if (surface == EGL_NO_SURFACE || ctx == EGL_NO_CONTEXT ||
      !eglMakeCurrent(*d, surface, surface, ctx)) {
    fprintf(stderr, "context creation failed: 0x%x\n", eglGetError());
    exit(2);
  }
}

static const uint8_t* expected_pixel(const uint8_t* rgba, int x, int y, int variant) {
  int sx = 2 * x;
  int sy = 2 * y;
  if (variant & 1) sx = 255 - sx;
  if (variant & 2) sy = 255 - sy;
  return rgba + ((size_t)sy * 256 + sx) * 4;
}

static double measure_mae(const uint8_t* out, const uint8_t* rgba, int variant, double* maxe) {
  double sum = 0.0;
  *maxe = 0.0;
  for (int y = 0; y < 128; ++y) {
    for (int x = 0; x < 128; ++x) {
      const uint8_t* exp = expected_pixel(rgba, x, y, variant);
      const uint8_t* got = out + ((size_t)y * 128 + x) * 4;
      for (int c = 0; c < 3; ++c) {
        double e = fabs((double)got[c] - (double)exp[c]);
        sum += e;
        if (e > *maxe) *maxe = e;
      }
    }
  }
  return sum / (128.0 * 128.0 * 3.0);
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  char path[512];
  float* input = malloc(256 * 256 * 3 * sizeof(float));
  uint8_t* rgba = malloc(256 * 256 * 4);
  uint8_t* out = malloc(128 * 128 * 4);
  if (!input || !rgba || !out) return 2;

  snprintf(path, sizeof(path), "%s/input_f32.bin", dir);
  read_exact(path, input, 256 * 256 * 3 * sizeof(float));
  for (int i = 0; i < 256 * 256; ++i) {
    rgba[4 * i + 0] = (uint8_t)lrintf(fminf(fmaxf(input[3 * i + 0], 0.0f), 1.0f) * 255.0f);
    rgba[4 * i + 1] = (uint8_t)lrintf(fminf(fmaxf(input[3 * i + 1], 0.0f), 1.0f) * 255.0f);
    rgba[4 * i + 2] = (uint8_t)lrintf(fminf(fmaxf(input[3 * i + 2], 0.0f), 1.0f) * 255.0f);
    rgba[4 * i + 3] = 255;
  }

  load_symbols();
  EGLDisplay display;
  setup_egl(&display);
  GLuint program = make_program();

  GLuint input_tex = 0, out_tex = 0, fbo = 0;
  glGenTextures(1, &input_tex);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, input_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
  check_gl("input texture upload");

  glGenTextures(1, &out_tex);
  glBindTexture(GL_TEXTURE_2D, out_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 128, 128, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_tex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "bad fbo\n");
    return 2;
  }
  check_gl("output framebuffer");

  const GLfloat verts[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
  GLuint vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  glUseProgram(program);
  GLint loc = glGetAttribLocation(program, "pos");
  glEnableVertexAttribArray((GLuint)loc);
  glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glUniform1i(glGetUniformLocation(program, "src"), 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, input_tex);
  glViewport(0, 0, 128, 128);
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  glFinish();
  check_gl("draw");

  glReadPixels(0, 0, 128, 128, GL_RGBA, GL_UNSIGNED_BYTE, out);
  check_gl("readback");

  const char* names[] = {"same", "xflip", "yflip", "xyflip"};
  for (int variant = 0; variant < 4; ++variant) {
    double maxe = 0.0;
    double mae = measure_mae(out, rgba, variant, &maxe);
    printf("texture_probe variant=%s mae_u8=%.6f max_u8=%.6f\n",
        names[variant], mae, maxe);
  }

  const int sample_xy[][2] = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {17, 33}, {127, 127}};
  for (size_t i = 0; i < sizeof(sample_xy) / sizeof(sample_xy[0]); ++i) {
    int x = sample_xy[i][0];
    int y = sample_xy[i][1];
    const uint8_t* got = out + ((size_t)y * 128 + x) * 4;
    const uint8_t* exp = expected_pixel(rgba, x, y, 0);
    printf("sample x=%d y=%d got=%u/%u/%u/%u expected=%u/%u/%u/%u\n",
        x, y, got[0], got[1], got[2], got[3], exp[0], exp[1], exp[2], exp[3]);
  }

  snprintf(path, sizeof(path), "%s/gles_texture_probe_out_u8.bin", dir);
  FILE* dump = fopen(path, "wb");
  if (dump) {
    fwrite(out, 1, 128 * 128 * 4, dump);
    fclose(dump);
  }

  eglTerminate(display);
  return 0;
}
