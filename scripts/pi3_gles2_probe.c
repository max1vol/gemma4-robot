// Minimal EGL/GLES2 probe for Raspberry Pi 3B+ without relying on dev headers.
// It creates an offscreen context, runs a fragment shader many times into an
// FBO, and reports draw + readback timing. This is a primitive compute-path
// smoke test for using VideoCore IV through GLES2 under vc4-kms-v3d.

#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
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

#define EGL_DEFAULT_DISPLAY ((EGLNativeDisplayType)0)
#define EGL_NO_DISPLAY ((EGLDisplay)0)
#define EGL_NO_CONTEXT ((EGLContext)0)
#define EGL_NO_SURFACE ((EGLSurface)0)
#define EGL_FALSE 0
#define EGL_TRUE 1
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
#define EGL_EXTENSIONS 0x3055
#define EGL_VENDOR 0x3053
#define EGL_VERSION 0x3054
#define EGL_PLATFORM_SURFACELESS_MESA 0x31DD

#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_ARRAY_BUFFER 0x8892
#define GL_STATIC_DRAW 0x88E4
#define GL_FLOAT 0x1406
#define GL_FALSE 0
#define GL_TRIANGLE_STRIP 0x0005
#define GL_COLOR_BUFFER_BIT 0x00004000
#define GL_TEXTURE_2D 0x0DE1
#define GL_TEXTURE0 0x84C0
#define GL_RGBA 0x1908
#define GL_UNSIGNED_BYTE 0x1401
#define GL_TEXTURE_MIN_FILTER 0x2801
#define GL_TEXTURE_MAG_FILTER 0x2800
#define GL_NEAREST 0x2600
#define GL_FRAMEBUFFER 0x8D40
#define GL_COLOR_ATTACHMENT0 0x8CE0
#define GL_FRAMEBUFFER_COMPLETE 0x8CD5
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_EXTENSIONS 0x1F03

#define EGL_PROC(ret, name, args) static ret (*name) args
#define GL_PROC(ret, name, args) static ret (*name) args

EGL_PROC(EGLDisplay, eglGetDisplay, (EGLNativeDisplayType));
EGL_PROC(EGLint, eglInitialize, (EGLDisplay, EGLint*, EGLint*));
EGL_PROC(const char*, eglQueryString, (EGLDisplay, EGLint));
EGL_PROC(EGLint, eglBindAPI, (EGLint));
EGL_PROC(EGLint, eglChooseConfig, (EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*));
EGL_PROC(EGLSurface, eglCreatePbufferSurface, (EGLDisplay, EGLConfig, const EGLint*));
EGL_PROC(EGLContext, eglCreateContext, (EGLDisplay, EGLConfig, EGLContext, const EGLint*));
EGL_PROC(EGLint, eglMakeCurrent, (EGLDisplay, EGLSurface, EGLSurface, EGLContext));
EGL_PROC(EGLint, eglSwapBuffers, (EGLDisplay, EGLSurface));
EGL_PROC(EGLint, eglDestroyContext, (EGLDisplay, EGLContext));
EGL_PROC(EGLint, eglDestroySurface, (EGLDisplay, EGLSurface));
EGL_PROC(EGLint, eglTerminate, (EGLDisplay));
EGL_PROC(EGLint, eglGetError, (void));
EGL_PROC(void*, eglGetProcAddress, (const char*));
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLint, void*, const EGLint*);

GL_PROC(const unsigned char*, glGetString, (GLenum));
GL_PROC(void, glViewport, (GLint, GLint, GLsizei, GLsizei));
GL_PROC(void, glClearColor, (GLfloat, GLfloat, GLfloat, GLfloat));
GL_PROC(void, glClear, (GLbitfield));
GL_PROC(GLuint, glCreateShader, (GLenum));
GL_PROC(void, glShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*));
GL_PROC(void, glCompileShader, (GLuint));
GL_PROC(void, glGetShaderiv, (GLuint, GLenum, GLint*));
GL_PROC(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*));
GL_PROC(GLuint, glCreateProgram, (void));
GL_PROC(void, glAttachShader, (GLuint, GLuint));
GL_PROC(void, glLinkProgram, (GLuint));
GL_PROC(void, glGetProgramiv, (GLuint, GLenum, GLint*));
GL_PROC(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*));
GL_PROC(void, glUseProgram, (GLuint));
GL_PROC(void, glGenBuffers, (GLsizei, GLuint*));
GL_PROC(void, glBindBuffer, (GLenum, GLuint));
GL_PROC(void, glBufferData, (GLenum, intptr_t, const void*, GLenum));
GL_PROC(GLint, glGetAttribLocation, (GLuint, const GLchar*));
GL_PROC(GLint, glGetUniformLocation, (GLuint, const GLchar*));
GL_PROC(void, glUniform1i, (GLint, GLint));
GL_PROC(void, glUniform1f, (GLint, GLfloat));
GL_PROC(void, glEnableVertexAttribArray, (GLuint));
GL_PROC(void, glVertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const void*));
GL_PROC(void, glDrawArrays, (GLenum, GLint, GLsizei));
GL_PROC(void, glFinish, (void));
GL_PROC(void, glGenTextures, (GLsizei, GLuint*));
GL_PROC(void, glActiveTexture, (GLenum));
GL_PROC(void, glBindTexture, (GLenum, GLuint));
GL_PROC(void, glTexParameteri, (GLenum, GLenum, GLint));
GL_PROC(void, glTexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*));
GL_PROC(void, glGenFramebuffers, (GLsizei, GLuint*));
GL_PROC(void, glBindFramebuffer, (GLenum, GLuint));
GL_PROC(void, glFramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint));
GL_PROC(GLenum, glCheckFramebufferStatus, (GLenum));
GL_PROC(void, glReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*));
GL_PROC(void, glDeleteProgram, (GLuint));
GL_PROC(void, glDeleteShader, (GLuint));

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void* must_sym(void* lib, const char* name) {
  void* p = dlsym(lib, name);
  if (!p) {
    fprintf(stderr, "missing symbol %s\n", name);
    exit(2);
  }
  return p;
}

#define LOAD_EGL(lib, name) name = must_sym(lib, #name)
#define LOAD_GL(lib, name) name = must_sym(lib, #name)

static void load_symbols(void) {
  void* egl = dlopen("libEGL.so.1", RTLD_NOW | RTLD_LOCAL);
  void* gles = dlopen("libGLESv2.so.2", RTLD_NOW | RTLD_LOCAL);
  if (!egl || !gles) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    exit(2);
  }
  LOAD_EGL(egl, eglGetDisplay);
  LOAD_EGL(egl, eglInitialize);
  LOAD_EGL(egl, eglQueryString);
  LOAD_EGL(egl, eglBindAPI);
  LOAD_EGL(egl, eglChooseConfig);
  LOAD_EGL(egl, eglCreatePbufferSurface);
  LOAD_EGL(egl, eglCreateContext);
  LOAD_EGL(egl, eglMakeCurrent);
  LOAD_EGL(egl, eglSwapBuffers);
  LOAD_EGL(egl, eglDestroyContext);
  LOAD_EGL(egl, eglDestroySurface);
  LOAD_EGL(egl, eglTerminate);
  LOAD_EGL(egl, eglGetError);
  LOAD_EGL(egl, eglGetProcAddress);

  LOAD_GL(gles, glGetString);
  LOAD_GL(gles, glViewport);
  LOAD_GL(gles, glClearColor);
  LOAD_GL(gles, glClear);
  LOAD_GL(gles, glCreateShader);
  LOAD_GL(gles, glShaderSource);
  LOAD_GL(gles, glCompileShader);
  LOAD_GL(gles, glGetShaderiv);
  LOAD_GL(gles, glGetShaderInfoLog);
  LOAD_GL(gles, glCreateProgram);
  LOAD_GL(gles, glAttachShader);
  LOAD_GL(gles, glLinkProgram);
  LOAD_GL(gles, glGetProgramiv);
  LOAD_GL(gles, glGetProgramInfoLog);
  LOAD_GL(gles, glUseProgram);
  LOAD_GL(gles, glGenBuffers);
  LOAD_GL(gles, glBindBuffer);
  LOAD_GL(gles, glBufferData);
  LOAD_GL(gles, glGetAttribLocation);
  LOAD_GL(gles, glGetUniformLocation);
  LOAD_GL(gles, glUniform1i);
  LOAD_GL(gles, glUniform1f);
  LOAD_GL(gles, glEnableVertexAttribArray);
  LOAD_GL(gles, glVertexAttribPointer);
  LOAD_GL(gles, glDrawArrays);
  LOAD_GL(gles, glFinish);
  LOAD_GL(gles, glGenTextures);
  LOAD_GL(gles, glActiveTexture);
  LOAD_GL(gles, glBindTexture);
  LOAD_GL(gles, glTexParameteri);
  LOAD_GL(gles, glTexImage2D);
  LOAD_GL(gles, glGenFramebuffers);
  LOAD_GL(gles, glBindFramebuffer);
  LOAD_GL(gles, glFramebufferTexture2D);
  LOAD_GL(gles, glCheckFramebufferStatus);
  LOAD_GL(gles, glReadPixels);
  LOAD_GL(gles, glDeleteProgram);
  LOAD_GL(gles, glDeleteShader);
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
    fprintf(stderr, "shader compile failed: %.*s\n", (int)n, log);
    exit(3);
  }
  return shader;
}

static GLuint make_program(int ops) {
  const char* vs =
    "attribute vec2 pos;\n"
    "varying vec2 uv;\n"
    "void main() { uv = pos * 0.5 + 0.5; gl_Position = vec4(pos, 0.0, 1.0); }\n";

  char fs[8192];
  snprintf(fs, sizeof(fs),
    "precision highp float;\n"
    "uniform sampler2D src;\n"
    "uniform float seed;\n"
    "varying vec2 uv;\n"
    "void main() {\n"
    "  vec4 x = texture2D(src, uv) + vec4(seed * 0.000001);\n"
    "  vec4 y = vec4(0.31, 0.37, 0.41, 0.43) + x.wxyz * 0.01;\n"
    "  for (int i = 0; i < %d; ++i) {\n"
    "    x = x * y + x.yzwx * vec4(0.001, 0.002, 0.003, 0.004);\n"
    "    y = y * x + y.wxyz * vec4(0.005, 0.006, 0.007, 0.008);\n"
    "  }\n"
    "  gl_FragColor = x + y;\n"
    "}\n", ops);

  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096];
    GLsizei n = 0;
    glGetProgramInfoLog(p, sizeof(log), &n, log);
    fprintf(stderr, "program link failed: %.*s\n", (int)n, log);
    exit(3);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  return p;
}

int main(int argc, char** argv) {
  int width = argc > 1 ? atoi(argv[1]) : 512;
  int height = argc > 2 ? atoi(argv[2]) : 512;
  int iters = argc > 3 ? atoi(argv[3]) : 200;
  int shader_ops = argc > 4 ? atoi(argv[4]) : 32;

  load_symbols();
  EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  EGLint major = 0, minor = 0;
  if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!get_platform) {
      fprintf(stderr, "no default EGL display and no eglGetPlatformDisplayEXT\n");
      return 2;
    }
    display = get_platform(EGL_PLATFORM_SURFACELESS_MESA, NULL, NULL);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
      fprintf(stderr, "eglInitialize failed: 0x%x\n", eglGetError());
      return 2;
    }
  }

  printf("EGL %d.%d vendor=%s version=%s\n", major, minor,
         eglQueryString(display, EGL_VENDOR), eglQueryString(display, EGL_VERSION));
  eglBindAPI(EGL_OPENGL_ES_API);

  EGLint cfg_attr[] = {
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
    EGL_DEPTH_SIZE, 0,
    EGL_NONE
  };
  EGLConfig cfg = NULL;
  EGLint num = 0;
  if (!eglChooseConfig(display, cfg_attr, &cfg, 1, &num) || num < 1) {
    fprintf(stderr, "eglChooseConfig failed: 0x%x\n", eglGetError());
    return 2;
  }
  EGLint pb_attr[] = {EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE};
  EGLSurface surface = eglCreatePbufferSurface(display, cfg, pb_attr);
  EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
  EGLContext ctx = eglCreateContext(display, cfg, EGL_NO_CONTEXT, ctx_attr);
  if (surface == EGL_NO_SURFACE || ctx == EGL_NO_CONTEXT ||
      !eglMakeCurrent(display, surface, surface, ctx)) {
    fprintf(stderr, "context creation failed: 0x%x\n", eglGetError());
    return 2;
  }

  printf("GL vendor=%s\nGL renderer=%s\nGL version=%s\nGL extensions=%s\n",
         glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION),
         glGetString(GL_EXTENSIONS));

  GLuint tex[2] = {0, 0}, fbo[2] = {0, 0};
  uint8_t* seed_pixels = malloc((size_t)width * (size_t)height * 4);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      size_t off = ((size_t)y * (size_t)width + (size_t)x) * 4;
      seed_pixels[off + 0] = (uint8_t)(x + y);
      seed_pixels[off + 1] = (uint8_t)(x * 3);
      seed_pixels[off + 2] = (uint8_t)(y * 5);
      seed_pixels[off + 3] = 255;
    }
  }
  glGenTextures(2, tex);
  glGenFramebuffers(2, fbo);
  for (int i = 0; i < 2; ++i) {
    glBindTexture(GL_TEXTURE_2D, tex[i]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, seed_pixels);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[i]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex[i], 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      fprintf(stderr, "incomplete framebuffer %d\n", i);
      return 2;
    }
  }
  free(seed_pixels);

  GLuint prog = make_program(shader_ops);
  glUseProgram(prog);
  GLint src_loc = glGetUniformLocation(prog, "src");
  GLint seed_loc = glGetUniformLocation(prog, "seed");
  glUniform1i(src_loc, 0);
  GLfloat verts[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
  GLuint vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
  GLint loc = glGetAttribLocation(prog, "pos");
  glEnableVertexAttribArray((GLuint)loc);
  glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glViewport(0, 0, width, height);

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  glFinish();

  double t0 = now_s();
  for (int i = 0; i < iters; ++i) {
    int src = i & 1;
    int dst = 1 - src;
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[dst]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex[src]);
    glUniform1f(seed_loc, (GLfloat)i);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }
  glFinish();
  double t1 = now_s();

  uint8_t* pixels = malloc((size_t)width * (size_t)height * 4);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo[iters & 1]);
  double r0 = now_s();
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
  glFinish();
  double r1 = now_s();
  unsigned checksum = 0;
  for (size_t i = 0; i < (size_t)width * (size_t)height * 4; i += 4096) checksum += pixels[i];
  free(pixels);

  double draw_s = t1 - t0;
  double px = (double)width * (double)height * (double)iters;
  // Each loop iteration does two vec4 multiply-adds: 8 mul + 8 add = 16 scalar FLOPs.
  double gflops_nominal = px * (double)shader_ops * 16.0 / draw_s / 1e9;
  printf("draw width=%d height=%d iters=%d shader_ops=%d seconds=%.6f frames_per_s=%.2f nominal_gflops=%.3f\n",
         width, height, iters, shader_ops, draw_s, (double)iters / draw_s, gflops_nominal);
  printf("readback seconds=%.6f MBps=%.2f checksum=%u\n", r1 - r0,
         ((double)width * (double)height * 4.0 / 1e6) / (r1 - r0), checksum);

  eglDestroyContext(display, ctx);
  eglDestroySurface(display, surface);
  eglTerminate(display);
  return 0;
}
