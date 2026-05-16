// GLES2 fragment-shader benchmark for the second Pose Landmarker Lite block.
//
// This standalone probe uploads extracted op9/op12 reference tensors as RGBA8
// textures and runs:
//   op15 depthwise 8ch, op18 pointwise 8->8, op19 add ReLU6,
//   op23 stride-2 depthwise 32ch, op26 pointwise 32->16.
//
// It intentionally includes the prefix GLES2 probe with main renamed so we can
// reuse the same headerless EGL/GLES loader and utility functions on the Pi.

#define main pose_prefix_unused_main
#include "pose_first_conv_gles2.c"
#undef main

enum {
  B2_H128 = 128,
  B2_W128 = 128,
  B2_H64 = 64,
  B2_W64 = 64,
  B2_C8 = 8,
  B2_C16 = 16,
  B2_C32 = 32,
};

static GLuint make_depthwise8_affine_to_relu6(const float* w, const float* bias, int group, float q_min, float q_range) {
  const char* vs = "attribute vec2 pos;void main(){gl_Position=vec4(pos,0.0,1.0);}";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D s0;uniform sampler2D s1;\n"
      "vec4 sample_at(float ix,float iy){\n"
      " if(ix<0.0||ix>=128.0||iy<0.0||iy>=128.0)return vec4(0.0);\n"
      " vec2 uv=vec2((ix+0.5)*%.12g,(iy+0.5)*%.12g);\n"
      " vec4 q=texture2D(%s,uv);\n"
      " return q*%.9g + (%.9g);\n"
      "}\n"
      "void main(){float ox=gl_FragCoord.x;float oy=gl_FragCoord.y;vec4 acc=vec4(%.9g,%.9g,%.9g,%.9g);\n",
      1.0 / 128.0, 1.0 / 128.0, group == 0 ? "s0" : "s1", q_range, q_min,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      const char* dx = kx == 0 ? " - 1.0" : (kx == 1 ? "" : " + 1.0");
      const char* dy = ky == 0 ? " - 1.0" : (ky == 1 ? "" : " + 1.0");
      int base = (ky * 3 + kx) * B2_C8 + group * 4;
      appendf(&fs,
          "acc+=sample_at(ox%s,oy%s)*vec4(%.9g,%.9g,%.9g,%.9g);\n",
          dx, dy,
          w[base + 0], w[base + 1], w[base + 2], w[base + 3]);
    }
  }
  appendf(&fs, "gl_FragColor=clamp(acc,0.0,6.0)*%.12g;}\n", 1.0 / 6.0);
  if (group == 0) {
    FILE* dbg = fopen("/tmp/pose_b2_op15_shader.glsl", "wb");
    if (dbg) {
      fwrite(fs.data, 1, fs.len, dbg);
      fclose(dbg);
    }
  }
  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[8192]; GLsizei n = 0; glGetProgramInfoLog(p, sizeof(log), &n, log); fprintf(stderr, "op15 link: %.*s\n", (int)n, log); exit(3); }
  free(fs.data);
  return p;
}

static GLuint make_pointwise8_relu6_to_affine(const float* w, const float* bias, int group, float q_min, float q_range) {
  const char* vs = "attribute vec2 pos;void main(){gl_Position=vec4(pos,0.0,1.0);}";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D s0;uniform sampler2D s1;\n"
      "void main(){vec2 uv=vec2((gl_FragCoord.x+0.5)*%.12g,(gl_FragCoord.y+0.5)*%.12g);"
      "vec4 v0=texture2D(s0,uv)*6.0;vec4 v1=texture2D(s1,uv)*6.0;"
      "vec4 acc=vec4(%.9g,%.9g,%.9g,%.9g);\n",
      1.0 / 128.0, 1.0 / 128.0,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ci = 0; ci < B2_C8; ++ci) {
    const char* src = ci < 4 ? "v0" : "v1";
    const char chan = (ci & 3) == 0 ? 'r' : (ci & 3) == 1 ? 'g' : (ci & 3) == 2 ? 'b' : 'a';
    float ww[4];
    for (int lane = 0; lane < 4; ++lane) ww[lane] = w[(group * 4 + lane) * B2_C8 + ci];
    appendf(&fs, "acc+=vec4(%.9g,%.9g,%.9g,%.9g)*%s.%c;\n", ww[0], ww[1], ww[2], ww[3], src, chan);
  }
  appendf(&fs, "gl_FragColor=clamp((acc-%.9g)*%.12g,0.0,1.0);}\n", q_min, 1.0 / q_range);
  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[8192]; GLsizei n = 0; glGetProgramInfoLog(p, sizeof(log), &n, log); fprintf(stderr, "op18 link: %.*s\n", (int)n, log); exit(3); }
  free(fs.data);
  return p;
}

static GLuint make_add8_affine_relu6(float a_min, float a_range, float b_min, float b_range, int group) {
  (void)group;
  const char* vs = "attribute vec2 pos;void main(){gl_Position=vec4(pos,0.0,1.0);}";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D a;uniform sampler2D b;\n"
      "void main(){vec2 uv=vec2((gl_FragCoord.x+0.5)*%.12g,(gl_FragCoord.y+0.5)*%.12g);"
      "vec4 va=texture2D(a,uv)*%.9g + (%.9g);"
      "vec4 vb=texture2D(b,uv)*%.9g + (%.9g);"
      "gl_FragColor=clamp(va+vb,0.0,6.0)*%.12g;}\n",
      1.0 / 128.0, 1.0 / 128.0, a_range, a_min, b_range, b_min, 1.0 / 6.0);
  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[8192]; GLsizei n = 0; glGetProgramInfoLog(p, sizeof(log), &n, log); fprintf(stderr, "op19 link: %.*s\n", (int)n, log); exit(3); }
  free(fs.data);
  return p;
}

static GLuint make_depthwise32_stride2_relu6(const float* w, const float* bias, int group) {
  const char* vs = "attribute vec2 pos;void main(){gl_Position=vec4(pos,0.0,1.0);}";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D src;\n"
      "vec4 sample_at(float ix,float iy){if(ix<0.0||ix>=128.0||iy<0.0||iy>=128.0)return vec4(0.0);"
      "return texture2D(src,vec2((ix+0.5)*%.12g,(iy+0.5)*%.12g))*6.0;}\n"
      "void main(){float ox=gl_FragCoord.x;float oy=gl_FragCoord.y;vec4 acc=vec4(%.9g,%.9g,%.9g,%.9g);\n",
      1.0 / 128.0, 1.0 / 128.0,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
      const char* dx = kx == 0 ? " - 1.0" : (kx == 1 ? "" : " + 1.0");
      const char* dy = ky == 0 ? " - 1.0" : (ky == 1 ? "" : " + 1.0");
      int base = (ky * 3 + kx) * B2_C32 + group * 4;
      appendf(&fs,
          "acc+=sample_at(ox * 2.0%s,oy * 2.0%s)*vec4(%.9g,%.9g,%.9g,%.9g);\n",
          dx, dy,
          w[base + 0], w[base + 1], w[base + 2], w[base + 3]);
    }
  }
  appendf(&fs, "gl_FragColor=clamp(acc,0.0,6.0)*%.12g;}\n", 1.0 / 6.0);
  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[8192]; GLsizei n = 0; glGetProgramInfoLog(p, sizeof(log), &n, log); fprintf(stderr, "op23 link: %.*s\n", (int)n, log); exit(3); }
  free(fs.data);
  return p;
}

static GLuint make_pointwise32_relu6_to_affine(const float* w, const float* bias, int group, float q_min, float q_range) {
  const char* vs = "attribute vec2 pos;void main(){gl_Position=vec4(pos,0.0,1.0);}";
  Str fs = {0};
  appendf(&fs,
      "precision highp float;\n"
      "uniform sampler2D s0;uniform sampler2D s1;uniform sampler2D s2;uniform sampler2D s3;"
      "uniform sampler2D s4;uniform sampler2D s5;uniform sampler2D s6;uniform sampler2D s7;\n"
      "void main(){vec2 uv=vec2((gl_FragCoord.x+0.5)*%.12g,(gl_FragCoord.y+0.5)*%.12g);"
      "vec4 acc=vec4(%.9g,%.9g,%.9g,%.9g);\n",
      1.0 / 64.0, 1.0 / 64.0,
      bias[group * 4 + 0], bias[group * 4 + 1], bias[group * 4 + 2], bias[group * 4 + 3]);
  for (int ig = 0; ig < 8; ++ig) {
    appendf(&fs, "{vec4 v=texture2D(s%d,uv)*6.0;\n", ig);
    for (int lane_in = 0; lane_in < 4; ++lane_in) {
      int ci = ig * 4 + lane_in;
      const char chan = lane_in == 0 ? 'r' : lane_in == 1 ? 'g' : lane_in == 2 ? 'b' : 'a';
      float ww[4];
      for (int lane = 0; lane < 4; ++lane) ww[lane] = w[(group * 4 + lane) * B2_C32 + ci];
      appendf(&fs, "acc+=vec4(%.9g,%.9g,%.9g,%.9g)*v.%c;\n", ww[0], ww[1], ww[2], ww[3], chan);
    }
    appendf(&fs, "}\n");
  }
  appendf(&fs, "gl_FragColor=clamp((acc-%.9g)*%.12g,0.0,1.0);}\n", q_min, 1.0 / q_range);
  GLuint p = glCreateProgram();
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs);
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs.data);
  glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
  GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) { char log[8192]; GLsizei n = 0; glGetProgramInfoLog(p, sizeof(log), &n, log); fprintf(stderr, "op26 link: %.*s\n", (int)n, log); exit(3); }
  free(fs.data);
  return p;
}

static GLuint make_texture_rgba(int w, int h, const uint8_t* data) {
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
  return tex;
}

static void make_fbo_tex(int w, int h, GLuint* tex, GLuint* fbo) {
  *tex = make_texture_rgba(w, h, NULL);
  glGenFramebuffers(1, fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, *tex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "incomplete fbo\n");
    exit(2);
  }
}

static void range_of(const float* x, size_t n, float* mn, float* mx) {
  *mn = x[0]; *mx = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] < *mn) *mn = x[i];
    if (x[i] > *mx) *mx = x[i];
  }
}

static uint8_t* pack_relu6(const float* src, int h, int w, int c, int group) {
  uint8_t* out = malloc((size_t)h * w * 4);
  for (int p = 0; p < h * w; ++p) {
    for (int lane = 0; lane < 4; ++lane) {
      float v = src[p * c + group * 4 + lane];
      out[p * 4 + lane] = (uint8_t)lrintf(fminf(fmaxf(v, 0.0f), 6.0f) * (255.0f / 6.0f));
    }
  }
  return out;
}

static uint8_t* pack_affine(const float* src, int h, int w, int c, int group, float mn, float range) {
  uint8_t* out = malloc((size_t)h * w * 4);
  for (int p = 0; p < h * w; ++p) {
    for (int lane = 0; lane < 4; ++lane) {
      float q = (src[p * c + group * 4 + lane] - mn) / range;
      out[p * 4 + lane] = (uint8_t)lrintf(fminf(fmaxf(q, 0.0f), 1.0f) * 255.0f);
    }
  }
  return out;
}

static void bind_quad(GLuint program, GLuint vbo) {
  glUseProgram(program);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  GLint loc = glGetAttribLocation(program, "pos");
  glEnableVertexAttribArray((GLuint)loc);
  glVertexAttribPointer((GLuint)loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

static void compare_relu6_u8(const char* label, const uint8_t* out, const float* ref, size_t n) {
  double mae = 0.0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float decoded = (float)out[i] * (6.0f / 255.0f);
    float d = fabsf(decoded - ref[i]);
    mae += d;
    if (d > max_abs) max_abs = d;
  }
  printf("%s max_abs=%.6f mean_abs=%.6f\n", label, max_abs, mae / (double)n);
}

static void compare_affine_u8(const char* label, const uint8_t* out, const float* ref, size_t n, float mn, float range) {
  double mae = 0.0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float decoded = (float)out[i] * (range / 255.0f) + mn;
    float d = fabsf(decoded - ref[i]);
    mae += d;
    if (d > max_abs) max_abs = d;
  }
  printf("%s max_abs=%.6f mean_abs=%.6f\n", label, max_abs, mae / (double)n);
}

int main(int argc, char** argv) {
  const char* dir = argc > 1 ? argv[1] : ".";
  int reps = argc > 2 ? atoi(argv[2]) : 300;
  char path[512];

#define READ_F32(name, ptr, count) do { snprintf(path, sizeof(path), "%s/%s", dir, name); read_exact(path, ptr, (count) * sizeof(float)); } while (0)
  float* op9 = malloc((size_t)B2_H128 * B2_W128 * B2_C8 * sizeof(float));
  float* op12 = malloc((size_t)B2_H128 * B2_W128 * B2_C32 * sizeof(float));
  float* op15_w = malloc(3 * 3 * B2_C8 * sizeof(float));
  float* op15_b = malloc(B2_C8 * sizeof(float));
  float* op18_w = malloc(B2_C8 * B2_C8 * sizeof(float));
  float* op18_b = malloc(B2_C8 * sizeof(float));
  float* op18_ref = malloc((size_t)B2_H128 * B2_W128 * B2_C8 * sizeof(float));
  float* op19_ref = malloc((size_t)B2_H128 * B2_W128 * B2_C8 * sizeof(float));
  float* op23_w = malloc(3 * 3 * B2_C32 * sizeof(float));
  float* op23_b = malloc(B2_C32 * sizeof(float));
  float* op23_ref = malloc((size_t)B2_H64 * B2_W64 * B2_C32 * sizeof(float));
  float* op26_w = malloc(B2_C16 * B2_C32 * sizeof(float));
  float* op26_b = malloc(B2_C16 * sizeof(float));
  float* op26_ref = malloc((size_t)B2_H64 * B2_W64 * B2_C16 * sizeof(float));
  READ_F32("op9_ref_f32.bin", op9, (size_t)B2_H128 * B2_W128 * B2_C8);
  READ_F32("op12_ref_f32.bin", op12, (size_t)B2_H128 * B2_W128 * B2_C32);
  READ_F32("op15_w_1hwc_f32.bin", op15_w, 3 * 3 * B2_C8);
  READ_F32("op15_b_f32.bin", op15_b, B2_C8);
  READ_F32("op18_w_ohwi_f32.bin", op18_w, B2_C8 * B2_C8);
  READ_F32("op18_b_f32.bin", op18_b, B2_C8);
  READ_F32("op18_ref_f32.bin", op18_ref, (size_t)B2_H128 * B2_W128 * B2_C8);
  READ_F32("op19_ref_f32.bin", op19_ref, (size_t)B2_H128 * B2_W128 * B2_C8);
  READ_F32("op23_w_1hwc_f32.bin", op23_w, 3 * 3 * B2_C32);
  READ_F32("op23_b_f32.bin", op23_b, B2_C32);
  READ_F32("op23_ref_f32.bin", op23_ref, (size_t)B2_H64 * B2_W64 * B2_C32);
  READ_F32("op26_w_ohwi_f32.bin", op26_w, B2_C16 * B2_C32);
  READ_F32("op26_b_f32.bin", op26_b, B2_C16);
  READ_F32("op26_ref_f32.bin", op26_ref, (size_t)B2_H64 * B2_W64 * B2_C16);

  float op9_min, op9_max, op18_min, op18_max, op26_min, op26_max;
  range_of(op9, (size_t)B2_H128 * B2_W128 * B2_C8, &op9_min, &op9_max);
  range_of(op18_ref, (size_t)B2_H128 * B2_W128 * B2_C8, &op18_min, &op18_max);
  range_of(op26_ref, (size_t)B2_H64 * B2_W64 * B2_C16, &op26_min, &op26_max);
  float op9_range = op9_max - op9_min;
  float op18_range = op18_max - op18_min;
  float op26_range = op26_max - op26_min;

  load_symbols();
  EGLDisplay display;
  EGLSurface surface;
  EGLContext ctx;
  setup_egl(&display, &surface, &ctx);

  GLuint op9_tex[2], op12_tex[8], op15_tex[2], op15_fbo[2], op18_tex[2], op18_fbo[2];
  GLuint op19_tex[2], op19_fbo[2], op23_tex[8], op23_fbo[8], op26_tex[4], op26_fbo[4];
  for (int g = 0; g < 2; ++g) {
    uint8_t* q = pack_affine(op9, B2_H128, B2_W128, B2_C8, g, op9_min, op9_range);
    op9_tex[g] = make_texture_rgba(B2_W128, B2_H128, q);
    free(q);
    make_fbo_tex(B2_W128, B2_H128, &op15_tex[g], &op15_fbo[g]);
    make_fbo_tex(B2_W128, B2_H128, &op18_tex[g], &op18_fbo[g]);
    make_fbo_tex(B2_W128, B2_H128, &op19_tex[g], &op19_fbo[g]);
  }
  for (int g = 0; g < 8; ++g) {
    uint8_t* q = pack_relu6(op12, B2_H128, B2_W128, B2_C32, g);
    op12_tex[g] = make_texture_rgba(B2_W128, B2_H128, q);
    free(q);
    make_fbo_tex(B2_W64, B2_H64, &op23_tex[g], &op23_fbo[g]);
  }
  for (int g = 0; g < 4; ++g) make_fbo_tex(B2_W64, B2_H64, &op26_tex[g], &op26_fbo[g]);

  GLuint p15[2], p18[2], p19[2], p23[8], p26[4];
  for (int g = 0; g < 2; ++g) {
    p15[g] = make_depthwise8_affine_to_relu6(op15_w, op15_b, g, op9_min, op9_range);
    p18[g] = make_pointwise8_relu6_to_affine(op18_w, op18_b, g, op18_min, op18_range);
    p19[g] = make_add8_affine_relu6(op9_min, op9_range, op18_min, op18_range, g);
  }
  for (int g = 0; g < 8; ++g) p23[g] = make_depthwise32_stride2_relu6(op23_w, op23_b, g);
  for (int g = 0; g < 4; ++g) p26[g] = make_pointwise32_relu6_to_affine(op26_w, op26_b, g, op26_min, op26_range);

  GLfloat verts[] = {-1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
  GLuint vbo = 0;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

  double t0 = now_s();
  for (int r = 0; r < reps; ++r) {
    glViewport(0, 0, B2_W128, B2_H128);
    for (int g = 0; g < 2; ++g) {
      bind_quad(p15[g], vbo);
      glUniform1i(glGetUniformLocation(p15[g], "s0"), 0);
      glUniform1i(glGetUniformLocation(p15[g], "s1"), 1);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, op9_tex[0]);
      glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, op9_tex[1]);
      glBindFramebuffer(GL_FRAMEBUFFER, op15_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glViewport(0, 0, B2_W64, B2_H64);
    for (int g = 0; g < 8; ++g) {
      bind_quad(p23[g], vbo);
      glUniform1i(glGetUniformLocation(p23[g], "src"), 0);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, op12_tex[g]);
      glBindFramebuffer(GL_FRAMEBUFFER, op23_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glViewport(0, 0, B2_W128, B2_H128);
    for (int g = 0; g < 2; ++g) {
      bind_quad(p18[g], vbo);
      glUniform1i(glGetUniformLocation(p18[g], "s0"), 0);
      glUniform1i(glGetUniformLocation(p18[g], "s1"), 1);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, op15_tex[0]);
      glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, op15_tex[1]);
      glBindFramebuffer(GL_FRAMEBUFFER, op18_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glViewport(0, 0, B2_W64, B2_H64);
    for (int g = 0; g < 4; ++g) {
      bind_quad(p26[g], vbo);
      for (int ig = 0; ig < 8; ++ig) {
        char uname[4];
        snprintf(uname, sizeof(uname), "s%d", ig);
        glUniform1i(glGetUniformLocation(p26[g], uname), ig);
        glActiveTexture(GL_TEXTURE0 + (GLenum)ig);
        glBindTexture(GL_TEXTURE_2D, op23_tex[ig]);
      }
      glBindFramebuffer(GL_FRAMEBUFFER, op26_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    glViewport(0, 0, B2_W128, B2_H128);
    for (int g = 0; g < 2; ++g) {
      bind_quad(p19[g], vbo);
      glUniform1i(glGetUniformLocation(p19[g], "a"), 0);
      glUniform1i(glGetUniformLocation(p19[g], "b"), 1);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, op9_tex[g]);
      glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, op18_tex[g]);
      glBindFramebuffer(GL_FRAMEBUFFER, op19_fbo[g]);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
  }
  glFinish();
  double t1 = now_s();

  double macs = (double)B2_H128 * B2_W128 * B2_C8 * 9.0 +
      (double)B2_H128 * B2_W128 * B2_C8 * B2_C8 +
      (double)B2_H64 * B2_W64 * B2_C32 * 9.0 +
      (double)B2_H64 * B2_W64 * B2_C32 * B2_C16;
  printf("gles2_block2 reps=%d avg_ms=%.3f fps=%.2f gmac_s=%.3f macs=%.0f\n",
      reps, (t1 - t0) * 1000.0 / reps, reps / (t1 - t0), macs * reps / (t1 - t0) / 1e9, macs);
  printf("ranges op9=[%.6f,%.6f] op18=[%.6f,%.6f] op26=[%.6f,%.6f]\n",
      op9_min, op9_max, op18_min, op18_max, op26_min, op26_max);

  uint8_t* tmp = malloc((size_t)B2_H128 * B2_W128 * 4);
  uint8_t* out19 = calloc((size_t)B2_H128 * B2_W128 * B2_C8, 1);
  for (int g = 0; g < 2; ++g) {
    glBindFramebuffer(GL_FRAMEBUFFER, op19_fbo[g]);
    glReadPixels(0, 0, B2_W128, B2_H128, GL_RGBA, GL_UNSIGNED_BYTE, tmp);
    for (int p = 0; p < B2_H128 * B2_W128; ++p)
      for (int c = 0; c < 4; ++c) out19[p * B2_C8 + g * 4 + c] = tmp[p * 4 + c];
  }
  compare_relu6_u8("op19_vs_float_ref", out19, op19_ref, (size_t)B2_H128 * B2_W128 * B2_C8);

  uint8_t* tmp64 = malloc((size_t)B2_H64 * B2_W64 * 4);
  uint8_t* out26 = calloc((size_t)B2_H64 * B2_W64 * B2_C16, 1);
  for (int g = 0; g < 4; ++g) {
    glBindFramebuffer(GL_FRAMEBUFFER, op26_fbo[g]);
    glReadPixels(0, 0, B2_W64, B2_H64, GL_RGBA, GL_UNSIGNED_BYTE, tmp64);
    for (int p = 0; p < B2_H64 * B2_W64; ++p)
      for (int c = 0; c < 4; ++c) out26[p * B2_C16 + g * 4 + c] = tmp64[p * 4 + c];
  }
  compare_affine_u8("op26_vs_float_ref", out26, op26_ref, (size_t)B2_H64 * B2_W64 * B2_C16, op26_min, op26_range);

  eglDestroyContext(display, ctx);
  eglDestroySurface(display, surface);
  eglTerminate(display);
  return 0;
}
