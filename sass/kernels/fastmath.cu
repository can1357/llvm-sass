// Differential tests for approximate math and the FP idioms folded into one
// instruction (sass/tools/difftest.py --inputs f32). Build with nvcc
// --use_fast_math and sassc --fast-math to compare the fast-math
// lowerings (all but p_*). Without fast math (both flush-free), f_exp2,
// f_sat*, f_step, f_select and the p_* kernels (explicit PTX approx
// instructions) compare; with -fapprox-func alone against nvcc
// -prec-div=false -prec-sqrt=false, so do f_div, f_div3, f_rcp, f_sqrt and
// f_rsqrt. Under flush-to-zero the backend flushes the non-ftz approx
// intrinsics too, as libdevice does, where ptxas keeps inline PTX's exact
// non-ftz semantics: p_* differ on denormal results there.
//
// Each kernel is one warp mapping 512 inputs x (and y, the input 256 further
// on) to 512 outputs.
#ifdef SASS_CLANG
// libm through the builtins: afn under -fapprox-func, as nvcc's
// --use_fast_math intrinsics.
#define M(f) __builtin_##f
// The PTX approx instructions (not flush-to-zero) as NVVM builtins.
#define EX2 __nvvm_ex2_approx_f
#define LG2 __nvvm_lg2_approx_f
#define RSQRT __nvvm_rsqrt_approx_f
#define SQRT __nvvm_sqrt_approx_f
#define SIN __nvvm_sin_approx_f
#define DIV __nvvm_div_approx_f
#else
#define M(f) f
#define PTX1(name, op)                                                        \
  static __device__ float name(float a) {                                     \
    float r;                                                                  \
    asm(op " %0, %1;" : "=f"(r) : "f"(a));                                    \
    return r;                                                                 \
  }
PTX1(EX2, "ex2.approx.f32")
PTX1(LG2, "lg2.approx.f32")
PTX1(RSQRT, "rsqrt.approx.f32")
PTX1(SQRT, "sqrt.approx.f32")
PTX1(SIN, "sin.approx.f32")
static __device__ float DIV(float a, float b) {
  float r;
  asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
#endif

#define K(name, expr)                                                         \
  extern "C" __global__ void name(const uint4 *in, uint4 *out) {              \
    const float *x = (const float *)in;                                       \
    float *o = (float *)out;                                                  \
    for (int i = threadIdx.x; i < 512; i += 32) {                             \
      float a = x[i];                                                         \
      [[maybe_unused]] float b = x[(i + 256) & 511];                          \
      o[i] = (expr);                                                          \
    }                                                                         \
  }

K(f_exp2, M(exp2f)(a))
// The backend implements libm calls, division and sqrt only as approximations
// (afn): with -fapprox-func (SASS_APPROX_FUNC) or --fast-math.
#if !defined(SASS_CLANG) || defined(__USE_FAST_MATH__) || defined(SASS_APPROX_FUNC)
K(f_exp, M(expf)(a))
K(f_exp10, M(exp10f)(a))
K(f_log, M(logf)(a))
K(f_log2, M(log2f)(a))
K(f_log10, M(log10f)(a))
K(f_pow, M(powf)(a, b))
K(f_sin, M(sinf)(a))
K(f_cos, M(cosf)(a))
K(f_sincos, M(sinf)(a) * M(cosf)(a))
K(f_tan, M(tanf)(a))
K(f_tanh, M(tanhf)(a))
K(f_sqrt, M(sqrtf)(a))
K(f_rsqrt, 1.0f / M(sqrtf)(a))
K(f_div, a / b)
K(f_div3, a / 3.0f)
K(f_rcp, 1.0f / b)
K(f_log_add, M(logf)(a) + b)
K(f_exp_mul, M(expf)(a * b))
#endif
K(f_sat, M(fminf)(M(fmaxf)(a, 0.f), 1.f))
K(f_sat_fma, M(fminf)(M(fmaxf)(a * b + 0.5f, 0.f), 1.f))
K(f_sat_mul, M(fminf)(M(fmaxf)(a * b, 0.f), 1.f))
K(f_step, a < b ? 1.f : 0.f)
K(f_select, a > 0.f ? a * 0.5f : a)
K(p_ex2, EX2(a))
K(p_lg2, LG2(a))
K(p_rsqrt, RSQRT(a))
K(p_sqrt, SQRT(a))
K(p_sin, SIN(a))
K(p_div, DIV(a, b))
