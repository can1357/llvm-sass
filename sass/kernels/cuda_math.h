// Single-precision CUDA math API for the llvm-sass flow (force-included after cuda_min.h).
//
// The function bodies are an op-for-op translation of the PTX nvcc 13 emits for each function
// (nvcc -ptx -arch=sm_90 -O3, libdevice inlined), produced by sass/tools/gen_cuda_math.py from
// sass/tools/cuda_math_ref.cu; results are bit-identical to CUDA 13 (validated on H100).
// PTX -> C mapping: fma.rn/rz/rm/rp.f32 -> __nvvm_fma_*_f (never contractible a*b+c);
// mul/add/sub.f32 (contractible in PTX) -> plain C operators; mul/add/sub.r[nzmp].f32 -> __nvvm_mul/add_*_f
// (sub.rX a,b == add.rX a,-b); cvt.sat.f32.f32 -> __nvvm_saturate_f; cvt.r{z,n,m,p}i.f32.f32 ->
// trunc/rint/floor/ceil; copysign.f32 d,a,b -> copysign(b, a); setp.<cmp>u -> negated ordered compare;
// ld/st.local on the reduction scratch -> a local array; __cudart_i2opi_f -> __sm_i2opi_f.
//
// Under -ffast-math (clang defines __FAST_MATH__) the bodies come from nvcc -use_fast_math -ftz=false
// instead (expf exp10f logf log2f log10f powf sinf cosf tanf sincosf become the __-intrinsics, sqrtf and
// hypotf use sqrt.approx); tanhf stays precise (tanh.approx.f32 has no clang builtin). Denormal flushing
// (nvcc's default -ftz=true under -use_fast_math) is not modelled.
//
// Without SASS_CLANG (nvcc, used only to validate the translation) the __nvvm builtins are replaced
// by equivalent inline PTX. SASS_MATH_NAME lets the validation build rename every function.
#pragma once

#ifndef SASS_MATH_NAME
#define SASS_MATH_NAME(n) n
#endif
#define SASS_MATH_FN(rt, n) SASS_INL rt SASS_MATH_NAME(n)
#ifdef __FAST_MATH__
#define SASS_MATH_FAST 1
#else
#define SASS_MATH_FAST 0
#endif

#ifdef SASS_CLANG
#define __sm_fma_rn __nvvm_fma_rn_f
#define __sm_fma_rz __nvvm_fma_rz_f
#define __sm_fma_rm __nvvm_fma_rm_f
#define __sm_fma_rp __nvvm_fma_rp_f
#define __sm_add_rn __nvvm_add_rn_f
#define __sm_add_rz __nvvm_add_rz_f
#define __sm_add_rm __nvvm_add_rm_f
#define __sm_add_rp __nvvm_add_rp_f
#define __sm_mul_rn __nvvm_mul_rn_f
#define __sm_mul_rz __nvvm_mul_rz_f
#define __sm_mul_rm __nvvm_mul_rm_f
#define __sm_mul_rp __nvvm_mul_rp_f
#define __sm_div_rn __nvvm_div_rn_f
#define __sm_div_rz __nvvm_div_rz_f
#define __sm_div_rm __nvvm_div_rm_f
#define __sm_div_rp __nvvm_div_rp_f
#define __sm_div_approx __nvvm_div_approx_f
#define __sm_div_approx_ftz __nvvm_div_approx_ftz_f
#define __sm_rcp_rn __nvvm_rcp_rn_f
#define __sm_rcp_rz __nvvm_rcp_rz_f
#define __sm_rcp_rm __nvvm_rcp_rm_f
#define __sm_rcp_rp __nvvm_rcp_rp_f
#define __sm_rcp_approx_ftz __nvvm_rcp_approx_ftz_f
#define __sm_sqrt_rn __nvvm_sqrt_rn_f
#define __sm_sqrt_rz __nvvm_sqrt_rz_f
#define __sm_sqrt_rm __nvvm_sqrt_rm_f
#define __sm_sqrt_rp __nvvm_sqrt_rp_f
#define __sm_sqrt_approx __nvvm_sqrt_approx_f
#define __sm_rsqrt_approx __nvvm_rsqrt_approx_f
#define __sm_rsqrt_approx_ftz __nvvm_rsqrt_approx_ftz_f
#define __sm_ex2_approx __nvvm_ex2_approx_f
#define __sm_ex2_approx_ftz __nvvm_ex2_approx_ftz_f
#define __sm_lg2_approx __nvvm_lg2_approx_f
#define __sm_lg2_approx_ftz __nvvm_lg2_approx_ftz_f
#define __sm_sin_approx __nvvm_sin_approx_f
#define __sm_cos_approx __nvvm_cos_approx_f
#define __sm_saturate __nvvm_saturate_f
#define __sm_f2i_rn __nvvm_f2i_rn
#define __sm_f2i_rz __nvvm_f2i_rz
#define __sm_f2i_rm __nvvm_f2i_rm
#define __sm_f2i_rp __nvvm_f2i_rp
#define __sm_f2ui_rn __nvvm_f2ui_rn
#define __sm_f2ui_rz __nvvm_f2ui_rz
#define __sm_f2ui_rm __nvvm_f2ui_rm
#define __sm_f2ui_rp __nvvm_f2ui_rp
#define __sm_f2ll_rn __nvvm_f2ll_rn
#define __sm_f2ll_rz __nvvm_f2ll_rz
#define __sm_f2ll_rm __nvvm_f2ll_rm
#define __sm_f2ll_rp __nvvm_f2ll_rp
#define __sm_f2ull_rn __nvvm_f2ull_rn
#define __sm_f2ull_rz __nvvm_f2ull_rz
#define __sm_f2ull_rm __nvvm_f2ull_rm
#define __sm_f2ull_rp __nvvm_f2ull_rp
#define __sm_i2f_rn __nvvm_i2f_rn
#define __sm_i2f_rz __nvvm_i2f_rz
#define __sm_i2f_rm __nvvm_i2f_rm
#define __sm_i2f_rp __nvvm_i2f_rp
#define __sm_ui2f_rn __nvvm_ui2f_rn
#define __sm_ui2f_rz __nvvm_ui2f_rz
#define __sm_ui2f_rm __nvvm_ui2f_rm
#define __sm_ui2f_rp __nvvm_ui2f_rp
#define __sm_ll2f_rn __nvvm_ll2f_rn
#define __sm_ll2f_rz __nvvm_ll2f_rz
#define __sm_ll2f_rm __nvvm_ll2f_rm
#define __sm_ll2f_rp __nvvm_ll2f_rp
#define __sm_ull2f_rn __nvvm_ull2f_rn
#define __sm_ull2f_rz __nvvm_ull2f_rz
#define __sm_ull2f_rm __nvvm_ull2f_rm
#define __sm_ull2f_rp __nvvm_ull2f_rp
#define __sm_trunc __builtin_truncf
#define __sm_floor __builtin_floorf
#define __sm_ceil __builtin_ceilf
#define __sm_rint __builtin_rintf
#define __sm_fabs __builtin_fabsf
#define __sm_copysign __builtin_copysignf
#define __sm_f32bits(u) __builtin_bit_cast(float, (unsigned)(u))
#define __sm_popc32(x) __builtin_popcount(x)
#define __sm_popc64(x) __builtin_popcountll(x)
#define __sm_clz32(x) __builtin_clzg((unsigned)(x), 32)
#define __sm_clz64(x) __builtin_clzg((unsigned long long)(x), 64)
#define __sm_brev32(x) __builtin_bitreverse32(x)
#define __sm_brev64(x) __builtin_bitreverse64(x)
#define __sm_prmt(a, b, s) ((unsigned)__nvvm_prmt((a), (b), (s)))
#else // nvcc: inline-PTX equivalents (validation build only)
#define __SM_F1(n, op) SASS_INL float n(float a) { float d; asm(op " %0, %1;" : "=f"(d) : "f"(a)); return d; }
#define __SM_F2(n, op) SASS_INL float n(float a, float b) { float d; asm(op " %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b)); return d; }
#define __SM_F3(n, op) \
  SASS_INL float n(float a, float b, float c) { float d; asm(op " %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c)); return d; }
#define __SM_CVT(n, op, dt, dc, st, sc) SASS_INL dt n(st a) { dt d; asm(op " %0, %1;" : "=" dc(d) : sc(a)); return d; }
#define __SM_RND4(m) m(rn) m(rz) m(rm) m(rp)
#define __SM_X(r) __SM_F3(__sm_fma_##r, "fma." #r ".f32") __SM_F2(__sm_add_##r, "add." #r ".f32") \
  __SM_F2(__sm_mul_##r, "mul." #r ".f32") __SM_F2(__sm_div_##r, "div." #r ".f32") __SM_F1(__sm_rcp_##r, "rcp." #r ".f32") \
  __SM_F1(__sm_sqrt_##r, "sqrt." #r ".f32") \
  __SM_CVT(__sm_f2i_##r, "cvt." #r "i.s32.f32", int, "r", float, "f") \
  __SM_CVT(__sm_f2ui_##r, "cvt." #r "i.u32.f32", unsigned, "r", float, "f") \
  __SM_CVT(__sm_f2ll_##r, "cvt." #r "i.s64.f32", long long, "l", float, "f") \
  __SM_CVT(__sm_f2ull_##r, "cvt." #r "i.u64.f32", unsigned long long, "l", float, "f") \
  __SM_CVT(__sm_i2f_##r, "cvt." #r ".f32.s32", float, "f", int, "r") \
  __SM_CVT(__sm_ui2f_##r, "cvt." #r ".f32.u32", float, "f", unsigned, "r") \
  __SM_CVT(__sm_ll2f_##r, "cvt." #r ".f32.s64", float, "f", long long, "l") \
  __SM_CVT(__sm_ull2f_##r, "cvt." #r ".f32.u64", float, "f", unsigned long long, "l")
__SM_RND4(__SM_X)
__SM_F2(__sm_div_approx, "div.approx.f32")
__SM_F2(__sm_div_approx_ftz, "div.approx.ftz.f32")
__SM_F1(__sm_rcp_approx_ftz, "rcp.approx.ftz.f32")
__SM_F1(__sm_rsqrt_approx, "rsqrt.approx.f32")
__SM_F1(__sm_sqrt_approx, "sqrt.approx.f32")
__SM_F1(__sm_rsqrt_approx_ftz, "rsqrt.approx.ftz.f32")
__SM_F1(__sm_ex2_approx, "ex2.approx.f32")
__SM_F1(__sm_ex2_approx_ftz, "ex2.approx.ftz.f32")
__SM_F1(__sm_lg2_approx, "lg2.approx.f32")
__SM_F1(__sm_lg2_approx_ftz, "lg2.approx.ftz.f32")
__SM_F1(__sm_sin_approx, "sin.approx.f32")
__SM_F1(__sm_cos_approx, "cos.approx.f32")
__SM_F1(__sm_saturate, "cvt.sat.f32.f32")
__SM_F1(__sm_trunc, "cvt.rzi.f32.f32")
__SM_F1(__sm_floor, "cvt.rmi.f32.f32")
__SM_F1(__sm_ceil, "cvt.rpi.f32.f32")
__SM_F1(__sm_rint, "cvt.rni.f32.f32")
__SM_F1(__sm_fabs, "abs.f32")
SASS_INL float __sm_copysign(float m, float s) { float d; asm("copysign.f32 %0, %2, %1;" : "=f"(d) : "f"(m), "f"(s)); return d; }
SASS_INL float __sm_f32bits(unsigned u) { return __uint_as_float(u); }
__SM_CVT(__sm_popc32, "popc.b32", unsigned, "r", unsigned, "r")
__SM_CVT(__sm_clz32, "clz.b32", unsigned, "r", unsigned, "r")
__SM_CVT(__sm_brev32, "brev.b32", unsigned, "r", unsigned, "r")
__SM_CVT(__sm_brev64, "brev.b64", unsigned long long, "l", unsigned long long, "l")
SASS_INL unsigned __sm_popc64(unsigned long long a) { unsigned d; asm("popc.b64 %0, %1;" : "=r"(d) : "l"(a)); return d; }
SASS_INL unsigned __sm_clz64(unsigned long long a) { unsigned d; asm("clz.b64 %0, %1;" : "=r"(d) : "l"(a)); return d; }
SASS_INL unsigned __sm_prmt(unsigned a, unsigned b, unsigned s) {
  unsigned d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
#endif

// bfi.b64 d, a, b, pos, len: insert the low `len` bits of a into b at bit `pos` (pos + len <= 64 here).
SASS_INL unsigned long long __sm_bfi64(unsigned long long a, unsigned long long b, unsigned pos, unsigned len) {
  unsigned long long m = (len >= 64 ? ~0ull : ((1ull << len) - 1)) << pos;
  return (b & ~m) | ((a << pos) & m);
}
// 2/pi bits for the Payne-Hanek reduction of sinf/cosf/tanf/sincosf (CUDA's __cudart_i2opi_f).
static __constant__ const unsigned __sm_i2opi_f[6] = {0x3c439041u, 0xdb629599u, 0xf534ddc0u,
                                                   0xfc2757d1u, 0x4e441529u, 0xa2f9836eu};

// ---- BEGIN GENERATED (gen_cuda_math.py) ----
SASS_MATH_FN(float, __expf)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = f1 * 0x1.715476p+0f;
  f3 = __sm_ex2_approx(f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __exp10f)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = f1 * 0x1.a934fp+1f;
  f3 = __sm_ex2_approx(f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __logf)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = __sm_lg2_approx(f1);
  f3 = f2 * 0x1.62e43p-1f;
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __log2f)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_lg2_approx(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __log10f)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = __sm_lg2_approx(f1);
  f3 = f2 * 0x1.344136p-2f;
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __sinf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sin_approx(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __cosf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_cos_approx(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __tanf)(float a0) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = __sm_sin_approx(f1);
  f3 = __sm_cos_approx(f1);
  f4 = __sm_div_approx(f2, f3);
  __ret = f4;
  return __ret;
}

SASS_MATH_FN(float, __frcp_rn)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rcp_rn(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __frcp_rz)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rcp_rz(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __frcp_ru)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rcp_rp(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __frcp_rd)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rcp_rm(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __fsqrt_rn)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sqrt_rn(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __fsqrt_rz)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sqrt_rz(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __fsqrt_ru)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sqrt_rp(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __fsqrt_rd)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sqrt_rm(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __frsqrt_rn)(float a0) {
  float __ret;
  float f1, f2, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22;
  bool p1, p2;
  unsigned r1, r3, r4, r5, r6, r7, r8, r9, r10, r11;
  f6 = a0;
  f1 = __sm_fabs(f6);
  p1 = f1 < 0x1p-126f;
  f7 = f6 * 0x1p+24f;
  f2 = p1 ? f7 : f6;
  r1 = __float_as_uint(f2);
  r3 = r1 + 0xff800000u;
  p2 = r3 < 0x7f000000u;
  if (p2) goto L72_2;
  goto L72_1;
L72_2:;
  r4 = r1 & 0xffffffu;
  r5 = r4 | 0x3f000000u;
  f8 = __sm_f32bits(r5);
  r6 = r5 - r1;
  r7 = r6 + 0xc000000u;
  r8 = p1 ? r7 : r6;
  f9 = __sm_rsqrt_approx_ftz(f8);
  f10 = f9 * f9;
  f11 = -f10;
  f12 = __sm_fma_rn(f9, f9, f11);
  f13 = -f8;
  f14 = 0x1p+0f;
  f15 = __sm_fma_rn(f10, f13, f14);
  f16 = __sm_fma_rn(f12, f13, f15);
  f17 = 0x1p-1f;
  f18 = 0x1.8p-2f;
  f19 = __sm_fma_rn(f18, f16, f17);
  f20 = f9 * f16;
  f21 = __sm_fma_rn(f19, f20, f9);
  r9 = (int)r8 >> 1u;
  r10 = __float_as_uint(f21);
  r11 = r9 + r10;
  f22 = __sm_f32bits(r11);
  goto L72_3;
L72_1:;
  f22 = __sm_rsqrt_approx_ftz(f2);
L72_3:;
  __ret = f22;
  return __ret;
}

SASS_MATH_FN(float, __saturatef)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_saturate(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, __powf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4, f5;
  f1 = a0;
  f2 = a1;
  f3 = __sm_lg2_approx(f1);
  f4 = f2 * f3;
  f5 = __sm_ex2_approx(f4);
  __ret = f5;
  return __ret;
}

SASS_MATH_FN(float, __fdividef)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_div_approx(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fdiv_rn)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_div_rn(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fdiv_rz)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_div_rz(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fdiv_ru)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_div_rp(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fdiv_rd)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_div_rm(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fadd_rn)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rn(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fadd_rz)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rz(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fadd_ru)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rp(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fadd_rd)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rm(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fsub_rn)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rn(f1, -(f2));
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fsub_rz)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rz(f1, -(f2));
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fsub_ru)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rp(f1, -(f2));
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fsub_rd)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rm(f1, -(f2));
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fmul_rn)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_mul_rn(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fmul_rz)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_mul_rz(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fmul_ru)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_mul_rp(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fmul_rd)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_mul_rm(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, __fmaf_rn)(float a0, float a1, float a2) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = a1;
  f3 = a2;
  f4 = __sm_fma_rn(f1, f2, f3);
  __ret = f4;
  return __ret;
}

SASS_MATH_FN(float, __fmaf_rz)(float a0, float a1, float a2) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = a1;
  f3 = a2;
  f4 = __sm_fma_rz(f1, f2, f3);
  __ret = f4;
  return __ret;
}

SASS_MATH_FN(float, __fmaf_ru)(float a0, float a1, float a2) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = a1;
  f3 = a2;
  f4 = __sm_fma_rp(f1, f2, f3);
  __ret = f4;
  return __ret;
}

SASS_MATH_FN(float, __fmaf_rd)(float a0, float a1, float a2) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = a1;
  f3 = a2;
  f4 = __sm_fma_rm(f1, f2, f3);
  __ret = f4;
  return __ret;
}

SASS_MATH_FN(int, __isnanf)(float a0) {
  int __ret;
  float f1, f2;
  bool p1;
  unsigned r2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = !(f2 <= __sm_f32bits(0x7f800000u));
  r2 = p1 ? 1u : 0u;
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(int, __isinff)(float a0) {
  int __ret;
  float f1, f2;
  bool p1;
  unsigned r2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = f2 == __sm_f32bits(0x7f800000u);
  r2 = p1 ? 1u : 0u;
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(int, __finitef)(float a0) {
  int __ret;
  float f1, f2;
  bool p1;
  unsigned r2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = f2 < __sm_f32bits(0x7f800000u);
  r2 = p1 ? 1u : 0u;
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(int, __signbitf)(float a0) {
  int __ret;
  unsigned r2, r3;
  r2 = __float_as_uint(a0);
  r3 = r2 >> 0x1fu;
  __ret = r3;
  return __ret;
}

SASS_MATH_FN(int, __float2int_rn)(float a0) {
  int __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2i_rn(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(int, __float2int_rz)(float a0) {
  int __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2i_rz(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(int, __float2int_ru)(float a0) {
  int __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2i_rp(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(int, __float2int_rd)(float a0) {
  int __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2i_rm(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(unsigned, __float2uint_rn)(float a0) {
  unsigned __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2ui_rn(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(unsigned, __float2uint_rz)(float a0) {
  unsigned __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2ui_rz(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(unsigned, __float2uint_ru)(float a0) {
  unsigned __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2ui_rp(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(unsigned, __float2uint_rd)(float a0) {
  unsigned __ret;
  float f1;
  unsigned r2;
  f1 = a0;
  r2 = __sm_f2ui_rm(f1);
  __ret = r2;
  return __ret;
}

SASS_MATH_FN(long long, __float2ll_rn)(float a0) {
  long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ll_rn(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(long long, __float2ll_rz)(float a0) {
  long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ll_rz(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(long long, __float2ll_ru)(float a0) {
  long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ll_rp(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(long long, __float2ll_rd)(float a0) {
  long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ll_rm(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(unsigned long long, __float2ull_rn)(float a0) {
  unsigned long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ull_rn(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(unsigned long long, __float2ull_rz)(float a0) {
  unsigned long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ull_rz(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(unsigned long long, __float2ull_ru)(float a0) {
  unsigned long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ull_rp(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(unsigned long long, __float2ull_rd)(float a0) {
  unsigned long long __ret;
  float f1;
  unsigned long long rd7;
  f1 = a0;
  rd7 = __sm_f2ull_rm(f1);
  __ret = rd7;
  return __ret;
}

SASS_MATH_FN(float, __int2float_rn)(int a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_i2f_rn((int)r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __int2float_rz)(int a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_i2f_rz((int)r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __int2float_ru)(int a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_i2f_rp((int)r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __int2float_rd)(int a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_i2f_rm((int)r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __uint2float_rn)(unsigned a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_ui2f_rn(r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __uint2float_rz)(unsigned a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_ui2f_rz(r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __uint2float_ru)(unsigned a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_ui2f_rp(r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __uint2float_rd)(unsigned a0) {
  float __ret;
  float f1;
  unsigned r2;
  r2 = a0;
  f1 = __sm_ui2f_rm(r2);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ll2float_rn)(long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ll2f_rn((long long)rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ll2float_rz)(long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ll2f_rz((long long)rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ll2float_ru)(long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ll2f_rp((long long)rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ll2float_rd)(long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ll2f_rm((long long)rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ull2float_rn)(unsigned long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ull2f_rn(rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ull2float_rz)(unsigned long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ull2f_rz(rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ull2float_ru)(unsigned long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ull2f_rp(rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(float, __ull2float_rd)(unsigned long long a0) {
  float __ret;
  float f1;
  unsigned long long rd7;
  rd7 = a0;
  f1 = __sm_ull2f_rm(rd7);
  __ret = f1;
  return __ret;
}

SASS_MATH_FN(void, __sincosf)(float a0, float *o0, float *o1) {
  float f1, f2, f3;
  f1 = a0;
  f2 = __sm_sin_approx(f1);
  *o0 = f2;
  f3 = __sm_cos_approx(f1);
  *o1 = f3;
  return;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, expf)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = f1 * 0x1.715476p+0f;
  f3 = __sm_ex2_approx(f2);
  __ret = f3;
  return __ret;
}
#else
SASS_MATH_FN(float, expf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17;
  unsigned r2, r3;
  f1 = a0;
  f2 = 0x1p-1f;
  f3 = 0x1.77313ap-8f;
  f4 = __sm_fma_rn(f1, f3, f2);
  f5 = __sm_saturate(f4);
  f6 = 0x1.800002p+23f;
  f7 = 0x1.f8p+7f;
  f8 = __sm_fma_rm(f5, f7, f6);
  f9 = f8 + -0x1.8000fep+23f;
  f10 = -f9;
  f11 = 0x1.715476p+0f;
  f12 = __sm_fma_rn(f1, f11, f10);
  f13 = 0x1.4ae0cp-26f;
  f14 = __sm_fma_rn(f1, f13, f12);
  r2 = __float_as_uint(f8);
  r3 = r2 << 0x17u;
  f15 = __sm_f32bits(r3);
  f16 = __sm_ex2_approx_ftz(f14);
  f17 = f16 * f15;
  __ret = f17;
  return __ret;
}
#endif

SASS_MATH_FN(float, exp2f)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_ex2_approx(f1);
  __ret = f2;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, exp10f)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = f1 * 0x1.a934fp+1f;
  f3 = __sm_ex2_approx(f2);
  __ret = f3;
  return __ret;
}
#else
SASS_MATH_FN(float, exp10f)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17;
  unsigned r2, r3;
  f1 = a0;
  f2 = 0x1p-1f;
  f3 = 0x1.aff4c4p-7f;
  f4 = __sm_fma_rn(f1, f3, f2);
  f5 = __sm_saturate(f4);
  f6 = 0x1.800002p+23f;
  f7 = 0x1.f8p+7f;
  f8 = __sm_fma_rm(f5, f7, f6);
  f9 = f8 + -0x1.8000fep+23f;
  f10 = -f9;
  f11 = 0x1.a934fp+1f;
  f12 = __sm_fma_rn(f1, f11, f10);
  f13 = 0x1.2f346ep-24f;
  f14 = __sm_fma_rn(f1, f13, f12);
  r2 = __float_as_uint(f8);
  r3 = r2 << 0x17u;
  f15 = __sm_f32bits(r3);
  f16 = __sm_ex2_approx_ftz(f14);
  f17 = f16 * f15;
  __ret = f17;
  return __ret;
}
#endif

SASS_MATH_FN(float, expm1f)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24;
  float f25, f26, f27, f28, f29, f30, f31, f32;
  bool p1, p2, p3, p4, p5;
  f1 = a0;
  f2 = f1 * 0x1.715476p+0f;
  f3 = __sm_rint(f2);
  f4 = __sm_fabs(f1);
  p1 = f4 < 0x1.a3d70ap-2f;
  f5 = p1 ? 0.0f : f3;
  f6 = -f5;
  f7 = 0x1.62e4p-1f;
  f8 = __sm_fma_rn(f6, f7, f1);
  f9 = 0x1.7f7d1cp-20f;
  f10 = __sm_fma_rn(f6, f9, f8);
  p2 = f5 == 0x1p+7f;
  f11 = f5 + -0x1p+0f;
  f12 = p2 ? f11 : f5;
  f13 = 0x1.12acc6p-7f;
  f14 = 0x1.6bd7ccp-10f;
  f15 = __sm_fma_rn(f14, f10, f13);
  f16 = 0x1.5557c6p-5f;
  f17 = __sm_fma_rn(f15, f10, f16);
  f18 = 0x1.5553ecp-3f;
  f19 = __sm_fma_rn(f17, f10, f18);
  f20 = 0x1.fffffcp-2f;
  f21 = __sm_fma_rn(f19, f10, f20);
  f22 = f10 * f21;
  f23 = __sm_fma_rn(f22, f10, f10);
  f24 = __sm_ex2_approx(f12);
  f25 = f24 + -0x1p+0f;
  f26 = __sm_fma_rn(f23, f24, f25);
  f27 = f26 + f26;
  f28 = p2 ? f27 : f26;
  p3 = f12 > 0x1p+7f;
  f29 = p3 ? __sm_f32bits(0x7f800000u) : f28;
  p4 = f12 < -0x1.9p+4f;
  f30 = p4 ? -0x1p+0f : f29;
  p5 = f1 == 0.0f;
  f31 = f1 + f1;
  f32 = p5 ? f31 : f30;
  __ret = f32;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, logf)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = __sm_lg2_approx(f1);
  f3 = f2 * 0x1.62e43p-1f;
  __ret = f3;
  return __ret;
}
#else
SASS_MATH_FN(float, logf)(float a0) {
  float __ret;
  float f1, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35;
  bool p1, p2, p3;
  unsigned r2, r3, r4, r5;
  f5 = a0;
  p1 = f5 < 0x1p-126f;
  f6 = f5 * 0x1p+23f;
  f1 = p1 ? f6 : f5;
  f7 = p1 ? -0x1.7p+4f : 0.0f;
  r2 = __float_as_uint(f1);
  r3 = r2 + 0xc0d55555u;
  r4 = r3 & 0xff800000u;
  r5 = r2 - r4;
  f8 = __sm_f32bits(r5);
  f9 = __sm_i2f_rn((int)r4);
  f10 = 0x1p-23f;
  f11 = __sm_fma_rn(f9, f10, f7);
  f12 = f8 + -0x1p+0f;
  f13 = 0x1.2073ecp-3f;
  f14 = -0x1.0aa04ep-3f;
  f15 = __sm_fma_rn(f14, f12, f13);
  f16 = -0x1.f19b98p-4f;
  f17 = __sm_fma_rn(f15, f12, f16);
  f18 = 0x1.1e52aap-3f;
  f19 = __sm_fma_rn(f17, f12, f18);
  f20 = -0x1.55b172p-3f;
  f21 = __sm_fma_rn(f19, f12, f20);
  f22 = 0x1.99da16p-3f;
  f23 = __sm_fma_rn(f21, f12, f22);
  f24 = -0x1.fffe44p-3f;
  f25 = __sm_fma_rn(f23, f12, f24);
  f26 = 0x1.5554fp-2f;
  f27 = __sm_fma_rn(f25, f12, f26);
  f28 = -0x1p-1f;
  f29 = __sm_fma_rn(f27, f12, f28);
  f30 = f12 * f29;
  f31 = __sm_fma_rn(f30, f12, f12);
  f32 = 0x1.62e43p-1f;
  f35 = __sm_fma_rn(f11, f32, f31);
  p2 = r2 < 0x7f800000u;
  if (p2) goto L4_2;
  f33 = __sm_f32bits(0x7f800000u);
  f35 = __sm_fma_rn(f1, f33, f33);
L4_2:;
  p3 = f1 == 0.0f;
  f34 = p3 ? __sm_f32bits(0xff800000u) : f35;
  __ret = f34;
  return __ret;
}
#endif

#if SASS_MATH_FAST
SASS_MATH_FN(float, log2f)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_lg2_approx(f1);
  __ret = f2;
  return __ret;
}
#else
SASS_MATH_FN(float, log2f)(float a0) {
  float __ret;
  float f1, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38;
  bool p1, p2, p3;
  unsigned r2, r3, r4, r5;
  f5 = a0;
  p1 = f5 < 0x1p-126f;
  f6 = f5 * 0x1p+23f;
  f1 = p1 ? f6 : f5;
  f7 = p1 ? -0x1.7p+4f : 0.0f;
  r2 = __float_as_uint(f1);
  r3 = r2 + 0xc0cafb0du;
  r4 = r3 & 0xff800000u;
  r5 = r2 - r4;
  f8 = __sm_f32bits(r5);
  f9 = __sm_i2f_rn((int)r4);
  f10 = 0x1p-23f;
  f11 = __sm_fma_rn(f9, f10, f7);
  f12 = f8 + -0x1p+0f;
  f13 = -0x1.58fe6p-3f;
  f14 = 0x1.8d64fep-4f;
  f15 = __sm_fma_rn(f14, f12, f13);
  f16 = 0x1.5f9e54p-3f;
  f17 = __sm_fma_rn(f15, f12, f16);
  f18 = -0x1.6e9c86p-3f;
  f19 = __sm_fma_rn(f17, f12, f18);
  f20 = 0x1.a417e8p-3f;
  f21 = __sm_fma_rn(f19, f12, f20);
  f22 = -0x1.ec7916p-3f;
  f23 = __sm_fma_rn(f21, f12, f22);
  f24 = 0x1.277f32p-2f;
  f25 = __sm_fma_rn(f23, f12, f24);
  f26 = -0x1.715492p-2f;
  f27 = __sm_fma_rn(f25, f12, f26);
  f28 = 0x1.ec7094p-2f;
  f29 = __sm_fma_rn(f27, f12, f28);
  f30 = -0x1.715476p-1f;
  f31 = __sm_fma_rn(f29, f12, f30);
  f32 = f12 * f31;
  f33 = f12 * f32;
  f34 = 0x1.715476p+0f;
  f35 = __sm_fma_rn(f12, f34, f33);
  f38 = f11 + f35;
  p2 = r2 < 0x7f800000u;
  if (p2) goto L5_2;
  f36 = __sm_f32bits(0x7f800000u);
  f38 = __sm_fma_rn(f1, f36, f36);
L5_2:;
  p3 = f1 == 0.0f;
  f37 = p3 ? __sm_f32bits(0xff800000u) : f38;
  __ret = f37;
  return __ret;
}
#endif

#if SASS_MATH_FAST
SASS_MATH_FN(float, log10f)(float a0) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = __sm_lg2_approx(f1);
  f3 = f2 * 0x1.344136p-2f;
  __ret = f3;
  return __ret;
}
#else
SASS_MATH_FN(float, log10f)(float a0) {
  float __ret;
  float f1, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35, f36;
  bool p1, p2, p3;
  unsigned r2, r3, r4, r5;
  f5 = a0;
  p1 = f5 < 0x1p-126f;
  f6 = f5 * 0x1p+23f;
  f1 = p1 ? f6 : f5;
  f7 = p1 ? -0x1.7p+4f : 0.0f;
  r2 = __float_as_uint(f1);
  r3 = r2 + 0xc0d55555u;
  r4 = r3 & 0xff800000u;
  r5 = r2 - r4;
  f8 = __sm_f32bits(r5);
  f9 = __sm_i2f_rn((int)r4);
  f10 = 0x1p-23f;
  f11 = __sm_fma_rn(f9, f10, f7);
  f12 = f8 + -0x1p+0f;
  f13 = 0x1.2073ecp-3f;
  f14 = -0x1.0aa04ep-3f;
  f15 = __sm_fma_rn(f14, f12, f13);
  f16 = -0x1.f19b98p-4f;
  f17 = __sm_fma_rn(f15, f12, f16);
  f18 = 0x1.1e52aap-3f;
  f19 = __sm_fma_rn(f17, f12, f18);
  f20 = -0x1.55b172p-3f;
  f21 = __sm_fma_rn(f19, f12, f20);
  f22 = 0x1.99da16p-3f;
  f23 = __sm_fma_rn(f21, f12, f22);
  f24 = -0x1.fffe44p-3f;
  f25 = __sm_fma_rn(f23, f12, f24);
  f26 = 0x1.5554fp-2f;
  f27 = __sm_fma_rn(f25, f12, f26);
  f28 = -0x1p-1f;
  f29 = __sm_fma_rn(f27, f12, f28);
  f30 = f12 * f29;
  f31 = __sm_fma_rn(f30, f12, f12);
  f32 = 0x1.62e43p-1f;
  f36 = __sm_fma_rn(f11, f32, f31);
  p2 = r2 < 0x7f800000u;
  if (p2) goto L6_2;
  f33 = __sm_f32bits(0x7f800000u);
  f36 = __sm_fma_rn(f1, f33, f33);
L6_2:;
  f34 = f36 * 0x1.bcb7b2p-2f;
  p3 = f1 == 0.0f;
  f35 = p3 ? __sm_f32bits(0xff800000u) : f34;
  __ret = f35;
  return __ret;
}
#endif

SASS_MATH_FN(float, log1pf)(float a0) {
  float __ret;
  float f1, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f39;
  bool p1, p2, p3;
  unsigned r2, r3, r4, r5, r6, r7, r8;
  f1 = a0;
  f7 = 0x1p+0f;
  f8 = __sm_add_rz(f1, f7);
  r2 = __float_as_uint(f8);
  r3 = r2 + 0xc0c00000u;
  r4 = r3 & 0xff800000u;
  r5 = __float_as_uint(f1);
  r6 = r5 - r4;
  f9 = __sm_f32bits(r6);
  r7 = 0x40800000u;
  r8 = r7 - r4;
  f10 = __sm_f32bits(r8);
  f11 = -0x1p+0f;
  f12 = 0x1p-2f;
  f13 = __sm_fma_rn(f12, f10, f11);
  f14 = f13 + f9;
  f15 = __sm_i2f_rn((int)r4);
  f16 = f15 * 0x1p-23f;
  f17 = 0x1.b00024p-4f;
  f18 = -0x1.737efp-5f;
  f19 = __sm_fma_rn(f18, f14, f17);
  f20 = -0x1.0ef1cp-3f;
  f21 = __sm_fma_rn(f19, f14, f20);
  f22 = 0x1.28c8eap-3f;
  f23 = __sm_fma_rn(f21, f14, f22);
  f24 = -0x1.54d1bap-3f;
  f25 = __sm_fma_rn(f23, f14, f24);
  f26 = 0x1.995f3cp-3f;
  f27 = __sm_fma_rn(f25, f14, f26);
  f28 = -0x1.000084p-2f;
  f29 = __sm_fma_rn(f27, f14, f28);
  f30 = 0x1.5555ccp-2f;
  f31 = __sm_fma_rn(f29, f14, f30);
  f32 = -0x1p-1f;
  f33 = __sm_fma_rn(f31, f14, f32);
  f34 = f14 * f33;
  f35 = __sm_fma_rn(f34, f14, f14);
  f36 = 0x1.62e43p-1f;
  f39 = __sm_fma_rn(f16, f36, f35);
  p1 = r5 < 0x7f800000u;
  if (p1) goto L7_4;
  p2 = (int)r5 < (int)0xbf800001u;
  if (p2) goto L7_3;
  f37 = __sm_f32bits(0x7f800000u);
  f39 = __sm_fma_rn(f1, f37, f37);
L7_3:;
  p3 = f1 == 0.0f;
  f39 = p3 ? -0.0f : f39;
L7_4:;
  __ret = f39;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, sqrtf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sqrt_approx(f1);
  __ret = f2;
  return __ret;
}
#else
SASS_MATH_FN(float, sqrtf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sqrt_rn(f1);
  __ret = f2;
  return __ret;
}
#endif

SASS_MATH_FN(float, rsqrtf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rsqrt_approx(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, cbrtf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16;
  bool p1, p2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = __sm_lg2_approx(f2);
  f4 = f3 * -0x1.555556p-1f;
  f5 = __sm_ex2_approx_ftz(f4);
  f6 = f2 * f5;
  f7 = f5 * f6;
  f8 = -f7;
  f9 = 0x1p+0f;
  f10 = __sm_fma_rn(f6, f8, f9);
  f11 = f10 * 0x1.555556p-2f;
  f12 = __sm_fma_rn(f6, f11, f6);
  p1 = f1 < 0.0f;
  f13 = -f12;
  f14 = p1 ? f13 : f12;
  f15 = f1 + f1;
  p2 = f15 == f1;
  f16 = p2 ? f15 : f14;
  __ret = f16;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, sinf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_sin_approx(f1);
  __ret = f2;
  return __ret;
}
#else
SASS_MATH_FN(float, sinf)(float a0) {
  float __ret;
  float f1, f3, f7, f8, f14, f15, f16, f17, f18, f19, f20, f21;
  float f22, f23, f25, f26, f27, f28, f29, f30, f31, f32, f34, f35;
  float f36, f37;
  double fd1, fd2;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p10;
  unsigned r2, r3, r4, r5, r8, r11, r18, r21, r22, r23, r24, r25;
  unsigned r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37;
  unsigned r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r50;
  unsigned r51, r52, r53, r54;
  unsigned long long rd1, rd17, rd18, rd19, rd20, rd21, rd22, rd23, rd24, rd28, rd29, rd30;
  unsigned __d[7];
  f1 = a0;
  rd1 = 0ull + 0ull;
  f14 = f1 * 0x1.45f306p-1f;
  r54 = __sm_f2i_rn(f14);
  f15 = __sm_i2f_rn((int)r54);
  f16 = -0x1.921fb4p+0f;
  f17 = __sm_fma_rn(f15, f16, f1);
  f18 = -0x1.4442dp-24f;
  f19 = __sm_fma_rn(f15, f18, f17);
  f20 = -0x1.84698ap-48f;
  f35 = __sm_fma_rn(f15, f20, f19);
  f3 = __sm_fabs(f1);
  p1 = !(f3 >= 0x1.9c8fp+16f);
  if (p1) goto L12_8;
  p2 = f3 == __sm_f32bits(0x7f800000u);
  if (p2) goto L12_7;
  goto L12_2;
L12_7:;
  f23 = 0.0f;
  f35 = __sm_mul_rn(f1, f23);
  r54 = 0u;
  goto L12_8;
L12_2:;
  r2 = __float_as_uint(f1);
  r21 = r2 >> 0x17u;
  r22 = r21 & 0xffu;
  r3 = r22 + 0xffffff80u;
  r23 = r2 << 8u;
  r4 = r23 | 0x80000000u;
  r5 = r3 >> 5u;
  rd30 = 0ull;
  r51 = 0u;
  rd29 = 0ull;
  rd28 = rd1;
L12_3:;
  r24 = __sm_i2opi_f[(rd29) >> 2];
  rd17 = (unsigned long long)r24 * (unsigned long long)r4 + rd30;
  rd30 = rd17 >> 0x20u;
  __d[(rd28) >> 2] = (unsigned)rd17;
  rd29 = rd29 + 4ull;
  rd28 = rd28 + 4ull;
  r51 = r51 + 1u;
  p3 = (int)r51 != (int)6u;
  if (p3) goto L12_3;
  __d[(rd1+24) >> 2] = (unsigned)rd30;
  r25 = 4u;
  r8 = r25 - r5;
  r26 = 6u;
  r27 = r26 - r5;
  rd18 = (long long)(int)r27 * (long long)(int)4u;
  rd19 = rd1 + rd18;
  r52 = __d[(rd19) >> 2];
  r53 = __d[(rd19-4) >> 2];
  r11 = r3 & 0x1fu;
  p4 = (int)r11 == (int)0u;
  if (p4) goto L12_6;
  r28 = 0x20u;
  r29 = r28 - r11;
  r30 = r53 >> r29;
  r31 = r52 << r11;
  r52 = r30 + r31;
  rd20 = (long long)(int)r8 * (long long)(int)4u;
  rd21 = rd1 + rd20;
  r32 = __d[(rd21) >> 2];
  r33 = r32 >> r29;
  r34 = r53 << r11;
  r53 = r33 + r34;
L12_6:;
  r35 = r2 & 0x80000000u;
  r36 = r53 >> 0x1eu;
  r37 = r52 << 2u;
  r38 = r36 | r37;
  r39 = r38 >> 0x1fu;
  r40 = r52 >> 0x1eu;
  r41 = r39 + r40;
  r42 = -r41;
  p5 = (int)r35 == (int)0u;
  r54 = p5 ? r41 : r42;
  p6 = (int)r39 != (int)0u;
  r43 = r35 ^ 0x80000000u;
  r44 = p6 ? r43 : r35;
  r45 = p6 ? 0xffffffffu : 0u;
  r46 = r38 ^ r45;
  r47 = r53 << 2u;
  r48 = r47 ^ r45;
  rd22 = (unsigned long long)r46;
  rd23 = (unsigned long long)r48;
  rd24 = __sm_bfi64(rd22, rd23, 0x20u, 0x20u);
  fd1 = (double)(long long)rd24;
  fd2 = fd1 * 0x1.921fb54442d19p-64;
  f21 = (float)fd2;
  p7 = (int)r44 == (int)0u;
  f22 = -f21;
  f35 = p7 ? f21 : f22;
L12_8:;
  r18 = r54 & 1u;
  p8 = (int)r18 == (int)0u;
  f7 = p8 ? f35 : 0x1p+0f;
  f8 = __sm_mul_rn(f35, f35);
  f36 = -0x1.9a82a6p-13f;
  if (p8) goto L12_10;
  f25 = -0x1.6c0fdap-10f;
  f26 = 0x1.9758p-16f;
  f36 = __sm_fma_rn(f26, f8, f25);
L12_10:;
  f27 = p8 ? 0x1.110bc8p-7f : 0x1.555576p-5f;
  f28 = __sm_fma_rn(f36, f8, f27);
  f29 = p8 ? -0x1.55555p-3f : -0x1.fffffep-2f;
  f30 = __sm_fma_rn(f28, f8, f29);
  f31 = 0.0f;
  f32 = __sm_fma_rn(f8, f7, f31);
  f37 = __sm_fma_rn(f30, f32, f7);
  r50 = r54 & 2u;
  p10 = (int)r50 == (int)0u;
  if (p10) goto L12_12;
  f34 = -0x1p+0f;
  f37 = __sm_fma_rn(f37, f34, f31);
L12_12:;
  __ret = f37;
  return __ret;
}
#endif

#if SASS_MATH_FAST
SASS_MATH_FN(float, cosf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_cos_approx(f1);
  __ret = f2;
  return __ret;
}
#else
SASS_MATH_FN(float, cosf)(float a0) {
  float __ret;
  float f1, f3, f7, f8, f14, f15, f16, f17, f18, f19, f20, f21;
  float f22, f23, f25, f26, f27, f28, f29, f30, f31, f32, f34, f35;
  float f36, f37;
  double fd1, fd2;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p10;
  unsigned r2, r3, r4, r5, r8, r11, r18, r19, r22, r23, r24, r25;
  unsigned r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37;
  unsigned r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r48, r49;
  unsigned r51, r52, r53, r54, r55;
  unsigned long long rd1, rd17, rd18, rd19, rd20, rd21, rd22, rd23, rd24, rd28, rd29, rd30;
  unsigned __d[7];
  f1 = a0;
  rd1 = 0ull + 0ull;
  f14 = f1 * 0x1.45f306p-1f;
  r55 = __sm_f2i_rn(f14);
  f15 = __sm_i2f_rn((int)r55);
  f16 = -0x1.921fb4p+0f;
  f17 = __sm_fma_rn(f15, f16, f1);
  f18 = -0x1.4442dp-24f;
  f19 = __sm_fma_rn(f15, f18, f17);
  f20 = -0x1.84698ap-48f;
  f35 = __sm_fma_rn(f15, f20, f19);
  f3 = __sm_fabs(f1);
  p1 = !(f3 >= 0x1.9c8fp+16f);
  if (p1) goto L13_8;
  p2 = f3 == __sm_f32bits(0x7f800000u);
  if (p2) goto L13_7;
  goto L13_2;
L13_7:;
  f23 = 0.0f;
  f35 = __sm_mul_rn(f1, f23);
  r55 = 0u;
  goto L13_8;
L13_2:;
  r2 = __float_as_uint(f1);
  r22 = r2 >> 0x17u;
  r23 = r22 & 0xffu;
  r3 = r23 + 0xffffff80u;
  r24 = r2 << 8u;
  r4 = r24 | 0x80000000u;
  r5 = r3 >> 5u;
  rd30 = 0ull;
  r52 = 0u;
  rd29 = 0ull;
  rd28 = rd1;
L13_3:;
  r25 = __sm_i2opi_f[(rd29) >> 2];
  rd17 = (unsigned long long)r25 * (unsigned long long)r4 + rd30;
  rd30 = rd17 >> 0x20u;
  __d[(rd28) >> 2] = (unsigned)rd17;
  rd29 = rd29 + 4ull;
  rd28 = rd28 + 4ull;
  r52 = r52 + 1u;
  p3 = (int)r52 != (int)6u;
  if (p3) goto L13_3;
  __d[(rd1+24) >> 2] = (unsigned)rd30;
  r26 = 4u;
  r8 = r26 - r5;
  r27 = 6u;
  r28 = r27 - r5;
  rd18 = (long long)(int)r28 * (long long)(int)4u;
  rd19 = rd1 + rd18;
  r53 = __d[(rd19) >> 2];
  r54 = __d[(rd19-4) >> 2];
  r11 = r3 & 0x1fu;
  p4 = (int)r11 == (int)0u;
  if (p4) goto L13_6;
  r29 = 0x20u;
  r30 = r29 - r11;
  r31 = r54 >> r30;
  r32 = r53 << r11;
  r53 = r31 + r32;
  rd20 = (long long)(int)r8 * (long long)(int)4u;
  rd21 = rd1 + rd20;
  r33 = __d[(rd21) >> 2];
  r34 = r33 >> r30;
  r35 = r54 << r11;
  r54 = r34 + r35;
L13_6:;
  r36 = r2 & 0x80000000u;
  r37 = r54 >> 0x1eu;
  r38 = r53 << 2u;
  r39 = r37 | r38;
  r40 = r39 >> 0x1fu;
  r41 = r53 >> 0x1eu;
  r42 = r40 + r41;
  r43 = -r42;
  p5 = (int)r36 == (int)0u;
  r55 = p5 ? r42 : r43;
  p6 = (int)r40 != (int)0u;
  r44 = r36 ^ 0x80000000u;
  r45 = p6 ? r44 : r36;
  r46 = p6 ? 0xffffffffu : 0u;
  r47 = r39 ^ r46;
  r48 = r54 << 2u;
  r49 = r48 ^ r46;
  rd22 = (unsigned long long)r47;
  rd23 = (unsigned long long)r49;
  rd24 = __sm_bfi64(rd22, rd23, 0x20u, 0x20u);
  fd1 = (double)(long long)rd24;
  fd2 = fd1 * 0x1.921fb54442d19p-64;
  f21 = (float)fd2;
  p7 = (int)r45 == (int)0u;
  f22 = -f21;
  f35 = p7 ? f21 : f22;
L13_8:;
  r18 = r55 + 1u;
  r19 = r18 & 1u;
  p8 = (int)r19 == (int)0u;
  f7 = p8 ? f35 : 0x1p+0f;
  f8 = __sm_mul_rn(f35, f35);
  f36 = -0x1.9a82a6p-13f;
  if (p8) goto L13_10;
  f25 = -0x1.6c0fdap-10f;
  f26 = 0x1.9758p-16f;
  f36 = __sm_fma_rn(f26, f8, f25);
L13_10:;
  f27 = p8 ? 0x1.110bc8p-7f : 0x1.555576p-5f;
  f28 = __sm_fma_rn(f36, f8, f27);
  f29 = p8 ? -0x1.55555p-3f : -0x1.fffffep-2f;
  f30 = __sm_fma_rn(f28, f8, f29);
  f31 = 0.0f;
  f32 = __sm_fma_rn(f8, f7, f31);
  f37 = __sm_fma_rn(f30, f32, f7);
  r51 = r18 & 2u;
  p10 = (int)r51 == (int)0u;
  if (p10) goto L13_12;
  f34 = -0x1p+0f;
  f37 = __sm_fma_rn(f37, f34, f31);
L13_12:;
  __ret = f37;
  return __ret;
}
#endif

#if SASS_MATH_FAST
SASS_MATH_FN(float, tanf)(float a0) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = __sm_sin_approx(f1);
  f3 = __sm_cos_approx(f1);
  f4 = __sm_div_approx(f2, f3);
  __ret = f4;
  return __ret;
}
#else
SASS_MATH_FN(float, tanf)(float a0) {
  float __ret;
  float f1, f3, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16;
  float f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28;
  float f29, f30, f31, f32, f33, f34, f35, f36;
  double fd1, fd2;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9;
  unsigned r2, r3, r4, r5, r8, r11, r20, r21, r22, r23, r24, r25;
  unsigned r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37;
  unsigned r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r49, r50;
  unsigned r51, r52, r53;
  unsigned long long rd1, rd17, rd18, rd19, rd20, rd21, rd22, rd23, rd24, rd28, rd29, rd30;
  unsigned __d[7];
  f1 = a0;
  rd1 = 0ull + 0ull;
  f7 = f1 * 0x1.45f306p-1f;
  r53 = __sm_f2i_rn(f7);
  f8 = __sm_i2f_rn((int)r53);
  f9 = -0x1.921fb4p+0f;
  f10 = __sm_fma_rn(f8, f9, f1);
  f11 = -0x1.4442dp-24f;
  f12 = __sm_fma_rn(f8, f11, f10);
  f13 = -0x1.84698ap-48f;
  f36 = __sm_fma_rn(f8, f13, f12);
  f3 = __sm_fabs(f1);
  p1 = !(f3 >= 0x1.9c8fp+16f);
  if (p1) goto L14_8;
  p2 = f3 == __sm_f32bits(0x7f800000u);
  if (p2) goto L14_7;
  goto L14_2;
L14_7:;
  f16 = 0.0f;
  f36 = __sm_mul_rn(f1, f16);
  r53 = 0u;
  goto L14_8;
L14_2:;
  r2 = __float_as_uint(f1);
  r20 = r2 >> 0x17u;
  r21 = r20 & 0xffu;
  r3 = r21 + 0xffffff80u;
  r22 = r2 << 8u;
  r4 = r22 | 0x80000000u;
  r5 = r3 >> 5u;
  rd30 = 0ull;
  r50 = 0u;
  rd29 = 0ull;
  rd28 = rd1;
L14_3:;
  r23 = __sm_i2opi_f[(rd29) >> 2];
  rd17 = (unsigned long long)r23 * (unsigned long long)r4 + rd30;
  rd30 = rd17 >> 0x20u;
  __d[(rd28) >> 2] = (unsigned)rd17;
  rd29 = rd29 + 4ull;
  rd28 = rd28 + 4ull;
  r50 = r50 + 1u;
  p3 = (int)r50 != (int)6u;
  if (p3) goto L14_3;
  __d[(rd1+24) >> 2] = (unsigned)rd30;
  r24 = 4u;
  r8 = r24 - r5;
  r25 = 6u;
  r26 = r25 - r5;
  rd18 = (long long)(int)r26 * (long long)(int)4u;
  rd19 = rd1 + rd18;
  r51 = __d[(rd19) >> 2];
  r52 = __d[(rd19-4) >> 2];
  r11 = r3 & 0x1fu;
  p4 = (int)r11 == (int)0u;
  if (p4) goto L14_6;
  r27 = 0x20u;
  r28 = r27 - r11;
  r29 = r52 >> r28;
  r30 = r51 << r11;
  r51 = r29 + r30;
  rd20 = (long long)(int)r8 * (long long)(int)4u;
  rd21 = rd1 + rd20;
  r31 = __d[(rd21) >> 2];
  r32 = r31 >> r28;
  r33 = r52 << r11;
  r52 = r32 + r33;
L14_6:;
  r34 = r2 & 0x80000000u;
  r35 = r52 >> 0x1eu;
  r36 = r51 << 2u;
  r37 = r35 | r36;
  r38 = r37 >> 0x1fu;
  r39 = r51 >> 0x1eu;
  r40 = r38 + r39;
  r41 = -r40;
  p5 = (int)r34 == (int)0u;
  r53 = p5 ? r40 : r41;
  p6 = (int)r38 != (int)0u;
  r42 = r34 ^ 0x80000000u;
  r43 = p6 ? r42 : r34;
  r44 = p6 ? 0xffffffffu : 0u;
  r45 = r37 ^ r44;
  r46 = r52 << 2u;
  r47 = r46 ^ r44;
  rd22 = (unsigned long long)r45;
  rd23 = (unsigned long long)r47;
  rd24 = __sm_bfi64(rd22, rd23, 0x20u, 0x20u);
  fd1 = (double)(long long)rd24;
  fd2 = fd1 * 0x1.921fb54442d19p-64;
  f14 = (float)fd2;
  p7 = (int)r43 == (int)0u;
  f15 = -f14;
  f36 = p7 ? f14 : f15;
L14_8:;
  f17 = f36 * f36;
  f18 = 0x1.acp-9f;
  f19 = 0x1.32p-7f;
  f20 = __sm_fma_rn(f19, f17, f18);
  f21 = 0x1.8ep-6f;
  f22 = __sm_fma_rn(f20, f17, f21);
  f23 = 0x1.b6p-5f;
  f24 = __sm_fma_rn(f22, f17, f23);
  f25 = 0x1.11287p-3f;
  f26 = __sm_fma_rn(f24, f17, f25);
  f27 = 0x1.55551p-2f;
  f28 = __sm_fma_rn(f26, f17, f27);
  f29 = __sm_mul_rn(f17, f36);
  f30 = __sm_fma_rn(f28, f29, f36);
  f31 = __sm_fabs(f36);
  p8 = f31 == 0x1.016878p-11f;
  f32 = p8 ? f36 : f30;
  r49 = r53 & 1u;
  p9 = r49 == 1u;
  f33 = -f32;
  f34 = __sm_rcp_approx_ftz(f33);
  f35 = p9 ? f34 : f32;
  __ret = f35;
  return __ret;
}
#endif

SASS_MATH_FN(float, tanhf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23;
  bool p1, p2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = f2 * 0x1.715476p+1f;
  f4 = __sm_ex2_approx_ftz(f3);
  f5 = f4 + 0x1p+0f;
  f6 = 0x1p+0f;
  f7 = __sm_rcp_approx_ftz(f5);
  f8 = -0x1p+1f;
  f9 = __sm_fma_rn(f7, f8, f6);
  p1 = f2 >= 0x1.205968p+3f;
  f10 = p1 ? 0x1p+0f : f9;
  f11 = __sm_copysign(f10, f1);
  f12 = f1 * f1;
  f13 = -0x1.ac795cp-5f;
  f14 = 0x1.01e104p-6f;
  f15 = __sm_fma_rn(f14, f12, f13);
  f16 = 0x1.10b282p-3f;
  f17 = __sm_fma_rn(f15, f12, f16);
  f18 = -0x1.5553dap-2f;
  f19 = __sm_fma_rn(f17, f12, f18);
  f20 = 0.0f;
  f21 = __sm_fma_rn(f19, f12, f20);
  f22 = __sm_fma_rn(f21, f1, f1);
  p2 = f2 >= 0x1.333334p-1f;
  f23 = p2 ? f11 : f22;
  __ret = f23;
  return __ret;
}

SASS_MATH_FN(float, erff)(float a0) {
  float __ret;
  float f1, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26;
  bool p1, p2;
  f1 = a0;
  f5 = __sm_fabs(f1);
  p1 = !(f5 >= 0x1.00c1fcp+0f);
  p2 = f5 >= 0x1.00c1fcp+0f;
  f6 = f1 * f1;
  f7 = p2 ? f5 : f6;
  f8 = p2 ? 0x1.d69874p-14f : 0x1.63d2d4p-14f;
  f9 = p2 ? -0x1.5c00b6p-10f : -0x1.ae9a4p-11f;
  f10 = __sm_fma_rn(f8, f7, f9);
  f11 = p2 ? 0x1.13233ep-7f : 0x1.55abd4p-8f;
  f12 = __sm_fma_rn(f10, f7, f11);
  f13 = p2 ? -0x1.49b334p-5f : -0x1.b837cep-6f;
  f14 = __sm_fma_rn(f12, f7, f13);
  f15 = p2 ? 0x1.46aa32p-3f : 0x1.ce315ep-4f;
  f16 = __sm_fma_rn(f14, f7, f15);
  f17 = p2 ? 0x1.d369f2p-1f : -0x1.812758p-2f;
  f18 = __sm_fma_rn(f16, f7, f17);
  f19 = p2 ? 0x1.421428p-1f : 0x1.06eba6p-3f;
  f20 = __sm_fma_rn(f18, f7, f19);
  f21 = -f5;
  f22 = p2 ? f21 : f1;
  f26 = __sm_fma_rn(f20, f22, f22);
  if (p1) goto L16_2;
  f23 = __sm_ex2_approx_ftz(f26);
  f24 = 0x1p+0f;
  f25 = f24 - f23;
  f26 = __sm_copysign(f25, f1);
L16_2:;
  __ret = f26;
  return __ret;
}

SASS_MATH_FN(float, erfcf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24;
  float f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36;
  float f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47, f48;
  float f49, f50, f51, f52, f53, f54, f55, f56, f57, f58, f59, f60;
  float f61, f62, f63, f64, f65, f66, f67;
  bool p1, p2, p3;
  unsigned r2, r3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = f2 + -0x1p+2f;
  f4 = -0x1p+2f;
  f5 = f2 + 0x1p+2f;
  f6 = __sm_rcp_approx_ftz(f5);
  f7 = __sm_mul_rn(f3, f6);
  f8 = f7 + 0x1p+0f;
  f9 = 0x1p+0f;
  f10 = __sm_fma_rn(f4, f8, f2);
  f11 = -f7;
  f12 = __sm_fma_rn(f11, f2, f10);
  f13 = __sm_fma_rn(f6, f12, f7);
  f14 = 0x1.cdc0b6p-8f;
  f15 = 0x1.d34122p-11f;
  f16 = __sm_fma_rn(f15, f13, f14);
  f17 = -0x1.03f696p-6f;
  f18 = __sm_fma_rn(f16, f13, f17);
  f19 = 0x1.2a6e76p-5f;
  f20 = __sm_fma_rn(f18, f13, f19);
  f21 = -0x1.10f8b4p-4f;
  f22 = __sm_fma_rn(f20, f13, f21);
  f23 = 0x1.8043aap-4f;
  f24 = __sm_fma_rn(f22, f13, f23);
  f25 = -0x1.9da848p-4f;
  f26 = __sm_fma_rn(f24, f13, f25);
  f27 = 0x1.16e9bcp-4f;
  f28 = __sm_fma_rn(f26, f13, f27);
  f29 = 0x1.f7e2ep-7f;
  f30 = __sm_fma_rn(f28, f13, f29);
  f31 = -0x1.1df1a8p-3f;
  f32 = __sm_fma_rn(f30, f13, f31);
  f33 = 0x1.3ba592p+0f;
  f34 = __sm_fma_rn(f32, f13, f33);
  f35 = 0x1p+1f;
  f36 = __sm_fma_rn(f35, f2, f9);
  f37 = __sm_rcp_approx_ftz(f36);
  f38 = __sm_mul_rn(f34, f37);
  f39 = f38 * -0x1p+1f;
  f40 = __sm_fma_rn(f2, f39, f34);
  f41 = f40 - f38;
  f42 = __sm_fma_rn(f41, f37, f38);
  f43 = f2 * f2;
  f44 = -f43;
  f45 = 0x1.715476p+0f;
  f46 = __sm_mul_rn(f44, f45);
  f47 = __sm_trunc(f46);
  f48 = __sm_fabs(f47);
  p1 = f48 > 0x1.f8p+6f;
  f49 = 0x1.f8p+6f;
  f50 = __sm_copysign(f49, f47);
  f51 = p1 ? f50 : f47;
  f52 = -0x1.62e43p-1f;
  f53 = __sm_fma_rn(f51, f52, f44);
  f54 = 0x1.05c61p-29f;
  f55 = __sm_fma_rn(f51, f54, f53);
  f56 = f55 * 0x1.715476p+0f;
  f57 = f51 + 0x1.8000fep+23f;
  r2 = __float_as_uint(f57);
  r3 = r2 << 0x17u;
  f58 = __sm_f32bits(r3);
  f59 = __sm_ex2_approx_ftz(f56);
  f60 = f59 * f58;
  f61 = -f2;
  f62 = __sm_fma_rn(f61, f2, f43);
  f63 = __sm_fma_rn(f60, f62, f60);
  f64 = f42 * f63;
  p2 = f2 > 0x1.41c29p+3f;
  f65 = p2 ? 0.0f : f64;
  p3 = f1 < 0.0f;
  f66 = f35 - f65;
  f67 = p3 ? f66 : f65;
  __ret = f67;
  return __ret;
}

SASS_MATH_FN(float, atanf)(float a0) {
  float __ret;
  float f1, f2, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30;
  bool p1, p2, p3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = !(f2 > 0x1p+0f);
  p2 = f2 > 0x1p+0f;
  f6 = __sm_rcp_approx_ftz(f2);
  f7 = p2 ? f6 : f2;
  f8 = f7 * f7;
  f9 = -0x1.d7c29ep-7f;
  f10 = 0x1.412154p-9f;
  f11 = __sm_fma_rn(f10, f8, f9);
  f12 = 0x1.4672fcp-5f;
  f13 = __sm_fma_rn(f11, f8, f12);
  f14 = -0x1.2914f4p-4f;
  f15 = __sm_fma_rn(f13, f8, f14);
  f16 = 0x1.aed642p-4f;
  f17 = __sm_fma_rn(f15, f8, f16);
  f18 = -0x1.223d1p-3f;
  f19 = __sm_fma_rn(f17, f8, f18);
  f20 = 0x1.995ecp-3f;
  f21 = __sm_fma_rn(f19, f8, f20);
  f22 = -0x1.55544ep-2f;
  f23 = __sm_fma_rn(f21, f8, f22);
  f24 = f8 * f23;
  f30 = __sm_fma_rn(f24, f7, f7);
  if (p1) goto L18_2;
  f25 = -f30;
  f26 = 0x1.aee9d6p+0f;
  f27 = 0x1.ddcb02p-1f;
  f30 = __sm_fma_rn(f27, f26, f25);
L18_2:;
  f28 = __sm_copysign(f30, f1);
  p3 = f2 <= __sm_f32bits(0x7f800000u);
  f29 = p3 ? f28 : f30;
  __ret = f29;
  return __ret;
}

SASS_MATH_FN(float, asinf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24;
  float f25, f26, f27, f28, f29, f30, f31, f32;
  bool p1, p2, p3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = -f2;
  f4 = 0x1p-1f;
  f5 = __sm_fma_rn(f4, f3, f4);
  f6 = __sm_rsqrt_approx_ftz(f5);
  f7 = f5 * f6;
  f8 = f6 * 0x1p-1f;
  f9 = -f7;
  f10 = __sm_fma_rn(f9, f8, f4);
  f11 = __sm_fma_rn(f7, f10, f7);
  p1 = f2 == 0x1p+0f;
  f12 = p1 ? 0.0f : f11;
  p2 = f2 > 0x1.1eb852p-1f;
  f13 = p2 ? f12 : f2;
  f14 = f13 * f13;
  f15 = 0x1.33952ep-6f;
  f16 = 0x1.9ba5eep-5f;
  f17 = __sm_fma_rn(f16, f14, f15);
  f18 = 0x1.7f21dp-5f;
  f19 = __sm_fma_rn(f17, f14, f18);
  f20 = 0x1.32799ep-4f;
  f21 = __sm_fma_rn(f19, f14, f20);
  f22 = 0x1.555808p-3f;
  f23 = __sm_fma_rn(f21, f14, f22);
  f24 = f14 * f23;
  f25 = __sm_fma_rn(f24, f13, f13);
  f26 = f25 * -0x1p+1f;
  f27 = 0x1.aee9d6p+0f;
  f28 = 0x1.ddcb02p-1f;
  f29 = __sm_fma_rn(f28, f27, f26);
  f30 = p2 ? f29 : f25;
  p3 = f30 <= __sm_f32bits(0x7f800000u);
  f31 = __sm_copysign(f30, f1);
  f32 = p3 ? f31 : f30;
  __ret = f32;
  return __ret;
}

SASS_MATH_FN(float, acosf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24;
  float f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36;
  bool p1, p2, p3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = -f2;
  f4 = 0x1p-1f;
  f5 = __sm_fma_rn(f4, f3, f4);
  f6 = __sm_rsqrt_approx_ftz(f5);
  f7 = f5 * f6;
  f8 = f6 * 0x1p-1f;
  f9 = -f7;
  f10 = __sm_fma_rn(f9, f8, f4);
  f11 = __sm_fma_rn(f7, f10, f7);
  p1 = f2 == 0x1p+0f;
  f12 = p1 ? 0.0f : f11;
  p2 = f2 > 0x1.1eb852p-1f;
  f13 = p2 ? f12 : f2;
  f14 = __sm_copysign(f13, f1);
  f15 = f14 * f14;
  f16 = 0x1.163576p-6f;
  f17 = 0x1.21d9dep-5f;
  f18 = __sm_fma_rn(f17, f15, f16);
  f19 = 0x1.f80518p-6f;
  f20 = __sm_fma_rn(f18, f15, f19);
  f21 = 0x1.6e4272p-5f;
  f22 = __sm_fma_rn(f20, f15, f21);
  f23 = 0x1.3327b6p-4f;
  f24 = __sm_fma_rn(f22, f15, f23);
  f25 = 0x1.55558cp-3f;
  f26 = __sm_fma_rn(f24, f15, f25);
  f27 = f15 * f26;
  f28 = __sm_fma_rn(f27, f14, f14);
  f29 = -f28;
  f30 = p2 ? f28 : f29;
  f31 = 0x1.aee9d6p+0f;
  f32 = 0x1.ddcb02p-1f;
  f33 = __sm_fma_rn(f32, f31, f30);
  p3 = f1 > 0x1.1eb852p-1f;
  f34 = p3 ? f28 : f33;
  f35 = f34 + f34;
  f36 = p2 ? f35 : f34;
  __ret = f36;
  return __ret;
}

SASS_MATH_FN(float, sinhf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24;
  float f25, f26, f27, f28, f29, f30, f31, f32, f33, f34;
  bool p1, p2;
  unsigned r2, r3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = 0x1.715476p+0f;
  f4 = __sm_mul_rn(f2, f3);
  f5 = __sm_trunc(f4);
  f6 = __sm_fabs(f5);
  p1 = f6 > 0x1.f8p+6f;
  f7 = 0x1.f8p+6f;
  f8 = __sm_copysign(f7, f5);
  f9 = p1 ? f8 : f5;
  f10 = -0x1.62e43p-1f;
  f11 = __sm_fma_rn(f9, f10, f2);
  f12 = 0x1.05c61p-29f;
  f13 = __sm_fma_rn(f9, f12, f11);
  f14 = f13 * 0x1.715476p+0f;
  f15 = f9 + 0x1.8000fap+23f;
  r2 = __float_as_uint(f15);
  r3 = r2 << 0x17u;
  f16 = __sm_f32bits(r3);
  f17 = __sm_ex2_approx_ftz(f14);
  f18 = f17 * f16;
  f19 = -0x1p-3f;
  f20 = __sm_div_approx_ftz(f19, f18);
  f21 = 0x1p+1f;
  f22 = __sm_fma_rn(f21, f18, f20);
  f23 = f1 * f1;
  f24 = 0x1.9ffe92p-13f;
  f25 = 0x1.7a15b4p-19f;
  f26 = __sm_fma_rn(f25, f23, f24);
  f27 = 0x1.111134p-7f;
  f28 = __sm_fma_rn(f26, f23, f27);
  f29 = 0x1.555556p-3f;
  f30 = __sm_fma_rn(f28, f23, f29);
  f31 = f23 * f30;
  f32 = __sm_fma_rn(f31, f1, f1);
  p2 = f2 >= 0x1p+0f;
  f33 = __sm_copysign(f22, f1);
  f34 = p2 ? f33 : f32;
  __ret = f34;
  return __ret;
}

SASS_MATH_FN(float, coshf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17, f18, f19, f20, f21, f22;
  bool p1;
  unsigned r2, r3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = 0x1.715476p+0f;
  f4 = __sm_mul_rn(f2, f3);
  f5 = __sm_trunc(f4);
  f6 = __sm_fabs(f5);
  p1 = f6 > 0x1.f8p+6f;
  f7 = 0x1.f8p+6f;
  f8 = __sm_copysign(f7, f5);
  f9 = p1 ? f8 : f5;
  f10 = -0x1.62e43p-1f;
  f11 = __sm_fma_rn(f9, f10, f2);
  f12 = 0x1.05c61p-29f;
  f13 = __sm_fma_rn(f9, f12, f11);
  f14 = f13 * 0x1.715476p+0f;
  f15 = f9 + 0x1.8000fap+23f;
  r2 = __float_as_uint(f15);
  r3 = r2 << 0x17u;
  f16 = __sm_f32bits(r3);
  f17 = __sm_ex2_approx_ftz(f14);
  f18 = f17 * f16;
  f19 = 0x1p-3f;
  f20 = __sm_div_approx_ftz(f19, f18);
  f21 = 0x1p+1f;
  f22 = __sm_fma_rn(f21, f18, f20);
  __ret = f22;
  return __ret;
}

SASS_MATH_FN(float, floorf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_floor(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, ceilf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_ceil(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, truncf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_trunc(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, roundf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5;
  f1 = a0;
  f2 = 0x1p-1f;
  f3 = __sm_copysign(f2, f1);
  f4 = __sm_add_rz(f1, f3);
  f5 = __sm_trunc(f4);
  __ret = f5;
  return __ret;
}

SASS_MATH_FN(float, rintf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rint(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, nearbyintf)(float a0) {
  float __ret;
  float f1, f2;
  f1 = a0;
  f2 = __sm_rint(f1);
  __ret = f2;
  return __ret;
}

SASS_MATH_FN(float, asinhf)(float a0) {
  float __ret;
  float f1, f2, f3, f9, f10, f11, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41;
  float f42, f43, f44, f45, f46, f47, f48, f49, f51;
  bool p1, p2, p3, p4, p6;
  unsigned r1, r3, r4, r5, r6, r7, r8;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f9 = 0x1p+0f;
  f10 = __sm_fma_rn(f1, f1, f9);
  f11 = __sm_rsqrt_approx_ftz(f10);
  f12 = __sm_fma_rn(f10, f11, f9);
  f13 = __sm_rcp_approx_ftz(f12);
  f14 = f2 * f13;
  f15 = __sm_fma_rn(f2, f14, f2);
  p1 = f2 > 0x1p+23f;
  f3 = p1 ? f2 : f15;
  f16 = __sm_add_rz(f3, f9);
  r3 = __float_as_uint(f16);
  r4 = r3 + 0xc0c00000u;
  r5 = r4 & 0xff800000u;
  r1 = __float_as_uint(f3);
  r6 = r1 - r5;
  f17 = __sm_f32bits(r6);
  r7 = 0x40800000u;
  r8 = r7 - r5;
  f18 = __sm_f32bits(r8);
  f19 = -0x1p+0f;
  f20 = 0x1p-2f;
  f21 = __sm_fma_rn(f20, f18, f19);
  f22 = f21 + f17;
  f23 = __sm_i2f_rn((int)r5);
  f24 = f23 * 0x1p-23f;
  f25 = 0x1.b00024p-4f;
  f26 = -0x1.737efp-5f;
  f27 = __sm_fma_rn(f26, f22, f25);
  f28 = -0x1.0ef1cp-3f;
  f29 = __sm_fma_rn(f27, f22, f28);
  f30 = 0x1.28c8eap-3f;
  f31 = __sm_fma_rn(f29, f22, f30);
  f32 = -0x1.54d1bap-3f;
  f33 = __sm_fma_rn(f31, f22, f32);
  f34 = 0x1.995f3cp-3f;
  f35 = __sm_fma_rn(f33, f22, f34);
  f36 = -0x1.000084p-2f;
  f37 = __sm_fma_rn(f35, f22, f36);
  f38 = 0x1.5555ccp-2f;
  f39 = __sm_fma_rn(f37, f22, f38);
  f40 = -0x1p-1f;
  f41 = __sm_fma_rn(f39, f22, f40);
  f42 = f22 * f41;
  f43 = __sm_fma_rn(f42, f22, f22);
  f44 = 0x1.62e43p-1f;
  f51 = __sm_fma_rn(f24, f44, f43);
  p2 = r1 < 0x7f800000u;
  if (p2) goto L39_4;
  p3 = (int)r1 < (int)0xbf800001u;
  if (p3) goto L39_3;
  f45 = __sm_f32bits(0x7f800000u);
  f51 = __sm_fma_rn(f3, f45, f45);
L39_3:;
  p4 = f3 == 0.0f;
  f51 = p4 ? -0.0f : f51;
L39_4:;
  f46 = f51 + 0x1.62e43p-1f;
  f47 = p1 ? f46 : f51;
  f48 = __sm_copysign(f47, f1);
  p6 = f2 <= __sm_f32bits(0x7f800000u);
  f49 = p6 ? f48 : f47;
  __ret = f49;
  return __ret;
}

SASS_MATH_FN(float, acoshf)(float a0) {
  float __ret;
  float f1, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41;
  float f42, f43, f44, f45, f46, f47, f48, f49, f50, f52;
  bool p1, p2, p3, p4, p5;
  unsigned r1, r2, r4, r5, r6, r7, r8, r9;
  f7 = a0;
  f8 = 0x1p+0f;
  f9 = __sm_add_rn(f7, -(f8));
  r1 = __float_as_uint(f9);
  p1 = r1 > 0x4b000000u;
  f10 = -0x1p+0f;
  f11 = __sm_fma_rn(f7, f7, f10);
  f12 = __sm_rsqrt_approx_ftz(f11);
  f13 = f11 * f12;
  f14 = f12 * 0x1p-1f;
  f15 = -f13;
  f16 = __sm_fma_rn(f15, f13, f11);
  f17 = __sm_fma_rn(f16, f14, f13);
  p2 = f11 == 0.0f;
  f18 = p2 ? 0.0f : f17;
  f19 = p1 ? -0x1.000002p+0f : f18;
  f1 = f9 + f19;
  f20 = __sm_add_rz(f1, f8);
  r4 = __float_as_uint(f20);
  r5 = r4 + 0xc0c00000u;
  r6 = r5 & 0xff800000u;
  r2 = __float_as_uint(f1);
  r7 = r2 - r6;
  f21 = __sm_f32bits(r7);
  r8 = 0x40800000u;
  r9 = r8 - r6;
  f22 = __sm_f32bits(r9);
  f23 = 0x1p-2f;
  f24 = __sm_fma_rn(f23, f22, f10);
  f25 = f24 + f21;
  f26 = __sm_i2f_rn((int)r6);
  f27 = f26 * 0x1p-23f;
  f28 = 0x1.b00024p-4f;
  f29 = -0x1.737efp-5f;
  f30 = __sm_fma_rn(f29, f25, f28);
  f31 = -0x1.0ef1cp-3f;
  f32 = __sm_fma_rn(f30, f25, f31);
  f33 = 0x1.28c8eap-3f;
  f34 = __sm_fma_rn(f32, f25, f33);
  f35 = -0x1.54d1bap-3f;
  f36 = __sm_fma_rn(f34, f25, f35);
  f37 = 0x1.995f3cp-3f;
  f38 = __sm_fma_rn(f36, f25, f37);
  f39 = -0x1.000084p-2f;
  f40 = __sm_fma_rn(f38, f25, f39);
  f41 = 0x1.5555ccp-2f;
  f42 = __sm_fma_rn(f40, f25, f41);
  f43 = -0x1p-1f;
  f44 = __sm_fma_rn(f42, f25, f43);
  f45 = f25 * f44;
  f46 = __sm_fma_rn(f45, f25, f25);
  f47 = 0x1.62e43p-1f;
  f52 = __sm_fma_rn(f27, f47, f46);
  p3 = r2 < 0x7f800000u;
  if (p3) goto L40_4;
  p4 = (int)r2 < (int)0xbf800001u;
  if (p4) goto L40_3;
  f48 = __sm_f32bits(0x7f800000u);
  f52 = __sm_fma_rn(f1, f48, f48);
L40_3:;
  p5 = f1 == 0.0f;
  f52 = p5 ? -0.0f : f52;
L40_4:;
  f49 = f52 + 0x1.62e43p-1f;
  f50 = p1 ? f49 : f52;
  __ret = f50;
  return __ret;
}

SASS_MATH_FN(float, atanhf)(float a0) {
  float __ret;
  float f1, f2, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41;
  float f42, f43, f44, f45, f46, f48;
  bool p1, p2, p3, p4;
  unsigned r1, r3, r4, r5, r6, r7, r8;
  f1 = a0;
  f8 = __sm_fabs(f1);
  f9 = 0x1p+0f;
  f10 = f9 - f8;
  f11 = __sm_rcp_approx_ftz(f10);
  f12 = f11 + f11;
  f13 = f8 * f12;
  p1 = f8 > 0x1p+126f;
  f2 = p1 ? -0x1p+1f : f13;
  f14 = __sm_add_rz(f2, f9);
  r3 = __float_as_uint(f14);
  r4 = r3 + 0xc0c00000u;
  r5 = r4 & 0xff800000u;
  r1 = __float_as_uint(f2);
  r6 = r1 - r5;
  f15 = __sm_f32bits(r6);
  r7 = 0x40800000u;
  r8 = r7 - r5;
  f16 = __sm_f32bits(r8);
  f17 = -0x1p+0f;
  f18 = 0x1p-2f;
  f19 = __sm_fma_rn(f18, f16, f17);
  f20 = f19 + f15;
  f21 = __sm_i2f_rn((int)r5);
  f22 = f21 * 0x1p-23f;
  f23 = 0x1.b00024p-4f;
  f24 = -0x1.737efp-5f;
  f25 = __sm_fma_rn(f24, f20, f23);
  f26 = -0x1.0ef1cp-3f;
  f27 = __sm_fma_rn(f25, f20, f26);
  f28 = 0x1.28c8eap-3f;
  f29 = __sm_fma_rn(f27, f20, f28);
  f30 = -0x1.54d1bap-3f;
  f31 = __sm_fma_rn(f29, f20, f30);
  f32 = 0x1.995f3cp-3f;
  f33 = __sm_fma_rn(f31, f20, f32);
  f34 = -0x1.000084p-2f;
  f35 = __sm_fma_rn(f33, f20, f34);
  f36 = 0x1.5555ccp-2f;
  f37 = __sm_fma_rn(f35, f20, f36);
  f38 = -0x1p-1f;
  f39 = __sm_fma_rn(f37, f20, f38);
  f40 = f20 * f39;
  f41 = __sm_fma_rn(f40, f20, f20);
  f42 = 0x1.62e43p-1f;
  f48 = __sm_fma_rn(f22, f42, f41);
  p2 = r1 < 0x7f800000u;
  if (p2) goto L41_4;
  p3 = (int)r1 < (int)0xbf800001u;
  if (p3) goto L41_3;
  f43 = __sm_f32bits(0x7f800000u);
  f48 = __sm_fma_rn(f2, f43, f43);
L41_3:;
  p4 = f2 == 0.0f;
  f48 = p4 ? -0.0f : f48;
L41_4:;
  f44 = 0x1p-1f;
  f45 = __sm_copysign(f44, f1);
  f46 = f45 * f48;
  __ret = f46;
  return __ret;
}

SASS_MATH_FN(float, erfinvf)(float a0) {
  float __ret;
  float f1, f2, f3, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39;
  float f40, f41, f42, f43, f44;
  bool p1, p2;
  f1 = a0;
  f7 = -f1;
  f8 = 0x1p+0f;
  f9 = __sm_fma_rn(f1, f7, f8);
  f2 = __sm_lg2_approx_ftz(f9);
  f3 = -f2;
  f10 = 0x1.43ec8ap-27f;
  f11 = -0x1.14c6ep-32f;
  f12 = __sm_fma_rn(f11, f3, f10);
  f13 = -0x1.02dfb4p-23f;
  f14 = __sm_fma_rn(f12, f3, f13);
  f15 = 0x1.d1f08cp-23f;
  f16 = __sm_fma_rn(f14, f3, f15);
  f17 = 0x1.0e8554p-17f;
  f18 = __sm_fma_rn(f16, f3, f17);
  f19 = -0x1.09b69ap-15f;
  f20 = __sm_fma_rn(f18, f3, f19);
  f21 = -0x1.95fd42p-11f;
  f22 = __sm_fma_rn(f20, f3, f21);
  f23 = 0x1.6b804ep-8f;
  f24 = __sm_fma_rn(f22, f3, f23);
  f25 = 0x1.495c1ep-3f;
  f26 = __sm_fma_rn(f24, f3, f25);
  f27 = 0x1.c5bf88p-1f;
  f28 = __sm_fma_rn(f26, f3, f27);
  f44 = f1 * f28;
  p1 = !(f2 < -0x1.066666p+3f);
  if (p1) goto L42_2;
  f29 = __sm_rsqrt_approx_ftz(f3);
  f30 = -0x1.53754ap-1f;
  f31 = -0x1.2e0942p-1f;
  f32 = __sm_fma_rn(f31, f29, f30);
  f33 = 0x1.98d5b8p+0f;
  f34 = __sm_fma_rn(f32, f29, f33);
  f35 = -0x1.59b5dap-1f;
  f36 = __sm_fma_rn(f34, f29, f35);
  f37 = -0x1.860a6ep-4f;
  f38 = __sm_fma_rn(f36, f29, f37);
  f39 = 0x1.abb372p-1f;
  f40 = __sm_fma_rn(f38, f29, f39);
  f41 = f29 * f3;
  f42 = f41 * f40;
  p2 = f2 == __sm_f32bits(0xff800000u);
  f43 = p2 ? f3 : f42;
  f44 = __sm_copysign(f43, f1);
L42_2:;
  __ret = f44;
  return __ret;
}

SASS_MATH_FN(float, erfcinvf)(float a0) {
  float __ret;
  float f1, f2, f3, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39;
  float f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51;
  float f52, f53, f54, f55;
  bool p1, p2, p3, p4, p5;
  f1 = a0;
  f2 = -f1;
  f7 = 0x1p+1f;
  f3 = __sm_add_rn(f7, f2);
  p1 = f1 >= 0x1.bda512p-9f;
  p2 = f1 <= 0x1.ff212ep+0f;
  p3 = p1 && p2;
  if (p3) goto L43_2;
  goto L43_1;
L43_2:;
  f33 = __sm_mul_rn(f3, f1);
  f34 = __sm_lg2_approx_ftz(f33);
  f35 = -f34;
  f36 = 0x1.43ec8ap-27f;
  f37 = -0x1.14c6ep-32f;
  f38 = __sm_fma_rn(f37, f35, f36);
  f39 = -0x1.02dfb4p-23f;
  f40 = __sm_fma_rn(f38, f35, f39);
  f41 = 0x1.d1f08cp-23f;
  f42 = __sm_fma_rn(f40, f35, f41);
  f43 = 0x1.0e8554p-17f;
  f44 = __sm_fma_rn(f42, f35, f43);
  f45 = -0x1.09b69ap-15f;
  f46 = __sm_fma_rn(f44, f35, f45);
  f47 = -0x1.95fd42p-11f;
  f48 = __sm_fma_rn(f46, f35, f47);
  f49 = 0x1.6b804ep-8f;
  f50 = __sm_fma_rn(f48, f35, f49);
  f51 = 0x1.495c1ep-3f;
  f52 = __sm_fma_rn(f50, f35, f51);
  f53 = 0x1.c5bf88p-1f;
  f54 = __sm_fma_rn(f52, f35, f53);
  f55 = __sm_fma_rn(f54, f2, f54);
  goto L43_3;
L43_1:;
  p4 = f1 > 0x1p+0f;
  f8 = p4 ? f3 : f1;
  f9 = __sm_lg2_approx(f8);
  f10 = -f9;
  f11 = __sm_rsqrt_approx_ftz(f10);
  f12 = 0x1.fdf052p+6f;
  f13 = -0x1.f8e7e2p+5f;
  f14 = __sm_fma_rn(f13, f11, f12);
  f15 = -0x1.c86c38p+6f;
  f16 = __sm_fma_rn(f14, f11, f15);
  f17 = 0x1.e29b36p+5f;
  f18 = __sm_fma_rn(f16, f11, f17);
  f19 = -0x1.5ca366p+4f;
  f20 = __sm_fma_rn(f18, f11, f19);
  f21 = 0x1.9dea08p+2f;
  f22 = __sm_fma_rn(f20, f11, f21);
  f23 = -0x1.d53c0ap+0f;
  f24 = __sm_fma_rn(f22, f11, f23);
  f25 = -0x1.f0e3e8p-6f;
  f26 = __sm_fma_rn(f24, f11, f25);
  f27 = 0x1.aa6eeap-1f;
  f28 = __sm_fma_rn(f26, f11, f27);
  f29 = f11 * f10;
  p5 = f9 == __sm_f32bits(0xff800000u);
  f30 = p5 ? f10 : f29;
  f31 = -f30;
  f32 = p4 ? f31 : f30;
  f55 = __sm_mul_rn(f28, f32);
L43_3:;
  __ret = f55;
  return __ret;
}

SASS_MATH_FN(float, lgammaf)(float a0) {
  float __ret;
  float f1, f2, f3, f11, f17, f18, f24, f29, f35, f36, f37, f38;
  float f39, f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50;
  float f51, f52, f53, f54, f55, f56, f57, f58, f59, f60, f61, f62;
  float f63, f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74;
  float f75, f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86;
  float f87, f88, f89, f90, f91, f92, f93, f94, f95, f96, f97, f98;
  float f99, f100, f101, f102, f103, f104, f105, f106, f107, f108, f109, f110;
  float f111, f112, f113, f114, f115, f116, f117, f118, f119, f120, f121, f122;
  float f123, f124, f125, f126, f127, f128, f129, f130, f131, f132, f133, f134;
  float f135, f136, f137, f138, f139, f140, f141, f142, f143, f144, f145, f146;
  float f147, f148, f149, f150, f151, f152, f153, f154, f155, f156, f157, f158;
  float f159, f160, f161, f162, f163, f164, f165, f166, f167, f168, f169, f170;
  float f171, f172, f173, f174, f175, f176, f177, f178, f179, f180, f181, f182;
  float f183, f185, f187, f188, f189, f190, f191, f192, f193, f194, f195, f196;
  float f197, f198, f199, f200, f202, f203, f204, f205, f206, f207, f208, f209;
  float f210, f211, f212, f213, f214, f215, f216, f217, f218, f219, f220, f221;
  float f222, f223, f224, f225, f226, f227, f228, f229, f230, f231, f232, f233;
  float f234, f235, f236, f237, f238, f239, f240, f241, f242, f243, f244, f245;
  float f246, f247, f248, f249, f250, f251, f252, f253, f254, f255, f256, f257;
  float f258, f259, f260, f261, f262, f263, f264, f265, f266, f267, f268, f269;
  float f270, f271, f272, f273, f274;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  bool p13, p14, p15, p17, p18, p19, p20, p21, p22, p23;
  unsigned r1, r2, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
  unsigned r14, r15, r16, r17, r18, r19, r20;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = !(f2 >= 0x1.8p+1f);
  if (p1) goto L44_6;
  goto L44_1;
L44_6:;
  p7 = !(f2 >= 0x1.8p+0f);
  if (p7) goto L44_8;
  goto L44_7;
L44_8:;
  p8 = !(f2 >= 0x1.666666p-1f);
  if (p8) goto L44_10;
  goto L44_9;
L44_10:;
  f140 = -0x1.6690fp-8f;
  f141 = 0x1.d6390cp-9f;
  f142 = __sm_fma_rn(f141, f2, f140);
  f143 = -0x1.6d95dep-5f;
  f144 = __sm_fma_rn(f142, f2, f143);
  f145 = 0x1.56aaaap-3f;
  f146 = __sm_fma_rn(f144, f2, f145);
  f147 = -0x1.592d8ep-5f;
  f148 = __sm_fma_rn(f146, f2, f147);
  f149 = -0x1.4fcdd6p-1f;
  f150 = __sm_fma_rn(f148, f2, f149);
  f151 = 0x1.2788c6p-1f;
  f152 = __sm_fma_rn(f150, f2, f151);
  f153 = f2 * f152;
  f154 = __sm_fma_rn(f153, f2, f2);
  p9 = f154 < 0x1p-126f;
  f155 = f154 * 0x1p+23f;
  f11 = p9 ? f155 : f154;
  f156 = p9 ? -0x1.7p+4f : 0.0f;
  r8 = __float_as_uint(f11);
  r9 = r8 + 0xc0d55555u;
  r10 = r9 & 0xff800000u;
  r11 = r8 - r10;
  f157 = __sm_f32bits(r11);
  f158 = __sm_i2f_rn((int)r10);
  f159 = 0x1p-23f;
  f160 = __sm_fma_rn(f158, f159, f156);
  f161 = f157 + -0x1p+0f;
  f162 = 0x1.2073ecp-3f;
  f163 = -0x1.0aa04ep-3f;
  f164 = __sm_fma_rn(f163, f161, f162);
  f165 = -0x1.f19b98p-4f;
  f166 = __sm_fma_rn(f164, f161, f165);
  f167 = 0x1.1e52aap-3f;
  f168 = __sm_fma_rn(f166, f161, f167);
  f169 = -0x1.55b172p-3f;
  f170 = __sm_fma_rn(f168, f161, f169);
  f171 = 0x1.99da16p-3f;
  f172 = __sm_fma_rn(f170, f161, f171);
  f173 = -0x1.fffe44p-3f;
  f174 = __sm_fma_rn(f172, f161, f173);
  f175 = 0x1.5554fp-2f;
  f176 = __sm_fma_rn(f174, f161, f175);
  f177 = -0x1p-1f;
  f178 = __sm_fma_rn(f176, f161, f177);
  f179 = f161 * f178;
  f180 = __sm_fma_rn(f179, f161, f161);
  f181 = 0x1.62e43p-1f;
  f268 = __sm_fma_rn(f160, f181, f180);
  p10 = r8 < 0x7f800000u;
  if (p10) goto L44_12;
  f182 = __sm_f32bits(0x7f800000u);
  f268 = __sm_fma_rn(f11, f182, f182);
L44_12:;
  f183 = -f268;
  p11 = f11 == 0.0f;
  f269 = p11 ? __sm_f32bits(0x7f800000u) : f183;
  goto L44_13;
L44_1:;
  p2 = !(f2 >= 0x1.f33334p+2f);
  if (p2) goto L44_5;
  goto L44_2;
L44_5:;
  f79 = f2 + -0x1.8p+1f;
  f80 = -0x1.81edfp+13f;
  f81 = -0x1.7671f6p+9f;
  f82 = __sm_fma_rn(f81, f79, f80);
  f83 = -0x1.40cacp+15f;
  f84 = __sm_fma_rn(f82, f79, f83);
  f85 = -0x1.796d54p+15f;
  f86 = __sm_fma_rn(f84, f79, f85);
  f87 = -0x1.175cb4p+17f;
  f88 = __sm_fma_rn(f86, f79, f87);
  f89 = f79 + -0x1.03404p+8f;
  f90 = -0x1.50c97p+13f;
  f91 = __sm_fma_rn(f89, f79, f90);
  f92 = -0x1.6a0d0cp+16f;
  f93 = __sm_fma_rn(f91, f79, f92);
  f94 = -0x1.9308cap+17f;
  f95 = __sm_fma_rn(f93, f79, f94);
  f96 = __sm_rcp_approx_ftz(f95);
  f269 = __sm_fma_rn(f88, f96, f79);
  goto L44_13;
L44_7:;
  f97 = f2 + -0x1p+1f;
  f98 = -0x1.cf4004p-13f;
  f99 = 0x1.a00ff4p-15f;
  f100 = __sm_fma_rn(f99, f97, f98);
  f101 = 0x1.1bcdf8p-11f;
  f102 = __sm_fma_rn(f100, f97, f101);
  f103 = -0x1.3bc1c4p-10f;
  f104 = __sm_fma_rn(f102, f97, f103);
  f105 = 0x1.7a0b6ep-9f;
  f106 = __sm_fma_rn(f104, f97, f105);
  f107 = -0x1.e3d62p-8f;
  f108 = __sm_fma_rn(f106, f97, f107);
  f109 = 0x1.51345p-6f;
  f110 = __sm_fma_rn(f108, f97, f109);
  f111 = -0x1.13e034p-4f;
  f112 = __sm_fma_rn(f110, f97, f111);
  f113 = 0x1.4a34ccp-2f;
  f114 = __sm_fma_rn(f112, f97, f113);
  f115 = 0x1.b0ee6p-2f;
  f116 = __sm_fma_rn(f114, f97, f115);
  f269 = f97 * f116;
  goto L44_13;
L44_2:;
  f35 = f2 * 0x1p+23f;
  p3 = f2 < 0x1p-126f;
  f3 = p3 ? f35 : f2;
  f36 = p3 ? -0x1.7p+4f : 0.0f;
  r4 = __float_as_uint(f3);
  r5 = r4 + 0xc0d55555u;
  r6 = r5 & 0xff800000u;
  r7 = r4 - r6;
  f37 = __sm_f32bits(r7);
  f38 = __sm_i2f_rn((int)r6);
  f39 = 0x1p-23f;
  f40 = __sm_fma_rn(f38, f39, f36);
  f41 = f37 + -0x1p+0f;
  f42 = 0x1.2073ecp-3f;
  f43 = -0x1.0aa04ep-3f;
  f44 = __sm_fma_rn(f43, f41, f42);
  f45 = -0x1.f19b98p-4f;
  f46 = __sm_fma_rn(f44, f41, f45);
  f47 = 0x1.1e52aap-3f;
  f48 = __sm_fma_rn(f46, f41, f47);
  f49 = -0x1.55b172p-3f;
  f50 = __sm_fma_rn(f48, f41, f49);
  f51 = 0x1.99da16p-3f;
  f52 = __sm_fma_rn(f50, f41, f51);
  f53 = -0x1.fffe44p-3f;
  f54 = __sm_fma_rn(f52, f41, f53);
  f55 = 0x1.5554fp-2f;
  f56 = __sm_fma_rn(f54, f41, f55);
  f57 = -0x1p-1f;
  f58 = __sm_fma_rn(f56, f41, f57);
  f59 = f41 * f58;
  f60 = __sm_fma_rn(f59, f41, f41);
  f61 = 0x1.62e43p-1f;
  f267 = __sm_fma_rn(f40, f61, f60);
  p4 = r4 < 0x7f800000u;
  if (p4) goto L44_4;
  f62 = __sm_f32bits(0x7f800000u);
  f267 = __sm_fma_rn(f3, f62, f62);
L44_4:;
  f63 = __sm_rcp_approx_ftz(f2);
  f64 = f63 * f63;
  f65 = f267 * 0x1p-1f;
  p5 = f3 == 0.0f;
  f66 = p5 ? __sm_f32bits(0xff800000u) : f65;
  f67 = f2 + -0x1p-1f;
  f68 = __sm_mul_rn(f66, f67);
  f69 = f68 - f2;
  f70 = -0x1.6c12a6p-9f;
  f71 = 0x1.97ceaap-11f;
  f72 = __sm_fma_rn(f71, f64, f70);
  f73 = 0x1.555546p-4f;
  f74 = __sm_fma_rn(f72, f64, f73);
  f75 = 0x1.d67f1cp-1f;
  f76 = __sm_fma_rn(f74, f63, f75);
  f77 = __sm_add_rn(f68, f76);
  f78 = f69 + f77;
  p6 = f2 == __sm_f32bits(0x7f800000u);
  f269 = p6 ? f2 : f78;
  goto L44_13;
L44_9:;
  f117 = 0x1p+0f;
  f118 = f117 - f2;
  f119 = 0x1.a8eaeep-4f;
  f120 = 0x1.77deecp-5f;
  f121 = __sm_fma_rn(f120, f118, f119);
  f122 = 0x1.f700f2p-4f;
  f123 = __sm_fma_rn(f121, f118, f122);
  f124 = 0x1.052b6ap-3f;
  f125 = __sm_fma_rn(f123, f118, f124);
  f126 = 0x1.254ecap-3f;
  f127 = __sm_fma_rn(f125, f118, f126);
  f128 = 0x1.5ad0cep-3f;
  f129 = __sm_fma_rn(f127, f118, f128);
  f130 = 0x1.a8c57ep-3f;
  f131 = __sm_fma_rn(f129, f118, f130);
  f132 = 0x1.1514e4p-2f;
  f133 = __sm_fma_rn(f131, f118, f132);
  f134 = 0x1.9a4d48p-2f;
  f135 = __sm_fma_rn(f133, f118, f134);
  f136 = 0x1.a51a64p-1f;
  f137 = __sm_fma_rn(f135, f118, f136);
  f138 = 0x1.2788dp-1f;
  f139 = __sm_fma_rn(f137, f118, f138);
  f269 = f118 * f139;
L44_13:;
  p12 = f1 >= 0.0f;
  f274 = f269;
  if (p12) goto L44_26;
  f185 = __sm_floor(f2);
  p13 = f2 == f185;
  f274 = __sm_f32bits(0x7f800000u);
  if (p13) goto L44_26;
  p14 = f2 < 0x1.d83c94p-64f;
  if (p14) goto L44_23;
  goto L44_16;
L44_23:;
  p21 = f2 < 0x1p-126f;
  f238 = f2 * 0x1p+23f;
  f29 = p21 ? f238 : f2;
  f239 = p21 ? -0x1.7p+4f : 0.0f;
  r17 = __float_as_uint(f29);
  r18 = r17 + 0xc0d55555u;
  r19 = r18 & 0xff800000u;
  r20 = r17 - r19;
  f240 = __sm_f32bits(r20);
  f241 = __sm_i2f_rn((int)r19);
  f242 = 0x1p-23f;
  f243 = __sm_fma_rn(f241, f242, f239);
  f244 = f240 + -0x1p+0f;
  f245 = 0x1.2073ecp-3f;
  f246 = -0x1.0aa04ep-3f;
  f247 = __sm_fma_rn(f246, f244, f245);
  f248 = -0x1.f19b98p-4f;
  f249 = __sm_fma_rn(f247, f244, f248);
  f250 = 0x1.1e52aap-3f;
  f251 = __sm_fma_rn(f249, f244, f250);
  f252 = -0x1.55b172p-3f;
  f253 = __sm_fma_rn(f251, f244, f252);
  f254 = 0x1.99da16p-3f;
  f255 = __sm_fma_rn(f253, f244, f254);
  f256 = -0x1.fffe44p-3f;
  f257 = __sm_fma_rn(f255, f244, f256);
  f258 = 0x1.5554fp-2f;
  f259 = __sm_fma_rn(f257, f244, f258);
  f260 = -0x1p-1f;
  f261 = __sm_fma_rn(f259, f244, f260);
  f262 = f244 * f261;
  f263 = __sm_fma_rn(f262, f244, f244);
  f264 = 0x1.62e43p-1f;
  f273 = __sm_fma_rn(f243, f264, f263);
  p22 = r17 < 0x7f800000u;
  if (p22) goto L44_25;
  f265 = __sm_f32bits(0x7f800000u);
  f273 = __sm_fma_rn(f29, f265, f265);
L44_25:;
  f266 = -f273;
  p23 = f29 == 0.0f;
  f274 = p23 ? __sm_f32bits(0x7f800000u) : f266;
  goto L44_26;
L44_16:;
  f187 = f2 + f2;
  f188 = __sm_rint(f187);
  r1 = __sm_f2i_rz(f188);
  f189 = -f188;
  f190 = 0x1p-1f;
  f191 = __sm_fma_rn(f189, f190, f2);
  f192 = f191 * 0x1.921fb6p+1f;
  f17 = __sm_mul_rn(f192, f192);
  r2 = r1 & 1u;
  p15 = (int)r2 == (int)0u;
  f18 = p15 ? f192 : 0x1p+0f;
  f270 = -0x1.9a82a6p-13f;
  if (p15) goto L44_18;
  f193 = -0x1.6c0fdap-10f;
  f194 = 0x1.9758p-16f;
  f270 = __sm_fma_rn(f194, f17, f193);
L44_18:;
  f195 = p15 ? 0x1.110bc8p-7f : 0x1.555576p-5f;
  f196 = __sm_fma_rn(f270, f17, f195);
  f197 = p15 ? -0x1.55555p-3f : -0x1.fffffep-2f;
  f198 = __sm_fma_rn(f196, f17, f197);
  f199 = 0.0f;
  f200 = __sm_fma_rn(f17, f18, f199);
  f271 = __sm_fma_rn(f198, f200, f18);
  r12 = r1 & 2u;
  p17 = (int)r12 == (int)0u;
  if (p17) goto L44_20;
  f202 = -0x1p+0f;
  f271 = __sm_fma_rn(f271, f202, f199);
L44_20:;
  f203 = __sm_fabs(f271);
  f204 = f2 * f203;
  p18 = f204 < 0x1p-126f;
  f205 = f204 * 0x1p+23f;
  f206 = p18 ? -0x1.7p+4f : 0.0f;
  f24 = p18 ? f205 : f204;
  r13 = __float_as_uint(f24);
  r14 = r13 + 0xc0d55555u;
  r15 = r14 & 0xff800000u;
  r16 = r13 - r15;
  f207 = __sm_f32bits(r16);
  f208 = __sm_i2f_rn((int)r15);
  f209 = 0x1p-23f;
  f210 = __sm_fma_rn(f208, f209, f206);
  f211 = f207 + -0x1p+0f;
  f212 = 0x1.2073ecp-3f;
  f213 = -0x1.0aa04ep-3f;
  f214 = __sm_fma_rn(f213, f211, f212);
  f215 = -0x1.f19b98p-4f;
  f216 = __sm_fma_rn(f214, f211, f215);
  f217 = 0x1.1e52aap-3f;
  f218 = __sm_fma_rn(f216, f211, f217);
  f219 = -0x1.55b172p-3f;
  f220 = __sm_fma_rn(f218, f211, f219);
  f221 = 0x1.99da16p-3f;
  f222 = __sm_fma_rn(f220, f211, f221);
  f223 = -0x1.fffe44p-3f;
  f224 = __sm_fma_rn(f222, f211, f223);
  f225 = 0x1.5554fp-2f;
  f226 = __sm_fma_rn(f224, f211, f225);
  f227 = -0x1p-1f;
  f228 = __sm_fma_rn(f226, f211, f227);
  f229 = f228 * f211;
  f230 = __sm_fma_rn(f229, f211, f211);
  f231 = 0x1.62e43p-1f;
  f272 = __sm_fma_rn(f210, f231, f230);
  p19 = r13 < 0x7f800000u;
  if (p19) goto L44_22;
  f232 = __sm_f32bits(0x7f800000u);
  f272 = __sm_fma_rn(f24, f232, f232);
L44_22:;
  f233 = 0x1.250d04p+0f;
  f234 = f233 - f272;
  p20 = f24 == 0.0f;
  f235 = p20 ? __sm_f32bits(0x7f800000u) : f234;
  f236 = -f269;
  f237 = 0x1p+0f;
  f274 = __sm_fma_rn(f235, f237, f236);
L44_26:;
  __ret = f274;
  return __ret;
}

SASS_MATH_FN(float, tgammaf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f12, f15, f16, f17, f18, f19;
  float f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31;
  float f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42, f43;
  float f44, f45, f46, f47, f48, f49, f50, f51, f52, f53, f54, f55;
  float f56, f57, f58, f59, f60, f61, f62, f63, f64, f65, f66, f67;
  float f68, f69, f70, f71, f72, f73, f74, f75, f76, f77, f78, f79;
  float f80, f81, f82, f83, f84, f85, f86, f87, f88, f89, f90, f91;
  float f92, f93, f94, f95, f96, f97, f98, f99, f100, f101, f102, f103;
  float f104, f105, f106, f107, f108, f109, f110, f111, f112, f113, f114, f115;
  float f116, f117, f118, f119, f120, f121, f122, f123, f124, f125, f126, f127;
  float f128, f129, f130, f131, f132, f133, f134, f136, f137, f138, f139, f140;
  float f141, f142, f143, f144, f145, f146, f147, f148, f149, f150, f151, f152;
  float f153, f154, f155, f156, f157, f158, f159, f160, f161, f162, f163, f164;
  float f165, f166, f167, f168, f169, f170, f171, f172, f173, f174, f175, f176;
  float f178, f179, f181;
  bool p1, p2, p3, p4, p5, p6, p8, p9, p10, p11, p12, p13;
  bool p14, p15, p16;
  unsigned r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
  unsigned r14, r15;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = !(f2 >= 0x1.8p+0f);
  if (p1) goto L45_4;
  goto L45_1;
L45_4:;
  p10 = f1 == 0.0f;
  f179 = f1;
  if (p10) goto L45_6;
  f154 = __sm_rint(f1);
  f179 = f1 - f154;
L45_6:;
  p11 = f1 < -0x1p-1f;
  f155 = p11 ? f1 : 0x1p+0f;
  f156 = 0x1p+0f;
  f157 = f155 * f179;
  p12 = f1 <= 0x1p-1f;
  f158 = p12 ? f157 : f155;
  f159 = 0x1.d207aep-8f;
  f160 = -0x1.193eccp-10f;
  f161 = __sm_fma_rn(f160, f179, f159);
  f162 = -0x1.3c016p-7f;
  f163 = __sm_fma_rn(f161, f179, f162);
  f164 = -0x1.598a44p-5f;
  f165 = __sm_fma_rn(f163, f179, f164);
  f166 = 0x1.551366p-3f;
  f167 = __sm_fma_rn(f165, f179, f166);
  f168 = -0x1.58180ap-5f;
  f169 = __sm_fma_rn(f167, f179, f168);
  f170 = -0x1.4fcf44p-1f;
  f171 = __sm_fma_rn(f169, f179, f170);
  f172 = 0x1.2788dp-1f;
  f173 = __sm_fma_rn(f171, f179, f172);
  f174 = 0.0f;
  f175 = __sm_fma_rn(f173, f158, f174);
  f176 = __sm_fma_rn(f175, f179, f158);
  f181 = __sm_div_approx(f156, f176);
  p13 = !(f1 < 0.0f);
  if (p13) goto L45_10;
  goto L45_7;
L45_1:;
  p2 = f1 < 0.0f;
  p3 = f2 > 0x1.48ccccp+5f;
  f3 = p3 ? 0x1.48ccccp+5f : f1;
  f4 = __sm_fabs(f3);
  p4 = f4 < 0x1p-126f;
  f15 = f4 * 0x1p+24f;
  f16 = p4 ? f15 : f4;
  f17 = p4 ? -0x1.8p+4f : 0.0f;
  r2 = __float_as_uint(f16);
  r3 = r2 + 0xc0cafb0du;
  r4 = r3 & 0xff800000u;
  r5 = r2 - r4;
  f18 = __sm_f32bits(r5);
  f19 = __sm_i2f_rn((int)r4);
  f20 = 0x1p-23f;
  f21 = __sm_fma_rn(f19, f20, f17);
  f22 = f18 + -0x1p+0f;
  f23 = f18 + 0x1p+0f;
  f24 = 0x1p+0f;
  f25 = __sm_rcp_approx_ftz(f23);
  f26 = f22 + f22;
  f27 = f26 * f25;
  f28 = f27 * f27;
  f29 = f22 - f27;
  f30 = f29 + f29;
  f31 = -f27;
  f32 = __sm_fma_rn(f31, f22, f30);
  f33 = __sm_mul_rn(f25, f32);
  f34 = 0x1.a5cfb6p-9f;
  f35 = 0x1.5865c8p-11f;
  f36 = __sm_fma_rn(f35, f28, f34);
  f37 = 0x1.2776e6p-6f;
  f38 = __sm_fma_rn(f36, f28, f37);
  f39 = 0x1.ec709ep-4f;
  f40 = __sm_fma_rn(f38, f28, f39);
  f41 = __sm_mul_rn(f40, f28);
  f42 = 0x1.715476p+0f;
  f43 = __sm_fma_rn(f27, f42, f21);
  f44 = f21 - f43;
  f45 = __sm_fma_rn(f27, f42, f44);
  f46 = __sm_fma_rn(f33, f42, f45);
  f47 = 0x1.4abc68p-26f;
  f48 = __sm_fma_rn(f27, f47, f46);
  f49 = __sm_fma_rn(f41, f27, f48);
  f50 = __sm_add_rn(f43, f49);
  f51 = -f43;
  f52 = __sm_add_rn(f50, f51);
  f53 = -f52;
  f54 = __sm_add_rn(f49, f53);
  f55 = f4 + -0x1p-1f;
  f56 = __sm_mul_rn(f50, f55);
  f57 = -f56;
  f58 = __sm_fma_rn(f50, f55, f57);
  f59 = __sm_fma_rn(f54, f55, f58);
  f60 = __sm_mul_rn(f42, f4);
  f61 = -f60;
  f62 = __sm_fma_rn(f42, f4, f61);
  f63 = 0x1.4ae0cp-26f;
  f64 = __sm_fma_rn(f63, f4, f62);
  f65 = __sm_add_rn(f56, f61);
  f66 = __sm_add_rn(f65, f60);
  f67 = -f66;
  f68 = __sm_add_rn(f65, f67);
  f69 = __sm_add_rn(f56, f67);
  f70 = -f68;
  f71 = __sm_add_rn(f61, f70);
  f72 = __sm_add_rn(f69, f71);
  f73 = f59 - f64;
  f74 = f73 + f72;
  f75 = -f74;
  p5 = f4 > 0x1.08p+5f;
  f76 = p5 ? 0x1.8p+5f : -0.0f;
  f77 = f76 - f65;
  f78 = p2 ? f77 : f65;
  f79 = p2 ? f75 : f74;
  f80 = __sm_rint(f78);
  f81 = f78 - f80;
  f82 = f81 + f79;
  f83 = 0x1.5f0bdap-10f;
  f84 = 0x1.3f971cp-13f;
  f85 = __sm_fma_rn(f84, f82, f83);
  f86 = 0x1.3b30acp-7f;
  f87 = __sm_fma_rn(f85, f82, f86);
  f88 = 0x1.c6af76p-5f;
  f89 = __sm_fma_rn(f87, f82, f88);
  f90 = 0x1.ebfbd8p-3f;
  f91 = __sm_fma_rn(f89, f82, f90);
  f92 = 0x1.62e43p-1f;
  f93 = __sm_fma_rn(f91, f82, f92);
  f94 = __sm_fma_rn(f93, f82, f24);
  r6 = __sm_f2i_rz(f80);
  p6 = f80 > 0.0f;
  r7 = p6 ? 0u : 0x83000000u;
  r8 = r7 + 0x7f000000u;
  f95 = __sm_f32bits(r8);
  f96 = f94 * f95;
  r9 = r6 << 0x17u;
  r10 = r9 - r7;
  f97 = __sm_f32bits(r10);
  f98 = f96 * f97;
  f5 = f98 * 0x1.40d932p+1f;
  f99 = __sm_rcp_approx_ftz(f4);
  f100 = -0x1.a87dbp-15f;
  f101 = 0x1.1ef2e2p-14f;
  f102 = __sm_fma_rn(f101, f99, f100);
  f103 = -0x1.bb4d7cp-12f;
  f104 = __sm_fma_rn(f102, f99, f103);
  f105 = 0x1.04157cp-10f;
  f106 = __sm_fma_rn(f104, f99, f105);
  f107 = -0x1.2415fap-12f;
  f108 = __sm_fma_rn(f106, f99, f107);
  f109 = -0x1.5e9ac8p-9f;
  f110 = __sm_fma_rn(f108, f99, f109);
  f111 = 0x1.c70e64p-9f;
  f112 = __sm_fma_rn(f110, f99, f111);
  f113 = 0x1.555558p-4f;
  f114 = __sm_fma_rn(f112, f99, f113);
  f6 = f99 * f114;
  if (p2) goto L45_3;
  goto L45_2;
L45_3:;
  f115 = f4 + f4;
  f116 = __sm_rint(f115);
  r11 = __sm_f2i_rz(f116);
  f117 = -0x1p-1f;
  f118 = __sm_fma_rn(f116, f117, f4);
  f119 = __sm_mul_rn(f118, f118);
  f120 = 0x1.466b2p+1f;
  f121 = -0x1.2f5992p-1f;
  f122 = __sm_fma_rn(f121, f119, f120);
  f123 = -0x1.55a5cp+0f;
  f124 = 0x1.d09c24p-3f;
  f125 = __sm_fma_rn(f124, f119, f123);
  f126 = -0x1.4abbecp+2f;
  f127 = __sm_fma_rn(f122, f119, f126);
  f128 = 0x1.03c19ep+2f;
  f129 = __sm_fma_rn(f125, f119, f128);
  f130 = 0.0f;
  f131 = __sm_fma_rn(f119, f118, f130);
  f132 = -0x1.3bd3ccp+2f;
  f133 = __sm_fma_rn(f129, f119, f132);
  f134 = __sm_fma_rn(f127, f131, f130);
  f136 = __sm_fma_rn(f133, f119, f24);
  f137 = 0x1.921fb6p+1f;
  f138 = __sm_fma_rn(f137, f118, f134);
  r12 = r11 & 1u;
  p8 = r12 == 1u;
  f139 = p8 ? f136 : f138;
  r13 = r11 & 2u;
  p9 = (int)r13 == (int)0u;
  f140 = f130 - f139;
  f141 = p9 ? f139 : f140;
  f142 = __sm_fma_rn(f6, f3, f3);
  f143 = __sm_mul_rn(f142, f141);
  f144 = -f143;
  f145 = __sm_fma_rn(f142, f141, f144);
  f146 = __sm_div_approx(f24, f143);
  f147 = f146 * f146;
  f148 = f145 * f147;
  f149 = f5 * f148;
  f150 = -f149;
  f151 = __sm_fma_rn(f5, f146, f150);
  f152 = f151 * 0x1p-1f;
  f153 = f152 * 0x1p-48f;
  f181 = p5 ? f153 : f152;
L45_7:;
  f12 = f181;
  f178 = __sm_trunc(f1);
  p14 = f178 == f1;
  f181 = __sm_f32bits(0x7fffffffu);
  if (p14) goto L45_10;
  p15 = !(f1 < -0x1.48ccccp+5f);
  f181 = f12;
  if (p15) goto L45_10;
  r14 = __sm_f2i_rz(f1);
  r15 = r14 & 1u;
  p16 = r15 == 1u;
  f181 = p16 ? 0.0f : f12;
  goto L45_10;
L45_2:;
  f181 = __sm_fma_rn(f6, f5, f5);
L45_10:;
  __ret = f181;
  return __ret;
}

SASS_MATH_FN(float, logbf)(float a0) {
  float __ret;
  float f1, f5, f6, f7, f8, f9;
  bool p1, p2, p3;
  unsigned r1, r3, r4, r5, r6, r7;
  f1 = a0;
  f5 = __sm_fabs(f1);
  r1 = __float_as_uint(f5);
  p1 = r1 < 0x800000u;
  if (p1) goto L46_2;
  goto L46_1;
L46_2:;
  r5 = __sm_clz32(r1);
  r6 = 0xffffff8au;
  r7 = r6 - r5;
  f8 = __sm_i2f_rn((int)r7);
  p3 = f1 == 0.0f;
  f9 = p3 ? __sm_f32bits(0xff800000u) : f8;
  goto L46_3;
L46_1:;
  r3 = r1 >> 0x17u;
  r4 = r3 + 0xffffff81u;
  f6 = __sm_i2f_rn((int)r4);
  f7 = f1 * f1;
  p2 = r1 > 0x7f7fffffu;
  f9 = p2 ? f7 : f6;
L46_3:;
  __ret = f9;
  return __ret;
}

SASS_MATH_FN(float, sinpif)(float a0) {
  float __ret;
  float f1, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f34;
  bool p1, p2, p3;
  unsigned r2, r3, r4;
  f1 = a0;
  f5 = f1 + f1;
  f6 = __sm_rint(f5);
  r2 = __sm_f2i_rz(f6);
  f7 = -f6;
  f8 = 0x1p-1f;
  f9 = __sm_fma_rn(f7, f8, f1);
  f10 = __sm_mul_rn(f9, f9);
  f11 = 0x1.466b2p+1f;
  f12 = -0x1.2f5992p-1f;
  f13 = __sm_fma_rn(f12, f10, f11);
  f14 = -0x1.55a5cp+0f;
  f15 = 0x1.d09c24p-3f;
  f16 = __sm_fma_rn(f15, f10, f14);
  f17 = -0x1.4abbecp+2f;
  f18 = __sm_fma_rn(f13, f10, f17);
  f19 = 0x1.03c19ep+2f;
  f20 = __sm_fma_rn(f16, f10, f19);
  f21 = 0.0f;
  f22 = __sm_fma_rn(f10, f9, f21);
  f23 = -0x1.3bd3ccp+2f;
  f24 = __sm_fma_rn(f20, f10, f23);
  f25 = __sm_fma_rn(f18, f22, f21);
  f26 = 0x1p+0f;
  f27 = __sm_fma_rn(f24, f10, f26);
  f28 = 0x1.921fb6p+1f;
  f29 = __sm_fma_rn(f28, f9, f25);
  r3 = r2 & 1u;
  p1 = r3 == 1u;
  f30 = p1 ? f27 : f29;
  r4 = r2 & 2u;
  p2 = (int)r4 == (int)0u;
  f31 = f21 - f30;
  f34 = p2 ? f30 : f31;
  f32 = __sm_trunc(f1);
  p3 = f32 != f1;
  if (p3) goto L48_2;
  f34 = __sm_mul_rn(f1, f21);
L48_2:;
  __ret = f34;
  return __ret;
}

SASS_MATH_FN(float, cospif)(float a0) {
  float __ret;
  float f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34;
  bool p1, p2, p3;
  unsigned r2, r3, r4, r5;
  f34 = a0;
  f4 = __sm_fabs(f34);
  p1 = !(f4 > 0x1p+24f);
  if (p1) goto L49_2;
  f5 = 0.0f;
  f34 = __sm_mul_rn(f34, f5);
L49_2:;
  f6 = f34 + f34;
  f7 = __sm_rint(f6);
  r2 = __sm_f2i_rz(f7);
  f8 = -f7;
  f9 = 0x1p-1f;
  f10 = __sm_fma_rn(f8, f9, f34);
  r3 = r2 + 1u;
  f11 = __sm_mul_rn(f10, f10);
  f12 = 0x1.466b2p+1f;
  f13 = -0x1.2f5992p-1f;
  f14 = __sm_fma_rn(f13, f11, f12);
  f15 = -0x1.55a5cp+0f;
  f16 = 0x1.d09c24p-3f;
  f17 = __sm_fma_rn(f16, f11, f15);
  f18 = -0x1.4abbecp+2f;
  f19 = __sm_fma_rn(f14, f11, f18);
  f20 = 0x1.03c19ep+2f;
  f21 = __sm_fma_rn(f17, f11, f20);
  f22 = 0.0f;
  f23 = __sm_fma_rn(f11, f10, f22);
  f24 = -0x1.3bd3ccp+2f;
  f25 = __sm_fma_rn(f21, f11, f24);
  f26 = __sm_fma_rn(f19, f23, f22);
  f27 = 0x1p+0f;
  f28 = __sm_fma_rn(f25, f11, f27);
  f29 = 0x1.921fb6p+1f;
  f30 = __sm_fma_rn(f29, f10, f26);
  r4 = r3 & 1u;
  p2 = r4 == 1u;
  f31 = p2 ? f28 : f30;
  r5 = r3 & 2u;
  p3 = (int)r5 == (int)0u;
  f32 = f22 - f31;
  f33 = p3 ? f31 : f32;
  __ret = f33;
  return __ret;
}

SASS_MATH_FN(float, rcbrtf)(float a0) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16, f17;
  bool p1, p2;
  unsigned r2, r3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  f3 = __sm_lg2_approx(f2);
  f4 = f3 * -0x1.555556p-2f;
  f5 = __sm_ex2_approx_ftz(f4);
  f6 = f5 * f5;
  f7 = f2 * f5;
  f8 = -f7;
  f9 = 0x1p+0f;
  f10 = __sm_fma_rn(f6, f8, f9);
  f11 = f5 * 0x1.555556p-2f;
  f12 = __sm_fma_rn(f10, f11, f5);
  p1 = f1 < 0.0f;
  f13 = -f12;
  f14 = p1 ? f13 : f12;
  f15 = f1 + f1;
  p2 = f15 == f1;
  r2 = __float_as_uint(f15);
  r3 = r2 ^ 0x7f800000u;
  f16 = __sm_f32bits(r3);
  f17 = p2 ? f16 : f14;
  __ret = f17;
  return __ret;
}

SASS_MATH_FN(float, normcdff)(float a0) {
  float __ret;
  float f1, f2, f3, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39;
  float f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51;
  float f52, f53, f54, f55, f56, f57, f58, f59, f60, f61, f62, f63;
  float f64, f65, f66, f67, f68, f69, f70, f71, f72, f73, f74, f75;
  float f76, f77, f78, f79, f80, f81, f82, f83, f84, f85, f86;
  bool p1, p2, p3, p4, p5;
  unsigned r2, r3;
  f7 = a0;
  f8 = __sm_fabs(f7);
  p1 = f8 > 0x1.dp+3f;
  f9 = 0x1.dp+3f;
  f10 = __sm_copysign(f9, f7);
  f11 = p1 ? f10 : f7;
  f12 = -0x1.6a09e6p-1f;
  f1 = __sm_mul_rn(f11, f12);
  f13 = -f1;
  f14 = __sm_fma_rn(f11, f12, f13);
  f15 = -0x1.9fcef4p-27f;
  f2 = __sm_fma_rn(f11, f15, f14);
  f3 = __sm_add_rn(f1, f2);
  f16 = __sm_fabs(f3);
  f17 = f16 + -0x1p+2f;
  f18 = -0x1p+2f;
  f19 = f16 + 0x1p+2f;
  f20 = __sm_rcp_approx_ftz(f19);
  f21 = __sm_mul_rn(f17, f20);
  f22 = f21 + 0x1p+0f;
  f23 = 0x1p+0f;
  f24 = __sm_fma_rn(f18, f22, f16);
  f25 = -f21;
  f26 = __sm_fma_rn(f25, f16, f24);
  f27 = __sm_fma_rn(f20, f26, f21);
  f28 = 0x1.cdc0b6p-8f;
  f29 = 0x1.d34122p-11f;
  f30 = __sm_fma_rn(f29, f27, f28);
  f31 = -0x1.03f696p-6f;
  f32 = __sm_fma_rn(f30, f27, f31);
  f33 = 0x1.2a6e76p-5f;
  f34 = __sm_fma_rn(f32, f27, f33);
  f35 = -0x1.10f8b4p-4f;
  f36 = __sm_fma_rn(f34, f27, f35);
  f37 = 0x1.8043aap-4f;
  f38 = __sm_fma_rn(f36, f27, f37);
  f39 = -0x1.9da848p-4f;
  f40 = __sm_fma_rn(f38, f27, f39);
  f41 = 0x1.16e9bcp-4f;
  f42 = __sm_fma_rn(f40, f27, f41);
  f43 = 0x1.f7e2ep-7f;
  f44 = __sm_fma_rn(f42, f27, f43);
  f45 = -0x1.1df1a8p-3f;
  f46 = __sm_fma_rn(f44, f27, f45);
  f47 = 0x1.3ba592p+0f;
  f48 = __sm_fma_rn(f46, f27, f47);
  f49 = 0x1p+1f;
  f50 = __sm_fma_rn(f49, f16, f23);
  f51 = __sm_rcp_approx_ftz(f50);
  f52 = __sm_mul_rn(f48, f51);
  f53 = f52 * -0x1p+1f;
  f54 = __sm_fma_rn(f16, f53, f48);
  f55 = f54 - f52;
  f56 = __sm_fma_rn(f55, f51, f52);
  f57 = f16 * f16;
  f58 = -f57;
  f59 = 0x1.715476p+0f;
  f60 = __sm_mul_rn(f58, f59);
  f61 = __sm_trunc(f60);
  f62 = __sm_fabs(f61);
  p2 = f62 > 0x1.f8p+6f;
  f63 = 0x1.f8p+6f;
  f64 = __sm_copysign(f63, f61);
  f65 = p2 ? f64 : f61;
  f66 = -0x1.62e43p-1f;
  f67 = __sm_fma_rn(f65, f66, f58);
  f68 = 0x1.05c61p-29f;
  f69 = __sm_fma_rn(f65, f68, f67);
  f70 = f69 * 0x1.715476p+0f;
  f71 = f65 + 0x1.8000fep+23f;
  r2 = __float_as_uint(f71);
  r3 = r2 << 0x17u;
  f72 = __sm_f32bits(r3);
  f73 = __sm_ex2_approx_ftz(f70);
  f74 = f73 * f72;
  f75 = -f16;
  f76 = __sm_fma_rn(f75, f16, f57);
  f77 = __sm_fma_rn(f74, f76, f74);
  f78 = f56 * f77;
  p3 = f16 > 0x1.41c29p+3f;
  f79 = p3 ? 0.0f : f78;
  p4 = f3 < 0.0f;
  f80 = f49 - f79;
  f86 = p4 ? f80 : f79;
  p5 = !(f11 < -0x1p+0f);
  if (p5) goto L51_2;
  f81 = f1 - f3;
  f82 = __sm_add_rn(f81, f2);
  f83 = f3 * -0x1p+1f;
  f84 = f83 * f86;
  f86 = __sm_fma_rn(f84, f82, f86);
L51_2:;
  f85 = f86 * 0x1p-1f;
  __ret = f85;
  return __ret;
}

SASS_MATH_FN(float, normcdfinvf)(float a0) {
  float __ret;
  float f1, f2, f3, f7, f8, f9, f10, f11, f12, f13, f14, f15;
  float f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27;
  float f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39;
  float f40, f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51;
  float f52, f53, f54, f55, f56, f57, f58, f59;
  bool p1, p2, p3, p4, p5;
  f7 = a0;
  f1 = f7 + f7;
  f2 = -f1;
  f8 = 0x1p+1f;
  f3 = __sm_add_rn(f8, f2);
  p1 = f1 >= 0x1.bda512p-9f;
  p2 = f1 <= 0x1.ff212ep+0f;
  p3 = p1 && p2;
  if (p3) goto L52_2;
  goto L52_1;
L52_2:;
  f34 = __sm_mul_rn(f3, f1);
  f35 = __sm_lg2_approx_ftz(f34);
  f36 = -f35;
  f37 = 0x1.43ec8ap-27f;
  f38 = -0x1.14c6ep-32f;
  f39 = __sm_fma_rn(f38, f36, f37);
  f40 = -0x1.02dfb4p-23f;
  f41 = __sm_fma_rn(f39, f36, f40);
  f42 = 0x1.d1f08cp-23f;
  f43 = __sm_fma_rn(f41, f36, f42);
  f44 = 0x1.0e8554p-17f;
  f45 = __sm_fma_rn(f43, f36, f44);
  f46 = -0x1.09b69ap-15f;
  f47 = __sm_fma_rn(f45, f36, f46);
  f48 = -0x1.95fd42p-11f;
  f49 = __sm_fma_rn(f47, f36, f48);
  f50 = 0x1.6b804ep-8f;
  f51 = __sm_fma_rn(f49, f36, f50);
  f52 = 0x1.495c1ep-3f;
  f53 = __sm_fma_rn(f51, f36, f52);
  f54 = 0x1.c5bf88p-1f;
  f55 = __sm_fma_rn(f53, f36, f54);
  f59 = __sm_fma_rn(f55, f2, f55);
  goto L52_3;
L52_1:;
  p4 = f1 > 0x1p+0f;
  f9 = p4 ? f3 : f1;
  f10 = __sm_lg2_approx(f9);
  f11 = -f10;
  f12 = __sm_rsqrt_approx_ftz(f11);
  f13 = 0x1.fdf052p+6f;
  f14 = -0x1.f8e7e2p+5f;
  f15 = __sm_fma_rn(f14, f12, f13);
  f16 = -0x1.c86c38p+6f;
  f17 = __sm_fma_rn(f15, f12, f16);
  f18 = 0x1.e29b36p+5f;
  f19 = __sm_fma_rn(f17, f12, f18);
  f20 = -0x1.5ca366p+4f;
  f21 = __sm_fma_rn(f19, f12, f20);
  f22 = 0x1.9dea08p+2f;
  f23 = __sm_fma_rn(f21, f12, f22);
  f24 = -0x1.d53c0ap+0f;
  f25 = __sm_fma_rn(f23, f12, f24);
  f26 = -0x1.f0e3e8p-6f;
  f27 = __sm_fma_rn(f25, f12, f26);
  f28 = 0x1.aa6eeap-1f;
  f29 = __sm_fma_rn(f27, f12, f28);
  f30 = f12 * f11;
  p5 = f10 == __sm_f32bits(0xff800000u);
  f31 = p5 ? f11 : f30;
  f32 = -f31;
  f33 = p4 ? f32 : f31;
  f59 = __sm_mul_rn(f29, f33);
L52_3:;
  f56 = 0.0f;
  f57 = -0x1.6a09e6p+0f;
  f58 = __sm_fma_rn(f57, f59, f56);
  __ret = f58;
  return __ret;
}

SASS_MATH_FN(float, erfcxf)(float a0) {
  float __ret;
  float f1, f2, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41;
  float f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53;
  float f54, f55, f56, f57, f58, f59, f60, f61, f62, f63, f64, f65;
  float f66, f67, f68, f69, f70, f71, f72, f73, f74, f75, f76, f77;
  float f78, f79, f80, f81, f82, f84;
  bool p1, p2, p3;
  unsigned r2, r3;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = f2 < 0x1.41c29p+3f;
  if (p1) goto L53_2;
  goto L53_1;
L53_2:;
  f22 = f2 + -0x1p+2f;
  f23 = -0x1p+2f;
  f24 = f2 + 0x1p+2f;
  f25 = __sm_rcp_approx_ftz(f24);
  f26 = __sm_mul_rn(f22, f25);
  f27 = f26 + 0x1p+0f;
  f28 = 0x1p+0f;
  f29 = __sm_fma_rn(f23, f27, f2);
  f30 = -f26;
  f31 = __sm_fma_rn(f30, f2, f29);
  f32 = __sm_fma_rn(f25, f31, f26);
  f33 = 0x1.cdc0b6p-8f;
  f34 = 0x1.d34122p-11f;
  f35 = __sm_fma_rn(f34, f32, f33);
  f36 = -0x1.03f696p-6f;
  f37 = __sm_fma_rn(f35, f32, f36);
  f38 = 0x1.2a6e76p-5f;
  f39 = __sm_fma_rn(f37, f32, f38);
  f40 = -0x1.10f8b4p-4f;
  f41 = __sm_fma_rn(f39, f32, f40);
  f42 = 0x1.8043aap-4f;
  f43 = __sm_fma_rn(f41, f32, f42);
  f44 = -0x1.9da848p-4f;
  f45 = __sm_fma_rn(f43, f32, f44);
  f46 = 0x1.16e9bcp-4f;
  f47 = __sm_fma_rn(f45, f32, f46);
  f48 = 0x1.f7e2ep-7f;
  f49 = __sm_fma_rn(f47, f32, f48);
  f50 = -0x1.1df1a8p-3f;
  f51 = __sm_fma_rn(f49, f32, f50);
  f52 = 0x1.3ba592p+0f;
  f53 = __sm_fma_rn(f51, f32, f52);
  f54 = 0x1p+1f;
  f55 = __sm_fma_rn(f54, f2, f28);
  f56 = __sm_rcp_approx_ftz(f55);
  f57 = __sm_mul_rn(f53, f56);
  f58 = f57 * -0x1p+1f;
  f59 = __sm_fma_rn(f2, f58, f53);
  f60 = f59 - f57;
  f84 = __sm_fma_rn(f60, f56, f57);
  goto L53_3;
L53_1:;
  f8 = f2 * 0x1p-2f;
  f9 = 0x1p-2f;
  f10 = __sm_div_approx(f9, f8);
  f11 = f10 * f10;
  f12 = -0x1.ep+0f;
  f13 = 0x1.a4p+2f;
  f14 = __sm_fma_rn(f13, f11, f12);
  f15 = 0x1.8p-1f;
  f16 = __sm_fma_rn(f14, f11, f15);
  f17 = -0x1p-1f;
  f18 = __sm_fma_rn(f16, f11, f17);
  f19 = 0x1p+0f;
  f20 = __sm_fma_rn(f18, f11, f19);
  f21 = f10 * 0x1.20dd76p-1f;
  f84 = f21 * f20;
L53_3:;
  p2 = !(f1 < 0.0f);
  if (p2) goto L53_5;
  f61 = __sm_mul_rz(f2, f2);
  f62 = -f61;
  f63 = __sm_fma_rn(f2, f2, f62);
  f64 = 0x1p-1f;
  f65 = 0x1.77313ap-8f;
  f66 = __sm_fma_rn(f61, f65, f64);
  f67 = __sm_saturate(f66);
  f68 = 0x1.800002p+23f;
  f69 = 0x1.f8p+7f;
  f70 = __sm_fma_rm(f67, f69, f68);
  f71 = f70 + -0x1.8000fep+23f;
  f72 = -f71;
  f73 = 0x1.715476p+0f;
  f74 = __sm_fma_rn(f61, f73, f72);
  f75 = 0x1.4ae0cp-26f;
  f76 = __sm_fma_rn(f61, f75, f74);
  r2 = __float_as_uint(f70);
  r3 = r2 << 0x17u;
  f77 = __sm_f32bits(r3);
  f78 = __sm_ex2_approx_ftz(f76);
  f79 = f78 * f77;
  f80 = __sm_fma_rn(f78, f77, f79);
  f81 = __sm_fma_rn(f80, f63, f80);
  f82 = f81 - f84;
  p3 = f80 == __sm_f32bits(0x7f800000u);
  f84 = p3 ? f80 : f82;
L53_5:;
  __ret = f84;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, powf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4, f5;
  f1 = a0;
  f2 = a1;
  f3 = __sm_lg2_approx(f1);
  f4 = f2 * f3;
  f5 = __sm_ex2_approx(f4);
  __ret = f5;
  return __ret;
}
#else
SASS_MATH_FN(float, powf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41;
  float f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52, f53;
  float f54, f55, f56, f57, f58, f59, f60, f61, f62, f63, f64, f65;
  float f66, f67, f68, f69, f70, f71, f72, f73, f74, f75, f76, f77;
  float f78, f79, f80, f81, f83, f84, f85, f86, f87;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  bool p13, p14, p15, p16, p17, p18, p19, p20;
  unsigned r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
  unsigned r14, r15;
  f3 = a0;
  f1 = a1;
  f12 = f1 * 0x1p-1f;
  f13 = __sm_trunc(f12);
  f14 = f13 + f13;
  f15 = f1 - f14;
  f2 = __sm_fabs(f15);
  f4 = __sm_fabs(f3);
  p1 = f4 < 0x1p-126f;
  f16 = f4 * 0x1p+24f;
  f17 = p1 ? f16 : f4;
  f18 = p1 ? -0x1.8p+4f : 0.0f;
  r2 = __float_as_uint(f17);
  r3 = r2 + 0xc0cafb0du;
  r4 = r3 & 0xff800000u;
  r5 = r2 - r4;
  f19 = __sm_f32bits(r5);
  f20 = __sm_i2f_rn((int)r4);
  f21 = 0x1p-23f;
  f22 = __sm_fma_rn(f20, f21, f18);
  f23 = f19 + -0x1p+0f;
  f24 = f19 + 0x1p+0f;
  f87 = 0x1p+0f;
  f25 = __sm_rcp_approx_ftz(f24);
  f26 = f23 + f23;
  f27 = f26 * f25;
  f28 = f27 * f27;
  f29 = f23 - f27;
  f30 = f29 + f29;
  f31 = -f27;
  f32 = __sm_fma_rn(f31, f23, f30);
  f33 = __sm_mul_rn(f25, f32);
  f34 = 0x1.a5cfb6p-9f;
  f35 = 0x1.5865c8p-11f;
  f36 = __sm_fma_rn(f35, f28, f34);
  f37 = 0x1.2776e6p-6f;
  f38 = __sm_fma_rn(f36, f28, f37);
  f39 = 0x1.ec709ep-4f;
  f40 = __sm_fma_rn(f38, f28, f39);
  f41 = __sm_mul_rn(f40, f28);
  f42 = 0x1.715476p+0f;
  f43 = __sm_fma_rn(f27, f42, f22);
  f44 = f22 - f43;
  f45 = __sm_fma_rn(f27, f42, f44);
  f46 = __sm_fma_rn(f33, f42, f45);
  f47 = 0x1.4abc68p-26f;
  f48 = __sm_fma_rn(f27, f47, f46);
  f49 = f41 * 0x1.8p+1f;
  f50 = __sm_fma_rn(f49, f33, f48);
  f51 = __sm_fma_rn(f41, f27, f50);
  f52 = __sm_add_rn(f43, f51);
  f53 = -f43;
  f54 = __sm_add_rn(f52, f53);
  f55 = -f54;
  f56 = __sm_add_rn(f51, f55);
  f57 = __sm_mul_rn(f52, f1);
  f58 = -f57;
  f59 = __sm_fma_rn(f52, f1, f58);
  f60 = __sm_fma_rn(f56, f1, f59);
  f61 = __sm_rint(f57);
  f62 = f57 - f61;
  f63 = f60 + f62;
  f64 = 0x1.5f0bdap-10f;
  f65 = 0x1.3f971cp-13f;
  f66 = __sm_fma_rn(f65, f63, f64);
  f67 = 0x1.3b30acp-7f;
  f68 = __sm_fma_rn(f66, f63, f67);
  f69 = 0x1.c6af76p-5f;
  f70 = __sm_fma_rn(f68, f63, f69);
  f71 = 0x1.ebfbd8p-3f;
  f72 = __sm_fma_rn(f70, f63, f71);
  f73 = 0x1.62e43p-1f;
  f74 = __sm_fma_rn(f72, f63, f73);
  f75 = __sm_fma_rn(f74, f63, f87);
  r6 = __sm_f2i_rz(f61);
  p2 = f61 > 0.0f;
  r7 = p2 ? 0u : 0x83000000u;
  r8 = r7 + 0x7f000000u;
  f76 = __sm_f32bits(r8);
  f77 = f75 * f76;
  r9 = r6 << 0x17u;
  r10 = r9 - r7;
  f78 = __sm_f32bits(r10);
  f79 = f77 * f78;
  f80 = __sm_fabs(f57);
  p3 = f80 > 0x1.3p+7f;
  p4 = f57 < 0.0f;
  f81 = p4 ? 0.0f : __sm_f32bits(0x7f800000u);
  f5 = p3 ? f81 : f79;
  p5 = f3 == 0x1p+0f;
  p6 = f1 == 0.0f;
  p7 = p5 || p6;
  if (p7) goto L8_9;
  p8 = !(f4 <= __sm_f32bits(0x7f800000u));
  if (p8) goto L8_8;
  f6 = __sm_fabs(f1);
  p9 = !(f6 <= __sm_f32bits(0x7f800000u));
  if (p9) goto L8_8;
  goto L8_3;
L8_8:;
  f87 = __sm_add_rn(f3, f1);
L8_9:;
  __ret = f87;
  return __ret;
L8_3:;
  p10 = f3 == 0.0f;
  p11 = f4 == __sm_f32bits(0x7f800000u);
  p12 = p10 || p11;
  if (p12) goto L8_7;
  goto L8_4;
L8_7:;
  p19 = f2 == 0x1p+0f;
  f86 = f3 + f3;
  r11 = __float_as_uint(f86);
  r12 = r11 ^ 0x7f800000u;
  p20 = f1 < 0.0f;
  r13 = p20 ? r12 : r11;
  r14 = r13 & 0x7fffffffu;
  r15 = p19 ? r13 : r14;
  f87 = __sm_f32bits(r15);
  goto L8_9;
L8_4:;
  p13 = f3 == -0x1p+0f;
  p14 = f6 == __sm_f32bits(0x7f800000u);
  p15 = p13 && p14;
  if (p15) goto L8_9;
  p16 = !(f3 < 0.0f);
  f87 = f5;
  if (p16) goto L8_9;
  p17 = f2 == 0x1p+0f;
  f83 = -f5;
  f84 = p17 ? f83 : f5;
  f85 = __sm_floor(f1);
  p18 = f85 != f1;
  f87 = p18 ? __sm_f32bits(0x7fffffffu) : f84;
  goto L8_9;
}
#endif

SASS_MATH_FN(float, atan2f)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4, f10, f11, f12, f13, f14, f15, f16, f17;
  float f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29;
  float f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41;
  float f42, f43, f44, f45, f46;
  bool p1, p2, p3, p4, p5, p6, p7;
  unsigned r1;
  f2 = a0;
  f10 = a1;
  f1 = __sm_fabs(f10);
  f3 = __sm_fabs(f2);
  p1 = f3 > f1;
  f4 = p1 ? f3 : f1;
  f11 = p1 ? f1 : f3;
  f12 = __sm_div_rn(f11, f4);
  f13 = f12 * f12;
  f14 = -0x1.00ee9p-6f;
  f15 = 0x1.66e216p-9f;
  f16 = __sm_fma_rn(f15, f13, f14);
  f17 = 0x1.59b564p-5f;
  f18 = __sm_fma_rn(f16, f13, f17);
  f19 = -0x1.325a2p-4f;
  f20 = __sm_fma_rn(f18, f13, f19);
  f21 = 0x1.b3d4d8p-4f;
  f22 = __sm_fma_rn(f20, f13, f21);
  f23 = -0x1.22f962p-3f;
  f24 = __sm_fma_rn(f22, f13, f23);
  f25 = 0x1.9979cp-3f;
  f26 = __sm_fma_rn(f24, f13, f25);
  f27 = -0x1.5554fap-2f;
  f28 = __sm_fma_rn(f26, f13, f27);
  f29 = f13 * f28;
  f46 = __sm_fma_rn(f29, f12, f12);
  r1 = __float_as_uint(f10);
  if (p1) goto L19_3;
  goto L19_1;
L19_3:;
  p3 = (int)r1 < (int)0u;
  if (p3) goto L19_5;
  f33 = -f46;
  f34 = 0x1.aee9d6p+0f;
  f35 = 0x1.ddcb02p-1f;
  f46 = __sm_fma_rn(f35, f34, f33);
  goto L19_6;
L19_1:;
  p2 = (int)r1 > (int)0xffffffffu;
  if (p2) goto L19_6;
  f30 = -f46;
  f31 = 0x1.aee9d6p+0f;
  f32 = 0x1.ddcb02p+0f;
  f46 = __sm_fma_rn(f32, f31, f30);
  goto L19_6;
L19_5:;
  f36 = 0x1.aee9d6p+0f;
  f37 = 0x1.ddcb02p-1f;
  f46 = __sm_fma_rn(f37, f36, f46);
L19_6:;
  p4 = (int)r1 < (int)0u;
  f38 = p4 ? 0x1.921fb6p+1f : 0.0f;
  f39 = p4 ? 0x1.2d97c8p+1f : 0x1.921fb6p-1f;
  p5 = f1 == f3;
  f40 = p5 ? f39 : f46;
  p6 = f4 == 0.0f;
  f41 = p6 ? f38 : f40;
  f42 = f1 + f3;
  f43 = __sm_fabs(f42);
  p7 = !(f43 <= __sm_f32bits(0x7f800000u));
  f44 = __sm_copysign(f41, f2);
  f45 = p7 ? f42 : f44;
  __ret = f45;
  return __ret;
}

SASS_MATH_FN(float, fmodf)(float a0, float a1) {
  float __ret;
  float f1, f2, f4, f6, f7, f14, f15, f20, f21, f22, f23, f24;
  float f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36;
  float f37, f38, f39, f40, f41, f42, f43, f44, f45, f46;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  bool p13, p14, p15;
  unsigned r1, r2, r10, r11, r13, r15, r16, r17, r18, r19, r21, r22;
  unsigned r23, r24, r25;
  f2 = a0;
  f20 = a1;
  f1 = __sm_fabs(f20);
  f46 = __sm_fabs(f2);
  p1 = f46 < f1;
  if (p1) goto L24_12;
  f4 = f1 * 0x1p+23f;
  p2 = !(f46 <= f4);
  if (p2) goto L24_8;
  goto L24_2;
L24_8:;
  r1 = __float_as_uint(f46);
  r13 = r1 & 0x7fffffu;
  r24 = r13 | 0x3f800000u;
  r2 = __float_as_uint(f4);
  f45 = __sm_f32bits(r24);
  r15 = r2 + 0xf4800000u;
  r16 = r15 & 0xff800000u;
  r17 = r1 - r16;
  r25 = r17 & 0xff800000u;
  p8 = (int)r25 == (int)0u;
  if (p8) goto L24_11;
  r18 = r2 & 0x7fffffu;
  r19 = r18 | 0x3f800000u;
  f29 = __sm_f32bits(r19);
  f14 = __sm_rcp_approx_ftz(f29);
  f15 = -f29;
L24_10:;
  r21 = r25 < 0xb800000u ? r25 : 0xb800000u;
  r22 = r21 + r24;
  f30 = __sm_f32bits(r22);
  f31 = -0.0f;
  f32 = __sm_fma_rn(f30, f14, f31);
  f33 = __sm_fma_rn(f15, f32, f30);
  f34 = __sm_fma_rn(f33, f14, f32);
  f35 = __sm_fma_rn(f15, f34, f30);
  f36 = __sm_fma_rz(f35, f14, f34);
  f37 = __sm_trunc(f36);
  f45 = __sm_fma_rn(f15, f37, f30);
  r25 = r25 - r21;
  r24 = __float_as_uint(f45);
  p9 = (int)r25 != (int)0u;
  p10 = (int)r24 != (int)0u;
  p11 = p9 && p10;
  if (p11) goto L24_10;
L24_11:;
  r23 = r2 & 0xff800000u;
  f38 = __sm_f32bits(r23);
  p12 = r1 > 0x7f7fffffu;
  p13 = !(f1 > 0.0f);
  p14 = p13 || p12;
  f39 = p14 ? __sm_f32bits(0x7fffffffu) : f38;
  f40 = f45 * 0x1p-23f;
  f46 = f39 * f40;
  goto L24_12;
L24_2:;
  f21 = __sm_div_approx(f46, f1);
  f44 = __sm_trunc(f21);
  f6 = -f1;
  f7 = __sm_fma_rn(f6, f44, f46);
  r10 = __float_as_uint(f7);
  r11 = __float_as_uint(f1);
  p3 = r10 < r11;
  if (p3) goto L24_7;
  p4 = r10 > 0x80000000u;
  if (p4) goto L24_6;
  goto L24_4;
L24_6:;
  f27 = f44 + -0x1p+0f;
  f28 = f27 + -0x1p+0f;
  p7 = f7 < f6;
  f44 = p7 ? f28 : f27;
  goto L24_7;
L24_4:;
  f44 = f44 + 0x1p+0f;
  f22 = f1 + f1;
  p5 = !(f7 >= f22);
  if (p5) goto L24_7;
  f23 = f44 + 0x1p+0f;
  f24 = -0x1.8p+1f;
  f25 = __sm_fma_rn(f24, f1, f7);
  p6 = f25 >= 0.0f;
  f26 = f23 + 0x1p+0f;
  f44 = p6 ? f26 : f23;
L24_7:;
  f46 = __sm_fma_rn(f6, f44, f46);
L24_12:;
  f41 = __sm_fabs(f46);
  p15 = !(f41 <= __sm_f32bits(0x7f800000u));
  f42 = __sm_copysign(f46, f2);
  f43 = p15 ? f46 : f42;
  __ret = f43;
  return __ret;
}

SASS_MATH_FN(float, remainderf)(float a0, float a1) {
  float __ret;
  float f1, f2, f4, f6, f7, f14, f15, f20, f25, f26, f27, f28;
  float f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40;
  float f41, f42, f43, f44, f45, f46, f47, f48, f49, f52;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  bool p13, p14, p15, p16, p17, p18, p19, p20, p21, p22;
  unsigned r1, r3, r4, r6, r19, r22, r23, r24, r25, r27, r28, r29;
  unsigned r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r42, r43;
  unsigned r44;
  f2 = a0;
  f25 = a1;
  f1 = __sm_fabs(f25);
  f52 = __sm_fabs(f2);
  p1 = f52 < f1;
  r44 = 0u;
  if (p1) goto L25_12;
  f4 = f1 * 0x1p+23f;
  p2 = !(f52 <= f4);
  if (p2) goto L25_8;
  goto L25_2;
L25_8:;
  r3 = __float_as_uint(f52);
  r4 = __float_as_uint(f4);
  r22 = r3 & 0x7fffffu;
  r39 = r22 | 0x3f800000u;
  f49 = __sm_f32bits(r39);
  r23 = r4 + 0xf4800000u;
  r24 = r23 & 0xff800000u;
  r25 = r3 - r24;
  r6 = r25 & 0xff800000u;
  p8 = (int)r6 == (int)0u;
  r42 = 0u;
  r43 = r42;
  if (p8) goto L25_11;
  r27 = r4 & 0x7fffffu;
  r28 = r27 | 0x3f800000u;
  f34 = __sm_f32bits(r28);
  f14 = __sm_rcp_approx_ftz(f34);
  f15 = -f34;
  r42 = 0u;
  r43 = r6;
L25_10:;
  r29 = r43 < 0xb800000u ? r43 : 0xb800000u;
  r30 = r29 + r39;
  f35 = __sm_f32bits(r30);
  f36 = -0.0f;
  f37 = __sm_fma_rn(f35, f14, f36);
  f38 = __sm_fma_rn(f15, f37, f35);
  f39 = __sm_fma_rn(f38, f14, f37);
  f40 = __sm_fma_rn(f15, f39, f35);
  f41 = __sm_fma_rz(f40, f14, f39);
  f42 = __sm_trunc(f41);
  f49 = __sm_fma_rn(f15, f42, f35);
  r43 = r43 - r29;
  r39 = __float_as_uint(f49);
  r31 = r29 >> 0x17u;
  r32 = r42 << r31;
  r33 = __sm_f2ui_rz(f42);
  r42 = r32 + r33;
  p9 = (int)r43 != (int)0u;
  p10 = (int)r39 != (int)0u;
  p11 = p9 && p10;
  if (p11) goto L25_10;
L25_11:;
  p12 = r43 < 0x2000000u;
  r34 = r43 >> 0x17u;
  r35 = p12 ? r34 : 3u;
  r44 = r42 << r35;
  p13 = r3 > 0x7f7fffffu;
  p14 = !(f1 > 0.0f);
  p15 = p14 || p13;
  r36 = r4 & 0xff800000u;
  f43 = __sm_f32bits(r36);
  f44 = p15 ? __sm_f32bits(0x7fffffffu) : f43;
  f45 = f49 * 0x1p-23f;
  f52 = f44 * f45;
  goto L25_12;
L25_2:;
  f26 = __sm_div_approx(f52, f1);
  f48 = __sm_trunc(f26);
  f6 = -f1;
  f7 = __sm_fma_rn(f6, f48, f52);
  r1 = __float_as_uint(f7);
  r19 = __float_as_uint(f1);
  p3 = r1 < r19;
  if (p3) goto L25_7;
  p4 = r1 > 0x80000000u;
  if (p4) goto L25_6;
  goto L25_4;
L25_6:;
  f32 = f48 + -0x1p+0f;
  f33 = f32 + -0x1p+0f;
  p7 = f7 < f6;
  f48 = p7 ? f33 : f32;
  goto L25_7;
L25_4:;
  f48 = f48 + 0x1p+0f;
  f27 = f1 + f1;
  p5 = !(f7 >= f27);
  if (p5) goto L25_7;
  f28 = f48 + 0x1p+0f;
  f29 = -0x1.8p+1f;
  f30 = __sm_fma_rn(f29, f1, f7);
  p6 = f30 >= 0.0f;
  f31 = f28 + 0x1p+0f;
  f48 = p6 ? f31 : f28;
L25_7:;
  r44 = __sm_f2ui_rz(f48);
  f52 = __sm_fma_rn(f6, f48, f52);
L25_12:;
  f20 = f52 + f52;
  p16 = f20 > f1;
  if (p16) goto L25_14;
  p17 = f20 != f1;
  r37 = r44 & 1u;
  p18 = r37 == 1u;
  p19 = !p18;
  p20 = p19 || p17;
  if (p20) goto L25_15;
L25_14:;
  f52 = f52 - f1;
L25_15:;
  r38 = __float_as_uint(f2);
  p21 = (int)r38 > (int)0xffffffffu;
  if (p21) goto L25_17;
  f46 = __sm_fabs(f52);
  p22 = f46 <= __sm_f32bits(0x7f800000u);
  f47 = -f52;
  f52 = p22 ? f47 : f52;
L25_17:;
  __ret = f52;
  return __ret;
}

SASS_MATH_FN(float, copysignf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3;
  f1 = a0;
  f2 = a1;
  f3 = __sm_copysign(f1, f2);
  __ret = f3;
  return __ret;
}

SASS_MATH_FN(float, fdimf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4;
  bool p1;
  f1 = a0;
  f2 = a1;
  f3 = __sm_add_rn(f1, -(f2));
  p1 = f1 <= f2;
  f4 = p1 ? 0.0f : f3;
  __ret = f4;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(float, hypotf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16;
  bool p1, p2;
  unsigned r2, r3, r4, r5, r6, r7, r8, r9;
  f1 = a0;
  f2 = a1;
  f3 = __sm_fabs(f1);
  f4 = __sm_fabs(f2);
  r2 = __float_as_uint(f4);
  r3 = __float_as_uint(f3);
  r4 = (int)r2 < (int)r3 ? (int)r2 : (int)r3;
  f5 = __sm_f32bits(r4);
  r5 = (int)r2 > (int)r3 ? (int)r2 : (int)r3;
  f6 = __sm_f32bits(r5);
  r6 = r5 & 0xfe000000u;
  r7 = 0x7e800000u;
  r8 = r7 - r6;
  f7 = __sm_f32bits(r8);
  f8 = f5 * f7;
  f9 = f6 * f7;
  f10 = f8 * f8;
  f11 = __sm_fma_rn(f9, f9, f10);
  f12 = __sm_sqrt_approx(f11);
  r9 = r6 | 0x800000u;
  f13 = __sm_f32bits(r9);
  f14 = f12 * f13;
  p1 = f5 == 0.0f;
  f15 = p1 ? f6 : f14;
  p2 = f5 == __sm_f32bits(0x7f800000u);
  f16 = p2 ? __sm_f32bits(0x7f800000u) : f15;
  __ret = f16;
  return __ret;
}
#else
SASS_MATH_FN(float, hypotf)(float a0, float a1) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
  float f13, f14, f15, f16;
  bool p1, p2;
  unsigned r2, r3, r4, r5, r6, r7, r8, r9;
  f1 = a0;
  f2 = a1;
  f3 = __sm_fabs(f1);
  f4 = __sm_fabs(f2);
  r2 = __float_as_uint(f4);
  r3 = __float_as_uint(f3);
  r4 = (int)r2 < (int)r3 ? (int)r2 : (int)r3;
  f5 = __sm_f32bits(r4);
  r5 = (int)r2 > (int)r3 ? (int)r2 : (int)r3;
  f6 = __sm_f32bits(r5);
  r6 = r5 & 0xfe000000u;
  r7 = 0x7e800000u;
  r8 = r7 - r6;
  f7 = __sm_f32bits(r8);
  f8 = f5 * f7;
  f9 = f6 * f7;
  f10 = f8 * f8;
  f11 = __sm_fma_rn(f9, f9, f10);
  f12 = __sm_sqrt_rn(f11);
  r9 = r6 | 0x800000u;
  f13 = __sm_f32bits(r9);
  f14 = f12 * f13;
  p1 = f5 == 0.0f;
  f15 = p1 ? f6 : f14;
  p2 = f5 == __sm_f32bits(0x7f800000u);
  f16 = p2 ? __sm_f32bits(0x7f800000u) : f15;
  __ret = f16;
  return __ret;
}
#endif

SASS_MATH_FN(float, fmaf)(float a0, float a1, float a2) {
  float __ret;
  float f1, f2, f3, f4;
  f1 = a0;
  f2 = a1;
  f3 = a2;
  f4 = __sm_fma_rn(f1, f2, f3);
  __ret = f4;
  return __ret;
}

SASS_MATH_FN(int, ilogbf)(float a0) {
  int __ret;
  float f1, f2;
  bool p1, p2, p3, p4;
  unsigned r1, r6, r7, r8, r9, r10, r11, r12;
  f1 = a0;
  f2 = __sm_fabs(f1);
  r1 = __float_as_uint(f2);
  p1 = r1 < 0x800000u;
  if (p1) goto L47_2;
  goto L47_1;
L47_2:;
  r9 = __sm_clz32(r1);
  r10 = 0xffffff8au;
  r11 = r10 - r9;
  p4 = (int)r1 == (int)0u;
  r12 = p4 ? 0x80000000u : r11;
  goto L47_3;
L47_1:;
  r6 = r1 >> 0x17u;
  r7 = r6 + 0xffffff81u;
  p2 = (int)r1 == (int)0x7f800000u;
  r8 = p2 ? 0x7fffffffu : r7;
  p3 = r1 > 0x7f800000u;
  r12 = p3 ? 0x80000000u : r8;
L47_3:;
  __ret = r12;
  return __ret;
}

SASS_MATH_FN(bool, isnan)(float a0) {
  bool __ret;
  float f1, f2;
  bool p1;
  unsigned r2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = !(f2 <= __sm_f32bits(0x7f800000u));
  r2 = p1 ? 1u : 0u;
  __ret = r2 != 0;
  return __ret;
}

SASS_MATH_FN(bool, isinf)(float a0) {
  bool __ret;
  float f1, f2;
  bool p1;
  unsigned r2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = f2 == __sm_f32bits(0x7f800000u);
  r2 = p1 ? 1u : 0u;
  __ret = r2 != 0;
  return __ret;
}

SASS_MATH_FN(bool, isfinite)(float a0) {
  bool __ret;
  float f1, f2;
  bool p1;
  unsigned r2;
  f1 = a0;
  f2 = __sm_fabs(f1);
  p1 = f2 < __sm_f32bits(0x7f800000u);
  r2 = p1 ? 1u : 0u;
  __ret = r2 != 0;
  return __ret;
}

SASS_MATH_FN(bool, signbit)(float a0) {
  bool __ret;
  unsigned r2, r3;
  r2 = __float_as_uint(a0);
  r3 = r2 >> 0x1fu;
  __ret = r3 != 0;
  return __ret;
}

#if SASS_MATH_FAST
SASS_MATH_FN(void, sincosf)(float a0, float *o0, float *o1) {
  float f1, f2, f3;
  f1 = a0;
  f2 = __sm_sin_approx(f1);
  *o0 = f2;
  f3 = __sm_cos_approx(f1);
  *o1 = f3;
  return;
}
#else
SASS_MATH_FN(void, sincosf)(float a0, float *o0, float *o1) {
  float f1, f3, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16;
  float f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28;
  float f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40;
  float f41;
  double fd1, fd2;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;
  unsigned r2, r3, r4, r5, r8, r11, r20, r21, r22, r23, r24, r25;
  unsigned r26, r27, r28, r29, r30, r31, r32, r33, r34, r35, r36, r37;
  unsigned r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, r49, r50;
  unsigned r51, r52, r53, r54, r55, r56;
  unsigned long long rd1, rd17, rd18, rd19, rd20, rd21, rd22, rd23, rd24, rd28, rd29, rd30;
  unsigned __d[7];
  f1 = a0;
  rd1 = 0ull + 0ull;
  f7 = f1 * 0x1.45f306p-1f;
  r56 = __sm_f2i_rn(f7);
  f8 = __sm_i2f_rn((int)r56);
  f9 = -0x1.921fb4p+0f;
  f10 = __sm_fma_rn(f8, f9, f1);
  f11 = -0x1.4442dp-24f;
  f12 = __sm_fma_rn(f8, f11, f10);
  f13 = -0x1.84698ap-48f;
  f41 = __sm_fma_rn(f8, f13, f12);
  f3 = __sm_fabs(f1);
  p1 = !(f3 >= 0x1.9c8fp+16f);
  if (p1) goto L134_8;
  p2 = f3 == __sm_f32bits(0x7f800000u);
  if (p2) goto L134_7;
  goto L134_2;
L134_7:;
  f16 = 0.0f;
  f41 = __sm_mul_rn(f1, f16);
  r56 = 0u;
  goto L134_8;
L134_2:;
  r2 = __float_as_uint(f1);
  r20 = r2 >> 0x17u;
  r21 = r20 & 0xffu;
  r3 = r21 + 0xffffff80u;
  r22 = r2 << 8u;
  r4 = r22 | 0x80000000u;
  r5 = r3 >> 5u;
  rd30 = 0ull;
  r53 = 0u;
  rd29 = 0ull;
  rd28 = rd1;
L134_3:;
  r23 = __sm_i2opi_f[(rd29) >> 2];
  rd17 = (unsigned long long)r23 * (unsigned long long)r4 + rd30;
  rd30 = rd17 >> 0x20u;
  __d[(rd28) >> 2] = (unsigned)rd17;
  rd29 = rd29 + 4ull;
  rd28 = rd28 + 4ull;
  r53 = r53 + 1u;
  p3 = (int)r53 != (int)6u;
  if (p3) goto L134_3;
  __d[(rd1+24) >> 2] = (unsigned)rd30;
  r24 = 4u;
  r8 = r24 - r5;
  r25 = 6u;
  r26 = r25 - r5;
  rd18 = (long long)(int)r26 * (long long)(int)4u;
  rd19 = rd1 + rd18;
  r54 = __d[(rd19) >> 2];
  r55 = __d[(rd19-4) >> 2];
  r11 = r3 & 0x1fu;
  p4 = (int)r11 == (int)0u;
  if (p4) goto L134_6;
  r27 = 0x20u;
  r28 = r27 - r11;
  r29 = r55 >> r28;
  r30 = r54 << r11;
  r54 = r29 + r30;
  rd20 = (long long)(int)r8 * (long long)(int)4u;
  rd21 = rd1 + rd20;
  r31 = __d[(rd21) >> 2];
  r32 = r31 >> r28;
  r33 = r55 << r11;
  r55 = r32 + r33;
L134_6:;
  r34 = r2 & 0x80000000u;
  r35 = r55 >> 0x1eu;
  r36 = r54 << 2u;
  r37 = r35 | r36;
  r38 = r37 >> 0x1fu;
  r39 = r54 >> 0x1eu;
  r40 = r38 + r39;
  r41 = -r40;
  p5 = (int)r34 == (int)0u;
  r56 = p5 ? r40 : r41;
  p6 = (int)r38 != (int)0u;
  r42 = r34 ^ 0x80000000u;
  r43 = p6 ? r42 : r34;
  r44 = p6 ? 0xffffffffu : 0u;
  r45 = r37 ^ r44;
  r46 = r55 << 2u;
  r47 = r46 ^ r44;
  rd22 = (unsigned long long)r45;
  rd23 = (unsigned long long)r47;
  rd24 = __sm_bfi64(rd22, rd23, 0x20u, 0x20u);
  fd1 = (double)(long long)rd24;
  fd2 = fd1 * 0x1.921fb54442d19p-64;
  f14 = (float)fd2;
  p7 = (int)r43 == (int)0u;
  f15 = -f14;
  f41 = p7 ? f14 : f15;
L134_8:;
  f17 = f41 * f41;
  f18 = -0x1.6c0fdap-10f;
  f19 = 0x1.9758p-16f;
  f20 = __sm_fma_rn(f19, f17, f18);
  f21 = 0x1.555576p-5f;
  f22 = __sm_fma_rn(f20, f17, f21);
  f23 = -0x1.fffffep-2f;
  f24 = __sm_fma_rn(f22, f17, f23);
  f25 = 0x1p+0f;
  f26 = __sm_fma_rn(f24, f17, f25);
  f27 = 0.0f;
  f28 = __sm_fma_rn(f17, f41, f27);
  f29 = 0x1.110bc8p-7f;
  f30 = -0x1.9a82a6p-13f;
  f31 = __sm_fma_rn(f30, f17, f29);
  f32 = -0x1.55555p-3f;
  f33 = __sm_fma_rn(f31, f17, f32);
  f34 = __sm_fma_rn(f33, f28, f41);
  r49 = r56 & 1u;
  p8 = r49 == 1u;
  f35 = p8 ? f26 : f34;
  f36 = p8 ? f34 : f26;
  r50 = r56 & 2u;
  p9 = (int)r50 == (int)0u;
  f37 = -f35;
  f38 = p9 ? f35 : f37;
  r51 = r56 + 1u;
  r52 = r51 & 2u;
  p10 = (int)r52 == (int)0u;
  f39 = -f36;
  f40 = p10 ? f36 : f39;
  *o0 = f38;
  *o1 = f40;
  return;
}
#endif

SASS_MATH_FN(void, sincospif)(float a0, float *o0, float *o1) {
  float f1, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20;
  float f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, f31, f32;
  float f33, f34, f35, f36, f38, f39, f40, f41, f42, f44, f45;
  bool p1, p2, p3, p4, p5;
  unsigned r2, r3, r4, r5, r6;
  f1 = a0;
  f10 = f1 + f1;
  f11 = __sm_rint(f10);
  r2 = __sm_f2i_rz(f11);
  f12 = -f11;
  f13 = 0x1p-1f;
  f14 = __sm_fma_rn(f12, f13, f1);
  f15 = __sm_mul_rn(f14, f14);
  f16 = 0x1.466b2p+1f;
  f17 = -0x1.2f5992p-1f;
  f18 = __sm_fma_rn(f17, f15, f16);
  f19 = -0x1.55a5cp+0f;
  f20 = 0x1.d09c24p-3f;
  f21 = __sm_fma_rn(f20, f15, f19);
  f22 = -0x1.4abbecp+2f;
  f23 = __sm_fma_rn(f18, f15, f22);
  f24 = 0x1.03c19ep+2f;
  f25 = __sm_fma_rn(f21, f15, f24);
  f26 = 0.0f;
  f27 = __sm_fma_rn(f15, f14, f26);
  f28 = -0x1.3bd3ccp+2f;
  f29 = __sm_fma_rn(f25, f15, f28);
  f30 = __sm_fma_rn(f23, f27, f26);
  f31 = 0x1p+0f;
  f32 = __sm_fma_rn(f29, f15, f31);
  f33 = 0x1.921fb6p+1f;
  f34 = __sm_fma_rn(f33, f14, f30);
  r3 = r2 & 1u;
  p1 = r3 == 1u;
  f35 = p1 ? f32 : f34;
  f45 = p1 ? f34 : f32;
  r4 = r2 & 2u;
  p2 = (int)r4 == (int)0u;
  f36 = -f35;
  f44 = p2 ? f35 : f36;
  r5 = r2 + 1u;
  r6 = r5 & 2u;
  p3 = (int)r6 == (int)0u;
  if (p3) goto L136_2;
  f38 = -0x1p+0f;
  f45 = __sm_fma_rn(f45, f38, f26);
L136_2:;
  f39 = __sm_trunc(f1);
  p4 = f39 != f1;
  if (p4) goto L136_4;
  f40 = 0.0f;
  f44 = __sm_mul_rn(f1, f40);
L136_4:;
  f41 = __sm_fabs(f1);
  p5 = !(f41 > 0x1p+24f);
  if (p5) goto L136_6;
  f42 = 0x1p+0f;
  f45 = __sm_add_rn(f44, f42);
L136_6:;
  *o0 = f44;
  *o1 = f45;
  return;
}

SASS_MATH_FN(float, ldexpf)(float a0, int a1) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7;
  bool p1;
  unsigned r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
  f1 = a0;
  r2 = a1;
  r3 = (int)r2 < (int)0x116u ? (int)r2 : (int)0x116u;
  r4 = (int)r3 > (int)0xfffffeeau ? (int)r3 : (int)0xfffffeeau;
  r5 = r4 + 0x1fcu;
  r6 = r5 >> 2u;
  r7 = r6 * 0xfffffffdu + r5;
  r8 = (int)r4 < 0 ? -(int)r4 : (int)r4;
  p1 = (int)r8 < (int)0x65u;
  r9 = r4 + 0x7fu;
  r10 = p1 ? r9 : r7;
  r11 = r6 << 0x17u;
  r12 = p1 ? 0x3f800000u : r11;
  f2 = __sm_f32bits(r12);
  r13 = r10 << 0x17u;
  f3 = __sm_f32bits(r13);
  f4 = __sm_mul_rn(f1, f3);
  f5 = __sm_mul_rn(f4, f2);
  f6 = __sm_mul_rn(f5, f2);
  f7 = __sm_mul_rn(f6, f2);
  __ret = f7;
  return __ret;
}

SASS_MATH_FN(float, scalbnf)(float a0, int a1) {
  float __ret;
  float f1, f2, f3, f4, f5, f6, f7;
  bool p1;
  unsigned r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;
  f1 = a0;
  r2 = a1;
  r3 = (int)r2 < (int)0x116u ? (int)r2 : (int)0x116u;
  r4 = (int)r3 > (int)0xfffffeeau ? (int)r3 : (int)0xfffffeeau;
  r5 = r4 + 0x1fcu;
  r6 = r5 >> 2u;
  r7 = r6 * 0xfffffffdu + r5;
  r8 = (int)r4 < 0 ? -(int)r4 : (int)r4;
  p1 = (int)r8 < (int)0x65u;
  r9 = r4 + 0x7fu;
  r10 = p1 ? r9 : r7;
  r11 = r6 << 0x17u;
  r12 = p1 ? 0x3f800000u : r11;
  f2 = __sm_f32bits(r12);
  r13 = r10 << 0x17u;
  f3 = __sm_f32bits(r13);
  f4 = __sm_mul_rn(f1, f3);
  f5 = __sm_mul_rn(f4, f2);
  f6 = __sm_mul_rn(f5, f2);
  f7 = __sm_mul_rn(f6, f2);
  __ret = f7;
  return __ret;
}

SASS_MATH_FN(float, frexpf)(float a0, int *o1) {
  float __ret;
  float f1, f2, f8, f9, f11, f12;
  bool p1, p2, p3, p4;
  unsigned r1, r2, r4, r5, r6, r7, r8;
  f8 = a0;
  f1 = __sm_fabs(f8);
  p1 = f1 < 0x1p-126f;
  f9 = f8 * 0x1p+24f;
  f2 = p1 ? f9 : f8;
  r1 = __float_as_uint(f2);
  r4 = r1 >> 0x17u;
  r2 = r4 & 0xffu;
  p2 = f1 == 0.0f;
  p3 = (int)r2 == (int)0xffu;
  p4 = p2 || p3;
  if (p4) goto L139_2;
  goto L139_1;
L139_2:;
  f11 = f2 + f2;
  f12 = 0.0f;
  goto L139_3;
L139_1:;
  r5 = p1 ? 0xffffff6au : 0xffffff82u;
  r6 = r5 + r2;
  r7 = r1 & 0x807fffffu;
  r8 = r7 | 0x3f000000u;
  f11 = __sm_f32bits(r8);
  f12 = __sm_f32bits(r6);
L139_3:;
  __ret = f11;
  *o1 = __float_as_int(f12);
  return __ret;
}

SASS_MATH_FN(float, modff)(float a0, float *o1) {
  float __ret;
  float f1, f2, f3, f4, f5;
  bool p1;
  unsigned r2, r3;
  f1 = a0;
  f2 = __sm_trunc(f1);
  *o1 = f2;
  p1 = f2 == f1;
  r2 = __float_as_uint(f1);
  r3 = r2 & 0x80000000u;
  f3 = __sm_f32bits(r3);
  f4 = f1 - f2;
  f5 = p1 ? f3 : f4;
  __ret = f5;
  return __ret;
}

SASS_MATH_FN(float, remquof)(float a0, float a1, int *o2) {
  float __ret;
  float f1, f2, f3, f5, f7, f8, f15, f16, f21, f26, f27, f28;
  float f29, f30, f31, f32, f33, f34, f35, f36, f37, f38, f39, f40;
  float f41, f42, f43, f44, f45, f46, f47, f48, f49, f50, f51, f52;
  float f55;
  bool p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  bool p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23;
  unsigned r1, r3, r4, r6, r19, r22, r25, r26, r27, r28, r30, r31;
  unsigned r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43;
  unsigned r44, r45, r46, r47, r50, r51, r52;
  f1 = a0;
  f2 = a1;
  f3 = __sm_fabs(f2);
  f55 = __sm_fabs(f1);
  p1 = f55 < f3;
  r52 = 0u;
  if (p1) goto L141_12;
  f5 = f3 * 0x1p+23f;
  p2 = !(f55 <= f5);
  if (p2) goto L141_8;
  goto L141_2;
L141_8:;
  r3 = __float_as_uint(f55);
  r4 = __float_as_uint(f5);
  r25 = r3 & 0x7fffffu;
  r47 = r25 | 0x3f800000u;
  f52 = __sm_f32bits(r47);
  r26 = r4 + 0xf4800000u;
  r27 = r26 & 0xff800000u;
  r28 = r3 - r27;
  r6 = r28 & 0xff800000u;
  p8 = (int)r6 == (int)0u;
  r50 = 0u;
  r51 = r50;
  if (p8) goto L141_11;
  r30 = r4 & 0x7fffffu;
  r31 = r30 | 0x3f800000u;
  f34 = __sm_f32bits(r31);
  f15 = __sm_rcp_approx_ftz(f34);
  f16 = -f34;
  r50 = 0u;
  r51 = r6;
L141_10:;
  r32 = r51 < 0xb800000u ? r51 : 0xb800000u;
  r33 = r32 + r47;
  f35 = __sm_f32bits(r33);
  f36 = -0.0f;
  f37 = __sm_fma_rn(f35, f15, f36);
  f38 = __sm_fma_rn(f16, f37, f35);
  f39 = __sm_fma_rn(f38, f15, f37);
  f40 = __sm_fma_rn(f16, f39, f35);
  f41 = __sm_fma_rz(f40, f15, f39);
  f42 = __sm_trunc(f41);
  f52 = __sm_fma_rn(f16, f42, f35);
  r51 = r51 - r32;
  r47 = __float_as_uint(f52);
  r34 = r32 >> 0x17u;
  r35 = r50 << r34;
  r36 = __sm_f2ui_rz(f42);
  r50 = r35 + r36;
  p9 = (int)r51 != (int)0u;
  p10 = (int)r47 != (int)0u;
  p11 = p9 && p10;
  if (p11) goto L141_10;
L141_11:;
  p12 = r51 < 0x2000000u;
  r37 = r51 >> 0x17u;
  r38 = p12 ? r37 : 3u;
  r52 = r50 << r38;
  p13 = r3 > 0x7f7fffffu;
  p14 = !(f3 > 0.0f);
  p15 = p14 || p13;
  r39 = r4 & 0xff800000u;
  f43 = __sm_f32bits(r39);
  f44 = p15 ? __sm_f32bits(0x7fffffffu) : f43;
  f45 = f52 * 0x1p-23f;
  f55 = f44 * f45;
  goto L141_12;
L141_2:;
  f26 = __sm_div_approx(f55, f3);
  f51 = __sm_trunc(f26);
  f7 = -f3;
  f8 = __sm_fma_rn(f7, f51, f55);
  r1 = __float_as_uint(f8);
  r22 = __float_as_uint(f3);
  p3 = r1 < r22;
  if (p3) goto L141_7;
  p4 = r1 > 0x80000000u;
  if (p4) goto L141_6;
  goto L141_4;
L141_6:;
  f32 = f51 + -0x1p+0f;
  f33 = f32 + -0x1p+0f;
  p7 = f8 < f7;
  f51 = p7 ? f33 : f32;
  goto L141_7;
L141_4:;
  f51 = f51 + 0x1p+0f;
  f27 = f3 + f3;
  p5 = !(f8 >= f27);
  if (p5) goto L141_7;
  f28 = f51 + 0x1p+0f;
  f29 = -0x1.8p+1f;
  f30 = __sm_fma_rn(f29, f3, f8);
  p6 = f30 >= 0.0f;
  f31 = f28 + 0x1p+0f;
  f51 = p6 ? f31 : f28;
L141_7:;
  r52 = __sm_f2ui_rz(f51);
  f55 = __sm_fma_rn(f7, f51, f55);
L141_12:;
  f21 = f55 + f55;
  p16 = f21 > f3;
  if (p16) goto L141_14;
  p17 = f21 != f3;
  r40 = r52 & 1u;
  p18 = r40 == 1u;
  p19 = !p18;
  p20 = p19 || p17;
  if (p20) goto L141_15;
L141_14:;
  r52 = r52 + 1u;
  f55 = f55 - f3;
L141_15:;
  r41 = __float_as_uint(f1);
  r42 = __float_as_uint(f2);
  r19 = r42 ^ r41;
  p21 = (int)r41 > (int)0xffffffffu;
  if (p21) goto L141_17;
  f46 = __sm_fabs(f55);
  p22 = f46 <= __sm_f32bits(0x7f800000u);
  f47 = -f55;
  f55 = p22 ? f47 : f55;
L141_17:;
  r43 = (int)r19 >> 0x1fu;
  r44 = r52 & 7u;
  r45 = r44 ^ r43;
  r46 = r45 - r43;
  f48 = __sm_fabs(f55);
  p23 = !(f48 <= __sm_f32bits(0x7f800000u));
  __ret = f55;
  f49 = __sm_f32bits(r46);
  f50 = p23 ? 0.0f : f49;
  *o2 = __float_as_int(f50);
  return __ret;
}
// ---- END GENERATED ----

// ---- integer helpers and overloads (CUDA device_functions / math_functions semantics) ----
SASS_MATH_FN(int, __popc)(unsigned x) { return (int)__sm_popc32(x); }
SASS_MATH_FN(int, __popcll)(unsigned long long x) { return (int)__sm_popc64(x); }
SASS_MATH_FN(int, __clz)(int x) { return (int)__sm_clz32((unsigned)x); }
SASS_MATH_FN(int, __clzll)(long long x) { return (int)__sm_clz64((unsigned long long)x); }
SASS_MATH_FN(int, __ffs)(int x) { return x == 0 ? 0 : (int)__sm_clz32(__sm_brev32((unsigned)x)) + 1; }
SASS_MATH_FN(int, __ffsll)(long long x) { return x == 0 ? 0 : (int)__sm_clz64(__sm_brev64((unsigned long long)x)) + 1; }
SASS_MATH_FN(unsigned, __brev)(unsigned x) { return __sm_brev32(x); }
SASS_MATH_FN(unsigned long long, __brevll)(unsigned long long x) { return __sm_brev64(x); }
SASS_MATH_FN(unsigned, __byte_perm)(unsigned x, unsigned y, unsigned s) { return __sm_prmt(x, y, s & 0x7777u); }
SASS_MATH_FN(unsigned, __funnelshift_r)(unsigned lo, unsigned hi, unsigned s) {
  return (unsigned)((((unsigned long long)hi << 32) | lo) >> (s & 31));
}
SASS_MATH_FN(unsigned, __funnelshift_lc)(unsigned lo, unsigned hi, unsigned s) {
  return (unsigned)(((((unsigned long long)hi << 32) | lo) << (s > 32 ? 32 : s)) >> 32);
}
SASS_MATH_FN(unsigned, __funnelshift_rc)(unsigned lo, unsigned hi, unsigned s) {
  return (unsigned)((((unsigned long long)hi << 32) | lo) >> (s > 32 ? 32 : s));
}
SASS_MATH_FN(int, __mul24)(int x, int y) { return (int)((unsigned)((x << 8) >> 8) * (unsigned)((y << 8) >> 8)); }
SASS_MATH_FN(unsigned, __umul24)(unsigned x, unsigned y) { return (x & 0xffffffu) * (y & 0xffffffu); }
SASS_MATH_FN(unsigned, __sad)(int x, int y, unsigned z) { return (x > y ? (unsigned)x - (unsigned)y : (unsigned)y - (unsigned)x) + z; }
SASS_MATH_FN(unsigned, __usad)(unsigned x, unsigned y, unsigned z) { return (x > y ? x - y : y - x) + z; }
SASS_MATH_FN(int, __hadd)(int x, int y) { return (int)(((long long)x + y) >> 1); }
SASS_MATH_FN(int, __rhadd)(int x, int y) { return (int)(((long long)x + y + 1) >> 1); }
SASS_MATH_FN(unsigned, __uhadd)(unsigned x, unsigned y) { return (unsigned)(((unsigned long long)x + y) >> 1); }
SASS_MATH_FN(unsigned, __urhadd)(unsigned x, unsigned y) { return (unsigned)(((unsigned long long)x + y + 1) >> 1); }
SASS_MATH_FN(long long, __mul64hi)(long long x, long long y) { return (long long)(((__int128)x * y) >> 64); }
SASS_MATH_FN(unsigned long long, __umul64hi)(unsigned long long x, unsigned long long y) {
  return (unsigned long long)(((unsigned __int128)x * y) >> 64);
}
SASS_MATH_FN(int, abs)(int x) { return x < 0 ? -x : x; }
SASS_MATH_FN(long long, llabs)(long long x) { return x < 0 ? -x : x; }
SASS_MATH_FN(long, labs)(long x) { return x < 0 ? -x : x; }
SASS_MATH_FN(long long, abs)(long long x) { return x < 0 ? -x : x; }
SASS_MATH_FN(long, abs)(long x) { return x < 0 ? -x : x; }
SASS_MATH_FN(float, abs)(float x) { return fabsf(x); }
SASS_MATH_FN(unsigned, umin)(unsigned a, unsigned b) { return a < b ? a : b; }
SASS_MATH_FN(unsigned, umax)(unsigned a, unsigned b) { return a > b ? a : b; }
SASS_MATH_FN(long long, llmin)(long long a, long long b) { return a < b ? a : b; }
SASS_MATH_FN(long long, llmax)(long long a, long long b) { return a > b ? a : b; }
SASS_MATH_FN(unsigned long long, ullmin)(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
SASS_MATH_FN(unsigned long long, ullmax)(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
SASS_MATH_FN(long long, min)(long long a, long long b) { return a < b ? a : b; }
SASS_MATH_FN(long long, max)(long long a, long long b) { return a > b ? a : b; }
SASS_MATH_FN(unsigned long long, min)(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
SASS_MATH_FN(unsigned long long, max)(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
SASS_MATH_FN(unsigned, min)(int a, unsigned b) { return (unsigned)a < b ? (unsigned)a : b; }
SASS_MATH_FN(unsigned, min)(unsigned a, int b) { return a < (unsigned)b ? a : (unsigned)b; }
SASS_MATH_FN(unsigned, max)(int a, unsigned b) { return (unsigned)a > b ? (unsigned)a : b; }
SASS_MATH_FN(unsigned, max)(unsigned a, int b) { return a > (unsigned)b ? a : (unsigned)b; }
SASS_MATH_FN(float, min)(float a, float b) { return fminf(a, b); }
SASS_MATH_FN(float, max)(float a, float b) { return fmaxf(a, b); }
SASS_MATH_FN(float, __uint_as_float)(unsigned x) { return __sm_f32bits(x); }
