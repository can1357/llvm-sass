// CUDA half/bfloat16/fp8 types, vector types, atomics, memory fences, warp
// intrinsics and cache-hinted loads for the llvm-sass flow (force-included
// through cuda_min.h; no CUDA SDK headers). Semantics follow CUDA 13 on
// sm_90: the same operations nvcc's cuda_fp16/cuda_bf16/cuda_fp8 headers and
// sm_*_intrinsics.h emit, built from clang builtins the SASS backend lowers
// (HADD2/HMUL2/HFMA2/HMNMX2/HSETP2, F2FP, I2F.F16, ATOM*/RED, MEMBAR, SHFL,
// VOTE, MATCH, REDUX).
#pragma once

#define SASS_INL_MEMBER __device__ __forceinline__

//===----------------------------------------------------------------------===//
// Vector types
//===----------------------------------------------------------------------===//

#define SASS_VEC2(T, N, A)                                                     \
  struct __attribute__((aligned(A))) N##2 { T x, y; };                         \
  SASS_INL N##2 make_##N##2(T x, T y) { return {x, y}; }
#define SASS_VEC3(T, N)                                                        \
  struct N##3 { T x, y, z; };                                                  \
  SASS_INL N##3 make_##N##3(T x, T y, T z) { return {x, y, z}; }
#define SASS_VEC4(T, N, A)                                                     \
  struct __attribute__((aligned(A))) N##4 { T x, y, z, w; };                   \
  SASS_INL N##4 make_##N##4(T x, T y, T z, T w) { return {x, y, z, w}; }
SASS_VEC2(float, float, 8)
SASS_VEC3(float, float)
SASS_VEC2(int, int, 8)
SASS_VEC3(int, int)
SASS_VEC4(int, int, 16)
SASS_VEC3(unsigned, uint)
SASS_VEC2(short, short, 4)
SASS_VEC4(short, short, 8)
SASS_VEC2(unsigned short, ushort, 4)
SASS_VEC4(unsigned short, ushort, 8)
SASS_VEC2(char, char, 2)
SASS_VEC4(char, char, 4)
SASS_VEC2(unsigned char, uchar, 2)
SASS_VEC4(unsigned char, uchar, 4)
SASS_VEC2(long long, longlong, 16)
SASS_VEC2(unsigned long long, ulonglong, 16)
#undef SASS_VEC2
#undef SASS_VEC3
#undef SASS_VEC4
SASS_INL uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }

//===----------------------------------------------------------------------===//
// __half / __nv_bfloat16 (native _Float16 / __bf16; packed pairs are
// two-element vectors, so .x/.y, operators and brace initialization work)
//===----------------------------------------------------------------------===//

typedef _Float16 __half;
typedef __bf16 __nv_bfloat16;
typedef _Float16 __half2 __attribute__((ext_vector_type(2)));
typedef __bf16 __nv_bfloat162 __attribute__((ext_vector_type(2)));
typedef __half half;
typedef __half2 half2;
typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

struct __half_raw {
  unsigned short x;
  __half_raw() = default;
  SASS_INL_MEMBER __half_raw(__half h) : x(__builtin_bit_cast(unsigned short, h)) {}
  SASS_INL_MEMBER operator __half() const { return __builtin_bit_cast(__half, x); }
};
struct __half2_raw {
  unsigned short x, y;
  __half2_raw() = default;
  // Through the whole vector: clang's __builtin_bit_cast of an ext-vector
  // element (h.y) reads element 0.
  SASS_INL_MEMBER __half2_raw(__half2 h)
      : x((unsigned short)__builtin_bit_cast(unsigned, h)),
        y((unsigned short)(__builtin_bit_cast(unsigned, h) >> 16)) {}
  SASS_INL_MEMBER operator __half2() const {
    return __builtin_bit_cast(__half2, *this);
  }
};
struct __nv_bfloat16_raw {
  unsigned short x;
  __nv_bfloat16_raw() = default;
  SASS_INL_MEMBER __nv_bfloat16_raw(__nv_bfloat16 h)
      : x(__builtin_bit_cast(unsigned short, h)) {}
  SASS_INL_MEMBER operator __nv_bfloat16() const {
    return __builtin_bit_cast(__nv_bfloat16, x);
  }
};
struct __nv_bfloat162_raw {
  unsigned short x, y;
  __nv_bfloat162_raw() = default;
  SASS_INL_MEMBER __nv_bfloat162_raw(__nv_bfloat162 h)
      : x((unsigned short)__builtin_bit_cast(unsigned, h)),
        y((unsigned short)(__builtin_bit_cast(unsigned, h) >> 16)) {}
  SASS_INL_MEMBER operator __nv_bfloat162() const {
    return __builtin_bit_cast(__nv_bfloat162, *this);
  }
};

// Bit casts.
SASS_INL unsigned short __half_as_ushort(__half h) { return __builtin_bit_cast(unsigned short, h); }
SASS_INL short __half_as_short(__half h) { return __builtin_bit_cast(short, h); }
SASS_INL __half __ushort_as_half(unsigned short x) { return __builtin_bit_cast(__half, x); }
SASS_INL __half __short_as_half(short x) { return __builtin_bit_cast(__half, x); }
SASS_INL unsigned short __bfloat16_as_ushort(__nv_bfloat16 h) { return __builtin_bit_cast(unsigned short, h); }
SASS_INL short __bfloat16_as_short(__nv_bfloat16 h) { return __builtin_bit_cast(short, h); }
SASS_INL __nv_bfloat16 __ushort_as_bfloat16(unsigned short x) { return __builtin_bit_cast(__nv_bfloat16, x); }
SASS_INL __nv_bfloat16 __short_as_bfloat16(short x) { return __builtin_bit_cast(__nv_bfloat16, x); }

// float <-> half (round to nearest even: F2FP / HADD2.F32).
SASS_INL __half __float2half(float f) { return (__half)f; }
SASS_INL __half __float2half_rn(float f) { return (__half)f; }
SASS_INL float __half2float(__half h) { return (float)h; }
SASS_INL __half2 __float2half2_rn(float f) { return __half2{(__half)f, (__half)f}; }
SASS_INL __half2 __floats2half2_rn(float lo, float hi) { return __half2{(__half)lo, (__half)hi}; }
SASS_INL __half2 __float22half2_rn(float2 f) { return __half2{(__half)f.x, (__half)f.y}; }
SASS_INL float2 __half22float2(__half2 h) { return {(float)h.x, (float)h.y}; }
SASS_INL float __low2float(__half2 h) { return (float)h.x; }
SASS_INL float __high2float(__half2 h) { return (float)h.y; }
SASS_INL __nv_bfloat16 __float2bfloat16(float f) { return (__nv_bfloat16)f; }
SASS_INL __nv_bfloat16 __float2bfloat16_rn(float f) { return (__nv_bfloat16)f; }
SASS_INL float __bfloat162float(__nv_bfloat16 h) { return (float)h; }
SASS_INL __nv_bfloat162 __float2bfloat162_rn(float f) {
  return __nv_bfloat162{(__nv_bfloat16)f, (__nv_bfloat16)f};
}
SASS_INL __nv_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
  return __nv_bfloat162{(__nv_bfloat16)lo, (__nv_bfloat16)hi};
}
SASS_INL __nv_bfloat162 __float22bfloat162_rn(float2 f) {
  return __nv_bfloat162{(__nv_bfloat16)f.x, (__nv_bfloat16)f.y};
}
SASS_INL float2 __bfloat1622float2(__nv_bfloat162 h) { return {(float)h.x, (float)h.y}; }
SASS_INL float __low2float(__nv_bfloat162 h) { return (float)h.x; }
SASS_INL float __high2float(__nv_bfloat162 h) { return (float)h.y; }

// Directed roundings of float -> half/bfloat16, from the round-to-nearest
// result: step one ulp toward the requested direction when it overshot.
SASS_INL unsigned short __sass_f2h_dir(float f, unsigned short rn, float back,
                                       int dir /* 0 rz, -1 rd, +1 ru */) {
  if (f != f || back == f)
    return rn;
  bool up = back > f; // the RN result is above the exact value
  bool step = dir == 0 ? (up == (f > 0)) : (dir > 0 ? !up : up);
  if (!step)
    return rn;
  // Moving toward -inf decrements the magnitude of a positive value and
  // increments a negative one (and vice versa).
  bool towardNeg = up; // overshot upward -> step down
  bool neg = rn & 0x8000;
  if ((rn & 0x7fff) == 0) // +-0: the neighbour is the smallest subnormal
    return towardNeg ? 0x8001 : 0x0001;
  return towardNeg != neg ? rn - 1 : rn + 1;
}
SASS_INL __half __float2half_rz(float f) {
  __half r = (__half)f;
  return __ushort_as_half(__sass_f2h_dir(f, __half_as_ushort(r), (float)r, 0));
}
SASS_INL __half __float2half_rd(float f) {
  __half r = (__half)f;
  return __ushort_as_half(__sass_f2h_dir(f, __half_as_ushort(r), (float)r, -1));
}
SASS_INL __half __float2half_ru(float f) {
  __half r = (__half)f;
  return __ushort_as_half(__sass_f2h_dir(f, __half_as_ushort(r), (float)r, 1));
}
SASS_INL __nv_bfloat16 __float2bfloat16_rz(float f) {
  __nv_bfloat16 r = (__nv_bfloat16)f;
  return __ushort_as_bfloat16(__sass_f2h_dir(f, __bfloat16_as_ushort(r), (float)r, 0));
}
SASS_INL __nv_bfloat16 __float2bfloat16_rd(float f) {
  __nv_bfloat16 r = (__nv_bfloat16)f;
  return __ushort_as_bfloat16(__sass_f2h_dir(f, __bfloat16_as_ushort(r), (float)r, -1));
}
SASS_INL __nv_bfloat16 __float2bfloat16_ru(float f) {
  __nv_bfloat16 r = (__nv_bfloat16)f;
  return __ushort_as_bfloat16(__sass_f2h_dir(f, __bfloat16_as_ushort(r), (float)r, 1));
}

// Integer conversions (I2F.F16/I2F.BF16 round to nearest; to-integer via the
// exact f32 widening).
SASS_INL __half __int2half_rn(int i) { return (__half)i; }
SASS_INL __half __uint2half_rn(unsigned i) { return (__half)i; }
SASS_INL __half __short2half_rn(short i) { return (__half)(int)i; }
SASS_INL __half __ushort2half_rn(unsigned short i) { return (__half)(unsigned)i; }
SASS_INL __nv_bfloat16 __int2bfloat16_rn(int i) { return (__nv_bfloat16)i; }
SASS_INL __nv_bfloat16 __uint2bfloat16_rn(unsigned i) { return (__nv_bfloat16)i; }
SASS_INL int __half2int_rz(__half h) { return (int)(float)h; }
SASS_INL unsigned __half2uint_rz(__half h) { return (unsigned)(float)h; }
SASS_INL short __half2short_rz(__half h) {
  float f = (float)h;
  return f != f ? 0 : (short)(f >= 32767.f ? 32767 : f <= -32768.f ? -32768 : (int)f);
}
SASS_INL int __half2int_rn(__half h) { return __float2int_rn((float)h); }
SASS_INL int __half2int_rd(__half h) { return __float2int_rd((float)h); }
SASS_INL int __half2int_ru(__half h) { return __float2int_ru((float)h); }
SASS_INL int __bfloat162int_rz(__nv_bfloat16 h) { return (int)(float)h; }
SASS_INL int __bfloat162int_rn(__nv_bfloat16 h) { return __float2int_rn((float)h); }
SASS_INL unsigned __bfloat162uint_rz(__nv_bfloat16 h) { return (unsigned)(float)h; }

// Packing.
#define SASS_HALF_PACK(H, H2, PFX)                                             \
  SASS_INL H2 make_##PFX##2(H lo, H hi) { return H2{lo, hi}; }                 \
  SASS_INL H2 __halves2##PFX##2(H lo, H hi) { return H2{lo, hi}; }             \
  SASS_INL H2 __##PFX##2##PFX##2(H h) { return H2{h, h}; }                     \
  SASS_INL H2 __lows2##PFX##2(H2 a, H2 b) { return H2{a.x, b.x}; }             \
  SASS_INL H2 __highs2##PFX##2(H2 a, H2 b) { return H2{a.y, b.y}; }            \
  SASS_INL H2 __low2##PFX##2(H2 a) { return H2{a.x, a.x}; }                    \
  SASS_INL H2 __high2##PFX##2(H2 a) { return H2{a.y, a.y}; }                   \
  SASS_INL H2 __lowhigh2highlow(H2 a) { return H2{a.y, a.x}; }                 \
  SASS_INL H __low2##PFX(H2 a) { return a.x; }                                 \
  SASS_INL H __high2##PFX(H2 a) { return a.y; }
SASS_HALF_PACK(__half, __half2, half)
SASS_HALF_PACK(__nv_bfloat16, __nv_bfloat162, bfloat16)
#undef SASS_HALF_PACK

// Arithmetic: one rounding per operation, like add/mul/fma.rn.{f16,bf16}{,x2}
// (the backend selects HADD2/HMUL2/HFMA2[.BF16_V2]); like nvcc, a separate
// multiply and add may still contract into an FMA. The *_rn forms do not.
#define SASS_HALF_ARITH(H)                                                     \
  SASS_INL H __hadd(H a, H b) { return a + b; }                                \
  SASS_INL H __hsub(H a, H b) { return a - b; }                                \
  SASS_INL H __hmul(H a, H b) { return a * b; }                                \
  SASS_INL H __hfma(H a, H b, H c) { return __builtin_elementwise_fma(a, b, c); } \
  SASS_INL H __hadd_rn(H a, H b) {                                             \
    _Pragma("clang fp contract(off)") return a + b;                            \
  }                                                                            \
  SASS_INL H __hsub_rn(H a, H b) {                                             \
    _Pragma("clang fp contract(off)") return a - b;                            \
  }                                                                            \
  SASS_INL H __hmul_rn(H a, H b) {                                             \
    _Pragma("clang fp contract(off)") return a * b;                            \
  }                                                                            \
  SASS_INL H __hneg(H a) { return -a; }                                        \
  SASS_INL H __habs(H a) { return __builtin_elementwise_abs(a); }              \
  SASS_INL H __hmax(H a, H b) { return __builtin_elementwise_maxnum(a, b); }   \
  SASS_INL H __hmin(H a, H b) { return __builtin_elementwise_minnum(a, b); }   \
  SASS_INL H __hdiv(H a, H b) { return (H)((float)a / (float)b); }             \
  SASS_INL H __hfma_relu(H a, H b, H c) {                                      \
    H r = __builtin_elementwise_fma(a, b, c);                                  \
    return r != r ? r : (r < (H)0 ? (H)0 : r);                                 \
  }                                                                            \
  /* fma.rn.sat: the rounded result clamped to [0, 1], NaN -> +0 */            \
  SASS_INL H __hfma_sat(H a, H b, H c) {                                       \
    return __hmin(__hmax(__builtin_elementwise_fma(a, b, c), (H)0), (H)1);     \
  }                                                                            \
  SASS_INL bool __heq(H a, H b) { return a == b; }                             \
  SASS_INL bool __hne(H a, H b) { return a != b && a == a && b == b; }         \
  SASS_INL bool __hlt(H a, H b) { return a < b; }                              \
  SASS_INL bool __hle(H a, H b) { return a <= b; }                             \
  SASS_INL bool __hgt(H a, H b) { return a > b; }                              \
  SASS_INL bool __hge(H a, H b) { return a >= b; }                             \
  SASS_INL bool __hequ(H a, H b) { return !(a != b); }                         \
  SASS_INL bool __hneu(H a, H b) { return a != b; }                            \
  SASS_INL bool __hltu(H a, H b) { return !(a >= b); }                         \
  SASS_INL bool __hleu(H a, H b) { return !(a > b); }                          \
  SASS_INL bool __hgtu(H a, H b) { return !(a <= b); }                         \
  SASS_INL bool __hgeu(H a, H b) { return !(a < b); }                          \
  SASS_INL bool __hisnan(H a) { return a != a; }                               \
  SASS_INL int __hisinf(H a) {                                                 \
    float f = (float)a;                                                        \
    return f == __builtin_inff() ? 1 : f == -__builtin_inff() ? -1 : 0;        \
  }
SASS_HALF_ARITH(__half)
SASS_HALF_ARITH(__nv_bfloat16)
#undef SASS_HALF_ARITH
template <typename H2, typename H> SASS_INL H2 __sass_mask2(bool lo, bool hi) {
  return H2{lo ? (H)1 : (H)0, hi ? (H)1 : (H)0};
}
#define SASS_HALF2_ARITH(H2, H)                                                \
  SASS_INL H2 __hadd2(H2 a, H2 b) { return a + b; }                            \
  SASS_INL H2 __hsub2(H2 a, H2 b) { return a - b; }                            \
  SASS_INL H2 __hmul2(H2 a, H2 b) { return a * b; }                            \
  SASS_INL H2 __hfma2(H2 a, H2 b, H2 c) { return __builtin_elementwise_fma(a, b, c); } \
  SASS_INL H2 __hadd2_rn(H2 a, H2 b) {                                         \
    _Pragma("clang fp contract(off)") return a + b;                            \
  }                                                                            \
  SASS_INL H2 __hmul2_rn(H2 a, H2 b) {                                         \
    _Pragma("clang fp contract(off)") return a * b;                            \
  }                                                                            \
  SASS_INL H2 __hneg2(H2 a) { return -a; }                                     \
  SASS_INL H2 __habs2(H2 a) { return __builtin_elementwise_abs(a); }           \
  SASS_INL H2 __hmax2(H2 a, H2 b) { return __builtin_elementwise_maxnum(a, b); } \
  SASS_INL H2 __hmin2(H2 a, H2 b) { return __builtin_elementwise_minnum(a, b); } \
  SASS_INL H2 __h2div(H2 a, H2 b) { return H2{__hdiv(a.x, b.x), __hdiv(a.y, b.y)}; } \
  SASS_INL H2 __hfma2_relu(H2 a, H2 b, H2 c) {                                 \
    return H2{__hfma_relu(a.x, b.x, c.x), __hfma_relu(a.y, b.y, c.y)};         \
  }                                                                            \
  SASS_INL H2 __hfma2_sat(H2 a, H2 b, H2 c) {                                  \
    return __hmin2(__hmax2(__builtin_elementwise_fma(a, b, c), H2{0, 0}), H2{1, 1}); \
  }                                                                            \
  SASS_INL H2 __heq2(H2 a, H2 b) { return __sass_mask2<H2, H>(a.x == b.x, a.y == b.y); } \
  SASS_INL H2 __hne2(H2 a, H2 b) { return __sass_mask2<H2, H>(__hne(a.x, b.x), __hne(a.y, b.y)); } \
  SASS_INL H2 __hlt2(H2 a, H2 b) { return __sass_mask2<H2, H>(a.x < b.x, a.y < b.y); } \
  SASS_INL H2 __hle2(H2 a, H2 b) { return __sass_mask2<H2, H>(a.x <= b.x, a.y <= b.y); } \
  SASS_INL H2 __hgt2(H2 a, H2 b) { return __sass_mask2<H2, H>(a.x > b.x, a.y > b.y); } \
  SASS_INL H2 __hge2(H2 a, H2 b) { return __sass_mask2<H2, H>(a.x >= b.x, a.y >= b.y); } \
  SASS_INL H2 __hisnan2(H2 a) { return __sass_mask2<H2, H>(a.x != a.x, a.y != a.y); } \
  SASS_INL bool __hbeq2(H2 a, H2 b) { return a.x == b.x && a.y == b.y; }       \
  SASS_INL bool __hbne2(H2 a, H2 b) { return __hne(a.x, b.x) && __hne(a.y, b.y); } \
  SASS_INL bool __hblt2(H2 a, H2 b) { return a.x < b.x && a.y < b.y; }         \
  SASS_INL bool __hble2(H2 a, H2 b) { return a.x <= b.x && a.y <= b.y; }       \
  SASS_INL bool __hbgt2(H2 a, H2 b) { return a.x > b.x && a.y > b.y; }         \
  SASS_INL bool __hbge2(H2 a, H2 b) { return a.x >= b.x && a.y >= b.y; }
SASS_HALF2_ARITH(__half2, __half)
SASS_HALF2_ARITH(__nv_bfloat162, __nv_bfloat16)
#undef SASS_HALF2_ARITH

// Elementary functions on halves: computed in f32 and rounded once.
#define SASS_HALF_FN(H, N, F)                                                  \
  SASS_INL H N(H a) { return (H)(F((float)a)); }
#define SASS_HALF2_FN(H2, H, N, F)                                             \
  SASS_INL H2 N(H2 a) { return H2{(H)(F((float)a.x)), (H)(F((float)a.y))}; }
#define SASS_HALF_FNS(H, H2)                                                   \
  SASS_HALF_FN(H, hsqrt, sqrtf) SASS_HALF_FN(H, hrsqrt, rsqrtf)                \
  SASS_HALF_FN(H, hexp, expf) SASS_HALF_FN(H, hexp2, exp2f)                    \
  SASS_HALF_FN(H, hexp10, exp10f) SASS_HALF_FN(H, hlog, logf)                  \
  SASS_HALF_FN(H, hlog2, log2f) SASS_HALF_FN(H, hlog10, log10f)                \
  SASS_HALF_FN(H, hsin, sinf) SASS_HALF_FN(H, hcos, cosf)                      \
  SASS_HALF_FN(H, hrcp, 1.0f /) SASS_HALF_FN(H, htrunc, truncf)                \
  SASS_HALF_FN(H, hfloor, floorf) SASS_HALF_FN(H, hceil, ceilf)                \
  SASS_HALF_FN(H, hrint, rintf) SASS_HALF_FN(H, htanh, tanhf)                  \
  SASS_HALF2_FN(H2, H, h2sqrt, sqrtf) SASS_HALF2_FN(H2, H, h2rsqrt, rsqrtf)    \
  SASS_HALF2_FN(H2, H, h2exp, expf) SASS_HALF2_FN(H2, H, h2exp2, exp2f)        \
  SASS_HALF2_FN(H2, H, h2log, logf) SASS_HALF2_FN(H2, H, h2log2, log2f)        \
  SASS_HALF2_FN(H2, H, h2sin, sinf) SASS_HALF2_FN(H2, H, h2cos, cosf)          \
  SASS_HALF2_FN(H2, H, h2rcp, 1.0f /) SASS_HALF2_FN(H2, H, h2trunc, truncf)    \
  SASS_HALF2_FN(H2, H, h2floor, floorf) SASS_HALF2_FN(H2, H, h2ceil, ceilf)    \
  SASS_HALF2_FN(H2, H, h2rint, rintf) SASS_HALF2_FN(H2, H, h2tanh, tanhf)
SASS_HALF_FNS(__half, __half2)
SASS_HALF_FNS(__nv_bfloat16, __nv_bfloat162)
#undef SASS_HALF_FNS
#undef SASS_HALF_FN
#undef SASS_HALF2_FN

//===----------------------------------------------------------------------===//
// FP8 (E4M3 / E5M2) conversions: F2FP.SATFINITE.E4M3/E5M2 and
// F2FP.F16.E4M3/E5M2.UNPACK_B. Byte 0 of a pair is the first (low) element.
//===----------------------------------------------------------------------===//

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short __nv_fp8x2_storage_t;
typedef unsigned int __nv_fp8x4_storage_t;
enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE };
enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 };

// Without saturation, out-of-range values become NaN (E4M3, which has no
// infinity) or infinity (E5M2); the hardware conversion saturates, so fix up
// what lies beyond the largest finite value after rounding.
SASS_INL unsigned short __sass_fp8_nosat(unsigned short sat, float lo, float hi,
                                         __nv_fp8_interpretation_t t) {
  float lim = t == __NV_E4M3 ? 464.0f : 61440.0f; // first value rounding past max
  unsigned short inf = t == __NV_E4M3 ? 0x7f : 0x7c;
  unsigned short r = sat;
  if (__builtin_fabsf(lo) >= lim && lo == lo && !(t == __NV_E4M3 && __builtin_fabsf(lo) == lim))
    r = (r & 0xff00) | inf | (lo < 0 ? 0x80 : 0);
  if (__builtin_fabsf(hi) >= lim && hi == hi && !(t == __NV_E4M3 && __builtin_fabsf(hi) == lim))
    r = (r & 0x00ff) | (unsigned short)((inf | (hi < 0 ? 0x80 : 0)) << 8);
  return r;
}
SASS_INL __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 v, __nv_saturation_t s,
                                                       __nv_fp8_interpretation_t t) {
  unsigned short r = t == __NV_E4M3 ? (unsigned short)__nvvm_ff_to_e4m3x2_rn(v.y, v.x)
                                    : (unsigned short)__nvvm_ff_to_e5m2x2_rn(v.y, v.x);
  return s == __NV_SATFINITE ? r : __sass_fp8_nosat(r, v.x, v.y, t);
}
SASS_INL __nv_fp8_storage_t __nv_cvt_float_to_fp8(float v, __nv_saturation_t s,
                                                  __nv_fp8_interpretation_t t) {
  return (__nv_fp8_storage_t)__nv_cvt_float2_to_fp8x2(make_float2(v, 0.0f), s, t);
}
SASS_INL __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw v, __nv_saturation_t s,
                                                         __nv_fp8_interpretation_t t) {
  __half2 h = v;
  unsigned short r = t == __NV_E4M3 ? (unsigned short)__nvvm_f16x2_to_e4m3x2_rn(h)
                                    : (unsigned short)__nvvm_f16x2_to_e5m2x2_rn(h);
  return s == __NV_SATFINITE ? r : __sass_fp8_nosat(r, (float)h.x, (float)h.y, t);
}
SASS_INL __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw v, __nv_saturation_t s,
                                                    __nv_fp8_interpretation_t t) {
  return (__nv_fp8_storage_t)__nv_cvt_float2_to_fp8x2(make_float2((float)(__half)v, 0.0f), s, t);
}
SASS_INL __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw v, __nv_saturation_t s,
                                                             __nv_fp8_interpretation_t t) {
  __nv_bfloat162 b = v;
  return __nv_cvt_float2_to_fp8x2(make_float2((float)b.x, (float)b.y), s, t);
}
SASS_INL __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw v, __nv_saturation_t s,
                                                        __nv_fp8_interpretation_t t) {
  return __nv_cvt_float_to_fp8((float)(__nv_bfloat16)v, s, t);
}
SASS_INL __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t v,
                                                __nv_fp8_interpretation_t t) {
  return t == __NV_E4M3 ? __half2_raw(__nvvm_e4m3x2_to_f16x2_rn((short)v))
                        : __half2_raw(__nvvm_e5m2x2_to_f16x2_rn((short)v));
}
SASS_INL __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t v, __nv_fp8_interpretation_t t) {
  __half2 h = __nv_cvt_fp8x2_to_halfraw2(v, t);
  return __half_raw(h.x);
}

struct __nv_fp8_e4m3 {
  __nv_fp8_storage_t __x;
  __nv_fp8_e4m3() = default;
  SASS_INL_MEMBER explicit __nv_fp8_e4m3(float f)
      : __x(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3)) {}
  SASS_INL_MEMBER explicit __nv_fp8_e4m3(__half h)
      : __x(__nv_cvt_halfraw_to_fp8(__half_raw(h), __NV_SATFINITE, __NV_E4M3)) {}
  SASS_INL_MEMBER explicit operator __half() const {
    return __half_raw(__nv_cvt_fp8_to_halfraw(__x, __NV_E4M3));
  }
  SASS_INL_MEMBER explicit operator float() const { return (float)(__half)(*this); }
};
struct __nv_fp8_e5m2 {
  __nv_fp8_storage_t __x;
  __nv_fp8_e5m2() = default;
  SASS_INL_MEMBER explicit __nv_fp8_e5m2(float f)
      : __x(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E5M2)) {}
  SASS_INL_MEMBER explicit __nv_fp8_e5m2(__half h)
      : __x(__nv_cvt_halfraw_to_fp8(__half_raw(h), __NV_SATFINITE, __NV_E5M2)) {}
  SASS_INL_MEMBER explicit operator __half() const {
    return __half_raw(__nv_cvt_fp8_to_halfraw(__x, __NV_E5M2));
  }
  SASS_INL_MEMBER explicit operator float() const { return (float)(__half)(*this); }
};
struct __nv_fp8x2_e4m3 {
  __nv_fp8x2_storage_t __x;
  __nv_fp8x2_e4m3() = default;
  SASS_INL_MEMBER explicit __nv_fp8x2_e4m3(float2 f)
      : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3)) {}
  SASS_INL_MEMBER explicit operator __half2() const {
    return __nv_cvt_fp8x2_to_halfraw2(__x, __NV_E4M3);
  }
  SASS_INL_MEMBER explicit operator float2() const {
    __half2 h = (__half2)(*this);
    return __half22float2(h);
  }
};
struct __nv_fp8x2_e5m2 {
  __nv_fp8x2_storage_t __x;
  __nv_fp8x2_e5m2() = default;
  SASS_INL_MEMBER explicit __nv_fp8x2_e5m2(float2 f)
      : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E5M2)) {}
  SASS_INL_MEMBER explicit operator __half2() const {
    return __nv_cvt_fp8x2_to_halfraw2(__x, __NV_E5M2);
  }
  SASS_INL_MEMBER explicit operator float2() const {
    __half2 h = (__half2)(*this);
    return __half22float2(h);
  }
};
struct __nv_fp8x4_e4m3 {
  __nv_fp8x4_storage_t __x;
  __nv_fp8x4_e4m3() = default;
  SASS_INL_MEMBER explicit __nv_fp8x4_e4m3(float4 f)
      : __x(__nv_cvt_float2_to_fp8x2(make_float2(f.x, f.y), __NV_SATFINITE, __NV_E4M3) |
            (unsigned)__nv_cvt_float2_to_fp8x2(make_float2(f.z, f.w), __NV_SATFINITE, __NV_E4M3)
                << 16) {}
};

//===----------------------------------------------------------------------===//
// Atomics (relaxed, device scope unless the name says otherwise; the
// backend selects ATOMG/ATOM/ATOMS, or REDG/RED when the result is unused)
//===----------------------------------------------------------------------===//

#define SASS_ATOMIC_SCOPED(SFX, SCOPE)                                         \
  template <typename T> SASS_INL T atomicAdd##SFX(T *p, T v) {                 \
    return __scoped_atomic_fetch_add(p, v, __ATOMIC_RELAXED, SCOPE);           \
  }                                                                            \
  template <typename T> SASS_INL T atomicSub##SFX(T *p, T v) {                 \
    return __scoped_atomic_fetch_sub(p, v, __ATOMIC_RELAXED, SCOPE);           \
  }                                                                            \
  template <typename T> SASS_INL T atomicExch##SFX(T *p, T v) {                \
    return __scoped_atomic_exchange_n(p, v, __ATOMIC_RELAXED, SCOPE);          \
  }                                                                            \
  template <typename T> SASS_INL T atomicMin##SFX(T *p, T v) {                 \
    return __scoped_atomic_fetch_min(p, v, __ATOMIC_RELAXED, SCOPE);           \
  }                                                                            \
  template <typename T> SASS_INL T atomicMax##SFX(T *p, T v) {                 \
    return __scoped_atomic_fetch_max(p, v, __ATOMIC_RELAXED, SCOPE);           \
  }                                                                            \
  template <typename T> SASS_INL T atomicAnd##SFX(T *p, T v) {                 \
    return __scoped_atomic_fetch_and(p, v, __ATOMIC_RELAXED, SCOPE);           \
  }                                                                            \
  template <typename T> SASS_INL T atomicOr##SFX(T *p, T v) {                  \
    return __scoped_atomic_fetch_or(p, v, __ATOMIC_RELAXED, SCOPE);            \
  }                                                                            \
  template <typename T> SASS_INL T atomicXor##SFX(T *p, T v) {                 \
    return __scoped_atomic_fetch_xor(p, v, __ATOMIC_RELAXED, SCOPE);           \
  }                                                                            \
  template <typename T> SASS_INL T atomicCAS##SFX(T *p, T cmp, T val) {        \
    __scoped_atomic_compare_exchange_n(p, &cmp, val, false, __ATOMIC_RELAXED,  \
                                       __ATOMIC_RELAXED, SCOPE);               \
    return cmp;                                                                \
  }
SASS_ATOMIC_SCOPED(, __MEMORY_SCOPE_DEVICE)
SASS_ATOMIC_SCOPED(_block, __MEMORY_SCOPE_WRKGRP)
SASS_ATOMIC_SCOPED(_system, __MEMORY_SCOPE_SYSTEM)
#undef SASS_ATOMIC_SCOPED
// Overload resolution for literals: atomicAdd(p, 1) with unsigned *p.
SASS_INL unsigned atomicAdd(unsigned *p, int v) { return atomicAdd<unsigned>(p, (unsigned)v); }
SASS_INL unsigned long long atomicAdd(unsigned long long *p, int v) {
  return atomicAdd<unsigned long long>(p, (unsigned long long)v);
}
SASS_INL unsigned atomicInc(unsigned *p, unsigned v) { return __nvvm_atom_inc_gen_ui(p, v); }
SASS_INL unsigned atomicDec(unsigned *p, unsigned v) { return __nvvm_atom_dec_gen_ui(p, v); }
// Halves: compare-and-swap on the containing word (as nvcc does for the
// scalar forms).
SASS_INL __half2 atomicAdd(__half2 *p, __half2 v) {
  unsigned *w = (unsigned *)p;
  unsigned old = *w, assumed;
  do {
    assumed = old;
    old = atomicCAS(w, assumed, __builtin_bit_cast(unsigned, __builtin_bit_cast(__half2, assumed) + v));
  } while (old != assumed);
  return __builtin_bit_cast(__half2, old);
}
SASS_INL __nv_bfloat162 atomicAdd(__nv_bfloat162 *p, __nv_bfloat162 v) {
  unsigned *w = (unsigned *)p;
  unsigned old = *w, assumed;
  do {
    assumed = old;
    old = atomicCAS(w, assumed,
                    __builtin_bit_cast(unsigned, __builtin_bit_cast(__nv_bfloat162, assumed) + v));
  } while (old != assumed);
  return __builtin_bit_cast(__nv_bfloat162, old);
}
SASS_INL __half atomicAdd(__half *p, __half v) {
  return __scoped_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}
SASS_INL __nv_bfloat16 atomicAdd(__nv_bfloat16 *p, __nv_bfloat16 v) {
  return __scoped_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}
// sm_90 vector adds, atomic per element (ATOMG/REDG .F32x2/.F32x4, as nvcc).
SASS_INL float2 atomicAdd(float2 *p, float2 v) {
  float2 r;
  asm volatile("atom.add.v2.f32 {%0,%1}, [%2], {%3,%4};"
               : "=f"(r.x), "=f"(r.y) : "l"(p), "f"(v.x), "f"(v.y) : "memory");
  return r;
}
SASS_INL float4 atomicAdd(float4 *p, float4 v) {
  float4 r;
  asm volatile("atom.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};"
               : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
               : "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
  return r;
}

// Fences: fence.sc.{cta,gpu,sys} (MEMBAR.SC + ERRBAR + L1 invalidation).
SASS_INL void __threadfence_block() { __nvvm_membar_cta(); }
SASS_INL void __threadfence() { __nvvm_membar_gl(); }
SASS_INL void __threadfence_system() { __nvvm_membar_sys(); }

//===----------------------------------------------------------------------===//
// Cache-hinted and coherent loads/stores
//===----------------------------------------------------------------------===//

// __ldg: read-only path (LDG.E.CONSTANT); the others use the default path,
// which is what ptxas emits for ld.global.cg/.cs/.lu/.cv on sm_90 except
// .cv (volatile: LDG.E.STRONG.SYS).
template <typename T> SASS_INL T __ldg(const T *p) {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 ||
                    sizeof(T) == 16,
                "__ldg: unsupported size");
  typedef int __i2 __attribute__((ext_vector_type(2)));
  typedef int __i4 __attribute__((ext_vector_type(4)));
  if constexpr (sizeof(T) == 1)
    return __builtin_bit_cast(T, __nvvm_ldg_c((const char *)p));
  else if constexpr (sizeof(T) == 2)
    return __builtin_bit_cast(T, __nvvm_ldg_s((const short *)p));
  else if constexpr (sizeof(T) == 4)
    return __builtin_bit_cast(T, __nvvm_ldg_i((const int *)p));
  else if constexpr (sizeof(T) == 8)
    return __builtin_bit_cast(T, __nvvm_ldg_i2((const __i2 *)p));
  else
    return __builtin_bit_cast(T, __nvvm_ldg_i4((const __i4 *)p));
}
template <typename T> SASS_INL T __ldcg(const T *p) { return *p; }
template <typename T> SASS_INL T __ldca(const T *p) { return *p; }
template <typename T> SASS_INL T __ldcs(const T *p) { return *p; }
template <typename T> SASS_INL T __ldlu(const T *p) { return *p; }
template <typename T> SASS_INL T __ldcv(const T *p) { return *(const volatile T *)p; }
template <typename T> SASS_INL void __stcg(T *p, T v) { *p = v; }
template <typename T> SASS_INL void __stcs(T *p, T v) { *p = v; }
template <typename T> SASS_INL void __stwb(T *p, T v) { *p = v; }
template <typename T> SASS_INL void __stwt(T *p, T v) { *(volatile T *)p = v; }

//===----------------------------------------------------------------------===//
// Warp intrinsics
//===----------------------------------------------------------------------===//

SASS_INL void __syncwarp(unsigned mask = 0xffffffffu) { __nvvm_bar_warp_sync(mask); }
SASS_INL unsigned __activemask() { return __nvvm_activemask(); }
SASS_INL unsigned __ballot_sync(unsigned mask, int p) { return __nvvm_vote_ballot_sync(mask, p); }
SASS_INL int __any_sync(unsigned mask, int p) { return __nvvm_vote_any_sync(mask, p); }
SASS_INL int __all_sync(unsigned mask, int p) { return __nvvm_vote_all_sync(mask, p); }
SASS_INL int __uni_sync(unsigned mask, int p) { return __nvvm_vote_uni_sync(mask, p); }
SASS_INL unsigned __match_any_sync(unsigned mask, unsigned v) { return __nvvm_match_any_sync_i32(mask, v); }
SASS_INL unsigned __match_any_sync(unsigned mask, int v) { return __nvvm_match_any_sync_i32(mask, v); }
SASS_INL unsigned __match_any_sync(unsigned mask, unsigned long long v) {
  return __nvvm_match_any_sync_i64(mask, v);
}
SASS_INL unsigned __match_any_sync(unsigned mask, long long v) { return __nvvm_match_any_sync_i64(mask, v); }
SASS_INL unsigned __match_all_sync(unsigned mask, unsigned v, int *pred) {
  return __nvvm_match_all_sync_i32p(mask, v, pred);
}
SASS_INL unsigned __match_all_sync(unsigned mask, int v, int *pred) {
  return __nvvm_match_all_sync_i32p(mask, v, pred);
}
SASS_INL unsigned __match_all_sync(unsigned mask, unsigned long long v, int *pred) {
  return __nvvm_match_all_sync_i64p(mask, v, pred);
}
SASS_INL unsigned __match_all_sync(unsigned mask, long long v, int *pred) {
  return __nvvm_match_all_sync_i64p(mask, v, pred);
}
SASS_INL unsigned __reduce_add_sync(unsigned mask, unsigned v) { return __nvvm_redux_sync_add(v, mask); }
SASS_INL int __reduce_add_sync(unsigned mask, int v) { return __nvvm_redux_sync_add(v, mask); }
SASS_INL unsigned __reduce_min_sync(unsigned mask, unsigned v) { return __nvvm_redux_sync_umin(v, mask); }
SASS_INL int __reduce_min_sync(unsigned mask, int v) { return __nvvm_redux_sync_min(v, mask); }
SASS_INL unsigned __reduce_max_sync(unsigned mask, unsigned v) { return __nvvm_redux_sync_umax(v, mask); }
SASS_INL int __reduce_max_sync(unsigned mask, int v) { return __nvvm_redux_sync_max(v, mask); }
SASS_INL unsigned __reduce_and_sync(unsigned mask, unsigned v) { return __nvvm_redux_sync_and(v, mask); }
SASS_INL unsigned __reduce_or_sync(unsigned mask, unsigned v) { return __nvvm_redux_sync_or(v, mask); }
SASS_INL unsigned __reduce_xor_sync(unsigned mask, unsigned v) { return __nvvm_redux_sync_xor(v, mask); }
SASS_INL unsigned __lanemask_lt() { return __nvvm_read_ptx_sreg_lanemask_lt(); }
SASS_INL unsigned __lanemask_le() { return __nvvm_read_ptx_sreg_lanemask_le(); }
SASS_INL unsigned __lanemask_gt() { return __nvvm_read_ptx_sreg_lanemask_gt(); }
SASS_INL unsigned __lanemask_ge() { return __nvvm_read_ptx_sreg_lanemask_ge(); }
SASS_INL unsigned __laneid() { return __nvvm_read_ptx_sreg_laneid(); }

// Shuffles: c = ((32 - width) << 8) | clamp, as nvcc encodes it.
#define SASS_SHFL(KIND, CLAMP)                                                 \
  SASS_INL int __shfl_##KIND##_i(unsigned mask, int v, unsigned l, int w) {    \
    return __nvvm_shfl_sync_##KIND##_i32(mask, v, l, ((32 - w) << 8) | (CLAMP)); \
  }                                                                            \
  SASS_INL float __shfl_##KIND##_f(unsigned mask, float v, unsigned l, int w) { \
    return __nvvm_shfl_sync_##KIND##_f32(mask, v, l, ((32 - w) << 8) | (CLAMP)); \
  }
SASS_SHFL(idx, 0x1f)
SASS_SHFL(up, 0)
SASS_SHFL(down, 0x1f)
SASS_SHFL(bfly, 0x1f)
#undef SASS_SHFL
#define SASS_SHFL_API(NAME, KIND, LANE_T)                                      \
  SASS_INL int NAME(unsigned m, int v, LANE_T l, int w = 32) { return __shfl_##KIND##_i(m, v, l, w); } \
  SASS_INL unsigned NAME(unsigned m, unsigned v, LANE_T l, int w = 32) {       \
    return (unsigned)__shfl_##KIND##_i(m, (int)v, l, w);                       \
  }                                                                            \
  SASS_INL float NAME(unsigned m, float v, LANE_T l, int w = 32) { return __shfl_##KIND##_f(m, v, l, w); } \
  SASS_INL long long NAME(unsigned m, long long v, LANE_T l, int w = 32) {     \
    int lo = __shfl_##KIND##_i(m, (int)v, l, w);                               \
    int hi = __shfl_##KIND##_i(m, (int)(v >> 32), l, w);                       \
    return (long long)(((unsigned long long)(unsigned)hi << 32) | (unsigned)lo); \
  }                                                                            \
  SASS_INL unsigned long long NAME(unsigned m, unsigned long long v, LANE_T l, int w = 32) { \
    return (unsigned long long)NAME(m, (long long)v, l, w);                    \
  }                                                                            \
  SASS_INL __half2 NAME(unsigned m, __half2 v, LANE_T l, int w = 32) {         \
    return __builtin_bit_cast(__half2, __shfl_##KIND##_i(m, __builtin_bit_cast(int, v), l, w)); \
  }                                                                            \
  SASS_INL __nv_bfloat162 NAME(unsigned m, __nv_bfloat162 v, LANE_T l, int w = 32) { \
    return __builtin_bit_cast(__nv_bfloat162,                                  \
                              __shfl_##KIND##_i(m, __builtin_bit_cast(int, v), l, w)); \
  }                                                                            \
  SASS_INL __half NAME(unsigned m, __half v, LANE_T l, int w = 32) {           \
    return __ushort_as_half((unsigned short)__shfl_##KIND##_i(m, __half_as_ushort(v), l, w)); \
  }                                                                            \
  SASS_INL __nv_bfloat16 NAME(unsigned m, __nv_bfloat16 v, LANE_T l, int w = 32) { \
    return __ushort_as_bfloat16(                                               \
        (unsigned short)__shfl_##KIND##_i(m, __bfloat16_as_ushort(v), l, w));  \
  }
SASS_SHFL_API(__shfl_sync, idx, int)
SASS_SHFL_API(__shfl_up_sync, up, unsigned)
SASS_SHFL_API(__shfl_down_sync, down, unsigned)
#undef SASS_SHFL_API
SASS_INL int __shfl_xor_sync(unsigned m, int v, int l, int w = 32) { return __shfl_bfly_i(m, v, l, w); }
SASS_INL float __shfl_xor_sync(unsigned m, float v, int l, int w = 32) { return __shfl_bfly_f(m, v, l, w); }
SASS_INL unsigned __shfl_xor_sync(unsigned m, unsigned v, int l, int w = 32) {
  return (unsigned)__shfl_bfly_i(m, (int)v, l, w);
}
SASS_INL long long __shfl_xor_sync(unsigned m, long long v, int l, int w = 32) {
  int lo = __shfl_bfly_i(m, (int)v, l, w);
  int hi = __shfl_bfly_i(m, (int)(v >> 32), l, w);
  return (long long)(((unsigned long long)(unsigned)hi << 32) | (unsigned)lo);
}
SASS_INL unsigned long long __shfl_xor_sync(unsigned m, unsigned long long v, int l, int w = 32) {
  return (unsigned long long)__shfl_xor_sync(m, (long long)v, l, w);
}
SASS_INL __half2 __shfl_xor_sync(unsigned m, __half2 v, int l, int w = 32) {
  return __builtin_bit_cast(__half2, __shfl_bfly_i(m, __builtin_bit_cast(int, v), l, w));
}
SASS_INL __nv_bfloat162 __shfl_xor_sync(unsigned m, __nv_bfloat162 v, int l, int w = 32) {
  return __builtin_bit_cast(__nv_bfloat162, __shfl_bfly_i(m, __builtin_bit_cast(int, v), l, w));
}
SASS_INL __half __shfl_xor_sync(unsigned m, __half v, int l, int w = 32) {
  return __ushort_as_half((unsigned short)__shfl_bfly_i(m, __half_as_ushort(v), l, w));
}
SASS_INL __nv_bfloat16 __shfl_xor_sync(unsigned m, __nv_bfloat16 v, int l, int w = 32) {
  return __ushort_as_bfloat16((unsigned short)__shfl_bfly_i(m, __bfloat16_as_ushort(v), l, w));
}

//===----------------------------------------------------------------------===//
// Miscellaneous
//===----------------------------------------------------------------------===//

SASS_INL void __nanosleep(unsigned ns) { __nvvm_nanosleep(ns); }
SASS_INL void __trap() { __builtin_trap(); }
SASS_INL void __brkpt() { __builtin_trap(); }
SASS_INL unsigned long long clock64() { return __nvvm_read_ptx_sreg_clock64(); }
SASS_INL long long clock() { return (long long)__nvvm_read_ptx_sreg_clock(); }
SASS_INL unsigned long long __globaltimer() { return __nvvm_read_ptx_sreg_globaltimer(); }
