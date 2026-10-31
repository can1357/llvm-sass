// Minimal CUDA prelude for `clang -x cuda --cuda-device-only -nocudainc`:
// kernel attributes and the threadIdx/blockIdx/blockDim/gridDim builtins.
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __attribute__((always_inline)) inline
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __grid_constant__ __attribute__((grid_constant))
#define __align__(n) __attribute__((aligned(n)))
#define SASS_CLANG 1
typedef __SIZE_TYPE__ size_t;
#define INFINITY __builtin_inff()
#include <__clang_cuda_builtin_vars.h>

#define SASS_INL static __device__ __forceinline__
SASS_INL unsigned __umulhi(unsigned a, unsigned b) {
  return (unsigned)(((unsigned long long)a * b) >> 32);
}
SASS_INL int __mulhi(int a, int b) { return (int)(((long long)a * b) >> 32); }
SASS_INL unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) {
  return __builtin_elementwise_fshl(hi, lo, s & 31);
}
SASS_INL int min(int a, int b) { return a < b ? a : b; }
SASS_INL int max(int a, int b) { return a > b ? a : b; }
SASS_INL unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
SASS_INL unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
SASS_INL float fabsf(float x) { return __builtin_fabsf(x); }
SASS_INL float fminf(float a, float b) { return __builtin_fminf(a, b); }
SASS_INL float fmaxf(float a, float b) { return __builtin_fmaxf(a, b); }
SASS_INL int __float_as_int(float x) { return __builtin_bit_cast(int, x); }
SASS_INL float __int_as_float(int x) { return __builtin_bit_cast(float, x); }

struct __attribute__((aligned(16))) float4 {
  float x, y, z, w;
};
SASS_INL float4 make_float4(float x, float y, float z, float w) {
  return {x, y, z, w};
}
struct __attribute__((aligned(16))) uint4 {
  unsigned x, y, z, w;
};
struct __attribute__((aligned(8))) uint2 {
  unsigned x, y;
};
SASS_INL uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) {
  return {x, y, z, w};
}
SASS_INL unsigned __float_as_uint(float x) {
  return __builtin_bit_cast(unsigned, x);
}
// Shared-window address of a generic pointer to shared memory, as taken by
// ldmatrix/stmatrix/cp.async operands.
SASS_INL size_t __cvta_generic_to_shared(const void *p) {
  return (size_t)(const __attribute__((address_space(3))) void *)p;
}
// ex2.approx.ftz.f32 / rcp.approx.ftz.f32 (kernels use the same PTX through
// inline asm under nvcc).
SASS_INL float ex2_approx(float x) { return __nvvm_ex2_approx_ftz_f(x); }
SASS_INL float rcp_approx(float x) { return __nvvm_rcp_approx_ftz_f(x); }
#include "cuda_math.h"
#include "cuda_types.h"
