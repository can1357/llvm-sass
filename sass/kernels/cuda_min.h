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
SASS_INL float __shfl_xor_sync(unsigned mask, float v, int lane) {
  return __nvvm_shfl_sync_bfly_f32(mask, v, lane, 0x1f);
}
// ex2.approx.ftz.f32 / rcp.approx.ftz.f32 (kernels use the same PTX through
// inline asm under nvcc).
SASS_INL float ex2_approx(float x) { return __nvvm_ex2_approx_ftz_f(x); }
SASS_INL float rcp_approx(float x) { return __nvvm_rcp_approx_ftz_f(x); }
