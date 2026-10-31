// Correctness suite for the SASS backend. Every kernel is elementwise:
// out[i] = f(a[i], b[i]) for i < n; tools/check.py holds the references.
#include <stdint.h>

#define BIN(name, TA, TB, TO, expr)                                            \
  extern "C" __global__ void name(const TA *__restrict__ pa,                   \
                                  const TB *__restrict__ pb,                   \
                                  TO *__restrict__ out, int n) {               \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i < n) {                                                               \
      TA a = pa[i];                                                            \
      TB b = pb[i];                                                            \
      out[i] = (expr);                                                         \
    }                                                                          \
  }

typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;

// 32-bit integer
BIN(t_add, i32, i32, i32, a + b)
BIN(t_sub, i32, i32, i32, a - b)
BIN(t_neg, i32, i32, i32, -a)
BIN(t_add3, i32, i32, i32, a + b + (a >> 3))
BIN(t_mul, i32, i32, i32, a * b)
BIN(t_mad, i32, i32, i32, a * b + (b ^ 0x55))
BIN(t_mulimm, i32, i32, i32, a * 37 - b * 3)
BIN(t_mulhi_u, u32, u32, u32, __umulhi(a, b))
BIN(t_mulhi_s, i32, i32, i32, __mulhi(a, b))
BIN(t_logic, u32, u32, u32, (a & b) ^ (a | 0xf0f0) ^ ~b)
BIN(t_andn, u32, u32, u32, a & ~b)
BIN(t_xor3, u32, u32, u32, a ^ b ^ (a >> 7))
BIN(t_shl, u32, u32, u32, a << (b & 31))
BIN(t_shr, u32, u32, u32, a >> (b & 31))
BIN(t_sar, i32, i32, i32, a >> (b & 31))
BIN(t_shimm, u32, u32, u32, (a << 5) + (a >> 27) + (u32)((i32)b >> 9))
BIN(t_rotl, u32, u32, u32, __funnelshift_l(a, a, b))
BIN(t_rotimm, u32, u32, u32, (a << 13) | (a >> 19))
BIN(t_minmax, i32, i32, i32, min(a, b) - max(a, 7))
BIN(t_uminmax, u32, u32, u32, min(a, b) + max(a, b >> 1))
BIN(t_cmpsel, i32, i32, i32, a < b ? a : b + 1)
BIN(t_ucmpsel, u32, u32, u32, a <= b ? a ^ b : 3u)
BIN(t_eqsel, i32, i32, i32, (a == b) + (a != 0) * 2 + (b > 5) * 4)
BIN(t_bool, i32, i32, i32, (a < b && b < 0) || a == 7)
BIN(t_sext8, i32, i32, i32, (i32)(int8_t)a + (i32)(int16_t)b)

// 64-bit integer
BIN(t_add64, i64, i64, i64, a + b)
BIN(t_sub64, i64, i64, i64, a - b)
BIN(t_mul64, i64, i64, i64, a * b)
BIN(t_widemul, i32, i32, i64, (i64)a * (i64)b + 5)
BIN(t_wideumul, u32, u32, u64, (u64)a * (u64)b)
BIN(t_shl64, u64, u64, u64, a << (b & 63))
BIN(t_shr64, u64, u64, u64, a >> (b & 63))
BIN(t_sar64, i64, i64, i64, a >> (b & 63))
BIN(t_shimm64, u64, u64, u64, (a << 3) ^ (a >> 40) ^ (u64)((i64)b >> 33) ^ (b << 32))
BIN(t_logic64, u64, u64, u64, (a & b) | (a ^ ~b))
BIN(t_cmp64, i64, i64, i32, (a < b) + (a <= b) * 2 + (a == b) * 4 + (a != b) * 8 + (a > b) * 16 + (a >= b) * 32)
BIN(t_ucmp64, u64, u64, i32, (a < b) + (a <= b) * 2 + (a > b) * 4 + (a >= b) * 8)
BIN(t_sel64, i64, i64, i64, a > b ? a : b * 3)
BIN(t_ext64, i32, u32, i64, (i64)a + (i64)(u64)b)
BIN(t_trunc64, i64, i64, i32, (i32)(a >> 17) + (i32)b)

// FP32
BIN(t_fma, float, float, float, a * b + 1.5f)
BIN(t_fadd, float, float, float, a + b)
BIN(t_fsub, float, float, float, a - b * 0.25f)
BIN(t_fneg, float, float, float, -a * b)
BIN(t_fabs, float, float, float, fabsf(a) + b)
BIN(t_fminmax, float, float, float, fminf(a, b) - fmaxf(a, 2.0f))
BIN(t_fcmp, float, float, i32, (a < b) + (a <= b) * 2 + (a > b) * 4 + (a >= b) * 8 + (a == b) * 16 + (a != b) * 32)
BIN(t_fcmpu, float, float, i32, !(a < b) + !(a >= b) * 2 + (a != a) * 4)
BIN(t_fsel, float, float, float, a > b ? a * 2.0f : b)
BIN(t_i2f, i32, u32, float, (float)a + (float)b)
BIN(t_f2i, float, float, i32, (i32)a + (i32)(u32)fabsf(b))
BIN(t_bits, float, i32, i32, __float_as_int(a) ^ __float_as_int((float)b))

// Control flow
extern "C" __global__ void t_loop(const float *__restrict__ pa,
                                  const i32 *__restrict__ pb,
                                  float *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = pa[i], acc = 0.f;
  int k = pb[i] & 15;
  for (int j = 0; j < k; ++j)
    acc = acc * x + (float)j;
  out[i] = acc;
}

extern "C" __global__ void t_diamond(const i32 *__restrict__ pa,
                                     const i32 *__restrict__ pb,
                                     i32 *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  i32 a = pa[i], b = pb[i], r;
  if (a > b)
    r = a * 3 - b;
  else
    r = (b << 2) ^ a;
  if (r & 1)
    r += 11;
  out[i] = r;
}

extern "C" __global__ void t_while(const u32 *__restrict__ pa,
                                   const u32 *__restrict__ pb,
                                   u32 *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  u32 x = pa[i] | 1, steps = 0;
  while (x != 1 && steps < 64) { // Collatz, bounded
    x = (x & 1) ? 3 * x + 1 : x >> 1;
    ++steps;
  }
  out[i] = steps * 1000 + (x & 0xff) + (pb[i] & 3);
}

// Shared-memory staging: 128-bit global->shared copies (LDGSTS in our
// backend), a barrier, then reads of other threads' elements. Assumes
// 128-thread blocks.
struct alignas(16) I4 {
  i32 v[4];
};
extern "C" __global__ void t_tile(const i32 *__restrict__ pa,
                                  const i32 *__restrict__ pb,
                                  i32 *__restrict__ out, int n) {
  __shared__ I4 s[128];
  int t = threadIdx.x, g = blockIdx.x * 32 + t;
  s[t] = reinterpret_cast<const I4 *>(pa)[min(g, n / 4 - 1)];
  __syncthreads();
  int base = blockIdx.x * blockDim.x, i = base + t, j = base + 127 - t;
  if (i < n)
    out[i] = (j < n ? reinterpret_cast<const i32 *>(s)[127 - t] : 0) + pb[i];
}

// Narrow memory
extern "C" __global__ void t_bytes(const uint8_t *__restrict__ pa,
                                   const int8_t *__restrict__ pb,
                                   int16_t *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = (int16_t)(pa[i] * 3 + pb[i]);
}

extern "C" __global__ void t_halves(const uint16_t *__restrict__ pa,
                                    const int16_t *__restrict__ pb,
                                    uint8_t *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = (uint8_t)(pa[i] + pb[i] * 5);
}

// Launch geometry
extern "C" __global__ void t_geom(const i32 *__restrict__ pa,
                                  const i32 *__restrict__ pb,
                                  i32 *__restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = threadIdx.x + blockIdx.x * 1000 + blockDim.x * 7 + gridDim.x * 13 +
             threadIdx.y + blockIdx.y;
}
