// Differential tests for scalar idioms lifted to one instruction
// (sass/tools/difftest.py --inputs u32): byte dot products (IDP.4A),
// full-warp butterfly reductions (REDUX), and bit operations (POPC, FLO,
// BREV, PRMT, IABS). Plain C, no intrinsics: nvcc lifts only some of these.
//
// Each kernel is one warp mapping 512 words x (and y, the word 256 further
// on) to 512 outputs.
typedef unsigned u32;
#ifdef SASS_CLANG
#define CLZ(x) (u32) __builtin_clzg((u32)(x), 32)
#define CTZ(x) (u32) __builtin_ctzg((u32)(x), 32)
#else
#define CLZ(x) (u32) __clz(x)
#define CTZ(x) (u32)(__ffs(x) ? __ffs(x) - 1 : 32)
#endif

#define K(name, expr)                                                         \
  extern "C" __global__ void name(const uint4 *in, uint4 *out) {              \
    const u32 *x = (const u32 *)in;                                           \
    u32 *o = (u32 *)out;                                                      \
    for (int i = threadIdx.x; i < 512; i += 32) {                             \
      u32 a = x[i];                                                           \
      [[maybe_unused]] u32 b = x[(i + 256) & 511];                            \
      [[maybe_unused]] u32 c = x[(i + 128) & 511];                            \
      o[i] = (expr);                                                          \
    }                                                                         \
  }

static __device__ int dot_ss(u32 a, u32 b, int acc) {
  for (int k = 0; k < 4; ++k)
    acc += (int)(signed char)(a >> (8 * k)) * (int)(signed char)(b >> (8 * k));
  return acc;
}
static __device__ u32 dot_uu(u32 a, u32 b, u32 acc) {
  for (int k = 0; k < 4; ++k)
    acc += ((a >> (8 * k)) & 0xff) * ((b >> (8 * k)) & 0xff);
  return acc;
}
static __device__ int dot_su(u32 a, u32 b, int acc) {
  for (int k = 0; k < 4; ++k)
    acc += (int)(signed char)(a >> (8 * k)) * (int)((b >> (8 * k)) & 0xff);
  return acc;
}

K(i_dot_ss, dot_ss(a, b, c))
K(i_dot_uu, dot_uu(a, b, c))
K(i_dot_su, dot_su(a, b, c))
K(i_dot_us, dot_su(b, a, 7))
K(i_dot2, dot_ss(a, b, 0) + dot_ss(c, a, 0) - 5)
K(i_dot_partial, (u32)((int)(signed char)a * (int)(signed char)b +
                       (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8)))
K(i_bswap, (a >> 24) | ((a >> 8) & 0xff00) | ((a << 8) & 0xff0000) | (a << 24))
K(i_popc, ({ u32 n = 0; for (int k = 0; k < 32; ++k) n += (a >> k) & 1; n; }))
K(i_brev, ({ u32 r = 0; for (int k = 0; k < 32; ++k) r |= ((a >> k) & 1) << (31 - k); r; }))
K(i_clz, CLZ(a >> (b & 31)))
K(i_ctz, CTZ(a << (b & 31)))
K(i_abs, (u32)((int)a < 0 ? -(int)a : (int)a))

#ifdef SASS_CLANG
#define SHFL(v, m) (decltype(v)) __nvvm_shfl_sync_bfly_i32(0xffffffffu, (int)(v), m, 0x1f)
#else
#define SHFL(v, m) __shfl_xor_sync(0xffffffffu, v, m)
#endif
#define WARP(name, T, combine)                                                \
  extern "C" __global__ void name(const uint4 *in, uint4 *out) {              \
    const T *x = (const T *)in;                                               \
    T *o = (T *)out;                                                          \
    for (int i = threadIdx.x; i < 512; i += 32) {                             \
      T v = x[i];                                                             \
      for (int m = 16; m; m >>= 1) {                                          \
        T w = SHFL(v, m);                                                     \
        v = combine;                                                          \
      }                                                                       \
      o[i] = v;                                                               \
    }                                                                         \
  }
WARP(w_sum, int, v + w)
WARP(w_max, int, v > w ? v : w)
WARP(w_min, int, v < w ? v : w)
WARP(w_umax, u32, v > w ? v : w)
WARP(w_umin, u32, v < w ? v : w)
WARP(w_and, int, v & w)
WARP(w_or, int, v | w)
WARP(w_xor, int, v ^ w)
// Not a full butterfly (no mask 16): stays shuffles.
extern "C" __global__ void w_half(const uint4 *in, uint4 *out) {
  const int *x = (const int *)in;
  int *o = (int *)out;
  for (int i = threadIdx.x; i < 512; i += 32) {
    int v = x[i];
    for (int m = 8; m; m >>= 1)
      v += SHFL(v, m);
    o[i] = v;
  }
}
