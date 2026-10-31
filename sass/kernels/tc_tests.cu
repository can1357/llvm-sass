// Warp-level tensor-core and async-copy tests. Every kernel runs one warp
// on in[0..127] (uint4) and writes out[0..127]; tools/difftest.py compares
// our build bit for bit against nvcc's, which defines the fragment layouts.
// The instructions are written as inline PTX, the way CUTLASS writes them,
// so the same source builds with nvcc and through SASSLowerInlinePTX.
#include <stdint.h>

#define KERNEL(name) \
  extern "C" __global__ void name(const uint4 *__restrict__ in, uint4 *__restrict__ out)

__device__ __forceinline__ unsigned smem(const void *p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

// ldmatrix: 4 8x8 b16 matrices (512 bytes) staged in shared memory, row
// addresses from every thread.
#define LDSM_TEST(name, num, trans, ...)                                        \
  KERNEL(name) {                                                                \
    __shared__ uint4 s[32];                                                     \
    int t = threadIdx.x;                                                        \
    s[t] = in[t];                                                               \
    __syncthreads();                                                            \
    unsigned r[4] = {0, 0, 0, 0};                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8." num trans ".shared.b16 " __VA_ARGS__ \
                 : "r"(smem(&s[31 - t])));                                      \
    out[t] = make_uint4(r[0], r[1], r[2], r[3]);                                \
  }
LDSM_TEST(t_ldsm_x1, "x1", "", "{%0}, [%1];" : "=r"(r[0]))
LDSM_TEST(t_ldsm_x2, "x2", "", "{%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]))
LDSM_TEST(t_ldsm_x4, "x4", "", "{%0,%1,%2,%3}, [%4];"
          : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]))
LDSM_TEST(t_ldsm_x1t, "x1", ".trans", "{%0}, [%1];" : "=r"(r[0]))
LDSM_TEST(t_ldsm_x2t, "x2", ".trans", "{%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]))
LDSM_TEST(t_ldsm_x4t, "x4", ".trans", "{%0,%1,%2,%3}, [%4];"
          : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]))

// stmatrix: registers from global memory into shared memory.
#define STSM_TEST(name, num, trans, ...)                                        \
  KERNEL(name) {                                                                \
    __shared__ uint4 s[32];                                                     \
    int t = threadIdx.x;                                                        \
    s[t] = make_uint4(0, 0, 0, 0);                                              \
    __syncthreads();                                                            \
    uint4 v = in[t];                                                            \
    asm volatile("stmatrix.sync.aligned.m8n8." num trans ".shared.b16 [%0], " __VA_ARGS__); \
    __syncthreads();                                                            \
    out[t] = s[t];                                                              \
  }
STSM_TEST(t_stsm_x1, "x1", "", "{%1};" ::"r"(smem(&s[t])), "r"(v.x))
STSM_TEST(t_stsm_x2, "x2", "", "{%1,%2};" ::"r"(smem(&s[t])), "r"(v.x), "r"(v.y))
STSM_TEST(t_stsm_x4, "x4", "", "{%1,%2,%3,%4};"
          ::"r"(smem(&s[t])), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w))
STSM_TEST(t_stsm_x4t, "x4", ".trans", "{%1,%2,%3,%4};"
          ::"r"(smem(&s[t])), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w))

// mma.m16n8k16 with fragments straight from registers.
#define MMA_TEST(name, type)                                                    \
  KERNEL(name) {                                                                \
    int t = threadIdx.x;                                                        \
    uint4 a = in[t], b = in[32 + t];                                            \
    float c[4] = {t * 0.25f, -1.5f, t * -0.125f, 3.0f};                         \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32." type "." type ".f32 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"         \
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])               \
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b.x), "r"(b.y)); \
    out[t] = make_uint4(__float_as_uint(c[0]), __float_as_uint(c[1]),           \
                        __float_as_uint(c[2]), __float_as_uint(c[3]));          \
  }
MMA_TEST(t_mma_f16, "f16")
MMA_TEST(t_mma_bf16, "bf16")

// A small GEMM step: A (16x64) and B (64x8) in shared memory, four k-steps
// of ldmatrix + mma accumulating into the same registers. The fragment
// registers are rewritten while earlier HMMAs may still read them.
KERNEL(t_mma_chain) {
  __shared__ uint4 sa[128], sb[64];
  int t = threadIdx.x;
  for (int i = t; i < 128; i += 32)
    sa[i] = in[i];
  for (int i = t; i < 64; i += 32)
    sb[i] = in[64 + i];
  __syncthreads();
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    unsigned a0, a1, a2, a3, b0, b1;
    // A rows are 8 uint4 (64 halves); this k-step's 16x16 block.
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(smem(&sa[(t % 16) * 8 + k * 2 + t / 16])));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b0), "=r"(b1)
                 : "r"(smem(&sb[k * 16 + t % 16])));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  out[t] = make_uint4(__float_as_uint(c[0]), __float_as_uint(c[1]),
                      __float_as_uint(c[2]), __float_as_uint(c[3]));
}

// cp.async of every size into shared memory, in two commit groups.
KERNEL(t_cp_async) {
  __shared__ uint4 s[128];
  int t = threadIdx.x;
  const char *g = reinterpret_cast<const char *>(in);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem(&s[t])),
               "l"(g + 16 * (127 - t)));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(smem(&s[32 + t])),
               "l"(g + 16 * (64 + t)));
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem(&s[64 + t])),
               "l"(g + 8 * t));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem(&s[96 + t])),
               "l"(g + 4 * (31 - t)));
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 1;" ::: "memory");
  __syncthreads();
  uint4 first = s[t];
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  for (int i = 0; i < 4; ++i)
    out[i * 32 + t] = i == 0 ? first : s[i * 32 + t];
}
