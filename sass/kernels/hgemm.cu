// FP16 GEMM on the warp-level tensor cores: C[M][N] (f32) = A[M][K] * B[K][N],
// A and B row-major f16. M, N must be multiples of 128 and K of 32.
//
// A 256-thread CTA computes a 128x128 tile of C, stepping K by 32 through a
// two-stage cp.async pipeline in shared memory. Its 8 warps (2 along M, 4
// along N) each own a 64x32 sub-tile: 4x4 mma.m16n8k16 per 16-wide k-step,
// A fragments from ldmatrix.x4, B fragments from ldmatrix.x2.trans of the
// row-major B tile. Shared rows are padded by 16 bytes so the 8 row
// addresses of each ldmatrix hit distinct bank groups.
//
// Inline PTX (as CUTLASS writes it): nvcc compiles it natively, our backend
// through SASSLowerInlinePTX. __launch_bounds__(256, 2) asks both compilers
// for two resident blocks per SM (at most 128 registers).
#include <stdint.h>

#define BM 128
#define BN 128
#define BK 32
#define AP (BK + 8)  // A tile row pitch, halves
#define BP (BN + 8)  // B tile row pitch, halves

__device__ __forceinline__ unsigned smem(const void *p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void copy16(void *dst, const void *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem(dst)), "l"(src));
}

extern "C" __global__ void __launch_bounds__(256, 2)
    hgemm(const uint16_t *__restrict__ A, const uint16_t *__restrict__ B,
          float *__restrict__ C, int M, int N, int K) {
  __shared__ __attribute__((aligned(16))) uint16_t As[2][BM * AP];
  __shared__ __attribute__((aligned(16))) uint16_t Bs[2][BK * BP];

  const int t = threadIdx.x, lane = t % 32, warp = t / 32;
  const int wm = warp / 4 * 64, wn = warp % 4 * 32;  // warp's sub-tile
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

  // Each thread copies two 16-byte chunks of A and of B per stage.
  auto load = [&](int stage, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      int c = t + i * 256;
      int ar = c / 4, ak = c % 4 * 8;   // A: 128 rows x 4 chunks
      copy16(&As[stage][ar * AP + ak], &A[(size_t)(m0 + ar) * K + k0 + ak]);
      int br = c / 16, bn = c % 16 * 8; // B: 32 rows x 16 chunks
      copy16(&Bs[stage][br * BP + bn], &B[(size_t)(k0 + br) * N + n0 + bn]);
    }
    asm volatile("cp.async.commit_group;");
  };

  float acc[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        acc[i][j][r] = 0.f;

  load(0, 0);
  for (int k0 = 0, s = 0; k0 < K; k0 += BK, s ^= 1) {
    if (k0 + BK < K) {
      load(s ^ 1, k0 + BK);
      asm volatile("cp.async.wait_group 1;" ::: "memory");
    } else {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      unsigned a[4][4], b[4][2];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3])
                     : "r"(smem(&As[s][(wm + i * 16 + lane % 16) * AP + kk + lane / 16 * 8])));
#pragma unroll
      for (int j = 0; j < 4; ++j)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                     : "=r"(b[j][0]), "=r"(b[j][1])
                     : "r"(smem(&Bs[s][(kk + lane % 16) * BP + wn + j * 8])));
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                       "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]),
                         "+f"(acc[i][j][3])
                       : "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3]),
                         "r"(b[j][0]), "r"(b[j][1]));
    }
    __syncthreads();
  }

  // Accumulator fragment: rows lane/4 and lane/4 + 8, columns 2*(lane%4) + {0,1}.
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int r = m0 + wm + i * 16 + lane / 4, c = n0 + wn + j * 8 + lane % 4 * 2;
      C[(size_t)r * N + c] = acc[i][j][0];
      C[(size_t)r * N + c + 1] = acc[i][j][1];
      C[(size_t)(r + 8) * N + c] = acc[i][j][2];
      C[(size_t)(r + 8) * N + c + 1] = acc[i][j][3];
    }
}
