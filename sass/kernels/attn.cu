// Flash-attention forward in FP32 on the SIMT pipes (no tensor cores),
// non-causal. Q, K, V, O are [H][S][D] with D = 64, stored as float4.
//
// One thread owns one query row (q and the output accumulator live in
// registers); a CTA owns BQ consecutive rows of one head and streams K/V
// through shared memory in tiles of BK keys, keeping a running max `m` and
// denominator `l` (online softmax in base 2: the caller folds log2(e) and
// 1/sqrt(D) into `scale_log2e`).
#define D 64
#define D4 (D / 4)
#define BQ 64
#define BK 64

#ifndef SASS_CLANG
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
#endif

extern "C" __global__ void __launch_bounds__(BQ)
    attn_fwd(const float4 *__restrict__ Q, const float4 *__restrict__ K,
             const float4 *__restrict__ V, float4 *__restrict__ O, int S,
             float scale_log2e) {
  __shared__ float4 Ks[BK * D4];
  __shared__ float4 Vs[BK * D4];

  const int t = threadIdx.x;
  const size_t head = (size_t)blockIdx.y * S * D4;
  const size_t row = head + (size_t)(blockIdx.x * BQ + t) * D4;

  float4 q[D4], acc[D4];
#pragma unroll
  for (int j = 0; j < D4; ++j) {
    q[j] = Q[row + j];
    acc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float m = -INFINITY, l = 0.f;

  for (int k0 = 0; k0 < S; k0 += BK) {
    const size_t tile = head + (size_t)k0 * D4;
#pragma unroll
    for (int i = 0; i < BK * D4 / BQ; ++i) {
      Ks[i * BQ + t] = K[tile + i * BQ + t];
      Vs[i * BQ + t] = V[tile + i * BQ + t];
    }
    __syncthreads();

    for (int kk = 0; kk < BK; ++kk) {
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < D4; ++j) {
        float4 k = Ks[kk * D4 + j];
        s += q[j].x * k.x;
        s += q[j].y * k.y;
        s += q[j].z * k.z;
        s += q[j].w * k.w;
      }
      s *= scale_log2e;
      float mn = fmaxf(m, s);
      float c = ex2_approx(m - mn), p = ex2_approx(s - mn);
      l = l * c + p;
#pragma unroll
      for (int j = 0; j < D4; ++j) {
        float4 v = Vs[kk * D4 + j];
        acc[j].x = acc[j].x * c + p * v.x;
        acc[j].y = acc[j].y * c + p * v.y;
        acc[j].z = acc[j].z * c + p * v.z;
        acc[j].w = acc[j].w * c + p * v.w;
      }
      m = mn;
    }
    __syncthreads();
  }

  float inv = rcp_approx(l);
#pragma unroll
  for (int j = 0; j < D4; ++j)
    O[row + j] = make_float4(acc[j].x * inv, acc[j].y * inv, acc[j].z * inv,
                             acc[j].w * inv);
}
