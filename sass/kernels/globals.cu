// Dynamic shared memory, __constant__ data and by-value struct /
// __grid_constant__ kernel parameters; tools/globals_test.py runs each kernel
// against the nvcc build of this file.

// ---- (A) dynamic shared memory ----
extern "C" __global__ void dsmem(float *o, const float *a) {
  extern __shared__ float s[];
  __shared__ float t[64];
  s[threadIdx.x] = a[threadIdx.x];
  t[threadIdx.x & 63] = 1.f;
  __syncthreads();
  o[threadIdx.x] = s[blockDim.x - 1 - threadIdx.x] + t[threadIdx.x & 63];
}

extern "C" __global__ void dyn_only(float *o, const float *a) {
  extern __shared__ float4 v4[];
  int t = threadIdx.x;
  float4 x;
  x.x = a[4 * t];
  x.y = a[4 * t + 1];
  x.z = a[4 * t + 2];
  x.w = a[4 * t + 3];
  v4[t] = x;
  __syncthreads();
  float4 y = v4[blockDim.x - 1 - t];
  o[4 * t] = y.x + 1.f;
  o[4 * t + 1] = y.y * 2.f;
  o[4 * t + 2] = y.z - 3.f;
  o[4 * t + 3] = y.w;
}

// Two extern declarations alias the same dynamic base.
extern "C" __global__ void dyn_alias(int *o) {
  extern __shared__ float fa[];
  extern __shared__ int ib[];
  __shared__ short st[259]; // 518 bytes: the dynamic base is rounded to 16
  int t = threadIdx.x;
  st[t] = t * 7 + 1;
  fa[t] = (float)t * 0.5f;
  __syncthreads();
  o[t] = ib[t ^ 1] + st[t % 3];
}

// 64 KB of dynamic shared memory (needs MAX_DYNAMIC_SHARED_SIZE_BYTES).
extern "C" __global__ void dyn_big(int *o) {
  extern __shared__ int big[];
  int t = threadIdx.x;
  for (int k = t; k < 16384; k += blockDim.x)
    big[k] = k * 3 + 1;
  __syncthreads();
  int s = 0;
  for (int k = 0; k < 64; ++k)
    s += big[(t * 64 + k * 17) & 16383];
  o[t] = s;
}

// ---- (B) __constant__ ----
__constant__ float ctab[64] = {1, 2, 3, 4, 5, 6, 7, 8};
__constant__ int cscal; // written by the host (cuModuleGetGlobal)
extern "C" __global__ void const_ops(float *o) {
  o[threadIdx.x] = ctab[threadIdx.x & 63] + ctab[5] + (float)cscal;
}

struct CS {
  int a;
  short b;
  signed char c;
  float f;
  long long q;
};
__constant__ CS cs = {-5, -300, -7, 2.5f, 0x123456789abcdefLL};
__constant__ char cstr[13] = "hello world!";
__constant__ unsigned short cus[5] = {1, 65535, 3, 40000, 5};
__constant__ float cmat[4][4] = {{1.5f, 2, 3, 4}, {5, 6, 7, 8},
                                  {9, 10, 11, 12}, {13, 14, 15, 16.25f}};
// Initializer-only constants, byte/short/struct fields, a loop.
extern "C" __global__ void const_init(int *o, long long *q, int n) {
  int t = threadIdx.x;
  o[t] = cs.a + cs.b + cs.c + (int)(cs.f * 2.f) + cstr[t % 13] + cus[t % 5];
  float acc = 0.f;
  for (int k = 0; k < n; ++k)
    acc += cmat[k & 3][(k >> 2) & 3] * (float)(t + 1);
  o[t + 256] = (int)acc;
  q[t] = cs.q + t;
}

// Pointers into __constant__ data through select and a loop induction.
extern "C" __global__ void const_ptr(float *o, int n) {
  int t = threadIdx.x;
  const float *q = (t & 1) ? ctab : ctab + 8;
  float acc = q[t & 7];
  for (const float *p = cmat[0]; p < cmat[0] + n; ++p)
    acc = acc * 0.5f + *p;
  o[t] = acc;
}

// ---- (C) by-value parameters ----
struct P {
  float *o;
  const float *a;
  int n;
  float s;
};
struct Big {
  float v[20];
};
extern "C" __global__ void struct_ops(P p, Big b) {
  int i = threadIdx.x;
  if (i < p.n)
    p.o[i] = p.a[i] * p.s + b.v[i % 20] + b.v[3];
}

struct __align__(16) V {
  float x, y, z, w;
};
struct alignas(64) TM {
  signed char b[128];
};
struct C3 {
  signed char a, b, c;
};
extern "C" __global__ void params_mix(signed char c, P p, Big b, long long ll, V v,
                                      C3 c3, short s, const __grid_constant__ TM tm,
                                      int last, int *oi) {
  int i = threadIdx.x;
  p.o[i] = p.a[i] * p.s + b.v[i % 20];
  oi[i] = c + c3.a + c3.b * 3 + c3.c + s + tm.b[i & 127] + last + p.n +
          (int)(ll >> 40) + (int)ll + (int)v.x + (int)v.y + (int)v.z + (int)v.w;
}

// A by-value parameter whose address is selected (a local copy).
extern "C" __global__ void psel(float *o, Big b) {
  int i = threadIdx.x;
  const float *q = (i & 1) ? &b.v[0] : &b.v[10];
  o[i] = q[i & 7] * 2.f;
}

// A by-value parameter that is written (dynamic index: a local copy).
extern "C" __global__ void pmod(int *o, Big b, P p) {
  int i = threadIdx.x;
  b.v[i % 20] += 1.f;
  p.n += i;
  o[i] = (int)b.v[(i + 1) % 20] + (int)b.v[i % 20] + p.n;
}
