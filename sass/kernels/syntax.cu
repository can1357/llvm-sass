// Probe kernels: compile with nvcc -arch=sm_121 and lift to learn instruction spellings.
extern "C" __global__ void k_int(int *out, const int *in, int n, unsigned s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int a = in[i];
  int b = in[i + 1];
  int r = a + b;
  r = r - (a << 3);
  r ^= b & 0xff00;
  r |= (unsigned)a >> s;
  r += a * b;
  r += (a < b) ? 7 : r;
  r += min(a, b) + max(a, n);
  r = (int)((unsigned)r >> 5) | (r << 27);
  out[i] = r;
  out[i + gridDim.x] = blockDim.y + gridDim.y + threadIdx.y + blockIdx.y;
}

extern "C" __global__ void k_fp(float *out, const float *in, float a) {
  int i = threadIdx.x;
  float x = in[i], y = in[i + 32];
  float r = x * y + a;
  r = r - x * 0.5f;
  r = fmaxf(r, y) + fminf(x, 2.0f);
  r = (x > y) ? r : -r;
  r += (float)i + fabsf(x);
  out[i] = r;
  out[i + 32] = __int_as_float((int)y);
}

extern "C" __global__ void k_loop(float *out, const float *in, int n) {
  float acc = 0.f;
  for (int k = 0; k < n; ++k)
    acc = acc * in[k] + 1.0f;
  out[threadIdx.x] = acc;
}

extern "C" __global__ void k_i64(long long *out, const long long *in, long long c) {
  int i = threadIdx.x;
  long long a = in[i];
  out[i] = (a + c) ^ (a << 7) ^ (a >> 3);
  out[i + 64] = a * c;
}

extern "C" __global__ void k_mem(float *out, const float4 *in, unsigned char *b, short *h) {
  __shared__ float s[256];
  int i = threadIdx.x;
  float4 v = in[i];
  s[i] = v.x + v.y + v.z + v.w;
  __syncthreads();
  out[i] = s[255 - i];
  b[i] = b[i + 1] + 1;
  h[i] = h[i + 2] * 3;
}
