// Latency-bound kernels: each thread runs one long dependent chain, so at low
// occupancy the time per round is the sum of issue-to-use latencies the
// compiler schedules for.

// murmur3 fmix32 per round: SHF, LOP3, IMAD in one dependent chain.
extern "C" __global__ void fmix(const unsigned *__restrict__ in,
                                unsigned *__restrict__ out, int rounds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned x = in[i];
  for (int r = 0; r < rounds; ++r) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    x += (unsigned)r;
  }
  out[i] = x;
}

// xorshift32 + add-rotate: pure ALU (no IMAD) chain.
extern "C" __global__ void xorshift(const unsigned *__restrict__ in,
                                    unsigned *__restrict__ out, int rounds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned x = in[i] | 1u, y = 0x9e3779b9u;
  for (int r = 0; r < rounds; ++r) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    y = ((y << 7) | (y >> 25)) + x;
  }
  out[i] = x ^ y;
}

// Horner-style FMA chain (FFMA -> FFMA).
extern "C" __global__ void horner(const float *__restrict__ in,
                                  float *__restrict__ out, int rounds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = in[i], acc = 1.0f;
  for (int r = 0; r < rounds; ++r) {
    acc = acc * x + 0.5f;
    acc = acc * x - 0.25f;
  }
  out[i] = acc;
}
