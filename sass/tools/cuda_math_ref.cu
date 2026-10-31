#define U(f) extern "C" __global__ void k_##f(const float*x,float*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define B(f) extern "C" __global__ void k_##f(const float*x,float*y){int i=threadIdx.x;y[i]=f(x[i],x[i+32]);}
#define T(f) extern "C" __global__ void k_##f(const float*x,float*y){int i=threadIdx.x;y[i]=f(x[i],x[i+32],x[i+64]);}
#define UI(f) extern "C" __global__ void k_##f(const float*x,int*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define UU(f) extern "C" __global__ void k_##f(const float*x,unsigned*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define UL(f) extern "C" __global__ void k_##f(const float*x,long long*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define UUL(f) extern "C" __global__ void k_##f(const float*x,unsigned long long*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define IF(f) extern "C" __global__ void k_##f(const int*x,float*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define UF(f) extern "C" __global__ void k_##f(const unsigned*x,float*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define LF(f) extern "C" __global__ void k_##f(const long long*x,float*y){int i=threadIdx.x;y[i]=f(x[i]);}
#define ULF(f) extern "C" __global__ void k_##f(const unsigned long long*x,float*y){int i=threadIdx.x;y[i]=f(x[i]);}
U(expf)U(exp2f)U(exp10f)U(expm1f)U(logf)U(log2f)U(log10f)U(log1pf)B(powf)U(sqrtf)U(rsqrtf)U(cbrtf)
U(sinf)U(cosf)U(tanf)U(tanhf)U(erff)U(erfcf)U(atanf)B(atan2f)U(asinf)U(acosf)U(sinhf)U(coshf)
B(fmodf)B(remainderf)U(floorf)U(ceilf)U(truncf)U(roundf)U(rintf)U(nearbyintf)U(fabsf)B(fminf)B(fmaxf)
B(copysignf)T(fmaf)B(fdimf)B(hypotf)U(asinhf)U(acoshf)U(atanhf)U(erfinvf)U(erfcinvf)U(lgammaf)U(tgammaf)U(logbf)UI(ilogbf)
U(sinpif)U(cospif)U(rcbrtf)U(normcdff)U(normcdfinvf)U(erfcxf)
U(__expf)U(__exp10f)U(__logf)U(__log2f)U(__log10f)B(__powf)U(__sinf)U(__cosf)U(__tanf)B(__fdividef)
U(__frcp_rn)U(__frcp_rz)U(__frcp_ru)U(__frcp_rd)U(__fsqrt_rn)U(__fsqrt_rz)U(__fsqrt_ru)U(__fsqrt_rd)U(__frsqrt_rn)
B(__fdiv_rn)B(__fdiv_rz)B(__fdiv_ru)B(__fdiv_rd)B(__fadd_rn)B(__fadd_rz)B(__fadd_ru)B(__fadd_rd)
B(__fsub_rn)B(__fsub_rz)B(__fsub_ru)B(__fsub_rd)B(__fmul_rn)B(__fmul_rz)B(__fmul_ru)B(__fmul_rd)
T(__fmaf_rn)T(__fmaf_rz)T(__fmaf_ru)T(__fmaf_rd)U(__saturatef)
UI(__float2int_rn)UI(__float2int_rz)UI(__float2int_ru)UI(__float2int_rd)
UU(__float2uint_rn)UU(__float2uint_rz)UU(__float2uint_ru)UU(__float2uint_rd)
UL(__float2ll_rn)UL(__float2ll_rz)UL(__float2ll_ru)UL(__float2ll_rd)
UUL(__float2ull_rn)UUL(__float2ull_rz)UUL(__float2ull_ru)UUL(__float2ull_rd)
IF(__int2float_rn)IF(__int2float_rz)IF(__int2float_ru)IF(__int2float_rd)
UF(__uint2float_rn)UF(__uint2float_rz)UF(__uint2float_ru)UF(__uint2float_rd)
LF(__ll2float_rn)LF(__ll2float_rz)LF(__ll2float_ru)LF(__ll2float_rd)
ULF(__ull2float_rn)ULF(__ull2float_rz)ULF(__ull2float_ru)ULF(__ull2float_rd)
UI(isnan)UI(isinf)UI(isfinite)UI(signbit)UI(__isnanf)UI(__isinff)UI(__finitef)UI(__signbitf)
extern "C" __global__ void k_sincosf(const float*x,float*y){int i=threadIdx.x;sincosf(x[i],y+i,y+i+32);}
extern "C" __global__ void k___sincosf(const float*x,float*y){int i=threadIdx.x;__sincosf(x[i],y+i,y+i+32);}
extern "C" __global__ void k_sincospif(const float*x,float*y){int i=threadIdx.x;sincospif(x[i],y+i,y+i+32);}
extern "C" __global__ void k_ldexpf(const float*x,float*y){int i=threadIdx.x;y[i]=ldexpf(x[i],__float_as_int(x[i+32]));}
extern "C" __global__ void k_scalbnf(const float*x,float*y){int i=threadIdx.x;y[i]=scalbnf(x[i],__float_as_int(x[i+32]));}
extern "C" __global__ void k_frexpf(const float*x,float*y){int i=threadIdx.x;int e;y[i]=frexpf(x[i],&e);y[i+32]=__int_as_float(e);}
extern "C" __global__ void k_modff(const float*x,float*y){int i=threadIdx.x;y[i]=modff(x[i],y+i+32);}
extern "C" __global__ void k_remquof(const float*x,float*y){int i=threadIdx.x;int q;y[i]=remquof(x[i],x[i+32],&q);y[i+64]=__int_as_float(q);}
