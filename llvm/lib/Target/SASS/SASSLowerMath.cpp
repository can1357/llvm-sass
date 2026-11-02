//===-- SASSLowerMath.cpp - Approximate f32 math to MUFU primitives -------===//
//
// Lowers approximate single-precision math to the MUFU primitives the
// instruction selector implements (the flush-to-zero NVVM approx intrinsics
// and afn llvm.tanh), the way nvcc and ptxas expand it:
//
// * afn llvm.{exp,exp10,log,log10,pow,tan} and fdiv become their CUDA
//   fast-intrinsic forms: __expf = ex2(x * log2 e), __logf = lg2(x) * ln 2,
//   __powf = ex2(y * lg2 x), __tanf = sin x / cos x, x / y = x * rcp(y),
//   1 / y = rcp(y), 1 / sqrt(x) = rsqrt(x). This is what
//   `nvcc --use_fast_math` emits for expf, logf, powf, ...; clang marks
//   every call afn under -fapprox-func or -ffast-math.
// * llvm.exp2 (CUDA's exp2f is ex2.approx), afn llvm.{log2,sqrt,sin,cos}.
// * The NVVM approx intrinsics that are not flush-to-zero.
//
// In functions whose f32 denormal mode is preserve-sign (-ftz, -fcuda-flush-
// denormals-to-zero) the MUFU primitive is used directly. Otherwise it is
// wrapped in ptxas's range fix-up, which scales a denormal input into the
// normal range and the result back; e.g. ex2.approx.f32 is
// `x >= -126 ? ex2(x) : ex2(x / 2)^2`. The non-ftz NVVM intrinsics flush too
// in flush-to-zero functions, as the libdevice forms nvcc selects with
// __nvvm_reflect("__CUDA_FTZ") do (and every SASS FP32 op there is .FTZ).
//
// Precise (non-afn) operations are left to instruction selection.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class SASSLowerMath : public FunctionPass {
public:
  static char ID;
  SASSLowerMath() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS lower approximate math"; }
  bool runOnFunction(Function &F) override;
};

/// \p V is >= 0 or NaN: an ex2.approx result, x * x, or a select of such.
static bool nonNegative(const Value *V, unsigned Depth = 0) {
  if (Depth > 4)
    return false;
  if (match(V, m_Intrinsic<Intrinsic::nvvm_ex2_approx_ftz>()) ||
      match(V, m_Intrinsic<Intrinsic::nvvm_ex2_approx>()))
    return true;
  const Value *A, *C;
  if (match(V, m_FMul(m_Value(A), m_Deferred(A))))
    return true;
  if (match(V, m_Select(m_Value(), m_Value(A), m_Value(C))))
    return nonNegative(A, Depth + 1) && nonNegative(C, Depth + 1);
  return false;
}

/// \p Y is >= 1 or NaN, so never denormal: a non-negative value plus a
/// constant >= 1 (the `1 + __expf(-x)` of a sigmoid).
static bool atLeastOne(const Value *Y) {
  const Value *A;
  const APFloat *C;
  return match(Y, m_c_FAdd(m_Value(A), m_APFloat(C))) && !C->isNaN() &&
         C->compare(APFloat(C->getSemantics(), 1)) != APFloat::cmpLessThan &&
         nonNegative(A);
}

/// Builds approximate operations in front of one instruction.
class Approx {
  IRBuilder<> B;
  Module &M;
  bool FTZ;

  Constant *c(float V) { return ConstantFP::get(B.getFloatTy(), V); }
  Value *prim(Intrinsic::ID ID, Value *X) {
    if (ID == Intrinsic::nvvm_ex2_approx_ftz)
      return B.CreateIntrinsic(ID, {B.getFloatTy()}, {X});
    return B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, ID), {X});
  }
  /// |x| >= FLT_MIN or NaN: x is not denormal (or zero).
  Value *normal(Value *X) {
    return B.CreateFCmpUGE(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                           c(0x1p-126f));
  }
  /// `P ? x : x * s`, with the original value kept when P holds.
  Value *scaleUnless(Value *P, Value *X, float S) {
    return B.CreateSelect(P, X, B.CreateFMul(X, c(S)));
  }

public:
  Approx(Instruction &I, bool FTZ)
      : B(&I), M(*I.getModule()), FTZ(FTZ) {
    B.setFastMathFlags(I.getFastMathFlags());
  }
  IRBuilder<> &builder() { return B; }

  /// ex2.approx.f32.
  Value *ex2(Value *X) {
    if (FTZ)
      return prim(Intrinsic::nvvm_ex2_approx_ftz, X);
    IRBuilder<>::FastMathFlagGuard G(B);
    B.clearFastMathFlags();
    Value *P = B.CreateFCmpUGE(X, c(-126));
    Value *R = prim(Intrinsic::nvvm_ex2_approx_ftz, scaleUnless(P, X, 0.5f));
    return B.CreateSelect(P, R, B.CreateFMul(R, R));
  }
  /// lg2.approx.f32.
  Value *lg2(Value *X) {
    if (FTZ)
      return prim(Intrinsic::nvvm_lg2_approx_ftz_f, X);
    IRBuilder<>::FastMathFlagGuard G(B);
    B.clearFastMathFlags();
    Value *P = normal(X);
    Value *R = prim(Intrinsic::nvvm_lg2_approx_ftz_f, scaleUnless(P, X, 0x1p24f));
    return B.CreateSelect(P, R, B.CreateFAdd(R, c(-24)));
  }
  /// rsqrt.approx.f32 and sqrt.approx.f32: scale a denormal input by 2^24
  /// and the result by 2^12 or 2^-12.
  Value *rsqrt(Value *X) { return root(Intrinsic::nvvm_rsqrt_approx_ftz_f, X, 0x1p12f); }
  Value *sqrt(Value *X) { return root(Intrinsic::nvvm_sqrt_approx_ftz_f, X, 0x1p-12f); }
  Value *root(Intrinsic::ID ID, Value *X, float Back) {
    if (FTZ)
      return prim(ID, X);
    IRBuilder<>::FastMathFlagGuard G(B);
    B.clearFastMathFlags();
    Value *P = normal(X);
    return scaleUnless(P, prim(ID, scaleUnless(P, X, 0x1p24f)), Back);
  }
  /// rcp.approx.f32: scale a denormal divisor by 2^24 and a huge one
  /// (|x| > 2^126, whose reciprocal is denormal) by 1/4.
  Value *rcp(Value *X) {
    if (FTZ)
      return prim(Intrinsic::nvvm_rcp_approx_ftz_f, X);
    IRBuilder<>::FastMathFlagGuard G(B);
    B.clearFastMathFlags();
    Value *Abs = B.CreateUnaryIntrinsic(Intrinsic::fabs, X);
    Value *Huge = B.CreateFCmpOGT(Abs, c(0x1p126f));
    Value *S = B.CreateSelect(normal(X), c(1), c(0x1p24f));
    S = B.CreateSelect(Huge, c(0.25f), S);
    return B.CreateFMul(S, prim(Intrinsic::nvvm_rcp_approx_ftz_f, B.CreateFMul(X, S)));
  }
  /// div.approx.f32 (\p Full false): x * rcp(y), scaling both by 2^24 for a
  /// denormal y. div.full.f32 also scales both by 1/4 for |y| > 2^126, whose
  /// reciprocal would be denormal; nvcc divides so with -prec-div=false.
  /// Flush-to-zero, both are x * rcp(y) as under --use_fast_math. A y known
  /// to be >= 1 needs no denormal scaling, as nvcc finds (sm_90).
  Value *div(Value *X, Value *Y, bool Full) {
    if (FTZ)
      return B.CreateFMul(X, prim(Intrinsic::nvvm_rcp_approx_ftz_f, Y));
    IRBuilder<>::FastMathFlagGuard G(B);
    B.clearFastMathFlags();
    bool NoDenorm = B.GetInsertBlock()
                        ->getParent()
                        ->getFnAttribute("target-cpu")
                        .getValueAsString()
                        .starts_with("sm_90") &&
                    atLeastOne(Y);
    Value *P = NoDenorm ? nullptr : normal(Y);
    if (Full) {
      Value *Huge = B.CreateFCmpOGT(B.CreateUnaryIntrinsic(Intrinsic::fabs, Y),
                                    c(0x1p126f));
      X = B.CreateSelect(Huge, B.CreateFMul(X, c(0.25f)), X);
      Y = B.CreateSelect(Huge, B.CreateFMul(Y, c(0.25f)), Y);
    }
    if (NoDenorm)
      return B.CreateFMul(prim(Intrinsic::nvvm_rcp_approx_ftz_f, Y), X);
    Value *R = prim(Intrinsic::nvvm_rcp_approx_ftz_f, scaleUnless(P, Y, 0x1p24f));
    return B.CreateFMul(R, scaleUnless(P, X, 0x1p24f));
  }
  /// sin.approx.f32 / cos.approx.f32 (flush-to-zero or not: the same SASS).
  Value *sin(Value *X) { return prim(Intrinsic::nvvm_sin_approx_ftz_f, X); }
  Value *cos(Value *X) { return prim(Intrinsic::nvvm_cos_approx_ftz_f, X); }
};
} // namespace

char SASSLowerMath::ID = 0;

INITIALIZE_PASS(SASSLowerMath, "sass-lower-math", "SASS lower approximate math",
                false, false)

/// `1 / sqrt(x)` with both approximate: rsqrt.approx.
static bool isRsqrt(const Instruction &Div, Value *&X) {
  return Div.getOpcode() == Instruction::FDiv && Div.hasApproxFunc() &&
         match(Div.getOperand(0), m_FPOne()) &&
         match(Div.getOperand(1),
               m_OneUse(m_Intrinsic<Intrinsic::sqrt>(m_Value(X)))) &&
         cast<Instruction>(Div.getOperand(1))->hasApproxFunc();
}

/// The approximate value of \p I, or null when \p I is not approximate math.
static Value *lower(Instruction &I, bool FTZ) {
  if (!I.getType()->isFloatTy() ||
      (I.getOpcode() != Instruction::FDiv && !isa<IntrinsicInst>(I)))
    return nullptr;
  bool AFn = I.hasApproxFunc();
  auto *II = dyn_cast<IntrinsicInst>(&I);
  Approx A(I, FTZ);
  IRBuilder<> &B = A.builder();

  if (I.getOpcode() == Instruction::FDiv) {
    if (!AFn)
      return nullptr;
    Value *X = I.getOperand(0), *Y = I.getOperand(1);
    Value *S;
    if (isRsqrt(I, S))
      return A.rsqrt(S);
    if (match(X, m_FPOne()))
      return A.rcp(Y);
    return A.div(X, Y, /*Full=*/true);
  }

  auto Arg = [&](unsigned N) { return II->getArgOperand(N); };
  auto Mul = [&](Value *X, float C) {
    return B.CreateFMul(X, ConstantFP::get(B.getFloatTy(), C));
  };
  Intrinsic::ID ID = II->getIntrinsicID();
  switch (ID) {
  case Intrinsic::exp:
  case Intrinsic::exp10:
  case Intrinsic::log2:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::pow:
  case Intrinsic::sqrt:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::tan:
    if (!AFn)
      return nullptr;
    break;
  default:
    break;
  }
  switch (ID) {
  case Intrinsic::exp2:
  case Intrinsic::nvvm_ex2_approx:
    return A.ex2(Arg(0));
  case Intrinsic::exp:
    return A.ex2(Mul(Arg(0), 1.44269502f));
  case Intrinsic::exp10:
    return A.ex2(Mul(Arg(0), 3.32192802f));
  case Intrinsic::log2:
  case Intrinsic::nvvm_lg2_approx_f:
    return A.lg2(Arg(0));
  case Intrinsic::log:
    return Mul(A.lg2(Arg(0)), 0.693147182f);
  case Intrinsic::log10:
    return Mul(A.lg2(Arg(0)), 0.30103001f);
  case Intrinsic::pow:
    return A.ex2(B.CreateFMul(Arg(1), A.lg2(Arg(0))));
  case Intrinsic::sqrt: {
    Value *X;
    if (I.hasOneUse() && isRsqrt(*cast<Instruction>(I.user_back()), X))
      return nullptr; // Becomes rsqrt with its division.
    return A.sqrt(Arg(0));
  }
  case Intrinsic::nvvm_sqrt_approx_f:
    return A.sqrt(Arg(0));
  case Intrinsic::nvvm_rsqrt_approx_f:
    return A.rsqrt(Arg(0));
  case Intrinsic::nvvm_div_approx_f:
    return A.div(Arg(0), Arg(1), /*Full=*/false);
  case Intrinsic::nvvm_div_approx_ftz_f:
    return B.CreateFMul(
        Arg(0), B.CreateCall(Intrinsic::getOrInsertDeclaration(
                                 I.getModule(), Intrinsic::nvvm_rcp_approx_ftz_f),
                             {Arg(1)}));
  case Intrinsic::sin:
  case Intrinsic::nvvm_sin_approx_f:
    return A.sin(Arg(0));
  case Intrinsic::cos:
  case Intrinsic::nvvm_cos_approx_f:
    return A.cos(Arg(0));
  case Intrinsic::tan:
    return A.div(A.sin(Arg(0)), A.cos(Arg(0)), /*Full=*/false);
  default:
    return nullptr;
  }
}

bool SASSLowerMath::runOnFunction(Function &F) {
  bool FTZ = F.getDenormalMode(APFloat::IEEEsingle()).Output ==
             DenormalMode::PreserveSign;
  SmallVector<std::pair<Instruction *, Value *>, 16> Done;
  for (Instruction &I : instructions(F))
    if (Value *V = lower(I, FTZ))
      Done.push_back({&I, V});
  for (auto [I, V] : Done) {
    V->takeName(I);
    I->replaceAllUsesWith(V);
  }
  for (auto [I, V] : Done) {
    // The sqrt under an rsqrt division dies with it.
    Value *Root = I->getOpcode() == Instruction::FDiv ? I->getOperand(1) : nullptr;
    I->eraseFromParent();
    if (auto *R = dyn_cast_or_null<IntrinsicInst>(Root);
        R && R->getIntrinsicID() == Intrinsic::sqrt && R->use_empty())
      R->eraseFromParent();
  }
  return !Done.empty();
}

FunctionPass *llvm::createSASSLowerMathPass() { return new SASSLowerMath(); }
