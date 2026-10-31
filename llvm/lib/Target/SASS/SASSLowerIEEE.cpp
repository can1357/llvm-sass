//===-- SASSLowerIEEE.cpp - IEEE f32 division, reciprocal, square root ----===//
//
// Expands the correctly rounded f32 operations (fdiv and llvm.nvvm.div.rn.f,
// llvm.nvvm.rcp.rn.f and 1 / x, llvm.sqrt and llvm.nvvm.sqrt.rn.f) into
// ptxas's sequences: a MUFU estimate refined by FFMA Newton steps, a range
// check, and for operands outside its range ptxas's slow path (a subroutine
// there, an inlined block here), so results match bit for bit:
//
// * a / b: r = rcp(b), r += r * (1 - b * r), q = a * r, q += r * (a - b * q);
//   FCHK a, b selects the slow path (special or denormal operands, results
//   near overflow or underflow), which divides the mantissas, rebuilds the
//   exponent and rounds denormal results from the RZ/RM/RP FFMAs.
// * 1 / x: r = rcp(x), r += r * -(x * r - 1) for exponents 1..252; the slow
//   path scales denormals by 2^64 and rounds the denormal reciprocals of
//   exponents 253 and 254.
// * sqrt(x): s = x * rsq(x), s += (x - s * s) * rsq(x) / 2 for positive
//   normal x >= 2^-102; the slow path handles zeros, negatives, NaN, infinity
//   and scales small inputs by 2^64.
//
// Approximate forms (afn) were lowered by SASSLowerMath before. In
// flush-to-zero functions the same sequences are used (their FFMAs take
// .FTZ like all f32 arithmetic there); ptxas's div.rn.ftz/sqrt.rn.ftz
// sequences are not replicated.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {
class SASSLowerIEEE : public FunctionPass {
public:
  static char ID;
  SASSLowerIEEE() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "SASS lower IEEE division and square root";
  }
  bool runOnFunction(Function &F) override;
};

enum class Op { Div, Rcp, Sqrt };

class IEEEBuilder {
  IRBuilder<> B;
  Module &M;
  Type *I32, *F32;

  Constant *f(float V) { return ConstantFP::get(F32, V); }
  Constant *i(uint32_t V) { return ConstantInt::get(I32, V); }
  Value *bits(Value *X) { return B.CreateBitCast(X, I32); }
  Value *flt(Value *X) { return B.CreateBitCast(X, F32); }
  Value *intr(Intrinsic::ID ID, ArrayRef<Value *> Ops) {
    return B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, ID), Ops);
  }
  Value *rcp(Value *X) { return intr(Intrinsic::nvvm_rcp_approx_ftz_f, X); }
  Value *rsq(Value *X) { return intr(Intrinsic::nvvm_rsqrt_approx_ftz_f, X); }
  Value *fma(Value *A, Value *X, Value *C) {
    return B.CreateIntrinsic(Intrinsic::fma, {F32}, {A, X, C});
  }
  Value *fmaRnd(Intrinsic::ID ID, Value *A, Value *X, Value *C) {
    return intr(ID, {A, X, C});
  }
  Value *sel(Value *P, Value *T, Value *F) { return B.CreateSelect(P, T, F); }
  /// |x| == inf, |x| > inf (NaN).
  Value *isInf(Value *X) {
    return B.CreateFCmpOEQ(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                           f(INFINITY));
  }
  Value *isNaN(Value *X) { return B.CreateFCmpUNO(X, X); }
  /// x << s with SHF.L.U32's result for s >= 32 (0).
  Value *shlClamped(Value *X, Value *S) {
    return sel(B.CreateICmpULT(S, i(32)), B.CreateShl(X, S), i(0));
  }

  //===--------------------------------------------------------------------===//
  // Fast paths
  //===--------------------------------------------------------------------===//

  Value *fastDiv(Value *A, Value *D) {
    Value *R = rcp(D);
    Value *ND = B.CreateFNeg(D);
    R = fma(R, fma(ND, R, f(1)), R);
    Value *Q = fma(A, R, f(0));
    return fma(R, fma(ND, Q, A), Q);
  }
  Value *fastRcp(Value *X) {
    Value *R = rcp(X);
    return fma(R, B.CreateFNeg(fma(X, R, f(-1))), R);
  }
  Value *fastSqrt(Value *X) {
    Value *R = rsq(X);
    Value *S = B.CreateFMul(X, R);
    Value *H = B.CreateFMul(R, f(0.5f));
    return fma(fma(B.CreateFNeg(S), S, X), H, S);
  }

  //===--------------------------------------------------------------------===//
  // Slow paths (ptxas's subroutines)
  //===--------------------------------------------------------------------===//

  // The slow paths branch like ptxas's subroutines (few predicates live at
  // once); their blocks go to the end of the function.
  BasicBlock *block(const Twine &Name) {
    Function *F = B.GetInsertBlock()->getParent();
    return BasicBlock::Create(F->getContext(), Name, F);
  }
  /// Branches to \p Done on \p P with \p V, else to \p Else.
  void exitIf(Value *P, Value *V, BasicBlock *Done, PHINode *Out,
              BasicBlock *Else) {
    Out->addIncoming(V, B.GetInsertBlock());
    B.CreateCondBr(P, Done, Else);
    B.SetInsertPoint(Else);
  }
  void exit(Value *V, BasicBlock *Done, PHINode *Out) {
    Out->addIncoming(V, B.GetInsertBlock());
    B.CreateBr(Done);
  }
  PHINode *result(BasicBlock *Done) {
    IRBuilder<>::InsertPointGuard G(B);
    B.SetInsertPoint(Done);
    return B.CreatePHI(I32, 4);
  }
  /// x is zero, infinite or NaN.
  Value *zeroInfNaN(Value *XI) {
    return B.CreateOr(B.CreateICmpEQ(B.CreateAnd(XI, i(0x7fffffff)), i(0)),
                      B.CreateICmpEQ(B.CreateAnd(XI, i(0x7f800000)), i(0x7f800000)));
  }

  Value *slowDiv(Value *A, Value *D) {
    BasicBlock *Entry = B.GetInsertBlock();
    BasicBlock *Spec = block("div.special"), *Main = block("div.main"),
               *Den = block("div.denorm"), *Done = block("div.done");
    PHINode *Out = result(Done);
    Value *AI = bits(A), *DI = bits(D);
    Value *ED = B.CreateAnd(B.CreateLShr(DI, 23), i(0xff));
    Value *EA = B.CreateAnd(B.CreateLShr(AI, 23), i(0xff));
    Value *Special = B.CreateOr(B.CreateICmpUGT(B.CreateSub(ED, i(1)), i(0xfd)),
                                B.CreateICmpUGT(B.CreateSub(EA, i(1)), i(0xfd)));
    B.CreateCondBr(Special, Spec, Main);

    // Special operands, in ptxas's order; denormal operands are scaled by
    // 2^64 (the exponent fields stay the original ones) and `Scale`
    // corrects the result exponent.
    B.SetInsertPoint(Spec);
    Value *XSign = B.CreateAnd(B.CreateXor(AI, DI), i(0x80000000));
    Value *AZero = B.CreateICmpEQ(B.CreateAnd(AI, i(0x7fffffff)), i(0));
    Value *DZero = B.CreateICmpEQ(B.CreateAnd(DI, i(0x7fffffff)), i(0));
    Value *AInf = isInf(A), *DInf = isInf(D);
    Value *S = sel(B.CreateOr(AInf, DZero), B.CreateOr(XSign, i(0x7f800000)), i(0));
    S = sel(B.CreateOr(DInf, AZero), XSign, S);
    S = sel(B.CreateOr(B.CreateAnd(AZero, DZero), B.CreateAnd(AInf, DInf)),
            i(0x7fffffff), S);
    S = sel(B.CreateOr(isNaN(A), isNaN(D)), bits(B.CreateFAdd(A, D)), S);
    BasicBlock *Scaled = block("div.scale");
    exitIf(B.CreateOr(zeroInfNaN(AI), zeroInfNaN(DI)), S, Done, Out, Scaled);
    Value *SA = B.CreateICmpEQ(EA, i(0)), *SD = B.CreateICmpEQ(ED, i(0));
    Value *A2 = sel(SA, fma(A, f(0x1p64f), f(0)), A);
    Value *D2 = sel(SD, fma(D, f(0x1p64f), f(0)), D);
    Value *Scale = B.CreateAdd(sel(SA, i(-64), i(0)), sel(SD, i(64), i(0)));
    B.CreateBr(Main);

    // Mantissas in [1, 2): the exponents moved into Ex.
    B.SetInsertPoint(Main);
    PHINode *AP = B.CreatePHI(F32, 2), *DP = B.CreatePHI(F32, 2),
            *ScaleP = B.CreatePHI(I32, 2);
    AP->addIncoming(A, Entry);
    AP->addIncoming(A2, Scaled);
    DP->addIncoming(D, Entry);
    DP->addIncoming(D2, Scaled);
    ScaleP->addIncoming(i(0), Entry);
    ScaleP->addIncoming(Scale, Scaled);
    Value *DN = flt(B.CreateSub(bits(DP), B.CreateAdd(B.CreateShl(ED, 23), i(0xc0800000))));
    Value *EA127 = B.CreateSub(EA, i(127));
    Value *R = rcp(DN);
    Value *ND = B.CreateFNeg(DN);
    Value *AN = flt(B.CreateAdd(bits(AP), B.CreateMul(EA127, i(-0x800000))));
    Value *Ex = B.CreateAdd(B.CreateSub(B.CreateAdd(EA127, i(127)), ED), ScaleP);
    Value *R2 = fma(R, fma(R, ND, f(1)), R);
    Value *Q0 = fma(AN, R2, f(0));
    Value *Q1 = fma(R2, fma(ND, Q0, AN), Q0);
    Value *E2 = fma(ND, Q1, AN);
    Value *Q2 = fma(R2, E2, Q1);
    Value *QI = bits(Q2);
    Value *E = B.CreateAdd(B.CreateAnd(B.CreateLShr(QI, 23), i(0xff)), Ex);
    Value *Sign = B.CreateAnd(QI, i(0x80000000));
    Value *V = sel(B.CreateICmpSLT(E, i(-24)), Sign, QI);
    V = sel(B.CreateICmpSGT(E, i(0xfe)), B.CreateOr(Sign, i(0x7f800000)), V);
    V = sel(B.CreateICmpULT(B.CreateSub(E, i(1)), i(0xfe)),
            B.CreateAdd(QI, B.CreateShl(Ex, 23)), V);
    exitIf(B.CreateOr(B.CreateICmpSGE(E, i(1)), B.CreateICmpSLT(E, i(-24))), V, Done,
           Out, Den);

    // Denormal result: round the mantissa shifted right by 1 - E from the
    // truncated quotient, with sticky bits from the RM/RP quotients and
    // the bits shifted out.
    Value *QZ = fmaRnd(Intrinsic::nvvm_fma_rz_f, R2, E2, Q1);
    Value *QM = fmaRnd(Intrinsic::nvvm_fma_rm_f, R2, E2, Q1);
    Value *QP = fmaRnd(Intrinsic::nvvm_fma_rp_f, R2, E2, Q1);
    Value *Mant = B.CreateOr(B.CreateAnd(bits(QZ), i(0x7fffff)), i(0x800000));
    Value *ENZ = B.CreateICmpNE(E, i(0));
    Value *Lost = shlClamped(Mant, B.CreateAdd(E, i(32)));
    Value *Shifted = B.CreateLShr(Mant, sel(ENZ, B.CreateNeg(E), i(0)));
    Value *Inexact = B.CreateOr(B.CreateFCmpUNE(QP, QM),
                                B.CreateAnd(B.CreateICmpNE(Lost, i(0)), ENZ));
    Value *Res = B.CreateLShr(Shifted, i(1));
    Value *Up = B.CreateAnd(B.CreateOr(B.CreateZExt(Inexact, I32), B.CreateAnd(Res, i(1))),
                            Shifted);
    exit(B.CreateOr(B.CreateAdd(Res, Up), Sign), Done, Out);
    B.SetInsertPoint(Done);
    return flt(Out);
  }

  Value *slowRcp(Value *X) {
    BasicBlock *Low = block("rcp.low"), *High = block("rcp.high"),
               *Den = block("rcp.denorm"), *Done = block("rcp.done");
    PHINode *Out = result(Done);
    Value *XI = bits(X);
    Value *E = B.CreateLShr(B.CreateShl(XI, 1), 24);
    B.CreateCondBr(B.CreateICmpEQ(E, i(0)), Low, High);

    // Zero or denormal: denormals scaled by 2^64 around the fast sequence.
    B.SetInsertPoint(Low);
    Value *Y = fma(X, f(0x1p64f), f(0));
    Value *Scaled = fma(fastRcp(Y), f(0x1p64f), f(0));
    exit(bits(sel(B.CreateICmpNE(B.CreateShl(XI, 1), i(0)), Scaled, rcp(X))), Done, Out);

    // Exponents 253 and 254: denormal reciprocal of the mantissa, rounded;
    // others (infinity, NaN) MUFU.RCP.
    B.SetInsertPoint(High);
    Value *S = B.CreateSub(E, i(253));
    exitIf(B.CreateICmpUGT(S, i(1)), bits(rcp(X)), Done, Out, Den);
    Value *MF = flt(B.CreateOr(B.CreateAnd(XI, i(0x7fffff)), i(0x3f800000)));
    Value *R = rcp(MF);
    Value *T = B.CreateFNeg(fma(MF, R, f(-1)));
    Value *RM = fmaRnd(Intrinsic::nvvm_fma_rm_f, R, T, R);
    Value *RP = fmaRnd(Intrinsic::nvvm_fma_rp_f, R, T, R);
    Value *Mant = B.CreateOr(B.CreateAnd(bits(RM), i(0x7fffff)), i(0x800000));
    Value *Kept = B.CreateLShr(B.CreateAnd(B.CreateShl(i(3), S), Mant), S);
    Value *Sticky = B.CreateOr(B.CreateFCmpUNE(RM, RP),
                               B.CreateICmpNE(B.CreateAnd(S, Mant), i(0)));
    Value *Up = B.CreateAnd(B.CreateICmpNE(B.CreateAnd(Kept, i(1)), i(0)),
                            B.CreateOr(Sticky, B.CreateICmpNE(B.CreateAnd(Kept, i(2)), i(0))));
    Value *M = B.CreateLShr(Mant, B.CreateSub(E, i(252)));
    M = B.CreateAdd(M, B.CreateZExt(Up, I32));
    M = sel(B.CreateICmpEQ(B.CreateAnd(XI, i(0x7fffff)), i(0)), B.CreateShl(M, 1), M);
    exit(B.CreateOr(M, B.CreateAnd(XI, i(0x80000000))), Done, Out);
    B.SetInsertPoint(Done);
    return flt(Out);
  }

  Value *slowSqrt(Value *X) {
    Value *XI = bits(X);
    Value *Y = fma(X, f(0x1p64f), f(0));
    Value *R = rsq(Y);
    Value *S = B.CreateFMul(Y, R);
    Value *H = B.CreateFMul(R, f(0.5f));
    Value *V = B.CreateFMul(fma(fma(S, B.CreateFNeg(S), Y), H, S), f(0x1p-32f));
    V = sel(isInf(X), X, V);
    V = sel(isNaN(X), B.CreateFAdd(X, f(1)), V);
    // FSETP.GEU.FTZ x, 0: negative denormals compare as zero.
    Value *NonNeg = B.CreateOr(B.CreateFCmpUGE(X, f(0)),
                               B.CreateFCmpOLT(B.CreateUnaryIntrinsic(Intrinsic::fabs, X),
                                               f(0x1p-126f)));
    V = sel(NonNeg, V, flt(i(0x7fffffff)));
    return sel(B.CreateICmpEQ(B.CreateAnd(XI, i(0x7fffffff)), i(0)), X, V);
  }

public:
  explicit IEEEBuilder(Instruction &I)
      : B(&I), M(*I.getModule()), I32(B.getInt32Ty()), F32(B.getFloatTy()) {}

  /// Replaces \p I (operation \p K on \p A, \p D) by the fast path and a
  /// branch to the slow path.
  void expand(Instruction &I, Op K, Value *A, Value *D) {
    Value *Fast, *Slow;
    switch (K) {
    case Op::Div:
      Fast = fastDiv(A, D);
      Slow = B.CreateCall(SASS::getFChk(M), {A, D});
      break;
    case Op::Rcp: {
      Fast = fastRcp(A);
      Value *E = B.CreateAnd(B.CreateAdd(bits(A), i(0x01800000)), i(0x7f800000));
      Slow = B.CreateICmpULE(E, i(0x01ffffff));
      break;
    }
    case Op::Sqrt:
      Fast = fastSqrt(A);
      Slow = B.CreateICmpUGT(B.CreateAdd(bits(A), i(0xf3000000)), i(0x727fffff));
      break;
    }
    BasicBlock *Head = I.getParent();
    BasicBlock *Tail = Head->splitBasicBlock(I.getIterator(), "ieee.join");
    // The slow path goes to the end of the function, so the fast path falls
    // through to the join.
    BasicBlock *SlowBB =
        BasicBlock::Create(Head->getContext(), "ieee.slow", Head->getParent());
    Head->getTerminator()->eraseFromParent();
    B.SetInsertPoint(Head);
    B.CreateCondBr(Slow, SlowBB, Tail);
    B.SetInsertPoint(SlowBB);
    Value *SlowV = K == Op::Div ? slowDiv(A, D) : K == Op::Rcp ? slowRcp(A) : slowSqrt(A);
    BasicBlock *SlowEnd = B.GetInsertBlock();
    B.CreateBr(Tail);
    B.SetInsertPoint(&Tail->front());
    PHINode *Phi = B.CreatePHI(F32, 2);
    Phi->addIncoming(Fast, Head);
    Phi->addIncoming(SlowV, SlowEnd);
    Phi->takeName(&I);
    I.replaceAllUsesWith(Phi);
    I.eraseFromParent();
  }
};
} // namespace

char SASSLowerIEEE::ID = 0;

INITIALIZE_PASS(SASSLowerIEEE, "sass-lower-ieee",
                "SASS lower IEEE division and square root", false, false)

Function *SASS::getFChk(Module &M) {
  LLVMContext &Ctx = M.getContext();
  FunctionCallee FC = M.getOrInsertFunction(
      FChkFn, FunctionType::get(Type::getInt1Ty(Ctx),
                                {Type::getFloatTy(Ctx), Type::getFloatTy(Ctx)}, false));
  auto *F = cast<Function>(FC.getCallee());
  F->setDoesNotAccessMemory();
  F->setDoesNotThrow();
  F->setWillReturn();
  F->setNoSync();
  return F;
}

/// The IEEE operation \p I computes, with its operands.
static bool classify(Instruction &I, Op &K, Value *&A, Value *&D) {
  if (!I.getType()->isFloatTy())
    return false;
  if (I.getOpcode() == Instruction::FDiv) {
    if (I.hasApproxFunc())
      return false;
    A = I.getOperand(0);
    D = I.getOperand(1);
    K = Op::Div;
    if (match(A, m_FPOne())) {
      K = Op::Rcp;
      A = D;
    }
    return true;
  }
  auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II)
    return false;
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_div_rn_f:
    K = Op::Div;
    A = II->getArgOperand(0);
    D = II->getArgOperand(1);
    return true;
  case Intrinsic::nvvm_rcp_rn_f:
    K = Op::Rcp;
    A = II->getArgOperand(0);
    return true;
  case Intrinsic::sqrt:
    if (I.hasApproxFunc())
      return false;
    [[fallthrough]];
  case Intrinsic::nvvm_sqrt_rn_f:
  case Intrinsic::nvvm_sqrt_f:
    K = Op::Sqrt;
    A = II->getArgOperand(0);
    return true;
  default:
    return false;
  }
}

bool SASSLowerIEEE::runOnFunction(Function &F) {
  SmallVector<Instruction *, 8> Work;
  for (Instruction &I : instructions(F)) {
    Op K;
    Value *A, *D;
    if (classify(I, K, A, D))
      Work.push_back(&I);
  }
  for (Instruction *I : Work) {
    Op K;
    Value *A, *D = nullptr;
    classify(*I, K, A, D);
    IEEEBuilder(*I).expand(*I, K, A, D);
  }
  return !Work.empty();
}

FunctionPass *llvm::createSASSLowerIEEEPass() { return new SASSLowerIEEE(); }
