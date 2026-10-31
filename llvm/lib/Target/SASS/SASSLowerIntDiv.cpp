//===-- SASSLowerIntDiv.cpp - Integer division by a variable divisor ------===//
//
// SASS has no integer divide. Division and remainder by a non-constant
// divisor are expanded the way ptxas expands PTX div/rem (constant divisors
// are left to the DAG's multiply-high sequences):
//
// * 32 bits: a reciprocal estimate of the divisor in floating point,
//   I2F[.U32].RP + MUFU.RCP, whose bits + 0x0ffffffe truncated by
//   F2I.FTZ.U32.TRUNC.NTZ give x ~ 2^32 / d from below, one Newton step
//   x += umulhi(x, -d * x), the quotient estimate q = umulhi(x, n) and two
//   corrections `r = n - q * d; if (r >= d) { r -= d; ++q; }`. A zero
//   divisor yields all ones (ptxas's `~d`). Signed forms divide magnitudes
//   (the reciprocal from a signed I2F.RP of |d|, as ptxas) and fix the
//   sign of the quotient from n ^ d and of the remainder from n.
// * 64 bits: as nvcc (whose NVVM bypasses slow division), operands whose
//   high words are all zero take the 32-bit unsigned sequence; the others
//   the 64-bit one: I2F.U64.RP + MUFU.RCP, bits + 0x1ffffffe, F2I.U64.TRUNC,
//   two Newton steps x += umulh(x, -d * x) with 128-bit high products built
//   from 32-bit IMAD.WIDE.U32 pieces, and the same corrections.
//
// * 64-bit constant divisors other than powers of two (which the DAG turns
//   into shifts) multiply by the magic reciprocal (umulh as above; the DAG's
//   own sequences need a 64-bit MULHU, and offering one would make it widen
//   32-bit divisions by constants too).
//
// Narrower types are extended to 32 bits.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Pass.h"
#include "llvm/Support/DivisionByConstantInfo.h"

using namespace llvm;

namespace {
class SASSLowerIntDiv : public FunctionPass {
public:
  static char ID;
  SASSLowerIntDiv() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "SASS lower integer division";
  }
  bool runOnFunction(Function &F) override;
};

/// Emits the expansions in front of one instruction.
class DivBuilder {
  IRBuilder<> B;
  Type *I32, *I64, *F32;

  Value *intr(Intrinsic::ID ID, Value *X) {
    return B.CreateIntrinsic(ID, {}, {X});
  }
  Value *c32(uint64_t V) { return ConstantInt::get(I32, V); }
  Value *c64(uint64_t V) { return ConstantInt::get(I64, V); }
  Value *lo(Value *X) { return B.CreateTrunc(X, I32); }
  Value *hi(Value *X) { return B.CreateTrunc(B.CreateLShr(X, 32), I32); }
  Value *wide(Value *X, Value *Y) {
    return B.CreateMul(B.CreateZExt(X, I64), B.CreateZExt(Y, I64));
  }
  Value *umulhi32(Value *X, Value *Y) { return hi(wide(X, Y)); }
  /// x + umulhi(x, m): one IMAD.WIDE.U32 with addend {0, x}, as ptxas's
  /// IMAD.HI.U32 x, m, {RZ', x}.
  Value *newton32(Value *X, Value *M) {
    return hi(B.CreateAdd(wide(X, M), B.CreateShl(B.CreateZExt(X, I64), 32)));
  }
  /// High 64 bits of the 128-bit product of \p X and \p Y.
  Value *umulh64(Value *X, Value *Y) {
    Value *XL = lo(X), *XH = hi(X), *YL = lo(Y), *YH = hi(Y);
    Value *LL = wide(XL, YL), *LH = wide(XL, YH), *HL = wide(XH, YL),
          *HH = wide(XH, YH);
    Value *Mask = c64(0xffffffffu);
    Value *Mid = B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)),
                             B.CreateAnd(HL, Mask));
    return B.CreateAdd(B.CreateAdd(HH, B.CreateLShr(LH, 32)),
                       B.CreateAdd(B.CreateLShr(HL, 32), B.CreateLShr(Mid, 32)));
  }
  Value *select(Value *P, Value *T, Value *F) { return B.CreateSelect(P, T, F); }

  /// Quotient estimate and remainder of \p N / \p D after the first
  /// correction (32 bits).
  std::pair<Value *, Value *> core32(Value *N, Value *D, bool SignedCvt) {
    Value *F = intr(SignedCvt ? Intrinsic::nvvm_i2f_rp : Intrinsic::nvvm_ui2f_rp, D);
    Value *R = intr(Intrinsic::nvvm_rcp_approx_ftz_f, F);
    R = B.CreateBitCast(B.CreateAdd(B.CreateBitCast(R, I32), c32(0x0ffffffe)), F32);
    Value *X = intr(Intrinsic::nvvm_f2ui_rz_ftz, R);
    X = newton32(X, B.CreateMul(B.CreateNeg(D), X));
    Value *Q = umulhi32(X, N);
    Value *Rem = B.CreateSub(N, B.CreateMul(Q, D));
    Value *P = B.CreateICmpUGE(Rem, D);
    return {select(P, B.CreateAdd(Q, c32(1)), Q), select(P, B.CreateSub(Rem, D), Rem)};
  }

  /// Quotient estimate and remainder of \p N / \p D after the first
  /// correction (64 bits).
  std::pair<Value *, Value *> core64(Value *N, Value *D) {
    Value *F = intr(Intrinsic::nvvm_ull2f_rp, D);
    Value *R = intr(Intrinsic::nvvm_rcp_approx_ftz_f, F);
    R = B.CreateBitCast(B.CreateAdd(B.CreateBitCast(R, I32), c32(0x1ffffffe)), F32);
    Value *X = intr(Intrinsic::nvvm_f2ull_rz, R);
    for (int I = 0; I < 2; ++I)
      X = B.CreateAdd(X, umulh64(X, B.CreateNeg(B.CreateMul(X, D))));
    Value *Q = umulh64(X, N);
    Value *Rem = B.CreateSub(N, B.CreateMul(Q, D));
    Value *P = B.CreateICmpUGE(Rem, D);
    return {select(P, B.CreateAdd(Q, c64(1)), Q), select(P, B.CreateSub(Rem, D), Rem)};
  }

  /// The division or remainder \p Op of \p N by \p D at width 32 or 64.
  Value *divide(Instruction::BinaryOps Op, Value *N, Value *D) {
    Type *T = N->getType();
    bool Is64 = T == I64;
    bool Signed = Op == Instruction::SDiv || Op == Instruction::SRem;
    bool Rem = Op == Instruction::URem || Op == Instruction::SRem;
    Value *Zero = ConstantInt::get(T, 0), *One = ConstantInt::get(T, 1);
    Value *AN = N, *AD = D;
    if (Signed) {
      AN = B.CreateBinaryIntrinsic(Intrinsic::abs, N, B.getFalse());
      AD = B.CreateBinaryIntrinsic(Intrinsic::abs, D, B.getFalse());
    }
    auto [Q, R] = Is64 ? core64(AN, AD) : core32(AN, AD, Signed);
    Value *P = B.CreateICmpUGE(R, AD);
    Value *V = Rem ? select(P, B.CreateSub(R, AD), R) : select(P, B.CreateAdd(Q, One), Q);
    if (Signed) {
      Value *Neg = B.CreateICmpSLT(Rem ? N : B.CreateXor(N, D), Zero);
      V = select(Neg, B.CreateNeg(V), V);
    }
    return select(B.CreateICmpEQ(D, Zero), Constant::getAllOnesValue(T), V);
  }

  /// Quotient of \p N by the 64-bit constant \p C (not 0, +-1 or +-2^k).
  Value *divideConst(bool Signed, Value *N, const APInt &C) {
    if (!Signed) {
      auto Magic = UnsignedDivisionByConstantInfo::get(C);
      Value *Q = B.CreateLShr(N, Magic.PreShift);
      Q = umulh64(Q, c64(Magic.Magic.getZExtValue()));
      if (Magic.IsAdd)
        Q = B.CreateAdd(B.CreateLShr(B.CreateSub(N, Q), 1), Q);
      return B.CreateLShr(Q, Magic.PostShift);
    }
    auto Magic = SignedDivisionByConstantInfo::get(C);
    Value *MC = c64(Magic.Magic.getZExtValue());
    // mulhs(n, m) = umulh(n, m) - (n < 0 ? m : 0) - (m < 0 ? n : 0).
    Value *Zero = c64(0);
    Value *Q = umulh64(N, MC);
    Q = B.CreateSub(Q, select(B.CreateICmpSLT(N, Zero), MC, Zero));
    if (Magic.Magic.isNegative())
      Q = B.CreateSub(Q, N);
    if (C.isStrictlyPositive() && Magic.Magic.isNegative())
      Q = B.CreateAdd(Q, N);
    if (C.isNegative() && Magic.Magic.isStrictlyPositive())
      Q = B.CreateSub(Q, N);
    Q = B.CreateAShr(Q, Magic.ShiftAmount);
    return B.CreateAdd(Q, B.CreateLShr(Q, 63));
  }

public:
  explicit DivBuilder(Instruction &I)
      : B(&I), I32(B.getInt32Ty()), I64(B.getInt64Ty()), F32(B.getFloatTy()) {}

  /// Expands \p I (a division or remainder of i64, i32 or narrower).
  void expand(BinaryOperator &I) {
    Instruction::BinaryOps Op = I.getOpcode();
    Value *N = I.getOperand(0), *D = I.getOperand(1);
    Type *T = I.getType();
    unsigned Bits = T->getIntegerBitWidth();
    if (Bits < 32) {
      bool Signed = Op == Instruction::SDiv || Op == Instruction::SRem;
      auto Ext = [&](Value *V) {
        return Signed ? B.CreateSExt(V, I32) : B.CreateZExt(V, I32);
      };
      I.replaceAllUsesWith(B.CreateTrunc(divide(Op, Ext(N), Ext(D)), T));
      I.eraseFromParent();
      return;
    }
    if (Bits == 32) {
      I.replaceAllUsesWith(divide(Op, N, D));
      I.eraseFromParent();
      return;
    }
    if (auto *C = dyn_cast<ConstantInt>(D)) {
      bool Signed = Op == Instruction::SDiv || Op == Instruction::SRem;
      Value *Q = divideConst(Signed, N, C->getValue());
      bool Rem = Op == Instruction::URem || Op == Instruction::SRem;
      I.replaceAllUsesWith(Rem ? B.CreateSub(N, B.CreateMul(Q, D)) : Q);
      I.eraseFromParent();
      return;
    }
    // 64 bits: high words of both zero -> the 32-bit unsigned sequence.
    BasicBlock *Head = I.getParent();
    Value *Small = B.CreateICmpEQ(B.CreateLShr(B.CreateOr(N, D), 32), c64(0));
    Instruction *Term = I.getNextNode();
    BasicBlock *Tail = Head->splitBasicBlock(Term, "div.join");
    LLVMContext &Ctx = Head->getContext();
    Function *F = Head->getParent();
    BasicBlock *Narrow = BasicBlock::Create(Ctx, "div.32", F, Tail);
    BasicBlock *Full = BasicBlock::Create(Ctx, "div.64", F, Tail);
    Head->getTerminator()->eraseFromParent();
    B.SetInsertPoint(Head);
    B.CreateCondBr(Small, Narrow, Full);

    B.SetInsertPoint(Narrow);
    Instruction::BinaryOps UOp =
        Op == Instruction::SDiv || Op == Instruction::UDiv ? Instruction::UDiv
                                                            : Instruction::URem;
    Value *VN = B.CreateZExt(divide(UOp, lo(N), lo(D)), I64);
    B.CreateBr(Tail);

    B.SetInsertPoint(Full);
    Value *VF = divide(Op, N, D);
    B.CreateBr(Tail);

    B.SetInsertPoint(&Tail->front());
    PHINode *Phi = B.CreatePHI(I64, 2);
    Phi->addIncoming(VN, Narrow);
    Phi->addIncoming(VF, Full);
    I.replaceAllUsesWith(Phi);
    I.eraseFromParent();
  }
};
} // namespace

char SASSLowerIntDiv::ID = 0;

INITIALIZE_PASS(SASSLowerIntDiv, "sass-lower-int-div",
                "SASS lower integer division", false, false)

bool SASSLowerIntDiv::runOnFunction(Function &F) {
  SmallVector<BinaryOperator *, 8> Divs;
  for (Instruction &I : instructions(F)) {
    auto *BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || !BO->getType()->isIntegerTy() ||
        BO->getType()->getIntegerBitWidth() > 64)
      continue;
    // The DAG divides by 32-bit constants (multiply-high) and by powers of
    // two (shifts).
    if (auto *C = dyn_cast<ConstantInt>(BO->getOperand(1))) {
      const APInt &V = C->getValue();
      if (V.getBitWidth() != 64 || V.isZero() || V.isPowerOf2() ||
          V.isNegatedPowerOf2() || V.isAllOnes())
        continue;
    }
    switch (BO->getOpcode()) {
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::URem:
    case Instruction::SRem:
      Divs.push_back(BO);
      break;
    default:
      break;
    }
  }
  for (BinaryOperator *BO : Divs)
    DivBuilder(*BO).expand(*BO);
  return !Divs.empty();
}

FunctionPass *llvm::createSASSLowerIntDivPass() { return new SASSLowerIntDiv(); }
