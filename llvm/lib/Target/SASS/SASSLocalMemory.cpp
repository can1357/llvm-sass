//===-- SASSLocalMemory.cpp - Per-thread local memory ---------------------===//
//
// Allocas live in the per-thread local window (addrspace 5, 32-bit
// addresses), in the frame below the stack pointer R1 (SASSFrameLowering,
// SASSRegisterInfo::eliminateFrameIndex):
//
// - SASSLowerAlloca replaces every generic alloca by an addrspace(5) alloca
//   plus an addrspacecast back to generic, so that InferAddressSpaces turns
//   the accesses into local ones (LDL/STL) wherever the pointer provably
//   stays local.
//   Allocas of 8 or 16 bytes or more are aligned to 8 or 16 bytes so the
//   load/store vectorizer can use LDL/STL.64/.128, as ptxas does.
// - Pointers that remain generic are converted as ptxas's cvta.local does:
//   generic = local + the 64-bit window base in constant bank 0
//   (SASSSubtarget::localBaseOffset), local = generic - base.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSISelLowering.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static constexpr unsigned LocalAS = 5;

namespace {
class SASSLowerAlloca : public FunctionPass {
public:
  static char ID;
  SASSLowerAlloca() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS lower allocas"; }

  bool runOnFunction(Function &F) override {
    SmallVector<AllocaInst *, 8> Allocas;
    for (Instruction &I : instructions(F))
      if (auto *AI = dyn_cast<AllocaInst>(&I); AI && AI->getAddressSpace() == 0)
        Allocas.push_back(AI);
    const DataLayout &DL = F.getDataLayout();
    for (AllocaInst *AI : Allocas) {
      Align A = AI->getAlign();
      if (std::optional<TypeSize> Size = AI->getAllocationSize(DL);
          Size && !Size->isScalable())
        A = std::max(A, Size->getFixedValue() >= 16  ? Align(16)
                        : Size->getFixedValue() >= 8 ? Align(8)
                                                     : A);
      auto *Local = new AllocaInst(AI->getAllocatedType(), LocalAS,
                                   AI->getArraySize(), A, "",
                                   AI->getIterator());
      Local->takeName(AI);
      auto *Generic = new AddrSpaceCastInst(Local, AI->getType(),
                                            Local->getName() + ".gen",
                                            AI->getIterator());
      // Lifetime markers must name the alloca itself.
      for (User *U : make_early_inc_range(AI->users())) {
        auto *II = dyn_cast<IntrinsicInst>(U);
        if (!II || !II->isLifetimeStartOrEnd())
          continue;
        IRBuilder<> B(II);
        if (II->getIntrinsicID() == Intrinsic::lifetime_start)
          B.CreateLifetimeStart(Local);
        else
          B.CreateLifetimeEnd(Local);
        II->eraseFromParent();
      }
      AI->replaceAllUsesWith(Generic);
      AI->eraseFromParent();
    }
    return !Allocas.empty();
  }
};
} // namespace

char SASSLowerAlloca::ID = 0;

INITIALIZE_PASS(SASSLowerAlloca, "sass-lower-alloca", "SASS lower allocas",
                false, false)

FunctionPass *llvm::createSASSLowerAllocaPass() { return new SASSLowerAlloca(); }

SDValue SASSTargetLowering::lowerLocalAddrSpaceCast(SDValue Op,
                                                    SelectionDAG &DAG) const {
  auto *N = cast<AddrSpaceCastSDNode>(Op);
  unsigned Src = N->getSrcAddressSpace(), Dst = N->getDestAddressSpace();
  SDLoc DL(Op);
  SDValue Ptr = Op.getOperand(0);
  SDValue BaseOff = DAG.getTargetConstant(
      DAG.getSubtarget<SASSSubtarget>().localBaseOffset(), DL, MVT::i32);
  if (Src == LocalAS && Dst == 0)
    return DAG.getNode(ISD::ADD, DL, MVT::i64,
                       DAG.getNode(SASSISD::LDC, DL, MVT::i64, BaseOff),
                       DAG.getZExtOrTrunc(Ptr, DL, MVT::i64));
  if (Src == 0 && Dst == LocalAS)
    return DAG.getNode(ISD::SUB, DL, MVT::i32,
                       DAG.getZExtOrTrunc(Ptr, DL, MVT::i32),
                       DAG.getNode(SASSISD::LDC, DL, MVT::i32, BaseOff));
  report_fatal_error("SASS: unsupported cast between address spaces " +
                     Twine(Src) + " and " + Twine(Dst));
}
