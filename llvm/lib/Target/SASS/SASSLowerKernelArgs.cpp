//===-- SASSLowerKernelArgs.cpp - Kernel argument address spaces -----------===//
//
// CUDA kernel pointer parameters always point to global memory. Rewriting
// each use of `ptr %p` to `addrspacecast (addrspacecast %p to ptr
// addrspace(1)) to ptr` lets InferAddressSpaces turn the derived generic
// accesses into global ones.
//
// By-value parameters (structs, __grid_constant__ objects) live in constant
// bank 0. Their uses become the generic cast of the addrspace(4) address
// `SASS::KernelParamsSymbol + bank offset`, so InferAddressSpaces turns the
// loads into constant loads (LDC c[0x0][...]). A parameter that is written,
// or whose address escapes, is first copied to a local alloca; SROA removes
// the copy when the accesses are known.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

SmallVector<SASS::ParamSlot, 8> SASS::paramLayout(const Function &F,
                                                  unsigned ParamBase) {
  const DataLayout &DL = F.getDataLayout();
  SmallVector<ParamSlot, 8> Slots;
  unsigned End = 0;
  for (const Argument &A : F.args()) {
    Type *Ty = A.hasByValAttr() ? A.getParamByValType() : A.getType();
    unsigned Size = DL.getTypeAllocSize(Ty);
    ParamSlot S;
    if (!A.hasByValAttr() && Size <= 8) {
      unsigned Slot = Size > 4 ? 8 : 4;
      S = {0, Slot, Slot, false};
    } else {
      Align Al = A.hasByValAttr()
                     ? A.getParamAlign().value_or(DL.getABITypeAlign(Ty))
                     : DL.getABITypeAlign(Ty);
      S = {0, Size, unsigned(Al.value()), true};
    }
    S.Offset = alignTo(ParamBase + End, S.Align) - ParamBase;
    End = S.Offset + S.Size;
    Slots.push_back(S);
  }
  return Slots;
}

/// True if \p Ptr is only read: loaded from, directly or through GEPs, or
/// the source of a memcpy.
static bool onlyRead(Value *Ptr) {
  SmallVector<Value *, 8> Work = {Ptr};
  while (!Work.empty()) {
    Value *V = Work.pop_back_val();
    for (Use &U : V->uses()) {
      User *Usr = U.getUser();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
        if (GEP->getPointerOperand() != V)
          return false;
        Work.push_back(GEP);
      } else if (auto *MT = dyn_cast<MemTransferInst>(Usr)) {
        if (&U != &MT->getRawSourceUse())
          return false;
      } else if (!isa<LoadInst>(Usr)) {
        return false;
      }
    }
  }
  return true;
}

namespace {
class SASSLowerKernelArgs : public FunctionPass {
public:
  static char ID;
  SASSLowerKernelArgs() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS lower kernel args"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override {
    if (F.getCallingConv() != CallingConv::PTX_Kernel || F.isDeclaration())
      return false;
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    unsigned ParamBase = 0;
    SmallVector<SASS::ParamSlot, 8> Slots;
    bool Changed = false;
    for (Argument &A : F.args()) {
      if (A.use_empty())
        continue;
      if (A.hasByValAttr()) {
        if (Slots.empty()) {
          const auto &TM = getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
          ParamBase = static_cast<const SASSSubtarget *>(TM.getSubtargetImpl(F))
                          ->paramBase();
          Slots = SASS::paramLayout(F, ParamBase);
        }
        lowerByVal(A, ParamBase, Slots[A.getArgNo()], B);
        Changed = true;
        continue;
      }
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT || PT->getAddressSpace() != 0)
        continue;
      Value *Global = B.CreateAddrSpaceCast(
          &A, PointerType::get(F.getContext(), 1), A.getName() + ".global");
      Value *Generic = B.CreateAddrSpaceCast(Global, PT);
      A.replaceUsesWithIf(Generic, [&](Use &U) { return U.getUser() != Global; });
      Changed = true;
    }
    return Changed;
  }

private:
  static void lowerByVal(Argument &A, unsigned ParamBase,
                         const SASS::ParamSlot &S, IRBuilder<> &B) {
    Module &M = *A.getParent()->getParent();
    LLVMContext &Ctx = M.getContext();
    auto *Params = M.getNamedGlobal(SASS::KernelParamsSymbol);
    if (!Params)
      Params = new GlobalVariable(M, Type::getInt8Ty(Ctx), /*isConstant=*/true,
                                  GlobalValue::ExternalLinkage, nullptr,
                                  SASS::KernelParamsSymbol, nullptr,
                                  GlobalValue::NotThreadLocal, /*AddressSpace=*/4);
    Constant *Addr = ConstantExpr::getGetElementPtr(
        Type::getInt8Ty(Ctx), Params,
        ConstantInt::get(Type::getInt32Ty(Ctx), ParamBase + S.Offset));
    if (onlyRead(&A)) {
      // (pointers loaded from it are global: SASSTargetMachine::
      // getAssumedAddrSpace)
      A.replaceAllUsesWith(ConstantExpr::getAddrSpaceCast(Addr, A.getType()));
      return;
    }
    Type *Ty = A.getParamByValType();
    Align Al(S.Align);
    AllocaInst *Copy = B.CreateAlloca(Ty, nullptr, A.getName() + ".copy");
    Copy->setAlignment(Al);
    A.replaceAllUsesWith(Copy);
    B.CreateMemCpy(Copy, Al, Addr, Al, S.Size);
  }
};
} // namespace

char SASSLowerKernelArgs::ID = 0;

INITIALIZE_PASS(SASSLowerKernelArgs, "sass-lower-kernel-args",
                "SASS lower kernel args", false, false)

FunctionPass *llvm::createSASSLowerKernelArgsPass() {
  return new SASSLowerKernelArgs();
}
