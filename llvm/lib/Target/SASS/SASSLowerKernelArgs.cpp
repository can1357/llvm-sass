//===-- SASSLowerKernelArgs.cpp - Kernel pointer args are global ----------===//
//
// CUDA kernel pointer parameters always point to global memory. Rewriting
// each use of `ptr %p` to `addrspacecast (addrspacecast %p to ptr
// addrspace(1)) to ptr` lets InferAddressSpaces turn the derived generic
// accesses into global ones.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class SASSLowerKernelArgs : public FunctionPass {
public:
  static char ID;
  SASSLowerKernelArgs() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS lower kernel args"; }

  bool runOnFunction(Function &F) override {
    if (F.getCallingConv() != CallingConv::PTX_Kernel || F.isDeclaration())
      return false;
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    bool Changed = false;
    for (Argument &A : F.args()) {
      auto *PT = dyn_cast<PointerType>(A.getType());
      if (!PT || PT->getAddressSpace() != 0 || A.use_empty())
        continue;
      Value *Global = B.CreateAddrSpaceCast(
          &A, PointerType::get(F.getContext(), 1), A.getName() + ".global");
      Value *Generic = B.CreateAddrSpaceCast(Global, PT);
      A.replaceUsesWithIf(Generic, [&](Use &U) { return U.getUser() != Global; });
      Changed = true;
    }
    return Changed;
  }
};
} // namespace

char SASSLowerKernelArgs::ID = 0;

INITIALIZE_PASS(SASSLowerKernelArgs, "sass-lower-kernel-args",
                "SASS lower kernel args", false, false)

FunctionPass *llvm::createSASSLowerKernelArgsPass() {
  return new SASSLowerKernelArgs();
}
