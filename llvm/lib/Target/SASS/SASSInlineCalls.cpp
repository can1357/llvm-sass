//===-- SASSInlineCalls.cpp - Inline every device function call ----------===//
//
// The backend has no call ABI: every `__device__` function (including
// `__noinline__` ones) is inlined into the kernels that call it, bottom-up,
// and the then-dead device functions are deleted. A CUDA translation unit is
// the whole device program, so no device function needs an external
// definition. Recursion, indirect calls and calls to functions without a
// body (printf, libm, ...) cannot be inlined and are reported here, naming
// the kernel and callee.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace {

class SASSInlineCalls : public ModulePass {
public:
  static char ID;
  SASSInlineCalls() : ModulePass(ID) {}
  StringRef getPassName() const override { return "SASS inline calls"; }
  bool runOnModule(Module &M) override;
};

bool isKernel(const Function &F) {
  return F.getCallingConv() == CallingConv::PTX_Kernel;
}

/// Calls in \p F that must be inlined; diagnoses the ones that cannot be.
SmallVector<CallBase *, 8> callsToInline(Function &F) {
  SmallVector<CallBase *, 8> Calls;
  for (Instruction &I : instructions(F)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || isa<IntrinsicInst>(CB) || CB->isInlineAsm())
      continue;
    Function *Callee = CB->getCalledFunction();
    if (!Callee)
      report_fatal_error("SASS: indirect call in '" + F.getName() +
                         "': function pointers are not supported");
    if (Callee->isIntrinsic())
      continue;
    if (Callee->isDeclaration())
      report_fatal_error("SASS: '" + F.getName() + "' calls '" +
                         Callee->getName() +
                         "', which has no device definition (library calls "
                         "such as printf are not supported)");
    if (isKernel(*Callee))
      report_fatal_error("SASS: '" + F.getName() + "' calls kernel '" +
                         Callee->getName() + "'");
    Calls.push_back(CB);
  }
  return Calls;
}

/// Reports a call cycle reachable from \p F (DFS over direct callees).
void checkRecursion(Function &F, SmallVectorImpl<Function *> &Stack,
                    SmallPtrSetImpl<Function *> &Done) {
  if (Done.contains(&F))
    return;
  if (is_contained(Stack, &F)) {
    std::string Cycle;
    for (Function *G : make_range(find(Stack, &F), Stack.end()))
      Cycle += G->getName().str() + " -> ";
    report_fatal_error("SASS: recursive device functions are not supported "
                       "(" + Twine(Cycle) + F.getName() + ")");
  }
  Stack.push_back(&F);
  for (CallBase *CB : callsToInline(F))
    checkRecursion(*CB->getCalledFunction(), Stack, Done);
  Stack.pop_back();
  Done.insert(&F);
}

/// Inlines all calls in \p F after inlining its callees' calls (bottom-up,
/// so each callee body is flattened once).
void inlineAll(Function &F, SmallPtrSetImpl<Function *> &Flat) {
  if (!Flat.insert(&F).second)
    return;
  for (CallBase *CB : callsToInline(F)) {
    inlineAll(*CB->getCalledFunction(), Flat);
    InlineFunctionInfo IFI;
    InlineResult R = InlineFunction(*CB, IFI);
    if (!R.isSuccess())
      report_fatal_error("SASS: cannot inline '" +
                         CB->getCalledFunction()->getName() + "' into '" +
                         F.getName() + "': " + R.getFailureReason());
  }
}

} // namespace

bool SASSInlineCalls::runOnModule(Module &M) {
  SmallVector<Function *, 8> Stack;
  SmallPtrSet<Function *, 16> Done, Flat;
  bool HasDeviceFns = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    HasDeviceFns |= !isKernel(F);
    checkRecursion(F, Stack, Done);
  }
  if (!HasDeviceFns)
    return false;
  for (Function &F : M)
    if (!F.isDeclaration() && isKernel(F))
      inlineAll(F, Flat);
  // Device functions are not entry points: drop them once inlined (and any
  // that are never called), so codegen sees kernels only.
  SetVector<Function *> Dead;
  for (Function &F : M)
    if (!F.isDeclaration() && !isKernel(F))
      Dead.insert(&F);
  for (Function *F : Dead)
    F->dropAllReferences();
  for (Function *F : Dead) {
    F->replaceAllUsesWith(PoisonValue::get(F->getType()));
    F->eraseFromParent();
  }
  return true;
}

char SASSInlineCalls::ID = 0;

INITIALIZE_PASS(SASSInlineCalls, "sass-inline-calls", "SASS inline calls",
                false, false)

ModulePass *llvm::createSASSInlineCallsPass() { return new SASSInlineCalls(); }
