//===-- SASSSpeculateLoads.cpp - Loads through select trees as selects -----===//
//
// Hopper. A lambda that captures values by reference and picks one with
// nested conditionals,
//
//   auto sel = [&](int i) { return (i & 2) ? ((i & 1) ? a3 : a2)
//                                          : ((i & 1) ? a1 : a0); };
//
// reaches codegen as a load at a select tree of constant offsets into the
// closure (an alloca of pointers), then a load through the pointer it read.
// SROA cannot promote an alloca read at a variable offset, nor one read
// through a nested select of pointers, so the closure and the captured values
// stay in local memory (STL/LDL, and stack the register allocator then also
// spills into), where nvcc keeps them in registers. This pass rewrites
//
//   load (gep P, select(c, K1, K2))  ->  select(c, load (gep P, K1),
//                                                  load (gep P, K2))
//   load (select(c, P1, P2))         ->  select(c, load P1, load P2)
//
// recursively over select trees of at most MaxLeaves leaves, when every leaf
// address is dereferenceable and aligned for the load (so the speculated loads
// cannot fault), the first form for constant offsets into an alloca. SROA
// (run after it) then promotes the allocas and the loads become SELs.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {
class SASSSpeculateLoads : public FunctionPass {
public:
  static char ID;
  SASSSpeculateLoads() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SASS speculate loads through select trees";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
  bool runOnFunction(Function &F) override;
};
} // namespace

char SASSSpeculateLoads::ID = 0;

INITIALIZE_PASS(SASSSpeculateLoads, "sass-speculate-loads",
                "SASS speculate loads through select trees", false, false)

FunctionPass *llvm::createSASSSpeculateLoadsPass() {
  return new SASSSpeculateLoads();
}

/// Largest select tree expanded into loads.
static constexpr unsigned MaxLeaves = 8;

/// Collects the leaves of the select tree rooted at \p V (V itself when it is
/// not a select); false when there are more than MaxLeaves.
static bool selectLeaves(Value *V, SmallVectorImpl<Value *> &Leaves) {
  if (auto *S = dyn_cast<SelectInst>(V))
    return selectLeaves(S->getTrueValue(), Leaves) &&
           selectLeaves(S->getFalseValue(), Leaves);
  Leaves.push_back(V);
  return Leaves.size() <= MaxLeaves;
}

namespace {
struct Speculator {
  LoadInst &LI;
  const DataLayout &DL;
  AssumptionCache &AC;
  DominatorTree &DT;
  IRBuilder<> B;

  Speculator(LoadInst &LI, AssumptionCache &AC, DominatorTree &DT)
      : LI(LI), DL(LI.getDataLayout()), AC(AC), DT(DT), B(&LI) {}

  bool safe(Value *Ptr) const {
    return isDereferenceableAndAlignedPointer(
        Ptr, LI.getType(), LI.getAlign(), SimplifyQuery(DL, &DT, &AC, &LI));
  }

  LoadInst *load(Value *Ptr) {
    LoadInst *L = B.CreateAlignedLoad(LI.getType(), Ptr, LI.getAlign());
    L->copyMetadata(LI);
    return L;
  }

  /// load (select-tree of pointers) -> select-tree of loads.
  Value *viaPointers(Value *Ptr) {
    if (auto *S = dyn_cast<SelectInst>(Ptr))
      return B.CreateSelect(S->getCondition(), viaPointers(S->getTrueValue()),
                            viaPointers(S->getFalseValue()));
    return load(Ptr);
  }

  /// load (gep Base, ..., select-tree of constants at operand OpNo, ...) ->
  /// select-tree of loads at constant offsets.
  Value *viaIndex(GetElementPtrInst &GEP, unsigned OpNo, Value *Idx) {
    if (auto *S = dyn_cast<SelectInst>(Idx))
      return B.CreateSelect(S->getCondition(),
                            viaIndex(GEP, OpNo, S->getTrueValue()),
                            viaIndex(GEP, OpNo, S->getFalseValue()));
    return load(leafGEP(GEP, OpNo, Idx));
  }

  Value *leafGEP(GetElementPtrInst &GEP, unsigned OpNo, Value *Idx) {
    SmallVector<Value *, 4> Ops(GEP.idx_begin(), GEP.idx_end());
    Ops[OpNo - 1] = Idx;
    return B.CreateGEP(GEP.getSourceElementType(), GEP.getPointerOperand(),
                       Ops, "", GEP.getNoWrapFlags());
  }

  Value *run() {
    Value *Ptr = LI.getPointerOperand();
    SmallVector<Value *, MaxLeaves> Leaves;
    if (isa<SelectInst>(Ptr)) {
      if (!selectLeaves(Ptr, Leaves) || !all_of(Leaves, [&](Value *L) {
            return safe(L);
          }))
        return nullptr;
      return viaPointers(Ptr);
    }
    auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
    if (!GEP || !isa<AllocaInst>(getUnderlyingObject(GEP->getPointerOperand())))
      return nullptr;
    // The one variable index: a select tree of constants.
    unsigned OpNo = 0;
    for (unsigned I = 1, E = GEP->getNumOperands(); I < E; ++I) {
      if (isa<Constant>(GEP->getOperand(I)))
        continue;
      if (OpNo)
        return nullptr;
      OpNo = I;
    }
    if (!OpNo || !isa<SelectInst>(GEP->getOperand(OpNo)) ||
        !selectLeaves(GEP->getOperand(OpNo), Leaves) ||
        !all_of(Leaves, [](Value *L) { return isa<ConstantInt>(L); }))
      return nullptr;
    // Every leaf address must be inside the alloca (checked on a GEP built
    // in front of the load and deleted again when one is out of bounds).
    SmallVector<Instruction *, MaxLeaves> Probes;
    bool AllSafe = true;
    for (Value *L : Leaves) {
      Value *P = leafGEP(*GEP, OpNo, L);
      if (auto *I = dyn_cast<Instruction>(P))
        Probes.push_back(I);
      AllSafe &= safe(P);
    }
    for (Instruction *I : Probes)
      I->eraseFromParent();
    if (!AllSafe)
      return nullptr;
    return viaIndex(*GEP, OpNo, GEP->getOperand(OpNo));
  }
};
} // namespace

bool SASSSpeculateLoads::runOnFunction(Function &F) {
  AssumptionCache &AC =
      getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SmallVector<LoadInst *, 16> Loads;
  for (Instruction &I : instructions(F))
    if (auto *LI = dyn_cast<LoadInst>(&I); LI && LI->isSimple())
      Loads.push_back(LI);
  bool Changed = false;
  for (LoadInst *LI : Loads) {
    Value *Ptr = LI->getPointerOperand();
    Value *New = Speculator(*LI, AC, DT).run();
    if (!New)
      continue;
    New->takeName(LI);
    LI->replaceAllUsesWith(New);
    LI->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(Ptr);
    Changed = true;
  }
  return Changed;
}
