//===-- SASSMemAlign.cpp - Alignment and element types of memory accesses --===//
//
// Two IR passes around the load/store vectorizer (Hopper only):
//
// sass-infer-align raises each load's and store's alignment to what the known
// bits of its address prove. Clang marks a scalar access with its type's
// alignment, so the four floats `S.cs[4 * i + e]` of a 1024-byte aligned
// __shared__ window are each "align 4"; the vectorizer only joins accesses
// whose first element is aligned to the vector size (LDS.128 needs 16), and,
// unlike the full optimization pipeline's InferAlignment, codegen has nothing
// that derives it from the address arithmetic (window base, lane * 16).
//
// sass-wide-elts retypes the 8- and 16-byte vector accesses with 8- or 16-bit
// elements (a `uint4` of bf16 pairs the vectorizer joins as <8 x bfloat>) to
// <2 x i32> / <4 x i32>: the SelectionDAG type legalizer splits an illegal
// <8 x bfloat> into four 32-bit loads, while the i32 vectors are one
// LDG/LDS/STS .64/.128 (SASSISelLowering's wide accesses).
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

namespace {
class SASSInferAlign : public FunctionPass {
public:
  static char ID;
  SASSInferAlign() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SASS infer memory access alignment";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
  bool runOnFunction(Function &F) override;
};

class SASSWideElts : public FunctionPass {
public:
  static char ID;
  SASSWideElts() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SASS 32-bit elements for narrow-element vector accesses";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
  bool runOnFunction(Function &F) override;
};
} // namespace

char SASSInferAlign::ID = 0;
char SASSWideElts::ID = 0;

INITIALIZE_PASS(SASSInferAlign, "sass-infer-align",
                "SASS infer memory access alignment", false, false)
INITIALIZE_PASS(SASSWideElts, "sass-wide-elts",
                "SASS 32-bit elements for narrow-element vector accesses",
                false, false)

FunctionPass *llvm::createSASSInferAlignPass() { return new SASSInferAlign(); }
FunctionPass *llvm::createSASSWideEltsPass() { return new SASSWideElts(); }

/// Beyond 16 bytes no access gets wider.
static const Align MaxUsefulAlign(16);

bool SASSInferAlign::runOnFunction(Function &F) {
  const DataLayout &DL = F.getDataLayout();
  AssumptionCache &AC =
      getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  bool Changed = false;
  for (Instruction &I : instructions(F)) {
    Value *Ptr = getLoadStorePointerOperand(&I);
    if (!Ptr)
      continue;
    Align Old = getLoadStoreAlignment(&I);
    if (Old >= MaxUsefulAlign)
      continue;
    Align Known = std::min(getKnownAlignment(Ptr, DL, &I, &AC, &DT),
                           MaxUsefulAlign);
    if (Known <= Old)
      continue;
    if (auto *LI = dyn_cast<LoadInst>(&I))
      LI->setAlignment(Known);
    else
      cast<StoreInst>(I).setAlignment(Known);
    Changed = true;
  }
  return Changed;
}

/// <2 x i32> / <4 x i32> for an 8- or 16-byte, naturally aligned access of a
/// vector with 8- or 16-bit elements; null otherwise.
static FixedVectorType *wideType(Type *Ty, Align A) {
  auto *VT = dyn_cast<FixedVectorType>(Ty);
  if (!VT)
    return nullptr;
  unsigned EltBits = VT->getElementType()->getPrimitiveSizeInBits();
  unsigned Bytes = VT->getNumElements() * EltBits / 8;
  if ((EltBits != 8 && EltBits != 16) || (Bytes != 8 && Bytes != 16) ||
      A.value() < Bytes)
    return nullptr;
  return FixedVectorType::get(Type::getInt32Ty(Ty->getContext()), Bytes / 4);
}

bool SASSWideElts::runOnFunction(Function &F) {
  SmallVector<Instruction *, 16> Work;
  for (Instruction &I : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->isSimple() && wideType(LI->getType(), LI->getAlign()))
        Work.push_back(LI);
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->isSimple() &&
          wideType(SI->getValueOperand()->getType(), SI->getAlign()))
        Work.push_back(SI);
    }
  }
  for (Instruction *I : Work) {
    IRBuilder<> B(I);
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      LoadInst *NL = B.CreateAlignedLoad(wideType(LI->getType(), LI->getAlign()),
                                         LI->getPointerOperand(),
                                         LI->getAlign());
      NL->copyMetadata(*LI);
      NL->takeName(LI);
      LI->replaceAllUsesWith(B.CreateBitCast(NL, LI->getType()));
    } else {
      auto *SI = cast<StoreInst>(I);
      Value *V = SI->getValueOperand();
      StoreInst *NS = B.CreateAlignedStore(
          B.CreateBitCast(V, wideType(V->getType(), SI->getAlign())),
          SI->getPointerOperand(), SI->getAlign());
      NS->copyMetadata(*SI);
    }
    I->eraseFromParent();
  }
  return !Work.empty();
}
