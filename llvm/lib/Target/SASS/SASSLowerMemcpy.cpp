//===-- SASSLowerMemcpy.cpp - Small memcpys as wide loads/stores -----------===//
//
// Struct copies such as `float4 a = b[i];` reach codegen as llvm.memcpy. The
// SelectionDAG expansion only uses legal scalar types (at most i64 here), so
// a 16-byte aligned copy would become two 64-bit loads/stores. This pass
// rewrites small constant-length copies into the widest naturally aligned
// <4 x i32>, <2 x i32> or integer accesses, which select to LDG/LDS/STS .128
// and .64.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
class SASSLowerMemcpy : public FunctionPass {
public:
  static char ID;
  SASSLowerMemcpy() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS lower small memcpy"; }
  bool runOnFunction(Function &F) override;
};
} // namespace

char SASSLowerMemcpy::ID = 0;

INITIALIZE_PASS(SASSLowerMemcpy, "sass-lower-memcpy", "SASS lower small memcpy",
                false, false)

FunctionPass *llvm::createSASSLowerMemcpyPass() { return new SASSLowerMemcpy(); }

/// Longest copy expanded inline, in accesses.
static constexpr uint64_t MaxAccesses = 64;

bool SASSLowerMemcpy::runOnFunction(Function &F) {
  SmallVector<MemCpyInst *, 16> Copies;
  for (Instruction &I : instructions(F))
    if (auto *MC = dyn_cast<MemCpyInst>(&I))
      if (isa<ConstantInt>(MC->getLength()) && !MC->isVolatile())
        Copies.push_back(MC);

  bool Changed = false;
  LLVMContext &Ctx = F.getContext();
  for (MemCpyInst *MC : Copies) {
    uint64_t Len = cast<ConstantInt>(MC->getLength())->getZExtValue();
    Align SrcA = MC->getSourceAlign().valueOrOne();
    Align DstA = MC->getDestAlign().valueOrOne();
    uint64_t Chunk = std::min<uint64_t>(std::min(SrcA, DstA).value(), 16);
    while (Chunk > 1 && Len % Chunk)
      Chunk /= 2;
    if (Len == 0 || Len / Chunk > MaxAccesses)
      continue;
    Type *Ty = Type::getIntNTy(Ctx, Chunk * 8);
    if (Chunk >= 8)
      Ty = FixedVectorType::get(Type::getInt32Ty(Ctx), Chunk / 4);
    IRBuilder<> B(MC);
    for (uint64_t Off = 0; Off < Len; Off += Chunk) {
      Value *S = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), MC->getSource(), Off);
      Value *D = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), MC->getDest(), Off);
      LoadInst *L = B.CreateAlignedLoad(Ty, S, commonAlignment(SrcA, Off));
      L->copyMetadata(*MC, {LLVMContext::MD_tbaa, LLVMContext::MD_alias_scope,
                            LLVMContext::MD_noalias});
      StoreInst *St = B.CreateAlignedStore(L, D, commonAlignment(DstA, Off));
      St->copyMetadata(*MC, {LLVMContext::MD_tbaa, LLVMContext::MD_alias_scope,
                             LLVMContext::MD_noalias});
    }
    MC->eraseFromParent();
    Changed = true;
  }
  return Changed;
}
