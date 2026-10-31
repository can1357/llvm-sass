//===-- SASSTargetTransformInfo.h - SASS TTI --------------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_SASS_SASSTARGETTRANSFORMINFO_H

#include "SASSTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

namespace llvm {

class SASSTTIImpl final : public BasicTTIImplBase<SASSTTIImpl> {
  using BaseT = BasicTTIImplBase<SASSTTIImpl>;
  friend BaseT;

  const SASSSubtarget *ST;
  const SASSTargetLowering *TLI;

  const SASSSubtarget *getST() const { return ST; }
  const SASSTargetLowering *getTLI() const { return TLI; }

public:
  SASSTTIImpl(const SASSTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getDataLayout()), ST(TM->getSubtargetImpl()),
        TLI(ST->getTargetLowering()) {}

  bool hasBranchDivergence(const Function *F = nullptr) const override {
    return true;
  }
  /// Address space 0 is CUDA's generic space.
  unsigned getFlatAddressSpace() const override { return 0; }

  // The load/store vectorizer may form naturally aligned 64/128-bit accesses
  // (LDG/LDS/LD .64/.128).
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const override {
    return 128;
  }
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes, Align Alignment,
                                   unsigned AddrSpace) const override {
    return Alignment >= ChainSizeInBytes;
  }
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes, Align Alignment,
                                    unsigned AddrSpace) const override {
    return Alignment >= ChainSizeInBytes;
  }

  /// Unrolls loops with a known trip count by two: SMs issue in order, so
  /// only the pre-RA scheduler (SASSPreRASched) can fill one iteration's
  /// dependent-latency gaps (a dot-product chain, a softmax) with the other
  /// iteration's independent work. Larger factors run out of the 255
  /// registers on the loops that need it most.
  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TargetTransformInfo::UnrollingPreferences &UP,
                               OptimizationRemarkEmitter *ORE) const override {
    BaseT::getUnrollingPreferences(L, SE, UP, ORE);
    UP.Partial = true;
    UP.Runtime = false;
    UP.MaxCount = 2;
    UP.PartialThreshold = 2048; // IR instructions in the unrolled body.
  }
};

} // namespace llvm

#endif
