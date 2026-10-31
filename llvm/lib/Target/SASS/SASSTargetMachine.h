//===-- SASSTargetMachine.h - SASS target machine ---------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSTARGETMACHINE_H
#define LLVM_LIB_TARGET_SASS_SASSTARGETMACHINE_H

#include "SASSSubtarget.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include <optional>

namespace llvm {

class SASSTargetMachine : public CodeGenTargetMachineImpl {
  std::unique_ptr<TargetLoweringObjectFile> TLOF;
  SASSSubtarget Subtarget;

public:
  SASSTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, const TargetOptions &Options,
                    std::optional<Reloc::Model> RM,
                    std::optional<CodeModel::Model> CM, CodeGenOptLevel OL,
                    bool JIT);
  ~SASSTargetMachine() override;

  const SASSSubtarget *getSubtargetImpl() const { return &Subtarget; }
  const SASSSubtarget *getSubtargetImpl(const Function &) const override {
    return &Subtarget;
  }

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;
  TargetTransformInfo getTargetTransformInfo(const Function &F) const override;
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
  MachineFunctionInfo *
  createMachineFunctionInfo(BumpPtrAllocator &Allocator, const Function &F,
                            const TargetSubtargetInfo *STI) const override;

  /// Global memory is identity-mapped into the generic address space.
  bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const override {
    return SrcAS <= 1 && DestAS <= 1;
  }
};

} // namespace llvm

#endif
