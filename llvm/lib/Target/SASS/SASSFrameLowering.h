//===-- SASSFrameLowering.h - SASS frame lowering ---------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSFRAMELOWERING_H
#define LLVM_LIB_TARGET_SASS_SASSFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

/// Kernels have no stack. The prologue loads the default global-memory
/// descriptor into UR4:UR5, which every LDG/STG names as desc[UR4:UR5].
class SASSFrameLowering : public TargetFrameLowering {
public:
  SASSFrameLowering()
      : TargetFrameLowering(StackGrowsDown, Align(8), 0) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override {}

protected:
  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};

} // namespace llvm

#endif
