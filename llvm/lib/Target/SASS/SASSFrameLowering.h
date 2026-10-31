//===-- SASSFrameLowering.h - SASS frame lowering ---------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSFRAMELOWERING_H
#define LLVM_LIB_TARGET_SASS_SASSFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

/// The prologue loads the default global-memory descriptor into UR4:UR5,
/// which every LDG/STG names as desc[UR4:UR5]. Kernels with stack objects
/// (allocas, spill slots) also get a per-thread local-memory frame: R1 is
/// loaded with the initial stack pointer from constant bank 0 and lowered by
/// the frame size; objects are addressed as [R1+off]. The frame is never
/// realigned: the incoming stack pointer is 16-byte aligned, enough for the
/// widest access (LDL/STL.128).
class SASSFrameLowering : public TargetFrameLowering {
public:
  SASSFrameLowering()
      : TargetFrameLowering(StackGrowsDown, Align(16), 0, Align(1),
                            /*StackReal=*/false) {}

  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override {}
  void processFunctionBeforeFrameFinalized(
      MachineFunction &MF, RegScavenger *RS = nullptr) const override;

protected:
  bool hasFPImpl(const MachineFunction &MF) const override { return false; }
};

} // namespace llvm

#endif
