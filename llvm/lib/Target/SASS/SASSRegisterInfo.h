//===-- SASSRegisterInfo.h - SASS register information ----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSREGISTERINFO_H
#define LLVM_LIB_TARGET_SASS_SASSREGISTERINFO_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "SASSGenRegisterInfo.inc"

namespace llvm {

class SASSRegisterInfo : public SASSGenRegisterInfo {
public:
  SASSRegisterInfo();

  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  /// Predicate/uniform-register spills are expanded through scavenged GPRs.
  bool requiresRegisterScavenging(const MachineFunction &MF) const override;
  bool requiresFrameIndexScavenging(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;
  Register getFrameRegister(const MachineFunction &MF) const override;
};

} // namespace llvm

#endif
