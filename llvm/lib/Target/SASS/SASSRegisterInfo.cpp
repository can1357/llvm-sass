//===-- SASSRegisterInfo.cpp - SASS register information ------------------===//

#include "SASSRegisterInfo.h"
#include "SASS.h"
#include "SASSFrameLowering.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "SASSGenRegisterInfo.inc"

SASSRegisterInfo::SASSRegisterInfo() : SASSGenRegisterInfo(SASS::R1) {}

const MCPhysReg *
SASSRegisterInfo::getCalleeSavedRegs(const MachineFunction *) const {
  static const MCPhysReg None[] = {0};
  return None;
}

BitVector SASSRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  // R1 is the ABI stack pointer; zero/true registers are not allocatable.
  for (MCPhysReg R : {SASS::R1, SASS::RZ, SASS::PT, SASS::URZ})
    markSuperRegs(Reserved, R);
  // UR4:UR5 hold the global memory descriptor (SASSFrameLowering).
  for (MCPhysReg R : {SASS::UR4, SASS::UR5})
    markSuperRegs(Reserved, R);
  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool SASSRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator,
                                           int, unsigned,
                                           RegScavenger *) const {
  report_fatal_error("SASS: stack frames (spills, allocas) are not supported");
}

Register SASSRegisterInfo::getFrameRegister(const MachineFunction &) const {
  return SASS::R1;
}
