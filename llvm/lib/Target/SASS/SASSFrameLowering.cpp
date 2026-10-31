//===-- SASSFrameLowering.cpp - SASS frame lowering -----------------------===//

#include "SASSFrameLowering.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

void SASSFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  if (MF.getFrameInfo().hasStackObjects())
    report_fatal_error("SASS: stack objects are not supported in " +
                       MF.getName());

  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  bool UsesDesc = false;
  for (const MachineBasicBlock &B : MF)
    for (const MachineInstr &MI : B)
      UsesDesc |= MI.readsRegister(SASS::URD2, TRI);
  if (!UsesDesc)
    return;

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  BuildMI(MBB, MBB.begin(), DebugLoc(), TII.get(SASS::LDCU64), SASS::URD2)
      .addImm(SASS::CBankGlobalDesc)
      .addReg(SASS::PT)
      .addImm(0);
}
