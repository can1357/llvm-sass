//===-- SASSFrameLowering.cpp - SASS frame lowering -----------------------===//

#include "SASSFrameLowering.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSSubtarget.h"
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

  // Blackwell loads constants into uniform registers with LDCU (scoreboarded);
  // Hopper has only the fixed-latency ULDC.
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  const TargetInstrInfo &TII = *ST.getInstrInfo();
  unsigned Opc = ST.hasBlackwellISA() ? SASS::LDCU64 : SASS::ULDC64;
  BuildMI(MBB, MBB.begin(), DebugLoc(), TII.get(Opc), SASS::URD2)
      .addImm(ST.globalDescOffset())
      .addReg(SASS::PT)
      .addImm(0);
}
