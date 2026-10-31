//===-- SASSFrameLowering.cpp - SASS frame lowering -----------------------===//

#include "SASSFrameLowering.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

void SASSFrameLowering::emitPrologue(MachineFunction &MF,
                                     MachineBasicBlock &MBB) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.hasVarSizedObjects())
    report_fatal_error("SASS: dynamically sized stack objects are not "
                       "supported in " + MF.getName());

  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  const TargetInstrInfo &TII = *ST.getInstrInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  bool UsesDesc = false;
  for (const MachineBasicBlock &B : MF)
    for (const MachineInstr &MI : B)
      UsesDesc |= MI.readsRegister(SASS::URD2, TRI);
  // Blackwell loads constants into uniform registers with LDCU (scoreboarded);
  // Hopper has only the fixed-latency ULDC.
  if (UsesDesc)
    BuildMI(MBB, MBB.begin(), DebugLoc(),
            TII.get(ST.hasBlackwellISA() ? SASS::LDCU64 : SASS::ULDC64),
            SASS::URD2)
        .addImm(ST.globalDescOffset())
        .addReg(SASS::PT)
        .addImm(0);

  // R1 = initial stack pointer - frame size, ahead of everything else.
  if (uint64_t Size = MFI.getStackSize()) {
    if (!isInt<24>(Size))
      report_fatal_error("SASS: stack frame of " + MF.getName() +
                         " is too large");
    auto I = BuildMI(MBB, MBB.begin(), DebugLoc(), TII.get(SASS::IADDri),
                     SASS::R1)
                 .addReg(SASS::R1)
                 .addImm(-int64_t(Size))
                 .addReg(SASS::PT)
                 .addImm(0);
    BuildMI(MBB, I.getInstr(), DebugLoc(), TII.get(SASS::LDC32), SASS::R1)
        .addImm(ST.stackPtrOffset())
        .addReg(SASS::PT)
        .addImm(0);
  }
}

// Predicate and uniform-register spills are expanded through a GPR that the
// register scavenger finds; give it a slot of its own for when none is free.
void SASSFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  if (!RS)
    return;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      switch (MI.getOpcode()) {
      case SASS::SPILL_P:
      case SASS::SPILL_UR32:
      case SASS::SPILL_UR64:
      case SASS::RELOAD_P:
      case SASS::RELOAD_UR32:
      case SASS::RELOAD_UR64:
        RS->addScavengingFrameIndex(
            MF.getFrameInfo().CreateSpillStackObject(4, Align(4)));
        return;
      }
}
