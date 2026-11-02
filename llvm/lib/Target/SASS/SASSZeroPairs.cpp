//===-- SASSZeroPairs.cpp - Zeroed registers as CS2R pairs ----------------===//
//
// Values zeroed for a loop (attention accumulators: 32 registers before the
// main loop) reach the allocator as one MOV of 0 and a copy of it into each
// accumulator, which the copies keep: MOV R31, 0x0 then 32 MOV Rn, R31.
// ptxas clears them in pairs with CS2R Rn, SRZ. After allocation (copies
// expanded), within each block:
//   1. a copy of a register known to hold 0 becomes MOV Rn, 0x0;
//   2. MOV R2k, 0x0 and MOV R2k+1, 0x0 with nothing between them that reads
//      or writes either register become one CS2R R2k, SRZ.
// sm_90.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace {
class SASSZeroPairs : public MachineFunctionPass {
public:
  static char ID;
  SASSZeroPairs() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS zero pairs"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // namespace

char SASSZeroPairs::ID = 0;

INITIALIZE_PASS(SASSZeroPairs, "sass-zero-pairs", "SASS zero pairs", false, false)

FunctionPass *llvm::createSASSZeroPairsPass() { return new SASSZeroPairs(); }

static bool isZeroMov(const MachineInstr &MI, const TargetInstrInfo &TII) {
  return MI.getOpcode() == SASS::MOVri && MI.getOperand(1).isImm() &&
         MI.getOperand(1).getImm() == 0 && !TII.isPredicated(MI);
}

bool SASSZeroPairs::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (ST.hasBlackwellISA())
    return false;
  const TargetInstrInfo &TII = *ST.getInstrInfo();
  const TargetRegisterInfo &TRI = *ST.getRegisterInfo();
  bool Changed = false;
  auto Guard = [](const MachineInstrBuilder &MIB) { MIB.addReg(SASS::PT).addImm(0); };
  for (MachineBasicBlock &MBB : MF) {
    // 1. Copies of known zeros.
    BitVector Zero(TRI.getNumRegs());
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (isZeroMov(MI, TII)) {
        Zero.set(MI.getOperand(0).getReg());
        continue;
      }
      if (MI.getOpcode() == SASS::MOVrr && !TII.isPredicated(MI) &&
          MI.getOperand(1).getReg().isPhysical() && Zero.test(MI.getOperand(1).getReg())) {
        Register D = MI.getOperand(0).getReg();
        Guard(BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(SASS::MOVri), D).addImm(0));
        MI.eraseFromParent();
        Zero.set(D);
        Changed = true;
        continue;
      }
      if (MI.isCall()) {
        Zero.reset();
        continue;
      }
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isRegMask()) {
          Zero.reset();
          break;
        }
        if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical())
          for (MCRegAliasIterator A(MO.getReg(), &TRI, true); A.isValid(); ++A)
            Zero.reset(*A);
      }
    }
    // 2. Halves of a pair, the second moved up to the first: nothing between
    // them may touch either register.
    for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
      MachineInstr &A = *I++;
      if (!isZeroMov(A, TII))
        continue;
      Register RA = A.getOperand(0).getReg();
      MCRegister Pair = TRI.getMatchingSuperReg(RA, SASS::sub0, &SASS::GPR64RegClass);
      Register Want = Pair ? Register(TRI.getSubReg(Pair, SASS::sub1)) : Register();
      if (!Pair) {
        Pair = TRI.getMatchingSuperReg(RA, SASS::sub1, &SASS::GPR64RegClass);
        Want = Pair ? Register(TRI.getSubReg(Pair, SASS::sub0)) : Register();
      }
      if (!Pair)
        continue;
      unsigned Seen = 0;
      for (auto J = I; J != E && Seen < 32; ++J, ++Seen) {
        MachineInstr &B = *J;
        if (isZeroMov(B, TII) && B.getOperand(0).getReg() == Want) {
          Guard(BuildMI(MBB, A, A.getDebugLoc(), TII.get(SASS::CS2R_SRZ), Pair));
          if (I == J)
            ++I;
          B.eraseFromParent();
          A.eraseFromParent();
          Changed = true;
          break;
        }
        if (B.readsRegister(RA, &TRI) || B.modifiesRegister(RA, &TRI) ||
            B.readsRegister(Want, &TRI) || B.modifiesRegister(Want, &TRI) ||
            B.isTerminator() || B.hasUnmodeledSideEffects() || B.isCall())
          break;
      }
    }
  }
  return Changed;
}
