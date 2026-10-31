//===-- SASSRegisterInfo.cpp - SASS register information ------------------===//

#include "SASSRegisterInfo.h"
#include "SASS.h"
#include "SASSFrameLowering.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_REGINFO_TARGET_DESC
#include "SASSGenRegisterInfo.inc"

unsigned llvm::SASS::maxRegCount(const Function &F) {
  // "nvvm.maxntid" is "x[,y[,z]]"; its product is the block size.
  uint64_t Threads = 0;
  if (Attribute A = F.getFnAttribute("nvvm.maxntid"); A.isStringAttribute()) {
    Threads = 1;
    SmallVector<StringRef, 3> Dims;
    A.getValueAsString().split(Dims, ',');
    for (StringRef D : Dims) {
      uint64_t N;
      if (D.trim().getAsInteger(10, N) || !N)
        report_fatal_error("SASS: " + F.getName() + ": malformed nvvm.maxntid");
      Threads *= N;
    }
  }
  uint64_t Limit = 255;
  if (Threads) {
    // As ptxas: without minBlocksPerSM, one block of maxThreads must fit.
    uint64_t Blocks = F.getFnAttributeAsParsedInteger("nvvm.minctasm", 1);
    // Each of the 4 SM sub-partitions owns 16K registers and its warps' share
    // is allocated in units of 256 (8 per thread); a block's warps are spread
    // over the sub-partitions, so the most loaded one holds
    // ceil(Blocks * Warps / 4) of them.
    uint64_t Warps = std::max<uint64_t>(Blocks, 1) * divideCeil(Threads, 32);
    Limit = 16384 / (divideCeil(Warps, 4) * 32) / 8 * 8;
  }
  // __maxnreg__ / .maxnreg.
  if (uint64_t N = F.getFnAttributeAsParsedInteger("nvvm.maxnreg", 0))
    Limit = std::min(Limit, N);
  return std::clamp<uint64_t>(Limit, 8, 255);
}

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
  // The reported register count is the highest GPR + 3
  // (SASSAsmPrinter::regCount); keep it within the kernel's limit (or a
  // setmaxnreg.inc target, whose regions SASSWGMMAHazards checks).
  for (unsigned R = SASS::setMaxRegLimit(MF) - 2; R <= 254; ++R)
    markSuperRegs(Reserved, SASS::GPR32RegClass.getRegister(R));
  assert(checkAllSuperRegsMarked(Reserved));
  return Reserved;
}

bool SASSRegisterInfo::requiresRegisterScavenging(
    const MachineFunction &MF) const {
  return MF.getFrameInfo().hasStackObjects();
}

bool SASSRegisterInfo::requiresFrameIndexScavenging(
    const MachineFunction &MF) const {
  return MF.getFrameInfo().hasStackObjects();
}

// Every frame index operand is the base of a (base, imm) pair: an LDL/STL
// address, IADDri's operands or IADD3rri's last two. The frame lies at
// [R1, R1 + stack size) after the prologue.
bool SASSRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                           int, unsigned FIOperandNum,
                                           RegScavenger *) const {
  MachineInstr &MI = *II;
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  int FI = MI.getOperand(FIOperandNum).getIndex();
  MachineOperand &OffOp = MI.getOperand(FIOperandNum + 1);
  int64_t Off =
      MFI.getObjectOffset(FI) + int64_t(MFI.getStackSize()) + OffOp.getImm();
  if (!isInt<24>(Off))
    report_fatal_error("SASS: stack offset out of range in " + MF.getName());
  MI.getOperand(FIOperandNum).ChangeToRegister(SASS::R1, false);
  OffOp.setImm(Off);

  const DebugLoc &DL = MI.getDebugLoc();
  auto Build = [&](unsigned Opc) { return BuildMI(MBB, II, DL, TII.get(Opc)); };
  auto Guard = [](MachineInstrBuilder MIB) { MIB.addReg(SASS::PT).addImm(0); };
  auto Tmp = [&] { return MRI.createVirtualRegister(&SASS::GPR32RegClass); };
  // LDL/STL of a scavenged GPR at R1 + Off + Disp.
  auto Store = [&](unsigned Opc, Register R, int64_t Disp) {
    auto MIB = Build(Opc).addReg(SASS::R1).addImm(Off + Disp).addReg(
        R, RegState::Kill);
    Guard(MIB);
    MIB.cloneMemRefs(MI);
  };
  auto Load = [&](unsigned Opc, Register R, int64_t Disp) {
    auto MIB = BuildMI(MBB, II, DL, TII.get(Opc), R)
                   .addReg(SASS::R1)
                   .addImm(Off + Disp);
    Guard(MIB);
    MIB.cloneMemRefs(MI);
  };

  // wgmma accumulator tuples: one STL.128/LDL.128 per 4-aligned quad.
  bool IsGacc = false;
  unsigned Opc = MI.getOpcode();
#define GACC_CASE(S)                                                           \
  IsGacc |= Opc == SASS::SPILL_GACC##S || Opc == SASS::RELOAD_GACC##S;
  GACC_CASE(8) GACC_CASE(12) GACC_CASE(16) GACC_CASE(20) GACC_CASE(24)
  GACC_CASE(28) GACC_CASE(32) GACC_CASE(36) GACC_CASE(40) GACC_CASE(44)
  GACC_CASE(48) GACC_CASE(52) GACC_CASE(56) GACC_CASE(60) GACC_CASE(64)
#undef GACC_CASE
  if (IsGacc) {
    bool IsStore = MI.mayStore();
    const MachineOperand &RegOp = MI.getOperand(IsStore ? 2 : 0);
    Register Tuple = RegOp.getReg();
    unsigned First = getEncodingValue(getSubReg(Tuple, SASS::sub0));
    unsigned Lanes = getRegSizeInBits(Tuple, MRI) / 32;
    for (unsigned Q = 0; Q < Lanes; Q += 4) {
      MCRegister Quad = getMatchingSuperReg(
          SASS::GPR32RegClass.getRegister(First + Q), SASS::sub0,
          &SASS::GPR128RegClass);
      if (IsStore) {
        auto MIB = Build(SASS::STL128).addReg(SASS::R1).addImm(Off + 4 * Q).addReg(
            Quad, getKillRegState(RegOp.isKill()));
        Guard(MIB);
        MIB.cloneMemRefs(MI);
      } else {
        Load(SASS::LDL128, Quad, 4 * Q);
      }
    }
    MI.eraseFromParent();
    return true;
  }

  switch (MI.getOpcode()) {
  case SASS::IADDri:
    // A frame object's address: MOV Rd, R1 when at the stack pointer.
    if (Off == 0) {
      MI.setDesc(TII.get(SASS::MOVrr));
      MI.removeOperand(FIOperandNum + 1);
    }
    return false;
  case SASS::SPILL_P: {
    // Predicate as a 0/1 byte.
    const MachineOperand &Src = MI.getOperand(2);
    Register T = Tmp();
    Guard(BuildMI(MBB, II, DL, TII.get(SASS::SELri), T)
              .addReg(SASS::RZ)
              .addImm(1)
              .addReg(Src.getReg(), getKillRegState(Src.isKill()))
              .addImm(1));
    Store(SASS::STLU8, T, 0);
    break;
  }
  case SASS::RELOAD_P: {
    Register T = Tmp();
    Load(SASS::LDLU8, T, 0);
    Guard(BuildMI(MBB, II, DL, TII.get(SASS::ISETPri),
                  MI.getOperand(0).getReg())
              .addReg(T, RegState::Kill)
              .addImm(0)
              .addImm(5) // NE
              .addReg(SASS::PT)
              .addImm(0));
    break;
  }
  case SASS::SPILL_UR32:
  case SASS::SPILL_UR64: {
    const MachineOperand &Src = MI.getOperand(2);
    bool Wide = MI.getOpcode() == SASS::SPILL_UR64;
    for (unsigned H = 0; H < (Wide ? 2u : 1u); ++H) {
      Register T = Tmp();
      Register UR = Src.getReg();
      if (Wide)
        UR = getSubReg(UR, H ? SASS::sub1 : SASS::sub0);
      Guard(BuildMI(MBB, II, DL, TII.get(SASS::MOVru), T)
                .addReg(UR, getKillRegState(Src.isKill())));
      Store(SASS::STL32, T, 4 * H);
    }
    break;
  }
  case SASS::RELOAD_UR32:
  case SASS::RELOAD_UR64: {
    Register Dst = MI.getOperand(0).getReg();
    bool Wide = MI.getOpcode() == SASS::RELOAD_UR64;
    for (unsigned H = 0; H < (Wide ? 2u : 1u); ++H) {
      Register T = Tmp();
      Register UR = Dst;
      if (Wide)
        UR = getSubReg(UR, H ? SASS::sub1 : SASS::sub0);
      Load(SASS::LDL32, T, 4 * H);
      Guard(BuildMI(MBB, II, DL, TII.get(SASS::R2UR), UR)
                .addReg(T, RegState::Kill));
    }
    break;
  }
  default:
    return false;
  }
  MI.eraseFromParent();
  return true;
}

Register SASSRegisterInfo::getFrameRegister(const MachineFunction &) const {
  return SASS::R1;
}
