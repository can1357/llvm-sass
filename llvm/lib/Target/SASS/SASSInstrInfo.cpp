//===-- SASSInstrInfo.cpp - SASS instruction information -------------------===//

#include "SASSInstrInfo.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define GET_INSTRINFO_CTOR_DTOR
#include "SASSGenInstrInfo.inc"

SASSInstrInfo::SASSInstrInfo(const SASSSubtarget &ST)
    : SASSGenInstrInfo(ST, RI), RI() {}

/// Appends the always-true guard (@PT).
static const MachineInstrBuilder &noGuard(const MachineInstrBuilder &MIB) {
  return MIB.addReg(SASS::PT).addImm(0);
}

void SASSInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                const DebugLoc &DL, Register Dst, Register Src,
                                bool KillSrc, bool, bool) const {
  auto Copy = [&](unsigned Opc, Register D, Register S, bool Kill) {
    noGuard(BuildMI(MBB, I, DL, get(Opc), D).addReg(S, getKillRegState(Kill)));
  };
  if (SASS::GPR32RegClass.contains(Dst, Src))
    return Copy(SASS::MOVrr, Dst, Src, KillSrc);
  if (SASS::GPR64RegClass.contains(Dst, Src))
    return Copy(SASS::MOV64rr, Dst, Src, KillSrc);
  if (SASS::GPR128RegClass.contains(Dst, Src)) {
    for (unsigned Idx : {SASS::sub0, SASS::sub1, SASS::sub2, SASS::sub3})
      Copy(SASS::MOVrr, RI.getSubReg(Dst, Idx), RI.getSubReg(Src, Idx),
           KillSrc);
    return;
  }
  if (SASS::UGPR32RegClass.contains(Dst, Src))
    return Copy(SASS::UMOV, Dst, Src, KillSrc);
  if (SASS::UGPR32RegClass.contains(Dst) && SASS::GPR32RegClass.contains(Src))
    return Copy(SASS::R2UR, Dst, Src, KillSrc);
  if (SASS::GPR32RegClass.contains(Dst) && SASS::UGPR32RegClass.contains(Src))
    return Copy(SASS::MOVru, Dst, Src, KillSrc);
  if (SASS::PREDRegClass.contains(Dst, Src)) {
    noGuard(BuildMI(MBB, I, DL, get(SASS::PMOV), Dst)
                .addReg(Src, getKillRegState(KillSrc))
                .addImm(0));
    return;
  }
  report_fatal_error("SASS: unsupported register copy");
}

void SASSInstrInfo::expandAdd64(MachineInstr &MI) const {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const DebugLoc &DL = MI.getDebugLoc();
  unsigned LoOpc, HiOpc;
  switch (MI.getOpcode()) {
  case SASS::IADD64rr: LoOpc = SASS::IADD3COrr; HiOpc = SASS::IADD3Xrr; break;
  case SASS::IADD64ri: LoOpc = SASS::IADD3COri; HiOpc = SASS::IADD3Xri; break;
  case SASS::ISUB64rr: LoOpc = SASS::IADD3COrn; HiOpc = SASS::IADD3Xrn; break;
  default: llvm_unreachable("not a 64-bit add");
  }
  Register A = MI.getOperand(1).getReg();
  const MachineOperand &B = MI.getOperand(2);
  Register Lo = MRI.createVirtualRegister(&SASS::GPR32RegClass);
  Register Hi = MRI.createVirtualRegister(&SASS::GPR32RegClass);
  Register Carry = MRI.createVirtualRegister(&SASS::PREDRegClass);
  auto L = BuildMI(MBB, MI, DL, get(LoOpc), Lo).addDef(Carry).addReg(A, {}, SASS::sub0);
  auto H = BuildMI(MBB, MI, DL, get(HiOpc), Hi).addReg(A, {}, SASS::sub1);
  if (B.isImm()) {
    // The immediate is sign-extended: its high word is 0 or -1.
    L.addImm(B.getImm());
    H.addImm(B.getImm() < 0 ? -1 : 0);
  } else {
    L.addReg(B.getReg(), {}, SASS::sub0);
    H.addReg(B.getReg(), {}, SASS::sub1);
  }
  noGuard(L);
  noGuard(H.addReg(Carry));
  BuildMI(MBB, MI, DL, get(TargetOpcode::REG_SEQUENCE), MI.getOperand(0).getReg())
      .addReg(Lo)
      .addImm(SASS::sub0)
      .addReg(Hi)
      .addImm(SASS::sub1);
  MRI.clearKillFlags(A);
  if (B.isReg())
    MRI.clearKillFlags(B.getReg());
  MI.eraseFromParent();
}

static bool isUncondBranch(unsigned Opc) { return Opc == SASS::BRA; }
static bool isCondBranch(unsigned Opc) { return Opc == SASS::BRAc; }

/// Guard operand index of \p MI (predicate register; the negate flag follows).
static unsigned guardIdx(const MachineInstr &MI) {
  int Idx = MI.findFirstPredOperandIdx();
  assert(Idx >= 0 && "SASS instructions all carry a guard");
  return Idx;
}

bool SASSInstrInfo::analyzeBranch(MachineBasicBlock &MBB,
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  SmallVectorImpl<MachineOperand> &Cond,
                                  bool AllowModify) const {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end() || !I->isTerminator())
    return false; // Falls through.

  MachineInstr &Last = *I;
  MachineInstr *Prev = nullptr;
  if (I != MBB.begin()) {
    MachineBasicBlock::iterator P = std::prev(I);
    while (P != MBB.begin() && P->isDebugInstr())
      --P;
    if (!P->isDebugInstr() && P->isTerminator())
      Prev = &*P;
  }

  if (isUncondBranch(Last.getOpcode())) {
    if (!Prev) {
      TBB = Last.getOperand(0).getMBB();
      return false;
    }
    if (isCondBranch(Prev->getOpcode())) {
      // A third terminator would be unanalyzable.
      if (Prev != &*MBB.getFirstTerminator())
        return true;
      unsigned G = guardIdx(*Prev);
      TBB = Prev->getOperand(0).getMBB();
      Cond.push_back(Prev->getOperand(G));
      Cond.push_back(Prev->getOperand(G + 1));
      FBB = Last.getOperand(0).getMBB();
      return false;
    }
    return true;
  }

  if (isCondBranch(Last.getOpcode()) && !Prev) {
    unsigned G = guardIdx(Last);
    TBB = Last.getOperand(0).getMBB();
    Cond.push_back(Last.getOperand(G));
    Cond.push_back(Last.getOperand(G + 1));
    return false;
  }
  return true; // EXIT and anything else.
}

unsigned SASSInstrInfo::removeBranch(MachineBasicBlock &MBB,
                                     int *BytesRemoved) const {
  unsigned Count = 0;
  MachineBasicBlock::iterator I = MBB.end();
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugInstr())
      continue;
    if (!isUncondBranch(I->getOpcode()) && !isCondBranch(I->getOpcode()))
      break;
    I->eraseFromParent();
    I = MBB.end();
    ++Count;
  }
  if (BytesRemoved)
    *BytesRemoved = Count * 16;
  return Count;
}

unsigned SASSInstrInfo::insertBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *TBB,
                                     MachineBasicBlock *FBB,
                                     ArrayRef<MachineOperand> Cond,
                                     const DebugLoc &DL,
                                     int *BytesAdded) const {
  assert(TBB && "insertBranch must not be told to insert a fallthrough");
  unsigned Count = 1;
  if (Cond.empty()) {
    noGuard(BuildMI(&MBB, DL, get(SASS::BRA)).addMBB(TBB));
  } else {
    assert(Cond.size() == 2 && "condition is [predicate, negate]");
    BuildMI(&MBB, DL, get(SASS::BRAc))
        .addMBB(TBB)
        .addReg(Cond[0].getReg())
        .addImm(Cond[1].getImm());
    if (FBB) {
      noGuard(BuildMI(&MBB, DL, get(SASS::BRA)).addMBB(FBB));
      ++Count;
    }
  }
  if (BytesAdded)
    *BytesAdded = Count * 16;
  return Count;
}

bool SASSInstrInfo::reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  Cond[1].setImm(!Cond[1].getImm());
  return false;
}

bool SASSInstrInfo::isPredicated(const MachineInstr &MI) const {
  int Idx = MI.findFirstPredOperandIdx();
  if (Idx < 0)
    return false;
  return MI.getOperand(Idx).getReg() != SASS::PT ||
         MI.getOperand(Idx + 1).getImm() != 0;
}

bool SASSInstrInfo::isPredicable(const MachineInstr &MI) const {
  return MI.getDesc().isPredicable() && !isPredicated(MI);
}

bool SASSInstrInfo::PredicateInstruction(MachineInstr &MI,
                                         ArrayRef<MachineOperand> Pred) const {
  if (!isPredicable(MI))
    return false;
  // Unconditional terminators become their non-barrier twins once guarded.
  if (MI.getOpcode() == SASS::BRA)
    MI.setDesc(get(SASS::BRAc));
  else if (MI.getOpcode() == SASS::EXIT)
    MI.setDesc(get(SASS::EXITc));
  unsigned G = guardIdx(MI);
  MI.getOperand(G).setReg(Pred[0].getReg());
  MI.getOperand(G).setIsKill(false);
  MI.getOperand(G + 1).setImm(Pred[1].getImm());
  return true;
}

bool SASSInstrInfo::SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                                      ArrayRef<MachineOperand> Pred2) const {
  return Pred1[0].getReg() == Pred2[0].getReg() &&
         Pred1[1].getImm() == Pred2[1].getImm();
}

bool SASSInstrInfo::ClobbersPredicate(MachineInstr &MI,
                                      std::vector<MachineOperand> &Pred,
                                      bool SkipDead) const {
  bool Found = false;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical() &&
        SASS::PREDRegClass.contains(MO.getReg()) &&
        !(SkipDead && MO.isDead())) {
      Pred.push_back(MO);
      Found = true;
    }
  }
  return Found;
}

// Predication saves a branch (min. stall 5) plus the 13-14 cycle
// predicate-to-branch latency; short blocks are always worth it.
bool SASSInstrInfo::isProfitableToIfCvt(MachineBasicBlock &, unsigned NumCycles,
                                        unsigned, BranchProbability) const {
  return NumCycles <= 12;
}

bool SASSInstrInfo::isProfitableToIfCvt(MachineBasicBlock &, unsigned NumT,
                                        unsigned, MachineBasicBlock &,
                                        unsigned NumF, unsigned,
                                        BranchProbability) const {
  return NumT + NumF <= 16;
}

bool SASSInstrInfo::isProfitableToDupForIfCvt(MachineBasicBlock &,
                                              unsigned NumCycles,
                                              BranchProbability) const {
  return NumCycles <= 2;
}

bool SASSInstrInfo::isHighLatencyDef(int Opc) const {
  return SASS::writesSB(get(Opc));
}
