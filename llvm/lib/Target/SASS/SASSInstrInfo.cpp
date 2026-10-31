//===-- SASSInstrInfo.cpp - SASS instruction information -------------------===//

#include "SASSInstrInfo.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
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
  if (SASS::GPR64RegClass.contains(Dst, Src)) {
    // MOV.64 exists from Blackwell; Hopper copies the halves.
    if (MBB.getParent()->getSubtarget<SASSSubtarget>().hasBlackwellISA())
      return Copy(SASS::MOV64rr, Dst, Src, KillSrc);
    for (unsigned Idx : {SASS::sub0, SASS::sub1})
      Copy(SASS::MOVrr, RI.getSubReg(Dst, Idx), RI.getSubReg(Src, Idx), KillSrc);
    return;
  }
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
  // wgmma accumulator tuples (GACC<n>) and descriptor quads: lane by lane,
  // in the order that does not clobber overlapping sources.
  if ((SASS::isGPRTupleClass(RI.getMinimalPhysRegClass(Dst)) &&
       SASS::isGPRTupleClass(RI.getMinimalPhysRegClass(Src))) ||
      SASS::UGPR128RegClass.contains(Dst, Src)) {
    SmallVector<MCRegister, 64> D, S;
    for (MCRegister R : RI.subregs(Dst))
      if (RI.subregs(R).empty())
        D.push_back(R);
    for (MCRegister R : RI.subregs(Src))
      if (RI.subregs(R).empty())
        S.push_back(R);
    if (D.size() == S.size() && !D.empty()) {
      bool Uniform = SASS::UGPR128RegClass.contains(Dst);
      auto ByEnc = [&](MCRegister A, MCRegister B) {
        return RI.getEncodingValue(A) < RI.getEncodingValue(B);
      };
      llvm::sort(D, ByEnc);
      llvm::sort(S, ByEnc);
      bool Down = RI.getEncodingValue(D[0]) > RI.getEncodingValue(S[0]);
      for (unsigned K = 0, E = D.size(); K != E; ++K) {
        unsigned I = Down ? E - 1 - K : K;
        Copy(Uniform ? SASS::UMOV : SASS::MOVrr, D[I], S[I], KillSrc);
      }
      return;
    }
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

/// Spill (STL) or reload (LDL) opcode for a register class. Predicates and
/// uniform registers use pseudos that SASSRegisterInfo::eliminateFrameIndex
/// expands through a scavenged GPR.
static unsigned spillOpcode(const TargetRegisterClass *RC, bool Store,
                            const TargetRegisterInfo &TRI) {
  if (SASS::GPR32RegClass.hasSubClassEq(RC))
    return Store ? SASS::STL32 : SASS::LDL32;
  if (SASS::GPR64RegClass.hasSubClassEq(RC))
    return Store ? SASS::STL64 : SASS::LDL64;
  if (SASS::GPR128RegClass.hasSubClassEq(RC))
    return Store ? SASS::STL128 : SASS::LDL128;
  if (SASS::PREDRegClass.hasSubClassEq(RC))
    return Store ? SASS::SPILL_P : SASS::RELOAD_P;
  if (SASS::UGPR32RegClass.hasSubClassEq(RC))
    return Store ? SASS::SPILL_UR32 : SASS::RELOAD_UR32;
  if (SASS::UGPR64RegClass.hasSubClassEq(RC))
    return Store ? SASS::SPILL_UR64 : SASS::RELOAD_UR64;
#define GACC_SPILL(S)                                                          \
  if (SASS::GACC##S##RegClass.hasSubClassEq(RC))                               \
    return Store ? SASS::SPILL_GACC##S : SASS::RELOAD_GACC##S;
  GACC_SPILL(8) GACC_SPILL(12) GACC_SPILL(16) GACC_SPILL(20) GACC_SPILL(24)
  GACC_SPILL(28) GACC_SPILL(32) GACC_SPILL(36) GACC_SPILL(40) GACC_SPILL(44)
  GACC_SPILL(48) GACC_SPILL(52) GACC_SPILL(56) GACC_SPILL(60) GACC_SPILL(64)
#undef GACC_SPILL
  report_fatal_error(Twine("SASS: cannot spill register class ") +
                     TRI.getRegClassName(RC));
}

static MachineMemOperand *spillMemOperand(MachineBasicBlock &MBB, int FI,
                                          MachineMemOperand::Flags F) {
  MachineFunction &MF = *MBB.getParent();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI), F,
                                 MFI.getObjectSize(FI), MFI.getObjectAlign(FI));
}

void SASSInstrInfo::storeRegToStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
    bool IsKill, int FI, const TargetRegisterClass *RC, Register,
    MachineInstr::MIFlag Flags) const {
  noGuard(BuildMI(MBB, MI, DebugLoc(), get(spillOpcode(RC, true, RI)))
              .addFrameIndex(FI)
              .addImm(0)
              .addReg(SrcReg, getKillRegState(IsKill)))
      .addMemOperand(spillMemOperand(MBB, FI, MachineMemOperand::MOStore))
      .setMIFlags(Flags);
}

void SASSInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register DestReg,
    int FI, const TargetRegisterClass *RC, Register, unsigned,
    MachineInstr::MIFlag Flags) const {
  noGuard(BuildMI(MBB, MI, DebugLoc(), get(spillOpcode(RC, false, RI)), DestReg)
              .addFrameIndex(FI)
              .addImm(0))
      .addMemOperand(spillMemOperand(MBB, FI, MachineMemOperand::MOLoad))
      .setMIFlags(Flags);
}

/// The register an LDL/STL (or spill pseudo) at [FI+0] loads or stores.
static Register stackSlotAccess(const MachineInstr &MI, bool Store,
                                int &FrameIndex) {
  unsigned Opc = MI.getOpcode();
  bool Match =
      Store ? is_contained({SASS::STL32, SASS::STL64, SASS::STL128,
                            SASS::SPILL_P, SASS::SPILL_UR32, SASS::SPILL_UR64},
                           Opc)
            : is_contained({SASS::LDL32, SASS::LDL64, SASS::LDL128,
                            SASS::RELOAD_P, SASS::RELOAD_UR32,
                            SASS::RELOAD_UR64},
                           Opc);
  unsigned Addr = Store ? 0 : 1;
  if (!Match || !MI.getOperand(Addr).isFI() ||
      MI.getOperand(Addr + 1).getImm() != 0)
    return Register();
  FrameIndex = MI.getOperand(Addr).getIndex();
  return MI.getOperand(Store ? 2 : 0).getReg();
}

Register SASSInstrInfo::isLoadFromStackSlot(const MachineInstr &MI,
                                            int &FrameIndex) const {
  return stackSlotAccess(MI, false, FrameIndex);
}

Register SASSInstrInfo::isStoreToStackSlot(const MachineInstr &MI,
                                           int &FrameIndex) const {
  return stackSlotAccess(MI, true, FrameIndex);
}

static bool isUncondBranch(unsigned Opc) { return Opc == SASS::BRA; }
static bool isCondBranch(unsigned Opc) {
  return Opc == SASS::BRAc || Opc == SASS::BRAcu;
}

/// Branch conditions are [predicate, negate, uniform]: uniform branches
/// (BRAcu) need no reconvergence (SASSReconverge).
static void pushCond(const MachineInstr &Br, unsigned G,
                     SmallVectorImpl<MachineOperand> &Cond) {
  Cond.push_back(Br.getOperand(G));
  Cond.push_back(Br.getOperand(G + 1));
  Cond.push_back(MachineOperand::CreateImm(Br.getOpcode() == SASS::BRAcu));
}

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
      pushCond(*Prev, G, Cond);
      FBB = Last.getOperand(0).getMBB();
      return false;
    }
    return true;
  }

  if (isCondBranch(Last.getOpcode()) && !Prev) {
    unsigned G = guardIdx(Last);
    TBB = Last.getOperand(0).getMBB();
    pushCond(Last, G, Cond);
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
    assert(Cond.size() == 3 && "condition is [predicate, negate, uniform]");
    BuildMI(&MBB, DL, get(Cond[2].getImm() ? SASS::BRAcu : SASS::BRAc))
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
  return MI.getDesc().isPredicable() && !SASS::notPredicable(MI.getDesc()) &&
         !isPredicated(MI);
}

bool SASSInstrInfo::PredicateInstruction(MachineInstr &MI,
                                         ArrayRef<MachineOperand> Pred) const {
  if (!isPredicable(MI))
    return false;
  // Unconditional terminators become their non-barrier twins once guarded.
  if (MI.getOpcode() == SASS::BRA)
    MI.setDesc(get(Pred.size() > 2 && Pred[2].getImm() ? SASS::BRAcu
                                                       : SASS::BRAc));
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
