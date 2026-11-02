//===-- SASSInstrInfo.cpp - SASS instruction information -------------------===//

#include "SASSInstrInfo.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static cl::opt<bool>
    RematALU("sass-remat-alu", cl::init(true), cl::Hidden,
             cl::desc("Rematerialize cheap integer ALU defs instead of "
                      "spilling them (sm_90)"));

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
  // Uniform pairs (SASSUniformRegs) to and from uniform or vector pairs.
  bool UDst = SASS::UGPR64RegClass.contains(Dst),
       USrc = SASS::UGPR64RegClass.contains(Src);
  if ((UDst || USrc) && (UDst || SASS::GPR64RegClass.contains(Dst)) &&
      (USrc || SASS::GPR64RegClass.contains(Src))) {
    unsigned Opc = UDst ? (USrc ? SASS::UMOV : SASS::R2UR) : SASS::MOVru;
    // Halves in the order that does not clobber an overlapping source.
    bool Down = RI.getEncodingValue(Dst) > RI.getEncodingValue(Src) &&
                UDst == USrc;
    unsigned First = Down ? SASS::sub1 : SASS::sub0;
    unsigned Second = Down ? SASS::sub0 : SASS::sub1;
    for (unsigned Idx : {First, Second})
      Copy(Opc, RI.getSubReg(Dst, Idx), RI.getSubReg(Src, Idx), KillSrc);
    return;
  }
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
  // A zero-extended operand's high word (zext: REG_SEQUENCE lo, (MOV 0)) is
  // read as RZ on Hopper, so no register holds the zero (an address plus a
  // 32-bit offset otherwise keeps one live, or spilled, across a loop).
  bool ZeroHi =
      !MBB.getParent()->getSubtarget<SASSSubtarget>().hasBlackwellISA();
  auto HiOf = [&](Register R) -> Register {
    const MachineInstr *D = ZeroHi ? MRI.getVRegDef(R) : nullptr;
    if (!D || !D->isRegSequence())
      return Register();
    for (unsigned I = 1, E = D->getNumOperands(); I + 1 < E; I += 2) {
      if (D->getOperand(I + 1).getImm() != SASS::sub1)
        continue;
      const MachineOperand &S = D->getOperand(I);
      const MachineInstr *Z = S.getSubReg() ? nullptr : MRI.getVRegDef(S.getReg());
      if (Z && Z->getOpcode() == SASS::MOVri && Z->getOperand(1).isImm() &&
          Z->getOperand(1).getImm() == 0 && !isPredicated(*Z))
        return SASS::RZ;
    }
    return Register();
  };
  auto L = BuildMI(MBB, MI, DL, get(LoOpc), Lo).addDef(Carry).addReg(A, {}, SASS::sub0);
  auto H = BuildMI(MBB, MI, DL, get(HiOpc), Hi);
  if (Register Z = HiOf(A))
    H.addReg(Z);
  else
    H.addReg(A, {}, SASS::sub1);
  if (B.isImm()) {
    // The immediate is sign-extended: its high word is 0 or -1.
    L.addImm(B.getImm());
    H.addImm(B.getImm() < 0 ? -1 : 0);
  } else {
    L.addReg(B.getReg(), {}, SASS::sub0);
    if (Register Z = HiOf(B.getReg()))
      H.addReg(Z);
    else
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

/// INIT_UNDEF (InitUndef's def of an undefined register that an early-clobber
/// instruction reads, e.g. the unused A half of an RS-form wgmma's descriptor
/// quad) emits nothing, but is no meta instruction: SASSControlCodes would
/// give it an issue slot and park the next instruction's wait in its stall,
/// which is never printed (an HGMMA then read UR2:UR3 one cycle after their
/// R2UR). After allocation it is just an IMPLICIT_DEF.
/// FAKE_USE (SASSPreRASched's hold on an HMMA's unread result lanes) has
/// done its work once registers are assigned.
bool SASSInstrInfo::expandPostRAPseudo(MachineInstr &MI) const {
  if (MI.isFakeUse()) {
    MI.eraseFromParent();
    return true;
  }
  if (MI.getOpcode() != TargetOpcode::INIT_UNDEF)
    return false;
  MI.setDesc(get(TargetOpcode::IMPLICIT_DEF));
  return true;
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
// On sm_90 the size is the instruction count: IfConverter adds each
// instruction's latency, which kept short guarded shared loads (a producer's
// page indices, ~25 cycles of BSSY/branch/BSYNC each) as branches.
static unsigned ifcvtSize(const MachineBasicBlock &MBB, unsigned NumCycles) {
  if (MBB.getParent()->getSubtarget<SASSSubtarget>().hasBlackwellISA())
    return NumCycles;
  unsigned N = 0;
  for (const MachineInstr &MI : MBB)
    N += !MI.isMetaInstruction() && !MI.isTerminator();
  return N;
}

bool SASSInstrInfo::isProfitableToIfCvt(MachineBasicBlock &MBB,
                                        unsigned NumCycles, unsigned,
                                        BranchProbability) const {
  return ifcvtSize(MBB, NumCycles) <= 12;
}

bool SASSInstrInfo::isProfitableToIfCvt(MachineBasicBlock &TMBB, unsigned NumT,
                                        unsigned, MachineBasicBlock &FMBB,
                                        unsigned NumF, unsigned,
                                        BranchProbability) const {
  return ifcvtSize(TMBB, NumT) + ifcvtSize(FMBB, NumF) <= 16;
}

bool SASSInstrInfo::isProfitableToDupForIfCvt(MachineBasicBlock &,
                                              unsigned NumCycles,
                                              BranchProbability) const {
  return NumCycles <= 2;
}

bool SASSInstrInfo::isHighLatencyDef(int Opc) const {
  return SASS::writesSB(get(Opc));
}

/// \p MI's block heads a triangle `MBB -> T -> J, MBB -> J` (an `if` without
/// `else`) and nothing in T reads \p MI: MachineSink can only move it into J,
/// which post-dominates the block, so no path skips it. With \p Poll, T may
/// instead be a self-loop L entered only from the block, possibly through a
/// preheader (a peeled mbarrier poll, `MBB -> [T ->] L -> L|J, MBB -> J`):
/// sinking only delays \p MI past the wait.
static bool sinksOnlyIntoJoin(const MachineInstr &MI, bool Poll = false) {
  const MachineBasicBlock *MBB = MI.getParent();
  if (MBB->succ_size() != 2)
    return false;
  const MachineBasicBlock *S0 = *MBB->succ_begin(),
                          *S1 = *std::next(MBB->succ_begin());
  auto PollLoop = [&](const MachineBasicBlock *L, const MachineBasicBlock *Pre,
                      const MachineBasicBlock *J) {
    return L != J && L->pred_size() == 2 && L->succ_size() == 2 &&
           L->isSuccessor(L) && L->isSuccessor(J) && L->isPredecessor(Pre);
  };
  for (auto [T, J] : {std::pair(S0, S1), std::pair(S1, S0)}) {
    const MachineBasicBlock *L = nullptr;
    if (T->pred_size() == 1 && T->succ_size() == 1 && *T->succ_begin() == J)
      ; // Triangle.
    else if (Poll && PollLoop(T, MBB, J))
      L = T;
    else if (Poll && T->pred_size() == 1 && T->succ_size() == 1 &&
             PollLoop(*T->succ_begin(), T, J))
      L = *T->succ_begin();
    else
      continue;
    const MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
    for (const MachineOperand &MO : MI.defs())
      if (MO.getReg().isVirtual())
        for (const MachineInstr &U : MRI.use_nodbg_instructions(MO.getReg()))
          if (U.getParent() == T || U.getParent() == L ||
              (U.isPHI() && U.getParent() == J))
            return false;
    return true;
  }
  return false;
}

// Uniform registers are one per warp: they are written only where the warp
// runs converged (SASSUniformRegs; SASSReconverge checks the final code).
// Sinking a uniform write into a successor may put it below a divergent
// branch.
static bool writesUniform(const MachineInstr &MI) {
  const MachineFunction &MF = *MI.getMF();
  if (!SASS::reconverges(MF.getSubtarget<SASSSubtarget>()))
    return false;
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  return any_of(MI.all_defs(), [&](const MachineOperand &MO) {
    Register R = MO.getReg();
    if (R.isVirtual()) {
      const TargetRegisterClass *RC = MRI.getRegClass(R);
      return RC == &SASS::UGPR32RegClass || RC == &SASS::UGPR64RegClass ||
             RC == &SASS::UGPR128RegClass || RC == &SASS::UPREDRegClass;
    }
    return R != SASS::URZ && R != SASS::UPT &&
           (SASS::UGPR32RegClass.contains(R) ||
            SASS::UGPR64RegClass.contains(R) ||
            SASS::UGPR128RegClass.contains(R) ||
            SASS::UPREDRegClass.contains(R));
  });
}

bool SASSInstrInfo::shouldPostRASink(const MachineInstr &MI) const {
  return !writesUniform(MI);
}

// MachineSink moves an instruction into the successor dominating its uses
// even when that successor post-dominates its block (as a step toward a
// later conditional block), and the per-block pre-RA scheduler cannot move
// it back. In an unrolled K loop whose steps are separated by conditional
// loads (a predicated gather's `if`s):
// * the MMA accumulator chain moved step by step into the epilogue block,
//   away from each step's loads, which then all stayed live across the chain
//   (the bs=1 MoE kernel spilled 912 bytes under a 128-register bound);
// * the prologue's read-only weight loads moved down `if` by `if` to their
//   first use, so each waited its full latency instead of overlapping the
//   steps before it.
// Neither saves work: MMAs stay with their operands, and a memory load (not
// a constant-bank LDC) does not move past a conditional load's `if`. Nor do
// uniform register writes (writesUniform).
bool SASSInstrInfo::shouldSink(const MachineInstr &MI) const {
  if (writesUniform(MI))
    return false;
  // sm_90: nor past a peeled mbarrier poll, whose fast path then waits for
  // work (an MMA accumulator sum) that could have overlapped the wait.
  if (!MI.getMF()->getSubtarget<SASSSubtarget>().hasBlackwellISA() &&
      sinksOnlyIntoJoin(MI, /*Poll=*/true) && !sinksOnlyIntoJoin(MI))
    return false;
  switch (MI.getOpcode()) {
  case SASS::HMMA16816F32:
  case SASS::HMMA16816F32BF16:
  case SASS::HMMA16816F32Z:
  case SASS::HMMA16816F32BF16Z:
    return false;
  default:
    break;
  }
  if (!MI.mayLoad() || !SASS::writesSB(MI.getDesc()) ||
      (!MI.memoperands_empty() &&
       all_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
         return MMO->getAddrSpace() == 4;
       })))
    return true;
  return !sinksOnlyIntoJoin(MI);
}

// Constant banks (kernel parameters, launch geometry, __constant__ data) are
// read-only while a kernel runs: an unpredicated LDC c[bank][imm] can be
// re-executed anywhere, so the register allocator reloads such values from
// the bank instead of spilling them (the generic check rejects these loads
// because the parameter symbol's memory operands are not dereferenceable).
bool SASSInstrInfo::isReMaterializableImpl(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case SASS::LDCB32_I:
  case SASS::LDCB64_I:
  case SASS::LDCBU8_I:
  case SASS::LDCBS8_I:
  case SASS::LDCBU16_I:
  case SASS::LDCBS16_I: {
    const MachineOperand &Def = MI.getOperand(0);
    if (Def.getSubReg() || MI.getNumExplicitDefs() != 1)
      return false;
    for (const MachineOperand &MO : drop_begin(MI.operands()))
      if (MO.isReg() && MO.getReg() != SASS::PT)
        return false;
    return !isPredicated(MI);
  }
  // Hopper: per-thread / per-CTA constants. A re-read costs an S2R
  // (scoreboarded, tens of cycles) instead of a local-memory reload.
  case SASS::S2R_TIDX:
  case SASS::S2R_TIDY:
  case SASS::S2R_TIDZ:
  case SASS::S2R_CTAIDX:
  case SASS::S2R_CTAIDY:
  case SASS::S2R_CTAIDZ:
  case SASS::S2R_LANEID:
  case SASS::S2R_SWINHI:
    if (!rematALU(MI))
      return false;
    return TargetInstrInfo::isReMaterializableImpl(MI);
  // Hopper: only P0-P6 hold predicates, and a spilled one costs SEL + STL.U8
  // and LDL.U8 + ISETP on the path that needs it; a compare of values still
  // live is issued again next to the use instead, as ptxas does.
  case SASS::ISETPrr:
  case SASS::ISETPri:
    if (!rematALU(MI))
      return false;
    return TargetInstrInfo::isReMaterializableImpl(MI);
  default:
    if (isCheapALU(MI) && !rematALU(MI))
      return false;
    return TargetInstrInfo::isReMaterializableImpl(MI);
  }
}

// Single-cycle integer ALU ops (IADD3, LOP3, SHF, LEA, SEL, IMAD on the fast
// path) with only register / immediate operands. On Hopper they are
// rematerialized next to their uses (operands still live there) instead of
// being spilled: a loop-invariant index or address otherwise costs a local
// memory round trip per iteration, and with shared memory taking nearly all
// of L1 that reload comes from L2.
bool SASSInstrInfo::isCheapALU(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case SASS::IADDrr:
  case SASS::IADDri:
  case SASS::IADD3rrr:
  case SASS::IADD3rri:
  case SASS::ISUBrr:
  case SASS::IMADrrr:
  case SASS::IMADrir:
  case SASS::IMADrri:
  case SASS::IMADSHLri:
  case SASS::IMADHIU32:
  case SASS::IMADHI:
  case SASS::LOP3rrr:
  case SASS::LOP3rir:
  case SASS::SHLrr:
  case SASS::SHLri:
  case SASS::SRLrr:
  case SASS::SRLri:
  case SASS::SRArr:
  case SASS::SRAri:
  case SASS::ROTLrr:
  case SASS::ROTLri:
  case SASS::SHLU64HI:
  case SASS::SHRU64:
  case SASS::SHRS64:
  case SASS::SELrr:
  case SASS::SELri:
  case SASS::LEArrr:
    return true;
  default:
    return false;
  }
}

// (live-range splitting rematerializes only as-cheap-as-a-move defs)
bool SASSInstrInfo::rematALU(const MachineInstr &MI) const {
  return RematALU && !isPredicated(MI) &&
         !MI.getMF()->getSubtarget<SASSSubtarget>().hasBlackwellISA();
}

bool SASSInstrInfo::isAsCheapAsAMove(const MachineInstr &MI) const {
  if ((isCheapALU(MI) || MI.getOpcode() == SASS::ISETPrr ||
       MI.getOpcode() == SASS::ISETPri) &&
      rematALU(MI))
    return true;
  return TargetInstrInfo::isAsCheapAsAMove(MI);
}
