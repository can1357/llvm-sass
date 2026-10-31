//===-- SASSAsyncCopy.cpp - Global->shared copies as LDGSTS ----------------===//
//
// A tile copy `v = LDG.128 [g]; STS.128 [s], v` keeps four registers busy
// per 16 bytes for a whole global round trip, so register pressure caps how
// many of a tile's loads are in flight and the copy takes several round
// trips. LDGSTS (PTX cp.async.cg) copies global to shared memory without
// registers: the whole tile is in flight at once, in half the instructions.
// ptxas never forms it from plain loads and stores.
//
// Legality, per block in SSA form:
// * the LDG is a simple (non-volatile, unordered) 128-bit global load whose
//   only reader is a 128-bit shared store in the same block, possibly
//   through an identity REG_SEQUENCE or full COPY;
// * the copy issues at the store, so between load and store nothing may
//   write memory other than shared memory, or order memory;
// * the copy's global read and shared write complete asynchronously: before
//   the next shared or generic access, store, side effect or the end of the
//   block, the group is committed (LDGDEPBAR) and waited for
//   (DEPBAR.LE SB5, 0).
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "sass-async-copy"

static cl::opt<bool> Enable("sass-async-copy", cl::init(true),
                            cl::desc("Turn global->shared copies into LDGSTS"));

namespace {
class SASSAsyncCopy : public MachineFunctionPass {
public:
  static char ID;
  SASSAsyncCopy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS async copy"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const SASSSubtarget *ST = nullptr;
  const SASSInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  MachineInstr *sourceLoad(MachineInstr &Store,
                           SmallVectorImpl<MachineInstr *> &Chain) const;
  bool convertBlock(MachineBasicBlock &MBB);
};
} // namespace

char SASSAsyncCopy::ID = 0;

INITIALIZE_PASS(SASSAsyncCopy, DEBUG_TYPE, "SASS async copy", false, false)

FunctionPass *llvm::createSASSAsyncCopyPass() { return new SASSAsyncCopy(); }

/// Address space of a single-memoperand access, or ~0u if unknown.
static unsigned addrSpace(const MachineInstr &MI) {
  return MI.hasOneMemOperand() ? (*MI.memoperands_begin())->getAddrSpace()
                               : ~0u;
}

/// Pending async copies must complete before \p MI: it may read their shared
/// destination or write their global source, or it orders memory.
static bool endsGroup(const MachineInstr &MI) {
  if (MI.getOpcode() == SASS::LDGSTS128)
    return false;
  if (MI.isTerminator() || MI.isCall() || MI.hasUnmodeledSideEffects() ||
      MI.hasOrderedMemoryRef() || MI.mayStore())
    return true;
  return MI.mayLoad() && addrSpace(MI) != 1;
}

/// The 128-bit global load whose value \p Store writes, if the store is its
/// only reader; \p Chain receives the identity copies in between.
MachineInstr *
SASSAsyncCopy::sourceLoad(MachineInstr &Store,
                          SmallVectorImpl<MachineInstr *> &Chain) const {
  Register V = Store.getOperand(2).getReg();
  const MachineInstr *Reader = &Store;
  while (V.isVirtual()) {
    for (const MachineInstr &U : MRI->use_nodbg_instructions(V))
      if (&U != Reader)
        return nullptr;
    MachineInstr *Def = MRI->getVRegDef(V);
    if (!Def)
      return nullptr;
    switch (Def->getOpcode()) {
    case SASS::LDG128:
    case SASS::LDG128_NC: {
      if (!Def->hasOneMemOperand())
        return nullptr;
      const MachineMemOperand &MMO = **Def->memoperands_begin();
      if (MMO.isVolatile() || !MMO.isUnordered() || MMO.getAddrSpace() != 1)
        return nullptr;
      return Def;
    }
    case TargetOpcode::COPY:
      if (Def->getOperand(1).getSubReg())
        return nullptr;
      V = Def->getOperand(1).getReg();
      break;
    case TargetOpcode::REG_SEQUENCE: {
      // Lane I of the result must be lane I of one 128-bit source.
      if (Def->getNumOperands() != 9)
        return nullptr;
      Register Src;
      for (unsigned I = 1; I < 9; I += 2) {
        const MachineOperand &Op = Def->getOperand(I);
        if (Op.getSubReg() != Def->getOperand(I + 1).getImm() ||
            (Src && Op.getReg() != Src))
          return nullptr;
        Src = Op.getReg();
      }
      V = Src;
      break;
    }
    default:
      return nullptr;
    }
    Chain.push_back(Def);
    Reader = Def;
  }
  return nullptr;
}

bool SASSAsyncCopy::convertBlock(MachineBasicBlock &MBB) {
  bool Changed = false;
  // Global addresses for offsets outside LDGSTS's signed 12 bits, per
  // (base, rounded offset); defined at their first use, which dominates
  // later ones in the block.
  DenseMap<std::pair<Register, int64_t>, Register> Bases;
  for (MachineInstr &St : make_early_inc_range(MBB)) {
    if (St.getOpcode() != SASS::STS128 || !isInt<20>(St.getOperand(1).getImm()))
      continue;
    SmallVector<MachineInstr *, 2> Chain;
    MachineInstr *Ld = sourceLoad(St, Chain);
    if (!Ld || Ld->getParent() != &MBB)
      continue;
    bool Movable = true;
    for (auto I = std::next(Ld->getIterator()); &*I != &St && Movable; ++I)
      Movable = !I->isCall() && !I->hasUnmodeledSideEffects() &&
                !I->hasOrderedMemoryRef() &&
                (!I->mayStore() || addrSpace(*I) == 3);
    if (!Movable)
      continue;

    const DebugLoc &DL = St.getDebugLoc();
    Register GBase = Ld->getOperand(1).getReg();
    int64_t GOff = Ld->getOperand(2).getImm();
    if (!isInt<12>(GOff)) {
      int64_t Hi = (GOff + 2048) & ~int64_t(4095);
      Register &Addr = Bases[{GBase, Hi}];
      if (!Addr) {
        Addr = MRI->createVirtualRegister(&SASS::GPR64RegClass);
        MachineInstr *Add =
            BuildMI(MBB, St, DL, TII->get(SASS::IADD64ri), Addr)
                .addReg(GBase)
                .addImm(Hi)
                .addReg(SASS::PT)
                .addImm(0);
        if (!ST->hasBlackwellISA())
          TII->expandAdd64(*Add);
      }
      GBase = Addr;
      GOff -= Hi;
    }
    BuildMI(MBB, St, DL, TII->get(SASS::LDGSTS128))
        .addReg(St.getOperand(0).getReg())
        .addImm(St.getOperand(1).getImm())
        .addReg(GBase)
        .addImm(GOff)
        .addReg(SASS::PT)
        .addImm(0)
        .cloneMergedMemRefs({Ld, &St});
    MRI->clearKillFlags(St.getOperand(0).getReg());
    MRI->clearKillFlags(Ld->getOperand(1).getReg());
    St.eraseFromParent();
    for (MachineInstr *MI : Chain)
      MI->eraseFromParent();
    Ld->eraseFromParent();
    Changed = true;
  }
  if (!Changed)
    return false;

  // Group boundaries: ptxas's prelude before a group, commit and wait
  // before whatever must observe it.
  bool Open = false;
  for (auto I = MBB.getFirstNonPHI();; ++I) {
    if (I == MBB.end() || endsGroup(*I)) {
      if (Open) {
        DebugLoc DL = I == MBB.end() ? DebugLoc() : I->getDebugLoc();
        BuildMI(MBB, I, DL, TII->get(SASS::LDGDEPBAR))
            .addReg(SASS::PT)
            .addImm(0);
        BuildMI(MBB, I, DL, TII->get(SASS::DEPBAR_LE))
            .addImm(0)
            .addReg(SASS::PT)
            .addImm(0);
        Open = false;
      }
      if (I == MBB.end())
        break;
      continue;
    }
    if (I->getOpcode() == SASS::LDGSTS128 && !Open) {
      for (int K = 0; K < 3; ++K)
        BuildMI(MBB, I, I->getDebugLoc(), TII->get(SASS::LDGSTS_PRELUDE))
            .addReg(SASS::PT)
            .addImm(0);
      Open = true;
    }
  }
  return true;
}

bool SASSAsyncCopy::runOnMachineFunction(MachineFunction &MF) {
  if (!Enable)
    return false;
  ST = &MF.getSubtarget<SASSSubtarget>();
  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= convertBlock(MBB);
  return Changed;
}
