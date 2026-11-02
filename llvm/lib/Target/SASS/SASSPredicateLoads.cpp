//===-- SASSPredicateLoads.cpp - Predicate guarded shared loads -----------===//
//
// Hopper. A conditional shared-memory read,
//
//   v = c ? smem[i] : 0;
//
// reaches the machine level as a triangle: the head branches around a block
// holding the address arithmetic and the LDS, and a PHI at the join merges
// the loaded value with the default. The post-RA IfConverter predicates such
// triangles, but only after register allocation: an unrolled row of them
// (four page-index reads of a tile copy) shares one predicate register and
// the rows come out strictly one after another, each with its compare-to-
// guard latency, where ptxas evaluates all compares, then issues all loads
// back to back.
//
// This pass converts the triangles in SSA form, before the pre-RA scheduler:
// the address arithmetic moves into the head unconditionally (it cannot
// fault), the load is guarded by the branch condition, and each PHI becomes
// a SEL between the loaded value and the default. Successive triangles then
// fall into one block, where the scheduler interleaves their rows and the
// allocator gives each row its own predicate.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "sass-predicate-loads"

static cl::opt<bool> Enable("sass-predicate-loads", cl::init(true),
                            cl::desc("Predicate conditional shared loads "
                                     "before scheduling (sm_90)"));

/// Instructions of the guarded block, the loads included.
static constexpr unsigned MaxInstrs = 8;

namespace {
class SASSPredicateLoads : public MachineFunctionPass {
public:
  static char ID;
  SASSPredicateLoads() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SASS predicate conditional shared loads";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const SASSInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  bool speculatable(const MachineInstr &MI) const;
  bool predicableLoad(const MachineInstr &MI) const;
  bool convert(MachineBasicBlock &H);
};
} // namespace

char SASSPredicateLoads::ID = 0;

INITIALIZE_PASS(SASSPredicateLoads, DEBUG_TYPE,
                "SASS predicate conditional shared loads", false, false)

FunctionPass *llvm::createSASSPredicateLoadsPass() {
  return new SASSPredicateLoads();
}

/// Pure GPR arithmetic, safe to run whatever the condition.
bool SASSPredicateLoads::speculatable(const MachineInstr &MI) const {
  if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall() ||
      MI.isInlineAsm() || MI.isTerminator() || MI.isPHI() ||
      TII->isPredicated(MI))
    return false;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    Register R = MO.getReg();
    if (R.isPhysical()) {
      if (MO.isDef() || !MRI->isConstantPhysReg(R))
        return false;
      continue;
    }
    if (MO.isDef() && !SASS::isGPRTupleClass(MRI->getRegClass(R)))
      return false;
  }
  return true;
}

/// A shared-memory load into one GPR value that a guard can switch off.
bool SASSPredicateLoads::predicableLoad(const MachineInstr &MI) const {
  if (!MI.mayLoad() || MI.mayStore() || MI.hasUnmodeledSideEffects() ||
      MI.hasOrderedMemoryRef() || !MI.hasOneMemOperand() ||
      (*MI.memoperands_begin())->getAddrSpace() != 3 ||
      !TII->isPredicable(MI) || MI.getNumExplicitDefs() != 1)
    return false;
  Register D = MI.getOperand(0).getReg();
  return D.isVirtual() && SASS::isGPRTupleClass(MRI->getRegClass(D));
}

bool SASSPredicateLoads::convert(MachineBasicBlock &H) {
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 3> Cond;
  if (H.succ_size() != 2 || TII->analyzeBranch(H, TBB, FBB, Cond, false) ||
      Cond.size() != 3 || !Cond[0].isReg() || !Cond[0].getReg().isVirtual())
    return false;
  if (!FBB)
    FBB = H.getNextNode();
  if (!TBB || !FBB || !H.isSuccessor(TBB) || !H.isSuccessor(FBB))
    return false;
  // T: the guarded block, J: the join. T runs when the branch is not taken
  // (it skips T) or when it is taken to T.
  MachineBasicBlock *T, *J;
  bool Negate;
  if (TBB->pred_size() == 1 && TBB->isSuccessor(FBB) && TBB->succ_size() == 1) {
    T = TBB, J = FBB, Negate = false;
  } else if (FBB->pred_size() == 1 && FBB->isSuccessor(TBB) &&
             FBB->succ_size() == 1) {
    T = FBB, J = TBB, Negate = true;
  } else {
    return false;
  }
  if (J->pred_size() != 2 || J == &H || T->hasAddressTaken() ||
      J->hasAddressTaken() || J->isEHPad())
    return false;

  // T: speculatable arithmetic and at least one shared load, then at most
  // an unconditional branch to J.
  unsigned N = 0, Loads = 0;
  for (MachineInstr &MI : *T) {
    if (MI.isDebugInstr())
      continue;
    if (MI.isTerminator()) {
      MachineBasicBlock *Dst = nullptr;
      SmallVector<MachineOperand, 3> TC;
      MachineBasicBlock *TF = nullptr;
      if (TII->analyzeBranch(*T, Dst, TF, TC, false) || !TC.empty() ||
          (Dst && Dst != J))
        return false;
      break;
    }
    if (++N > MaxInstrs)
      return false;
    if (predicableLoad(MI))
      ++Loads;
    else if (!speculatable(MI))
      return false;
  }
  if (!Loads)
    return false;
  // J's PHIs: one 32-bit value from each side.
  for (MachineInstr &Phi : J->phis()) {
    if (Phi.getNumOperands() != 5 ||
        MRI->getRegClass(Phi.getOperand(0).getReg()) != &SASS::GPR32RegClass)
      return false;
    for (unsigned I = 1; I < 5; I += 2)
      if (Phi.getOperand(I).getSubReg() ||
          !Phi.getOperand(I).getReg().isVirtual())
        return false;
  }

  // The guard of T's loads and of the SELs: T's condition.
  Register P = Cond[0].getReg();
  int64_t Neg = Cond[1].getImm() ^ (Negate ? 1 : 0);
  SmallVector<MachineOperand, 3> Guard = {MachineOperand::CreateReg(P, false),
                                          MachineOperand::CreateImm(Neg)};
  MRI->clearKillFlags(P);

  // T's body into H, then H falls into J.
  MachineBasicBlock::iterator At = H.getFirstTerminator();
  for (MachineInstr &MI : make_early_inc_range(*T)) {
    if (MI.isTerminator())
      break;
    H.splice(At, T, MI.getIterator());
    if (MI.mayLoad())
      TII->PredicateInstruction(MI, Guard);
  }
  DebugLoc DL = At != H.end() ? At->getDebugLoc() : DebugLoc();
  for (MachineInstr &Phi : make_early_inc_range(J->phis())) {
    Register D = Phi.getOperand(0).getReg(), VT, VH;
    for (unsigned I = 1; I < 5; I += 2)
      (Phi.getOperand(I + 1).getMBB() == T ? VT : VH) =
          Phi.getOperand(I).getReg();
    MachineInstr *Def = MRI->getVRegDef(VH);
    MachineInstrBuilder SEL;
    if (Def && Def->getOpcode() == SASS::MOVri && !TII->isPredicated(*Def) &&
        Def->getOperand(1).isImm())
      SEL = BuildMI(H, At, DL, TII->get(SASS::SELri), D)
                .addReg(VT)
                .addImm(Def->getOperand(1).getImm());
    else
      SEL = BuildMI(H, At, DL, TII->get(SASS::SELrr), D).addReg(VT).addReg(VH);
    SEL.addReg(P).addImm(Neg).addReg(SASS::PT).addImm(0);
    Phi.eraseFromParent();
    if (Def && Def->getOpcode() == SASS::MOVri && MRI->use_nodbg_empty(VH))
      Def->eraseFromParent();
  }
  TII->removeBranch(H);
  H.removeSuccessor(T);
  T->removeSuccessor(J);
  T->eraseFromParent();

  // J now has H as its only predecessor: merge it. Where J fell through, H
  // branches to that block unless it follows H.
  MachineBasicBlock *FT = J->getFallThrough();
  H.splice(H.end(), J, J->begin(), J->end());
  H.removeSuccessor(J);
  H.transferSuccessorsAndUpdatePHIs(J);
  J->eraseFromParent();
  if (FT && H.getNextNode() != FT)
    TII->insertBranch(H, FT, nullptr, {}, DebugLoc());
  return true;
}

bool SASSPredicateLoads::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (!Enable || ST.hasBlackwellISA() || !MF.getRegInfo().isSSA())
    return false;
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  bool Changed = false;
  for (bool Again = true; Again;) {
    Again = false;
    for (MachineBasicBlock &MBB : MF)
      if (convert(MBB)) {
        Again = Changed = true;
        break;
      }
  }
  return Changed;
}
