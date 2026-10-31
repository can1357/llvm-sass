//===-- SASSUniformAddr.cpp - Uniform address induction for shared loads ---===//
//
// A shared load in a loop typically addresses `base + (iv << s) + imm`:
//
//   loop:  %iv = PHI [%init, preheader], [%iv.next, loop]
//          %a  = LEA %iv, %base, s
//          LDS.128 %v, [%a+imm]
//          %iv.next = IADD %iv, step
//
// LDS reads a GPR address when it leaves the LSU queue, so the next
// iteration's LEA must wait for every queued load of this one (a read
// scoreboard), and the first load of an iteration waits for the LEA. A
// uniform register is captured when the load enters the queue. When %init
// and %base are warp-uniform the address becomes a uniform induction
// variable, as ptxas does:
//
//   preheader: %u0 = R2UR (LEA %init, %base, s)
//   loop:      %u  = PHI [%u0, preheader], [%u.next, loop]
//              LDS.128 %v, [%u+imm]
//              %u.next = UIADD3 %u, step << s
//
// The increment then never waits on queued loads and can be scheduled
// anywhere in the iteration.
//
// Uniformity is conservative: values derived only from constants, kernel
// parameters (constant bank), CTA ids and arithmetic on uniform values; PHIs
// only when every conditional branch in the function is itself uniform, so
// no PHI merges values from a divergent region.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "sass-uniform-addr"

static cl::opt<bool> Enable("sass-uniform-addr", cl::init(true),
                            cl::desc("Uniform address induction for LDS"));

namespace {
class SASSUniformAddr : public MachineFunctionPass {
public:
  static char ID;
  SASSUniformAddr() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS uniform addresses"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  /// Virtual registers that may differ between threads of a warp.
  DenseSet<Register> Divergent;

  void computeDivergence(MachineFunction &MF);
  bool isUniform(const MachineOperand &MO) const {
    return MO.isImm() || (MO.isReg() && MO.getReg().isVirtual() &&
                          !Divergent.count(MO.getReg()));
  }
};
} // namespace

char SASSUniformAddr::ID = 0;

INITIALIZE_PASS_BEGIN(SASSUniformAddr, DEBUG_TYPE, "SASS uniform addresses",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SASSUniformAddr, DEBUG_TYPE, "SASS uniform addresses",
                    false, false)

FunctionPass *llvm::createSASSUniformAddrPass() { return new SASSUniformAddr(); }

/// Instructions whose results are the same in every thread whenever their
/// register sources are.
static bool isUniformOp(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case SASS::S2R_TIDX: case SASS::S2R_TIDY: case SASS::S2R_TIDZ:
  case SASS::S2R_LANEID:
    return false;
  default:
    break;
  }
  if (MI.mayLoadOrStore())
    return MI.getOpcode() == SASS::LDC32 || MI.getOpcode() == SASS::LDC64;
  return !MI.hasUnmodeledSideEffects() && !MI.isCall();
}

void SASSUniformAddr::computeDivergence(MachineFunction &MF) {
  Divergent.clear();
  // Seeds, then propagation to users until nothing changes. PHIs join their
  // inputs; with any divergent branch they are all divergent.
  bool PhisDivergent = false;
  SmallVector<Register, 64> Work;
  auto Mark = [&](Register R) {
    if (R.isVirtual() && Divergent.insert(R).second)
      Work.push_back(R);
  };
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      bool Seed = !MI.isPHI() && !MI.isCopy() && !isUniformOp(MI);
      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && MO.getReg().isPhysical() &&
            !MRI->isReserved(MO.getReg()))
          Seed = true; // Unknown physical input.
      if (Seed)
        for (const MachineOperand &MO : MI.defs())
          Mark(MO.getReg());
    }
  auto Propagate = [&] {
    while (!Work.empty()) {
      Register R = Work.pop_back_val();
      for (MachineInstr &U : MRI->use_nodbg_instructions(R))
        for (const MachineOperand &MO : U.defs())
          Mark(MO.getReg());
    }
  };
  Propagate();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &T : MBB.terminators())
      for (const MachineOperand &MO : T.uses())
        if (MO.isReg() && Divergent.count(MO.getReg()))
          PhisDivergent = true;
  if (PhisDivergent) {
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &Phi : MBB.phis())
        Mark(Phi.getOperand(0).getReg());
    Propagate();
  }
}

/// Replaces \p Ld (a GPR-addressed LDS) by its uniform-address form.
static unsigned urForm(unsigned Opc) {
  switch (Opc) {
  case SASS::LDS32: return SASS::LDS32_UR;
  case SASS::LDS64: return SASS::LDS64_UR;
  case SASS::LDS128: return SASS::LDS128_UR;
  default: return 0;
  }
}

bool SASSUniformAddr::runOnMachineFunction(MachineFunction &MF) {
  if (!Enable)
    return false;
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  const MachineLoopInfo &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  computeDivergence(MF);

  // Uniform address IV per (LEA result): shared by all loads using it.
  DenseMap<Register, Register> UR;
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    const MachineLoop *L = MLI.getLoopFor(&MBB);
    if (!L)
      continue;
    for (MachineInstr &Ld : make_early_inc_range(MBB)) {
      unsigned NewOpc = urForm(Ld.getOpcode());
      Register A = NewOpc ? Ld.getOperand(1).getReg() : Register();
      MachineInstr *Lea = A.isVirtual() ? MRI->getVRegDef(A) : nullptr;
      if (!Lea || Lea->getOpcode() != SASS::LEArrr)
        continue;
      Register &U = UR[A];
      if (!U) {
        // %a = LEA %iv, %base, s with %iv a header PHI stepping by a
        // constant and %base defined outside the loop.
        MachineInstr *Phi = MRI->getVRegDef(Lea->getOperand(1).getReg());
        Register Base = Lea->getOperand(2).getReg();
        MachineInstr *BaseDef = MRI->getVRegDef(Base);
        int64_t Shift = Lea->getOperand(3).getImm();
        if (!Phi || !Phi->isPHI() || Phi->getParent() != L->getHeader() ||
            Phi->getNumOperands() != 5 || !BaseDef ||
            L->contains(BaseDef->getParent()) ||
            !isUniform(Lea->getOperand(2)))
          continue;
        unsigned InIdx = L->contains(Phi->getOperand(2).getMBB()) ? 3 : 1;
        unsigned NextIdx = InIdx == 1 ? 3 : 1;
        MachineBasicBlock *Pre = Phi->getOperand(InIdx + 1).getMBB();
        MachineBasicBlock *Latch = Phi->getOperand(NextIdx + 1).getMBB();
        Register Init = Phi->getOperand(InIdx).getReg();
        MachineInstr *Next = MRI->getVRegDef(Phi->getOperand(NextIdx).getReg());
        if (!isUniform(Phi->getOperand(InIdx)) || !Next ||
            Next->getOpcode() != SASS::IADDri ||
            Next->getOperand(1).getReg() != Phi->getOperand(0).getReg() ||
            L->contains(Pre) || !L->contains(Latch))
          continue;
        int64_t Step = Next->getOperand(2).getImm() << Shift;
        if (!isInt<32>(Step))
          continue;
        const DebugLoc &DL = Lea->getDebugLoc();
        // Preheader: the first iteration's address, moved to a UR once.
        Register A0 = MRI->createVirtualRegister(&SASS::GPR32RegClass);
        Register U0 = MRI->createVirtualRegister(&SASS::UGPR32RegClass);
        MachineBasicBlock::iterator PreEnd = Pre->getFirstTerminator();
        BuildMI(*Pre, PreEnd, DL, TII->get(SASS::LEArrr), A0)
            .addReg(Init)
            .addReg(Base)
            .addImm(Shift)
            .addReg(SASS::PT)
            .addImm(0);
        BuildMI(*Pre, PreEnd, DL, TII->get(SASS::R2UR), U0)
            .addReg(A0)
            .addReg(SASS::PT)
            .addImm(0);
        // Loop: a uniform PHI stepped once per iteration.
        U = MRI->createVirtualRegister(&SASS::UGPR32RegClass);
        Register UNext = MRI->createVirtualRegister(&SASS::UGPR32RegClass);
        BuildMI(*L->getHeader(), L->getHeader()->begin(), DL,
                TII->get(TargetOpcode::PHI), U)
            .addReg(U0)
            .addMBB(Pre)
            .addReg(UNext)
            .addMBB(Latch);
        BuildMI(*Latch, Latch->getFirstTerminator(), DL,
                TII->get(SASS::UIADD3ri), UNext)
            .addReg(U)
            .addImm(Step)
            .addReg(SASS::PT)
            .addImm(0);
        MRI->clearKillFlags(Init);
        MRI->clearKillFlags(Base);
      }
      MachineInstrBuilder MIB =
          BuildMI(MBB, Ld, Ld.getDebugLoc(), TII->get(NewOpc),
                  Ld.getOperand(0).getReg())
              .addReg(U)
              .addImm(Ld.getOperand(2).getImm())
              .addReg(Ld.getOperand(3).getReg())
              .addImm(Ld.getOperand(4).getImm());
      MIB.cloneMemRefs(Ld);
      Ld.eraseFromParent();
      if (MRI->use_nodbg_empty(A))
        Lea->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}
