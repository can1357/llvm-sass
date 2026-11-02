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
#include "SASSSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

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

//===----------------------------------------------------------------------===//
// Uniform registers
//===----------------------------------------------------------------------===//
//
// ptxas keeps warp-uniform integers (kernel parameters, CTA ids, values
// broadcast with __shfl_sync(~0, x, 0), loop-invariant address parts and the
// arithmetic on them) in uniform registers, computed once per warp on the
// uniform datapath (ULDC, S2UR, UIADD3, UIMAD, ULOP3, USHF, ULEA). That
// frees the GPRs they would occupy for the whole kernel. This pass does the
// same on Hopper:
//
//   1. Divergence. A GPR value is divergent when it comes from a thread
//      index, a load (constant bank excepted), a warp collective (except
//      SHFL.IDX of a uniform lane over the whole warp, which broadcasts), a
//      call or an unknown physical register; when an operand is divergent;
//      when it is a PHI at a join of a divergent branch; or when it leaves a
//      loop with a divergent exit (the threads left at different
//      iterations). Branches are divergent when their predicate is.
//   2. Placement. Threads that took different sides of a divergent branch
//      may interleave and share one set of uniform registers, so uniform
//      values are written only while the warp runs converged: never between
//      a branch that may split it (splitsWarp: every branch not selected
//      uniform, whatever the divergence analysis proves or assumes, except
//      warp-uniform tests) and its reconvergence point. Where the sides
//      never reconverge (one side exits), the exiting side may run alongside
//      the rest of the kernel: it neither writes nor reads uniform registers.
//   3. Hoisting. Uniform values computed in split code (a kernel parameter
//      loaded again in each divergent row) move to the nearest converged
//      dominator, one def per value, and are only read inside.
//   4. Selection. Uniform values defined by instructions with a uniform twin
//      move to UR, together with the PHIs between them, when the value is
//      used by another uniform instruction or lives past its block (hoisted
//      ones: when they save a load or are read as UR). Uniform operands
//      computed elsewhere enter by R2UR; per-thread users read the UR
//      through a rematerializable MOV in their block (or directly: LDS
//      [UR+imm], R2UR). At most URBudget uniform registers are live at once,
//      so the allocator never rematerializes or splits a uniform value next
//      to a user in split code.
//   5. Later passes keep it: MachineSink and PostRAMachineSink do not move
//      uniform writes (SASSInstrInfo::shouldSink, shouldPostRASink);
//      SASSReconverge gathers the warp (WARPSYNC.ALL) before uniform writes
//      below divergent regions it could not bracket and, with assertions,
//      checks the final code (-sass-verify-uniform). SASSControlCodes keeps
//      ULDC/UMOV from overwriting a UR that a uniform ALU instruction issued
//      less than 3 cycles earlier still reads.
//
//===----------------------------------------------------------------------===//

static cl::opt<bool> EnableRegs("sass-uniform-regs", cl::init(true),
                                cl::desc("Keep warp-uniform integers in "
                                         "uniform registers (sm_90)"));

static cl::opt<unsigned> URBudget(
    "sass-uniform-regs-budget", cl::init(48),
    cl::desc("Uniform registers live at once that promotion may reach (of "
             "63): beyond it the allocator would rematerialize or spill "
             "uniform values, possibly where the warp is split"));

namespace {
class SASSUniformRegs : public MachineFunctionPass {
public:
  static char ID;
  SASSUniformRegs() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS uniform registers"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *MLI = nullptr;
  MachinePostDominatorTree *PDT = nullptr;

  DenseSet<Register> Div;
  SmallVector<Register, 64> Work;
  /// Divergent conditional branches whose joins have been marked.
  SmallPtrSet<MachineBasicBlock *, 16> DivBranches;
  /// Blocks that must not write uniform registers / read promoted ones.
  SmallPtrSet<const MachineBasicBlock *, 32> NoUDef, ExitSide;
  /// The first WARPSYNC.ALL of each block that has one: the hardware gathers
  /// the whole warp there, which ends any divergent region (BAR.SYNC and
  /// other .aligned operations do not regather a split warp).
  DenseMap<const MachineBasicBlock *, const MachineInstr *> GatherAt;
  /// The first .aligned operation of each block that has one.
  DenseMap<const MachineBasicBlock *, const MachineInstr *> AlignedAt;
  DenseMap<const MachineInstr *, unsigned> Order;
  MachineDominatorTree *MDT = nullptr;
  /// Values moved to UR and their uniform twins.
  DenseMap<Register, Register> UMap;

  bool isGPRValue(Register R) const {
    if (!R.isVirtual())
      return false;
    const TargetRegisterClass *RC = MRI->getRegClass(R);
    return RC == &SASS::GPR32RegClass || RC == &SASS::GPR64RegClass;
  }
  bool isDivergent(const MachineOperand &MO) const {
    if (!MO.isReg() || !MO.getReg())
      return false;
    Register R = MO.getReg();
    if (R.isPhysical())
      return !MRI->isReserved(R) && !MRI->isConstantPhysReg(R);
    return Div.count(R);
  }
  /// Registers of the uniform datapath hold one value per warp.
  bool isUniformClass(Register R) const {
    const TargetRegisterClass *RC = MRI->getRegClass(R);
    return RC == &SASS::UGPR32RegClass || RC == &SASS::UGPR64RegClass ||
           RC == &SASS::UGPR128RegClass || RC == &SASS::UPREDRegClass ||
           RC == &SASS::UPREDTRegClass;
  }
  void mark(Register R) {
    if (R.isVirtual() && !isUniformClass(R) && Div.insert(R).second)
      Work.push_back(R);
  }
  void markDefs(const MachineInstr &MI) {
    for (const MachineOperand &MO : MI.all_defs())
      mark(MO.getReg());
  }
  bool isSeed(const MachineInstr &MI) const;
  bool usePropagates(const MachineInstr &User, const MachineOperand &MO) const;
  /// The kernel reads only threadIdx.x (SASS::readsOnlyTidX).
  bool OnlyTidX = false;
  bool warpUniform(const MachineInstr &Br) const {
    return SASS::branchesWarpUniformly(
        Br, OnlyTidX, [&](const MachineInstr &, Register R) {
          return R.isVirtual() ? MRI->getVRegDef(R) : nullptr;
        });
  }
  bool splitsWarp(MachineBasicBlock &MBB) const;
  MachineBasicBlock *divergentBranch(MachineBasicBlock &MBB) const;
  bool witnessesUniform(MachineBasicBlock &MBB) const;
  void reach(MachineBasicBlock *From, MachineBasicBlock *Stop,
             SmallPtrSetImpl<MachineBasicBlock *> &Out,
             bool StopAtAligned = false) const;
  /// Whether \p MI may run while another part of the warp runs elsewhere.
  bool fenced(const MachineInstr &MI) const {
    const MachineBasicBlock *B = MI.getParent();
    if (!NoUDef.count(B))
      return false;
    auto G = GatherAt.find(B);
    return G == GatherAt.end() || Order.lookup(&MI) <= Order.lookup(G->second);
  }
  bool fencedAtEnd(const MachineBasicBlock *B) const {
    return NoUDef.count(B) && !GatherAt.count(B);
  }
  void computeDivergence(MachineFunction &MF);
  void computePlacement(MachineFunction &MF);
  /// Defs moved out of split code by hoist(): their blocks and order before.
  DenseMap<MachineInstr *, std::pair<MachineBasicBlock *, unsigned>> Hoisted;
  /// Hoisted defs that replaced identical ones from other blocks: they stay.
  SmallPtrSet<MachineInstr *, 16> Pinned;
  void hoist(MachineFunction &MF);
  void unhoist(MachineInstr &MI);
  void trimToBudget(MachineFunction &MF, DenseSet<Register> &C);
  bool promote(MachineFunction &MF);
};
} // namespace

char SASSUniformRegs::ID = 0;

INITIALIZE_PASS_BEGIN(SASSUniformRegs, "sass-uniform-regs",
                      "SASS uniform registers", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SASSUniformRegs, "sass-uniform-regs",
                    "SASS uniform registers", false, false)

FunctionPass *llvm::createSASSUniformRegsPass() { return new SASSUniformRegs(); }

static bool isShflIdx(unsigned Opc) {
  switch (Opc) {
  case SASS::SHFL_IDX_rr: case SASS::SHFL_IDX_ir: case SASS::SHFL_IDX_ri:
  case SASS::SHFL_IDX_ii: case SASS::SHFLP_IDX_rr: case SASS::SHFLP_IDX_ir:
  case SASS::SHFLP_IDX_ri: case SASS::SHFLP_IDX_ii:
    return true;
  default:
    return false;
  }
}

/// Warp votes: one result for the warp whatever the voting predicates.
static bool isVote(unsigned Opc) {
  switch (Opc) {
  case SASS::VOTE_BALLOT: case SASS::VOTE_ANY: case SASS::VOTE_ALL:
  case SASS::VOTE_EQ: case SASS::VOTE_ACTIVE:
    return true;
  default:
    return false;
  }
}

bool SASSUniformRegs::isSeed(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  switch (Opc) {
  case SASS::S2R_CTAIDX: case SASS::S2R_CTAIDY: case SASS::S2R_CTAIDZ:
  case SASS::S2R_CGA: case SASS::S2R_SWINHI:
  case SASS::LDC32: case SASS::LDC64:
  case SASS::LDCB32: case SASS::LDCB64: case SASS::LDCBU8: case SASS::LDCBS8:
  case SASS::LDCBU16: case SASS::LDCBS16:
  case SASS::LDCB32_I: case SASS::LDCB64_I: case SASS::LDCBU8_I:
  case SASS::LDCBS8_I: case SASS::LDCBU16_I: case SASS::LDCBS16_I:
    return false;
  default:
    break;
  }
  // A broadcast of a uniform lane over the whole warp: SHFL.IDX with
  // c = 0x1f (no segments, clamp 31). Its data operand does not propagate.
  if (isShflIdx(Opc)) {
    unsigned D = MI.getNumExplicitDefs();
    const MachineOperand &C = MI.getOperand(D + 2);
    return !C.isImm() || C.getImm() != 0x1f;
  }
  if (isVote(Opc))
    return false;
  StringRef Name = TII->getName(Opc);
  if (Name.starts_with("S2R") || Name.starts_with("CS2R"))
    return true;
  if (MI.isCall() || MI.isInlineAsm() || MI.hasUnmodeledSideEffects() ||
      MI.mayLoad())
    return true;
  for (const MachineOperand &MO : MI.uses())
    if (MO.isReg() && MO.getReg().isPhysical() && isDivergent(MO))
      return true;
  return false;
}

bool SASSUniformRegs::usePropagates(const MachineInstr &User,
                                    const MachineOperand &MO) const {
  if (isVote(User.getOpcode()))
    return false;
  if (isShflIdx(User.getOpcode())) // all but the data operand
    return MO.getOperandNo() != User.getNumExplicitDefs();
  return true;
}

static MachineBasicBlock *ipdom(MachinePostDominatorTree &PDT,
                                MachineBasicBlock *B) {
  MachineDomTreeNode *N = PDT.getNode(B);
  return N && N->getIDom() ? N->getIDom()->getBlock() : nullptr;
}

/// PTX .aligned operations (barriers, setmaxnreg, wgmma): every thread of
/// the warp executes the same one together.
static bool isAligned(unsigned Opc) {
  switch (Opc) {
  case SASS::BAR_SYNC: case SASS::BAR_SYNC_I: case SASS::BAR_SYNC_R:
  case SASS::BAR_SYNC_II: case SASS::BAR_SYNC_IR: case SASS::BAR_SYNC_RI:
  case SASS::BAR_SYNC_RR: case SASS::BAR_ARV_II: case SASS::BAR_ARV_IR:
  case SASS::BAR_ARV_RI: case SASS::BAR_ARV_RR: case SASS::WARPSYNC_ALL:
  case SASS::SETMAXNREG_INC: case SASS::SETMAXNREG_DEC:
  case SASS::USETMAXREG_DEALLOC: case SASS::USETMAXREG_TRY_ALLOC:
  case SASS::WARPGROUP_ARRIVE: case SASS::WARPGROUP_DEPBAR:
  case SASS::WGMMA_COMMIT: case SASS::GMMA_SS: case SASS::GMMA_RS:
  case SASS::GMMA_SS_Z: case SASS::GMMA_RS_Z:
    return true;
  default:
    return false;
  }
}

/// A side of the branch that every thread taking it runs into an aligned
/// operation, which the other side cannot reach without coming back through
/// the branch: well-defined programs take such a branch warp-uniformly
/// (`if (threadIdx.x >= 256) { setmaxnreg.dec ... }`).
bool SASSUniformRegs::witnessesUniform(MachineBasicBlock &MBB) const {
  for (MachineBasicBlock *S : MBB.successors()) {
    if (S->pred_size() != 1)
      continue;
    for (MachineBasicBlock *X = S; X && MDT->dominates(S, X);
         X = ipdom(*PDT, X))
      if (AlignedAt.count(X))
        return true;
  }
  return false;
}

/// The block's conditional branch when its predicate is divergent.
MachineBasicBlock *
SASSUniformRegs::divergentBranch(MachineBasicBlock &MBB) const {
  for (MachineInstr &T : MBB.terminators()) {
    if (T.getOpcode() == SASS::BRAcu || T.getOpcode() == SASS::BRA_U_ANY)
      continue;
    if (!T.isConditionalBranch() && !(T.isBranch() && TII->isPredicated(T)))
      continue;
    for (const MachineOperand &MO : T.uses())
      if (isDivergent(MO))
        return witnessesUniform(MBB) || warpUniform(T) ? nullptr : &MBB;
  }
  return nullptr;
}

/// Whether part of the warp may go each way at the block's branch as the
/// final code runs it: any conditional branch not selected uniform (BRAcu),
/// which SASSReconverge brackets as divergent, unless it is warp-uniform by
/// construction (SASS::branchesWarpUniformly, which SASSReconverge's check
/// of uniform writes knows too). The divergence analysis may know more, or
/// assume more (witnessesUniform), but uniform registers are written only
/// where the warp runs converged in fact.
bool SASSUniformRegs::splitsWarp(MachineBasicBlock &MBB) const {
  if (MBB.succ_size() != 2)
    return false;
  for (MachineInstr &T : MBB.terminators()) {
    if (T.getOpcode() == SASS::BRAcu || T.getOpcode() == SASS::BRA_U_ANY)
      continue;
    if (T.isConditionalBranch() || (T.isBranch() && TII->isPredicated(T)))
      return !warpUniform(T);
  }
  return false;
}

void SASSUniformRegs::reach(MachineBasicBlock *From, MachineBasicBlock *Stop,
                            SmallPtrSetImpl<MachineBasicBlock *> &Out,
                            bool StopAtAligned) const {
  SmallVector<MachineBasicBlock *, 16> Stack;
  if (From != Stop && Out.insert(From).second)
    Stack.push_back(From);
  while (!Stack.empty()) {
    MachineBasicBlock *B = Stack.pop_back_val();
    if (StopAtAligned && GatherAt.count(B))
      continue;
    for (MachineBasicBlock *S : B->successors())
      if (S != Stop && Out.insert(S).second)
        Stack.push_back(S);
  }
}

void SASSUniformRegs::computeDivergence(MachineFunction &MF) {
  Div.clear();
  DivBranches.clear();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (isSeed(MI))
        markDefs(MI);
  bool Changed = true;
  while (Changed) {
    while (!Work.empty()) {
      Register R = Work.pop_back_val();
      for (MachineOperand &MO : MRI->use_nodbg_operands(R)) {
        MachineInstr &U = *MO.getParent();
        if (usePropagates(U, MO))
          markDefs(U);
      }
    }
    Changed = false;
    for (MachineBasicBlock &MBB : MF) {
      if (DivBranches.count(&MBB) || !divergentBranch(MBB) ||
          MBB.succ_size() != 2)
        continue;
      DivBranches.insert(&MBB);
      Changed = true;
      MachineBasicBlock *S1 = *MBB.succ_begin(), *S2 = *std::next(MBB.succ_begin());
      MachineBasicBlock *P = ipdom(*PDT, &MBB);
      // Joins: blocks both sides reach before the post-dominator, and the
      // post-dominator itself.
      SmallPtrSet<MachineBasicBlock *, 32> R1, R2;
      reach(S1, P, R1);
      reach(S2, P, R2);
      auto MarkPhis = [&](MachineBasicBlock *J) {
        for (MachineInstr &Phi : J->phis())
          markDefs(Phi);
      };
      for (MachineBasicBlock *B : R1)
        if (R2.count(B))
          MarkPhis(B);
      if (P)
        MarkPhis(P);
      // Values leaving a loop through a divergent exit.
      for (MachineLoop *L = MLI->getLoopFor(&MBB); L; L = L->getParentLoop()) {
        if (L->contains(S1) && L->contains(S2))
          break;
        for (MachineBasicBlock *B : L->blocks())
          for (MachineInstr &MI : *B)
            for (const MachineOperand &D : MI.all_defs()) {
              if (!D.getReg().isVirtual())
                continue;
              for (MachineInstr &U : MRI->use_nodbg_instructions(D.getReg()))
                if (!L->contains(U.getParent())) {
                  mark(D.getReg());
                  break;
                }
            }
      }
    }
  }
}

void SASSUniformRegs::computePlacement(MachineFunction &MF) {
  NoUDef.clear();
  ExitSide.clear();
  for (MachineBasicBlock &B : MF) {
    if (!splitsWarp(B))
      continue;
    MachineBasicBlock *MBB = &B;
    MachineBasicBlock *S1 = *MBB->succ_begin(),
                      *S2 = *std::next(MBB->succ_begin());
    MachineBasicBlock *P = ipdom(*PDT, MBB);
    SmallPtrSet<MachineBasicBlock *, 32> R1, R2;
    reach(S1, P, R1, true);
    reach(S2, P, R2, true);
    // Reconverging sides (SASSReconverge's BSSY/BSYNC): no uniform writes
    // until the post-dominator. Regions that may sleep get no barrier, and
    // regions that end at an exit none either: there the sides stay split
    // until a WARPSYNC.ALL.
    auto Sleeps = [](const SmallPtrSetImpl<MachineBasicBlock *> &S) {
      return any_of(S, [](MachineBasicBlock *B) {
        return any_of(*B, [](const MachineInstr &MI) {
          return MI.getOpcode() == SASS::NANOSLEEPi ||
                 MI.getOpcode() == SASS::NANOSLEEPr;
        });
      });
    };
    bool Barrier = P && !P->succ_empty() && !Sleeps(R1) && !Sleeps(R2);
    if (Barrier) {
      NoUDef.insert(R1.begin(), R1.end());
      NoUDef.insert(R2.begin(), R2.end());
      continue;
    }
    R1.clear();
    R2.clear();
    reach(S1, nullptr, R1, true);
    reach(S2, nullptr, R2, true);
    // Sides that part ways to exit: the smaller one is fenced off from
    // uniform registers when only exit blocks are shared.
    bool Disjoint = true;
    for (MachineBasicBlock *B : R1)
      if (R2.count(B) && B->getFirstNonDebugInstr() != B->getFirstTerminator())
        Disjoint = false;
    if (!Disjoint) {
      NoUDef.insert(R1.begin(), R1.end());
      NoUDef.insert(R2.begin(), R2.end());
      continue;
    }
    auto Size = [](const SmallPtrSetImpl<MachineBasicBlock *> &S) {
      size_t N = 0;
      for (MachineBasicBlock *B : S)
        N += B->size();
      return N;
    };
    SmallPtrSetImpl<MachineBasicBlock *> &E = Size(R1) <= Size(R2) ? R1 : R2;
    SmallPtrSetImpl<MachineBasicBlock *> &O = &E == &R1 ? R2 : R1;
    // An exiting side that uses uniform registers itself could see the
    // other side's.
    bool TouchesUR = false;
    for (MachineBasicBlock *B : E)
      for (MachineInstr &MI : *B)
        for (const MachineOperand &MO : MI.operands())
          if (MO.isReg() && MO.getReg() &&
              (MO.getReg().isVirtual()
                   ? isUniformClass(MO.getReg())
                   : (SASS::UGPR32RegClass.contains(MO.getReg()) ||
                      SASS::UGPR64RegClass.contains(MO.getReg())) &&
                         !MRI->isReserved(MO.getReg())))
            TouchesUR = true;
    if (TouchesUR) {
      NoUDef.insert(R1.begin(), R1.end());
      NoUDef.insert(R2.begin(), R2.end());
      continue;
    }
    for (MachineBasicBlock *B : E) {
      NoUDef.insert(B);
      ExitSide.insert(B);
    }
    for (MachineBasicBlock *B : O)
      if (E.count(B)) // shared exit blocks
        NoUDef.insert(B), ExitSide.insert(B);
  }
}

/// The uniform twin of a GPR instruction with the same operand list.
static unsigned uniformOpcode(unsigned Opc) {
  switch (Opc) {
  case SASS::LDC32: return SASS::ULDC32;
  case SASS::LDC64: return SASS::ULDC64;
  case SASS::LDCB32_I: return SASS::ULDCB32;
  case SASS::LDCB64_I: return SASS::ULDCB64;
  case SASS::MOVri: return SASS::UMOVi;
  case SASS::S2R_CTAIDX: return SASS::S2UR_CTAIDX;
  case SASS::S2R_CTAIDY: return SASS::S2UR_CTAIDY;
  case SASS::S2R_CTAIDZ: return SASS::S2UR_CTAIDZ;
  case SASS::S2R_CGA: return SASS::S2UR_CGA;
  case SASS::IADDrr: return SASS::UIADDrr;
  case SASS::IADDri: return SASS::UIADD3ri;
  case SASS::IADD3rrr: return SASS::UIADD3rrr;
  case SASS::IADD3rri: return SASS::UIADD3rri;
  case SASS::ISUBrr: return SASS::UISUBrr;
  case SASS::IMADrrr: return SASS::UIMADrrr;
  case SASS::IMADrir: return SASS::UIMADrir;
  case SASS::IMADWrrr: return SASS::UIMADWrrr;
  case SASS::IMADWrir: return SASS::UIMADWrir;
  case SASS::IMADWrr0: return SASS::UIMADWrr0;
  case SASS::IMADWUrrr: return SASS::UIMADWUrrr;
  case SASS::IMADWUrir: return SASS::UIMADWUrir;
  case SASS::IMADWUrr0: return SASS::UIMADWUrr0;
  case SASS::LOP3rrr: return SASS::ULOP3rrr;
  case SASS::LOP3rir: return SASS::ULOP3rir;
  case SASS::SHLrr: return SASS::USHLrr;
  case SASS::SHLri: return SASS::USHLri;
  case SASS::SRLrr: return SASS::USRLrr;
  case SASS::SRLri: return SASS::USRLri;
  case SASS::SRArr: return SASS::USRArr;
  case SASS::SRAri: return SASS::USRAri;
  case SASS::SHLU64HI: return SASS::USHLU64HI;
  case SASS::SHRU64: return SASS::USHRU64;
  case SASS::SHRS64: return SASS::USHRS64;
  case SASS::LEArrr: return SASS::ULEArrr;
  default: return 0;
  }
}

/// LDS with a uniform address register.
static unsigned ldsURForm(unsigned Opc) {
  switch (Opc) {
  case SASS::LDS32: return SASS::LDS32_UR;
  case SASS::LDS64: return SASS::LDS64_UR;
  case SASS::LDS128: return SASS::LDS128_UR;
  default: return 0;
  }
}

namespace {
/// A per-thread instruction rewritten to read one uniform register: the new
/// opcode, and the operand order (indices into the old instruction) with the
/// UR in slot B.
struct URUse {
  unsigned Opc = 0;
  SmallVector<unsigned, 6> Ops;
  unsigned B = 0;     // Index in Ops of the uniform operand.
  int Lut = -1;       // LOP3: the truth table after reordering.
  int CC = -1;        // ISETP: the condition after swapping.
  int Neg = -1;       // SEL: the select negation after swapping.
  bool ZeroA = false; // Slot A is RZ (a comparison with 0, swapped).
};
} // namespace

/// LOP3 truth table with sources X and Y (0 = a, 1 = b, 2 = c) exchanged.
static unsigned swapLut(unsigned Lut, unsigned X, unsigned Y) {
  unsigned Out = 0;
  for (unsigned I = 0; I < 8; ++I) {
    unsigned Bits[3] = {I >> 2 & 1, I >> 1 & 1, I & 1}; // a, b, c
    std::swap(Bits[X], Bits[Y]);
    unsigned J = Bits[0] << 2 | Bits[1] << 1 | Bits[2];
    Out |= (Lut >> J & 1) << I;
  }
  return Out;
}

/// ISETP condition with the operands exchanged (bits [2:0] LT LE GT GE EQ NE,
/// bit 3 unsigned).
static unsigned swapCC(unsigned CC) {
  static const unsigned Swapped[6] = {2, 3, 0, 1, 4, 5};
  return (CC & 8) | Swapped[CC & 7];
}

/// How \p MI can read its operand \p Idx (a uniform value) from a UR.
static URUse urUse(const MachineInstr &MI, unsigned Idx) {
  URUse U;
  auto Set = [&](unsigned Opc, std::initializer_list<unsigned> Ops,
                 unsigned B) {
    U.Opc = Opc;
    U.Ops.assign(Ops);
    U.B = B;
  };
  switch (MI.getOpcode()) {
  case SASS::IADDrr:
    if (Idx == 2) Set(SASS::IADDru, {1, 2}, 1);
    if (Idx == 1) Set(SASS::IADDru, {2, 1}, 1);
    break;
  case SASS::IADD3rrr:
    if (Idx == 2) Set(SASS::IADD3rur, {1, 2, 3}, 1);
    if (Idx == 1) Set(SASS::IADD3rur, {2, 1, 3}, 1);
    if (Idx == 3) Set(SASS::IADD3rur, {1, 3, 2}, 1);
    break;
  case SASS::ISUBrr:
    if (Idx == 2) Set(SASS::ISUBru, {1, 2}, 1);
    if (Idx == 1) Set(SASS::ISUBur, {1, 2}, 0);
    break;
  case SASS::IMADrrr:
  case SASS::IMADWrrr:
  case SASS::IMADWUrrr: {
    unsigned Opc = MI.getOpcode() == SASS::IMADrrr    ? SASS::IMADrur
                   : MI.getOpcode() == SASS::IMADWrrr ? SASS::IMADWrur
                                                      : SASS::IMADWUrur;
    if (Idx == 2) Set(Opc, {1, 2, 3}, 1);
    if (Idx == 1) Set(Opc, {2, 1, 3}, 1);
    break;
  }
  case SASS::IMADWrr0:
  case SASS::IMADWUrr0: {
    unsigned Opc =
        MI.getOpcode() == SASS::IMADWrr0 ? SASS::IMADWru0 : SASS::IMADWUru0;
    if (Idx == 2) Set(Opc, {1, 2}, 1);
    if (Idx == 1) Set(Opc, {2, 1}, 1);
    break;
  }
  case SASS::LOP3rrr: {
    unsigned Lut = MI.getOperand(4).getImm();
    if (Idx == 2) Set(SASS::LOP3rur, {1, 2, 3, 4}, 1), U.Lut = Lut;
    if (Idx == 1) Set(SASS::LOP3rur, {2, 1, 3, 4}, 1), U.Lut = swapLut(Lut, 0, 1);
    if (Idx == 3) Set(SASS::LOP3rur, {1, 3, 2, 4}, 1), U.Lut = swapLut(Lut, 1, 2);
    break;
  }
  case SASS::SHLrr:
    if (Idx == 2) Set(SASS::SHLru, {1, 2}, 1);
    break;
  case SASS::SRLrr:
    if (Idx == 2) Set(SASS::SRLru, {1, 2}, 1);
    break;
  case SASS::SRArr:
    if (Idx == 2) Set(SASS::SRAru, {1, 2}, 1);
    break;
  case SASS::SHLU64HIrr:
    if (Idx == 2) Set(SASS::SHLU64HIru, {1, 2, 3}, 1);
    break;
  case SASS::LEArrr:
    if (Idx == 2) Set(SASS::LEAru, {1, 2, 3}, 1);
    break;
  case SASS::ISETPri:
    // x cc 0 as 0 cc' x: ISETP P, PT, RZ, UR (ptxas: ISETP.NE.AND P4, PT, RZ, UR10).
    if (Idx == 1 && MI.getOperand(2).getImm() == 0) {
      Set(SASS::ISETPru, {1, 1, 3, 4, 5}, 1);
      U.CC = swapCC(MI.getOperand(3).getImm());
      U.ZeroA = true;
    }
    break;
  case SASS::ISETPORrr:
    if (Idx == 2) Set(SASS::ISETPORru, {1, 2, 3, 4, 5}, 1);
    if (Idx == 1) {
      Set(SASS::ISETPORru, {2, 1, 3, 4, 5}, 1);
      U.CC = swapCC(MI.getOperand(3).getImm());
    }
    break;
  case SASS::ISETPrr:
    if (Idx == 2) Set(SASS::ISETPru, {1, 2, 3, 4, 5}, 1);
    if (Idx == 1) {
      Set(SASS::ISETPru, {2, 1, 3, 4, 5}, 1);
      U.CC = swapCC(MI.getOperand(3).getImm());
    }
    break;
  case SASS::SELrr:
    if (Idx == 2) Set(SASS::SELru, {1, 2, 3, 4}, 1);
    if (Idx == 1) {
      Set(SASS::SELru, {2, 1, 3, 4}, 1);
      U.Neg = !MI.getOperand(4).getImm();
    }
    break;
  case SASS::IADD3COrr:
    if (Idx == 3) Set(SASS::IADD3COru, {2, 3}, 1);
    if (Idx == 2) Set(SASS::IADD3COru, {3, 2}, 1);
    break;
  case SASS::IADD3Xrr:
    if (Idx == 2) Set(SASS::IADD3Xru, {1, 2, 3}, 1);
    if (Idx == 1) Set(SASS::IADD3Xru, {2, 1, 3}, 1);
    break;
  default:
    break;
  }
  return U;
}
/// IMAD.WIDE[.U32] a, 2^s, c whose 64-bit addend c can stay in a UR pair:
/// the shift s, else -1. Rewritten as a carry pair reading the UR (IADD3 /
/// LEA with carry-out, then IADD3.X / LEA.HI.X[.SX32]) instead of two MOV
/// R, UR and the IMAD.WIDE.
static int wideAddShift(const MachineInstr &MI, unsigned Idx) {
  unsigned Opc = MI.getOpcode();
  if ((Opc != SASS::IMADWrir && Opc != SASS::IMADWUrir) || Idx != 3 ||
      MI.getOperand(3).getSubReg())
    return -1;
  int64_t K = MI.getOperand(2).getImm();
  if (K <= 0 || K > (int64_t(1) << 31) || !isPowerOf2_64(K))
    return -1;
  return Log2_64(K);
}

/// Uniform values computed where the warp may be split (kernel parameters
/// loaded inside a divergent row, the arithmetic on them) move to the
/// nearest dominator that runs converged, when their operands are available
/// there and it lies in every loop they do; as ptxas does, the uniform
/// register is then written once before the region and only read inside.
/// promote() moves back what it does not keep.
void SASSUniformRegs::hoist(MachineFunction &MF) {
  Hoisted.clear();
  Pinned.clear();
  DenseMap<MachineBasicBlock *, SmallVector<MachineInstr *, 8>> Into;
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *D : RPOT) {
    if (!NoUDef.count(D) || ExitSide.count(D))
      continue;
    for (MachineInstr &MI : make_early_inc_range(*D)) {
      if (MI.getNumExplicitDefs() != 1 || !MI.getOperand(0).isReg() ||
          MI.getOperand(0).getSubReg() || !fenced(MI))
        continue;
      Register V = MI.getOperand(0).getReg();
      unsigned Opc = MI.getOpcode();
      // An immediate move costs no less than its MOV R, UR would.
      if (!isGPRValue(V) || Div.count(V) || !uniformOpcode(Opc) ||
          Opc == SASS::MOVri || TII->isPredicated(MI) ||
          !MRI->hasOneDef(V))
        continue;
      MachineBasicBlock *T = nullptr;
      for (MachineDomTreeNode *N = MDT->getNode(D)->getIDom(); N;
           N = N->getIDom())
        if (!fencedAtEnd(N->getBlock())) {
          T = N->getBlock();
          break;
        }
      if (!T)
        continue;
      if (const MachineLoop *L = MLI->getLoopFor(T); L && !L->contains(D))
        continue;
      bool Available = all_of(MI.uses(), [&](const MachineOperand &MO) {
        if (!MO.isReg())
          return MO.isImm();
        Register R = MO.getReg();
        if (R.isPhysical())
          return R == SASS::RZ || R == SASS::PT;
        const MachineInstr *Def = MRI->getVRegDef(R);
        return Def && !Def->isPHI() ? MDT->dominates(Def->getParent(), T)
                                    : Def && MDT->properlyDominates(
                                                 Def->getParent(), T);
      });
      if (!Available)
        continue;
      // The same value hoisted from another block (each unrolled row loads
      // the parameter again): one def serves both, also when it went to a
      // dominator of T (the previous row's converged head).
      MachineInstr *Same = nullptr;
      for (auto &[HB, Hs] : Into)
        if (HB == T || MDT->dominates(HB, T))
          for (MachineInstr *H : Hs)
            if (H->isIdenticalTo(MI, MachineInstr::IgnoreVRegDefs) &&
                MRI->getRegClass(H->getOperand(0).getReg()) ==
                    MRI->getRegClass(V))
              Same = H;
      if (Same) {
        MRI->replaceRegWith(V, Same->getOperand(0).getReg());
        Order.erase(&MI);
        MI.eraseFromParent();
        // It and the hoisted values it reads stay.
        SmallVector<MachineInstr *, 4> Work{Same};
        while (!Work.empty()) {
          MachineInstr *X = Work.pop_back_val();
          if (!Pinned.insert(X).second)
            continue;
          for (const MachineOperand &MO : X->uses())
            if (MO.isReg() && MO.getReg().isVirtual())
              if (MachineInstr *D = MRI->getVRegDef(MO.getReg());
                  D && Hoisted.count(D))
                Work.push_back(D);
        }
        continue;
      }
      Into[T].push_back(&MI);
      auto At = T->getFirstTerminator();
      T->splice(At, D, MI.getIterator());
      Hoisted[&MI] = {D, Order.lookup(&MI)};
      Order[&MI] = At == T->end() ? ~0u : Order.lookup(&*At);
    }
  }
}

/// Back into its block, after the operands it reads there.
void SASSUniformRegs::unhoist(MachineInstr &MI) {
  auto [D, Ord] = Hoisted.lookup(&MI);
  Order[&MI] = Ord;
  Hoisted.erase(&MI);
  auto At = D->getFirstNonPHI();
  for (auto I = D->begin(), E = D->end(); I != E; ++I)
    for (const MachineOperand &MO : MI.uses())
      if (MO.isReg() && MO.getReg().isVirtual() &&
          I->definesRegister(MO.getReg(), /*TRI=*/nullptr))
        At = std::next(I);
  D->splice(At, MI.getParent(), MI.getIterator());
}

/// Uniform registers live at once (32-bit units) stay within URBudget:
/// beyond the register file the allocator rematerializes (ULDC, UMOV) or
/// splits uniform values next to their users, which may be where the warp
/// is split. The values live over the most blocks at the worst point lose
/// their promotion first.
void SASSUniformRegs::trimToBudget(MachineFunction &MF,
                                   DenseSet<Register> &C) {
  // Values: the candidates and the uniform registers already in use.
  SmallVector<Register, 256> Vals;
  SmallVector<unsigned, 256> Weight;
  DenseMap<Register, unsigned> Idx;
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(R);
    unsigned W = 0;
    if (C.count(R))
      W = RC == &SASS::GPR64RegClass ? 2 : 1;
    else if (RC == &SASS::UGPR32RegClass)
      W = 1;
    else if (RC == &SASS::UGPR64RegClass)
      W = 2;
    else if (RC == &SASS::UGPR128RegClass)
      W = 4;
    if (!W)
      continue;
    Idx[R] = Vals.size();
    Vals.push_back(R);
    Weight.push_back(W);
  }
  unsigned N = Vals.size();
  if (!N)
    return;
  // Liveness at block boundaries.
  unsigned NB = MF.getNumBlockIDs();
  std::vector<BitVector> LiveIn(NB, BitVector(N)), LiveOut(NB, BitVector(N)),
      Use(NB, BitVector(N)), Def(NB, BitVector(N)), PhiOut(NB, BitVector(N));
  for (MachineBasicBlock &B : MF) {
    unsigned K = B.getNumber();
    for (MachineInstr &MI : B) {
      if (MI.isDebugInstr())
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        auto It = Idx.find(MO.getReg());
        if (It == Idx.end())
          continue;
        if (MO.isDef()) {
          Def[K].set(It->second);
        } else if (MI.isPHI()) {
          MachineBasicBlock *P = MI.getOperand(MO.getOperandNo() + 1).getMBB();
          PhiOut[P->getNumber()].set(It->second);
        } else if (!Def[K].test(It->second)) {
          Use[K].set(It->second);
        }
      }
    }
  }
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (MachineBasicBlock *B : post_order(&MF)) {
      unsigned K = B->getNumber();
      BitVector Out = PhiOut[K];
      for (MachineBasicBlock *S : B->successors())
        Out |= LiveIn[S->getNumber()];
      BitVector In = Out;
      In.reset(Def[K]);
      In |= Use[K];
      if (Out != LiveOut[K] || In != LiveIn[K]) {
        LiveOut[K] = std::move(Out);
        LiveIn[K] = std::move(In);
        Changed = true;
      }
    }
  }
  SmallVector<unsigned, 256> Span(N);
  for (MachineBasicBlock &B : MF) {
    BitVector Here = LiveIn[B.getNumber()];
    Here |= Def[B.getNumber()];
    for (unsigned I : Here.set_bits())
      ++Span[I];
  }
  // Per block, bottom up: defs (2i+1) then uses (2i) of each instruction,
  // then -1 where the pressure is taken.
  std::vector<SmallVector<int, 32>> Events(NB);
  for (MachineBasicBlock &B : MF) {
    SmallVector<int, 32> &Ev = Events[B.getNumber()];
    for (MachineInstr &MI : reverse(B)) {
      if (MI.isDebugInstr())
        continue;
      for (bool D : {true, false}) {
        if (!D && MI.isPHI())
          break;
        for (const MachineOperand &MO : MI.operands())
          if (MO.isReg() && MO.getReg().isVirtual() && MO.isDef() == D)
            if (auto It = Idx.find(MO.getReg()); It != Idx.end())
              Ev.push_back(2 * It->second + D);
      }
      Ev.push_back(-1);
    }
  }

  // The worst point under the remaining candidates, then drop one there.
  BitVector Active(N, true);
  for (;;) {
    unsigned Worst = 0;
    BitVector WorstLive;
    for (MachineBasicBlock &B : MF) {
      BitVector Live = LiveOut[B.getNumber()];
      Live &= Active;
      unsigned P = 0;
      for (unsigned I : Live.set_bits())
        P += Weight[I];
      auto Note = [&] {
        if (P > Worst) {
          Worst = P;
          WorstLive = Live;
        }
      };
      Note();
      for (int E : Events[B.getNumber()]) {
        if (E < 0) {
          Note();
          continue;
        }
        unsigned I = E >> 1;
        if (E & 1) {
          if (Live.test(I)) {
            Live.reset(I);
            P -= Weight[I];
          }
        } else if (Active.test(I) && !Live.test(I)) {
          Live.set(I);
          P += Weight[I];
        }
      }
    }
    if (Worst <= URBudget) {
      DEBUG_WITH_TYPE("sass-uniform-regs",
                      dbgs() << "  uniform pressure " << Worst << "\n");
      return;
    }
    // Hoisted values first (they relieved no GPR, only saved a load in
    // split code), the longest first; then the values relieving the GPRs
    // least: the shortest.
    auto Key = [&](unsigned I) {
      MachineInstr *D = MRI->getVRegDef(Vals[I]);
      bool H = D && Hoisted.count(D);
      return std::make_pair(!H, H ? -int(Span[I]) : int(Span[I]));
    };
    int Victim = -1;
    for (unsigned I : WorstLive.set_bits())
      if (C.count(Vals[I]) && (Victim < 0 || Key(I) < Key(Victim)))
        Victim = I;
    if (Victim < 0)
      return;
    Active.reset(Victim);
    C.erase(Vals[Victim]);
  }
}

bool SASSUniformRegs::promote(MachineFunction &MF) {
  hoist(MF);
  // Candidates: uniform single-def GPR values with a uniform twin, outside
  // the regions that must not write uniform registers.
  DenseSet<Register> C;
  // Loops with a branch that may split the warp: if later CFG changes
  // (if-conversion, block placement) leave its region without a barrier,
  // the iteration ends split and a uniform induction variable would be
  // stepped once per group (m_fdivs: a uniform loop counter read past the
  // end of its buffer). Their header PHIs and the values they carry stay
  // per thread.
  SmallPtrSet<const MachineLoop *, 8> SplitLoops;
  for (MachineBasicBlock &MBB : MF)
    if (splitsWarp(MBB))
      for (MachineLoop *L = MLI->getLoopFor(&MBB); L; L = L->getParentLoop())
        SplitLoops.insert(L);
  auto Carried = [&](const MachineInstr &MI, Register D) {
    if (MI.isPHI()) {
      const MachineLoop *L = MLI->getLoopFor(MI.getParent());
      return L && L->getHeader() == MI.getParent() && SplitLoops.count(L);
    }
    for (const MachineInstr &U : MRI->use_nodbg_instructions(D))
      if (U.isPHI()) {
        const MachineLoop *L = MLI->getLoopFor(U.getParent());
        if (L && L->getHeader() == U.getParent() && SplitLoops.count(L) &&
            L->contains(MI.getParent()))
          return true;
      }
    return false;
  };
  auto Movable = [&](const MachineInstr &MI) {
    if (MI.getNumExplicitDefs() != 1 || !MI.getOperand(0).isReg())
      return false;
    Register D = MI.getOperand(0).getReg();
    if (!isGPRValue(D) || Div.count(D) || MI.getOperand(0).getSubReg() ||
        fenced(MI) || Carried(MI, D))
      return false;
    if (MI.isPHI()) {
      for (unsigned I = 2, E = MI.getNumOperands(); I < E; I += 2)
        if (fencedAtEnd(MI.getOperand(I).getMBB()))
          return false;
      return true;
    }
    if (MI.isImplicitDef())
      return true;
    if (MI.isCopy() || MI.isRegSequence()) {
      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && (!MO.getReg().isVirtual() ||
                           (!isGPRValue(MO.getReg()) && !isUniformClass(MO.getReg()))))
          return false;
      return true;
    }
    if (!uniformOpcode(MI.getOpcode()) || TII->isPredicated(MI))
      return false;
    // Registers and immediates only (a frame index becomes R1 + offset).
    for (const MachineOperand &MO : MI.uses())
      if ((MO.isReg() && MO.getReg().isPhysical() && MO.getReg() != SASS::RZ &&
           MO.getReg() != SASS::PT) ||
          (!MO.isReg() && !MO.isImm()))
        return false;
    return true;
  };
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (Movable(MI))
        C.insert(MI.getOperand(0).getReg());
  DEBUG_WITH_TYPE("sass-uniform-regs", {
    unsigned Uni = 0, Vals = 0;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB)
        for (const MachineOperand &MO : MI.all_defs())
          if (isGPRValue(MO.getReg())) {
            ++Vals;
            Uni += !Div.count(MO.getReg());
          }
    dbgs() << "  " << Uni << "/" << Vals << " GPR values uniform, " << C.size()
           << " movable\n";
  });

  // Where a value is read: the user's block, or a PHI's incoming block.
  auto UseBlock = [](const MachineOperand &MO) {
    const MachineInstr &U = *MO.getParent();
    if (U.isPHI())
      return U.getOperand(MO.getOperandNo() + 1).getMBB();
    return const_cast<MachineBasicBlock *>(U.getParent());
  };
  // Direct readers of a UR: R2UR becomes a copy, LDS takes [UR+imm].
  auto ReadsUR = [&](const MachineOperand &MO) {
    const MachineInstr &U = *MO.getParent();
    unsigned Opc = U.getOpcode();
    if (Opc == SASS::R2UR || Opc == SASS::R2UR_G)
      return true;
    if (urUse(U, MO.getOperandNo()).Opc || wideAddShift(U, MO.getOperandNo()) >= 0)
      return true;
    return ldsURForm(Opc) && MO.getOperandNo() == 1 && !MO.getSubReg() &&
           MRI->getRegClass(MO.getReg()) == &SASS::GPR32RegClass;
  };
  // Keep a value only where it pays: another uniform instruction reads it,
  // or it lives past its block. Never read in an exiting side.
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB) {
        if (!MI.getNumOperands() || !MI.getOperand(0).isReg() ||
            !C.count(MI.getOperand(0).getReg()))
          continue;
        Register V = MI.getOperand(0).getReg();
        // A hoisted value pays when it saves a load (a kernel parameter, a
        // CTA id) in split code, or is read as a uniform register there.
        MachineBasicBlock *From = Hoisted.lookup(&MI).first;
        bool Pays = From && (MI.mayLoad() || TII->getName(MI.getOpcode())
                                                 .starts_with("S2R"));
        bool Fenced = false;
        // An immediate read only by PHIs (accumulators zeroed for a loop)
        // is cheaper as MOV R, imm at each incoming edge than as UMOV and
        // MOV R, UR: one UR copied into 32 accumulators became a MOV R, UR
        // and 32 MOV R, R. Read elsewhere, the UR still frees a GPR.
        // A zero needs no register of either kind past its block: per-thread
        // readers rematerialize MOV R, 0x0 (or pair them as CS2R), uniform
        // ones read URZ (ToUR).
        bool PhiImm = MI.getOpcode() == SASS::MOVri &&
                      (MI.getOperand(1).getImm() == 0 ||
                       all_of(MRI->use_nodbg_instructions(V),
                              [](const MachineInstr &U) { return U.isPHI(); }));
        for (MachineOperand &MO : MRI->use_nodbg_operands(V)) {
          MachineInstr &U = *MO.getParent();
          MachineBasicBlock *B = UseBlock(MO);
          if (ExitSide.count(B))
            Fenced = true;
          if ((!From && !PhiImm && (B != &MBB || U.isPHI())) || ReadsUR(MO) ||
              (U.getNumOperands() && U.getOperand(0).isReg() &&
               U.getOperand(0).isDef() && C.count(U.getOperand(0).getReg())))
            Pays = true;
        }
        // In a loop, an operand left per thread enters by R2UR every
        // iteration, and the result usually goes back by MOV R, UR: a
        // round trip of ~25 cycles of latency on what was one vector
        // instruction (a stage index (i + 2) % 3 from a per-thread loop
        // counter: R2UR, UIADD3, MOV, R2UR, UIMAD, USHF). Such values stay
        // per thread, as do the ones computed from them. Innermost loops
        // only: an outer loop around the whole body (a speculative
        // segment's redo loop) runs once.
        const MachineLoop *InLoop = MLI->getLoopFor(&MBB);
        if (Pays && !Fenced && !From && InLoop && InLoop->isInnermost())
          for (const MachineOperand &MO : MI.uses())
            if (MO.isReg() && MO.getReg().isVirtual() && !C.count(MO.getReg()) &&
                isGPRValue(MO.getReg())) {
              Pays = false;
              break;
            }
        if (!Pays || Fenced) {
          C.erase(V);
          Changed = true;
        }
      }
  }
  // Within the budget, then hoisted values not kept go back, with the
  // hoisted values reading them. Going back lengthens what they read: the
  // budget is checked again.
  bool Moved = false;
  for (;;) {
    trimToBudget(MF, C);
    for (bool Changed = true; Changed;) {
      Changed = false;
      for (auto &[MI, From] : Hoisted) {
        Register V = MI->getOperand(0).getReg();
        if (!C.count(V))
          continue;
        for (const MachineOperand &MO : MI->uses())
          if (MO.isReg() && MO.getReg().isVirtual())
            if (MachineInstr *D = MRI->getVRegDef(MO.getReg());
                D && Hoisted.count(D) && !Pinned.count(D) &&
                !C.count(D->getOperand(0).getReg())) {
              C.erase(V);
              Changed = true;
              break;
            }
      }
    }
    SmallVector<MachineInstr *, 16> Back;
    for (auto &[MI, From] : Hoisted)
      if (!C.count(MI->getOperand(0).getReg()) && !Pinned.count(MI))
        Back.push_back(MI);
    if (Back.empty())
      break;
    // Operands first: a value goes back after what it reads in its block.
    llvm::sort(Back, [&](MachineInstr *A, MachineInstr *B) {
      return Hoisted.lookup(A).second < Hoisted.lookup(B).second;
    });
    for (MachineInstr *MI : Back)
      unhoist(*MI);
    Moved = true;
  }
  DEBUG_WITH_TYPE("sass-uniform-regs",
                  dbgs() << "  " << C.size() << " promoted, " << Hoisted.size()
                         << " hoisted\n");
  if (C.empty())
    return Moved;

  // Uniform twins.
  UMap.clear();
  for (Register V : C)
    UMap[V] = MRI->createVirtualRegister(
        MRI->getRegClass(V) == &SASS::GPR64RegClass ? &SASS::UGPR64RegClass
                                                    : &SASS::UGPR32RegClass);

  // R2UR of a uniform GPR operand computed outside C, one per block (at the
  // first reader, or the end of a PHI's incoming block).
  DenseMap<std::tuple<Register, unsigned, MachineBasicBlock *, unsigned>, Register>
      In, Out;
  auto Guard = [](const MachineInstrBuilder &MIB) { MIB.addReg(SASS::PT).addImm(0); };
  auto ToUR = [&](const MachineOperand &MO, MachineBasicBlock &B,
                  MachineBasicBlock::iterator At, bool AtEnd) -> std::pair<Register, unsigned> {
    Register W = MO.getReg();
    if (W == SASS::RZ)
      return {SASS::URZ, 0};
    if (const MachineInstr *D = W.isVirtual() ? MRI->getUniqueVRegDef(W) : nullptr;
        D && D->getOpcode() == SASS::MOVri && D->getOperand(1).getImm() == 0 &&
        !C.count(W) && !MO.getSubReg())
      return {SASS::URZ, 0};
    if (auto It = UMap.find(W); It != UMap.end())
      return {It->second, MO.getSubReg()};
    if (isUniformClass(W))
      return {W, MO.getSubReg()};
    Register &U = In[{W, MO.getSubReg(), &B, AtEnd}];
    if (U)
      return {U, 0};
    const DebugLoc &DL = At == B.end() ? DebugLoc() : At->getDebugLoc();
    bool Wide = !MO.getSubReg() && MRI->getRegClass(W) == &SASS::GPR64RegClass;
    if (!Wide) {
      U = MRI->createVirtualRegister(&SASS::UGPR32RegClass);
      Guard(BuildMI(B, At, DL, TII->get(SASS::R2UR), U).addReg(W, {}, MO.getSubReg()));
      return {U, 0};
    }
    Register Lo = MRI->createVirtualRegister(&SASS::UGPR32RegClass);
    Register Hi = MRI->createVirtualRegister(&SASS::UGPR32RegClass);
    Guard(BuildMI(B, At, DL, TII->get(SASS::R2UR), Lo).addReg(W, {}, SASS::sub0));
    Guard(BuildMI(B, At, DL, TII->get(SASS::R2UR), Hi).addReg(W, {}, SASS::sub1));
    U = MRI->createVirtualRegister(&SASS::UGPR64RegClass);
    BuildMI(B, At, DL, TII->get(TargetOpcode::REG_SEQUENCE), U)
        .addReg(Lo).addImm(SASS::sub0).addReg(Hi).addImm(SASS::sub1);
    return {U, 0};
  };
  // GPR copy of a promoted value for per-thread readers.
  auto ToGPR = [&](Register V, unsigned Sub, MachineBasicBlock &B,
                   MachineBasicBlock::iterator At, bool AtEnd) {
    Register &R = Out[{V, Sub, &B, AtEnd}];
    if (R)
      return R;
    const DebugLoc &DL = At == B.end() ? DebugLoc() : At->getDebugLoc();
    Register U = UMap[V];
    if (Sub || MRI->getRegClass(V) == &SASS::GPR32RegClass) {
      R = MRI->createVirtualRegister(&SASS::GPR32RegClass);
      Guard(BuildMI(B, At, DL, TII->get(SASS::MOVur), R).addReg(U, {}, Sub));
      return R;
    }
    Register Lo = MRI->createVirtualRegister(&SASS::GPR32RegClass);
    Register Hi = MRI->createVirtualRegister(&SASS::GPR32RegClass);
    Guard(BuildMI(B, At, DL, TII->get(SASS::MOVur), Lo).addReg(U, {}, SASS::sub0));
    Guard(BuildMI(B, At, DL, TII->get(SASS::MOVur), Hi).addReg(U, {}, SASS::sub1));
    R = MRI->createVirtualRegister(&SASS::GPR64RegClass);
    BuildMI(B, At, DL, TII->get(TargetOpcode::REG_SEQUENCE), R)
        .addReg(Lo).addImm(SASS::sub0).addReg(Hi).addImm(SASS::sub1);
    return R;
  };

  SmallVector<MachineInstr *, 64> Dead;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      bool Def = MI.getNumOperands() && MI.getOperand(0).isReg() &&
                 MI.getOperand(0).isDef() && C.count(MI.getOperand(0).getReg());
      if (Def) {
        // The uniform twin, operand for operand.
        Register U = UMap[MI.getOperand(0).getReg()];
        const DebugLoc &DL = MI.getDebugLoc();
        if (MI.isPHI()) {
          MachineInstrBuilder MIB =
              BuildMI(MBB, MI, DL, TII->get(TargetOpcode::PHI), U);
          for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
            MachineBasicBlock *Pred = MI.getOperand(I + 1).getMBB();
            auto [R, Sub] = ToUR(MI.getOperand(I), *Pred,
                                 Pred->getFirstTerminator(), true);
            MIB.addReg(R, {}, Sub).addMBB(Pred);
          }
        } else {
          unsigned Opc = MI.isCopy() || MI.isRegSequence() || MI.isImplicitDef()
                             ? MI.getOpcode()
                             : uniformOpcode(MI.getOpcode());
          MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII->get(Opc), U);
          unsigned NumOps = MI.isCopy() || MI.isRegSequence() || MI.isImplicitDef()
                                ? MI.getNumOperands()
                                : MI.getNumExplicitOperands() - 2; // guard
          for (unsigned I = 1; I < NumOps; ++I) {
            const MachineOperand &MO = MI.getOperand(I);
            if (MO.isReg()) {
              auto [R, Sub] = ToUR(MO, MBB, MIB.getInstr()->getIterator(), false);
              MIB.addReg(R, {}, Sub);
            } else {
              MIB.add(MO);
            }
          }
          if (!MI.isCopy() && !MI.isRegSequence() && !MI.isImplicitDef())
            Guard(MIB);
        }
        Dead.push_back(&MI);
        continue;
      }
      // A per-thread reader of promoted values.
      if (MI.isDebugInstr())
        continue;
      unsigned Opc = MI.getOpcode();
      if ((Opc == SASS::R2UR || Opc == SASS::R2UR_G) &&
          C.count(MI.getOperand(1).getReg())) {
        // Its readers take the uniform register itself: no copy, which
        // would write a uniform register where the R2UR was (maybe split
        // code, e.g. the operands of an elected bulk copy).
        const MachineOperand &Src = MI.getOperand(1);
        Register Dst = MI.getOperand(0).getReg(), U = UMap[Src.getReg()];
        // A fixed uniform register (the operands of TMA instructions) keeps
        // its copy, at the same place as the R2UR it replaces.
        if (!Src.getSubReg() && Dst.isVirtual() &&
            MRI->constrainRegClass(U, MRI->getRegClass(Dst))) {
          MRI->replaceRegWith(Dst, U);
          MI.getOperand(0).setReg(Dst); // Erased below.
        } else
          BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), Dst)
              .addReg(U, {}, Src.getSubReg());
        Dead.push_back(&MI);
        continue;
      }
      if (unsigned UOpc = ldsURForm(Opc)) {
        const MachineOperand &A = MI.getOperand(1);
        if (A.isReg() && C.count(A.getReg()) && !A.getSubReg()) {
          MachineInstrBuilder MIB =
              BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(UOpc),
                      MI.getOperand(0).getReg())
                  .addReg(UMap[A.getReg()]);
          for (unsigned I = 2, E = MI.getNumOperands(); I < E; ++I)
            MIB.add(MI.getOperand(I));
          MIB.cloneMemRefs(MI);
          Dead.push_back(&MI);
          continue;
        }
      }
      if (int S = wideAddShift(MI, 3); S >= 0 && C.count(MI.getOperand(3).getReg()) &&
                                       !TII->isPredicated(MI)) {
        const DebugLoc &DL = MI.getDebugLoc();
        Register UC = UMap[MI.getOperand(3).getReg()];
        Register A = MI.getOperand(1).getReg();
        unsigned ASub = MI.getOperand(1).getSubReg();
        if (C.count(A))
          A = ToGPR(A, ASub, MBB, MI.getIterator(), false), ASub = 0;
        bool Signed = MI.getOpcode() == SASS::IMADWrir;
        Register Lo = MRI->createVirtualRegister(&SASS::GPR32RegClass);
        Register Hi = MRI->createVirtualRegister(&SASS::GPR32RegClass);
        Register Co = MRI->createVirtualRegister(&SASS::PREDRegClass);
        if (S == 0)
          Guard(BuildMI(MBB, MI, DL, TII->get(SASS::IADD3COru), Lo)
              .addReg(Co, RegState::Define).addReg(A, {}, ASub).addReg(UC, {}, SASS::sub0));
        else
          Guard(BuildMI(MBB, MI, DL, TII->get(SASS::LEACOru), Lo)
              .addReg(Co, RegState::Define).addReg(A, {}, ASub).addReg(UC, {}, SASS::sub0)
              .addImm(S));
        if (S == 0 && !Signed)
          Guard(BuildMI(MBB, MI, DL, TII->get(SASS::IADD3Xru), Hi)
              .addReg(SASS::RZ).addReg(UC, {}, SASS::sub1).addReg(Co));
        else // (sign word of a: a >> 31, the high word of a << 1)
          Guard(BuildMI(MBB, MI, DL,
                  TII->get(Signed ? SASS::LEAHIXSX32ru : SASS::LEAHIXru), Hi)
              .addReg(A, {}, ASub).addReg(UC, {}, SASS::sub1)
              .addImm(Signed && S == 0 ? 1 : S).addReg(Co));
        BuildMI(MBB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE),
                MI.getOperand(0).getReg())
            .addReg(Lo).addImm(SASS::sub0).addReg(Hi).addImm(SASS::sub1);
        Dead.push_back(&MI);
        continue;
      }
      // One promoted operand read straight from its UR.
      MachineInstr *User = &MI;
      if (!MI.isPHI())
        for (unsigned I = MI.getNumExplicitDefs(), E = MI.getNumOperands();
             I < E; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (!MO.isReg() || !MO.isUse() || !C.count(MO.getReg()))
            continue;
          URUse U = urUse(MI, I);
          if (!U.Opc)
            continue;
          MachineInstrBuilder MIB =
              BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(U.Opc));
          for (const MachineOperand &D : MI.defs())
            MIB.add(D);
          for (auto [K, From] : enumerate(U.Ops)) {
            const MachineOperand &Src = MI.getOperand(From);
            if (K == U.B)
              MIB.addReg(UMap[Src.getReg()], {}, Src.getSubReg());
            else if (K == 0 && U.ZeroA)
              MIB.addReg(SASS::RZ);
            else if (int(K) == 2 && U.CC >= 0)
              MIB.addImm(U.CC);
            else if (int(K) == 3 && U.Lut >= 0)
              MIB.addImm(U.Lut);
            else if (int(K) == 3 && U.Neg >= 0)
              MIB.addImm(U.Neg);
            else
              MIB.add(Src);
          }
          unsigned G = MI.getNumExplicitOperands() - 2;
          MIB.add(MI.getOperand(G)).add(MI.getOperand(G + 1));
          Dead.push_back(&MI);
          User = MIB.getInstr();
          break;
        }
      for (unsigned I = 0, E = User->getNumOperands(); I < E; ++I) {
        MachineInstr &MI = *User;
        MachineOperand &MO = MI.getOperand(I);
        if (!MO.isReg() || !MO.isUse() || !C.count(MO.getReg()))
          continue;
        Register R;
        if (MI.isPHI()) {
          MachineBasicBlock *Pred = MI.getOperand(I + 1).getMBB();
          R = ToGPR(MO.getReg(), MO.getSubReg(), *Pred,
                    Pred->getFirstTerminator(), true);
        } else {
          R = ToGPR(MO.getReg(), MO.getSubReg(), MBB, MI.getIterator(), false);
        }
        MO.setReg(R);
        MO.setSubReg(0);
        MO.setIsKill(false);
      }
    }
  }
  for (MachineInstr *MI : Dead)
    MI->eraseFromParent();
  // Debug users of the removed GPR values.
  for (Register V : C)
    for (MachineOperand &MO : make_early_inc_range(MRI->reg_operands(V)))
      if (MO.getParent()->isDebugInstr())
        MO.setReg(Register());
  for (auto &[V, U] : UMap)
    MRI->clearKillFlags(U);
  return true;
}

/// The shared window base, (SR_CgaCtaId << 24) + 0x400, is selected per block
/// (SASSISD::SMEM_BASE), and MachineCSE keeps the MOV of 0x400 next to each
/// LEA: every block that addresses shared memory computes it again (MOV, MOV
/// R, UR, LEA in each unrolled copy row). One def at the kernel entry instead,
/// which promotion keeps in a UR for the whole kernel as ptxas does (UR4).
static bool unifySmemBase(MachineFunction &MF, MachineRegisterInfo &MRI,
                          const TargetInstrInfo &TII) {
  auto DefIs = [&](const MachineOperand &MO, unsigned Opc) -> MachineInstr * {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      return nullptr;
    MachineInstr *D = MRI.getVRegDef(MO.getReg());
    return D && D->getOpcode() == Opc && !TII.isPredicated(*D) ? D : nullptr;
  };
  SmallVector<MachineInstr *, 32> Bases;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.getOpcode() == SASS::LEArrr && !TII.isPredicated(MI) &&
          MI.getOperand(3).getImm() == 24 && DefIs(MI.getOperand(1), SASS::S2R_CGA))
        if (MachineInstr *K = DefIs(MI.getOperand(2), SASS::MOVri);
            K && K->getOperand(1).isImm() && K->getOperand(1).getImm() == 0x400)
          Bases.push_back(&MI);
  if (Bases.size() < 2)
    return false;
  MachineBasicBlock &Entry = MF.front();
  auto At = Entry.getFirstNonPHI();
  DebugLoc DL;
  auto Guard = [](const MachineInstrBuilder &MIB) { MIB.addReg(SASS::PT).addImm(0); };
  Register Cga = MRI.createVirtualRegister(&SASS::GPR32RegClass);
  Register K = MRI.createVirtualRegister(&SASS::GPR32RegClass);
  Register Base = MRI.createVirtualRegister(&SASS::GPR32RegClass);
  Guard(BuildMI(Entry, At, DL, TII.get(SASS::S2R_CGA), Cga));
  Guard(BuildMI(Entry, At, DL, TII.get(SASS::MOVri), K).addImm(0x400));
  Guard(BuildMI(Entry, At, DL, TII.get(SASS::LEArrr), Base)
            .addReg(Cga).addReg(K).addImm(24));
  for (MachineInstr *MI : Bases) {
    Register Old = MI->getOperand(0).getReg();
    SmallVector<MachineInstr *, 2> Ops;
    for (unsigned I : {1u, 2u})
      Ops.push_back(MRI.getVRegDef(MI->getOperand(I).getReg()));
    MRI.replaceRegWith(Old, Base);
    MI->eraseFromParent();
    for (MachineInstr *D : Ops)
      if (MRI.use_nodbg_empty(D->getOperand(0).getReg()))
        D->eraseFromParent();
  }
  MRI.clearKillFlags(Base);
  return true;
}

bool SASSUniformRegs::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (!EnableRegs || ST.hasBlackwellISA() || !MF.getRegInfo().isSSA())
    return false;
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  PDT = &getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  OnlyTidX = SASS::readsOnlyTidX(MF);
  bool Unified = unifySmemBase(MF, *MRI, *TII);
  GatherAt.clear();
  AlignedAt.clear();
  Order.clear();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      Order[&MI] = Order.size();
      if (isAligned(MI.getOpcode()))
        AlignedAt.try_emplace(&MBB, &MI);
      if (MI.getOpcode() == SASS::WARPSYNC_ALL)
        GatherAt.try_emplace(&MBB, &MI);
    }
  computeDivergence(MF);
  computePlacement(MF);
  DEBUG_WITH_TYPE("sass-uniform-regs", {
    dbgs() << MF.getName() << ": " << DivBranches.size()
           << " divergent branches, " << NoUDef.size() << "/" << MF.size()
           << " blocks fenced, " << ExitSide.size() << " exiting\n";
    for (MachineBasicBlock *B : DivBranches) {
      dbgs() << "  divergent " << printMBBReference(*B) << ": "
             << *B->getFirstTerminator();
      for (const MachineOperand &MO : B->getFirstTerminator()->uses())
        if (MO.isReg() && MO.getReg().isVirtual())
          if (MachineInstr *D = MRI->getVRegDef(MO.getReg()))
            dbgs() << "    cond " << *D;
    }
  });
  return promote(MF) || Unified;
}
