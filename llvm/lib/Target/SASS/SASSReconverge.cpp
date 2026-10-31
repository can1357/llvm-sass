//===-- SASSReconverge.cpp - Warp reconvergence (BSSY/BSYNC) --------------===//
//
// Volta and later schedule the threads of a warp independently: after a
// divergent branch the two groups run one after the other and nothing merges
// them again. BAR.SYNC counts a partially arrived warp as arrived (threads of
// the other group then race past the CTA barrier), warp collectives see a
// partial warp, and every later instruction issues twice. ptxas therefore
// brackets each divergent region with a convergence barrier:
//
//     BSSY Bn, J          ; arm barrier n with the active threads
//     @P BRA ...          ; the divergent branch
//     ...                 ; the region
//     BSYNC Bn            ; wait for all of them (exited threads excepted)
//   J:                    ; reconverged
//
// This pass does the same on the final block layout. A region is a divergent
// conditional branch (BRAc; uniform branches were selected as BRAcu from
// UniformityInfo) and its immediate post-dominator J: BSSY goes before the
// branch, or, when the branch sits in a loop that the region contains, into
// a new preheader of that loop; every edge from the region into J goes
// through a new block holding BSYNC, laid out right before its target.
// Regions that only part ways to exit (J is the virtual exit, or a lone
// EXIT) need no barrier, except that when exactly one side still reaches a
// warp-synchronous instruction (BAR.SYNC, a shuffle, ...) that side's entry
// is the join, so the exiting threads finish before the barrier.
//
// The IR side (SASSTTIImpl::getValueUniformity) names the sources of
// divergence for UniformityInfo.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSSubtarget.h"
#include "SASSTargetTransformInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "sass-reconverge"

static cl::opt<bool> ReconvergeAll(
    "sass-reconverge",
    cl::desc("Bracket divergent regions with BSSY/BSYNC on Blackwell too "
             "(always done on sm_90)"),
    cl::init(false));

bool SASS::reconverges(const SASSSubtarget &ST) {
  return !ST.hasBlackwellISA() || ReconvergeAll;
}

//===----------------------------------------------------------------------===//
// IR: sources of divergence
//===----------------------------------------------------------------------===//

static bool isDivergentSource(const Value *V) {
  if (const auto *A = dyn_cast<Argument>(V))
    return A->getParent()->getCallingConv() != CallingConv::PTX_Kernel;
  const auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  // Generic pointers may point to (per-thread) local memory.
  if (const auto *LI = dyn_cast<LoadInst>(I)) {
    unsigned AS = LI->getPointerAddressSpace();
    return AS == 0 || AS == 5;
  }
  // Threads of a warp perform atomics one after the other.
  if (isa<AtomicRMWInst, AtomicCmpXchgInst>(I))
    return true;
  const auto *CB = dyn_cast<CallBase>(I);
  if (!CB || CB->getType()->isVoidTy())
    return false;
  const Function *F = CB->getCalledFunction();
  if (!F || !F->isIntrinsic())
    return true; // Calls and inline asm.
  switch (F->getIntrinsicID()) {
  case Intrinsic::nvvm_read_ptx_sreg_tid_x:
  case Intrinsic::nvvm_read_ptx_sreg_tid_y:
  case Intrinsic::nvvm_read_ptx_sreg_tid_z:
  case Intrinsic::nvvm_read_ptx_sreg_laneid:
  case Intrinsic::nvvm_read_ptx_sreg_lanemask_eq:
  case Intrinsic::nvvm_read_ptx_sreg_lanemask_le:
  case Intrinsic::nvvm_read_ptx_sreg_lanemask_lt:
  case Intrinsic::nvvm_read_ptx_sreg_lanemask_ge:
  case Intrinsic::nvvm_read_ptx_sreg_lanemask_gt:
  case Intrinsic::nvvm_read_ptx_sreg_clock:
  case Intrinsic::nvvm_read_ptx_sreg_clock64:
  case Intrinsic::nvvm_read_ptx_sreg_globaltimer:
    return true;
  default:
    break;
  }
  // Shuffles, votes, matrix loads, mbarrier polls, atomics: anything that
  // touches memory or state may differ per thread.
  return CB->mayReadOrWriteMemory() || CB->mayHaveSideEffects() ||
         CB->isConvergent();
}

ValueUniformity SASSTTIImpl::getValueUniformity(const Value *V) const {
  if (SASS::reconverges(*ST) && isDivergentSource(V))
    return ValueUniformity::NeverUniform;
  return ValueUniformity::Default;
}

//===----------------------------------------------------------------------===//
// Machine: convergence barriers
//===----------------------------------------------------------------------===//

namespace {

struct Region {
  MachineBasicBlock *Branch; // Ends in the divergent BRAc.
  MachineBasicBlock *Join;
  /// Blocks between Branch (included) and Join, then the barrier blocks of
  /// regions nested in it.
  SmallPtrSet<MachineBasicBlock *, 16> Blocks;
  /// Loop header inside the region that is its entry (BSSY in a new
  /// preheader); null when BSSY goes before Branch's terminator.
  MachineBasicBlock *Header = nullptr;
  MachineBasicBlock *Sync = nullptr;     // BSYNC block before Join.
  MachineBasicBlock *Preheader = nullptr;
  MachineInstr *Arm = nullptr;           // The BSSY.
  unsigned Barrier = 0;
};

class SASSReconverge : public MachineFunctionPass {
public:
  static char ID;
  SASSReconverge() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS reconvergence"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const SASSInstrInfo *TII = nullptr;
  bool findRegion(MachineBasicBlock &B, MachinePostDominatorTree &PDT,
                  const SmallPtrSetImpl<MachineBasicBlock *> &ReachesSync,
                  Region &R);
  void fixFallthroughs(MachineFunction &MF,
                       const SmallPtrSetImpl<MachineBasicBlock *> &SyncBlocks);
};

} // namespace

char SASSReconverge::ID = 0;

INITIALIZE_PASS_BEGIN(SASSReconverge, DEBUG_TYPE, "SASS reconvergence", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SASSReconverge, DEBUG_TYPE, "SASS reconvergence", false,
                    false)

FunctionPass *llvm::createSASSReconvergePass() { return new SASSReconverge(); }

static const unsigned BSSYOpc[16] = {
    SASS::BSSY0,  SASS::BSSY1,  SASS::BSSY2,  SASS::BSSY3,
    SASS::BSSY4,  SASS::BSSY5,  SASS::BSSY6,  SASS::BSSY7,
    SASS::BSSY8,  SASS::BSSY9,  SASS::BSSY10, SASS::BSSY11,
    SASS::BSSY12, SASS::BSSY13, SASS::BSSY14, SASS::BSSY15};
static const unsigned BSYNCOpc[16] = {
    SASS::BSYNC0,  SASS::BSYNC1,  SASS::BSYNC2,  SASS::BSYNC3,
    SASS::BSYNC4,  SASS::BSYNC5,  SASS::BSYNC6,  SASS::BSYNC7,
    SASS::BSYNC8,  SASS::BSYNC9,  SASS::BSYNC10, SASS::BSYNC11,
    SASS::BSYNC12, SASS::BSYNC13, SASS::BSYNC14, SASS::BSYNC15};

/// Only an EXIT: reconverging there is pointless.
static bool isExitOnly(const MachineBasicBlock &MBB) {
  auto I = MBB.getFirstNonDebugInstr();
  return I != MBB.end() && I->getOpcode() == SASS::EXIT;
}

bool SASSReconverge::findRegion(
    MachineBasicBlock &B, MachinePostDominatorTree &PDT,
    const SmallPtrSetImpl<MachineBasicBlock *> &ReachesSync, Region &R) {
  auto T = B.getFirstTerminator();
  if (T == B.end() || T->getOpcode() != SASS::BRAc || B.succ_size() != 2)
    return false;
  MachineDomTreeNode *N = PDT.getNode(&B);
  if (!N)
    return false; // Never exits (infinite loop).
  MachineBasicBlock *J = N->getIDom() ? N->getIDom()->getBlock() : nullptr;
  if (!J || isExitOnly(*J)) {
    // The sides part ways to exit. Only when one of them still reaches a
    // warp-synchronous instruction must the other finish first.
    J = nullptr;
    unsigned Reaching = 0;
    for (MachineBasicBlock *S : B.successors())
      if (ReachesSync.count(S)) {
        J = S;
        ++Reaching;
      }
    if (Reaching != 1 || J == &B)
      return false;
  }
  R.Branch = &B;
  R.Join = J;
  SmallVector<MachineBasicBlock *, 16> Work(B.successors());
  SmallPtrSet<MachineBasicBlock *, 16> Body;
  while (!Work.empty()) {
    MachineBasicBlock *X = Work.pop_back_val();
    if (X == J || !Body.insert(X).second)
      continue;
    append_range(Work, X->successors());
  }
  bool InCycle = Body.count(&B);
  Body.insert(&B);
  // Entry: Branch, or the loop header containing it. Anything else entering
  // the region from outside would reach BSYNC unarmed.
  MachineBasicBlock *Entry = nullptr;
  for (MachineBasicBlock *X : Body) {
    for (MachineBasicBlock *P : X->predecessors())
      if (!Body.count(P)) {
        if (Entry && Entry != X)
          return false;
        Entry = X;
      }
    // Every way out leads to the join.
    for (MachineBasicBlock *S : X->successors())
      if (!Body.count(S) && S != J)
        return false;
  }
  if (!InCycle) {
    if (Entry && Entry != &B)
      return false;
  } else {
    if (!Entry)
      return false;
    R.Header = Entry;
  }
  R.Blocks = std::move(Body);
  return true;
}

void SASSReconverge::fixFallthroughs(
    MachineFunction &MF, const SmallPtrSetImpl<MachineBasicBlock *> &SyncBlocks) {
  for (auto It = MF.begin(); It != MF.end(); ++It) {
    MachineBasicBlock &X = *It;
    SmallPtrSet<MachineBasicBlock *, 2> Explicit;
    bool Barrier = false;
    for (MachineInstr &MI : X.terminators()) {
      for (MachineOperand &MO : MI.operands())
        if (MO.isMBB())
          Explicit.insert(MO.getMBB());
      Barrier |= MI.isBarrier();
    }
    MachineBasicBlock *Implicit = nullptr;
    for (MachineBasicBlock *S : X.successors())
      if (!Explicit.count(S))
        Implicit = S;
    auto Next = std::next(It);
    if (!Implicit || Barrier ||
        (Next != MF.end() && &*Next == Implicit))
      continue;
    if (SyncBlocks.count(&X)) {
      // BSYNC resumes at the instruction after it: a separate block.
      MachineBasicBlock *Tramp = MF.CreateMachineBasicBlock();
      MF.insert(Next, Tramp);
      for (const auto &LI : Implicit->liveins())
        Tramp->addLiveIn(LI);
      X.replaceSuccessor(Implicit, Tramp);
      Tramp->addSuccessor(Implicit);
      BuildMI(Tramp, DebugLoc(), TII->get(SASS::BRA))
          .addMBB(Implicit)
          .addReg(SASS::PT)
          .addImm(0);
      continue;
    }
    BuildMI(&X, DebugLoc(), TII->get(SASS::BRA))
        .addMBB(Implicit)
        .addReg(SASS::PT)
        .addImm(0);
  }
}

bool SASSReconverge::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (!SASS::reconverges(ST))
    return false;
  TII = ST.getInstrInfo();
  MachinePostDominatorTree &PDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();

  // Blocks from which a warp-synchronous instruction is reachable.
  SmallPtrSet<MachineBasicBlock *, 32> ReachesSync;
  SmallVector<MachineBasicBlock *, 32> Work;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isConvergent()) {
        Work.push_back(&MBB);
        break;
      }
  while (!Work.empty()) {
    MachineBasicBlock *X = Work.pop_back_val();
    if (ReachesSync.insert(X).second)
      append_range(Work, X->predecessors());
  }

  std::vector<Region> Regions;
  for (MachineBasicBlock &MBB : MF) {
    Region R;
    if (findRegion(MBB, PDT, ReachesSync, R))
      Regions.push_back(std::move(R));
  }
  if (Regions.empty())
    return false;
  // Inner regions first: an enclosing region then redirects the inner
  // BSYNC block's edge into the shared join.
  llvm::stable_sort(Regions, [](const Region &A, const Region &B) {
    return A.Blocks.size() < B.Blocks.size();
  });

  SmallPtrSet<MachineBasicBlock *, 16> SyncBlocks;
  DebugLoc DL;
  for (unsigned I = 0, E = Regions.size(); I != E; ++I) {
    Region &R = Regions[I];
    MachineBasicBlock *J = R.Join;
    // BSYNC block on every edge from the region into the join.
    MachineBasicBlock *Sync = MF.CreateMachineBasicBlock();
    MF.insert(J->getIterator(), Sync);
    for (const auto &LI : J->liveins())
      Sync->addLiveIn(LI);
    SmallVector<MachineBasicBlock *, 4> Into;
    for (MachineBasicBlock *P : J->predecessors())
      if (R.Blocks.count(P))
        Into.push_back(P);
    for (MachineBasicBlock *P : Into)
      P->ReplaceUsesOfBlockWith(J, Sync);
    Sync->addSuccessor(J);
    BuildMI(Sync, DL, TII->get(SASS::BSYNC0)).addReg(SASS::PT).addImm(0);
    R.Sync = Sync;
    SyncBlocks.insert(Sync);

    // BSSY (its target is set once the layout is final).
    MachineBasicBlock *ArmBB = R.Branch;
    MachineBasicBlock::iterator At = R.Branch->getFirstTerminator();
    if (MachineBasicBlock *H = R.Header) {
      MachineBasicBlock *Pre = MF.CreateMachineBasicBlock();
      MF.insert(H->getIterator(), Pre);
      for (const auto &LI : H->liveins())
        Pre->addLiveIn(LI);
      SmallVector<MachineBasicBlock *, 4> Outside;
      for (MachineBasicBlock *P : H->predecessors())
        if (!R.Blocks.count(P))
          Outside.push_back(P);
      for (MachineBasicBlock *P : Outside)
        P->ReplaceUsesOfBlockWith(H, Pre);
      Pre->addSuccessor(H);
      R.Preheader = ArmBB = Pre;
      At = Pre->end();
    }
    R.Arm = BuildMI(*ArmBB, At, DL, TII->get(SASS::BSSY0))
                .addMBB(J)
                .addReg(SASS::PT)
                .addImm(0);

    // Enclosing regions now contain this one's new blocks.
    for (unsigned K = I + 1; K != E; ++K)
      if (Regions[K].Blocks.count(R.Branch)) {
        Regions[K].Blocks.insert(Sync);
        if (R.Preheader)
          Regions[K].Blocks.insert(R.Preheader);
      }
  }

  fixFallthroughs(MF, SyncBlocks);

  // Barrier registers: regions sharing a block are live together.
  for (unsigned I = 0, E = Regions.size(); I != E; ++I) {
    Region &R = Regions[I];
    R.Blocks.insert(R.Sync);
    if (R.Preheader)
      R.Blocks.insert(R.Preheader);
  }
  for (unsigned I = Regions.size(); I-- != 0;) {
    Region &R = Regions[I];
    unsigned Used = 0;
    for (unsigned K = I + 1; K != Regions.size(); ++K) {
      const Region &O = Regions[K];
      if (llvm::any_of(R.Blocks, [&](MachineBasicBlock *X) {
            return O.Blocks.count(X);
          }))
        Used |= 1u << O.Barrier;
    }
    unsigned Free = llvm::countr_one(Used);
    if (Free >= 16)
      report_fatal_error("SASS: more than 16 nested divergent regions in " +
                         MF.getName());
    R.Barrier = Free;
    R.Arm->setDesc(TII->get(BSSYOpc[Free]));
    R.Arm->getOperand(0).setMBB(&*std::next(R.Sync->getIterator()));
    R.Sync->begin()->setDesc(TII->get(BSYNCOpc[Free]));
  }
  return true;
}
