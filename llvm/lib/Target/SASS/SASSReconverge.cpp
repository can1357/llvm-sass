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
// A thread that sleeps (NANOSLEEP) leaves the convergence barriers it is in,
// so a BSYNC would let the others run on without it. As ptxas does, regions
// that can sleep get no barrier; WARPSYNC.ALL gathers the warp at the next
// instruction that needs all of it (gatherSplitWarps), as after branches
// that could not be bracketed.
//
// Branches that cannot split a warp although selected divergent (warp-
// specialized roles: threadIdx.x against a multiple of 32) leave the whole
// warp below them, and paths into a trap (a spin wait's timeout) do not keep
// the sides of a branch from meeting. With assertions (-sass-verify-uniform)
// the final code is checked for uniform register writes where another group
// of a split warp may use the same register (verifyUniformWrites).
//
// The IR side (SASSTTIImpl::getValueUniformity) names the sources of
// divergence for UniformityInfo.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSSubtarget.h"
#include "SASSTargetTransformInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
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

static cl::opt<bool> UniformMBarPoll(
    "sass-uniform-mbar-poll",
    cl::desc("An mbarrier try_wait/test_wait on a uniform barrier and phase "
             "is uniform (one warp-wide poll, as ptxas assumes)"),
    cl::init(true));

namespace {
enum class UniformCheck { Off, Warn, Fatal };
} // namespace

static cl::opt<UniformCheck> VerifyUniform(
    "sass-verify-uniform",
    cl::desc("After reconvergence, report uniform register / predicate "
             "writes where the warp may run split"),
#ifndef NDEBUG
    cl::init(UniformCheck::Warn),
#else
    cl::init(UniformCheck::Off),
#endif
    cl::values(clEnumValN(UniformCheck::Off, "off", "no check"),
               clEnumValN(UniformCheck::Warn, "warn", "print to stderr"),
               clEnumValN(UniformCheck::Fatal, "fatal", "abort")));

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
  // An mbarrier poll reads the barrier's phase once for the warp: with a
  // uniform barrier and phase operand every active thread sees the same
  // result, so a spin loop on it needs no BSSY/BSYNC (ptxas emits a bare
  // TRYWAIT and a branch to an out-of-line retry).
  StringRef Name = F->getName();
  if (UniformMBarPoll && (Name.starts_with("llvm.nvvm.mbarrier.try.wait") ||
                          Name.starts_with("llvm.nvvm.mbarrier.test.wait")))
    return false;
  // Shuffles, votes, matrix loads, atomics: anything that touches memory or
  // state may differ per thread.
  return CB->mayReadOrWriteMemory() || CB->mayHaveSideEffects() ||
         CB->isConvergent();
}

/// A full-warp shfl.sync.idx of a constant lane with no sub-warp segments
/// hands every lane the same value: the lane-0 broadcast kernels use to mark
/// values warp-uniform (SASSTargetLowering::isSDNodeAlwaysUniform matches it
/// in the DAG).
static bool isLaneBroadcast(const Value *V) {
  const auto *II = dyn_cast<IntrinsicInst>(V);
  if (!II || (II->getIntrinsicID() != Intrinsic::nvvm_shfl_sync_idx_i32 &&
              II->getIntrinsicID() != Intrinsic::nvvm_shfl_sync_idx_f32))
    return false;
  const auto *M = dyn_cast<ConstantInt>(II->getArgOperand(0));
  const auto *L = dyn_cast<ConstantInt>(II->getArgOperand(2));
  const auto *C = dyn_cast<ConstantInt>(II->getArgOperand(3));
  return M && M->isAllOnesValue() && L && L->getZExtValue() < 32 && C &&
         C->getZExtValue() == 0x1f;
}

ValueUniformity SASSTTIImpl::getValueUniformity(const Value *V) const {
  if (!SASS::reconverges(*ST))
    return ValueUniformity::Default;
  if (isLaneBroadcast(V))
    return ValueUniformity::AlwaysUniform;
  return isDivergentSource(V) ? ValueUniformity::NeverUniform : ValueUniformity::Default;
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
  bool Gather = false; // WARPSYNC.ALL before the BSSY (syncAfterSleep).
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
  bool reconverge(MachineFunction &MF);
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
  if (!N) {
    LLVM_DEBUG(dbgs() << "no region at " << printMBBReference(B)
                      << ": never exits\n");
    return false; // Never exits (infinite loop).
  }
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
    if (Reaching != 1 || J == &B) {
      LLVM_DEBUG(dbgs() << "no region at " << printMBBReference(B)
                        << ": parts ways to exit, " << Reaching
                        << " sides reach a warp-synchronous instruction\n");
      return false;
    }
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
        if (Entry && Entry != X) {
          LLVM_DEBUG(dbgs() << "no region at " << printMBBReference(B)
                            << ": entered at " << printMBBReference(*Entry)
                            << " and " << printMBBReference(*X) << '\n');
          return false;
        }
        Entry = X;
      }
    // Every way out leads to the join.
    for (MachineBasicBlock *S : X->successors())
      if (!Body.count(S) && S != J) {
        LLVM_DEBUG(dbgs() << "no region at " << printMBBReference(B)
                          << ": leaves to " << printMBBReference(*S) << '\n');
        return false;
      }
  }
  if (!InCycle) {
    if (Entry && Entry != &B) {
      LLVM_DEBUG(dbgs() << "no region at " << printMBBReference(B)
                        << ": entered at " << printMBBReference(*Entry)
                        << '\n');
      return false;
    }
  } else {
    if (!Entry) {
      LLVM_DEBUG(dbgs() << "no region at " << printMBBReference(B)
                        << ": cycle without entry\n");
      return false;
    }
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

/// A warp collective needs every thread of its member mask at once. Below a
/// divergent branch the two groups only meet again at the immediate
/// post-dominator J; a block reachable from both sides before J (an
/// unstructured merge, e.g. a join block tail-duplicated into both arms)
/// runs once per group, so a full-warp collective there would see half a
/// warp. As ptxas does for collectives it cannot prove converged, a WARPSYNC
/// on the full mask right before it gathers the warp first (every thread of
/// a well-defined program reaches that collective). Collectives on another
/// mask already follow their own WARPSYNC (SASSISelWarp).
static bool isCollective(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case SASS::WARPSYNC_R:
  case SASS::WARPSYNC_ALL:
  case SASS::VOTE_ACTIVE:
    return false;
  default:
    return MI.isConvergent() && !MI.isBranch();
  }
}

static bool syncMerges(MachineFunction &MF, MachinePostDominatorTree &PDT,
                       const SASSInstrInfo *TII) {
  SmallPtrSet<MachineInstr *, 8> Need;
  for (MachineBasicBlock &B : MF) {
    auto T = B.getFirstTerminator();
    if (T == B.end() || T->getOpcode() != SASS::BRAc || B.succ_size() != 2)
      continue;
    MachineDomTreeNode *N = PDT.getNode(&B);
    MachineBasicBlock *J =
        N && N->getIDom() ? N->getIDom()->getBlock() : nullptr;
    SmallPtrSet<MachineBasicBlock *, 16> Seen[2];
    for (auto [K, S] : enumerate(B.successors())) {
      SmallVector<MachineBasicBlock *, 16> Work{S};
      while (!Work.empty()) {
        MachineBasicBlock *X = Work.pop_back_val();
        if (X == J || X == &B || !Seen[K].insert(X).second)
          continue;
        append_range(Work, X->successors());
      }
    }
    for (MachineBasicBlock *X : Seen[0]) {
      if (!Seen[1].count(X))
        continue;
      // Full-mask collectives: not preceded (past non-collectives) by a
      // WARPSYNC in their block.
      bool Synced = false;
      for (MachineInstr &MI : *X) {
        if (MI.getOpcode() == SASS::WARPSYNC_R ||
            MI.getOpcode() == SASS::WARPSYNC_ALL) {
          Synced = true;
          continue;
        }
        if (!isCollective(MI))
          continue;
        if (!Synced)
          Need.insert(&MI);
        Synced = false;
      }
    }
  }
  for (MachineInstr *MI : Need) {
    LLVM_DEBUG(dbgs() << "WARPSYNC.ALL in merged " << printMBBReference(
                             *MI->getParent()) << " before " << *MI);
    BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
            TII->get(SASS::WARPSYNC_ALL))
        .addReg(SASS::PT)
        .addImm(0);
  }
  return !Need.empty();
}

/// Writes a uniform register or predicate (R2UR, ULDC, the uniform
/// datapath). There is one uniform register file per warp: two groups of a
/// split warp running the rest of a region apart (one after the other, or
/// interleaved) overwrite each other's values, which register allocation,
/// seeing one thread's paths, does not know about. They run gathered.
static bool isUniformPhys(Register R) {
  return R.isPhysical() && R != SASS::URZ && R != SASS::UPT &&
         (SASS::UGPR32RegClass.contains(R) ||
          SASS::UGPR64RegClass.contains(R) ||
          SASS::UGPR128RegClass.contains(R) ||
          SASS::UPREDRegClass.contains(R));
}
static bool definesUniform(const MachineInstr &MI) {
  return any_of(MI.all_defs(), [](const MachineOperand &MO) {
    return isUniformPhys(MO.getReg());
  });
}
static bool touchesUniform(const MachineInstr &MI) {
  return any_of(MI.operands(), [](const MachineOperand &MO) {
    return MO.isReg() && isUniformPhys(MO.getReg());
  });
}

/// Warp-wide (.sync.aligned) matrix instructions not marked convergent.
static bool isWarpMatrix(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case SASS::HMMA16816F32:
  case SASS::HMMA16816F32BF16:
  case SASS::HMMA16816F32Z:
  case SASS::HMMA16816F32BF16Z:
  case SASS::MAT1_LD:
  case SASS::MAT1_LDT:
  case SASS::MAT1_ST:
  case SASS::MAT1_STT:
  case SASS::MAT2_LD:
  case SASS::MAT2_LDT:
  case SASS::MAT2_ST:
  case SASS::MAT2_STT:
  case SASS::MAT4_LD:
  case SASS::MAT4_LDT:
  case SASS::MAT4_ST:
  case SASS::MAT4_STT:
    return true;
  default:
    return false;
  }
}

static bool sleeps(const MachineBasicBlock &MBB) {
  return any_of(MBB, [](const MachineInstr &MI) {
    return MI.getOpcode() == SASS::NANOSLEEPi ||
           MI.getOpcode() == SASS::NANOSLEEPr;
  });
}

static const MachineInstr *reachingDef(const MachineInstr &At, Register R,
                                       const TargetRegisterInfo *TRI,
                                       unsigned Depth = 0);

/// Blocks from which every path ends in a trap (BPT.TRAP, e.g. the timeout
/// of a spin wait): threads there never come back, and the kernel fails.
static SmallPtrSet<MachineBasicBlock *, 8> trapBlocks(MachineFunction &MF) {
  SmallPtrSet<MachineBasicBlock *, 8> Doomed;
  for (MachineBasicBlock &B : MF)
    if (any_of(B, [](const MachineInstr &MI) {
          return MI.getOpcode() == SASS::BPT_TRAP;
        }))
      Doomed.insert(&B);
  for (bool Again = !Doomed.empty(); Again;) {
    Again = false;
    for (MachineBasicBlock &B : MF)
      if (!B.succ_empty() && !Doomed.count(&B) &&
          all_of(B.successors(),
                 [&](MachineBasicBlock *S) { return Doomed.count(S); }))
        Again |= Doomed.insert(&B).second;
  }
  return Doomed;
}

/// Where the sides of \p B meet when paths into a trap are ignored (their
/// post-dominator is the virtual exit only because of those paths): the
/// nearest block that every other way from B to an exit passes.
static MachineBasicBlock *
joinPastTraps(MachineBasicBlock &B,
              const SmallPtrSetImpl<MachineBasicBlock *> &Doomed) {
  if (Doomed.empty())
    return nullptr;
  // Whether an exit is reachable from B's successors without passing X.
  auto Escapes = [&](MachineBasicBlock *X) {
    SmallPtrSet<MachineBasicBlock *, 32> Seen;
    SmallVector<MachineBasicBlock *, 16> Work(B.successors());
    while (!Work.empty()) {
      MachineBasicBlock *Y = Work.pop_back_val();
      if (Y == X || Doomed.count(Y) || !Seen.insert(Y).second)
        continue;
      if (Y->succ_empty())
        return true;
      append_range(Work, Y->successors());
    }
    return false;
  };
  // Candidates in breadth-first order from the branch.
  SmallVector<MachineBasicBlock *, 64> Order;
  SmallPtrSet<MachineBasicBlock *, 32> Seen;
  Seen.insert(&B);
  for (MachineBasicBlock *S : B.successors())
    if (!Doomed.count(S) && Seen.insert(S).second)
      Order.push_back(S);
  for (unsigned I = 0; I < Order.size() && I < 64; ++I) {
    MachineBasicBlock *X = Order[I];
    if (!Escapes(X))
      return X;
    for (MachineBasicBlock *S : X->successors())
      if (!Doomed.count(S) && Seen.insert(S).second)
        Order.push_back(S);
  }
  return nullptr;
}

/// Divergent branches without a convergence barrier leave the warp split
/// after their join:
///  * A sleeping thread (NANOSLEEP) drops out of its convergence barriers,
///    so BSYNC lets the others go on without it (on sm_90 a lane-0 spin with
///    __nanosleep followed by __syncwarp leaves the warp split for good). As
///    ptxas does, regions that can sleep get no BSSY/BSYNC (they are erased
///    from \p Regions).
///  * findRegion cannot bracket a branch whose region has side entries
///    (e.g. left by tail duplication).
/// The BSYNC of an enclosing region gathers the warp again; otherwise a
/// WARPSYNC.ALL goes right before the next instruction that needs the whole
/// warp: __syncwarp() (replacing its NOP), a CTA barrier, a full-mask
/// collective or matrix instruction, a uniform register write in a block the
/// whole warp runs (definesUniform), or the BSSY of a later region
/// (Region::Gather). A region to
/// be armed while the warp may be split below another divergent branch,
/// where not the whole warp runs, cannot gather it and goes without its
/// barrier as well.
static bool gatherSplitWarps(MachineFunction &MF,
                             MachinePostDominatorTree &PDT,
                             std::vector<Region> &Regions,
                             const SASSInstrInfo *TII) {
  auto InBody = [](const Region &R, MachineBasicBlock *X) {
    return R.Blocks.count(X) && (X != R.Branch || R.Header);
  };
  bool Changed = false;
  llvm::erase_if(Regions, [&](const Region &R) {
    bool Sleeps = any_of(R.Blocks, [&](MachineBasicBlock *X) {
      return InBody(R, X) && sleeps(*X);
    });
    if (Sleeps)
      LLVM_DEBUG(dbgs() << "no barrier for " << printMBBReference(*R.Branch)
                        << ": may sleep\n");
    Changed |= Sleeps;
    return Sleeps;
  });
  SmallPtrSet<MachineBasicBlock *, 32> Bracketed;
  for (const Region &R : Regions)
    Bracketed.insert(R.Branch);

  // Edges into joins after which the warp may be split, and the blocks below
  // a divergent branch (not the whole warp runs there).
  DenseSet<std::pair<MachineBasicBlock *, MachineBasicBlock *>> SplitInto;
  auto SplitAt = [&](MachineBasicBlock *J,
                     const SmallPtrSetImpl<MachineBasicBlock *> &From) {
    for (MachineBasicBlock *P : J->predecessors())
      if (From.count(P))
        SplitInto.insert({P, J});
  };
  SmallPtrSet<MachineBasicBlock *, 32> Partial;
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> JoinAfter;
  SmallPtrSet<MachineBasicBlock *, 8> Doomed = trapBlocks(MF);
  // Warp-specialized roles split no warp: the whole warp runs below them.
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  bool OnlyTidX = SASS::readsOnlyTidX(MF);
  auto DefOf = [&](const MachineInstr &User, Register R) {
    return reachingDef(User, R, TRI);
  };
  for (MachineBasicBlock &B : MF) {
    auto T = B.getFirstTerminator();
    if (T == B.end() || T->getOpcode() != SASS::BRAc || B.succ_size() != 2 ||
        SASS::branchesWarpUniformly(*T, OnlyTidX, DefOf))
      continue;
    // Sides that never meet again (J is the virtual exit) are separate
    // warps in practice (a producer warp split off and exiting). Sides that
    // only fail to meet on the way into a trap (a spin wait's timeout) meet
    // where the others do.
    MachineDomTreeNode *N = PDT.getNode(&B);
    MachineBasicBlock *J =
        N && N->getIDom() ? N->getIDom()->getBlock() : nullptr;
    if (!J && N && (J = joinPastTraps(B, Doomed)) && isExitOnly(*J))
      J = nullptr;
    if (!J)
      continue;
    SmallPtrSet<MachineBasicBlock *, 16> Body;
    SmallVector<MachineBasicBlock *, 16> Work(B.successors());
    while (!Work.empty()) {
      MachineBasicBlock *X = Work.pop_back_val();
      if (X != J && Body.insert(X).second)
        append_range(Work, X->successors());
    }
    Partial.insert(Body.begin(), Body.end());
    JoinAfter[&B] = J;
    if (!Bracketed.count(&B)) {
      Body.insert(&B);
      SplitAt(J, Body);
    }
  }
  if (SplitInto.empty())
    return Changed;

  // Walks B from state Split (the warp may be split); returns the state at
  // its end. With Apply, gathers the warp where needed.
  auto Walk = [&](MachineBasicBlock &B, bool Split, bool Apply) {
    bool Masked = false; // A WARPSYNC on a member mask precedes.
    for (auto I = B.begin(), E = B.end(); I != E;) {
      MachineInstr &MI = *I++;
      if (MI.getOpcode() == SASS::WARPSYNC_ALL) {
        Split = Masked = false;
        continue;
      }
      if (MI.getOpcode() == SASS::WARPSYNC_R) {
        Masked = true;
        continue;
      }
      if (!isCollective(MI) && !isWarpMatrix(MI) &&
          (!definesUniform(MI) || Partial.count(&B))) {
        // Below a divergent branch only part of the warp comes: a uniform
        // write there cannot gather it (elected operands under lane == 0).
        LLVM_DEBUG(if (Apply && Split && !Masked && definesUniform(MI))
                     dbgs() << "uniform write while the warp may be split in "
                            << printMBBReference(B) << ": " << MI);
        continue;
      }
      if (Split && !Masked) {
        if (Apply) {
          LLVM_DEBUG(dbgs() << "WARPSYNC.ALL after sleep in "
                            << printMBBReference(B) << " before " << MI);
          BuildMI(B, MI, MI.getDebugLoc(), TII->get(SASS::WARPSYNC_ALL))
              .addReg(SASS::PT)
              .addImm(0);
          if (MI.getOpcode() == SASS::WARPSYNC_NOP)
            MI.eraseFromParent();
          Changed = true;
        }
        Split = false;
      }
      Masked = false;
    }
    return Split;
  };

  // Forward may-analysis of "the warp may be split" on block exits. A kept
  // region gathers the warp before its BSSY: before Branch's terminator, or
  // on the edges into its loop header from outside; its BSYNC gathers it on
  // the edges into its join.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  DenseMap<MachineBasicBlock *, bool> Out;
  DenseMap<MachineBasicBlock *, SmallVector<Region *, 1>> ArmIn, JoinOf;
  DenseMap<MachineBasicBlock *, Region *> ArmAfter;
  // Only part of the warp takes edge P -> X: P is below a divergent branch,
  // or is one and X is not its join.
  auto PartialEdge = [&](MachineBasicBlock *P, MachineBasicBlock *X) {
    auto It = JoinAfter.find(P);
    return Partial.count(P) || (It != JoinAfter.end() && It->second != X);
  };
  SmallPtrSet<Region *, 4> Blocked; // Would have to gather a partial warp.
  auto InState = [&](MachineBasicBlock *B) {
    bool In = false;
    auto HI = ArmIn.find(B), JI = JoinOf.find(B);
    for (MachineBasicBlock *P : B->predecessors()) {
      if (!Out.lookup(P) && !SplitInto.count({P, B}))
        continue;
      if (JI != JoinOf.end() && any_of(JI->second, [&](Region *K) {
            return K->Blocks.count(P);
          }))
        continue;
      bool Outside = false;
      if (HI != ArmIn.end())
        for (Region *H : HI->second)
          if (!H->Blocks.count(P)) {
            Outside = true;
            if (PartialEdge(P, B))
              Blocked.insert(H);
            else
              H->Gather = true;
          }
      // Gathered in the preheader, unless only part of the warp runs there.
      In |= !Outside || PartialEdge(P, B);
    }
    return In;
  };
  for (;;) {
    ArmIn.clear();
    JoinOf.clear();
    ArmAfter.clear();
    Blocked.clear();
    for (Region &R : Regions) {
      R.Gather = false;
      JoinOf[R.Join].push_back(&R);
      if (R.Header)
        ArmIn[R.Header].push_back(&R);
      else
        ArmAfter[R.Branch] = &R;
    }
    Out.clear();
    for (bool Again = true; Again;) {
      Again = false;
      for (MachineBasicBlock *B : RPOT) {
        bool O = Walk(*B, InState(B), /*Apply=*/false);
        if (Region *R = ArmAfter.lookup(B); R && O) {
          if (Partial.count(B)) {
            Blocked.insert(R);
          } else {
            R->Gather = true;
            O = false;
          }
        }
        if (O != Out.lookup(B)) {
          Out[B] = O;
          Again = true;
        }
      }
    }
    // Regions that would have to gather the warp where only part of it runs
    // lose their barrier, and the warp may be split after them too.
    bool Dropped = false;
    llvm::erase_if(Regions, [&](const Region &R) {
      if (!Blocked.count(const_cast<Region *>(&R)))
        return false;
      LLVM_DEBUG(dbgs() << "no barrier for " << printMBBReference(*R.Branch)
                        << ": entered while the warp may be split\n");
      SplitAt(R.Join, R.Blocks);
      for (MachineBasicBlock *X : R.Blocks)
        if (InBody(R, X))
          Partial.insert(X);
      return Dropped = true;
    });
    if (!Dropped)
      break;
    Changed = true;
  }
  for (MachineBasicBlock *B : RPOT)
    Walk(*B, InState(B), /*Apply=*/true);
  return Changed || any_of(Regions, [](const Region &R) { return R.Gather; });
}

static bool isBSync(const MachineInstr &MI) {
  return is_contained(BSYNCOpc, MI.getOpcode());
}

/// Whether \p A and \p B compute the same value (one is a rematerialized
/// copy of the other).
static bool sameValue(const MachineInstr &A, const MachineInstr &B,
                      const TargetRegisterInfo *TRI, unsigned Depth) {
  if (&A == &B)
    return true;
  if (Depth > 3 || !A.isIdenticalTo(B))
    return false;
  for (const MachineOperand &MO : A.uses()) {
    if (!MO.isReg() || !MO.getReg() || MO.getReg() == SASS::RZ ||
        MO.getReg() == SASS::PT || MO.getReg() == SASS::URZ ||
        MO.getReg() == SASS::UPT)
      continue;
    const MachineInstr *DA = reachingDef(A, MO.getReg(), TRI, Depth + 1);
    const MachineInstr *DB = reachingDef(B, MO.getReg(), TRI, Depth + 1);
    if (!DA || !DB || !sameValue(*DA, *DB, TRI, Depth + 1))
      return false;
  }
  return true;
}

/// After register allocation: the write of \p R that \p At reads, when one
/// value reaches it on every path (one instruction, or copies of it).
static const MachineInstr *reachingDef(const MachineInstr &At, Register R,
                                       const TargetRegisterInfo *TRI,
                                       unsigned Depth) {
  auto Before = [&](const MachineBasicBlock &B,
                    MachineBasicBlock::const_iterator I) -> const MachineInstr * {
    while (I != B.begin())
      if (const MachineInstr &MI = *--I; MI.modifiesRegister(R, TRI))
        return &MI;
    return nullptr;
  };
  if (const MachineInstr *Def = Before(*At.getParent(), At))
    return Def;
  const MachineInstr *Def = nullptr;
  SmallPtrSet<const MachineBasicBlock *, 32> Seen;
  SmallVector<const MachineBasicBlock *, 16> Work(At.getParent()->predecessors());
  while (!Work.empty()) {
    const MachineBasicBlock *B = Work.pop_back_val();
    if (!Seen.insert(B).second)
      continue;
    if (const MachineInstr *D = Before(*B, B->end())) {
      if (Def && !sameValue(*Def, *D, TRI, Depth))
        return nullptr;
      Def = D;
      continue;
    }
    if (B->pred_empty() || Seen.size() > 512)
      return nullptr;
    append_range(Work, B->predecessors());
  }
  return Def;
}

bool SASS::readsOnlyTidX(const MachineFunction &MF) {
  return none_of(MF, [](const MachineBasicBlock &X) {
    return any_of(X, [](const MachineInstr &MI) {
      return MI.getOpcode() == SASS::S2R_TIDY ||
             MI.getOpcode() == SASS::S2R_TIDZ;
    });
  });
}

bool SASS::branchesWarpUniformly(
    const MachineInstr &Br, bool OnlyTidX,
    function_ref<const MachineInstr *(const MachineInstr &, Register)> DefOf) {
  if (Br.getOpcode() != SASS::BRAc || !Br.getOperand(1).isReg())
    return false;
  // Through copies.
  auto Def = [&](const MachineInstr &User, Register R) {
    const MachineInstr *D = DefOf(User, R);
    for (unsigned N = 0; D && D->isCopy() && N < 8; ++N)
      D = D->getOperand(1).getSubReg() ? nullptr
                                       : DefOf(*D, D->getOperand(1).getReg());
    return D;
  };
  const MachineInstr *P = Def(Br, Br.getOperand(1).getReg());
  if (!P)
    return false;
  switch (P->getOpcode()) {
  case SASS::PLOP3_UP:
  case SASS::VOTE_ANY:
  case SASS::VOTE_ALL:
    return true;
  case SASS::ISETPri:
    break;
  default:
    return false;
  }
  // p = a <cc> imm (and PT), unguarded: cc bits [2:0] LT LE GT GE EQ NE,
  // bit 3 unsigned.
  if (!OnlyTidX || !P->getOperand(1).isReg() ||
      P->getOperand(1).getSubReg() || P->getOperand(4).getReg() != SASS::PT ||
      P->getOperand(5).getImm() ||
      P->getOperand(P->getNumExplicitOperands() - 2).getReg() != SASS::PT)
    return false;
  int64_t C = P->getOperand(2).getImm();
  unsigned CC = P->getOperand(3).getImm();
  // threadIdx.x shifted right by 5 or more (the warp index and coarser) is
  // the same in a warp; threadIdx.x itself on either side of a multiple of
  // 32 is too.
  const MachineInstr *A = Def(*P, P->getOperand(1).getReg());
  bool Coarse = false;
  while (A && (A->getOpcode() == SASS::SRLri ||
               A->getOpcode() == SASS::SRAri) &&
         A->getOperand(2).getImm() >= 5 && !A->getOperand(1).getSubReg()) {
    Coarse = true;
    A = Def(*A, A->getOperand(1).getReg());
  }
  if (!A || A->getOpcode() != SASS::S2R_TIDX)
    return false;
  if (Coarse)
    return true;
  if (C < 0 && !(CC & 8))
    return false;
  switch (CC & 7) {
  case 0: case 3: return C % 32 == 0;       // LT, GE
  case 1: case 2: return (C + 1) % 32 == 0; // LE, GT
  default: return false;
  }
}

/// The successor of \p B that at most one thread of a warp takes: \p Br
/// tests lane == 0 (%laneid, or threadIdx.x & 31, against 0), or
/// threadIdx.x == 0 in a kernel that reads only threadIdx.x.
static MachineBasicBlock *
singleThreadSide(MachineBasicBlock &B, const MachineInstr &Br, bool OnlyTidX,
                 function_ref<const MachineInstr *(const MachineInstr &,
                                                   Register)> DefOf) {
  if (!Br.getOperand(1).isReg())
    return nullptr;
  const MachineInstr *P = DefOf(Br, Br.getOperand(1).getReg());
  if (!P || P->getOpcode() != SASS::ISETPri || P->getOperand(2).getImm() ||
      !P->getOperand(1).isReg() || P->getOperand(4).getReg() != SASS::PT ||
      P->getOperand(5).getImm() ||
      P->getOperand(P->getNumExplicitOperands() - 2).getReg() != SASS::PT)
    return nullptr;
  unsigned CC = P->getOperand(3).getImm() & 7;
  if (CC != 4 && CC != 5) // EQ, NE
    return nullptr;
  const MachineInstr *L = DefOf(*P, P->getOperand(1).getReg());
  if (!L)
    return nullptr;
  if (L->getOpcode() == SASS::LOP3rir) { // threadIdx.x & 31
    const MachineInstr *T =
        L->getOperand(1).isReg() ? DefOf(*L, L->getOperand(1).getReg())
                                 : nullptr;
    if (!T || T->getOpcode() != SASS::S2R_TIDX ||
        L->getOperand(2).getImm() != 31 ||
        L->getOperand(3).getReg() != SASS::RZ ||
        (L->getOperand(4).getImm() & 0xff) != 0xc0)
      return nullptr;
  } else if (L->getOpcode() != SASS::S2R_LANEID &&
             (L->getOpcode() != SASS::S2R_TIDX || !OnlyTidX)) {
    return nullptr;
  }
  MachineBasicBlock *Taken = Br.getOperand(0).getMBB();
  bool TakenIsLane0 = (CC == 4) != (Br.getOperand(2).getImm() != 0);
  if (TakenIsLane0)
    return Taken;
  for (MachineBasicBlock *S : B.successors())
    if (S != Taken)
      return S;
  return nullptr;
}

/// Uniform registers and predicates are one per warp, and register allocation
/// assumes that values on different sides of a branch never meet. Two groups
/// of a split warp can run interleaved, so the rule (ptxas's) is that uniform
/// values are written only while the warp runs converged and read anywhere;
/// where code must write one while the warp may be split (an R2UR feeding an
/// elected bulk copy), nothing the other group runs meanwhile may use that
/// register. On the final code, with the convergence barriers in place, this
/// reports the uniform writes that break it:
///  * on one side of a divergent region with code on both sides (if/else),
///    to a register the other side uses (a group that skipped a one-sided
///    region waits at its BSYNC);
///  * in a region without a barrier, to a register used below its join until
///    the warp is gathered (WARPSYNC.ALL, or the BSYNC of an enclosing
///    region), and below that join at all, which both groups run;
///  * on a side of a branch parting ways to exit, to a register the other
///    side uses.
/// Branches that split no warp are left out: warp-uniform ones
/// (SASS::branchesWarpUniformly), and those below a lane == 0 test, which a
/// single thread runs.
static void verifyUniformWrites(MachineFunction &MF) {
  MachinePostDominatorTree PDT(MF);
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned NU = TRI->getNumRegUnits();
  MapVector<const MachineInstr *, std::string> Bad;
  auto AddUnits = [&](MCRegister R, BitVector &Out) {
    for (MCRegUnit U : TRI->regunits(R))
      Out.set(static_cast<unsigned>(U));
  };
  // Where a stretch of split code ends in a block: at the first instruction
  // that gathers the warp.
  auto Gathers = [](const MachineInstr &MI) {
    return MI.getOpcode() == SASS::WARPSYNC_ALL || isBSync(MI);
  };
  auto Upto = [&](MachineBasicBlock *X, bool Cut) {
    return make_range(X->begin(), Cut ? find_if(*X, Gathers) : X->end());
  };
  // Uniform register units a stretch reads, writes or carries (live-ins);
  // per whole block: Uses, and Defs written.
  std::vector<BitVector> Uses(MF.getNumBlockIDs()), Defs(Uses.size());
  auto Summarize = [&](MachineBasicBlock *X, bool Cut, BitVector &T,
                       BitVector *D) {
    for (const auto &LI : X->liveins())
      if (isUniformPhys(LI.PhysReg))
        AddUnits(LI.PhysReg, T);
    for (const MachineInstr &MI : Upto(X, Cut))
      for (const MachineOperand &MO : MI.operands())
        if (MO.isReg() && isUniformPhys(MO.getReg())) {
          AddUnits(MO.getReg(), T);
          if (D && MO.isDef())
            AddUnits(MO.getReg(), *D);
        }
  };
  for (MachineBasicBlock &X : MF) {
    Uses[X.getNumber()].resize(NU);
    Defs[X.getNumber()].resize(NU);
    Summarize(&X, false, Uses[X.getNumber()], &Defs[X.getNumber()]);
  }
  auto Touched = [&](const SmallPtrSetImpl<MachineBasicBlock *> &Blocks,
                     bool Cut) {
    BitVector T(NU);
    for (MachineBasicBlock *X : Blocks)
      if (Cut && any_of(*X, Gathers))
        Summarize(X, true, T, nullptr);
      else
        T |= Uses[X->getNumber()];
    return T;
  };
  auto Conflicts = [&](const SmallPtrSetImpl<MachineBasicBlock *> &Blocks,
                       bool Cut, const BitVector &Other, const char *Why,
                       const MachineBasicBlock &B) {
    for (MachineBasicBlock *X : Blocks)
      if (Defs[X->getNumber()].anyCommon(Other))
        for (const MachineInstr &MI : Upto(X, Cut))
          for (const MachineOperand &MO : MI.all_defs()) {
            if (!isUniformPhys(MO.getReg()) || Bad.count(&MI))
              continue;
            BitVector D(NU);
            AddUnits(MO.getReg(), D);
            if (!D.anyCommon(Other))
              continue;
            std::string S;
            raw_string_ostream(S) << Why << " of " << printMBBReference(B);
            Bad.insert({&MI, S});
          }
  };
  // Blocks reachable from From before Stop (and before B, around a loop).
  auto Reach = [](MachineBasicBlock *From, MachineBasicBlock *Stop,
                  MachineBasicBlock *B,
                  const SmallPtrSetImpl<MachineBasicBlock *> &Skip,
                  SmallPtrSetImpl<MachineBasicBlock *> &Out, bool Cut,
                  function_ref<bool(MachineBasicBlock *)> Ends) {
    SmallVector<MachineBasicBlock *, 16> Work{From};
    while (!Work.empty()) {
      MachineBasicBlock *X = Work.pop_back_val();
      if (X == Stop || X == B || Skip.count(X) || !Out.insert(X).second)
        continue;
      if (!Cut || !Ends(X))
        append_range(Work, X->successors());
    }
  };
  auto EndsSplit = [&](MachineBasicBlock *X) {
    return any_of(*X, Gathers);
  };
  // A side that goes straight to the barrier or the join.
  auto Empty = [](MachineBasicBlock *X, MachineBasicBlock *J) {
    for (unsigned N = 0; N < 8; ++N) {
      if (X == J)
        return true;
      auto I = X->getFirstNonDebugInstr();
      if (I != X->end() && isBSync(*I))
        return true;
      if (I != X->getFirstTerminator() || X->succ_size() != 1)
        return false;
      X = *X->succ_begin();
    }
    return false;
  };
  SmallPtrSet<MachineBasicBlock *, 8> Doomed = trapBlocks(MF);
  bool OnlyTidX = SASS::readsOnlyTidX(MF);
  auto DefOf = [&](const MachineInstr &User, Register R) {
    return reachingDef(User, R, TRI);
  };
  auto JoinOf = [&](MachineBasicBlock &B) -> MachineBasicBlock * {
    MachineDomTreeNode *N = PDT.getNode(&B);
    MachineBasicBlock *J =
        N && N->getIDom() ? N->getIDom()->getBlock() : nullptr;
    return J ? J : joinPastTraps(B, Doomed);
  };

  // Divergent branches, and the code below lane == 0 tests.
  SmallVector<MachineBasicBlock *, 32> Branches;
  SmallPtrSet<MachineBasicBlock *, 32> Single, None;
  for (MachineBasicBlock &B : MF) {
    auto T = B.getFirstTerminator();
    if (T == B.end() || T->getOpcode() != SASS::BRAc || B.succ_size() != 2 ||
        !PDT.getNode(&B) || SASS::branchesWarpUniformly(*T, OnlyTidX, DefOf))
      continue;
    Branches.push_back(&B);
    if (MachineBasicBlock *S = singleThreadSide(B, *T, OnlyTidX, DefOf)) {
      LLVM_DEBUG(dbgs() << "uniform-write check: one thread from "
                        << printMBBReference(*S) << "\n");
      SmallPtrSet<MachineBasicBlock *, 32> Side;
      Reach(S, JoinOf(B), &B, None, Side, false, EndsSplit);
      Single.insert(Side.begin(), Side.end());
    }
  }

  for (MachineBasicBlock *BP : Branches) {
    MachineBasicBlock &B = *BP;
    if (Single.count(&B))
      continue; // One thread: it cannot split.
    MachineBasicBlock *J = JoinOf(B);
    SmallPtrSet<MachineBasicBlock *, 32> Side[2];
    if (!J || isExitOnly(*J)) {
      // Each side up to a gather, which completes once the other side has
      // exited.
      for (auto [K, S] : enumerate(B.successors()))
        Reach(S, nullptr, &B, Doomed, Side[K], true, EndsSplit);
      for (unsigned K = 0; K < 2; ++K)
        Conflicts(Side[K], true, Touched(Side[1 - K], true),
                  "side parting ways to exit", B);
      continue;
    }
    for (auto [K, S] : enumerate(B.successors()))
      Reach(S, J, &B, None, Side[K], false, EndsSplit);
    if (none_of(B.successors(),
                [&](MachineBasicBlock *S) { return Empty(S, J); }))
      for (unsigned K = 0; K < 2; ++K)
        Conflicts(Side[K], false, Touched(Side[1 - K], false),
                  "two-sided region", B);
    // The BSYNC block itself post-dominates a region that is bracketed with
    // a single barrier block (or every way into J passes one).
    SmallPtrSet<MachineBasicBlock *, 32> Body(Side[0].begin(), Side[0].end());
    Body.insert(Side[1].begin(), Side[1].end());
    auto JI = J->getFirstNonDebugInstr();
    bool Bracketed = true;
    if (JI == J->end() || !isBSync(*JI))
      for (MachineBasicBlock *P : J->predecessors())
        if (Body.count(P) || P == &B) {
          auto I = P->getFirstNonDebugInstr();
          Bracketed &= P != &B && I != P->end() && isBSync(*I);
        }
    if (Bracketed)
      continue;
    // Without a barrier: the groups part at the join, and both run what
    // follows until the warp is gathered.
    LLVM_DEBUG(dbgs() << "uniform-write check: no barrier for "
                      << printMBBReference(B) << " -> "
                      << printMBBReference(*J) << " (" << Body.size()
                      << " blocks)\n");
    // A loop through the branch runs it again apart.
    if (any_of(B.predecessors(),
               [&](MachineBasicBlock *P) { return Body.count(P); }))
      Body.insert(&B);
    SmallPtrSet<MachineBasicBlock *, 32> Below;
    Reach(J, nullptr, nullptr, None, Below, true, EndsSplit);
    BitVector TB = Touched(Below, true);
    Conflicts(Body, false, TB, "region without a barrier", B);
    Conflicts(Below, true, Touched(Body, false), "split warp below", B);
    Conflicts(Below, true, TB, "split warp below", B);
  }

  for (auto &[MI, Why] : Bad)
    errs() << "sass-verify-uniform: " << MF.getName() << ": uniform write in "
           << Why << ", " << printMBBReference(*MI->getParent()) << ": "
           << *MI;
  if (!Bad.empty() && VerifyUniform == UniformCheck::Fatal)
    report_fatal_error("SASS: " + Twine(Bad.size()) +
                       " uniform register writes where the warp may be split "
                       "in " + MF.getName());
}

bool SASSReconverge::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = reconverge(MF);
  if (VerifyUniform != UniformCheck::Off &&
      SASS::reconverges(MF.getSubtarget<SASSSubtarget>()))
    verifyUniformWrites(MF);
  return Changed;
}

bool SASSReconverge::reconverge(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (!SASS::reconverges(ST))
    return false;
  TII = ST.getInstrInfo();
  MachinePostDominatorTree &PDT =
      getAnalysis<MachinePostDominatorTreeWrapperPass>().getPostDomTree();
  bool Changed = syncMerges(MF, PDT, TII);

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
  Changed |= gatherSplitWarps(MF, PDT, Regions, TII);
  // A region nested in another with the same join needs no barrier of its
  // own: the outer BSYNC at that join already waits for every thread the
  // inner branch split (`if (pos < end) { if (pos < seq) copy; } else zero;`
  // per unrolled row: ptxas brackets the row once). sm_90.
  if (!ST.hasBlackwellISA()) {
    SmallPtrSet<const Region *, 8> Redundant;
    for (const Region &R : Regions)
      if (!R.Header && !R.Gather)
        for (const Region &O : Regions)
          if (&O != &R && O.Join == R.Join && O.Blocks.count(R.Branch) &&
              O.Blocks.size() > R.Blocks.size()) {
            Redundant.insert(&R);
            break;
          }
    if (!Redundant.empty()) {
      std::vector<Region> Kept;
      for (Region &R : Regions)
        if (!Redundant.count(&R))
          Kept.push_back(std::move(R));
      Regions = std::move(Kept);
    }
  }
  if (Regions.empty())
    return Changed;
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
    if (R.Gather)
      BuildMI(*ArmBB, At, DL, TII->get(SASS::WARPSYNC_ALL))
          .addReg(SASS::PT)
          .addImm(0);
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
