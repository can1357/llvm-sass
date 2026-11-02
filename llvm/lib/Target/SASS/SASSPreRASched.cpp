//===-- SASSPreRASched.cpp - Pre-RA list scheduling for SASS ---------------===//
//
// Orders each block's instructions (in SSA, before PHI elimination) for an
// in-order issue machine without interlocks:
//
// * Top-down list scheduling over the block's dependence graph: data edges
//   carry the producer's issue-to-use latency (fixed pipes from the ptxas
//   tables, typical values for scoreboarded ones), memory edges keep stores,
//   barriers and side effects ordered against accesses they may alias, and
//   the block's terminators stay last.
// * Priority is the critical-path height to the block end, so long chains
//   (and loads feeding them) start first and independent work fills their
//   latency, e.g. two unrolled dot-product chains interleave, and a tile
//   copy issues its global loads before the shared stores that wait on them.
// * Loads, stores and shuffles pass through a ~4-entry LSU queue per
//   sub-core, drained at a quarter of the SM's ~0.5 instructions/clock when
//   all sub-cores load (sass-dec notes/sm120/mio_lsu_xu_topology.md). A
//   burst beyond the queue blocks issue, so memory instructions wait for a
//   queue slot and independent compute fills the gap.
// * Register pressure is bounded by the function's peak in the original
//   order plus a margin: the allocator must fit the worst block anyway, so
//   any block may use registers up to that level without costing occupancy.
//   Live predicates are bounded by the 7 allocatable ones: an unrolled tile
//   otherwise evaluates all of its mask compares before their selects and
//   spills most of them. Over a limit, instructions that free registers win.
//   Under a launch-bounds limit, rematerializable values (parameter LDCs,
//   constant MOVs) that a block only carries through do not count: the
//   allocator reloads them at their uses rather than exceed the limit.
//
// The pressure model counts GPR units and predicates of SSA values with exact
// block liveness. A COPY between virtual registers (one lane of a 128-bit load)
// is coalesced into its source, which without sub-register liveness stays
// live until its last lane is read, so copies are aliases of their root.
// On Hopper (sub-register liveness), a REG_SEQUENCE whose sources feed no
// other REG_SEQUENCE is coalesced with them: it shares their registers, which
// stay live until both the tuple and the source are dead (e.g. the HMMA B
// pairs of loop-carried 32-bit operands cost nothing beyond the operands).
//
// LLVM's GenericScheduler is disabled for SASS (SASSSubtarget): on unrolled
// tiles it hoisted every 128-bit load while under-reporting pressure.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSRegisterInfo.h"
#include "SASSSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "sass-prera-sched"

static cl::opt<unsigned> ExtraRegs(
    "sass-sched-regs", cl::init(48),
    cl::desc("32-bit registers a schedule may use over the function's peak "
             "pressure in the original order"));
static cl::opt<bool> Disable("sass-disable-prera-sched", cl::init(false),
                             cl::desc("Keep the instruction selector's order"));
static cl::opt<unsigned> RegSlack(
    "sass-sched-reg-slack", cl::init(8),
    cl::desc("Registers kept free below a launch-bounds register limit for "
             "allocation fragmentation"));
// The measured full-load interval is 8 cycles, but this kernel mix runs
// fewer than four loading warps per SM most of the time: 4 measured best
// on attention (2-5 within noise, 6+ slower).
static cl::opt<unsigned> LSUInterval(
    "sass-lsu-interval", cl::init(4),
    cl::desc("Cycles between a sub-core's LSU dispatches "
             "(0: no LSU queue model)"));
static cl::opt<unsigned> PredRegs(
    "sass-sched-preds", cl::init(7),
    cl::desc("Live predicates a schedule may hold (P0-P6 are allocatable)"));
static cl::opt<unsigned> Window(
    "sass-sched-window", cl::init(24),
    cl::desc("Lookahead (in original-order instructions) of the retry of a "
             "block whose schedule exceeds the register limit (0: no retry)"));
// Hopper's integer ALU pipe (LOP3, SHF, PRMT, IADD3, LEA, ISETP, SEL, ...) and
// its FMA-heavy pipe (IMAD in all forms, HMUL2/HFMA2) each accept one warp
// instruction every other cycle, so ptxas alternates them in dequantizing
// loops (LOP3, IMAD.SHL, SHF, HMUL2, ...): a run of either issues at half
// rate, and warps that reach such runs together stall on the pipe.
static cl::opt<unsigned> PipeInterval(
    "sass-pipe-interval", cl::init(2),
    cl::desc("Hopper: cycles between issues to the ALU or the FMA-heavy pipe "
             "(0: no pipe model)"));
// ptxas issues a dequantizing loop's packed-half multiplies by a shared scale
// back to back (with the scale in source b, which the integer instructions
// between them leave free: LOP3 a, imm, c), so the operand reuse cache serves
// the scale instead of the register file, whose two parity banks deliver one
// operand per clock to all of a sub-core's warps.
static cl::opt<bool> SchedReuse(
    "sass-sched-reuse", cl::init(false),
    cl::desc("Hopper: among equally urgent packed-half multiplies, continue "
             "with one sharing a source with the previous one, and put "
             "shared sources in slot b"));
// An HMMA writes all four lanes of its result even when the program reads
// only two (an f16 m16n8k16 product whose upper or lower rows are dropped).
// Dead at the HMMA, those registers were handed to the very next
// instruction, whose write then waited out the tensor-core latency (WAW:
// 26 stall cycles twice per attention tile). ptxas keeps them unassigned
// until the result lands.
static cl::opt<unsigned> MMAHold(
    "sass-mma-hold", cl::init(24),
    cl::desc("Hopper: instructions an HMMA's unread result lanes stay "
             "allocated after it (0: no hold)"));
static cl::opt<unsigned> ReuseMargin(
    "sass-sched-reuse-margin", cl::init(8),
    cl::desc("Registers below the limit the sass-sched-reuse grouping keeps "
             "free (it lengthens the multiplies' sources' live ranges)"));
constexpr unsigned LSUQueueDepth = 4;

namespace {

#include "SASSLatencyTables.inc"

struct Node {
  MachineInstr *MI;
  SmallVector<std::pair<unsigned, unsigned>, 4> Succs; // (node, latency)
  unsigned NumPreds = 0;
  unsigned Height = 0;
  int Earliest = 0;
  /// Number of the first later block reading the result (Hopper), else ~0u.
  unsigned NextUse = ~0u;
};

/// One block's scheduling region (after the PHIs, before the terminators)
/// with its dependence graph and pressure-model state.
struct Region {
  MachineBasicBlock *MBB = nullptr;
  SmallVector<Node, 0> Nodes;
  /// Remaining uses in the region per root register.
  DenseMap<Register, unsigned> UsesLeft;
  /// Roots still needed after the region.
  DenseSet<Register> Out;
  /// Pressure at the region start and the original order's peak.
  int Start = 0, Peak = 0;
  /// Live predicates at the region start.
  int PredStart = 0;
  /// GPR units (in Start) of rematerializable values the region only
  /// carries through: a parameter's LDC, a constant MOV.
  int Carried = 0;
};

using RegSet = DenseSet<Register>;

class SASSPreRASched : public MachineFunctionPass {
public:
  static char ID;
  SASSPreRASched() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS pre-RA scheduler"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const TargetRegisterInfo *TRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  /// Fixed latencies of the function's architecture.
  const LatencyTables *Lat = nullptr;
  DenseMap<const MachineBasicBlock *, RegSet> LiveIn, LiveOut;
  /// Coalesced REG_SEQUENCE tuples: tuple root -> source roots, and back
  /// (identical REG_SEQUENCEs, e.g. one per unrolled copy of a loop body,
  /// all share their sources' registers).
  DenseMap<Register, SmallVector<Register, 4>> Tuples;
  DenseMap<Register, SmallVector<Register, 2>> TupleOf;

  Register root(Register R) const;
  unsigned width(Register R) const;
  bool isPred(Register R) const;
  bool tracked(Register R) const { return width(R) || isPred(R); }
  void computeLiveness(MachineFunction &MF);
  void findTuples(MachineFunction &MF);
  Region buildRegion(MachineBasicBlock &MBB) const;
  std::pair<int, int> delta(const Region &Rg, const MachineInstr &MI,
                            DenseMap<Register, unsigned> &Left,
                            bool Commit) const;
  /// A block order, its peak pressure and length in cycles.
  struct Schedule {
    SmallVector<unsigned, 256> Order;
    int Peak = 0, Length = 0;
  };
  Schedule listSchedule(const Region &Rg, int Limit, unsigned Win, bool Pipes) const;
  bool schedule(Region &Rg, int Limit, int Slack) const;
  bool holdMMALanes(MachineFunction &MF) const;
};

} // namespace

char SASSPreRASched::ID = 0;

INITIALIZE_PASS(SASSPreRASched, DEBUG_TYPE, "SASS pre-RA scheduler", false,
                false)

FunctionPass *llvm::createSASSPreRASchedPass() { return new SASSPreRASched(); }

static bool isPseudo(const MachineInstr &MI) {
  return MI.isCopy() || MI.isRegSequence() || MI.isImplicitDef() ||
         MI.isInsertSubreg() || MI.isSubregToReg() || MI.isKill();
}

/// Low half of a Hopper 64-bit add, producing a carry predicate.
static bool isCarryOut(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == SASS::IADD3COrr || Opc == SASS::IADD3COri ||
         Opc == SASS::IADD3COrn;
}
/// High half of a Hopper 64-bit add, consuming a carry predicate.
static bool isCarryIn(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == SASS::IADD3Xrr || Opc == SASS::IADD3Xri ||
         Opc == SASS::IADD3Xrn;
}

/// Half-rate issue pipe of a Hopper instruction: 1 integer ALU, 2 FMA-heavy,
/// 0 neither (full-rate FP32, memory, uniform, control, tensor).
static unsigned halfRatePipe(const MachineInstr &MI) {
  if (isPseudo(MI))
    return 0;
  switch (SASS::latMnem(MI.getDesc())) {
  case LAT_IADD:
  case LAT_IADD3:
  case LAT_LOP3:
  case LAT_SHF:
  case LAT_ISETP:
  case LAT_SEL:
  case LAT_VIMNMX:
  case LAT_MOV:
  case LAT_PLOP3:
  case LAT_PRMT:
  case LAT_LEA:
  case LAT_IABS:
  case LAT_FSETP:
  case LAT_FSEL:
  case LAT_FMNMX:
    return 1;
  case LAT_IMAD:
  case LAT_HFMA2:
    return 2;
  default:
    return 0;
  }
}

/// A packed-half multiply (or FMA) with register sources a and b
/// (sass-sched-reuse).
static bool isPackedHalfMul(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case SASS::HBF16X2MUL: case SASS::HBF16X2FMA:
  case SASS::HF16X2MUL: case SASS::HF16X2FMA:
    return true;
  default:
    return false;
  }
}

/// Goes through the sub-core's LSU queue.
static bool usesLSU(const MachineInstr &MI) {
  return MI.mayLoadOrStore() || SASS::latMnem(MI.getDesc()) == LAT_SHFL;
}

/// Issue-to-use cycles from \p P to \p C under latency tables \p L.
static unsigned latency(const LatencyTables &L, const MachineInstr &P,
                        const MachineInstr &C) {
  if (isPseudo(P))
    return 0;
  const MCInstrDesc &D = P.getDesc();
  unsigned M = SASS::latMnem(D);
  if (SASS::writesSB(D)) {
    switch (M) {
    case LAT_LDG: case LAT_LD: return 400;
    case LAT_LDS: case LAT_LDSM: case LAT_SHFL: return 30;
    case LAT_LDC: case LAT_LDCU: return 15;
    case LAT_S2R: case LAT_S2UR: return 25;
    default: return 20; // MUFU, F2I, ...
    }
  }
  if (isPseudo(C)) {
    // Hopper: a copy or tuple of the value (an MMA accumulator's lanes, the
    // next MMA's operand) passes the latency on to its users in the block;
    // else independent MMA chains look free and are issued one after another.
    const MachineFunction &MF = *P.getMF();
    if (MF.getSubtarget<SASSSubtarget>().hasBlackwellISA() ||
        !(C.isCopy() || C.isRegSequence() || C.isInsertSubreg()))
      return 0;
    unsigned V = 0;
    for (const MachineOperand &MO : C.defs())
      if (MO.getReg().isVirtual())
        for (const MachineInstr &U :
             MF.getRegInfo().use_nodbg_instructions(MO.getReg()))
          if (U.getParent() == C.getParent() && !U.isPHI())
            V = std::max(V, latency(L, P, U));
    return V;
  }
  unsigned CM = SASS::latMnem(C.getDesc());
  bool Uniform = false;
  for (const MachineOperand &MO : P.defs()) {
    if (!MO.getReg().isVirtual())
      continue;
    const TargetRegisterClass *RC =
        P.getMF()->getRegInfo().getRegClass(MO.getReg());
    if (RC == &SASS::PREDRegClass) {
      // A Hopper 64-bit add's carry into its high half (SASSInstrInfo::
      // expandAdd64); other predicate uses plan for a guard/branch.
      return isCarryIn(C) && L.TruePred[M][CM] ? L.TruePred[M][CM] : 13;
    }
    Uniform |= RC == &SASS::UGPR32RegClass;
  }
  unsigned V = Uniform ? L.TrueUGPR[M][CM] : L.TrueGPR[M][CM];
  return V ? V : 6;
}

/// Address space of a single-memoperand access, or ~0u if unknown.
static unsigned addrSpace(const MachineInstr &MI) {
  return MI.hasOneMemOperand() ? (*MI.memoperands_begin())->getAddrSpace()
                               : ~0u;
}

/// Accesses in distinct, non-generic address spaces never alias.
static bool mayAlias(const MachineInstr &A, const MachineInstr &B) {
  unsigned SA = addrSpace(A), SB = addrSpace(B);
  return SA == ~0u || SB == ~0u || SA == 0 || SB == 0 || SA == SB;
}

/// Orders against every memory access and every other such instruction.
static bool isOrderingPoint(const MachineInstr &MI) {
  if (MI.hasUnmodeledSideEffects() || MI.isCall() || MI.hasOrderedMemoryRef())
    return true;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical())
      return true;
  return false;
}

/// The register a COPY chain starting at \p R copies from.
Register SASSPreRASched::root(Register R) const {
  while (R.isVirtual()) {
    const MachineInstr *Def = MRI->getVRegDef(R);
    if (!Def || !Def->isCopy() || !Def->getOperand(1).getReg().isVirtual())
      break;
    R = Def->getOperand(1).getReg();
  }
  return R;
}

/// 32-bit GPR units of \p R (0 for predicates, uniform and physical regs).
unsigned SASSPreRASched::width(Register R) const {
  if (!R.isVirtual())
    return 0;
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  if (!SASS::isGPRTupleClass(RC))
    return 0;
  return TRI->getRegSizeInBits(*RC) / 32;
}

/// A virtual predicate (the allocator has only P0-P6 for them).
bool SASSPreRASched::isPred(Register R) const {
  return R.isVirtual() && MRI->getRegClass(R) == &SASS::PREDRegClass;
}

/// Backward dataflow of live root registers per block. A PHI operand is
/// live out of its incoming block, not into the PHI's block.
void SASSPreRASched::computeLiveness(MachineFunction &MF) {
  LiveIn.clear();
  LiveOut.clear();
  DenseMap<const MachineBasicBlock *, RegSet> Gen, Kill, PhiUses;
  for (MachineBasicBlock &MBB : MF) {
    RegSet &G = Gen[&MBB], &K = Kill[&MBB];
    for (MachineInstr &MI : MBB) {
      if (MI.isPHI()) {
        if (tracked(MI.getOperand(0).getReg()))
          K.insert(MI.getOperand(0).getReg());
        for (unsigned I = 1; I + 1 < MI.getNumOperands(); I += 2) {
          Register R = root(MI.getOperand(I).getReg());
          if (tracked(R))
            PhiUses[MI.getOperand(I + 1).getMBB()].insert(R);
        }
        continue;
      }
      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && MO.getReg().isVirtual()) {
          Register R = root(MO.getReg());
          if (tracked(R) && !K.count(R))
            G.insert(R);
        }
      for (const MachineOperand &MO : MI.defs())
        if (MO.getReg().isVirtual() && root(MO.getReg()) == MO.getReg() &&
            tracked(MO.getReg()))
          K.insert(MO.getReg());
    }
  }
  // Sets only grow, so comparing sizes detects the fixpoint.
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (MachineBasicBlock *MBB : post_order(&MF)) {
      RegSet Out = PhiUses.lookup(MBB);
      for (MachineBasicBlock *S : MBB->successors())
        Out.insert_range(LiveIn[S]);
      RegSet In = Gen[MBB];
      for (Register R : Out)
        if (!Kill[MBB].count(R))
          In.insert(R);
      Changed |= In.size() != LiveIn[MBB].size() ||
                 Out.size() != LiveOut[MBB].size();
      LiveIn[MBB] = std::move(In);
      LiveOut[MBB] = std::move(Out);
    }
  }
}

/// REG_SEQUENCEs the register coalescer folds into their sources: every
/// source a distinct GPR root read by no REG_SEQUENCE of another shape
/// (sources and lanes), together filling the tuple.
void SASSPreRASched::findTuples(MachineFunction &MF) {
  Tuples.clear();
  TupleOf.clear();
  // Each source root's first reading REG_SEQUENCE; a root read by one of
  // another shape is no tuple source.
  DenseMap<Register, const MachineInstr *> Reader;
  DenseSet<Register> Mixed;
  auto SameShape = [&](const MachineInstr &A, const MachineInstr &B) {
    if (A.getNumOperands() != B.getNumOperands())
      return false;
    for (unsigned I = 1; I + 1 < A.getNumOperands(); I += 2) {
      const MachineOperand &X = A.getOperand(I), &Y = B.getOperand(I);
      if (!X.isReg() || !Y.isReg() || root(X.getReg()) != root(Y.getReg()) ||
          A.getOperand(I + 1).getImm() != B.getOperand(I + 1).getImm())
        return false;
    }
    return true;
  };
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isRegSequence())
        for (unsigned I = 1; I < MI.getNumOperands(); I += 2)
          if (MI.getOperand(I).isReg()) {
            Register S = root(MI.getOperand(I).getReg());
            auto [It, New] = Reader.try_emplace(S, &MI);
            if (!New && !SameShape(*It->second, MI))
              Mixed.insert(S);
          }
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (!MI.isRegSequence())
        continue;
      Register T = MI.getOperand(0).getReg();
      SmallVector<Register, 4> Srcs;
      unsigned W = 0;
      bool OK = width(T) != 0;
      for (unsigned I = 1; OK && I < MI.getNumOperands(); I += 2) {
        const MachineOperand &MO = MI.getOperand(I);
        Register S = MO.isReg() && !MO.isUndef() ? root(MO.getReg()) : Register();
        OK = width(S) && !Mixed.count(S) && !is_contained(Srcs, S);
        if (OK) {
          Srcs.push_back(S);
          W += width(S);
        }
      }
      if (!OK || W != width(T))
        continue;
      for (Register S : Srcs)
        TupleOf[S].push_back(T);
      Tuples[T] = std::move(Srcs);
    }
}

/// Pressure change from issuing \p MI given the remaining uses \p Left;
/// \p Commit consumes its uses. Returns the change in GPR units and in
/// predicates.
std::pair<int, int> SASSPreRASched::delta(const Region &Rg,
                                          const MachineInstr &MI,
                                          DenseMap<Register, unsigned> &Left,
                                          bool Commit) const {
  int D = 0, PD = 0;
  SmallDenseMap<Register, unsigned, 8> Seen;
  for (const MachineOperand &MO : MI.uses()) {
    Register R = MO.isReg() ? root(MO.getReg()) : Register();
    if (tracked(R))
      ++Seen[R];
  }
  auto LiveAfter = [&](Register R) {
    return Rg.Out.count(R) || Left.lookup(R) > Seen.lookup(R);
  };
  // A coalesced tuple occupies its sources' registers.
  Register NewTuple;
  for (const MachineOperand &MO : MI.defs()) {
    Register R = MO.getReg();
    if (root(R) == R && (Left.count(R) || Rg.Out.count(R))) {
      if (Tuples.count(R)) {
        NewTuple = R;
        continue;
      }
      D += width(R);
      PD += isPred(R);
    }
  }
  for (auto [R, K] : Seen) {
    if (LiveAfter(R))
      continue;
    // R dies here. A tuple source keeps its registers while a tuple of it
    // lives; a dying tuple frees the sources that are already dead and held
    // by no other live tuple.
    auto Held = [&](Register S, Register Except) {
      for (Register T : TupleOf.lookup(S))
        if (T != Except && (T == NewTuple || LiveAfter(T)))
          return true;
      return false;
    };
    if (TupleOf.count(R)) {
      if (!Held(R, Register()))
        D -= width(R);
      continue;
    }
    if (auto It = Tuples.find(R); It != Tuples.end()) {
      for (Register S : It->second)
        if (!Rg.Out.count(S) && !Left.lookup(S) && !Held(S, R))
          D -= width(S);
      continue;
    }
    D -= width(R);
    PD -= isPred(R);
  }
  if (Commit)
    for (auto [R, K] : Seen)
      Left[R] -= K;
  return {D, PD};
}

Region SASSPreRASched::buildRegion(MachineBasicBlock &MBB) const {
  Region Rg;
  Rg.MBB = &MBB;
  SmallVector<Node, 0> &Nodes = Rg.Nodes;
  DenseMap<const MachineInstr *, unsigned> Idx;
  MachineBasicBlock::iterator End = MBB.getFirstTerminator();
  for (auto I = MBB.getFirstNonPHI(); I != End; ++I) {
    if (I->isDebugInstr())
      continue;
    Idx[&*I] = Nodes.size();
    Nodes.emplace_back().MI = &*I;
  }
  unsigned N = Nodes.size();

  auto AddEdge = [&](unsigned From, unsigned To, unsigned Lat) {
    Nodes[From].Succs.push_back({To, Lat});
    ++Nodes[To].NumPreds;
  };

  // Dependences, in original (topological) order.
  int LastOrder = -1;
  int LastSync = -1; // WARPGROUP.ARRIVE/DEPBAR: nothing moves across them.
  SmallVector<unsigned, 32> MemSinceOrder;
  for (unsigned I = 0; I < N; ++I) {
    MachineInstr &MI = *Nodes[I].MI;
    SmallDenseSet<unsigned, 8> Preds;
    for (const MachineOperand &MO : MI.uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      MachineInstr *Def = MRI->getVRegDef(MO.getReg());
      auto It = Def ? Idx.find(Def) : Idx.end();
      if (It != Idx.end() && Preds.insert(It->second).second)
        AddEdge(It->second, I, latency(*Lat, *Def, MI));
    }
    if (SASS::isWarpgroupSync(MI)) {
      for (int J = LastSync + 1; J < (int)I; ++J)
        if (Preds.insert(J).second)
          AddEdge(J, I, 0);
      LastSync = I;
    } else if (LastSync >= 0 && Preds.insert(LastSync).second) {
      AddEdge(LastSync, I, 0);
    }
    bool Order = isOrderingPoint(MI);
    if (!Order && !MI.mayLoadOrStore())
      continue;
    if (LastOrder >= 0 && Preds.insert(LastOrder).second)
      AddEdge(LastOrder, I, 0);
    if (Order) {
      for (unsigned M : MemSinceOrder)
        if (Preds.insert(M).second)
          AddEdge(M, I, 0);
      MemSinceOrder.clear();
      LastOrder = I;
      continue;
    }
    for (unsigned M : MemSinceOrder) {
      const MachineInstr &Prev = *Nodes[M].MI;
      if ((Prev.mayStore() || MI.mayStore()) && mayAlias(Prev, MI) &&
          Preds.insert(M).second)
        AddEdge(M, I, 0);
    }
    MemSinceOrder.push_back(I);
  }

  // Heights: latency-weighted longest path to the block end, including the
  // latency to the terminators' uses (the branch predicate).
  for (unsigned I = N; I-- > 0;) {
    Node &Nd = Nodes[I];
    unsigned H = 0;
    for (auto [S, Lat] : Nd.Succs)
      H = std::max(H, Lat + Nodes[S].Height);
    for (MachineInstr &T : make_range(End, MBB.end()))
      for (const MachineOperand &MO : T.uses())
        if (MO.isReg() && MO.getReg().isVirtual() &&
            MRI->getVRegDef(MO.getReg()) == Nd.MI)
          H = std::max(H, latency(*Lat, *Nd.MI, T));
    Nd.Height = H;
  }

  // Hopper: equal heights go to the result a later block needs first. The
  // selector emits independent loads in reverse, so a prologue issuing the
  // loads of a K loop's first steps otherwise issued step 0's last and the
  // loop's first MMA waited for all of them.
  if (!MBB.getParent()->getSubtarget<SASSSubtarget>().hasBlackwellISA())
    for (Node &Nd : Nodes)
      for (const MachineOperand &MO : Nd.MI->defs())
        if (MO.getReg().isVirtual())
          for (const MachineInstr &U : MRI->use_nodbg_instructions(MO.getReg()))
            if (U.getParent() != &MBB &&
                U.getParent()->getNumber() > MBB.getNumber())
              Nd.NextUse = std::min(Nd.NextUse,
                                    unsigned(U.getParent()->getNumber()));

  // Pressure model: live roots at the region start, region uses, and roots
  // still needed afterwards (live out, or read by a terminator).
  Rg.Out = LiveOut.lookup(&MBB);
  for (MachineInstr &T : make_range(End, MBB.end()))
    for (const MachineOperand &MO : T.uses())
      if (MO.isReg() && tracked(root(MO.getReg())))
        Rg.Out.insert(root(MO.getReg()));
  for (Node &Nd : Nodes)
    for (const MachineOperand &MO : Nd.MI->uses())
      if (MO.isReg() && tracked(root(MO.getReg())))
        ++Rg.UsesLeft[root(MO.getReg())];
  RegSet StartSet = LiveIn.lookup(&MBB);
  for (MachineInstr &Phi : MBB.phis()) {
    Register R = Phi.getOperand(0).getReg();
    if (Rg.UsesLeft.count(R) || Rg.Out.count(R))
      StartSet.insert(R);
  }
  for (Register R : StartSet) {
    // (a live tuple counts its sources' registers)
    if (any_of(TupleOf.lookup(R), [&](Register T) { return StartSet.count(T); }))
      continue;
    Rg.Start += width(R);
    Rg.PredStart += isPred(R);
    if (!Rg.UsesLeft.count(R))
      if (const MachineInstr *Def = MRI->getVRegDef(R);
          Def && TII->isTriviallyReMaterializable(*Def))
        Rg.Carried += width(R);
  }
  int P = Rg.Peak = Rg.Start;
  DenseMap<Register, unsigned> Left = Rg.UsesLeft;
  for (Node &Nd : Nodes) {
    P += delta(Rg, *Nd.MI, Left, /*Commit=*/true).first;
    Rg.Peak = std::max(Rg.Peak, P);
  }
  return Rg;
}

/// Top-down list schedule of \p Rg under register limit \p Limit. A nonzero
/// \p Win only considers instructions fewer than \p Win positions after the
/// first unscheduled one in the original order. \p Pipes models Hopper's
/// half-rate ALU and FMA-heavy pipes.
SASSPreRASched::Schedule
SASSPreRASched::listSchedule(const Region &Rg, int Limit, unsigned Win, bool Pipes) const {
  const SmallVector<Node, 0> &Nodes = Rg.Nodes;
  unsigned N = Nodes.size();
  SmallVector<unsigned, 0> NumPreds;
  SmallVector<int, 0> Earliest;
  SmallVector<unsigned, 64> Avail;
  for (unsigned I = 0; I < N; ++I) {
    NumPreds.push_back(Nodes[I].NumPreds);
    Earliest.push_back(Nodes[I].Earliest);
    if (!Nodes[I].NumPreds)
      Avail.push_back(I);
  }
  Schedule S;
  SmallVector<unsigned, 256> &Order = S.Order;
  int Cycle = 0, P = Rg.Start, SchedPeak = Rg.Start, PP = Rg.PredStart;
  DenseMap<Register, unsigned> Left = Rg.UsesLeft;
  // Dispatch cycles of the last LSUQueueDepth LSU instructions, oldest at
  // QHead: the next one enters the queue once the oldest has left it.
  bool ModelLSU = LSUInterval != 0;
  int Dispatch[LSUQueueDepth];
  std::fill(std::begin(Dispatch), std::end(Dispatch), INT_MIN / 2);
  unsigned QHead = 0;
  int LastDispatch = INT_MIN / 2;
  // Cycle from which each half-rate pipe (halfRatePipe) accepts the next
  // instruction.
  bool ModelPipes = Pipes && PipeInterval != 0 && Lat == &SM90Latencies;
  int PipeFree[3] = {0, 0, 0};
  auto ReadyAt = [&](unsigned C) {
    int R = Earliest[C];
    if (ModelLSU && usesLSU(*Nodes[C].MI))
      R = std::max(R, Dispatch[QHead]);
    if (ModelPipes)
      if (unsigned Pp = halfRatePipe(*Nodes[C].MI))
        R = std::max(R, PipeFree[Pp]);
    return R;
  };
  // A carry predicate cannot be spilled: a 64-bit add's carry-in half
  // follows its carry-out half at once, so at most one carry is live (as in
  // ptxas's code).
  unsigned CarryNext = ~0u;
  // The last packed-half multiply's opcode and sources a, b: an equally
  // urgent one reading one of them follows it, and so does the last step of
  // such a multiply's other source (sass-sched-reuse).
  bool ModelReuse = SchedReuse && Lat == &SM90Latencies;
  unsigned LastOpc = 0;
  Register LastSrc[2];
  auto Shares = [&](const MachineInstr &MI) {
    if (MI.getOpcode() != LastOpc)
      return false;
    for (unsigned I : {1, 2}) {
      Register R = MI.getOperand(I).getReg();
      if (R == LastSrc[0] || R == LastSrc[1])
        return true;
    }
    return false;
  };
  // The packed-half multiply \p MI is or whose source it computes (in the
  // block), if any.
  auto feedsPackedHalfMul = [&](const MachineInstr &MI) -> const MachineInstr * {
    if (isPackedHalfMul(MI))
      return &MI;
    if (MI.getNumDefs() != 1 || !MI.getOperand(0).isReg() ||
        !MI.getOperand(0).getReg().isVirtual())
      return nullptr;
    for (const MachineInstr &U :
         MRI->use_nodbg_instructions(MI.getOperand(0).getReg()))
      if (U.getParent() == MI.getParent() && isPackedHalfMul(U))
        return &U;
    return nullptr;
  };
  auto SharesSource = [&](const MachineInstr &MI) {
    if (!ModelReuse || !LastOpc)
      return false;
    if (isPackedHalfMul(MI))
      return Shares(MI);
    const MachineInstr *M = feedsPackedHalfMul(MI);
    return M && Shares(*M);
  };
  // The first unscheduled instruction in the original order (its
  // predecessors precede it there, so it is always available).
  SmallVector<bool, 0> Done(N, false);
  unsigned Oldest = 0;
  while (!Avail.empty()) {
    // In the window, fits the limits, else exceeds them the least; then
    // ready; then height; then a packed-half multiply sharing a source with
    // the previous one; then the earliest later block reading it; then
    // original order.
    using Key =
        std::tuple<bool, bool, int, bool, unsigned, bool, unsigned, int>;
    unsigned BestPos = 0;
    Key Best;
    while (Oldest < N && Done[Oldest])
      ++Oldest;
    auto Forced = find(Avail, CarryNext);
    for (unsigned K = 0; K < Avail.size() && Forced == Avail.end(); ++K) {
      unsigned C = Avail[K];
      auto [D, PD] = delta(Rg, *Nodes[C].MI, Left, /*Commit=*/false);
      // Registers over the limits after issuing C: predicates over P0-P6
      // spill too (one byte each, through a GPR).
      int Over = std::max(P + D - Limit, 0) +
                 std::max(PP + PD - int(PredRegs), 0);
      bool Fits = !Over;
      bool InWin = !Win || C < Oldest + Win;
      Key Cand = {InWin, Fits, -Over, ReadyAt(C) <= Cycle, Nodes[C].Height,
                  P + D + int(ReuseMargin) <= Limit &&
                      SharesSource(*Nodes[C].MI),
                  ~Nodes[C].NextUse, -int(C)};
      if (K == 0 || Cand > Best) {
        Best = Cand;
        BestPos = K;
      }
    }
    if (Forced != Avail.end())
      BestPos = Forced - Avail.begin();
    unsigned C = Avail[BestPos];
    Avail.erase(Avail.begin() + BestPos);
    const Node &Nd = Nodes[C];
    if (ModelReuse)
      if (const MachineInstr *M = feedsPackedHalfMul(*Nd.MI)) {
        LastOpc = M->getOpcode();
        LastSrc[0] = M->getOperand(1).getReg();
        LastSrc[1] = M->getOperand(2).getReg();
      }
    CarryNext = ~0u;
    if (isCarryOut(*Nd.MI))
      for (auto [S, Lat] : Nd.Succs)
        if (isCarryIn(*Nodes[S].MI))
          CarryNext = S;
    LLVM_DEBUG(dbgs() << "  @" << std::max(Cycle, ReadyAt(C)) << " h"
                      << Nd.Height << " e" << Earliest[C] << " P" << P << " "
                      << *Nd.MI);
    Cycle = std::max(Cycle, ReadyAt(C));
    if (ModelPipes)
      if (unsigned Pp = halfRatePipe(*Nd.MI))
        PipeFree[Pp] = Cycle + int(PipeInterval);
    auto [D, PD] = delta(Rg, *Nd.MI, Left, /*Commit=*/true);
    P += D;
    PP += PD;
    SchedPeak = std::max(SchedPeak, P);
    Order.push_back(C);
    Done[C] = true;
    // A queued memory instruction's latency starts at its dispatch.
    int Start = Cycle;
    if (ModelLSU && usesLSU(*Nd.MI)) {
      Start = std::max(Cycle, LastDispatch + int(LSUInterval));
      LastDispatch = Dispatch[QHead] = Start;
      QHead = (QHead + 1) % LSUQueueDepth;
    }
    for (auto [Sc, Lat] : Nd.Succs) {
      Earliest[Sc] = std::max(Earliest[Sc], (Lat ? Start : Cycle) + int(Lat));
      if (--NumPreds[Sc] == 0)
        Avail.push_back(Sc);
    }
    if (!isPseudo(*Nd.MI))
      Cycle += std::max(1u, SASS::minStall(Nd.MI->getDesc()));
  }
  assert(Order.size() == N && "dependence cycle in a block");
  S.Peak = SchedPeak;
  S.Length = Cycle;
  LLVM_DEBUG(dbgs() << printMBBReference(*Rg.MBB) << ": " << N
                    << " instrs, pressure " << Rg.Peak << " -> " << SchedPeak
                    << " (limit " << Limit << ", window " << Win
                    << "), schedule length " << Cycle << "\n");
  return S;
}

/// Reorders \p Rg's block. A schedule over the register limit retries with a
/// lookahead window: unbounded, the list scheduler hoists every load and
/// starts every independent chain of an unrolled block before finishing any
/// (a dequantizing MMA loop then spills), while the source order of such a
/// loop body is already chunk by chunk. Before that, it retries with the
/// \p Slack registers kept free for allocation fragmentation: a block that
/// cannot stay under the limit fills it with the longest chains and then
/// waits on whichever instruction frees a register (mid32's two-tile MoE
/// unit: 2196 cycles at the limit, 1096 and a lower peak with the slack).
bool SASSPreRASched::schedule(Region &Rg, int Limit, int Slack) const {
  // Over the limit, the pipe model's interleaving can cost registers: each
  // attempt also runs without it and keeps the lower peak (mid32's two-tile
  // MoE unit spilled 21 more registers with it).
  auto Attempt = [&](int L, unsigned Win) {
    Schedule A = listSchedule(Rg, L, Win, /*Pipes=*/true);
    if (A.Peak > Limit) {
      Schedule B = listSchedule(Rg, L, Win, /*Pipes=*/false);
      if (B.Peak < A.Peak)
        return B;
    }
    return A;
  };
  Schedule S = Attempt(Limit, 0);
  if (S.Peak > Limit && Slack) {
    Schedule H = Attempt(Limit + Slack, 0);
    if (H.Peak < S.Peak || (H.Peak == S.Peak && H.Length < S.Length))
      S = std::move(H);
  }
  if (S.Peak > Limit && Window) {
    Schedule W = Attempt(Limit, Window);
    if (W.Peak < S.Peak)
      S = std::move(W);
  }
  const SmallVector<unsigned, 256> &Order = S.Order;
  unsigned N = Order.size();
  MachineBasicBlock &MBB = *Rg.MBB;
  MachineBasicBlock::iterator End = MBB.getFirstTerminator();
  bool Changed = false;
  for (unsigned K = 0; K < N; ++K) {
    MachineInstr *MI = Rg.Nodes[Order[K]].MI;
    Changed |= Order[K] != K;
    MBB.splice(End, &MBB, MI);
    for (MachineOperand &MO : MI->uses())
      if (MO.isReg())
        MO.setIsKill(false);
  }
  // The source most packed-half multiplies of the block read (a scale) goes
  // to slot b, where the reuse cache keeps it across the integer
  // instructions between them (sass-sched-reuse).
  if (SchedReuse && Lat == &SM90Latencies) {
    DenseMap<Register, unsigned> Reads;
    for (const Node &Nd : Rg.Nodes)
      if (isPackedHalfMul(*Nd.MI)) {
        ++Reads[Nd.MI->getOperand(1).getReg()];
        ++Reads[Nd.MI->getOperand(2).getReg()];
      }
    for (const Node &Nd : Rg.Nodes) {
      if (!isPackedHalfMul(*Nd.MI))
        continue;
      MachineOperand &A = Nd.MI->getOperand(1), &B = Nd.MI->getOperand(2);
      Register RA = A.getReg(), RB = B.getReg();
      if (Reads[RA] <= Reads[RB])
        continue;
      unsigned SA = A.getSubReg(), SB = B.getSubReg();
      A.setReg(RB);
      A.setSubReg(SB);
      B.setReg(RA);
      B.setSubReg(SA);
      Changed = true;
    }
  }
  return Changed;
}

bool SASSPreRASched::runOnMachineFunction(MachineFunction &MF) {
  if (Disable)
    return false;
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  Lat = ST.hasBlackwellISA() ? &SM120Latencies : &SM90Latencies;
  Tuples.clear();
  TupleOf.clear();
  if (ST.enableSubRegLiveness())
    findTuples(MF);
  computeLiveness(MF);
  SmallVector<Region, 8> Regions;
  int FuncPeak = 0;
  for (MachineBasicBlock &MBB : MF) {
    Regions.push_back(buildRegion(MBB));
    FuncPeak = std::max(FuncPeak, Regions.back().Peak);
  }
  // Under a register limit (__launch_bounds__ minimum blocks) the schedule
  // stays below it, reordering to free registers where the original order
  // exceeds it. The limit counts 3 registers the model does not
  // (SASSAsmPrinter::regCount) and tuple alignment can waste a few more.
  // With setmaxnreg a consumer region after setmaxnreg.inc is scheduled under
  // its larger count (under the launch allocation its independent MMA chains
  // were serialized). Regions after setmaxnreg.dec keep the launch count:
  // under their smaller one, an async-copy producer computed each address
  // right before its LDGSTS, the allocator gave all of them one register
  // pair, and every address then waited for the previous LDGSTS to read its
  // operands out of the MIO queue (~380 cycles per attention tile).
  const bool SetMax = SASS::setMaxRegEntryCount(MF) != 0;
  const unsigned MaxRegs = SetMax ? SASS::setMaxRegLimit(MF)
                                  : SASS::maxRegCount(MF.getFunction());
  const auto &BlockRegs = MF.getInfo<SASSFunctionInfo>()->RegLimit;
  bool Changed = false;
  for (Region &Rg : Regions) {
    if (Rg.Nodes.size() <= 2)
      continue;
    int Limit = FuncPeak + int(ExtraRegs);
    unsigned Regs = MaxRegs;
    if (SetMax)
      if (auto It = BlockRegs.find(Rg.MBB); It != BlockRegs.end())
        Regs = std::max(It->second, SASS::setMaxRegEntryCount(MF));
    if (Regs < 255)
      Limit = std::min(Limit, int(Regs) - 3 - int(RegSlack));
    // A block over a launch-bounds count does not hold the rematerializable
    // values it only carries through: the allocator reloads them at their
    // uses rather than exceed the count. Counted, the bs=1 MoE kernel's K
    // loop blocks (28 such registers) stayed over the limit in any order and
    // the fallback serialized their dequant chains.
    if (!ST.hasBlackwellISA() && !SetMax && Regs < 255 && Rg.Peak > Limit &&
        Rg.Carried) {
      LLVM_DEBUG(dbgs() << printMBBReference(*Rg.MBB) << ": " << Rg.Carried
                        << " rematerializable registers carried through\n");
      Rg.Start -= Rg.Carried;
      Rg.Peak -= Rg.Carried;
    }
    Changed |= schedule(Rg, Limit,
                        !ST.hasBlackwellISA() && Regs < 255 ? int(RegSlack) : 0);
  }
  if (!ST.hasBlackwellISA() && MMAHold)
    Changed |= holdMMALanes(MF);
  return Changed;
}

/// Hopper: keeps the lanes no instruction reads of each HMMA result
/// allocated (FAKE_USE, erased after allocation) until its first reader or
/// MMAHold instructions later, so the allocator does not give them to an
/// instruction that would wait for the HMMA to land before writing them.
bool SASSPreRASched::holdMMALanes(MachineFunction &MF) const {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != SASS::HMMA16816F32 && Opc != SASS::HMMA16816F32BF16 &&
          Opc != SASS::HMMA16816F32Z && Opc != SASS::HMMA16816F32BF16Z)
        continue;
      Register D = MI.getOperand(0).getReg();
      if (!D.isVirtual() || MI.getOperand(0).getSubReg())
        continue;
      LaneBitmask Read = LaneBitmask::getNone();
      for (const MachineOperand &MO : MRI->use_nodbg_operands(D))
        Read |= MO.getSubReg() ? TRI->getSubRegIndexLaneMask(MO.getSubReg())
                               : MRI->getMaxLaneMaskForVReg(D);
      if ((Read & MRI->getMaxLaneMaskForVReg(D)) ==
          MRI->getMaxLaneMaskForVReg(D))
        continue;
      auto At = std::next(MI.getIterator());
      for (unsigned N = 0; At != MBB.end() && !At->isTerminator() &&
                           !At->readsVirtualRegister(D) && N < MMAHold;
           ++At)
        N += !isPseudo(*At);
      BuildMI(MBB, At, MI.getDebugLoc(), TII->get(TargetOpcode::FAKE_USE))
          .addReg(D);
      Changed = true;
    }
  return Changed;
}

//===----------------------------------------------------------------------===//
// SASSFoldLanes
//
// Hopper, in SSA before register allocation. 64-bit values built from 32-bit
// halves (REG_SEQUENCE: a zero extension's `MOV 0` high word, a generic
// shared address's window high word, a descriptor's constant high word) are
// split again into 32-bit LOP3s and adds after the DAG combiner has run, so
// nothing folds `hi ^ hi` (the high word of an XOR toggle between two shared
// addresses or descriptors) or `0 | c`. Each such value then occupies a
// register for its whole live range, across loops. This pass
// * reads a lane of a REG_SEQUENCE (and the source of a vreg COPY) directly,
// * reads a vreg holding `MOV 0` as RZ in integer ALU operands,
// * re-evaluates LOP3 truth tables with known-zero / all-ones inputs and
//   repeated operands: a constant result becomes a MOV, a single input a COPY,
// and deletes the instructions this leaves unused.
//===----------------------------------------------------------------------===//

namespace {
class SASSFoldLanes : public MachineFunctionPass {
public:
  static char ID;
  SASSFoldLanes() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS fold lanes"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  bool forward(MachineInstr &MI);
  bool zeroToRZ(MachineInstr &MI);
  bool simplifyLOP3(MachineInstr &MI);
  /// The immediate a vreg holds (an unpredicated MOV), if any.
  std::optional<int64_t> constOf(const MachineOperand &MO) const;
};
} // namespace

char SASSFoldLanes::ID = 0;

INITIALIZE_PASS(SASSFoldLanes, "sass-fold-lanes", "SASS fold lanes", false,
                false)

FunctionPass *llvm::createSASSFoldLanesPass() { return new SASSFoldLanes(); }

std::optional<int64_t>
SASSFoldLanes::constOf(const MachineOperand &MO) const {
  if (MO.isImm())
    return MO.getImm();
  if (!MO.isReg())
    return std::nullopt;
  if (MO.getReg() == SASS::RZ)
    return 0;
  if (!MO.getReg().isVirtual() || MO.getSubReg())
    return std::nullopt;
  const MachineInstr *D = MRI->getVRegDef(MO.getReg());
  if (!D || D->getOpcode() != SASS::MOVri || TII->isPredicated(*D) ||
      !D->getOperand(1).isImm())
    return std::nullopt;
  return D->getOperand(1).getImm();
}

/// Reads through vreg COPYs and REG_SEQUENCE lanes where the instruction
/// accepts the source's class.
bool SASSFoldLanes::forward(MachineInstr &MI) {
  if (MI.isPHI() || MI.isRegSequence() || MI.isInlineAsm() ||
      (MI.isCopy() && !MI.getOperand(0).getReg().isVirtual()))
    return false;
  bool Changed = false;
  for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse() || MO.isImplicit() || MO.isTied() ||
        !MO.getReg().isVirtual())
      continue;
    for (;;) {
      const MachineInstr *D = MRI->getVRegDef(MO.getReg());
      if (!D)
        break;
      Register Src;
      if (D->isCopy() && !MO.getSubReg() && !D->getOperand(1).getSubReg() &&
          D->getOperand(1).getReg().isVirtual()) {
        Src = D->getOperand(1).getReg();
      } else if (D->isRegSequence() && MO.getSubReg()) {
        for (unsigned K = 1; K + 1 < D->getNumOperands(); K += 2)
          if (D->getOperand(K + 1).getImm() == MO.getSubReg() &&
              !D->getOperand(K).getSubReg() &&
              D->getOperand(K).getReg().isVirtual())
            Src = D->getOperand(K).getReg();
      }
      if (!Src)
        break;
      const TargetRegisterClass *SrcRC = MRI->getRegClass(Src);
      const TargetRegisterClass *Want =
          MI.isCopy() ? MRI->getRegClass(MI.getOperand(0).getReg())
                      : MI.getRegClassConstraint(I, TII, TRI);
      if (MI.isCopy() && MO.getSubReg())
        Want = nullptr; // (a lane COPY's source class is the lane's)
      if (Want && !Want->hasSubClassEq(SrcRC))
        break;
      if (!Want && TRI->getRegSizeInBits(*SrcRC) !=
                       (MO.getSubReg()
                            ? TRI->getSubRegIdxSize(MO.getSubReg())
                            : TRI->getRegSizeInBits(*MRI->getRegClass(
                                  MO.getReg()))))
        break;
      MO.setReg(Src);
      MO.setSubReg(0);
      MO.setIsKill(false);
      Changed = true;
    }
  }
  return Changed;
}

/// `MOV 0` operands of integer ALU instructions become RZ.
bool SASSFoldLanes::zeroToRZ(MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if (!SASSInstrInfo::isCheapALU(MI) && Opc != SASS::ISETPrr &&
      Opc != SASS::ISETPri && Opc != SASS::IADD3COrr &&
      Opc != SASS::IADD3Xrr && Opc != SASS::PRMTrrr && Opc != SASS::PRMTrir)
    return false;
  bool Changed = false;
  for (unsigned I = MI.getNumExplicitDefs(), E = MI.getNumExplicitOperands();
       I != E; ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || MO.isTied() || !MO.getReg().isVirtual())
      continue;
    std::optional<int64_t> C = constOf(MO);
    const TargetRegisterClass *RC = MI.getRegClassConstraint(I, TII, TRI);
    if (!C || *C != 0 || !RC || !RC->contains(SASS::RZ))
      continue;
    MO.setReg(SASS::RZ);
    MO.setIsKill(false);
    Changed = true;
  }
  return Changed;
}

/// LOP3.LUT a, b, c with zero / all-ones inputs and repeated registers.
bool SASSFoldLanes::simplifyLOP3(MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  if ((Opc != SASS::LOP3rrr && Opc != SASS::LOP3rir) || TII->isPredicated(MI))
    return false;
  // Inputs a, b, c: a register (index into Vars) or a constant 0 / ~0.
  SmallVector<Register, 3> Vars;
  int In[3]; // >= 0: variable; -1: zero; -2: all ones
  for (unsigned K = 0; K < 3; ++K) {
    const MachineOperand &MO = MI.getOperand(1 + K);
    std::optional<int64_t> C = constOf(MO);
    if (C && (uint32_t(*C) == 0 || uint32_t(*C) == ~0u)) {
      In[K] = uint32_t(*C) ? -2 : -1;
      continue;
    }
    if (!MO.isReg() || MO.getSubReg() || !MO.getReg().isVirtual())
      return false;
    auto It = find(Vars, MO.getReg());
    In[K] = It - Vars.begin();
    if (It == Vars.end())
      Vars.push_back(MO.getReg());
  }
  unsigned Lut = MI.getOperand(4).getImm() & 0xff;
  // Truth table over the distinct variables, in LOP3 order (var 0 = a).
  unsigned NewLut = 0;
  for (unsigned Row = 0; Row < 8; ++Row) {
    unsigned Bits[3] = {(Row >> 2) & 1, (Row >> 1) & 1, Row & 1};
    unsigned Idx = 0;
    for (unsigned K = 0; K < 3; ++K) {
      unsigned V = In[K] >= 0 ? Bits[In[K]] : In[K] == -2;
      Idx |= V << (2 - K);
    }
    NewLut |= ((Lut >> Idx) & 1) << Row;
  }
  const unsigned VarLut[3] = {0xf0, 0xcc, 0xaa};
  MachineBasicBlock &MBB = *MI.getParent();
  Register Dst = MI.getOperand(0).getReg();
  if (MI.getOperand(0).getSubReg())
    return false;
  if (NewLut == 0x00 || NewLut == 0xff) {
    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(SASS::MOVri), Dst)
        .addImm(NewLut ? -1 : 0)
        .addReg(SASS::PT)
        .addImm(0);
    MI.eraseFromParent();
    return true;
  }
  for (unsigned V = 0; V < Vars.size(); ++V)
    if (NewLut == VarLut[V] &&
        MRI->getRegClass(Dst)->hasSubClassEq(MRI->getRegClass(Vars[V]))) {
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), Dst)
          .addReg(Vars[V]);
      MI.eraseFromParent();
      return true;
    }
  // Fewer distinct registers or a known operand: rewrite with RZ for unused
  // inputs (a constant input is folded into the table).
  bool Known = any_of(In, [](int X) { return X < 0; });
  bool Repeat = Vars.size() < 3 && !Known;
  if (!Known && !Repeat)
    return false;
  if (Opc == SASS::LOP3rir && Vars.size() == 3)
    return false;
  Register Ops[3];
  for (unsigned K = 0; K < 3; ++K)
    Ops[K] = K < Vars.size() ? Vars[K] : Register(SASS::RZ);
  if (Opc == SASS::LOP3rrr && NewLut == Lut &&
      all_of(seq(3u), [&](unsigned K) {
        return MI.getOperand(1 + K).getReg() == Ops[K];
      }))
    return false;
  MachineInstrBuilder B =
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(SASS::LOP3rrr), Dst);
  for (Register R : Ops)
    B.addReg(R);
  B.addImm(NewLut).addReg(SASS::PT).addImm(0);
  MI.eraseFromParent();
  return true;
}

static cl::opt<bool> FoldLanes("sass-fold-lanes", cl::init(true), cl::Hidden,
                               cl::desc("Fold REG_SEQUENCE lanes, zero "
                                        "operands and LOP3 tables (sm_90)"));

bool SASSFoldLanes::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (!FoldLanes || ST.hasBlackwellISA())
    return false;
  MRI = &MF.getRegInfo();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  bool Changed = false;
  for (bool Again = true; Again;) {
    Again = false;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB)) {
        bool C = forward(MI);
        C |= zeroToRZ(MI);
        C |= simplifyLOP3(MI);
        Again |= C;
      }
    Changed |= Again;
  }
  // Delete what lost its last use (side-effect-free single defs).
  for (bool Again = Changed; Again;) {
    Again = false;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(reverse(MBB))) {
        if (MI.getNumDefs() == 0 || MI.isInlineAsm() || MI.mayStore() ||
            MI.hasUnmodeledSideEffects() || MI.isTerminator())
          continue;
        if (all_of(MI.defs(), [&](const MachineOperand &MO) {
              return MO.getReg().isVirtual() && MRI->use_nodbg_empty(MO.getReg());
            }) && (MI.isCopy() || MI.isRegSequence() || MI.isPHI() ||
                   MI.isImplicitDef() || MI.getOpcode() == SASS::MOVri ||
                   SASSInstrInfo::isCheapALU(MI) ||
                   MI.getOpcode() == SASS::LOP3rrr)) {
          MI.eraseFromParent();
          Again = true;
        }
      }
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// SASSNarrowRegs
//
// After register coalescing, a GPR pair whose every definition and use names
// the same one 32-bit lane (`undef %p.sub0 = ...`, `... %p.sub0 ...`) is left
// over where a zero extension or 64-bit operation fed only one half (the
// coalescer joins the 32-bit value into the pair's lane; the other half is
// never written or read). Such a vreg still allocates as an aligned pair:
// it competes only for even (or odd) registers, spills and reloads 64 bits,
// and a reload writes both registers of the pair. Under sub-register liveness
// (Hopper) each one becomes a 32-bit vreg of the lane's class before the
// allocator runs, as AMDGPU's GCNRewritePartialRegUses does.
//===----------------------------------------------------------------------===//

namespace {
class SASSNarrowRegs : public MachineFunctionPass {
public:
  static char ID;
  SASSNarrowRegs() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "SASS narrow single-lane registers";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // namespace

char SASSNarrowRegs::ID = 0;
char &llvm::SASSNarrowRegsID = SASSNarrowRegs::ID;

INITIALIZE_PASS(SASSNarrowRegs, "sass-narrow-regs",
                "SASS narrow single-lane registers", false, false)

FunctionPass *llvm::createSASSNarrowRegsPass() { return new SASSNarrowRegs(); }

bool SASSNarrowRegs::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (!ST.enableSubRegLiveness())
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  const TargetInstrInfo *TII = ST.getInstrInfo();
  auto *LISW = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
  LiveIntervals *LIS = LISW ? &LISW->getLIS() : nullptr;
  bool Changed = false;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(R) ||
        !SASS::GPR64RegClass.hasSubClassEq(MRI.getRegClass(R)))
      continue;
    unsigned Sub = 0;
    bool OneLane = true;
    for (const MachineOperand &MO : MRI.reg_nodbg_operands(R)) {
      if (!MO.getSubReg() || (Sub && MO.getSubReg() != Sub)) {
        OneLane = false;
        break;
      }
      Sub = MO.getSubReg();
    }
    const TargetRegisterClass *NewRC =
        OneLane ? TRI->getSubRegisterClass(MRI.getRegClass(R), Sub) : nullptr;
    // Every instruction must accept a plain 32-bit register there.
    for (const MachineOperand &MO : MRI.reg_nodbg_operands(R)) {
      if (!NewRC)
        break;
      const MachineInstr &MI = *MO.getParent();
      if (const TargetRegisterClass *OpRC =
              MI.getRegClassConstraint(MI.getOperandNo(&MO), TII, TRI))
        NewRC = TRI->getCommonSubClass(NewRC, OpRC);
    }
    if (!NewRC)
      continue;
    Register N = MRI.createVirtualRegister(NewRC);
    for (MachineOperand &MO : make_early_inc_range(MRI.reg_operands(R))) {
      MO.setReg(N);
      if (MO.isDebug() && !MO.getSubReg())
        continue;
      MO.setSubReg(0);
      if (MO.isDef())
        MO.setIsUndef(false);
    }
    if (LIS && LIS->hasInterval(R)) {
      LIS->removeInterval(R);
      LIS->createAndComputeVirtRegInterval(N);
    }
    Changed = true;
  }
  return Changed;
}
