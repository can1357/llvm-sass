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
//   Over the limit, instructions that free registers win.
//
// The pressure model counts GPR units of SSA values with exact block
// liveness. A COPY between virtual registers (one lane of a 128-bit load)
// is coalesced into its source, which without sub-register liveness stays
// live until its last lane is read, so copies are aliases of their root.
//
// LLVM's GenericScheduler is disabled for SASS (SASSSubtarget): on unrolled
// tiles it hoisted every 128-bit load while under-reporting pressure.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSRegisterInfo.h"
#include "SASSSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
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
constexpr unsigned LSUQueueDepth = 4;

namespace {

#include "SASSLatencyTables.inc"

struct Node {
  MachineInstr *MI;
  SmallVector<std::pair<unsigned, unsigned>, 4> Succs; // (node, latency)
  unsigned NumPreds = 0;
  unsigned Height = 0;
  int Earliest = 0;
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
  const MachineRegisterInfo *MRI = nullptr;
  /// Fixed latencies of the function's architecture.
  const LatencyTables *Lat = nullptr;
  DenseMap<const MachineBasicBlock *, RegSet> LiveIn, LiveOut;

  Register root(Register R) const;
  unsigned width(Register R) const;
  void computeLiveness(MachineFunction &MF);
  Region buildRegion(MachineBasicBlock &MBB) const;
  int delta(const Region &Rg, const MachineInstr &MI,
            DenseMap<Register, unsigned> &Left, bool Commit) const;
  bool schedule(Region &Rg, int Limit) const;
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
  if (isPseudo(C))
    return 0;
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
        if (width(MI.getOperand(0).getReg()))
          K.insert(MI.getOperand(0).getReg());
        for (unsigned I = 1; I + 1 < MI.getNumOperands(); I += 2) {
          Register R = root(MI.getOperand(I).getReg());
          if (width(R))
            PhiUses[MI.getOperand(I + 1).getMBB()].insert(R);
        }
        continue;
      }
      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && MO.getReg().isVirtual()) {
          Register R = root(MO.getReg());
          if (width(R) && !K.count(R))
            G.insert(R);
        }
      for (const MachineOperand &MO : MI.defs())
        if (MO.getReg().isVirtual() && root(MO.getReg()) == MO.getReg() &&
            width(MO.getReg()))
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

/// Pressure change from issuing \p MI given the remaining uses \p Left;
/// \p Commit consumes its uses.
int SASSPreRASched::delta(const Region &Rg, const MachineInstr &MI,
                          DenseMap<Register, unsigned> &Left,
                          bool Commit) const {
  int D = 0;
  for (const MachineOperand &MO : MI.defs()) {
    Register R = MO.getReg();
    if (root(R) == R && (Left.count(R) || Rg.Out.count(R)))
      D += width(R);
  }
  SmallDenseMap<Register, unsigned, 8> Seen;
  for (const MachineOperand &MO : MI.uses()) {
    Register R = MO.isReg() ? root(MO.getReg()) : Register();
    if (!width(R))
      continue;
    unsigned Now = Left.lookup(R) - Seen[R]++;
    if (Now == 1 && !Rg.Out.count(R))
      D -= width(R);
  }
  if (Commit)
    for (auto [R, K] : Seen)
      Left[R] -= K;
  return D;
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

  // Pressure model: live roots at the region start, region uses, and roots
  // still needed afterwards (live out, or read by a terminator).
  Rg.Out = LiveOut.lookup(&MBB);
  for (MachineInstr &T : make_range(End, MBB.end()))
    for (const MachineOperand &MO : T.uses())
      if (MO.isReg() && width(root(MO.getReg())))
        Rg.Out.insert(root(MO.getReg()));
  for (Node &Nd : Nodes)
    for (const MachineOperand &MO : Nd.MI->uses())
      if (MO.isReg() && width(root(MO.getReg())))
        ++Rg.UsesLeft[root(MO.getReg())];
  RegSet StartSet = LiveIn.lookup(&MBB);
  for (MachineInstr &Phi : MBB.phis()) {
    Register R = Phi.getOperand(0).getReg();
    if (Rg.UsesLeft.count(R) || Rg.Out.count(R))
      StartSet.insert(R);
  }
  for (Register R : StartSet)
    Rg.Start += width(R);
  int P = Rg.Peak = Rg.Start;
  DenseMap<Register, unsigned> Left = Rg.UsesLeft;
  for (Node &Nd : Nodes) {
    P += delta(Rg, *Nd.MI, Left, /*Commit=*/true);
    Rg.Peak = std::max(Rg.Peak, P);
  }
  return Rg;
}

bool SASSPreRASched::schedule(Region &Rg, int Limit) const {
  SmallVector<Node, 0> &Nodes = Rg.Nodes;
  unsigned N = Nodes.size();
  SmallVector<unsigned, 64> Avail;
  for (unsigned I = 0; I < N; ++I)
    if (!Nodes[I].NumPreds)
      Avail.push_back(I);
  SmallVector<unsigned, 256> Order;
  int Cycle = 0, P = Rg.Start, SchedPeak = Rg.Start;
  DenseMap<Register, unsigned> Left = Rg.UsesLeft;
  // Dispatch cycles of the last LSUQueueDepth LSU instructions, oldest at
  // QHead: the next one enters the queue once the oldest has left it.
  bool ModelLSU = LSUInterval != 0;
  int Dispatch[LSUQueueDepth];
  std::fill(std::begin(Dispatch), std::end(Dispatch), INT_MIN / 2);
  unsigned QHead = 0;
  int LastDispatch = INT_MIN / 2;
  auto ReadyAt = [&](unsigned C) {
    int R = Nodes[C].Earliest;
    if (ModelLSU && usesLSU(*Nodes[C].MI))
      R = std::max(R, Dispatch[QHead]);
    return R;
  };
  // The pressure model counts GPRs only, and predicates cannot be spilled:
  // a 64-bit add's carry-in half follows its carry-out half at once, so at
  // most one carry is live (as in ptxas's code).
  unsigned CarryNext = ~0u;
  while (!Avail.empty()) {
    // Fits the limit, else frees the most; then ready; then height; then
    // original order.
    using Key = std::tuple<bool, int, bool, unsigned, int>;
    unsigned BestPos = 0;
    Key Best;
    auto Forced = find(Avail, CarryNext);
    for (unsigned K = 0; K < Avail.size() && Forced == Avail.end(); ++K) {
      unsigned C = Avail[K];
      int D = delta(Rg, *Nodes[C].MI, Left, /*Commit=*/false);
      bool Fits = P + D <= Limit;
      Key Cand = {Fits, Fits ? 0 : -D, ReadyAt(C) <= Cycle, Nodes[C].Height,
                  -int(C)};
      if (K == 0 || Cand > Best) {
        Best = Cand;
        BestPos = K;
      }
    }
    if (Forced != Avail.end())
      BestPos = Forced - Avail.begin();
    unsigned C = Avail[BestPos];
    Avail.erase(Avail.begin() + BestPos);
    Node &Nd = Nodes[C];
    CarryNext = ~0u;
    if (isCarryOut(*Nd.MI))
      for (auto [S, Lat] : Nd.Succs)
        if (isCarryIn(*Nodes[S].MI))
          CarryNext = S;
    LLVM_DEBUG(dbgs() << "  @" << std::max(Cycle, ReadyAt(C)) << " h"
                      << Nd.Height << " e" << Nd.Earliest << " P" << P << " "
                      << *Nd.MI);
    Cycle = std::max(Cycle, ReadyAt(C));
    P += delta(Rg, *Nd.MI, Left, /*Commit=*/true);
    SchedPeak = std::max(SchedPeak, P);
    Order.push_back(C);
    // A queued memory instruction's latency starts at its dispatch.
    int Start = Cycle;
    if (ModelLSU && usesLSU(*Nd.MI)) {
      Start = std::max(Cycle, LastDispatch + int(LSUInterval));
      LastDispatch = Dispatch[QHead] = Start;
      QHead = (QHead + 1) % LSUQueueDepth;
    }
    for (auto [S, Lat] : Nd.Succs) {
      Nodes[S].Earliest =
          std::max(Nodes[S].Earliest, (Lat ? Start : Cycle) + int(Lat));
      if (--Nodes[S].NumPreds == 0)
        Avail.push_back(S);
    }
    if (!isPseudo(*Nd.MI))
      Cycle += std::max(1u, SASS::minStall(Nd.MI->getDesc()));
  }
  assert(Order.size() == N && "dependence cycle in a block");

  MachineBasicBlock &MBB = *Rg.MBB;
  MachineBasicBlock::iterator End = MBB.getFirstTerminator();
  bool Changed = false;
  for (unsigned K = 0; K < N; ++K) {
    MachineInstr *MI = Nodes[Order[K]].MI;
    Changed |= Order[K] != K;
    MBB.splice(End, &MBB, MI);
    for (MachineOperand &MO : MI->uses())
      if (MO.isReg())
        MO.setIsKill(false);
  }
  LLVM_DEBUG(dbgs() << printMBBReference(MBB) << ": " << N
                    << " instrs, pressure " << Rg.Peak << " -> " << SchedPeak
                    << " (limit " << Limit << "), schedule length " << Cycle
                    << "\n");
  return Changed;
}

bool SASSPreRASched::runOnMachineFunction(MachineFunction &MF) {
  if (Disable)
    return false;
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  Lat = ST.hasBlackwellISA() ? &SM120Latencies : &SM90Latencies;
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
  int Limit = FuncPeak + int(ExtraRegs);
  unsigned MaxRegs = SASS::maxRegCount(MF.getFunction());
  if (MaxRegs < 255)
    Limit = std::min(Limit, int(MaxRegs) - 3 - int(RegSlack));
  bool Changed = false;
  for (Region &Rg : Regions)
    if (Rg.Nodes.size() > 2)
      Changed |= schedule(Rg, Limit);
  return Changed;
}
