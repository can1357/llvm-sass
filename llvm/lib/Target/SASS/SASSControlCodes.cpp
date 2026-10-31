//===-- SASSControlCodes.cpp - Stalls, yields and scoreboards -------------===//
//
// SASS has no hardware interlocks for fixed-latency pipelines and six
// counting scoreboards (SB0-SB5) for variable-latency ones. This pass assigns
// every instruction's control word:
//
// * Fixed latency: a consumer may issue only `lat(producer, consumer)` cycles
//   after its producer (ptxas's own sm_90 or sm_120 tables,
//   SASSLatencyTables.inc).
//   Missing cycles are added to the previous instruction's stall (stall N =
//   the next instruction issues N cycles later). Each block drains pending
//   fixed results before control leaves it, so blocks start clean.
// * Variable latency: a decoupled producer increments a write scoreboard at
//   issue; readers and re-writers of its result wait on it (`req`). Decoupled
//   instructions that read sources late increment SB0 until the read is done,
//   so later writers of those sources wait on SB0 (WAR). Write scoreboards
//   are SB1..SB5: a producer takes one with no pending claims, else the one
//   set longest ago. Sharing is always correct, it only over-waits. A
//   scoreboard takes effect two cycles after issue, so waiters are kept at
//   least two cycles away.
// * Pending scoreboard claims flow across blocks (union over predecessors,
//   iterated to a fixpoint).
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "sass-control-codes"

namespace {

#include "SASSLatencyTables.inc"

enum class RegKind { GPR, Pred, PredGuard, UGPR };

constexpr unsigned MaxStall = 15;
// usched values 28-31 (stall >= 12 without yield) are not encodable.
constexpr unsigned MaxTransStall = 11;
// Shortest stall worth a warp switch (which costs one issue cycle).
constexpr unsigned YieldStall = 3;
constexpr unsigned ReadSB = 0;
constexpr unsigned FirstWriteSB = 1, LastWriteSB = 5;
constexpr int SBSetDelay = 2;
constexpr int Never = -1000;

/// 32-bit integer-pipe ALU ops.
bool isIntPipe(unsigned Mnem) {
  return Mnem == LAT_LOP3 || Mnem == LAT_SHF || Mnem == LAT_IADD3 ||
         Mnem == LAT_LEA;
}
/// Integer ops on the FMA pipe (IMAD) or with its timing (two-input IADD).
bool isIntFma(unsigned Mnem) { return Mnem == LAT_IMAD || Mnem == LAT_IADD; }

unsigned tableLat(const LatencyTables &L, unsigned Prod, unsigned Cons,
                  RegKind K) {
  unsigned V = 0;
  switch (K) {
  case RegKind::GPR: V = L.TrueGPR[Prod][Cons]; break;
  case RegKind::Pred: V = L.TruePred[Prod][Cons]; break;
  case RegKind::PredGuard: V = L.TruePredGuard[Prod][Cons]; break;
  case RegKind::UGPR: V = L.TrueUGPR[Prod][Cons]; break;
  }
  // No table entry for a fixed-latency pair: assume the common int latency.
  return V ? V : 6;
}

/// Worst-case latency of a fixed result over the consumer mnemonics in
/// \p Consumers (a bit per LatMnemonic present in the function), used when
/// the consumer is in another block.
unsigned drainLat(const LatencyTables &L, uint64_t Consumers, unsigned Prod,
                  RegKind K) {
  static_assert(NumLatMnem <= 64);
  unsigned V = 6;
  for (unsigned C = 0; C < NumLatMnem; ++C) {
    if (!(Consumers >> C & 1))
      continue;
    if (K == RegKind::Pred) {
      V = std::max<unsigned>(V, L.TruePred[Prod][C]);
      V = std::max<unsigned>(V, L.TruePredGuard[Prod][C]);
    } else {
      V = std::max(V, tableLat(L, Prod, C, K));
    }
  }
  return V;
}

/// Typical issue-to-result cycles of a scoreboarded producer.
int resultLat(unsigned Mnem) {
  switch (Mnem) {
  case LAT_LDG: case LAT_LD: return 400;
  case LAT_LDS: case LAT_SHFL: return 30;
  case LAT_S2R: case LAT_S2UR: return 25;
  case LAT_LDC: case LAT_LDCU: return 15;
  default: return 20; // MUFU, F2I, ...
  }
}

/// Pending scoreboard claims per register unit.
struct SBState {
  SmallVector<uint8_t, 0> Wr, Rd;
  explicit SBState(unsigned Units = 0) : Wr(Units, 0), Rd(Units, 0) {}
  bool join(const SBState &O) {
    bool Changed = false;
    for (unsigned U = 0, E = Wr.size(); U != E; ++U) {
      uint8_t W = Wr[U] | O.Wr[U], R = Rd[U] | O.Rd[U];
      Changed |= W != Wr[U] || R != Rd[U];
      Wr[U] = W;
      Rd[U] = R;
    }
    return Changed;
  }
};

/// Last fixed-latency producer of a register unit in the current block.
struct FixedDef {
  int Issue = Never;
  unsigned Mnem = 0;
  RegKind K = RegKind::GPR;
  bool Wide = false; // 64-bit result (IMAD.WIDE, IADD.64): never tightened.
};

/// Issue-to-use cycles from \p F to a consumer \p Cons reading as \p K.
///
/// The dumped tables give 6 cycles between dependent integer instructions,
/// but ptxas itself schedules int-pipe -> int-pipe 4 cycles apart and
/// int-pipe <-> IMAD/IADD 5 apart (lifted nvcc code for kernels/chains.cu).
/// Those observed distances replace the table for exactly those pairs:
/// nothing shorter than ptxas emits (a blanket 4, IMAD included, gave wrong
/// fmix/xorshift results on GB10), and IMAD -> IMAD, 64-bit results and
/// cross-block uses (drainLat) keep the table.
unsigned useLat(const LatencyTables &L, const FixedDef &F, unsigned Cons,
                RegKind K) {
  unsigned V = tableLat(L, F.Mnem, Cons, K);
  if (K != RegKind::GPR || F.Wide)
    return V;
  bool PI = isIntPipe(F.Mnem), CI = isIntPipe(Cons);
  if (PI && CI)
    return std::min(V, 4u);
  if ((PI && isIntFma(Cons)) || (isIntFma(F.Mnem) && CI))
    return std::min(V, 5u);
  return V;
}

class SASSControlCodes : public MachineFunctionPass {
public:
  static char ID;
  SASSControlCodes() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS control codes"; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const TargetRegisterInfo *TRI = nullptr;
  const SASSInstrInfo *TII = nullptr;
  /// Fixed latencies of the function's architecture.
  const LatencyTables *Lat = nullptr;
  /// LatMnemonic bits of the instructions in the function.
  uint64_t Consumers = 0;
  /// The function has tensor-core instructions, whose results can take
  /// longer than one stall field: blocks drain before their terminators.
  bool DrainBeforeTerminators = false;
  /// Highest write scoreboard for results: below SASS::AsyncCopySB when the
  /// function commits async copies to it.
  unsigned LastResultSB = LastWriteSB;

  SBState processBlock(MachineBasicBlock &MBB, SBState S,
                       SASSFunctionInfo *Commit);
  RegKind kindOf(Register R, bool IsGuard) const;
  SmallVector<unsigned, 4> units(Register R) const {
    SmallVector<unsigned, 4> Us;
    for (MCRegUnit U : TRI->regunits(R))
      Us.push_back(static_cast<unsigned>(U));
    return Us;
  }
};

} // namespace

char SASSControlCodes::ID = 0;

INITIALIZE_PASS(SASSControlCodes, DEBUG_TYPE, "SASS control codes", false,
                false)

FunctionPass *llvm::createSASSControlCodesPass() {
  return new SASSControlCodes();
}

static bool isConstReg(Register R) {
  return R == SASS::RZ || R == SASS::PT || R == SASS::URZ;
}

/// Instructions whose explicit sources map 1:1 onto SASS slots a, b, c and
/// whose encodings accept reuse bits.
static bool reuseCapable(unsigned Opc) {
  switch (Opc) {
  case SASS::FFMArrr: case SASS::FFMArir: case SASS::FFMArri:
  case SASS::FFMArrn: case SASS::FFMAnrr: case SASS::FMULrr:
  case SASS::FMULri: case SASS::FADDrr: case SASS::FADDri:
  case SASS::FSUBrr: case SASS::IMADrrr: case SASS::IMADrir:
  case SASS::LOP3rrr: case SASS::LOP3rir: case SASS::IADD3rrr:
    return true;
  default:
    return false;
  }
}

// The operand reuse cache keeps a source for exactly the next instruction,
// per slot. Never on a source the instruction also writes (the successor
// would read the stale value), and only with the trans (yield 0) usched form
// and stalls <= 11.
template <typename SlotVec> static void setReuseBits(SlotVec &Out) {
  for (size_t I = 0; I + 1 < Out.size(); ++I) {
    MachineInstr *A = Out[I].MI, *B = Out[I + 1].MI;
    if (!A || !B || !reuseCapable(A->getOpcode()) ||
        !reuseCapable(B->getOpcode()) || Out[I].C.Stall > MaxTransStall)
      continue;
    unsigned NA = A->getDesc().getNumDefs(), NB = B->getDesc().getNumDefs();
    unsigned Bits = 0;
    for (unsigned S = 0; S < 3; ++S) {
      if (NA + S >= A->getNumExplicitOperands() || NB + S >= B->getNumExplicitOperands())
        break;
      const MachineOperand &OA = A->getOperand(NA + S), &OB = B->getOperand(NB + S);
      if (!OA.isReg() || !OB.isReg() || OA.getReg() != OB.getReg() ||
          isConstReg(OA.getReg()) || A->modifiesRegister(OA.getReg(), nullptr) ||
          A->findFirstPredOperandIdx() == int(NA + S))
        continue;
      Bits |= 1u << S;
    }
    if (Bits)
      Out[I].C.Reuse = Bits;
  }
}

RegKind SASSControlCodes::kindOf(Register R, bool IsGuard) const {
  if (SASS::PREDRegClass.contains(R))
    return IsGuard ? RegKind::PredGuard : RegKind::Pred;
  if (SASS::UGPR32RegClass.contains(R) || SASS::UGPR64RegClass.contains(R) ||
      SASS::UGPR128RegClass.contains(R))
    return RegKind::UGPR;
  return RegKind::GPR;
}

SBState SASSControlCodes::processBlock(MachineBasicBlock &MBB, SBState S,
                                       SASSFunctionInfo *Commit) {
  unsigned NumUnits = TRI->getNumRegUnits();
  SmallVector<FixedDef, 0> Fixed(NumUnits);
  // Last read of each register unit by a tensor-core instruction, which
  // collects its sources over several cycles: a later writer of the unit
  // waits AntiGPR[reader][writer] cycles after it issued.
  struct SlowRead {
    int Issue = Never;
    unsigned Mnem = 0;
  };
  SmallVector<SlowRead, 0> Slow(NumUnits);
  auto MaxAnti = [&](unsigned Reader) {
    unsigned V = 1;
    for (unsigned W = 0; W < NumLatMnem; ++W)
      V = std::max<unsigned>(V, Lat->AntiGPR[Reader][W]);
    return V;
  };
  int SBSet[6] = {Never, Never, Never, Never, Never, Never};
  // wgmma and setmaxnreg (ptxas's GMMA_SCOREBOARD, GMMA_GROUP_SCOREBOARD
  // and RPC tables): GMMA issues 6 cycles after WARPGROUP.ARRIVE; DEPBAR,
  // EXIT and the next gsb0 GMMA 3 after a gsb0 GMMA; USETMAXREG 9 after the
  // previous instruction. NextReq: scoreboard the next instruction waits on.
  int LastArrive = Never, LastGsb = Never, PrevIssue = Never;
  unsigned NextReq = 0;
  auto IsGMMA = [](unsigned Opc) {
    return Opc == SASS::GMMA_SS || Opc == SASS::GMMA_RS ||
           Opc == SASS::GMMA_SS_Z || Opc == SASS::GMMA_RS_Z;
  };
  auto HasGsb = [&](const MachineInstr &MI) {
    return IsGMMA(MI.getOpcode()) &&
           MI.getOperand(MI.getDesc().getNumOperands() - 3).getImm();
  };
  int T = 0; // Issue cycle of the next instruction.
  // Expected latency of the slowest result pending on each scoreboard.
  int SBLat[6] = {0, 0, 0, 0, 0, 0};
  // A free write scoreboard if any, else the one set longest ago. Sharing it
  // makes every reader of its pending results also wait for the new one
  // (ncu: FFMAs stalled on shared loads issued 98 cycles earlier), so when
  // those results should have landed by now (short-latency producers set
  // long enough ago), the new producer waits for them first and takes the
  // scoreboard over, as ptxas does. Long-latency (global) results are shared.
  auto PickWriteSB = [&](bool &WaitFirst) {
    uint8_t Pending = 0;
    for (unsigned U = 0; U < NumUnits; ++U)
      Pending |= S.Wr[U];
    unsigned Best = FirstWriteSB;
    for (unsigned SB = FirstWriteSB; SB <= LastResultSB; ++SB) {
      if (!(Pending & (1u << SB))) {
        SBLat[SB] = 0;
        return SB;
      }
      if (SBSet[SB] < SBSet[Best])
        Best = SB;
    }
    WaitFirst = T - SBSet[Best] >= SBLat[Best];
    return Best;
  };

  struct Slot {
    MachineInstr *MI;
    SASSCtrl C;
  };
  SmallVector<Slot, 32> Out;

  // Delays the instruction about to issue by N cycles: extends the previous
  // instruction's stall, spilling into NOPs past the 4-bit stall field.
  auto Delay = [&](unsigned N, MachineInstr *Before) {
    assert(!Out.empty() && "blocks start with no pending fixed results");
    SASSCtrl &P = Out.back().C;
    unsigned Total = P.Stall + N;
    P.Stall = std::min(Total, MaxStall);
    for (Total -= P.Stall; Total; Total -= Out.back().C.Stall) {
      // Padding at the block end only runs on fallthrough.
      if (!Before && MBB.getFirstTerminator() != MBB.end())
        report_fatal_error("SASS: block-exit drain exceeds the stall field");
      MachineInstr *Nop = nullptr;
      if (Commit)
        Nop = BuildMI(MBB, Before ? Before->getIterator() : MBB.end(),
                      DebugLoc(), TII->get(SASS::NOP))
                  .addReg(SASS::PT)
                  .addImm(0);
      SASSCtrl NC;
      NC.Stall = std::min(Total, MaxStall);
      Out.push_back({Nop, NC});
    }
  };

  // Cycle by which every fixed result is usable, every scoreboard set and
  // every slow read done: blocks are left in that state.
  auto DrainReady = [&] {
    int Ready = T;
    for (const SlowRead &R : Slow)
      if (R.Issue != Never)
        Ready = std::max<int>(Ready, R.Issue + MaxAnti(R.Mnem));
    for (const FixedDef &F : Fixed)
      if (F.Issue != Never)
        Ready = std::max<int>(Ready, F.Issue + drainLat(*Lat, Consumers, F.Mnem, F.K));
    for (int Set : SBSet)
      Ready = std::max(Ready, Set + SBSetDelay);
    Ready = std::max({Ready, LastArrive + 6, LastGsb + 3});
    return Ready;
  };

  bool Drained = false;
  for (MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction())
      continue;
    // Tensor-core results take longer than a stall field: wait before the
    // branch, padding with NOPs, rather than in its stall.
    if (DrainBeforeTerminators && MI.isTerminator() && !Drained) {
      Drained = true;
      int Ready = DrainReady();
      if (Ready > T && !Out.empty()) {
        Delay(Ready - T, &MI);
        T = Ready;
      }
    }
    const MCInstrDesc &D = MI.getDesc();
    unsigned M = SASS::latMnem(D);
    int Guard = MI.findFirstPredOperandIdx();

    unsigned Req = 0;
    int Need = T;
    for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
      const MachineOperand &MO = MI.getOperand(I);
      if (!MO.isReg() || !MO.getReg() || isConstReg(MO.getReg()))
        continue;
      RegKind K = kindOf(MO.getReg(), (int)I == Guard);
      for (unsigned U : units(MO.getReg())) {
        Req |= S.Wr[U];
        if (MO.isDef())
          Req |= S.Rd[U]; // WAR against a late reader.
        if (MO.isDef() && Slow[U].Issue != Never)
          Need = std::max<int>(Need, Slow[U].Issue +
                                         Lat->AntiGPR[Slow[U].Mnem][M]);
        const FixedDef &F = Fixed[U];
        if (F.Issue != Never)
          Need = std::max<int>(Need, F.Issue + useLat(*Lat, F, M, K));
      }
    }
    unsigned Opc = MI.getOpcode();
    if (IsGMMA(Opc))
      Need = std::max(Need, LastArrive + 6);
    if (HasGsb(MI) || Opc == SASS::WARPGROUP_DEPBAR || MI.isReturn())
      Need = std::max(Need, LastGsb + 3);
    if (Opc == SASS::WARPGROUP_ARRIVE) {
      // Registers a GMMA reads must be written back before the arrive
      // (TABLE_TRUE(GPR) column GMMA_SB): every pending result.
      for (unsigned U = 0; U < NumUnits; ++U) {
        Req |= S.Wr[U];
        const FixedDef &F = Fixed[U];
        if (F.Issue == Never)
          continue;
        unsigned L = 8;
        for (unsigned Cn = 0; Cn < NumLatMnem; ++Cn)
          L = std::max<unsigned>(L, Lat->TrueGPR[F.Mnem][Cn] + 2);
        Need = std::max<int>(Need, F.Issue + L);
      }
    }
    if (Opc == SASS::USETMAXREG_DEALLOC || Opc == SASS::USETMAXREG_TRY_ALLOC) {
      Need = std::max(Need, PrevIssue + 9);
      // Freed registers must have no results in flight.
      if (Opc == SASS::USETMAXREG_DEALLOC)
        for (unsigned U = 0; U < NumUnits; ++U)
          Req |= S.Wr[U] | S.Rd[U];
    }
    Req |= NextReq;
    NextReq = 0;
    for (unsigned SB = 0; SB < 6; ++SB)
      if (Req & (1u << SB))
        Need = std::max(Need, SBSet[SB] + SBSetDelay);
    if (Need > T) {
      Delay(Need - T, &MI);
      T = Need;
    }
    if (Req)
      for (unsigned U = 0; U < NumUnits; ++U) {
        S.Wr[U] &= ~Req;
        S.Rd[U] &= ~Req;
      }

    SASSCtrl C;
    C.Req = Req;
    bool HasDefs = false, HasLateSrc = false;
    for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
      const MachineOperand &MO = MI.getOperand(I);
      if (MO.isReg() && MO.getReg() && !isConstReg(MO.getReg())) {
        HasDefs |= MO.isDef();
        HasLateSrc |= MO.isUse() && (int)I != Guard;
      }
    }
    if (SASS::writesSB(D) && HasDefs) {
      bool WaitFirst = false;
      C.Wr = PickWriteSB(WaitFirst);
      if (WaitFirst) {
        C.Req |= 1u << C.Wr;
        for (unsigned U = 0; U < NumUnits; ++U) {
          S.Wr[U] &= ~(1u << C.Wr);
          S.Rd[U] &= ~(1u << C.Wr);
        }
        SBLat[C.Wr] = 0;
      }
      SBSet[C.Wr] = T;
      SBLat[C.Wr] = std::max(SBLat[C.Wr], resultLat(M));
    }
    if (MI.getOpcode() == SASS::LDGDEPBAR) {
      C.Wr = SASS::AsyncCopySB;
      SBSet[C.Wr] = T;
    }
    // USETMAXREG.DEALLOC completes on a scoreboard the next instruction
    // waits on (as ptxas emits it).
    if (MI.getOpcode() == SASS::USETMAXREG_DEALLOC) {
      bool WaitFirst = false;
      C.Wr = PickWriteSB(WaitFirst);
      if (WaitFirst)
        C.Req |= 1u << C.Wr;
      SBSet[C.Wr] = T;
      SBLat[C.Wr] = resultLat(M);
      NextReq = 1u << C.Wr;
    }
    // A bulk-copy commit counts on SB0; DEPBAR.LE SB0, N waits for it
    // (cp.async.bulk.wait_group).
    if (MI.getOpcode() == SASS::UTMACMDFLUSH) {
      C.Rd = ReadSB;
      SBSet[ReadSB] = T;
    }
    if (SASS::readsLate(D) && HasLateSrc) {
      C.Rd = ReadSB;
      SBSet[ReadSB] = T;
    }
    for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
      const MachineOperand &MO = MI.getOperand(I);
      if (!MO.isReg() || !MO.getReg() || isConstReg(MO.getReg()))
        continue;
      RegKind K = kindOf(MO.getReg(), false);
      bool Wide = SASS::GPR64RegClass.contains(MO.getReg()) ||
                  SASS::GPR128RegClass.contains(MO.getReg());
      // GMMA results land asynchronously, behind WARPGROUP.DEPBAR.
      if (IsGMMA(Opc) && MO.isDef()) {
        for (unsigned U : units(MO.getReg()))
          Fixed[U] = FixedDef();
        continue;
      }
      for (unsigned U : units(MO.getReg())) {
        if (MO.isDef()) {
          Slow[U] = SlowRead();
          if (C.Wr != 7) {
            S.Wr[U] = 1u << C.Wr;
            Fixed[U] = FixedDef();
          } else {
            Fixed[U] = {T, M, K, Wide};
          }
        } else if (C.Rd != 7 && (int)I != Guard) {
          S.Rd[U] |= 1u << ReadSB;
        } else if (M == LAT_HMMA && (int)I != Guard) {
          Slow[U] = {T, M};
        }
      }
    }
    if (MI.getOpcode() == SASS::WARPGROUP_ARRIVE)
      LastArrive = T;
    if (HasGsb(MI))
      LastGsb = T;
    PrevIssue = T;
    C.Stall = std::max(1u, SASS::minStall(D));
    Out.push_back({&MI, C});
    T += C.Stall;
  }

  int Ready = DrainReady();
  if (Ready > T && !Out.empty())
    Delay(Ready - T, nullptr);

  if (Commit) {
    // A yield switches warps at the cost of one dead issue cycle (sm120
    // measurements), so like ptxas only yield where the warp stalls for 3+
    // cycles anyway; never with reuse bits, always past the trans form.
    setReuseBits(Out);
    for (Slot &Sl : Out) {
      Sl.C.Yield = Sl.C.Stall > MaxTransStall ||
                   (Sl.C.Stall >= YieldStall && !Sl.C.Reuse);
      Commit->Ctrl[Sl.MI] = Sl.C;
    }
  }
  return S;
}

bool SASSControlCodes::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  Lat = ST.hasBlackwellISA() ? &SM120Latencies : &SM90Latencies;
  Consumers = 0;
  for (const MachineBasicBlock &B : MF)
    for (const MachineInstr &MI : B)
      if (!MI.isMetaInstruction())
        Consumers |= uint64_t(1) << SASS::latMnem(MI.getDesc());
  DrainBeforeTerminators = Consumers >> LAT_HMMA & 1;
  unsigned NumUnits = TRI->getNumRegUnits();
  static_assert(SASS::AsyncCopySB == LastWriteSB);
  LastResultSB = LastWriteSB;
  for (MachineBasicBlock &B : MF)
    for (MachineInstr &MI : B)
      if (MI.getOpcode() == SASS::LDGDEPBAR)
        LastResultSB = LastWriteSB - 1;

  // Fixpoint over pending scoreboard claims; outputs only grow, so this
  // terminates and over-approximates what may be in flight.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  DenseMap<const MachineBasicBlock *, SBState> In, Out;
  for (MachineBasicBlock *B : RPOT) {
    In.try_emplace(B, NumUnits);
    Out.try_emplace(B, NumUnits);
  }
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (MachineBasicBlock *B : RPOT) {
      SBState Entry(NumUnits);
      for (MachineBasicBlock *P : B->predecessors())
        if (auto It = Out.find(P); It != Out.end())
          Entry.join(It->second);
      In.find(B)->second = Entry;
      Changed |= Out.find(B)->second.join(processBlock(*B, Entry, nullptr));
    }
  }

  auto *FI = MF.getInfo<SASSFunctionInfo>();
  FI->Ctrl.clear();
  for (MachineBasicBlock &B : MF)
    processBlock(B, In.count(&B) ? In.find(&B)->second : SBState(NumUnits),
                 FI);
  return true;
}
