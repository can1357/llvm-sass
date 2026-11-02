//===-- SASSPredSpills.cpp - Predicate spills to free GPRs -----------------===//
//
// Only P0-P6 hold predicates, so unrolled kernels spill them. In memory a
// spill is SEL + STL.U8 and a reload LDL.U8 + ISETP; ptxas keeps spilled
// predicates in GPRs instead. After register allocation, before rewriting,
// this pass moves each predicate spill slot into a GPR that no allocated
// value occupies while the slot is live:
//
//   SPILL_P  %stack.N, P   ->  SEL %v, RZ, 1, !P
//   RELOAD_P P, %stack.N   ->  ISETP.NE P, %v, 0
//
// %v gets the free GPR. Only registers within the setmaxnreg limit of every
// block %v is live in, and without setmaxnreg no higher than the allocation
// already uses, are taken, so the register count stays; when none is free the
// slot stays in memory (SASSRegisterInfo::eliminateFrameIndex). Giving the
// predicate allocator a GPR class to spill into instead costs more than it
// saves when GPRs are scarce: the new values push others into memory.
//
// A slot is converted when each reload has a spill of the slot dominating it,
// so %v is defined on every path to its uses.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSMachineFunctionInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "sass-pred-spills"

namespace {

class SASSPredSpills : public MachineFunctionPass {
public:
  static char ID;
  SASSPredSpills() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SASS predicate spills to GPRs";
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char SASSPredSpills::ID = 0;

INITIALIZE_PASS_BEGIN(SASSPredSpills, DEBUG_TYPE,
                      "SASS predicate spills to GPRs", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SASSPredSpills, DEBUG_TYPE, "SASS predicate spills to GPRs",
                    false, false)

FunctionPass *llvm::createSASSPredSpillsPass() { return new SASSPredSpills(); }

bool SASSPredSpills::runOnMachineFunction(MachineFunction &MF) {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasStackObjects())
    return false;

  // Spills and reloads per slot; slots with any other reference stay.
  struct Slot {
    SmallVector<MachineInstr *, 4> Spills, Reloads;
    bool Other = false;
  };
  MapVector<int, Slot> Slots;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isFI())
          continue;
        Slot &S = Slots[MO.getIndex()];
        if (MI.getOpcode() == SASS::SPILL_P && MO.getOperandNo() == 0)
          S.Spills.push_back(&MI);
        else if (MI.getOpcode() == SASS::RELOAD_P && MO.getOperandNo() == 1)
          S.Reloads.push_back(&MI);
        else
          S.Other = true;
      }
  if (none_of(Slots, [](const auto &KV) {
        return !KV.second.Other && !KV.second.Spills.empty();
      }))
    return false;

  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LiveIntervals &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  VirtRegMap &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LiveRegMatrix &Matrix = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  const auto &Limits = MF.getInfo<SASSFunctionInfo>()->RegLimit;
  RegisterClassInfo RCI;
  RCI.runOnMachineFunction(MF);
  ArrayRef<MCPhysReg> Order = RCI.getOrder(&SASS::GPR32RegClass);
  // Highest GPR the allocation uses: the register count, unless setmaxnreg
  // fixes that at the entry allocation.
  unsigned MaxEnc = ~0u;
  if (!SASS::setMaxRegEntryCount(MF)) {
    MaxEnc = 0;
    for (MCPhysReg R : Order)
      if (Matrix.isPhysRegUsed(R))
        MaxEnc = std::max<unsigned>(MaxEnc, TRI.getEncodingValue(R));
  }

  auto Dominates = [&](const MachineInstr *A, const MachineInstr *B) {
    if (A->getParent() != B->getParent())
      return MDT.dominates(A->getParent(), B->getParent());
    return LIS.getInstructionIndex(*A) < LIS.getInstructionIndex(*B);
  };
  auto Guard = [](MachineInstrBuilder MIB) {
    MIB.addReg(SASS::PT).addImm(0);
  };
  // Swaps \p Old for \p New at the same slot index; \p Old is kept detached.
  auto Swap = [&](MachineInstr &Old, MachineInstr &New) {
    LIS.ReplaceMachineInstrInMaps(Old, New);
    Old.removeFromParent();
  };

  bool Changed = false;
  for (auto &KV : Slots) {
    int FI = KV.first;
    Slot &S = KV.second;
    if (S.Other || S.Spills.empty() ||
        !all_of(S.Reloads, [&](const MachineInstr *R) {
          return any_of(S.Spills,
                        [&](const MachineInstr *W) { return Dominates(W, R); });
        }))
      continue;
    if (S.Reloads.empty()) {
      for (MachineInstr *MI : S.Spills) {
        LIS.RemoveMachineInstrFromMaps(*MI);
        MI->eraseFromParent();
      }
      MFI.RemoveStackObject(FI);
      Changed = true;
      continue;
    }

    Register V = MRI.createVirtualRegister(&SASS::GPR32RegClass);
    VRM.grow();
    SmallVector<std::pair<MachineInstr *, MachineInstr *>, 8> Swapped;
    for (MachineInstr *MI : S.Spills) {
      const MachineOperand &Src = MI->getOperand(2);
      auto MIB = BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                         TII.get(SASS::SELri), V)
                     .addReg(SASS::RZ)
                     .addImm(1)
                     .addReg(Src.getReg(), getKillRegState(Src.isKill()) |
                                               getUndefRegState(Src.isUndef()))
                     .addImm(1); // !P: 1 when P is true.
      Guard(MIB);
      Swapped.push_back({MI, MIB});
    }
    for (MachineInstr *MI : S.Reloads) {
      auto MIB = BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                         TII.get(SASS::ISETPri), MI->getOperand(0).getReg())
                     .addReg(V)
                     .addImm(0)
                     .addImm(5) // NE
                     .addReg(SASS::PT)
                     .addImm(0);
      Guard(MIB);
      Swapped.push_back({MI, MIB});
    }
    for (auto [Old, New] : Swapped)
      Swap(*Old, *New);
    LiveInterval &LI = LIS.createAndComputeVirtRegInterval(V);

    // The lowest free register within every limit %v is live under.
    unsigned Limit = ~0u;
    for (auto [MBB, L] : Limits)
      if (LI.overlaps(LIS.getMBBStartIdx(MBB), LIS.getMBBEndIdx(MBB)))
        Limit = std::min(Limit, L);
    MCRegister Found;
    for (MCPhysReg R : Order) {
      unsigned Enc = TRI.getEncodingValue(R);
      if (Enc <= MaxEnc && Enc + 3 <= Limit &&
          Matrix.checkInterference(LI, R) == LiveRegMatrix::IK_Free) {
        Found = R;
        break;
      }
    }
    if (Found.isValid()) {
      Matrix.assign(LI, Found);
      for (auto [Old, New] : Swapped)
        MF.deleteMachineInstr(Old);
      MFI.RemoveStackObject(FI);
      Changed = true;
      continue;
    }
    LLVM_DEBUG({
      unsigned Free = 0;
      for (MCPhysReg R : Order)
        Free += Matrix.checkInterference(LI, R) == LiveRegMatrix::IK_Free;
      dbgs() << "slot " << FI << ": no register under limit " << Limit
             << " (" << Free << " free above it) for " << LI << '\n';
    });
    // No free register: back to the slot.
    LIS.removeInterval(V);
    for (auto [Old, New] : Swapped) {
      New->getParent()->insert(New->getIterator(), Old);
      LIS.ReplaceMachineInstrInMaps(*New, *Old);
      New->eraseFromParent();
    }
  }
  return Changed;
}
