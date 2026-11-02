//===-- SASSPipeBalance.cpp - Move integer shifts onto the FMA pipe --------===//
//
// Hopper issues one instruction per cycle per sub-core, but its integer ALU
// pipe (LOP3, SHF, IADD3, LEA, PRMT, ...) and the FMA-heavy pipe (IMAD,
// HFMA2/HMUL2) each accept a warp instruction only every other cycle. A
// stretch dominated by one of them runs at half the issue rate. ptxas
// balances them by issuing a left shift by a constant as a multiply,
//
//   SHF.L.U32 R1, R2, 0x4, RZ   ->   IMAD.SHL.U32 R1, R2, 0x10, RZ
//
// when the surrounding code is ALU-bound, e.g. an FP4 dequantization loop of
// LOP3/SHF bit extracts feeding HMUL2s.
//
// After the pre-RA scheduler has fixed each block's order (nothing reorders
// it later), every shift by a constant whose window of nearby instructions
// holds more ALU than FMA-heavy work becomes an IMAD.SHL.U32. FP32 FFMA,
// FMUL and FADD issue to either FMA half and count half. Earlier conversions
// count in later windows.
//
// The pass also joins the per-element float atomic adds of an
// atom/red.global.v2/v4.f32.add (SASSExpandInlinePTX, first element aligned
// to the vector) back into one ATOMG/REDG .F32x2/.F32x4, as nvcc emits; the
// same for packed .f16x2/.bf16x2 words (.F16x4/.F16x8, .BF16x4/.BF16x8).
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "sass-pipe-balance"

static cl::opt<unsigned>
    PipeWindow("sass-pipe-window", cl::init(8),
               cl::desc("Instructions on each side of a shift weighed for "
                        "pipe balance (0: never move shifts to IMAD)"));

namespace {

#include "SASSLatencyTables.inc"

class SASSPipeBalance : public MachineFunctionPass {
public:
  static char ID;
  SASSPipeBalance() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "SASS pipe balance"; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

char SASSPipeBalance::ID = 0;

INITIALIZE_PASS(SASSPipeBalance, DEBUG_TYPE, "SASS pipe balance", false, false)

FunctionPass *llvm::createSASSPipeBalancePass() {
  return new SASSPipeBalance();
}

/// Load of \p MI on the integer ALU pipe minus its load on the FMA-heavy
/// pipe, in half-instructions.
static int pipeLoad(const MachineInstr &MI) {
  switch (SASS::latMnem(MI.getDesc())) {
  case LAT_LOP3: case LAT_SHF: case LAT_IADD3: case LAT_LEA: case LAT_PRMT:
  case LAT_SEL: case LAT_ISETP: case LAT_PLOP3: case LAT_VIMNMX:
    return 2;
  case LAT_IMAD: case LAT_IADD: case LAT_HFMA2:
    return -2;
  case LAT_FFMA: case LAT_FMUL: case LAT_FADD:
    return -1;
  default:
    return 0;
  }
}

/// The 2- and 4-word forms of a global per-word float atomic add (ATOMG,
/// REDG), or nullptr.
static const unsigned *joinedAtomics(unsigned Opc) {
  static const unsigned T[][6] = {
      {SASS::AT_FADDG, SASS::AT_FADDRG, SASS::AT_FADDX2G, SASS::AT_FADDX2RG,
       SASS::AT_FADDX4G, SASS::AT_FADDX4RG},
      {SASS::AT_HADD2G, SASS::AT_HADD2RG, SASS::AT_HADD4G, SASS::AT_HADD4RG,
       SASS::AT_HADD8G, SASS::AT_HADD8RG},
      {SASS::AT_BADD2G, SASS::AT_BADD2RG, SASS::AT_BADD4G, SASS::AT_BADD4RG,
       SASS::AT_BADD8G, SASS::AT_BADD8RG}};
  for (const unsigned *R : T)
    if (Opc == R[0] || Opc == R[1])
      return R;
  return nullptr;
}

/// Joins the global float atomic adds starting at \p I0 (F32 or packed
/// F16x2/BF16x2 words, ATOMG or REDG) on consecutive words of one aligned
/// vector into a vector atomic; returns whether it did.
static bool joinAtomics(MachineInstr &I0, const unsigned *Forms,
                        const TargetInstrInfo &TII, MachineRegisterInfo &MRI) {
  bool Ret = I0.getOpcode() == Forms[0];
  unsigned A = Ret ? 1 : 0; // base, offset, value, scope, guard
  if (!I0.hasOneMemOperand() || !I0.getOperand(A).isReg() ||
      I0.getOperand(A + 4).getReg() != SASS::PT || I0.getOperand(A + 5).getImm())
    return false;
  uint64_t Al = (*I0.memoperands_begin())->getAlign().value();
  unsigned N = Al >= 16 ? 4 : Al >= 8 ? 2 : 1;
  SmallVector<MachineInstr *, 4> Grp = {&I0};
  for (MachineBasicBlock::iterator It = std::next(I0.getIterator());
       Grp.size() < N && It != I0.getParent()->end(); ++It) {
    MachineInstr &M = *It;
    if (M.getOpcode() == I0.getOpcode() && M.hasOneMemOperand() &&
        M.getOperand(A).getReg() == I0.getOperand(A).getReg() &&
        M.getOperand(A + 1).getImm() == I0.getOperand(A + 1).getImm() + 4 * int64_t(Grp.size()) &&
        M.getOperand(A + 3).getImm() == I0.getOperand(A + 3).getImm() &&
        M.getOperand(A + 4).getReg() == SASS::PT && !M.getOperand(A + 5).getImm()) {
      Grp.push_back(&M);
      continue;
    }
    if (M.mayLoadOrStore() || M.hasUnmodeledSideEffects() || M.isTerminator())
      break;
  }
  if (Grp.size() == 3)
    Grp.pop_back();
  if (Grp.size() < 2)
    return false;
  // The joined atomic replaces the last one (every value is defined before
  // it); no earlier member's result may be read before that.
  MachineBasicBlock &MBB = *I0.getParent();
  MachineInstr *Last = Grp.back();
  if (Ret) {
    SmallPtrSet<const MachineInstr *, 16> Between;
    for (MachineBasicBlock::iterator It = I0.getIterator(); &*It != Last; ++It)
      Between.insert(&*It);
    for (MachineInstr *M : Grp)
      for (const MachineInstr &U : MRI.use_nodbg_instructions(M->getOperand(0).getReg()))
        if (Between.count(&U))
          return false;
  }
  MachineBasicBlock::iterator At = std::next(Last->getIterator());
  bool Four = Grp.size() == 4;
  static const unsigned Sub[] = {SASS::sub0, SASS::sub1, SASS::sub2, SASS::sub3};
  const TargetRegisterClass *RC = Four ? &SASS::GPR128RegClass : &SASS::GPR64RegClass;
  DebugLoc DL = I0.getDebugLoc();
  Register V = MRI.createVirtualRegister(RC);
  MachineInstrBuilder Seq = BuildMI(MBB, At, DL, TII.get(TargetOpcode::REG_SEQUENCE), V);
  for (auto [K, M] : enumerate(Grp))
    Seq.addReg(M->getOperand(A + 2).getReg()).addImm(Sub[K]);
  unsigned Opc = Forms[(Four ? 4 : 2) + !Ret];
  MachineInstrBuilder J = BuildMI(MBB, At, DL, TII.get(Opc));
  Register D;
  if (Ret)
    J.addDef(D = MRI.createVirtualRegister(RC));
  J.add(I0.getOperand(A)).add(I0.getOperand(A + 1)).addReg(V)
      .add(I0.getOperand(A + 3)).addReg(SASS::PT).addImm(0);
  MachineFunction &MF = *MBB.getParent();
  J.addMemOperand(MF.getMachineMemOperand(*I0.memoperands_begin(), 0, 4 * Grp.size()));
  for (auto [K, M] : enumerate(Grp))
    if (Ret)
      BuildMI(MBB, At, DL, TII.get(TargetOpcode::COPY), M->getOperand(0).getReg())
          .addReg(D, {}, Sub[K]);
  for (MachineInstr *M : Grp)
    M->eraseFromParent();
  return true;
}

bool SASSPipeBalance::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<SASSSubtarget>();
  if (ST.hasBlackwellISA())
    return false;
  const TargetInstrInfo &TII = *ST.getInstrInfo();
  bool Changed = false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (MachineBasicBlock &MBB : MF)
    for (MachineBasicBlock::iterator It = MBB.begin(); It != MBB.end();) {
      MachineInstr &MI = *It++;
      const unsigned *Forms = joinedAtomics(MI.getOpcode());
      if (Forms && joinAtomics(MI, Forms, TII, MRI)) {
        Changed = true;
        It = MBB.begin(); // (the group's later members are gone)
      }
    }
  if (!PipeWindow)
    return Changed;
  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 0> Insts;
    SmallVector<int, 0> Load;
    for (MachineInstr &MI : MBB)
      if (!MI.isDebugInstr() && !MI.isPseudo()) {
        Insts.push_back(&MI);
        Load.push_back(pipeLoad(MI));
      }
    int W = PipeWindow, N = Insts.size();
    for (int I = 0; I < N; ++I) {
      MachineInstr &MI = *Insts[I];
      if (MI.getOpcode() != SASS::SHLri)
        continue;
      int64_t Sh = MI.getOperand(2).getImm();
      if (Sh < 0 || Sh > 31)
        continue;
      int Bal = 0;
      for (int J = std::max(0, I - W); J < std::min(N, I + W + 1); ++J)
        Bal += Load[J];
      if (Bal <= 0)
        continue;
      MI.setDesc(TII.get(SASS::IMADSHLri));
      MI.getOperand(2).setImm(int32_t(1u << Sh));
      Load[I] = pipeLoad(MI);
      Changed = true;
    }
  }
  return Changed;
}
