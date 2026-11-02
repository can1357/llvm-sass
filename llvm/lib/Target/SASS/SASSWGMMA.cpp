//===-- SASSWGMMA.cpp - Hopper warpgroup MMA and setmaxnreg ----------------===//
//
// wgmma.mma_async has no NVVM intrinsic: SASSExpandInlinePTX re-emits each
// one as a canonical single-instruction inline asm call
//
//   wgmma.mma_async.sync.aligned.<shape>.<dtype>.<atype>.<btype>
//       {$0, ..., $N-1}, A, $B, SCALE_D, imm-scale-a, imm-scale-b[, trans...];
//
// with the accumulators as outputs tied to inputs, A a 64-bit descriptor or
// a {$a, $b, $c, $d} register fragment, and SCALE_D a 0/1 register or
// immediate. The INLINEASM survives instruction selection (the "r", "f" and
// "l" constraints below) and SASSExpandWGMMA rewrites it, in SSA form before
// register allocation, into GMMA_* (SASSInstrWGMMA.td): REG_SEQUENCEs build
// the accumulator tuple, A fragment and the uniform descriptor tuple
// (R2UR: .sync.aligned operands are warp-uniform; B is always the upper
// pair, also in the RS form), and sub-register COPYs
// hand the results back, so the register coalescer keeps a whole
// accumulator chain in one tuple. Doing this on MachineInstrs rather than
// in DAG selection keeps the inline-asm operand plumbing (copies, tied
// operands) generic and lets one place choose tuple classes. f32 shapes with
// N > 128 (more than the 64 lanes a tuple can have) become two GMMAs on the
// column halves, the second B descriptor advanced past the first 128
// columns (CUTLASS's canonical GMMA layouts: 16 SBO strides for K-major and
// no-swizzle MN-major B, 128/(swizzle width in elements) LBO strides for
// swizzled MN-major B).
//
// A GMMA reads its A fragment until the wgmma.wait_group that retires its
// group; with -sass-wgmma-live-frags SASSExpandWGMMA gives each
// WARPGROUP.DEPBAR that retires an RS GMMA an implicit use of the fragment
// (through PHIs when the wait is on the next loop iteration), so the
// allocator keeps the registers until then.
//
// SASSExpandWGMMA also expands setmaxnreg as ptxas does: WARPSYNC.ALL, then
// USETMAXREG.DEALLOC, or a USETMAXREG.TRY_ALLOC retry loop. Like ptxas it
// ignores setmaxnreg in kernels without __launch_bounds__(threads,
// minBlocks): the entry register count must be known.
//
// After register allocation SASSWGMMAHazards
// * sets gsb0 on the last GMMA before each wgmma.commit_group of its block
//   (the group's scoreboard; WARPGROUP.DEPBAR.LE gsb0, N waits for all but N
//   groups); a commit without such a GMMA (empty group, group spanning
//   blocks) makes every GMMA its own group and every wait a full wait,
// * keeps instructions off registers that in-flight GMMAs read or write
//   (accumulators, A fragment), GMMAs included unless they only share A
//   fragments or chain on one accumulator tuple, by inserting
//   WARPGROUP.DEPBAR.LE gsb0, 0x0,
//   and inserts WARPGROUP.ARRIVE before a GMMA whose registers (accumulators,
//   write-only ones included, and A fragment) other instructions read or
//   wrote since the last one (ptxas remarks C7514 / C7519), each arrive
//   preceded by WARPSYNC.ALL to reconverge the warp,
// * checks that each region after a setmaxnreg uses only registers below
//   its count.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SASSGMMAFlags.h"
#include "SASS.h"
#include "SASSISelLowering.h"
#include "SASSInstrInfo.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSRegisterInfo.h"
#include "SASSSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
namespace GMMA = SASS::GMMA;

static cl::opt<bool>
    HoistDesc("sass-wgmma-hoist-desc", cl::init(true), cl::Hidden,
              cl::desc("Hoist loop-invariant wgmma descriptor halves to "
                       "uniform registers in the loop preheader"));

static cl::opt<bool>
    BankHints("sass-bank-hints", cl::init(true), cl::Hidden,
              cl::desc("Hopper: prefer the register-file bank (register "
                       "parity) other than that of the registers read "
                       "together with a 32-bit virtual register"));

static cl::opt<bool>
    LiveFrags("sass-wgmma-live-frags", cl::init(false), cl::Hidden,
              cl::desc("Keep RS wgmma A fragments allocated until the "
                       "wait_group that retires them instead of waiting "
                       "before their registers are reused"));

//===----------------------------------------------------------------------===//
// Inline asm constraints
//===----------------------------------------------------------------------===//

TargetLowering::ConstraintType
SASSTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1)
    switch (Constraint[0]) {
    case 'r':
    case 'f':
    case 'l':
      return C_RegisterClass;
    default:
      break;
    }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
SASSTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                 StringRef Constraint,
                                                 MVT VT) const {
  if (Constraint.size() == 1)
    switch (Constraint[0]) {
    case 'r':
    case 'f':
      if (VT.getSizeInBits() <= 32)
        return {0U, &SASS::GPR32RegClass};
      break;
    case 'l':
      return {0U, &SASS::GPR64RegClass};
    default:
      break;
    }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

static const unsigned SubIdx[64] = {
    SASS::sub0,  SASS::sub1,  SASS::sub2,  SASS::sub3,  SASS::sub4,  SASS::sub5,
    SASS::sub6,  SASS::sub7,  SASS::sub8,  SASS::sub9,  SASS::sub10, SASS::sub11,
    SASS::sub12, SASS::sub13, SASS::sub14, SASS::sub15, SASS::sub16, SASS::sub17,
    SASS::sub18, SASS::sub19, SASS::sub20, SASS::sub21, SASS::sub22, SASS::sub23,
    SASS::sub24, SASS::sub25, SASS::sub26, SASS::sub27, SASS::sub28, SASS::sub29,
    SASS::sub30, SASS::sub31, SASS::sub32, SASS::sub33, SASS::sub34, SASS::sub35,
    SASS::sub36, SASS::sub37, SASS::sub38, SASS::sub39, SASS::sub40, SASS::sub41,
    SASS::sub42, SASS::sub43, SASS::sub44, SASS::sub45, SASS::sub46, SASS::sub47,
    SASS::sub48, SASS::sub49, SASS::sub50, SASS::sub51, SASS::sub52, SASS::sub53,
    SASS::sub54, SASS::sub55, SASS::sub56, SASS::sub57, SASS::sub58, SASS::sub59,
    SASS::sub60, SASS::sub61, SASS::sub62, SASS::sub63};

constexpr unsigned MaxGroups = 7; // WARPGROUP.DEPBAR.LE takes 0..7.

static const TargetRegisterClass *const AccClasses[] = {
    &SASS::GACC8RegClass,  &SASS::GACC12RegClass, &SASS::GACC16RegClass,
    &SASS::GACC20RegClass, &SASS::GACC24RegClass, &SASS::GACC28RegClass,
    &SASS::GACC32RegClass, &SASS::GACC36RegClass, &SASS::GACC40RegClass,
    &SASS::GACC44RegClass, &SASS::GACC48RegClass, &SASS::GACC52RegClass,
    &SASS::GACC56RegClass, &SASS::GACC60RegClass, &SASS::GACC64RegClass};

/// Tuple class for \p Regs accumulator registers (at most 64).
static const TargetRegisterClass *accClass(unsigned Regs) {
  if (Regs <= 2)
    return &SASS::GPR64RegClass;
  if (Regs <= 4)
    return &SASS::GPR128RegClass;
  return AccClasses[(alignTo(Regs, 4) - 8) / 4];
}

bool SASS::isGPRTupleClass(const TargetRegisterClass *RC) {
  if (SASS::GPR32RegClass.hasSubClassEq(RC) ||
      SASS::GPR64RegClass.hasSubClassEq(RC) ||
      SASS::GPR128RegClass.hasSubClassEq(RC))
    return true;
  return is_contained(AccClasses, RC);
}

static bool isGMMA(unsigned Opc) {
  return Opc == SASS::GMMA_SS || Opc == SASS::GMMA_RS ||
         Opc == SASS::GMMA_SS_Z || Opc == SASS::GMMA_RS_Z;
}

bool SASS::isWarpgroupSync(const MachineInstr &MI) {
  return MI.getOpcode() == SASS::WARPGROUP_ARRIVE ||
         MI.getOpcode() == SASS::WARPGROUP_DEPBAR;
}

static bool hasLaunchBounds(const Function &F) {
  return F.hasFnAttribute("nvvm.maxntid") && F.hasFnAttribute("nvvm.minctasm");
}

unsigned SASS::setMaxRegEntryCount(const MachineFunction &MF) {
  for (const MachineBasicBlock &B : MF)
    for (const MachineInstr &MI : B)
      if (MI.getOpcode() == SASS::USETMAXREG_DEALLOC ||
          MI.getOpcode() == SASS::USETMAXREG_TRY_ALLOC)
        return maxRegCount(MF.getFunction());
  return 0;
}

unsigned SASS::setMaxRegLimit(const MachineFunction &MF) {
  unsigned Limit = maxRegCount(MF.getFunction());
  if (!hasLaunchBounds(MF.getFunction()))
    return Limit;
  // Reserved registers are frozen at the end of selection, before the
  // pseudos are expanded: look at both forms.
  for (const MachineBasicBlock &B : MF)
    for (const MachineInstr &MI : B)
      if (MI.getOpcode() == SASS::SETMAXNREG_INC ||
          MI.getOpcode() == SASS::USETMAXREG_TRY_ALLOC)
        Limit = std::max<unsigned>(Limit, MI.getOperand(MI.getNumDefs()).getImm());
  return std::min(Limit, 255u);
}

static void noGuard(const MachineInstrBuilder &MIB) { MIB.addReg(SASS::PT).addImm(0); }

/// Register count after \p MI if it is a USETMAXREG, else \p L.
static unsigned limitAfter(const MachineInstr &MI, unsigned L) {
  if (MI.getOpcode() == SASS::USETMAXREG_DEALLOC ||
      MI.getOpcode() == SASS::USETMAXREG_TRY_ALLOC)
    return MI.getOperand(MI.getNumDefs()).getImm();
  return L;
}

/// Register count in effect at each block entry: \p Entry at the function
/// entry, then the minimum over incoming paths.
static DenseMap<const MachineBasicBlock *, unsigned>
regLimitsIn(MachineFunction &MF, unsigned Entry) {
  DenseMap<const MachineBasicBlock *, unsigned> In, Out;
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (MachineBasicBlock *B : RPOT) {
      unsigned L = B == &MF.front() ? Entry : ~0u;
      for (MachineBasicBlock *P : B->predecessors())
        if (auto It = Out.find(P); It != Out.end())
          L = std::min(L, It->second);
      In[B] = L;
      for (const MachineInstr &MI : *B)
        L = limitAfter(MI, L);
      auto [It, New] = Out.try_emplace(B, L);
      if (New || It->second != L) {
        It->second = L;
        Changed = true;
      }
    }
  }
  return In;
}

bool SASSRegisterInfo::getRegAllocationHints(
    Register VirtReg, ArrayRef<MCPhysReg> Order,
    SmallVectorImpl<MCPhysReg> &Hints, const MachineFunction &MF,
    const VirtRegMap *VRM, const LiveRegMatrix *Matrix) const {
  bool Hard = TargetRegisterInfo::getRegAllocationHints(VirtReg, Order, Hints,
                                                        MF, VRM, Matrix);
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  // Hopper's register file has two banks, by register parity, each reading
  // one warp-wide operand per clock for all of a sub-core's warps
  // (sass-dec notes/sm90/arch/register_bandwidth.md): an instruction reading
  // two registers of one parity takes a second clock. A 32-bit register
  // prefers the parity its fellow sources (already assigned) have least,
  // e.g. a dequantized weight the other parity than the scale it is
  // multiplied by.
  int Parity = -1;
  if (BankHints && VRM && !MF.getSubtarget<SASSSubtarget>().hasBlackwellISA() &&
      MRI.getRegClass(VirtReg) == &SASS::GPR32RegClass) {
    int Votes[2] = {0, 0};
    for (const MachineOperand &MO : MRI.use_nodbg_operands(VirtReg))
      for (const MachineOperand &U : MO.getParent()->uses()) {
        if (!U.isReg() || U.isImplicit() || !U.getReg() ||
            U.getReg() == VirtReg)
          continue;
        MCRegister P;
        if (U.getReg().isPhysical())
          P = U.getReg();
        else if (U.getReg().isVirtual() && VRM->hasPhys(U.getReg()))
          P = VRM->getPhys(U.getReg());
        if (P && U.getSubReg())
          P = getSubReg(P, U.getSubReg());
        if (P && P != SASS::RZ && SASS::GPR32RegClass.contains(P))
          ++Votes[getEncodingValue(P) & 1];
      }
    if (Votes[0] != Votes[1])
      Parity = Votes[0] > Votes[1];
  }
  const auto &Limits = MF.getInfo<SASSFunctionInfo>()->RegLimit;
  unsigned L = ~0u;
  if (!Limits.empty())
    for (const MachineInstr &MI : MRI.reg_nodbg_instructions(VirtReg))
      if (auto It = Limits.find(MI.getParent()); It != Limits.end())
        L = std::min(L, It->second);
  if (L == ~0u && Parity < 0)
    return Hard;
  // Only registers whose 32-bit lanes all lie below L - 2.
  auto Fits = [&](MCPhysReg R) {
    if (L == ~0u)
      return true;
    for (MCRegUnit U : regunits(R))
      for (MCRegUnitRootIterator Root(U, this); Root.isValid(); ++Root)
        if (SASS::GPR32RegClass.contains(*Root) && getEncodingValue(*Root) + 3 > L)
          return false;
    return true;
  };
  SmallVector<MCPhysReg, 16> Pref(Hints.begin(), Hints.end());
  Hints.clear();
  for (MCPhysReg R : Pref)
    if (Fits(R) && is_contained(Order, R))
      Hints.push_back(R);
  if (Parity >= 0)
    for (MCPhysReg R : Order)
      if (int(getEncodingValue(R) & 1) == Parity && Fits(R) &&
          !is_contained(Hints, R))
        Hints.push_back(R);
  if (L == ~0u)
    return Hard;
  for (MCPhysReg R : Order)
    if (Fits(R) && !is_contained(Hints, R))
      Hints.push_back(R);
  return true;
}

//===----------------------------------------------------------------------===//
// SASSExpandWGMMA
//===----------------------------------------------------------------------===//

namespace {

class SASSExpandWGMMA : public MachineFunctionPass {
public:
  static char ID;
  SASSExpandWGMMA() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS expand wgmma"; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineRegisterInfo *MRI = nullptr;
  const SASSInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  /// The descriptor R2URs built by expandWGMMA.
  SmallVector<MachineInstr *, 16> DescR2URs;
  /// RS GMMAs (the last part) and their A fragments.
  SmallVector<std::pair<MachineInstr *, Register>, 8> RSFrags;

  void expandWGMMA(MachineInstr &MI);
  void extendFragments(MachineFunction &MF);
  void expandSetMaxNReg(MachineInstr &MI);
  bool hoistableChain(Register R, const MachineLoop *L, unsigned Depth,
                      const MachineInstr *User,
                      SmallVectorImpl<MachineInstr *> &Chain);
  void hoistInvariantDescriptors(const MachineLoopInfo &MLI);
  Register vreg(const TargetRegisterClass *RC) {
    return MRI->createVirtualRegister(RC);
  }
};

} // namespace

char SASSExpandWGMMA::ID = 0;

INITIALIZE_PASS(SASSExpandWGMMA, "sass-expand-wgmma", "SASS expand wgmma",
                false, false)

FunctionPass *llvm::createSASSExpandWGMMAPass() { return new SASSExpandWGMMA(); }

/// Top-level comma-separated operands (braces group).
static SmallVector<StringRef, 8> splitOperands(StringRef S) {
  SmallVector<StringRef, 8> Out;
  int Depth = 0;
  size_t Start = 0;
  for (size_t I = 0; I < S.size(); ++I) {
    if (S[I] == '{')
      ++Depth;
    else if (S[I] == '}')
      --Depth;
    else if (S[I] == ',' && Depth == 0) {
      Out.push_back(S.slice(Start, I).trim());
      Start = I + 1;
    }
  }
  if (!S.slice(Start, S.size()).trim().empty())
    Out.push_back(S.slice(Start, S.size()).trim());
  return Out;
}

void SASSExpandWGMMA::expandWGMMA(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  StringRef Text = MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName();
  Text = Text.trim().rtrim(';').trim();
  StringRef Mnem, Rest;
  std::tie(Mnem, Rest) = Text.split(' ');
  auto Fail = [&](const Twine &Why) {
    report_fatal_error("SASS: " + MF.getName() + ": " + Mnem + ": " + Why);
  };

  // Operand groups: outputs are $0..$N-1, inputs follow in order.
  SmallVector<Register, 64> Defs;
  SmallVector<MachineOperand *, 80> Uses;
  for (unsigned I = InlineAsm::MIOp_FirstOperand, E = MI.getNumOperands(); I < E;) {
    const MachineOperand &FO = MI.getOperand(I);
    if (!FO.isImm()) {
      ++I;
      continue;
    }
    InlineAsm::Flag F(FO.getImm());
    if (F.isRegDefKind() || F.isRegDefEarlyClobberKind())
      Defs.push_back(MI.getOperand(I + 1).getReg());
    else if (F.isRegUseKind() || F.isImmKind())
      Uses.push_back(&MI.getOperand(I + 1));
    I += 1 + F.getNumOperandRegisters();
  }
  // $k -> operand.
  auto Ref = [&](StringRef S) -> int {
    unsigned K;
    if (!S.consume_front("$") || S.getAsInteger(10, K) || K < Defs.size() ||
        K - Defs.size() >= Uses.size())
      return -1;
    return K - Defs.size();
  };
  auto Imm = [&](StringRef S, std::initializer_list<int> Allowed, const char *What) {
    int V;
    if (S.getAsInteger(10, V) || !is_contained(Allowed, V))
      Fail(Twine("unsupported ") + What + " '" + S + "'");
    return V;
  };

  SmallVector<StringRef, 12> Mods;
  Mnem.split(Mods, '.');
  if (is_contained(Mods, "sp"))
    Fail("sparse wgmma is not supported");
  auto ShapeIt = find_if(Mods, [](StringRef M) { return M.starts_with("m64n"); });
  if (ShapeIt == Mods.end() || Mods.end() - ShapeIt != 4)
    Fail("unrecognized shape/types");
  StringRef Shape = *ShapeIt, DType = ShapeIt[1], AType = ShapeIt[2],
            BType = ShapeIt[3];
  unsigned N = 0, K = 0;
  {
    StringRef S = Shape.drop_front(4);
    auto [NS, KS] = S.split('k');
    if (NS.getAsInteger(10, N) || KS.getAsInteger(10, K) || N < 8 || N > 256 || N % 8)
      Fail("unsupported shape " + Shape);
  }
  auto Fmt = [&](StringRef T) -> GMMA::Fmt {
    if (T == "f16") return GMMA::F16;
    if (T == "bf16") return GMMA::BF16;
    if (T == "tf32") return GMMA::TF32;
    if (T == "e4m3") return GMMA::E4M3;
    if (T == "e5m2") return GMMA::E5M2;
    Fail("." + T + " inputs are not supported (integer/b1 wgmma)");
    llvm_unreachable("fatal");
  };
  GMMA::Fmt FA = Fmt(AType), FB = Fmt(BType);
  bool Q = FA >= GMMA::E4M3;
  if (Q != (FB >= GMMA::E4M3) || (!Q && FA != FB))
    Fail("mismatched A/B types");
  if (DType != "f32" && (DType != "f16" || FA == GMMA::TF32))
    Fail("unsupported accumulator type ." + DType);
  bool F32 = DType == "f32";
  unsigned Flags = GMMA::make(N, F32, FA, FB);
  if (K != GMMA::k(Flags))
    Fail("unsupported shape " + Shape);
  unsigned Regs = GMMA::accRegs(Flags);

  SmallVector<StringRef, 8> Ops = splitOperands(Rest);
  bool HasTrans = FA == GMMA::F16 || FA == GMMA::BF16;
  if (Ops.size() < 3)
    Fail("bad operands");
  bool RS = Ops[1].starts_with("{");
  unsigned Want = 6 + (HasTrans ? (RS ? 1 : 2) : 0);
  if (Ops.size() != Want)
    Fail("expected " + Twine(Want) + " operands");

  // Accumulators: the output list in order, with their tied inputs.
  SmallVector<StringRef, 128> DOps;
  {
    StringRef D = Ops[0];
    if (!D.consume_front("{") || !D.consume_back("}"))
      Fail("bad accumulator list");
    D.split(DOps, ',');
  }
  if (DOps.size() != Regs)
    Fail("expected " + Twine(Regs) + " accumulator registers");
  SmallVector<Register, 128> DOut(Regs), DIn(Regs);
  for (unsigned I = 0; I < Regs; ++I) {
    StringRef S = DOps[I].trim();
    unsigned K2 = 0;
    if (!S.consume_front("$") || S.getAsInteger(10, K2) || K2 >= Defs.size())
      Fail("bad accumulator operand");
    DOut[I] = Defs[K2];
  }
  for (auto [UI, U] : enumerate(Uses)) {
    unsigned FlagIdx = 0; // Flag operand of this use: scan back.
    for (unsigned J = U->getOperandNo(); J-- > 0;)
      if (MI.getOperand(J).isImm()) {
        FlagIdx = J;
        break;
      }
    InlineAsm::Flag F(MI.getOperand(FlagIdx).getImm());
    unsigned DefGroup;
    if (F.isRegUseKind() && F.isUseOperandTiedToDef(DefGroup)) {
      Register Out = Defs[DefGroup];
      for (unsigned I = 0; I < Regs; ++I)
        if (DOut[I] == Out)
          DIn[I] = U->getReg();
    }
  }

  // Scale-d: immediate, constant register, or a runtime predicate.
  int SDConst = -1;
  Register SDReg;
  if (int R = Ref(Ops[3]); R >= 0) {
    SDReg = Uses[R]->isReg() ? Uses[R]->getReg() : Register();
    if (!SDReg)
      SDConst = Uses[R]->getImm() != 0;
    else {
      MachineInstr *D = MRI->getVRegDef(SDReg);
      while (D && D->isCopy() && D->getOperand(1).getReg().isVirtual() &&
             !D->getOperand(1).getSubReg())
        D = MRI->getVRegDef(D->getOperand(1).getReg());
      if (D && D->getOpcode() == SASS::MOVri)
        SDConst = D->getOperand(1).getImm() != 0;
    }
  } else {
    SDConst = Imm(Ops[3], {0, 1}, "scale-d");
  }
  if (SDConst < 0)
    for (Register R : DIn)
      if (!R)
        Fail("accumulator inputs missing");
  if (Imm(Ops[4], {1, -1}, "imm-scale-a") < 0)
    Flags |= GMMA::NegA;
  if (Imm(Ops[5], {1, -1}, "imm-scale-b") < 0)
    Flags |= GMMA::NegB;
  bool TransB = false;
  if (HasTrans && !RS && Imm(Ops[6], {0, 1}, "imm-trans-a"))
    Flags |= GMMA::TnspA;
  if (HasTrans && Imm(Ops[RS ? 6 : 7], {0, 1}, "imm-trans-b")) {
    Flags |= GMMA::TnspB;
    TransB = true;
  }

  auto ToUR = [&](Register G64, unsigned Sub) {
    Register U = vreg(&SASS::UGPR32RegClass);
    auto MIB = BuildMI(MBB, MI, DL, TII->get(SASS::R2UR), U).addReg(G64, {}, Sub);
    noGuard(MIB);
    DescR2URs.push_back(MIB);
    return U;
  };
  auto ToURReg = [&](Register G32) {
    Register U = vreg(&SASS::UGPR32RegClass);
    noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::R2UR), U).addReg(G32));
    return U;
  };
  auto Desc = [&](StringRef S, const char *What) {
    int R = Ref(S);
    if (R < 0 || !Uses[R]->isReg() ||
        !SASS::GPR64RegClass.hasSubClassEq(MRI->getRegClass(Uses[R]->getReg())))
      Fail(Twine("bad ") + What + " descriptor operand");
    return Uses[R]->getReg();
  };

  // A: descriptor (low half of the SS quad) or a register fragment.
  Register AFrag, AUR0, AUR1;
  if (RS) {
    StringRef S = Ops[1].drop_front().drop_back();
    SmallVector<StringRef, 4> Parts;
    S.split(Parts, ',');
    if (Parts.size() != 4)
      Fail("A fragment must have 4 registers");
    AFrag = vreg(&SASS::GPR128RegClass);
    auto RSeq = BuildMI(MBB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), AFrag);
    for (auto [I, P] : enumerate(Parts)) {
      int R = Ref(P.trim());
      if (R < 0 || !Uses[R]->isReg())
        Fail("bad A fragment operand");
      RSeq.addReg(Uses[R]->getReg()).addImm(SubIdx[I]);
    }
  } else {
    Register A = Desc(Ops[1], "A");
    AUR0 = ToUR(A, SASS::sub0);
    AUR1 = ToUR(A, SASS::sub1);
  }
  Register B = Desc(Ops[2], "B");
  Register BHi = ToUR(B, SASS::sub1);

  Register UP = SASS::UPT;
  if (SDConst < 0) {
    Register P = vreg(&SASS::PREDRegClass);
    noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::ISETPri), P)
                .addReg(SDReg)
                .addImm(0)
                .addImm(5) // NE
                .addReg(SASS::PT)
                .addImm(0));
    UP = vreg(&SASS::UPREDRegClass);
    noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::VOTEU_ANY), UP).addReg(P));
  }

  // Low word of B advanced by Cols columns (Cols = 128).
  auto AdvanceB = [&]() {
    auto Op = [&](unsigned Opc) {
      Register R = vreg(&SASS::GPR32RegClass);
      return std::make_pair(R, BuildMI(MBB, MI, DL, TII->get(Opc), R));
    };
    // 16 * SBO (bits 45:32, in 16-byte units).
    auto [SBO, I1] = Op(SASS::LOP3rir);
    noGuard(I1.addReg(B, {}, SASS::sub1).addImm(0x3fff).addReg(SASS::RZ).addImm(0xc0));
    auto [Delta, I2] = Op(SASS::SHLri);
    noGuard(I2.addReg(SBO).addImm(4));
    if (TransB) {
      // Swizzled MN-major: (128 / (swizzle bytes / 2)) * LBO = LBO << mode
      // for modes 1 (128B), 2 (64B), 3 (32B); no swizzle (0) uses SBO.
      auto [L0, I3] = Op(SASS::SRLri);
      noGuard(I3.addReg(B, {}, SASS::sub0).addImm(16));
      auto [LBO, I4] = Op(SASS::LOP3rir);
      noGuard(I4.addReg(L0).addImm(0x3fff).addReg(SASS::RZ).addImm(0xc0));
      auto [Mode, I5] = Op(SASS::SRLri);
      noGuard(I5.addReg(B, {}, SASS::sub1).addImm(30));
      auto [LSh, I6] = Op(SASS::SHLrr);
      noGuard(I6.addReg(LBO).addReg(Mode));
      Register P = vreg(&SASS::PREDRegClass);
      noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::ISETPri), P)
                  .addReg(Mode)
                  .addImm(0)
                  .addImm(4) // EQ
                  .addReg(SASS::PT)
                  .addImm(0));
      auto [Sel, I7] = Op(SASS::SELrr);
      noGuard(I7.addReg(Delta).addReg(LSh).addReg(P).addImm(0));
      Delta = Sel;
    }
    auto [Lo, I8] = Op(SASS::IADDrr);
    noGuard(I8.addReg(B, {}, SASS::sub0).addReg(Delta));
    return ToURReg(Lo);
  };

  // One GMMA per part of at most 64 accumulator registers.
  unsigned Parts = Regs > 64 ? 2 : 1;
  MachineInstr *LastG = nullptr;
  for (unsigned Part = 0; Part < Parts; ++Part) {
    unsigned First = Part * 64, Cnt = Parts == 1 ? Regs : (Part ? Regs - 64 : 64);
    unsigned PFlags = Parts == 1 ? Flags
                                 : (Flags & ~0x3fu) | (Part ? (N - 128) / 8 : 16);
    Register BLo = Part ? AdvanceB() : ToUR(B, SASS::sub0);
    Register DescR;
    // {A desc, B desc}; the RS form leaves the A half undefined.
    if (RS) {
      AUR0 = vreg(&SASS::UGPR32RegClass);
      AUR1 = vreg(&SASS::UGPR32RegClass);
      BuildMI(MBB, MI, DL, TII->get(TargetOpcode::IMPLICIT_DEF), AUR0);
      BuildMI(MBB, MI, DL, TII->get(TargetOpcode::IMPLICIT_DEF), AUR1);
    }
    DescR = vreg(&SASS::UGPR128RegClass);
    BuildMI(MBB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), DescR)
        .addReg(AUR0).addImm(SASS::sub0).addReg(AUR1).addImm(SASS::sub1)
        .addReg(BLo).addImm(SASS::sub2).addReg(BHi).addImm(SASS::sub3);
    const TargetRegisterClass *RC = accClass(Cnt);
    unsigned Lanes = TRI->getRegSizeInBits(*RC) / 32;
    Register In;
    if (SDConst != 0) {
      // An accumulator chain: the inputs are lanes of one earlier tuple.
      Register Chain;
      for (unsigned I = 0; I < Cnt; ++I) {
        Register V = DIn[First + I];
        MachineInstr *D = MRI->getVRegDef(V);
        while (D && D->isCopy() && !D->getOperand(1).getSubReg() &&
               D->getOperand(1).getReg().isVirtual())
          D = MRI->getVRegDef(D->getOperand(1).getReg());
        Register T = D && D->isCopy() ? D->getOperand(1).getReg() : Register();
        if (!T || !T.isVirtual() || D->getOperand(1).getSubReg() != SubIdx[I] ||
            MRI->getRegClass(T) != RC || (Chain && Chain != T)) {
          Chain = Register();
          break;
        }
        Chain = T;
      }
      if (Chain) {
        In = Chain;
      } else {
        In = vreg(RC);
        auto RSeq = BuildMI(MBB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), In);
        for (unsigned I = 0; I < Lanes; ++I) {
          Register V;
          if (I < Cnt) {
            V = DIn[First + I];
          } else {
            V = vreg(&SASS::GPR32RegClass);
            BuildMI(MBB, MI, DL, TII->get(TargetOpcode::IMPLICIT_DEF), V);
          }
          RSeq.addReg(V).addImm(SubIdx[I]);
        }
      }
    }
    Register Out = vreg(RC);
    MachineInstrBuilder G;
    if (SDConst == 0) {
      G = BuildMI(MBB, MI, DL, TII->get(RS ? SASS::GMMA_RS_Z : SASS::GMMA_SS_Z), Out)
              .addImm(PFlags);
      if (RS)
        G.addReg(AFrag);
      G.addReg(DescR).addImm(0);
    } else {
      G = BuildMI(MBB, MI, DL, TII->get(RS ? SASS::GMMA_RS : SASS::GMMA_SS), Out)
              .addImm(PFlags);
      if (RS)
        G.addReg(AFrag).addReg(DescR).addReg(In);
      else
        G.addReg(In).addReg(DescR);
      G.addReg(UP).addImm(0);
    }
    noGuard(G);
    LastG = G;
    for (unsigned I = 0; I < Cnt; ++I)
      BuildMI(MBB, MI, DL, TII->get(TargetOpcode::COPY), DOut[First + I])
          .addReg(Out, {}, SubIdx[I]);
  }
  if (RS)
    RSFrags.emplace_back(LastG, AFrag);
  MI.eraseFromParent();
}

/// The A fragment of an RS GMMA is read until the WARPGROUP.DEPBAR that
/// retires the GMMA's group, but its last use is the GMMA: the allocator
/// hands its registers to the next values (the next step's fragment, a
/// following GMMA's accumulators) and SASSWGMMAHazards then waits for every
/// group before they are written. With -sass-wgmma-live-frags an implicit
/// use on each retiring DEPBAR keeps the fragment live instead, so the
/// group stays in flight; MachineSSAUpdater threads it through PHIs
/// (undefined on paths that do not issue the GMMA) when the wait is on the
/// next loop iteration. Groups count as in PTX: the GMMA joins the group of
/// the next commit_group, and wait_group N retires it once more than N
/// commits followed it. When a path reaches the fragment's definition again
/// first, the next instance would need the same registers while this one
/// is in flight: no extension. Off by default: under a tight register count
/// (moe MAIN at 96, two groups in flight) the longer live ranges spilled
/// 456 bytes more than waiting does.
void SASSExpandWGMMA::extendFragments(MachineFunction &MF) {
  constexpr unsigned Saturated = MaxGroups + 1; // retired by any DEPBAR
  for (auto [G, Frag] : RSFrags) {
    const MachineInstr *Def = MRI->getVRegDef(Frag);
    MachineBasicBlock *GB = G->getParent();
    // Retiring DEPBARs; true when after G in G's block.
    SmallVector<std::pair<MachineInstr *, bool>, 4> Waits;
    SmallVector<std::tuple<MachineBasicBlock *, MachineBasicBlock::iterator,
                           unsigned>, 8>
        Work;
    DenseSet<std::pair<MachineBasicBlock *, unsigned>> Seen;
    Work.emplace_back(GB, std::next(G->getIterator()), 0);
    bool Reached = false;
    while (!Reached && !Work.empty()) {
      MachineBasicBlock *B;
      MachineBasicBlock::iterator I;
      unsigned K;
      std::tie(B, I, K) = Work.pop_back_val();
      bool Direct = B == GB && I != B->begin(), Retired = false;
      for (; I != B->end(); ++I) {
        MachineInstr *MI = &*I;
        if (MI == Def) {
          Reached = true;
          break;
        }
        if (MI->getOpcode() == SASS::WGMMA_COMMIT) {
          K = std::min(K + 1, Saturated);
        } else if (MI->getOpcode() == SASS::WARPGROUP_DEPBAR &&
                   K > static_cast<unsigned>(MI->getOperand(0).getImm())) {
          if (none_of(Waits, [MI](auto &W) { return W.first == MI; }))
            Waits.emplace_back(MI, Direct);
          Retired = true;
          break;
        }
      }
      if (Reached || Retired)
        continue;
      for (MachineBasicBlock *S : B->successors())
        if (Seen.insert({S, K}).second)
          Work.emplace_back(S, S->begin(), K);
    }
    if (Reached || Waits.empty())
      continue;
    MachineSSAUpdater SSA(MF);
    SSA.Initialize(Frag);
    SSA.AddAvailableValue(GB, Frag);
    for (auto [W, Direct] : Waits) {
      Register V = Direct ? Frag : SSA.GetValueInMiddleOfBlock(W->getParent());
      W->addOperand(MF, MachineOperand::CreateReg(V, /*isDef=*/false,
                                                  /*isImp=*/true));
    }
  }
}

void SASSExpandWGMMA::expandSetMaxNReg(MachineInstr &MI) {
  MachineBasicBlock &MBB = *MI.getParent();
  MachineFunction &MF = *MBB.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  int64_t N = MI.getOperand(0).getImm();
  if (N < 24 || N > 256 || N % 8)
    report_fatal_error("SASS: " + MF.getName() + ": setmaxnreg count " +
                       Twine(N) + " is not a multiple of 8 in [24, 256]");
  // ptxas ignores setmaxnreg when the entry allocation is unknown.
  if (!hasLaunchBounds(MF.getFunction())) {
    MI.eraseFromParent();
    return;
  }
  if (MI.getOpcode() == SASS::SETMAXNREG_DEC) {
    noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::WARPSYNC_ALL)));
    noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::USETMAXREG_DEALLOC)).addImm(N));
    MI.eraseFromParent();
    return;
  }
  // Retry until the CTA pool has the registers:
  //   Loop: WARPSYNC.ALL; USETMAXREG.TRY_ALLOC.CTAPOOL UP, N;
  //         PLOP3.LUT P, PT, PT, PT, UP, 0x80, 0x0; @!P BRA Loop
  MachineBasicBlock *Loop = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *Cont = MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.insert(std::next(MBB.getIterator()), Loop);
  MF.insert(std::next(Loop->getIterator()), Cont);
  Cont->splice(Cont->begin(), &MBB, std::next(MI.getIterator()), MBB.end());
  Cont->transferSuccessorsAndUpdatePHIs(&MBB);
  MBB.addSuccessor(Loop);
  Loop->addSuccessor(Loop);
  Loop->addSuccessor(Cont);
  noGuard(BuildMI(Loop, DL, TII->get(SASS::WARPSYNC_ALL)));
  Register UP = vreg(&SASS::UPREDRegClass);
  noGuard(BuildMI(Loop, DL, TII->get(SASS::USETMAXREG_TRY_ALLOC), UP).addImm(N));
  Register P = vreg(&SASS::PREDRegClass);
  noGuard(BuildMI(Loop, DL, TII->get(SASS::PLOP3_UP), P).addReg(UP));
  BuildMI(Loop, DL, TII->get(SASS::BRAc)).addMBB(Loop).addReg(P).addImm(1);
  MI.eraseFromParent();
}

/// Collects, operands first, the in-loop instructions to hoist out of \p L so
/// that \p R is loop-invariant: pure single-def arithmetic on invariant
/// operands whose only users are R2URs or \p User (so no GPR value starts to
/// live across the loop). False when \p R cannot be made invariant.
bool SASSExpandWGMMA::hoistableChain(Register R, const MachineLoop *L,
                                     unsigned Depth, const MachineInstr *User,
                                     SmallVectorImpl<MachineInstr *> &Chain) {
  if (R.isPhysical())
    return R == SASS::RZ || R == SASS::PT || R == SASS::URZ || R == SASS::UPT;
  MachineInstr *D = nullptr;
  for (MachineInstr &Def : MRI->def_instructions(R))
    if (L->contains(Def.getParent())) {
      if (D || !MRI->hasOneDef(R))
        return false;
      D = &Def;
    }
  if (!D || is_contained(Chain, D))
    return true;
  if (!Depth || D->isPHI() || D->isInlineAsm() || D->mayLoadOrStore() ||
      D->hasUnmodeledSideEffects() || D->isConvergent() || D->isTerminator() ||
      D->getNumExplicitDefs() != 1 || D->getNumImplicitOperands())
    return false;
  for (const MachineInstr &U : MRI->use_nodbg_instructions(R))
    if (U.getOpcode() != SASS::R2UR && &U != User)
      return false;
  for (const MachineOperand &MO : D->uses())
    if (MO.isReg() && MO.getReg() &&
        !hoistableChain(MO.getReg(), L, Depth - 1, D, Chain))
      return false;
  Chain.push_back(D);
  return true;
}

/// A descriptor half that does not change in a loop (the constant high word,
/// a loop-invariant base) is moved to uniform registers in the preheader,
/// with the GPR arithmetic only it uses, instead of occupying a GPR pair
/// across the loop (spilled and reloaded every iteration under pressure).
void SASSExpandWGMMA::hoistInvariantDescriptors(const MachineLoopInfo &MLI) {
  for (MachineInstr *MI : DescR2URs) {
    for (MachineLoop *L = MLI.getLoopFor(MI->getParent()); L;
         L = L->getParentLoop()) {
      MachineBasicBlock *PH = L->getLoopPreheader();
      SmallVector<MachineInstr *, 4> Chain;
      if (!PH || !hoistableChain(MI->getOperand(1).getReg(), L, 3, MI, Chain))
        break;
      Chain.push_back(MI);
      for (MachineInstr *H : Chain)
        H->moveBefore(&*PH->getFirstTerminator());
    }
  }
}

bool SASSExpandWGMMA::runOnMachineFunction(MachineFunction &MF) {
  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget<SASSSubtarget>().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  DescR2URs.clear();
  RSFrags.clear();
  SmallVector<MachineInstr *, 8> Work;
  for (MachineBasicBlock &B : MF)
    for (MachineInstr &MI : B)
      if ((MI.isInlineAsm() &&
           StringRef(MI.getOperand(InlineAsm::MIOp_AsmString).getSymbolName())
               .ltrim()
               .starts_with("wgmma.mma_async")) ||
          MI.getOpcode() == SASS::SETMAXNREG_INC ||
          MI.getOpcode() == SASS::SETMAXNREG_DEC)
        Work.push_back(&MI);
  for (MachineInstr *MI : Work) {
    if (MI->isInlineAsm())
      expandWGMMA(*MI);
    else
      expandSetMaxNReg(*MI);
  }
  if (LiveFrags)
    extendFragments(MF);
  if (HoistDesc && !DescR2URs.empty()) {
    // (after the setmaxnreg retry loops are in the CFG)
    MachineDominatorTree MDT(MF);
    hoistInvariantDescriptors(MachineLoopInfo(MDT));
  }
  // Blocks where fewer registers than the function may use are available.
  auto *FI = MF.getInfo<SASSFunctionInfo>();
  FI->RegLimit.clear();
  if (unsigned Entry = SASS::setMaxRegEntryCount(MF)) {
    unsigned Max = SASS::setMaxRegLimit(MF);
    auto In = regLimitsIn(MF, Entry);
    for (MachineBasicBlock &B : MF) {
      unsigned L = In.lookup(&B), Min = L;
      for (const MachineInstr &MI : B)
        Min = std::min(Min, L = limitAfter(MI, L));
      if (Min < Max)
        FI->RegLimit[&B] = Min;
    }
  }
  return !Work.empty();
}

//===----------------------------------------------------------------------===//
// SASSWGMMAHazards
//===----------------------------------------------------------------------===//

namespace {

/// Registers of in-flight GMMAs.
struct GRegs {
  /// Units of the accumulators and of the A fragments.
  BitVector Acc, Src;
  /// The accumulator tuples, sorted.
  SmallVector<Register, 4> AccRegs;

  explicit GRegs(unsigned NumUnits = 0) : Acc(NumUnits), Src(NumUnits) {}
  bool operator==(const GRegs &O) const {
    return Acc == O.Acc && Src == O.Src && AccRegs == O.AccRegs;
  }
  bool any() const { return Acc.any() || Src.any(); }
  BitVector all() const {
    BitVector A = Acc;
    return A |= Src;
  }
  void reset() {
    Acc.reset();
    Src.reset();
    AccRegs.clear();
  }
  void addAcc(Register R) {
    auto It = lower_bound(AccRegs, R);
    if (It == AccRegs.end() || *It != R)
      AccRegs.insert(It, R);
  }
  GRegs &operator|=(const GRegs &O) {
    Acc |= O.Acc;
    Src |= O.Src;
    for (Register R : O.AccRegs)
      addAcc(R);
    return *this;
  }
};

/// GMMA register state at a program point.
struct HState {
  /// Registers of committed, unretired groups, oldest first.
  SmallVector<GRegs, 8> Groups;
  /// Registers of GMMAs issued since the last gsb0 (not yet a group).
  GRegs Open;
  /// Units read or written by other instructions since the last
  /// WARPGROUP.ARRIVE.
  BitVector Dirty;
  bool Valid = false;

  bool operator==(const HState &O) const {
    return Valid == O.Valid && Groups == O.Groups && Open == O.Open &&
           Dirty == O.Dirty;
  }
  /// Union, with group lists aligned at the most recent group.
  void join(const HState &O) {
    if (!O.Valid)
      return;
    if (!Valid) {
      *this = O;
      return;
    }
    if (O.Groups.size() > Groups.size())
      Groups.insert(Groups.begin(), O.Groups.size() - Groups.size(),
                    GRegs(Dirty.size()));
    for (unsigned I = 0, E = O.Groups.size(); I != E; ++I)
      Groups[Groups.size() - E + I] |= O.Groups[I];
    Open |= O.Open;
    Dirty |= O.Dirty;
  }
};

class SASSWGMMAHazards : public MachineFunctionPass {
public:
  static char ID;
  SASSWGMMAHazards() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "SASS wgmma hazards"; }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const TargetRegisterInfo *TRI = nullptr;
  const SASSInstrInfo *TII = nullptr;
  /// A hazard on a GMMA not yet in a group: rerun with every GMMA a group.
  bool NeedSerialize = false;

  HState process(MachineBasicBlock &MBB, HState S, bool Rewrite);
  void addUnits(BitVector &BV, Register R) const {
    for (MCRegUnit U : TRI->regunits(R))
      BV.set(static_cast<unsigned>(U));
  }
  void checkSetMaxReg(MachineFunction &MF);
};

} // namespace

char SASSWGMMAHazards::ID = 0;

INITIALIZE_PASS(SASSWGMMAHazards, "sass-wgmma-hazards", "SASS wgmma hazards",
                false, false)

FunctionPass *llvm::createSASSWGMMAHazardsPass() {
  return new SASSWGMMAHazards();
}

static unsigned gsbOperand(const MachineInstr &MI) {
  return MI.getDesc().getNumOperands() - 3; // ..., gsb, pred, pred-neg
}

static bool isConstReg(Register R) {
  return R == SASS::RZ || R == SASS::PT || R == SASS::URZ || R == SASS::UPT;
}

HState SASSWGMMAHazards::process(MachineBasicBlock &MBB, HState S,
                                 bool Rewrite) {
  // The last GMMA of this block not yet in a group. A hazard on the open
  // group ends the group there (gsb0, then the full wait): every path
  // executes it, and the waits after it count the same groups, the split
  // one retired. Without one (the group began in a predecessor) every GMMA
  // becomes its own group.
  MachineInstr *LastOpen = nullptr;
  auto CloseOpen = [&]() {
    if (!LastOpen) {
      NeedSerialize = true;
      S.Open.reset();
      return;
    }
    LastOpen->getOperand(gsbOperand(*LastOpen)).setImm(1);
    LastOpen = nullptr;
    S.Open.reset();
  };
  for (MachineInstr &MI : make_early_inc_range(MBB)) {
    if (MI.isMetaInstruction())
      continue;
    unsigned Opc = MI.getOpcode();
    const DebugLoc &DL = MI.getDebugLoc();
    if (isGMMA(Opc)) {
      // The accumulator inputs are tied to the outputs.
      Register D = MI.getOperand(0).getReg();
      GRegs R(S.Dirty.size());
      bool RS = Opc == SASS::GMMA_RS || Opc == SASS::GMMA_RS_Z;
      if (RS)
        addUnits(R.Src, MI.getOperand(2).getReg());
      addUnits(R.Acc, D);
      R.addAcc(D);
      BitVector All = R.all();
      // In-flight GMMAs may share A fragments, and chain on one accumulator
      // tuple (same shape, ordered by the hardware); a GMMA must not write
      // another's A fragment (the next QK^T over a P*V still in flight),
      // read an accumulator as its A fragment, or overlap another
      // accumulator tuple. Wait for every group first.
      auto Conflicts = [&](const GRegs &P) {
        if (R.Src.anyCommon(P.Acc) || R.Acc.anyCommon(P.Src))
          return true;
        return R.Acc.anyCommon(P.Acc) &&
               any_of(P.AccRegs, [&](Register A) {
                 return A != D && TRI->regsOverlap(A, D);
               });
      };
      bool OpenHazard = Conflicts(S.Open);
      if (OpenHazard || any_of(S.Groups, Conflicts)) {
        // DEPBAR does not wait for GMMAs outside a group.
        if (OpenHazard)
          CloseOpen();
        if (Rewrite)
          noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::WARPGROUP_DEPBAR)).addImm(0));
        S.Groups.clear();
      }
      // wgmma.fence orders every earlier access of the GMMA's registers,
      // write-only accumulators (scale-d = 0) included: a register the
      // allocator reused as a temporary after the source's fence corrupts
      // the result unless the GMMA has its own arrive.
      if (All.anyCommon(S.Dirty)) {
        if (Rewrite)
          noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::WARPGROUP_ARRIVE)));
        S.Dirty.reset();
      }
      S.Open |= R;
      LastOpen = &MI;
      if (MI.getOperand(gsbOperand(MI)).getImm()) {
        S.Groups.push_back(S.Open);
        S.Open.reset();
        LastOpen = nullptr;
        if (S.Groups.size() > MaxGroups) {
          S.Groups[1] |= S.Groups[0];
          S.Groups.erase(S.Groups.begin());
        }
      }
      continue;
    }
    if (Opc == SASS::WARPGROUP_ARRIVE) {
      S.Dirty.reset();
      continue;
    }
    if (Opc == SASS::WARPGROUP_DEPBAR) {
      unsigned Keep = MI.getOperand(0).getImm();
      if (S.Groups.size() > Keep)
        S.Groups.erase(S.Groups.begin(), S.Groups.end() - Keep);
      continue;
    }
    BitVector Pending = S.Open.all();
    for (const GRegs &G : S.Groups)
      Pending |= G.all();
    BitVector Open = S.Open.all();
    bool Exit = MI.isReturn() || MI.isCall();
    bool Touch = Exit && Pending.any(), TouchOpen = Exit && S.Open.any();
    BitVector Accessed(S.Dirty.size());
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg() || !MO.getReg().isPhysical() ||
          isConstReg(MO.getReg()))
        continue;
      for (MCRegUnit U : TRI->regunits(MO.getReg())) {
        Touch |= Pending.test(static_cast<unsigned>(U));
        TouchOpen |= Open.test(static_cast<unsigned>(U));
        Accessed.set(static_cast<unsigned>(U));
      }
    }
    if (Touch) {
      // DEPBAR does not wait for GMMAs outside a group.
      if (TouchOpen)
        CloseOpen();
      if (Rewrite)
        noGuard(BuildMI(MBB, MI, DL, TII->get(SASS::WARPGROUP_DEPBAR)).addImm(0));
      S.Groups.clear();
    }
    S.Dirty |= Accessed;
  }
  return S;
}

void SASSWGMMAHazards::checkSetMaxReg(MachineFunction &MF) {
  unsigned Entry = SASS::setMaxRegEntryCount(MF);
  if (!Entry)
    return;
  auto In = regLimitsIn(MF, Entry);
  for (MachineBasicBlock &B : MF) {
    unsigned L = In.lookup(&B);
    for (MachineInstr &MI : B) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(MO.getReg()))
          for (MCRegUnitRootIterator R(U, TRI); R.isValid(); ++R) {
            if (!SASS::GPR32RegClass.contains(*R) || *R == SASS::RZ ||
                TRI->getEncodingValue(*R) + 3 <= L)
              continue;
            std::string S;
            raw_string_ostream OS(S);
            MI.print(OS, /*IsStandalone=*/true, /*SkipOpers=*/false,
                     /*SkipDebugLoc=*/true);
            report_fatal_error("SASS: " + MF.getName() + ": R" +
                               Twine(TRI->getEncodingValue(*R)) +
                               " is used where setmaxnreg limits the warp to " +
                               Twine(L) + " registers (the top two are reserved): " + S);
          }
      }
      L = limitAfter(MI, L);
    }
  }
}

bool SASSWGMMAHazards::runOnMachineFunction(MachineFunction &MF) {
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget<SASSSubtarget>().getInstrInfo();
  checkSetMaxReg(MF);

  // gsb0 on the last GMMA of each commit group.
  bool Any = false, Serialize = false;
  for (MachineBasicBlock &B : MF) {
    MachineInstr *Last = nullptr;
    for (MachineInstr &MI : make_early_inc_range(B)) {
      if (isGMMA(MI.getOpcode())) {
        Last = &MI;
        Any = true;
      } else if (MI.getOpcode() == SASS::WGMMA_COMMIT) {
        if (Last)
          Last->getOperand(gsbOperand(*Last)).setImm(1);
        else
          Serialize = true;
        Last = nullptr;
        MI.eraseFromParent();
        Any = true;
      } else if (MI.getOpcode() == SASS::WARPGROUP_DEPBAR) {
        // The A fragment uses from SASSExpandWGMMA served the allocator.
        while (MI.getNumOperands() > MI.getNumExplicitOperands())
          MI.removeOperand(MI.getNumOperands() - 1);
      }
    }
  }
  if (!Any)
    return false;

  unsigned NumUnits = TRI->getNumRegUnits();
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  DenseMap<const MachineBasicBlock *, HState> In, Out;
  for (;;) {
    if (Serialize)
      for (MachineBasicBlock &B : MF)
        for (MachineInstr &MI : B) {
          if (isGMMA(MI.getOpcode()))
            MI.getOperand(gsbOperand(MI)).setImm(1);
          else if (MI.getOpcode() == SASS::WARPGROUP_DEPBAR)
            MI.getOperand(0).setImm(0);
        }
    NeedSerialize = false;
    In.clear();
    Out.clear();
    bool Changed = true;
    for (unsigned Iter = 0; Changed && Iter < 64; ++Iter) {
      Changed = false;
      for (MachineBasicBlock *B : RPOT) {
        HState Entry;
        if (B == &MF.front()) {
          Entry.Valid = true;
          Entry.Open = GRegs(NumUnits);
          Entry.Dirty.resize(NumUnits);
        }
        for (MachineBasicBlock *P : B->predecessors())
          if (auto It = Out.find(P); It != Out.end())
            Entry.join(It->second);
        if (!Entry.Valid)
          continue;
        HState O = process(*B, Entry, false);
        In[B] = std::move(Entry);
        auto [It, New] = Out.try_emplace(B, O);
        if (New || !(It->second == O)) {
          It->second = std::move(O);
          Changed = true;
        }
      }
    }
    if (Changed)
      NeedSerialize = true; // No fixpoint: fall back to the simple model.
    if (NeedSerialize && !Serialize) {
      Serialize = true;
      continue;
    }
    break;
  }
  for (MachineBasicBlock *B : RPOT)
    if (auto It = In.find(B); It != In.end())
      process(*B, It->second, true);
  // wgmma is .sync.aligned, but nothing reconverges the warp after divergent
  // code (a guarded shared-memory fill): WARPSYNC.ALL before each arrive,
  // which precedes every GMMA group (diverged GMMAs fault on the H100).
  for (MachineBasicBlock &B : MF)
    for (MachineInstr &MI : B)
      if (MI.getOpcode() == SASS::WARPGROUP_ARRIVE &&
          (MI.getIterator() == B.begin() ||
           std::prev(MI.getIterator())->getOpcode() != SASS::WARPSYNC_ALL))
        noGuard(BuildMI(B, MI, MI.getDebugLoc(), TII->get(SASS::WARPSYNC_ALL)));
  return true;
}
