//===-- SASSISelLowering.cpp - SASS DAG lowering ---------------------------===//

#include "SASSISelLowering.h"
#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASS.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSSubtarget.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;

SASSTargetLowering::SASSTargetLowering(const TargetMachine &TM,
                                       const SASSSubtarget &ST)
    : TargetLowering(TM, ST) {
  addRegisterClass(MVT::i1, &SASS::PREDRegClass);
  addRegisterClass(MVT::i32, &SASS::GPR32RegClass);
  addRegisterClass(MVT::f32, &SASS::GPR32RegClass);
  addRegisterClass(MVT::i64, &SASS::GPR64RegClass);
  // f64: conversions and DADD/DMUL/DFMA/DSETP (initConvActions).
  addRegisterClass(MVT::f64, &SASS::GPR64RegClass);
  // Halves live in 32-bit registers: scalars in the low 16 bits, packed
  // pairs in the whole register (operation actions: initTypeActions).
  for (MVT VT : {MVT::v2f16, MVT::v2bf16, MVT::f16, MVT::bf16})
    addRegisterClass(VT, &SASS::GPR32RegClass);
  computeRegisterProperties(ST.getRegisterInfo());
  for (MVT VT : {MVT::v2f16, MVT::v2bf16}) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
      setOperationAction(Op, VT, Expand);
    setOperationAction(ISD::BITCAST, VT, Legal);
    setOperationAction({ISD::LOAD, ISD::STORE}, VT, Promote);
    AddPromotedToType(ISD::LOAD, VT, MVT::i32);
    AddPromotedToType(ISD::STORE, VT, MVT::i32);
  }

  setBooleanContents(ZeroOrOneBooleanContent);
  // Without a pre-RA machine scheduler (see SASSSubtarget), the DAG
  // scheduler's order is what the allocator sees: balance latency (e.g. the
  // FMUL feeding each FFMA of an accumulator update) against pressure.
  setSchedulingPreference(Sched::Hybrid);
  setStackPointerRegisterToSaveRestore(SASS::R1);
  // Hopper: a branch on `a && b` stays one branch on the combined predicate
  // (CodeGenPrepare and the DAG builder otherwise split it into two, each a
  // divergent region with its own BSSY/BSYNC), as NVPTX and ptxas.
  if (!ST.hasBlackwellISA())
    setJumpIsExpensive(true);

  for (MVT VT : {MVT::i32, MVT::i64}) {
    setOperationAction({ISD::SDIV, ISD::UDIV, ISD::SREM, ISD::UREM,
                        ISD::SDIVREM, ISD::UDIVREM, ISD::SMUL_LOHI,
                        ISD::UMUL_LOHI, ISD::BSWAP, ISD::CTPOP, ISD::CTLZ,
                        ISD::CTTZ, ISD::BITREVERSE, ISD::BR_CC,
                        ISD::SELECT_CC, ISD::ABS, ISD::SADDO, ISD::UADDO,
                        ISD::SSUBO, ISD::USUBO, ISD::SMULO, ISD::UMULO,
                        ISD::UADDO_CARRY, ISD::USUBO_CARRY,
                        ISD::SADDO_CARRY, ISD::SSUBO_CARRY},
                       VT, Expand);
  }
  // PRMT, POPC, FLO, BREV, IABS (SASSInstrBits.td); CTTZ expands through
  // CTPOP.
  setOperationAction({ISD::BSWAP, ISD::CTPOP, ISD::CTLZ, ISD::BITREVERSE,
                      ISD::ABS},
                     MVT::i32, Legal);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction({ISD::FSHL, ISD::FSHR}, MVT::i32, Custom);
  setOperationAction({ISD::MULHU, ISD::MULHS, ISD::ROTL, ISD::ROTR, ISD::FSHL,
                      ISD::FSHR, ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX},
                     MVT::i64, Expand);
  setOperationAction({ISD::SETCC, ISD::MUL, ISD::SHL, ISD::SRL, ISD::SRA},
                     MVT::i64, Custom);
  for (MVT VT : {MVT::i1, MVT::i8, MVT::i16})
    setOperationAction(ISD::SIGN_EXTEND_INREG, VT, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);

  // i1 values live in predicates; anything beyond logic ops goes via i32.
  setOperationAction({ISD::SELECT, ISD::SETCC}, MVT::i1, Promote);
  AddPromotedToType(ISD::SELECT, MVT::i1, MVT::i32);
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);
  setOperationAction({ISD::BR_CC, ISD::SELECT_CC}, MVT::f32, Expand);
  setOperationAction({ISD::BR_JT, ISD::BRIND}, MVT::Other, Expand);

  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction({ISD::FMINNUM, ISD::FMAXNUM}, MVT::f32, Legal);
  // afn tanh is MUFU.TANH (SASSInstrMath.td); the other approximate math is
  // lowered to NVVM approx intrinsics by SASSLowerMath.
  setOperationAction(ISD::FTANH, MVT::f32, Legal);
  setOperationAction({ISD::FCOPYSIGN, ISD::FMINIMUM, ISD::FMAXIMUM,
                      ISD::FMINIMUMNUM, ISD::FMAXIMUMNUM},
                     MVT::f32, Expand);

  for (MVT VT : {MVT::i32, MVT::i64})
    for (auto Ext : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD})
      setLoadExtAction(Ext, VT, MVT::i1, Promote);
  for (MVT MemVT : {MVT::i8, MVT::i16, MVT::i32})
    for (auto Ext : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD})
      setLoadExtAction(Ext, MVT::i64, MemVT, Expand);
  for (MVT MemVT : {MVT::i8, MVT::i16, MVT::i32})
    setTruncStoreAction(MVT::i64, MemVT, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);

  setOperationAction(ISD::GlobalAddress, {MVT::i32, MVT::i64}, Custom);
  setOperationAction(ISD::ADDRSPACECAST, {MVT::i32, MVT::i64}, Custom);
  for (MVT VT : {MVT::v2f32, MVT::v2i32, MVT::v4f32, MVT::v4i32})
    setOperationAction({ISD::LOAD, ISD::STORE}, VT, Custom);

  setTargetDAGCombine({ISD::ADD, ISD::LOAD, ISD::BR});
  initTypeActions();
  initPTXActions();
  initConvActions();
}

/// Whether kernel \p F refers to \p G (device functions are inlined into
/// their kernels by then, so every use is in a kernel or a constant).
static bool usedBy(const Value &G, const Function &F) {
  for (const User *U : G.users()) {
    if (const auto *I = dyn_cast<Instruction>(U)) {
      if (I->getFunction() == &F)
        return true;
    } else if (isa<Constant>(U) && usedBy(*U, F)) {
      return true;
    }
  }
  return false;
}

/// Walks the static shared variables kernel \p F uses, in module order;
/// returns the offset of \p Stop, or the total size when \p Stop is null.
/// Every extern __shared__ array is the dynamic window right after the
/// static variables, aligned like ptxas to at least 16 bytes and to the
/// strictest extern declaration; when there is one, the total is padded to
/// that base.
static uint64_t sharedLayout(const Function &F, const GlobalVariable *Stop) {
  const Module &M = *F.getParent();
  const DataLayout &DL = M.getDataLayout();
  uint64_t Off = 0;
  std::optional<Align> DynAlign;
  for (const GlobalVariable &G : M.globals()) {
    if (G.getAddressSpace() != 3 || (&G != Stop && !usedBy(G, F)))
      continue;
    Align A = G.getAlign().value_or(DL.getPrefTypeAlign(G.getValueType()));
    if (G.isDeclaration()) {
      DynAlign = std::max(DynAlign.value_or(Align(16)), A);
      continue;
    }
    uint64_t Size = DL.getTypeAllocSize(G.getValueType());
    if (Size == 0)
      report_fatal_error("SASS: zero-sized __shared__ variable: " + G.getName());
    Off = alignTo(Off, A);
    if (&G == Stop)
      return Off;
    Off += Size;
  }
  if (DynAlign)
    Off = alignTo(Off, *DynAlign);
  assert((!Stop || Stop->isDeclaration()) && "shared variable not in its module");
  return Off;
}

bool SASS::hasDynamicShared(const Function &F) {
  return any_of(F.getParent()->globals(), [&](const GlobalVariable &G) {
    return G.getAddressSpace() == 3 && G.isDeclaration() && usedBy(G, F);
  });
}

uint64_t SASS::sharedOffset(const GlobalVariable &GV, const Function &F) {
  return sharedLayout(F, &GV);
}

uint64_t SASS::sharedBytes(const Function &F) {
  return sharedLayout(F, nullptr);
}

/// Vector accesses done as one LDG/LDS/LDL/LD.64/.128: 2 or 4 x 32-bit elements,
/// naturally aligned.
static bool isWideAccess(EVT VT, Align A, unsigned AS) {
  if (!VT.isSimple() || (AS != 0 && AS != 1 && AS != 3 && AS != 5))
    return false;
  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::v2f32:
  case MVT::v2i32:
  case MVT::v4f32:
  case MVT::v4i32:
    return A.value() >= VT.getStoreSize();
  default:
    return false;
  }
}

void SASSTargetLowering::ReplaceNodeResults(SDNode *N,
                                            SmallVectorImpl<SDValue> &Results,
                                            SelectionDAG &DAG) const {
  if (replaceTypesResults(N, Results, DAG) || replacePTXResults(N, Results, DAG))
    return;
  auto *LD = dyn_cast<LoadSDNode>(N);
  if (!LD || LD->getExtensionType() != ISD::NON_EXTLOAD || !LD->isUnindexed())
    return;
  EVT VT = LD->getValueType(0);
  if (!isWideAccess(VT, LD->getAlign(), LD->getAddressSpace()))
    return;
  SDLoc DL(N);
  unsigned NumElts = VT.getVectorNumElements();
  SmallVector<EVT, 5> VTs(NumElts, VT.getVectorElementType());
  VTs.push_back(MVT::Other);
  SDValue Ld = DAG.getMemIntrinsicNode(
      NumElts == 4 ? SASSISD::LDV4 : SASSISD::LDV2, DL, DAG.getVTList(VTs),
      {LD->getChain(), LD->getBasePtr()}, LD->getMemoryVT(),
      LD->getMemOperand());
  SmallVector<SDValue, 4> Elts;
  for (unsigned I = 0; I < NumElts; ++I)
    Elts.push_back(Ld.getValue(I));
  Results.push_back(DAG.getBuildVector(VT, DL, Elts));
  Results.push_back(Ld.getValue(NumElts));
}

SDValue SASSTargetLowering::lowerVectorStore(SDValue Op,
                                             SelectionDAG &DAG) const {
  auto *ST = cast<StoreSDNode>(Op);
  SDValue V = ST->getValue();
  EVT VT = V.getValueType();
  if (ST->isTruncatingStore() || !ST->isUnindexed() ||
      !isWideAccess(VT, ST->getAlign(), ST->getAddressSpace()))
    return SDValue(); // Default: split into scalar stores.
  SDLoc DL(Op);
  unsigned NumElts = VT.getVectorNumElements();
  SmallVector<SDValue, 6> Ops = {ST->getChain()};
  for (unsigned I = 0; I < NumElts; ++I)
    Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL,
                              VT.getVectorElementType(), V,
                              DAG.getVectorIdxConstant(I, DL)));
  Ops.push_back(ST->getBasePtr());
  return DAG.getMemIntrinsicNode(NumElts == 4 ? SASSISD::STV4 : SASSISD::STV2,
                                 DL, DAG.getVTList(MVT::Other), Ops,
                                 ST->getMemoryVT(), ST->getMemOperand());
}

SDValue SASSTargetLowering::lowerGlobalAddress(SDValue Op,
                                               SelectionDAG &DAG) const {
  auto *GA = cast<GlobalAddressSDNode>(Op);
  auto *GV = dyn_cast<GlobalVariable>(GA->getGlobal());
  unsigned AS = GA->getAddressSpace();
  if (!GV || (AS != 3 && AS != 4) || Op.getValueType() != MVT::i32)
    report_fatal_error(
        Twine("SASS: only __shared__ and __constant__ variables (with "
              "-fcuda-short-ptr) are supported, not ") +
        (AS == 1 ? "__device__ variable " : "") + GA->getGlobal()->getName());
  SDLoc DL(Op);
  // Constant bank addresses are plain offsets; the bank an access reads is
  // recovered from its memory operand (SASS::constBank).
  if (AS == 4) {
    uint64_t Off = GV->getName() == SASS::KernelParamsSymbol
                       ? 0
                       : SASS::constOffset(*GV);
    return DAG.getConstant(Off + GA->getOffset(), DL, MVT::i32);
  }
  SDValue Base = DAG.getNode(SASSISD::SMEM_BASE, DL, MVT::i32);
  return DAG.getNode(
      ISD::ADD, DL, MVT::i32, Base,
      DAG.getConstant(SASS::sharedOffset(*GV, DAG.getMachineFunction().getFunction()) +
                          GA->getOffset(),
                      DL, MVT::i32));
}

SDValue SASSTargetLowering::lowerConstAddrSpaceCast(SDValue Op,
                                                    SelectionDAG &DAG) const {
  auto *ASC = cast<AddrSpaceCastSDNode>(Op);
  unsigned Src = ASC->getSrcAddressSpace(), Dst = ASC->getDestAddressSpace();
  if (Src == 4 || Dst == 4)
    report_fatal_error("SASS: generic pointers to __constant__ memory are not "
                       "supported (only direct __constant__ accesses)");
  // As ptxas's cvta.shared / cvta.to.shared: a generic shared address is the
  // 32-bit shared address under the shared window's high word (SR_SWINHI).
  SDLoc DL(Op);
  SDValue Ptr = Op.getOperand(0);
  if (Src == 3 && Dst == 0)
    return DAG.getNode(SASSISD::PAIR, DL, MVT::i64,
                       DAG.getZExtOrTrunc(Ptr, DL, MVT::i32),
                       DAG.getNode(SASSISD::SWINHI, DL, MVT::i32));
  if (Src == 0 && Dst == 3)
    return DAG.getZExtOrTrunc(Ptr, DL, Op.getValueType());
  return Op;
}

/// Matches a 64-bit value that is a widened 32-bit value times a 32-bit
/// factor: (ext X), (shl (ext X), C), (mul (ext X), C), (mul (ext X), (ext Y)).
static bool matchWideProduct(SDValue V, SelectionDAG &DAG, SDValue &X,
                             SDValue &Factor, bool &Signed) {
  SDLoc DL(V);
  auto Widened = [&](SDValue E, SDValue &Narrow, bool &S) {
    if ((E.getOpcode() != ISD::SIGN_EXTEND &&
         E.getOpcode() != ISD::ZERO_EXTEND) ||
        E.getOperand(0).getValueType() != MVT::i32)
      return false;
    Narrow = E.getOperand(0);
    S = E.getOpcode() == ISD::SIGN_EXTEND;
    return true;
  };
  if (Widened(V, X, Signed)) {
    Factor = DAG.getConstant(1, DL, MVT::i32);
    return true;
  }
  if (V.getOpcode() != ISD::SHL && V.getOpcode() != ISD::MUL)
    return false;
  // Shared products are fine: each IMAD.WIDE replaces a 64-bit shift + add.
  // The widened value may come from another block (a CopyFromReg); then
  // its known sign/zero bits say whether it is an extended 32-bit value.
  SDValue Wide = V.getOperand(0);
  if (!Widened(Wide, X, Signed)) {
    if (Wide.getValueType() != MVT::i64 || isa<ConstantSDNode>(Wide))
      return false;
    if (DAG.ComputeNumSignBits(Wide) > 32)
      Signed = true;
    else if (DAG.computeKnownBits(Wide).countMinLeadingZeros() >= 32)
      Signed = false;
    else
      return false;
    X = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, Wide);
  }
  auto *C = dyn_cast<ConstantSDNode>(V.getOperand(1));
  if (V.getOpcode() == ISD::SHL) {
    if (!C || C->getZExtValue() >= 31)
      return false;
    Factor = DAG.getConstant(1u << C->getZExtValue(), DL, MVT::i32);
    return true;
  }
  if (C) {
    int64_t F = C->getSExtValue();
    if (Signed ? !isInt<32>(F) : !isUInt<32>(F))
      return false;
    Factor = DAG.getConstant(F, DL, MVT::i32);
    return true;
  }
  bool OtherSigned;
  return Widened(V.getOperand(1), Factor, OtherSigned) && OtherSigned == Signed;
}

/// The DAG combiner folds `select c, (load a), (load b)` into a load through
/// `select c, a, b` whose memory operand has no IR pointer, so the constant
/// bank it reads (SASS::constBank) is lost. Before legalization the
/// addresses are still GlobalAddress nodes: when they all lie in one bank,
/// point the memory operand at one of them.
static void recoverConstBankPointer(LoadSDNode &LD) {
  MachineMemOperand *MMO = LD.getMemOperand();
  if (LD.getAddressSpace() != 4 || !MMO->getPointerInfo().V.isNull())
    return;
  SmallVector<SDValue, 4> Work = {LD.getBasePtr()};
  const GlobalValue *Leaf = nullptr;
  std::optional<bool> Params;
  while (!Work.empty()) {
    SDValue V = Work.pop_back_val();
    if (V.getOpcode() == ISD::SELECT) {
      Work.append({V.getOperand(1), V.getOperand(2)});
      continue;
    }
    if (V.getOpcode() == ISD::SELECT_CC) {
      Work.append({V.getOperand(2), V.getOperand(3)});
      continue;
    }
    if (V.getOpcode() == ISD::ADD && isa<ConstantSDNode>(V.getOperand(1))) {
      Work.push_back(V.getOperand(0));
      continue;
    }
    auto *GA = dyn_cast<GlobalAddressSDNode>(V);
    if (!GA || GA->getAddressSpace() != 4)
      return;
    bool P = GA->getGlobal()->getName() == SASS::KernelParamsSymbol;
    if (Params && *Params != P)
      return;
    Params = P;
    Leaf = GA->getGlobal();
  }
  MMO->setValue(Leaf);
}

// sm_90: `brcond (setcc p, C, eq|ne), T; br F` on a predicate p (a negated
// mbarrier try_wait result: the generic combiner turns `xor p, -1` into that
// setcc, which i1 promotion selects as SEL + ISETP, ~20 cycles per poll)
// branches on p itself, swapping the targets when the setcc negates it.
static SDValue combineBranchOnPredicate(SDNode *Br, SelectionDAG &DAG) {
  SDValue BC = Br->getOperand(0);
  if (DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA() ||
      BC.getOpcode() != ISD::BRCOND || !BC.hasOneUse())
    return SDValue();
  SDValue Cond = BC.getOperand(1);
  if (Cond.getOpcode() != ISD::SETCC ||
      Cond.getOperand(0).getValueType() != MVT::i1)
    return SDValue();
  auto *C = dyn_cast<ConstantSDNode>(Cond.getOperand(1));
  ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
  if (!C || (CC != ISD::SETEQ && CC != ISD::SETNE))
    return SDValue();
  bool Negated = (CC == ISD::SETNE) == bool(C->getZExtValue() & 1);
  SDValue T = BC.getOperand(2), F = Br->getOperand(1);
  if (Negated)
    std::swap(T, F);
  SDLoc DL(Br);
  SDValue NewBC = DAG.getNode(ISD::BRCOND, DL, MVT::Other, BC.getOperand(0),
                              Cond.getOperand(0), T);
  return DAG.getNode(ISD::BR, DL, MVT::Other, NewBC, F);
}

// Address arithmetic `base + (i64)idx * scale` is one IMAD.WIDE; forming it
// before legalization keeps the 64-bit shift/mul from being split.
SDValue SASSTargetLowering::PerformDAGCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  if (auto *LD = dyn_cast<LoadSDNode>(N)) {
    recoverConstBankPointer(*LD);
    return SDValue();
  }
  if (N->getOpcode() == ISD::BR)
    return combineBranchOnPredicate(N, DCI.DAG);
  if (N->getOpcode() != ISD::ADD || N->getValueType(0) != MVT::i64)
    return SDValue();
  SelectionDAG &DAG = DCI.DAG;
  for (unsigned I = 0; I < 2; ++I) {
    SDValue X, Factor;
    bool Signed;
    if (matchWideProduct(N->getOperand(I), DAG, X, Factor, Signed))
      return DAG.getNode(Signed ? SASSISD::MAD_WIDE : SASSISD::MAD_WIDE_U,
                         SDLoc(N), MVT::i64, X, Factor, N->getOperand(1 - I));
  }
  // Hopper, as ptxas: base + X * C with a full 64-bit X and a 32-bit
  // constant C is IMAD.WIDE.U32 lo(X), C, base plus hi(X) * C into the high
  // word, not a 64-bit multiply and a 64-bit add.
  if (DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA())
    return SDValue();
  for (unsigned I = 0; I < 2; ++I) {
    SDValue M = N->getOperand(I);
    auto *C = M.getOpcode() == ISD::MUL && M.hasOneUse()
                  ? dyn_cast<ConstantSDNode>(M.getOperand(1))
                  : nullptr;
    if (!C || !isUInt<32>(C->getZExtValue()))
      continue;
    SDLoc DL(N);
    SDValue X = M.getOperand(0);
    SDValue CL = DAG.getConstant(C->getZExtValue(), DL, MVT::i32);
    auto Half = [&](unsigned Opc, SDValue V) {
      return DAG.getNode(Opc, DL, MVT::i32, V);
    };
    SDValue Wide = DAG.getNode(SASSISD::MAD_WIDE_U, DL, MVT::i64,
                               Half(SASSISD::LO, X), CL, N->getOperand(1 - I));
    SDValue Hi = DAG.getNode(
        ISD::ADD, DL, MVT::i32, Half(SASSISD::HI, Wide),
        DAG.getNode(ISD::MUL, DL, MVT::i32, Half(SASSISD::HI, X), CL));
    return DAG.getNode(SASSISD::PAIR, DL, MVT::i64, Half(SASSISD::LO, Wide), Hi);
  }
  return SDValue();
}

MachineBasicBlock *
SASSTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                MachineBasicBlock *MBB) const {
  if (MachineBasicBlock *Next = SASS::expandPTXUniform(MI, MBB))
    return Next;
  const auto &ST = MBB->getParent()->getSubtarget<SASSSubtarget>();
  if (!ST.hasBlackwellISA())
    ST.getInstrInfo()->expandAdd64(MI);
  return MBB;
}

void SASSTargetLowering::getTgtMemIntrinsic(
    SmallVectorImpl<IntrinsicInfo> &Infos, const CallBase &I,
    MachineFunction &MF, unsigned Intrinsic) const {
  IntrinsicInfo Info;
  Info.offset = 0;
  Info.align = Align(16);
  switch (Intrinsic) {
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16:
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getIntegerVT(I.getType()->isStructTy()
                                       ? I.getType()->getStructNumElements() * 32
                                       : 32);
    Info.flags = MachineMemOperand::MOLoad;
    break;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_trans_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16:
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16:
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::getIntegerVT((I.arg_size() - 1) * 32);
    Info.flags = MachineMemOperand::MOStore;
    break;
  case Intrinsic::nvvm_cp_async_cg_shared_global_16:
  case Intrinsic::nvvm_cp_async_ca_shared_global_16:
  case Intrinsic::nvvm_cp_async_ca_shared_global_8:
  case Intrinsic::nvvm_cp_async_ca_shared_global_4:
  case Intrinsic::nvvm_cp_async_cg_shared_global_16_s:
  case Intrinsic::nvvm_cp_async_ca_shared_global_16_s:
  case Intrinsic::nvvm_cp_async_ca_shared_global_8_s:
  case Intrinsic::nvvm_cp_async_ca_shared_global_4_s:
    // The shared-memory write: ordering against other shared accesses
    // comes from the explicit commit/wait (DEPBAR) the program issues.
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::i128;
    Info.flags = MachineMemOperand::MOStore | MachineMemOperand::MOLoad;
    break;
  default:
    return;
  }
  Info.ptrVal = I.getArgOperand(0); // The shared-memory address.
  Infos.push_back(Info);
}

MachineMemOperand::Flags
SASSTargetLowering::getTargetMMOFlags(const Instruction &I) const {
  const MDNode *MD = I.getMetadata(SASS::LoadHintMD);
  if (!MD)
    return MachineMemOperand::MONone;
  return SASS::loadHintToMMOFlags(
      mdconst::extract<ConstantInt>(MD->getOperand(0))->getZExtValue());
}

/// A wide access's 32-bit-element type (isWideAccess) re-typed to a vector of
/// narrower elements: the type legalizer would split that into 32-bit pieces.
static bool narrowsWideAccess(EVT VT, EVT NewVT) {
  return VT.isSimple() && NewVT.isVector() &&
         NewVT.getScalarSizeInBits() < 32 &&
         is_contained({MVT::v2f32, MVT::v2i32, MVT::v4f32, MVT::v4i32},
                      VT.getSimpleVT().SimpleTy);
}

bool SASSTargetLowering::isLoadBitCastBeneficial(
    EVT LoadVT, EVT BitcastVT, const SelectionDAG &DAG,
    const MachineMemOperand &MMO) const {
  if (!DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA() &&
      narrowsWideAccess(LoadVT, BitcastVT))
    return false;
  return TargetLowering::isLoadBitCastBeneficial(LoadVT, BitcastVT, DAG, MMO);
}

bool SASSTargetLowering::isStoreBitCastBeneficial(
    EVT StoreVT, EVT BitcastVT, const SelectionDAG &DAG,
    const MachineMemOperand &MMO) const {
  if (!DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA() &&
      narrowsWideAccess(StoreVT, BitcastVT))
    return false;
  return TargetLowering::isStoreBitCastBeneficial(StoreVT, BitcastVT, DAG, MMO);
}

bool SASSTargetLowering::isSDNodeAlwaysUniform(const SDNode *N) const {
  unsigned Base = N->getOpcode() == ISD::INTRINSIC_W_CHAIN ? 1 : 0;
  if (N->getOpcode() != ISD::INTRINSIC_WO_CHAIN && N->getOpcode() != ISD::INTRINSIC_W_CHAIN)
    return false;
  uint64_t ID = N->getConstantOperandVal(Base);
  if (ID != Intrinsic::nvvm_shfl_sync_idx_i32 && ID != Intrinsic::nvvm_shfl_sync_idx_f32)
    return false;
  // (mask, value, lane, clamp)
  const auto *M = dyn_cast<ConstantSDNode>(N->getOperand(Base + 1));
  const auto *L = dyn_cast<ConstantSDNode>(N->getOperand(Base + 3));
  const auto *C = dyn_cast<ConstantSDNode>(N->getOperand(Base + 4));
  return M && M->isAllOnes() && L && L->getZExtValue() < 32 && C &&
         C->getZExtValue() == 0x1f;
}

// Hopper: nodes that start divergence, matching SASSTTIImpl::getValueUniformity
// for IR values (other blocks' values through their virtual registers). Only
// the uniform-operand selection of SASSISelPTX reads SDNode divergence.
bool SASSTargetLowering::isSDNodeSourceOfDivergence(
    const SDNode *N, FunctionLoweringInfo *FLI, UniformityInfo *UA) const {
  if (!FLI || FLI->MF->getSubtarget<SASSSubtarget>().hasBlackwellISA() ||
      isSDNodeAlwaysUniform(N))
    return false;
  switch (N->getOpcode()) {
  case ISD::CopyFromReg: {
    Register R = cast<RegisterSDNode>(N->getOperand(1))->getReg();
    // (FunctionLoweringInfo::getValueFromVirtualReg freezes its map, and
    // instruction selection still creates value registers afterwards.)
    if (RegValueMF != FLI->MF || RegValueSize != FLI->ValueMap.size()) {
      RegValue.clear();
      for (auto [Val, Reg] : FLI->ValueMap)
        RegValue[Reg] = Val;
      RegValueMF = FLI->MF;
      RegValueSize = FLI->ValueMap.size();
    }
    const Value *V = R.isVirtual() ? RegValue.lookup(R) : nullptr;
    if (!V || !UA || UA->isDivergentAtDef(V))
      return true;
    // A value leaving a loop with a divergent exit is divergent at its uses.
    const BasicBlock *BB = FLI->MBB ? FLI->MBB->getBasicBlock() : nullptr;
    for (const Use &U : V->uses()) {
      const auto *I = dyn_cast<Instruction>(U.getUser());
      if ((!BB || (I && I->getParent() == BB)) && UA->isDivergentAtUse(U))
        return true;
    }
    return false;
  }
  case ISD::LOAD: {
    unsigned AS = cast<LoadSDNode>(N)->getAddressSpace();
    return AS == 0 || AS == 5; // generic (may be local) or local
  }
  case ISD::INTRINSIC_WO_CHAIN:
    switch (N->getConstantOperandVal(0)) {
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
      return false;
    }
  case ISD::INTRINSIC_W_CHAIN: // memory, state, collectives
  case ISD::INLINEASM:
  case ISD::INLINEASM_BR:
    return true;
  default:
    return isa<AtomicSDNode>(N);
  }
}

EVT SASSTargetLowering::getSetCCResultType(const DataLayout &, LLVMContext &Ctx,
                                           EVT VT) const {
  if (VT.isVector())
    return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
  return MVT::i1;
}

bool SASSTargetLowering::isLegalAddressingMode(const DataLayout &,
                                               const AddrMode &AM, Type *,
                                               unsigned, Instruction *) const {
  if (AM.BaseGV || !isInt<24>(AM.BaseOffs))
    return false;
  switch (AM.Scale) {
  case 0:
    return true;
  case 1:
    return !AM.HasBaseReg;
  default:
    return false;
  }
}

/// Addresses of the memory intrinsics that select to reg + offset forms
/// (LDGSTS, a small memset's stores): CodeGenPrepare then sinks an address
/// computed in another block (clang hoists a tile's shared-memory slot
/// addresses to the kernel entry) into the user, where the offset folds,
/// instead of each slot address holding a register across the kernel.
bool SASSTargetLowering::getAddrModeArguments(const IntrinsicInst *II,
                                              SmallVectorImpl<Value *> &Ops,
                                              Type *&AccessTy) const {
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_cp_async_ca_shared_global_4:
  case Intrinsic::nvvm_cp_async_ca_shared_global_4_s:
  case Intrinsic::nvvm_cp_async_ca_shared_global_8:
  case Intrinsic::nvvm_cp_async_ca_shared_global_8_s:
  case Intrinsic::nvvm_cp_async_ca_shared_global_16:
  case Intrinsic::nvvm_cp_async_ca_shared_global_16_s:
  case Intrinsic::nvvm_cp_async_cg_shared_global_16:
  case Intrinsic::nvvm_cp_async_cg_shared_global_16_s:
    Ops.push_back(II->getArgOperand(0));
    Ops.push_back(II->getArgOperand(1));
    break;
  case Intrinsic::memset:
    Ops.push_back(II->getArgOperand(0));
    break;
  default:
    return false;
  }
  AccessTy = Type::getInt8Ty(II->getContext());
  return true;
}

//===----------------------------------------------------------------------===//
// Calling convention: kernels only, parameters in constant bank 0.
//===----------------------------------------------------------------------===//

SDValue SASSTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const Function &F = MF.getFunction();
  if (CallConv != CallingConv::PTX_Kernel)
    report_fatal_error("SASS: only kernels are supported; device function '" +
                       F.getName() + "' must be inlined");

  // Parameter offsets in constant bank 0 (SASS::paramLayout); the sass-dec
  // assembler lays out the printed `#fn` parameters the same way.
  auto *FI = MF.getInfo<SASSFunctionInfo>();
  unsigned ParamBase = DAG.getSubtarget<SASSSubtarget>().paramBase();
  SmallVector<SASS::ParamSlot, 8> Slots = SASS::paramLayout(F, ParamBase);
  for (const SASS::ParamSlot &S : Slots) {
    FI->ParamSizes.push_back(S.Size);
    FI->ParamAligns.push_back(S.Explicit ? S.Align : 0);
  }

  for (const ISD::InputArg &In : Ins) {
    // SASSLowerKernelArgs rewrote every access to a by-value parameter as a
    // constant bank 0 load; the pointer itself is dead.
    if (In.Flags.isByVal()) {
      InVals.push_back(DAG.getUNDEF(In.VT));
      continue;
    }
    unsigned CBOff =
        ParamBase + Slots[In.getOrigArgIndex()].Offset + In.PartOffset;
    SDValue OffV = DAG.getTargetConstant(CBOff, DL, MVT::i32);
    EVT ArgVT = In.ArgVT;
    if (In.VT == MVT::i64 || In.VT == MVT::f32 || In.VT == MVT::f64) {
      InVals.push_back(DAG.getNode(SASSISD::LDC, DL, In.VT, OffV));
      continue;
    }
    // Narrow integers occupy the low bytes of a 4-byte slot whose upper
    // bytes are whatever the host passed: extend explicitly.
    SDValue V = DAG.getNode(SASSISD::LDC, DL, MVT::i32, OffV);
    if (ArgVT.getSizeInBits() < 32) {
      EVT NarrowVT = ArgVT == MVT::i1 ? EVT(MVT::i8) : ArgVT;
      V = In.Flags.isSExt()
              ? DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32, V,
                            DAG.getValueType(NarrowVT))
              : DAG.getZeroExtendInReg(V, DL, NarrowVT);
    }
    if (In.VT == MVT::i1)
      V = DAG.getSetCC(DL, MVT::i1, V, DAG.getConstant(0, DL, MVT::i32),
                       ISD::SETNE);
    else if (In.VT != MVT::i32)
      report_fatal_error("SASS: unsupported kernel parameter type");
    InVals.push_back(V);
  }
  return Chain;
}

SDValue SASSTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID, bool, const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &, const SDLoc &DL, SelectionDAG &DAG) const {
  if (!Outs.empty())
    report_fatal_error("SASS: kernels return void");
  return DAG.getNode(SASSISD::RET, DL, MVT::Other, Chain);
}

SDValue SASSTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                      SmallVectorImpl<SDValue> &InVals) const {
  if (SDValue Chain = lowerBuiltinCall(CLI, InVals))
    return Chain;
  std::string Callee = "<indirect>";
  if (auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee))
    Callee = G->getGlobal()->getName().str();
  else if (auto *E = dyn_cast<ExternalSymbolSDNode>(CLI.Callee))
    Callee = E->getSymbol();
  report_fatal_error("SASS: calls are not supported (callee " + Twine(Callee) +
                     ")");
}

//===----------------------------------------------------------------------===//
// Custom lowering: 64-bit operations split into 32-bit halves.
//===----------------------------------------------------------------------===//

SDValue SASSTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::SETCC:
    return lowerSETCC64(Op, DAG);
  case ISD::MUL:
    return lowerMUL64(Op, DAG);
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    return lowerShift64(Op, DAG);
  case ISD::FSHL:
  case ISD::FSHR:
    return lowerFunnelShift(Op, DAG);
  case ISD::GlobalAddress:
    return lowerGlobalAddress(Op, DAG);
  case ISD::ADDRSPACECAST: {
    auto *ASC = cast<AddrSpaceCastSDNode>(Op);
    if (ASC->getSrcAddressSpace() == 5 || ASC->getDestAddressSpace() == 5)
      return lowerLocalAddrSpaceCast(Op, DAG);
    return lowerConstAddrSpaceCast(Op, DAG);
  }
  case ISD::STORE:
    return lowerVectorStore(Op, DAG);
  default:
    if (SDValue V = lowerPTXOp(Op, DAG))
      return V;
    return lowerTypesOp(Op, DAG);
  }
}

static SDValue lo32(SDValue V, const SDLoc &DL, SelectionDAG &DAG) {
  return DAG.getNode(SASSISD::LO, DL, MVT::i32, V);
}
static SDValue hi32(SDValue V, const SDLoc &DL, SelectionDAG &DAG) {
  return DAG.getNode(SASSISD::HI, DL, MVT::i32, V);
}
static SDValue pair(SDValue Lo, SDValue Hi, const SDLoc &DL, SelectionDAG &DAG) {
  return DAG.getNode(SASSISD::PAIR, DL, MVT::i64, Lo, Hi);
}

// 64-bit compares: an unsigned compare of the low words feeds ISETP.*.EX on
// the high words, which takes the low result as carry-in.
SDValue SASSTargetLowering::lowerSETCC64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue A = Op.getOperand(0), B = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  unsigned Rel;
  bool Signed = false;
  switch (CC) {
  case ISD::SETEQ: Rel = 4; break;
  case ISD::SETNE: Rel = 5; break;
  case ISD::SETLT: Rel = 0; Signed = true; break;
  case ISD::SETLE: Rel = 1; Signed = true; break;
  case ISD::SETGT: Rel = 2; Signed = true; break;
  case ISD::SETGE: Rel = 3; Signed = true; break;
  case ISD::SETULT: Rel = 0; break;
  case ISD::SETULE: Rel = 1; break;
  case ISD::SETUGT: Rel = 2; break;
  case ISD::SETUGE: Rel = 3; break;
  default:
    llvm_unreachable("unexpected integer condition");
  }
  static const ISD::CondCode LoCC[] = {ISD::SETULT, ISD::SETULE, ISD::SETUGT,
                                       ISD::SETUGE, ISD::SETEQ,  ISD::SETNE};
  SDValue Lo = DAG.getSetCC(DL, MVT::i1, lo32(A, DL, DAG), lo32(B, DL, DAG),
                            LoCC[Rel]);
  unsigned HiCC = Rel | (Signed ? 0 : 8);
  return DAG.getNode(SASSISD::SETCC_EX, DL, MVT::i1, hi32(A, DL, DAG),
                     hi32(B, DL, DAG), DAG.getTargetConstant(HiCC, DL, MVT::i32),
                     Lo);
}

SDValue SASSTargetLowering::lowerMUL64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue A = Op.getOperand(0), B = Op.getOperand(1);
  auto IsExt = [](SDValue V, unsigned Opc) {
    return V.getOpcode() == Opc && V.getOperand(0).getValueType() == MVT::i32;
  };
  // Widening 32x32 multiplies select IMAD.WIDE(.U32) directly.
  if ((IsExt(A, ISD::ZERO_EXTEND) && IsExt(B, ISD::ZERO_EXTEND)) ||
      (IsExt(A, ISD::SIGN_EXTEND) && IsExt(B, ISD::SIGN_EXTEND)))
    return Op;
  SDValue ALo = lo32(A, DL, DAG), AHi = hi32(A, DL, DAG);
  SDValue BLo = lo32(B, DL, DAG), BHi = hi32(B, DL, DAG);
  SDValue Wide = DAG.getNode(ISD::MUL, DL, MVT::i64,
                             DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, ALo),
                             DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, BLo));
  SDValue Cross = DAG.getNode(ISD::ADD, DL, MVT::i32,
                              DAG.getNode(ISD::MUL, DL, MVT::i32, ALo, BHi),
                              DAG.getNode(ISD::MUL, DL, MVT::i32, AHi, BLo));
  SDValue Hi = DAG.getNode(ISD::ADD, DL, MVT::i32, hi32(Wide, DL, DAG), Cross);
  return pair(lo32(Wide, DL, DAG), Hi, DL, DAG);
}

SDValue SASSTargetLowering::lowerShift64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  unsigned Opc = Op.getOpcode();
  SDValue A = Op.getOperand(0), Amt = Op.getOperand(1);
  SDValue Lo = lo32(A, DL, DAG), Hi = hi32(A, DL, DAG);
  auto I32 = [&](unsigned O, SDValue X, uint64_t C) {
    return DAG.getNode(O, DL, MVT::i32, X, DAG.getConstant(C, DL, MVT::i32));
  };
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);

  if (auto *C = dyn_cast<ConstantSDNode>(Amt)) {
    uint64_t S = C->getZExtValue() & 63;
    if (S == 0)
      return A;
    if (Opc == ISD::SHL) {
      if (S < 32)
        return pair(I32(ISD::SHL, Lo, S),
                    DAG.getNode(ISD::FSHL, DL, MVT::i32, Hi, Lo,
                                DAG.getConstant(S, DL, MVT::i32)),
                    DL, DAG);
      return pair(Zero, S == 32 ? Lo : I32(ISD::SHL, Lo, S - 32), DL, DAG);
    }
    // The low word of a 64-bit right shift is the same funnel for SRL and SRA.
    SDValue NewHi = Opc == ISD::SRL ? Zero : I32(ISD::SRA, Hi, 31);
    if (S < 32)
      return pair(DAG.getNode(ISD::FSHR, DL, MVT::i32, Hi, Lo,
                              DAG.getConstant(S, DL, MVT::i32)),
                  I32(Opc, Hi, S), DL, DAG);
    return pair(S == 32 ? Hi : I32(Opc, Hi, S - 32), NewHi, DL, DAG);
  }

  // sm_90: a register amount takes the clamping funnels, as ptxas's
  // shl.b64 (SHF.L.U32 lo; SHF.L.U64.HI hi) instead of the generic
  // expansion's compare and selects around amount 32.
  if (Opc == ISD::SHL && !DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA()) {
    SDValue S = DAG.getZExtOrTrunc(Amt, DL, MVT::i32);
    SDValue G = DAG.getRegister(SASS::PT, MVT::i1);
    SDValue G0 = DAG.getTargetConstant(0, DL, MVT::i32);
    SDValue NewLo(DAG.getMachineNode(SASS::SHLrr, DL, MVT::i32, {Lo, S, G, G0}), 0);
    SDValue NewHi(DAG.getMachineNode(SASS::SHLU64HIrr, DL, MVT::i32, {Lo, S, Hi, G, G0}), 0);
    return pair(NewLo, NewHi, DL, DAG);
  }
  unsigned PartsOpc = Opc == ISD::SHL   ? ISD::SHL_PARTS
                      : Opc == ISD::SRL ? ISD::SRL_PARTS
                                        : ISD::SRA_PARTS;
  SDValue Parts = DAG.getNode(PartsOpc, DL, DAG.getVTList(MVT::i32, MVT::i32),
                              Lo, Hi, DAG.getZExtOrTrunc(Amt, DL, MVT::i32));
  SDValue NewLo, NewHi;
  expandShiftParts(Parts.getNode(), NewLo, NewHi, DAG);
  return pair(NewLo, NewHi, DL, DAG);
}

// Immediate funnel shifts are SHF.{L.U64.HI,R.U64}; variable ones expand.
SDValue SASSTargetLowering::lowerFunnelShift(SDValue Op,
                                             SelectionDAG &) const {
  return isa<ConstantSDNode>(Op.getOperand(2)) ? Op : SDValue();
}
