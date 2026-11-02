//===-- SASSISelDAGToDAG.cpp - SASS instruction selection ------------------===//

#include "SASS.h"
#include "SASSSelectionDAGInfo.h"
#include "SASSSubtarget.h"
#include "SASSTargetMachine.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/GlobalVariable.h"

using namespace llvm;

#define DEBUG_TYPE "sass-isel"
#define PASS_NAME "SASS DAG->DAG Pattern Instruction Selection"

namespace {

class SASSDAGToDAGISel : public SelectionDAGISel {
public:
  SASSDAGToDAGISel(SASSTargetMachine &TM, CodeGenOptLevel OL)
      : SelectionDAGISel(TM, OL) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<SASSSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

  void Select(SDNode *N) override;

  /// 64-bit base register pair plus a signed 24-bit offset (LDG/LD/STG/ST).
  bool SelectADDRri(SDValue Addr, SDValue &Base, SDValue &Offset);
  /// 32-bit shared-window address plus a signed 24-bit offset (LDS/STS).
  bool SelectADDRs(SDValue Addr, SDValue &Base, SDValue &Offset);
  /// 32-bit local-window address (or frame index) plus a signed 24-bit
  /// offset (LDL/STL).
  bool SelectADDRl(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool isAddrSpace(const SDNode *N, unsigned AS) const {
    return cast<MemSDNode>(N)->getAddressSpace() == AS;
  }

  /// Global memory that is read-only for the whole kernel, so the
  /// non-coherent path (LDG.E.CONSTANT) is safe.
  bool isNonCoherent(const MemSDNode *N) const;

#include "SASSGenDAGISel.inc"

private:
  /// Read by the generated pattern predicates (HasBlackwellISA, ...).
  const SASSSubtarget *Subtarget = nullptr;

  /// BRCOND of a block whose IR terminator is warp-uniform -> BRAcu.
  /// A BRCOND on a promoted i1 compared with 0/1 -> on the i1 itself.
  SDNode *foldPredicateCompare(SDNode *N);
  bool selectUniformBranch(SDNode *N);
  void selectVectorLoad(SDNode *N);
  void selectVectorStore(SDNode *N);
  /// LDH1/LDH2/LDH4/STH: global/generic accesses with an L2 cache policy.
  void selectPolicyAccess(SDNode *N);
  /// Tensor-core, matrix and async-copy intrinsics; false if \p N is
  /// another intrinsic (left to the generated matcher).
  bool selectIntrinsic(SDNode *N);
  void selectMMA(SDNode *N, unsigned Opc, unsigned ZeroAccOpc);
  void selectMatrixLoad(SDNode *N, unsigned Opc, unsigned Count);
  void selectMatrixStore(SDNode *N, unsigned Opc, unsigned Count);
  void selectAsyncCopy(SDNode *N, unsigned Opc);
  /// REG_SEQUENCE of 2 or 4 32-bit values into a GPR64/GPR128 tuple.
  SDValue tuple(const SDLoc &DL, ArrayRef<SDValue> Vals);
  SDValue guard(const SDLoc &DL) {
    return CurDAG->getRegister(SASS::PT, MVT::i1);
  }
};

class SASSDAGToDAGISelLegacy : public SelectionDAGISelLegacy {
public:
  static char ID;
  SASSDAGToDAGISelLegacy(SASSTargetMachine &TM, CodeGenOptLevel OL)
      : SelectionDAGISelLegacy(ID, std::make_unique<SASSDAGToDAGISel>(TM, OL)) {}

  // Branch uniformity for SASSReconverge.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<UniformityInfoWrapperPass>();
    SelectionDAGISelLegacy::getAnalysisUsage(AU);
  }
};

} // namespace

char SASSDAGToDAGISelLegacy::ID;

INITIALIZE_PASS(SASSDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createSASSISelDag(SASSTargetMachine &TM,
                                      CodeGenOptLevel OL) {
  return new SASSDAGToDAGISelLegacy(TM, OL);
}

static const unsigned SubIdx[] = {SASS::sub0, SASS::sub1, SASS::sub2,
                                  SASS::sub3};

void SASSDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;
  }
  // Atomics, fences, volatile accesses (SASSISelAtomics.cpp).
  if (SDNode *M = SASS::selectAtomic(*CurDAG, N)) {
    ReplaceUses(SDValue(N, N->getNumValues() - 1),
                SDValue(M, M->getNumValues() - 1));
    if (N->getNumValues() == 2 && M->getNumValues() == 2)
      ReplaceUses(SDValue(N, 0), SDValue(M, 0));
    CurDAG->RemoveDeadNode(N);
    return;
  }
  switch (N->getOpcode()) {
  case SASSISD::LDV2:
  case SASSISD::LDV4:
    return selectVectorLoad(N);
  case SASSISD::STV2:
  case SASSISD::STV4:
    return selectVectorStore(N);
  case SASSISD::LDH1:
  case SASSISD::LDH2:
  case SASSISD::LDH4:
  case SASSISD::STH:
    return selectPolicyAccess(N);
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    if (selectIntrinsic(N) || SASS::selectPTXIntrinsic(*CurDAG, N) ||
        SASS::selectWarpIntrinsic(*CurDAG, N))
      return;
    break;
  case ISD::BRCOND:
    N = foldPredicateCompare(N);
    if (selectUniformBranch(N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

/// `brcond (setcc (zext p), 0|1, eq|ne)`: a branch on an i1 that the type
/// legalizer promoted (e.g. an inverted mbarrier try_wait result, as a
/// fallthrough loop exit) becomes a branch on p or !p, not SEL + ISETP.
SDNode *SASSDAGToDAGISel::foldPredicateCompare(SDNode *N) {
  SDValue Cond = N->getOperand(1);
  if (Cond.getOpcode() != ISD::SETCC || !Cond.hasOneUse())
    return N;
  auto *C = dyn_cast<ConstantSDNode>(Cond.getOperand(1));
  ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
  if (!C || C->getZExtValue() > 1 || (CC != ISD::SETEQ && CC != ISD::SETNE))
    return N;
  SDValue X = Cond.getOperand(0), P;
  if (X.getOpcode() == ISD::ZERO_EXTEND &&
      X.getOperand(0).getValueType() == MVT::i1)
    P = X.getOperand(0);
  else if (X.getOpcode() == ISD::SELECT && isOneConstant(X.getOperand(1)) &&
           isNullConstant(X.getOperand(2)))
    P = X.getOperand(0);
  if (!P)
    return N;
  SDLoc DL(N);
  if ((CC == ISD::SETNE) == (C->getZExtValue() == 1))
    P = CurDAG->getNode(ISD::XOR, DL, MVT::i1, P,
                        CurDAG->getAllOnesConstant(DL, MVT::i1));
  N = CurDAG->UpdateNodeOperands(N, N->getOperand(0), P, N->getOperand(2));
  if (Cond->use_empty())
    CurDAG->RemoveDeadNode(Cond.getNode());
  return N;
}

bool SASSDAGToDAGISel::selectUniformBranch(SDNode *N) {
  auto *UA = const_cast<UniformityInfo *>(CurDAG->getUniformityInfo());
  const BasicBlock *BB = FuncInfo->MBB->getBasicBlock();
  if (!UA || !BB || !SASS::reconverges(*Subtarget) ||
      UA->hasDivergentTerminator(*BB))
    return false;
  SDValue Cond = N->getOperand(1);
  unsigned Neg = 0;
  if (Cond.getOpcode() == ISD::XOR && isAllOnesConstant(Cond.getOperand(1))) {
    Cond = Cond.getOperand(0);
    Neg = 1;
  }
  SDLoc DL(N);
  SDValue Ops[] = {N->getOperand(2), Cond,
                   CurDAG->getTargetConstant(Neg, DL, MVT::i32),
                   N->getOperand(0)};
  CurDAG->SelectNodeTo(N, SASS::BRAcu, MVT::Other, Ops);
  return true;
}

bool SASSDAGToDAGISel::selectIntrinsic(SDNode *N) {
  bool HasChain = N->getOpcode() != ISD::INTRINSIC_WO_CHAIN;
  unsigned ID = N->getConstantOperandVal(HasChain ? 1 : 0);
  SDLoc DL(N);
  switch (ID) {
  case Intrinsic::nvvm_mma_m16n8k16_row_col_f32_f32:
    selectMMA(N, SASS::HMMA16816F32, SASS::HMMA16816F32Z);
    return true;
  case Intrinsic::nvvm_mma_m16n8k16_row_col_bf16:
    selectMMA(N, SASS::HMMA16816F32BF16, SASS::HMMA16816F32BF16Z);
    return true;
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_b16:
    selectMatrixLoad(N, SASS::MAT1_LD, 1);
    return true;
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_b16:
    selectMatrixLoad(N, SASS::MAT2_LD, 2);
    return true;
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_b16:
    selectMatrixLoad(N, SASS::MAT4_LD, 4);
    return true;
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16:
    selectMatrixLoad(N, SASS::MAT1_LDT, 1);
    return true;
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16:
    selectMatrixLoad(N, SASS::MAT2_LDT, 2);
    return true;
  case Intrinsic::nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16:
    selectMatrixLoad(N, SASS::MAT4_LDT, 4);
    return true;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_b16:
    selectMatrixStore(N, SASS::MAT1_ST, 1);
    return true;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_b16:
    selectMatrixStore(N, SASS::MAT2_ST, 2);
    return true;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_b16:
    selectMatrixStore(N, SASS::MAT4_ST, 4);
    return true;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x1_trans_b16:
    selectMatrixStore(N, SASS::MAT1_STT, 1);
    return true;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16:
    selectMatrixStore(N, SASS::MAT2_STT, 2);
    return true;
  case Intrinsic::nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16:
    selectMatrixStore(N, SASS::MAT4_STT, 4);
    return true;
  case Intrinsic::nvvm_cp_async_cg_shared_global_16:
    selectAsyncCopy(N, SASS::LDGSTS128);
    return true;
  case Intrinsic::nvvm_cp_async_cg_shared_global_16_s:
    selectAsyncCopy(N, SASS::LDGSTS128_P);
    return true;
  case Intrinsic::nvvm_cp_async_ca_shared_global_16_s:
    selectAsyncCopy(N, SASS::LDGSTS_CA128_P);
    return true;
  case Intrinsic::nvvm_cp_async_ca_shared_global_8_s:
    selectAsyncCopy(N, SASS::LDGSTS_CA64_P);
    return true;
  case Intrinsic::nvvm_cp_async_ca_shared_global_4_s:
    selectAsyncCopy(N, SASS::LDGSTS_CA32_P);
    return true;
  case Intrinsic::nvvm_cp_async_ca_shared_global_16:
    selectAsyncCopy(N, SASS::LDGSTS_CA128);
    return true;
  case Intrinsic::nvvm_cp_async_ca_shared_global_8:
    selectAsyncCopy(N, SASS::LDGSTS_CA64);
    return true;
  case Intrinsic::nvvm_cp_async_ca_shared_global_4:
    selectAsyncCopy(N, SASS::LDGSTS_CA32);
    return true;
  case Intrinsic::nvvm_cp_async_commit_group: {
    SDValue Ops[] = {guard(DL), CurDAG->getTargetConstant(0, DL, MVT::i32),
                     N->getOperand(0)};
    ReplaceNode(N, CurDAG->getMachineNode(SASS::LDGDEPBAR, DL, MVT::Other, Ops));
    return true;
  }
  case Intrinsic::nvvm_cp_async_wait_group:
  case Intrinsic::nvvm_cp_async_wait_all: {
    unsigned Pending = ID == Intrinsic::nvvm_cp_async_wait_group
                           ? N->getConstantOperandVal(2)
                           : 0;
    SDValue Ops[] = {CurDAG->getTargetConstant(Pending, DL, MVT::i32), guard(DL),
                     CurDAG->getTargetConstant(0, DL, MVT::i32), N->getOperand(0)};
    ReplaceNode(N, CurDAG->getMachineNode(SASS::DEPBAR_LE, DL, MVT::Other, Ops));
    return true;
  }
  default:
    return false;
  }
}

SDValue SASSDAGToDAGISel::tuple(const SDLoc &DL, ArrayRef<SDValue> Vals) {
  SmallVector<SDValue, 9> Ops = {CurDAG->getTargetConstant(
      Vals.size() == 4 ? SASS::GPR128RegClassID : SASS::GPR64RegClassID, DL,
      MVT::i32)};
  for (auto [I, V] : enumerate(Vals)) {
    Ops.push_back(V);
    Ops.push_back(CurDAG->getTargetConstant(SubIdx[I], DL, MVT::i32));
  }
  return SDValue(
      CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::Untyped, Ops),
      0);
}

// mma.m16n8k16 (id, a0..a3, b0..b1, c0..c3) -> HMMA on register tuples.
void SASSDAGToDAGISel::selectMMA(SDNode *N, unsigned Opc, unsigned ZeroAccOpc) {
  SDLoc DL(N);
  auto Op = [&](unsigned I) { return N->getOperand(1 + I); };
  // sm_90: a +0.0 accumulator is RZ, as ptxas emits it (no zeroed tuple
  // to keep live and no C register reads).
  auto PosZero = [](SDValue V) {
    if (auto *C = dyn_cast<ConstantFPSDNode>(V))
      return C->isZero() && !C->isNegative();
    return isNullConstant(V);
  };
  bool ZeroAcc = !Subtarget->hasBlackwellISA() && PosZero(Op(6)) &&
                 PosZero(Op(7)) && PosZero(Op(8)) && PosZero(Op(9));
  SmallVector<SDValue, 5> Ops = {tuple(DL, {Op(0), Op(1), Op(2), Op(3)}),
                                 tuple(DL, {Op(4), Op(5)})};
  if (!ZeroAcc)
    Ops.push_back(tuple(DL, {Op(6), Op(7), Op(8), Op(9)}));
  Ops.push_back(guard(DL));
  Ops.push_back(CurDAG->getTargetConstant(0, DL, MVT::i32));
  MachineSDNode *MMA = CurDAG->getMachineNode(ZeroAcc ? ZeroAccOpc : Opc, DL,
                                              MVT::Untyped, Ops);
  for (unsigned I = 0; I < 4; ++I)
    ReplaceUses(SDValue(N, I), CurDAG->getTargetExtractSubreg(
                                   SubIdx[I], DL, MVT::f32, SDValue(MMA, 0)));
  CurDAG->RemoveDeadNode(N);
}

// ldmatrix (chain, id, addr) -> LDSM into 1, 2 or 4 registers.
void SASSDAGToDAGISel::selectMatrixLoad(SDNode *N, unsigned Opc,
                                        unsigned Count) {
  SDLoc DL(N);
  SDValue Base, Off;
  SelectADDRs(N->getOperand(2), Base, Off);
  SDValue Ops[] = {Base, Off, guard(DL), CurDAG->getTargetConstant(0, DL, MVT::i32),
                   N->getOperand(0)};
  MachineSDNode *Ld = CurDAG->getMachineNode(
      Opc, DL, Count == 1 ? MVT::i32 : MVT::Untyped, MVT::Other, Ops);
  CurDAG->setNodeMemRefs(Ld, {cast<MemSDNode>(N)->getMemOperand()});
  for (unsigned I = 0; I < Count; ++I)
    ReplaceUses(SDValue(N, I),
                Count == 1 ? SDValue(Ld, 0)
                           : CurDAG->getTargetExtractSubreg(SubIdx[I], DL, MVT::i32,
                                                            SDValue(Ld, 0)));
  ReplaceUses(SDValue(N, Count), SDValue(Ld, 1));
  CurDAG->RemoveDeadNode(N);
}

// stmatrix (chain, id, addr, v0..) -> STSM from 1, 2 or 4 registers.
void SASSDAGToDAGISel::selectMatrixStore(SDNode *N, unsigned Opc,
                                         unsigned Count) {
  SDLoc DL(N);
  SDValue Base, Off;
  SelectADDRs(N->getOperand(2), Base, Off);
  SmallVector<SDValue, 4> Vals;
  for (unsigned I = 0; I < Count; ++I)
    Vals.push_back(N->getOperand(3 + I));
  SDValue Ops[] = {Base, Off, Count == 1 ? Vals[0] : tuple(DL, Vals), guard(DL),
                   CurDAG->getTargetConstant(0, DL, MVT::i32), N->getOperand(0)};
  MachineSDNode *St = CurDAG->getMachineNode(Opc, DL, MVT::Other, Ops);
  CurDAG->setNodeMemRefs(St, {cast<MemSDNode>(N)->getMemOperand()});
  ReplaceNode(N, St);
}

// cp.async (chain, id, shared dst, global src) -> LDGSTS. Its offsets are
// narrower than other memory instructions': signed 20 (shared) and 12
// (global) bits.
void SASSDAGToDAGISel::selectAsyncCopy(SDNode *N, unsigned Opc) {
  SDLoc DL(N);
  SDValue SBase, SOff, GBase, GOff;
  SelectADDRs(N->getOperand(2), SBase, SOff);
  if (!isInt<20>(cast<ConstantSDNode>(SOff)->getSExtValue())) {
    SBase = N->getOperand(2);
    SOff = CurDAG->getTargetConstant(0, DL, MVT::i32);
  }
  SelectADDRri(N->getOperand(3), GBase, GOff);
  if (!isInt<12>(cast<ConstantSDNode>(GOff)->getSExtValue())) {
    GBase = N->getOperand(3);
    GOff = CurDAG->getTargetConstant(0, DL, MVT::i32);
  }
  SmallVector<SDValue, 9> Ops = {SBase, SOff, GBase, GOff};
  if (N->getNumOperands() > 4) {
    // _s forms: SASSLowerInlinePTX only emits a src-size of 0 or the copy
    // size, so the source is read iff it is non-zero.
    SDValue Size = N->getOperand(4);
    if (auto *C = dyn_cast<ConstantSDNode>(Size)) {
      Ops.push_back(CurDAG->getRegister(SASS::PT, MVT::i1));
      Ops.push_back(CurDAG->getTargetConstant(C->isZero(), DL, MVT::i32));
    } else {
      SDValue Pt = CurDAG->getRegister(SASS::PT, MVT::i1);
      SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
      Ops.push_back(SDValue(
          CurDAG->getMachineNode(SASS::ISETPri, DL, MVT::i1,
                                 {Size, Zero,
                                  CurDAG->getTargetConstant(5, DL, MVT::i32),
                                  Pt, Zero, guard(DL), Zero}),
          0));
      Ops.push_back(Zero);
    }
  }
  Ops.append({guard(DL), CurDAG->getTargetConstant(0, DL, MVT::i32),
              N->getOperand(0)});
  MachineSDNode *Cp = CurDAG->getMachineNode(Opc, DL, MVT::Other, Ops);
  CurDAG->setNodeMemRefs(Cp, {cast<MemSDNode>(N)->getMemOperand()});
  ReplaceNode(N, Cp);
}

static bool baseOffset(SelectionDAG &DAG, SDValue Addr, SDValue &Base,
                       SDValue &Offset) {
  SDLoc DL(Addr);
  if (DAG.isBaseWithConstantOffset(Addr)) {
    int64_t C = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<24>(C)) {
      Base = Addr.getOperand(0);
      Offset = DAG.getSignedTargetConstant(C, DL, MVT::i32);
      return true;
    }
  }
  Base = Addr;
  Offset = DAG.getTargetConstant(0, DL, MVT::i32);
  return true;
}

bool SASSDAGToDAGISel::SelectADDRri(SDValue Addr, SDValue &Base,
                                    SDValue &Offset) {
  return baseOffset(*CurDAG, Addr, Base, Offset);
}

bool SASSDAGToDAGISel::SelectADDRs(SDValue Addr, SDValue &Base,
                                   SDValue &Offset) {
  return baseOffset(*CurDAG, Addr, Base, Offset);
}

bool SASSDAGToDAGISel::SelectADDRl(SDValue Addr, SDValue &Base,
                                   SDValue &Offset) {
  baseOffset(*CurDAG, Addr, Base, Offset);
  if (auto *FI = dyn_cast<FrameIndexSDNode>(Base))
    Base = CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
  return true;
}

bool SASSDAGToDAGISel::isNonCoherent(const MemSDNode *N) const {
  if (N->getAddressSpace() != 1 || !N->readMem() || N->writeMem())
    return false;
  if (N->isInvariant())
    return true;
  const Value *Ptr = N->getMemOperand()->getValue();
  if (!Ptr)
    return false;
  SmallVector<const Value *, 4> Objs;
  getUnderlyingObjects(Ptr, Objs);
  return !Objs.empty() && all_of(Objs, [](const Value *V) {
    if (const auto *A = dyn_cast<Argument>(V))
      return A->onlyReadsMemory() && A->hasNoAliasAttr();
    if (const auto *GV = dyn_cast<GlobalVariable>(V))
      return GV->isConstant();
    return false;
  });
}

// LDV2/LDV4 (chain, addr) -> one 64/128-bit load into a register tuple whose
// sub-registers replace the scalar results.
void SASSDAGToDAGISel::selectVectorLoad(SDNode *N) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  bool Four = N->getOpcode() == SASSISD::LDV4;
  unsigned NumElts = Four ? 4 : 2;
  SDValue Base, Off;
  unsigned Opc;
  switch (M->getAddressSpace()) {
  case 3:
    SelectADDRs(N->getOperand(1), Base, Off);
    Opc = Four ? SASS::LDS128 : SASS::LDS64;
    break;
  case 5:
    SelectADDRl(N->getOperand(1), Base, Off);
    Opc = Four ? SASS::LDL128 : SASS::LDL64;
    break;
  case 1:
    SelectADDRri(N->getOperand(1), Base, Off);
    if (isNonCoherent(M))
      Opc = Four ? SASS::LDG128_NC : SASS::LDG64_NC;
    else
      Opc = Four ? SASS::LDG128 : SASS::LDG64;
    break;
  default:
    SelectADDRri(N->getOperand(1), Base, Off);
    Opc = Four ? SASS::LD128 : SASS::LD64;
    break;
  }
  SDValue Ops[] = {Base, Off, guard(DL), CurDAG->getTargetConstant(0, DL, MVT::i32),
                   N->getOperand(0)};
  MachineSDNode *Ld =
      CurDAG->getMachineNode(Opc, DL, MVT::Untyped, MVT::Other, Ops);
  CurDAG->setNodeMemRefs(Ld, {M->getMemOperand()});
  EVT EltVT = N->getValueType(0);
  for (unsigned I = 0; I < NumElts; ++I)
    ReplaceUses(SDValue(N, I), CurDAG->getTargetExtractSubreg(
                                   SubIdx[I], DL, EltVT, SDValue(Ld, 0)));
  ReplaceUses(SDValue(N, NumElts), SDValue(Ld, 1));
  CurDAG->RemoveDeadNode(N);
}

// LDH* (chain, addr, policy) / STH (chain, addr, policy, words...) ->
// LDG/STG/LD/ST desc[policy][addr], the policy in a uniform register pair
// (Hopper; Blackwell keeps the default descriptor). 1/2-byte loads
// zero-extend; 8/16-byte accesses use register tuples of the 32-bit words.
void SASSDAGToDAGISel::selectPolicyAccess(SDNode *N) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  bool Load = N->getOpcode() != SASSISD::STH;
  unsigned Bytes = M->getMemoryVT().getStoreSize();
  unsigned Size = Bytes == 1 ? 0 : Bytes == 2 ? 1 : Bytes == 4 ? 2 : Bytes == 8 ? 3 : 4;
  bool Desc = !Subtarget->hasBlackwellISA();
  bool Global = M->getAddressSpace() == 1;
  bool NC = Load && isNonCoherent(M);
  // [desc][global][size]
  static const unsigned Loads[2][2][5] = {
      {{SASS::LDU8, SASS::LDU16, SASS::LD32, SASS::LD64, SASS::LD128},
       {SASS::LDGU8, SASS::LDGU16, SASS::LDG32, SASS::LDG64, SASS::LDG128}},
      {{SASS::LDU8_D, SASS::LDU16_D, SASS::LD32_D, SASS::LD64_D, SASS::LD128_D},
       {SASS::LDGU8_D, SASS::LDGU16_D, SASS::LDG32_D, SASS::LDG64_D, SASS::LDG128_D}}};
  static const unsigned LoadsNC[2][5] = {
      {SASS::LDGU8_NC, SASS::LDGU16_NC, SASS::LDG32_NC, SASS::LDG64_NC, SASS::LDG128_NC},
      {SASS::LDGU8_D_NC, SASS::LDGU16_D_NC, SASS::LDG32_D_NC, SASS::LDG64_D_NC,
       SASS::LDG128_D_NC}};
  static const unsigned Stores[2][2][5] = {
      {{SASS::STU8, SASS::STU16, SASS::ST32, SASS::ST64, SASS::ST128},
       {SASS::STGU8, SASS::STGU16, SASS::STG32, SASS::STG64, SASS::STG128}},
      {{SASS::STU8_D, SASS::STU16_D, SASS::ST32_D, SASS::ST64_D, SASS::ST128_D},
       {SASS::STGU8_D, SASS::STGU16_D, SASS::STG32_D, SASS::STG64_D, SASS::STG128_D}}};
  unsigned Opc = Load ? (NC ? LoadsNC[Desc][Size] : Loads[Desc][Global][Size])
                      : Stores[Desc][Global][Size];
  SmallVector<SDValue, 8> Ops;
  if (Desc)
    Ops.push_back(SASS::policyDesc(*CurDAG, N->getOperand(2), DL));
  SDValue Base, Off;
  SelectADDRri(N->getOperand(1), Base, Off);
  Ops.append({Base, Off});
  unsigned Words = std::max(1u, Bytes / 4);
  if (!Load) {
    if (Words == 1) {
      Ops.push_back(N->getOperand(3));
    } else {
      SmallVector<SDValue, 4> Vals(N->op_begin() + 3, N->op_begin() + 3 + Words);
      Ops.push_back(tuple(DL, Vals));
    }
  }
  Ops.append({guard(DL), CurDAG->getTargetConstant(0, DL, MVT::i32), N->getOperand(0)});
  MachineSDNode *MI;
  if (!Load) {
    MI = CurDAG->getMachineNode(Opc, DL, MVT::Other, Ops);
  } else {
    MI = CurDAG->getMachineNode(Opc, DL, Words == 1 ? MVT::i32 : MVT::Untyped, MVT::Other,
                                Ops);
    for (unsigned I = 0; I < Words; ++I)
      ReplaceUses(SDValue(N, I),
                  Words == 1 ? SDValue(MI, 0)
                             : CurDAG->getTargetExtractSubreg(SubIdx[I], DL, MVT::i32,
                                                              SDValue(MI, 0)));
    ReplaceUses(SDValue(N, Words), SDValue(MI, 1));
  }
  CurDAG->setNodeMemRefs(MI, {M->getMemOperand()});
  if (Load)
    CurDAG->RemoveDeadNode(N);
  else
    ReplaceNode(N, MI);
}

// STV2/STV4 (chain, v0..vN-1, addr) -> REG_SEQUENCE + one wide store.
void SASSDAGToDAGISel::selectVectorStore(SDNode *N) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  bool Four = N->getOpcode() == SASSISD::STV4;
  unsigned NumElts = Four ? 4 : 2;
  if (M->getAddressSpace() == 3 && !Subtarget->hasBlackwellISA() &&
      all_of(seq(1u, NumElts + 1), [&](unsigned I) { return isNullConstant(N->getOperand(I)); })) {
    SDValue Base, Off;
    SelectADDRs(N->getOperand(1 + NumElts), Base, Off);
    SDValue Ops[] = {Base, Off, guard(DL), CurDAG->getTargetConstant(0, DL, MVT::i32),
                     N->getOperand(0)};
    MachineSDNode *St = CurDAG->getMachineNode(Four ? SASS::STS128Z : SASS::STS64Z, DL,
                                               MVT::Other, Ops);
    CurDAG->setNodeMemRefs(St, {M->getMemOperand()});
    ReplaceNode(N, St);
    return;
  }
  SmallVector<SDValue, 9> Seq = {CurDAG->getTargetConstant(
      Four ? SASS::GPR128RegClassID : SASS::GPR64RegClassID, DL, MVT::i32)};
  for (unsigned I = 0; I < NumElts; ++I) {
    Seq.push_back(N->getOperand(1 + I));
    Seq.push_back(CurDAG->getTargetConstant(SubIdx[I], DL, MVT::i32));
  }
  SDValue Tuple(CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                       MVT::Untyped, Seq),
                0);
  SDValue Addr = N->getOperand(1 + NumElts);
  SDValue Base, Off;
  unsigned Opc;
  switch (M->getAddressSpace()) {
  case 3:
    SelectADDRs(Addr, Base, Off);
    Opc = Four ? SASS::STS128 : SASS::STS64;
    break;
  case 5:
    SelectADDRl(Addr, Base, Off);
    Opc = Four ? SASS::STL128 : SASS::STL64;
    break;
  case 1:
    SelectADDRri(Addr, Base, Off);
    Opc = Four ? SASS::STG128 : SASS::STG64;
    break;
  default:
    SelectADDRri(Addr, Base, Off);
    Opc = Four ? SASS::ST128 : SASS::ST64;
    break;
  }
  SDValue Ops[] = {Base, Off, Tuple, guard(DL),
                   CurDAG->getTargetConstant(0, DL, MVT::i32), N->getOperand(0)};
  MachineSDNode *St = CurDAG->getMachineNode(Opc, DL, MVT::Other, Ops);
  CurDAG->setNodeMemRefs(St, {M->getMemOperand()});
  ReplaceNode(N, St);
}
