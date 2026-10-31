//===-- SASSISelDAGToDAG.cpp - SASS instruction selection ------------------===//

#include "SASS.h"
#include "SASSSelectionDAGInfo.h"
#include "SASSSubtarget.h"
#include "SASSTargetMachine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
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

  void selectVectorLoad(SDNode *N);
  void selectVectorStore(SDNode *N);
  /// Tensor-core, matrix and async-copy intrinsics; false if \p N is
  /// another intrinsic (left to the generated matcher).
  bool selectIntrinsic(SDNode *N);
  void selectMMA(SDNode *N, unsigned Opc);
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
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
    if (selectIntrinsic(N) || SASS::selectPTXIntrinsic(*CurDAG, N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

bool SASSDAGToDAGISel::selectIntrinsic(SDNode *N) {
  bool HasChain = N->getOpcode() != ISD::INTRINSIC_WO_CHAIN;
  unsigned ID = N->getConstantOperandVal(HasChain ? 1 : 0);
  SDLoc DL(N);
  switch (ID) {
  case Intrinsic::nvvm_mma_m16n8k16_row_col_f32_f32:
    selectMMA(N, SASS::HMMA16816F32);
    return true;
  case Intrinsic::nvvm_mma_m16n8k16_row_col_bf16:
    selectMMA(N, SASS::HMMA16816F32BF16);
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
void SASSDAGToDAGISel::selectMMA(SDNode *N, unsigned Opc) {
  SDLoc DL(N);
  auto Op = [&](unsigned I) { return N->getOperand(1 + I); };
  SDValue Ops[] = {tuple(DL, {Op(0), Op(1), Op(2), Op(3)}),
                   tuple(DL, {Op(4), Op(5)}),
                   tuple(DL, {Op(6), Op(7), Op(8), Op(9)}), guard(DL),
                   CurDAG->getTargetConstant(0, DL, MVT::i32)};
  MachineSDNode *MMA = CurDAG->getMachineNode(Opc, DL, MVT::Untyped, Ops);
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
  SDValue Ops[] = {SBase, SOff, GBase, GOff, guard(DL),
                   CurDAG->getTargetConstant(0, DL, MVT::i32), N->getOperand(0)};
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

// STV2/STV4 (chain, v0..vN-1, addr) -> REG_SEQUENCE + one wide store.
void SASSDAGToDAGISel::selectVectorStore(SDNode *N) {
  SDLoc DL(N);
  auto *M = cast<MemSDNode>(N);
  bool Four = N->getOpcode() == SASSISD::STV4;
  unsigned NumElts = Four ? 4 : 2;
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
