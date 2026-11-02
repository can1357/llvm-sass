//===-- SASSISelPTX.cpp - Selection of inline-PTX intrinsics --------------===//
//
// DAG lowering and selection for the NVVM intrinsics SASSExpandInlinePTX
// emits for inline PTX (simple ones are patterns in SASSInstrPTX.td):
//
// * FP8 conversions carry i16 values: rewritten into i32 target nodes.
// * mbarrier operations become SYNCS on [Ra + off]; fences become
//   MEMBAR/FENCE.VIEW.ASYNC with a wait on the fence's scoreboard.
// * Instructions that only take uniform registers (mbarrier.init's
//   SYNCS.EXCH, ARRIVES, bulk and tensor copies, tensor-map prefetch) are
//   selected to UNI_* pseudos, expanded after selection into the loop ptxas
//   emits: elect one pending thread, move its operands into uniform
//   registers, issue the instruction, repeat while any thread is pending
//   (BRA.U.ANY keeps the warp together).
// * L2 cache policies (.L2::cache_hint) become the descriptor pair of the
//   instruction (desc[URa]): UMOVs of a constant policy or R2URs of a
//   computed one. ld/st with a policy arrive as SASS::L2HintLoadFn/
//   L2HintStoreFn calls, lowered to SASSISD::LDH*/STH memory nodes.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSISelLowering.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// FP8 conversions (i16 operands/results)
//===----------------------------------------------------------------------===//

void SASSTargetLowering::initPTXActions() {
  // llvm.nvvm.{ff,f16x2}.to.e{4m3,5m2}x2.rn return i16 and
  // llvm.nvvm.e{4m3,5m2}x2.to.f16x2.rn take one: lowered to i32 nodes.
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i16, Custom);
  // Bulk/tensor copies take an i16 CTA mask (unused: no multicast).
  setOperationAction(ISD::INTRINSIC_VOID, MVT::i16, Custom);
  // trap: BPT.TRAP.
  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  // `br (not p)` must stay a brcond on the negated predicate.
  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
}

static SDValue fp8Node(SDNode *N, SelectionDAG &DAG) {
  SDLoc DL(N);
  unsigned ID = N->getConstantOperandVal(0);
  auto Op = [&](unsigned I) { return N->getOperand(I + 1); };
  switch (ID) {
  case Intrinsic::nvvm_ff_to_e4m3x2_rn:
    return DAG.getNode(SASSISD::F32_TO_E4M3X2, DL, MVT::i32, Op(0), Op(1));
  case Intrinsic::nvvm_ff_to_e5m2x2_rn:
    return DAG.getNode(SASSISD::F32_TO_E5M2X2, DL, MVT::i32, Op(0), Op(1));
  case Intrinsic::nvvm_f16x2_to_e4m3x2_rn:
    return DAG.getNode(SASSISD::F16_TO_E4M3X2, DL, MVT::i32, Op(0));
  case Intrinsic::nvvm_f16x2_to_e5m2x2_rn:
    return DAG.getNode(SASSISD::F16_TO_E5M2X2, DL, MVT::i32, Op(0));
  case Intrinsic::nvvm_e4m3x2_to_f16x2_rn:
  case Intrinsic::nvvm_e5m2x2_to_f16x2_rn: {
    // The i16 operand, widened: the node reads the low 16 bits.
    SDValue A = Op(0);
    if (A.getOpcode() == ISD::TRUNCATE && A.getOperand(0).getValueType() == MVT::i32)
      A = A.getOperand(0);
    else
      A = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, A);
    return DAG.getNode(ID == Intrinsic::nvvm_e4m3x2_to_f16x2_rn ? SASSISD::E4M3X2_TO_F16
                                                               : SASSISD::E5M2X2_TO_F16,
                       DL, MVT::v2f16, A);
  }
  default:
    return SDValue();
  }
}

SDValue SASSTargetLowering::lowerPTXOp(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN)
    return fp8Node(Op.getNode(), DAG);
  if (Op.getOpcode() != ISD::INTRINSIC_VOID)
    return SDValue();
  // Widen i16 operands (CTA masks) to i32; selection only checks their flags.
  SmallVector<SDValue, 16> Ops;
  bool Changed = false;
  for (const SDValue &O : Op->op_values()) {
    if (O.getValueType() == MVT::i16) {
      Ops.push_back(DAG.getNode(ISD::ANY_EXTEND, SDLoc(Op), MVT::i32, O));
      Changed = true;
    } else {
      Ops.push_back(O);
    }
  }
  if (!Changed)
    return SDValue();
  return DAG.getNode(ISD::INTRINSIC_VOID, SDLoc(Op), Op->getVTList(), Ops);
}

bool SASSTargetLowering::replacePTXResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  if (N->getOpcode() != ISD::INTRINSIC_WO_CHAIN || N->getValueType(0) != MVT::i16)
    return false;
  SDValue V = fp8Node(N, DAG);
  if (!V)
    return false;
  Results.push_back(DAG.getNode(ISD::TRUNCATE, SDLoc(N), MVT::i16, V));
  return true;
}

SDValue SASSTargetLowering::lowerL2HintCall(CallLoweringInfo &CLI,
                                            SmallVectorImpl<SDValue> &InVals) const {
  auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee);
  if (!G || !CLI.CB)
    return SDValue();
  StringRef Name = G->getGlobal()->getName();
  bool Load = Name.starts_with(SASS::L2HintLoadFn);
  if (!Load && !Name.starts_with(SASS::L2HintStoreFn))
    return SDValue();
  // <prefix>.<bytes>.p<addrspace>
  unsigned Bytes;
  if (Name.split('.').second.split('.').first.getAsInteger(10, Bytes))
    report_fatal_error("SASS: bad L2 cache-hint builtin " + Name);
  CLI.IsTailCall = false; // A memory access: the kernel still returns.
  SelectionDAG &DAG = CLI.DAG;
  MachineFunction &MF = DAG.getMachineFunction();
  const Value *Ptr = CLI.CB->getArgOperand(0);
  MachineMemOperand::Flags F = Load ? MachineMemOperand::MOLoad : MachineMemOperand::MOStore;
  if (Load) {
    uint64_t Flags = cast<ConstantSDNode>(CLI.OutVals[2])->getZExtValue();
    if (Flags & 1)
      F |= MachineMemOperand::MOInvariant;
    F |= SASS::loadHintToMMOFlags(Flags);
  }
  MachineMemOperand *MMO = MF.getMachineMemOperand(MachinePointerInfo(Ptr), F,
                                                    LocationSize::precise(Bytes), Align(Bytes));
  EVT MemVT = EVT::getIntegerVT(*DAG.getContext(), Bytes * 8);
  unsigned Words = std::max(1u, Bytes / 4);
  SDValue Addr = CLI.OutVals[0], Pol = CLI.OutVals[1];
  if (Load) {
    unsigned Opc = Words == 4 ? SASSISD::LDH4 : Words == 2 ? SASSISD::LDH2 : SASSISD::LDH1;
    SmallVector<EVT, 5> VTs(Words, MVT::i32);
    VTs.push_back(MVT::Other);
    SDValue Ld = DAG.getMemIntrinsicNode(Opc, CLI.DL, DAG.getVTList(VTs),
                                         {CLI.Chain, Addr, Pol}, MemVT, MMO);
    for (unsigned I = 0; I < Words; ++I)
      InVals.push_back(Ld.getValue(I));
    return Ld.getValue(Words);
  }
  SmallVector<SDValue, 7> Ops = {CLI.Chain, Addr, Pol};
  Ops.append(CLI.OutVals.begin() + 2, CLI.OutVals.end());
  // sm_90: STG.U8/.U16 store only the low bits of their word; its zero
  // extension (an LOP3 after the F2FP of a bf16 store) is dropped.
  if (Bytes < 4 && !DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA()) {
    SDValue &V = Ops[3];
    auto *M = V.getOpcode() == ISD::AND ? dyn_cast<ConstantSDNode>(V.getOperand(1))
                                        : nullptr;
    if (V.getOpcode() == ISD::ZERO_EXTEND)
      V = DAG.getNode(ISD::ANY_EXTEND, CLI.DL, MVT::i32, V.getOperand(0));
    else if (M && M->getAPIntValue().countr_one() >= Bytes * 8)
      V = V.getOperand(0);
  }
  return DAG.getMemIntrinsicNode(SASSISD::STH, CLI.DL, DAG.getVTList(MVT::Other), Ops, MemVT,
                                 MMO);
}

//===----------------------------------------------------------------------===//
// Selection
//===----------------------------------------------------------------------===//

namespace {
class PTXSelector {
  SelectionDAG &DAG;
  SDNode *N;
  SDLoc DL;
  SDValue Chain;

public:
  PTXSelector(SelectionDAG &DAG, SDNode *N) : DAG(DAG), N(N), DL(N) {
    if (N->getOpcode() != ISD::INTRINSIC_WO_CHAIN)
      Chain = N->getOperand(0);
  }
  bool select();

private:
  unsigned id() const {
    return N->getConstantOperandVal(N->getOpcode() == ISD::INTRINSIC_WO_CHAIN ? 0 : 1);
  }
  /// Intrinsic argument \p I.
  SDValue arg(unsigned I) const {
    return N->getOperand(I + (N->getOpcode() == ISD::INTRINSIC_WO_CHAIN ? 1 : 2));
  }
  /// \p Uni if every operand in \p Ops is warp-uniform (Hopper; SDNode
  /// divergence from SASSTargetLowering::isSDNodeSourceOfDivergence), else
  /// \p Any (a waterfall loop over the distinct values).
  unsigned uniformOr(unsigned Uni, unsigned Any, ArrayRef<SDValue> Ops) const {
    if (DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA() ||
        !DAG.getUniformityInfo())
      return Any;
    for (SDValue V : Ops)
      if (V->isDivergent())
        return Any;
    return Uni;
  }
  SDValue imm(int64_t V) { return DAG.getTargetConstant(V, DL, MVT::i32); }
  SDValue rz() { return DAG.getRegister(SASS::RZ, MVT::i32); }
  /// A 32-bit register operand: RZ for 0, a MOV for other constants
  /// (machine-node operands that are constants become immediates).
  SDValue reg32(SDValue V) {
    if (auto *C = dyn_cast<ConstantSDNode>(V))
      return C->isZero() ? rz() : node(SASS::MOVri, MVT::i32, {imm(C->getSExtValue())});
    return V;
  }
  SDValue reg32(int64_t V) { return reg32(DAG.getConstant(V, DL, MVT::i32)); }
  /// Shared address as (base, non-negative 24-bit offset).
  std::pair<SDValue, SDValue> shared(SDValue A) {
    if (A.getOpcode() == ISD::ADD)
      if (auto *C = dyn_cast<ConstantSDNode>(A.getOperand(1)))
        if (int64_t V = C->getSExtValue(); V >= 0 && V < (1 << 23))
          return {reg32(A.getOperand(0)), imm(V)};
    if (auto *C = dyn_cast<ConstantSDNode>(A); C && C->getSExtValue() >= 0 &&
                                                C->getSExtValue() < (1 << 23))
      return {rz(), imm(C->getSExtValue())};
    return {A, imm(0)};
  }
  /// Builds \p Opc with \p Ops, the always-true guard and the chain.
  MachineSDNode *emit(unsigned Opc, ArrayRef<EVT> VTs, ArrayRef<SDValue> Ops,
                      bool WithChain = true) {
    SmallVector<SDValue, 12> All(Ops);
    All.push_back(DAG.getRegister(SASS::PT, MVT::i1));
    All.push_back(imm(0));
    SmallVector<EVT, 3> Tys(VTs);
    if (WithChain) {
      All.push_back(Chain);
      Tys.push_back(MVT::Other);
    }
    MachineSDNode *M = DAG.getMachineNode(Opc, DL, DAG.getVTList(Tys), All);
    if (WithChain)
      Chain = SDValue(M, Tys.size() - 1);
    return M;
  }
  SDValue node(unsigned Opc, EVT VT, ArrayRef<SDValue> Ops) {
    return SDValue(emit(Opc, {VT}, Ops, /*WithChain=*/false), 0);
  }
  /// Replaces N's results (values in order, then the chain).
  void done(ArrayRef<SDValue> Vals = {}) {
    for (auto [I, V] : enumerate(Vals))
      DAG.ReplaceAllUsesOfValueWith(SDValue(N, I), V);
    if (N->getOpcode() != ISD::INTRINSIC_WO_CHAIN)
      DAG.ReplaceAllUsesOfValueWith(SDValue(N, N->getNumValues() - 1), Chain);
    DAG.RemoveDeadNode(N);
  }
  /// FENCE.VIEW.ASYNC.{S,G} and an immediate wait for it.
  SDValue fenceView(bool Global) {
    SDValue Done(emit(Global ? SASS::FENCE_VIEW_ASYNC_G : SASS::FENCE_VIEW_ASYNC_S,
                      {MVT::i32}, {}),
                 0);
    emit(SASS::SBWAIT, {}, {Done});
    return Done;
  }
  void releaseCluster() {
    for (unsigned Opc : {SASS::MEMBAR_ALL_CTA, SASS::MEMBAR_ALL_GPU, SASS::ERRBAR, SASS::CGAERRBAR})
      emit(Opc, {}, {});
  }
  /// The L2 cache policy of a bulk or tensor copy whose cache-hint flag
  /// (argument \p FlagArg) is set, as a uniform register pair; null without
  /// one. Blackwell copies keep the default policy.
  SDValue policy(unsigned PolArg, unsigned FlagArg) {
    if (!cast<ConstantSDNode>(arg(FlagArg))->getZExtValue() ||
        DAG.getSubtarget<SASSSubtarget>().hasBlackwellISA())
      return SDValue();
    return SASS::policyDesc(DAG, arg(PolArg), DL);
  }
  /// Bulk copy sizes are given in 16-byte units.
  SDValue units16(SDValue Size) {
    if (auto *C = dyn_cast<ConstantSDNode>(Size))
      return reg32(C->getZExtValue() >> 4);
    return node(SASS::SRLri, MVT::i32, {Size, imm(4)});
  }
  bool mbarrier(unsigned ID);
  bool tensor(unsigned ID);
};
} // namespace

bool PTXSelector::mbarrier(unsigned ID) {
  bool Cluster = false;
  switch (ID) {
  case Intrinsic::nvvm_mbarrier_init_shared: {
    // State: pending and expected arrival counts in a 20-bit field each,
    // as ptxas initializes it.
    SDValue Cnt = arg(1), Lo, Hi;
    if (auto *C = dyn_cast<ConstantSDNode>(Cnt)) {
      uint32_t V = 0x100000 - C->getZExtValue();
      Lo = reg32(V << 1);
      Hi = reg32(V << 11);
    } else {
      SDValue V = node(SASS::ISUBrr, MVT::i32, {reg32(0x100000), Cnt});
      Lo = node(SASS::SHLri, MVT::i32, {V, imm(1)});
      Hi = node(SASS::SHLri, MVT::i32, {V, imm(11)});
    }
    SDValue A = reg32(arg(0));
    emit(uniformOr(SASS::UNIU_SYNCS_EXCH, SASS::UNI_SYNCS_EXCH, {A, Lo, Hi}), {},
         {A, Lo, Hi});
    done();
    return true;
  }
  case Intrinsic::nvvm_mbarrier_inval_shared: {
    auto [A, O] = shared(arg(0));
    emit(SASS::SYNCS_INVAL, {}, {A, O});
    done();
    return true;
  }
  case Intrinsic::nvvm_mbarrier_arrive_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cluster_space_cta:
    Cluster = true;
    [[fallthrough]];
  case Intrinsic::nvvm_mbarrier_arrive_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_arrive_relaxed_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_arrive_relaxed_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cluster_space_cta: {
    bool Tx = ID == Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cluster_space_cta ||
              ID == Intrinsic::nvvm_mbarrier_arrive_expect_tx_scope_cta_space_cta ||
              ID == Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cta_space_cta ||
              ID == Intrinsic::nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cluster_space_cta;
    if (Cluster)
      releaseCluster();
    auto [A, O] = shared(arg(0));
    SDValue N1 = arg(1);
    bool One = !Tx && isa<ConstantSDNode>(N1) && cast<ConstantSDNode>(N1)->getZExtValue() == 1;
    bool Used = !SDValue(N, 0).use_empty();
    unsigned Opc = Tx    ? (Used ? SASS::SYNCS_ARRIVE_TX : SASS::SYNCS_ARRIVE_TX_X)
                   : One ? (Used ? SASS::SYNCS_ARRIVE : SASS::SYNCS_ARRIVE_X)
                         : (Used ? SASS::SYNCS_ARRIVE_N : SASS::SYNCS_ARRIVE_N_X);
    SmallVector<SDValue, 3> Ops = {A, O};
    if (!One)
      Ops.push_back(reg32(N1));
    if (Used) {
      MachineSDNode *M = emit(Opc, {MVT::i64}, Ops);
      done({SDValue(M, 0)});
    } else {
      emit(Opc, {}, Ops);
      done({DAG.getUNDEF(MVT::i64)});
    }
    return true;
  }
  case Intrinsic::nvvm_mbarrier_arrive_scope_cluster_space_cluster:
  case Intrinsic::nvvm_mbarrier_arrive_relaxed_scope_cluster_space_cluster: {
    if (ID == Intrinsic::nvvm_mbarrier_arrive_scope_cluster_space_cluster)
      releaseCluster();
    if (!isa<ConstantSDNode>(arg(1)) || cast<ConstantSDNode>(arg(1))->getZExtValue() != 1)
      report_fatal_error("SASS: remote mbarrier.arrive with a count is not supported");
    auto [A, O] = shared(arg(0));
    emit(SASS::SYNCS_ARRIVE_REMOTE, {}, {A, O});
    done();
    return true;
  }
  case Intrinsic::nvvm_mbarrier_expect_tx_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_expect_tx_scope_cluster_space_cta: {
    auto [A, O] = shared(arg(0));
    emit(SASS::SYNCS_EXPECT_TX, {}, {A, O, reg32(arg(1))});
    done();
    return true;
  }
  case Intrinsic::nvvm_mbarrier_try_wait_parity_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_parity_tl_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_tl_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_test_wait_parity_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_test_wait_scope_cluster_space_cta:
    Cluster = true;
    [[fallthrough]];
  case Intrinsic::nvvm_mbarrier_try_wait_parity_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_parity_tl_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_tl_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_test_wait_parity_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_test_wait_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_parity_relaxed_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_try_wait_parity_relaxed_scope_cluster_space_cta:
  case Intrinsic::nvvm_mbarrier_test_wait_parity_relaxed_scope_cta_space_cta:
  case Intrinsic::nvvm_mbarrier_test_wait_parity_relaxed_scope_cluster_space_cta: {
    bool Test = ID == Intrinsic::nvvm_mbarrier_test_wait_parity_scope_cluster_space_cta ||
                ID == Intrinsic::nvvm_mbarrier_test_wait_scope_cluster_space_cta ||
                ID == Intrinsic::nvvm_mbarrier_test_wait_parity_scope_cta_space_cta ||
                ID == Intrinsic::nvvm_mbarrier_test_wait_scope_cta_space_cta ||
                ID == Intrinsic::nvvm_mbarrier_test_wait_parity_relaxed_scope_cta_space_cta ||
                ID == Intrinsic::nvvm_mbarrier_test_wait_parity_relaxed_scope_cluster_space_cta;
    // The phase to wait for is bit 31 of the operand: the parity shifted
    // up, or the high word of an arrive state. A suspend-time hint (tl)
    // only bounds the wait: TRYWAIT's own bound is a valid choice.
    SDValue Ph = arg(1);
    if (Ph.getValueType() == MVT::i64) {
      Ph = DAG.getTargetExtractSubreg(SASS::sub1, DL, MVT::i32, Ph);
    } else if (auto *C = dyn_cast<ConstantSDNode>(Ph)) {
      Ph = C->getZExtValue() & 1 ? reg32(int32_t(0x80000000u)) : rz();
    } else {
      Ph = node(SASS::SHLri, MVT::i32, {Ph, imm(31)});
    }
    auto [A, O] = shared(arg(0));
    MachineSDNode *W = emit(Test ? SASS::SYNCS_TESTWAIT : SASS::SYNCS_TRYWAIT, {MVT::i1}, {A, O, Ph});
    SDValue P(W, 0);
    if (Cluster) {
      // Acquire at cluster scope: invalidate L1 once the phase completed.
      SDValue Ops[] = {P, imm(0), Chain};
      Chain = SDValue(DAG.getMachineNode(SASS::CCTL_IVALL, DL, MVT::Other, Ops), 0);
    }
    done({P});
    return true;
  }
  case Intrinsic::nvvm_cp_async_mbarrier_arrive_noinc_shared: {
    SDValue A = reg32(arg(0));
    emit(uniformOr(SASS::UNIU_ARRIVES_ARVCNT, SASS::UNI_ARRIVES_ARVCNT, {A}), {}, {A});
    done();
    return true;
  }
  case Intrinsic::nvvm_cp_async_mbarrier_arrive_shared: {
    auto [A, O] = shared(arg(0));
    emit(SASS::SYNCS_EXPECT_ONE, {}, {A, O});
    SDValue B = reg32(arg(0));
    emit(uniformOr(SASS::UNIU_ARRIVES_TRANSCNT, SASS::UNI_ARRIVES_TRANSCNT, {B}), {}, {B});
    done();
    return true;
  }
  default:
    return false;
  }
}

bool PTXSelector::tensor(unsigned ID) {
  // g2s: dst, mbar, map, coords..., cta_mask, hint, flag_mask, flag_hint,
  // cta_group. s2g: src, map, coords..., hint, flag_hint.
  struct T {
    Intrinsic::ID IID;
    unsigned Dim;
    bool Load;
  };
  static const T Table[] = {
      {Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_1d, 1, true},
      {Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_2d, 2, true},
      {Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_3d, 3, true},
      {Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_4d, 4, true},
      {Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_5d, 5, true},
      {Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_1d, 1, false},
      {Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_2d, 2, false},
      {Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_3d, 3, false},
      {Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_4d, 4, false},
      {Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_5d, 5, false}};
  const T *E = find_if(Table, [&](const T &E) { return E.IID == ID; });
  if (E == std::end(Table))
    return false;
  SmallVector<SDValue, 11> Ops = {imm(E->Dim)};
  unsigned C0;
  SDValue Pol;
  if (E->Load) {
    if (cast<ConstantSDNode>(arg(3 + E->Dim + 2))->getZExtValue())
      report_fatal_error("SASS: multicast tensor copies are not supported");
    Ops.append({arg(2), reg32(arg(0)), reg32(arg(1))});
    C0 = 3;
    Pol = policy(3 + E->Dim + 1, 3 + E->Dim + 3);
  } else {
    Ops.append({arg(1), reg32(arg(0))});
    C0 = 2;
    Pol = policy(2 + E->Dim, 2 + E->Dim + 1);
  }
  for (unsigned I = 0; I < 5; ++I)
    Ops.push_back(I < E->Dim ? reg32(arg(C0 + I)) : rz());
  if (Pol)
    Ops.push_back(Pol);
  emit(E->Load ? (Pol ? SASS::UNI_UTMALDG_H : SASS::UNI_UTMALDG)
               : (Pol ? SASS::UNI_UTMASTG_H : SASS::UNI_UTMASTG),
       {}, Ops);
  done();
  return true;
}

bool PTXSelector::select() {
  if (N->getOpcode() != ISD::INTRINSIC_W_CHAIN && N->getOpcode() != ISD::INTRINSIC_VOID &&
      N->getOpcode() != ISD::INTRINSIC_WO_CHAIN)
    return false;
  unsigned ID = id();
  if (mbarrier(ID) || tensor(ID))
    return true;
  switch (ID) {
  case Intrinsic::nvvm_fence_proxy_async_shared_cta:
    emit(SASS::MEMBAR_ALL_CTA, {}, {});
    fenceView(false);
    break;
  case Intrinsic::nvvm_fence_proxy_async:
    emit(SASS::MEMBAR_ALL_GPU, {}, {});
    fenceView(false);
    break;
  case Intrinsic::nvvm_fence_proxy_async_global:
    fenceView(true);
    break;
  case Intrinsic::nvvm_fence_mbarrier_init_release_cluster:
    // ptxas emits nothing: SYNCS.EXCH completes before later barrier use.
    break;
  case Intrinsic::nvvm_cp_async_bulk_global_to_shared_cluster:
  case Intrinsic::nvvm_cp_async_bulk_global_to_shared_cta: {
    // cluster: dst, mbar, src, size, cta_mask, hint, flag_mask, flag_hint;
    // cta: dst, mbar, src, size, hint, flag_hint.
    bool Cluster = ID == Intrinsic::nvvm_cp_async_bulk_global_to_shared_cluster;
    if (Cluster && cast<ConstantSDNode>(arg(6))->getZExtValue())
      report_fatal_error("SASS: multicast bulk copies are not supported");
    SDValue Pol = Cluster ? policy(5, 7) : policy(4, 5);
    SmallVector<SDValue, 5> Ops = {reg32(arg(0)), reg32(arg(1)), arg(2), units16(arg(3))};
    unsigned Opc = Pol ? uniformOr(SASS::UNIU_UBLKCP_SG_H, SASS::UNI_UBLKCP_SG_H, Ops)
                       : uniformOr(SASS::UNIU_UBLKCP_SG, SASS::UNI_UBLKCP_SG, Ops);
    if (Pol)
      Ops.push_back(Pol);
    emit(Opc, {}, Ops);
    break;
  }
  case Intrinsic::nvvm_cp_async_bulk_shared_cta_to_global: {
    // dst, src, size, hint, flag_hint
    SDValue Pol = policy(3, 4);
    SmallVector<SDValue, 4> Ops = {arg(0), reg32(arg(1)), units16(arg(2))};
    if (Pol)
      Ops.push_back(Pol);
    emit(Pol ? SASS::UNI_UBLKCP_GS_H : SASS::UNI_UBLKCP_GS, {}, Ops);
    break;
  }
  case Intrinsic::nvvm_cp_async_bulk_prefetch_L2: {
    // src, size, hint, flag_hint
    SDValue Pol = policy(2, 3);
    SmallVector<SDValue, 3> Ops = {arg(0), units16(arg(1))};
    if (Pol)
      Ops.push_back(Pol);
    emit(Pol ? SASS::UNI_UBLKPF_L2_H : SASS::UNI_UBLKPF_L2, {}, Ops);
    break;
  }
  case Intrinsic::nvvm_cp_async_bulk_commit_group:
    emit(SASS::UTMACMDFLUSH, {}, {});
    break;
  case Intrinsic::nvvm_cp_async_bulk_wait_group:
  case Intrinsic::nvvm_cp_async_bulk_wait_group_read:
    emit(SASS::DEPBAR_LE_SB0, {}, {imm(cast<ConstantSDNode>(arg(0))->getZExtValue())});
    // Waiting for the writes (not just the source reads) also makes them
    // visible to this SM's L1.
    if (ID == Intrinsic::nvvm_cp_async_bulk_wait_group)
      emit(SASS::CCTL_IVALL, {}, {});
    break;
  case Intrinsic::nvvm_prefetch_tensormap:
    if (arg(0).getValueType() != MVT::i64)
      report_fatal_error("SASS: prefetch.tensormap needs a generic address");
    emit(SASS::UNI_UTMACCTL_PF, {}, {arg(0)});
    break;
  case Intrinsic::nvvm_barrier_cluster_arrive:
  case Intrinsic::nvvm_barrier_cluster_arrive_aligned:
    releaseCluster();
    emit(SASS::UCGABAR_ARV, {}, {});
    break;
  case Intrinsic::nvvm_barrier_cluster_arrive_relaxed:
  case Intrinsic::nvvm_barrier_cluster_arrive_relaxed_aligned:
    emit(SASS::UCGABAR_ARV, {}, {});
    break;
  case Intrinsic::nvvm_barrier_cluster_wait:
  case Intrinsic::nvvm_barrier_cluster_wait_aligned:
    emit(SASS::UCGABAR_WAIT, {}, {});
    emit(SASS::CCTL_IVALL, {}, {});
    break;
  case Intrinsic::nvvm_elect_sync: {
    // Elects among the active threads (the member mask is all of them in
    // the supported use).
    MachineSDNode *E = emit(SASS::ELECT, {MVT::i1, MVT::i32}, {});
    SDValue Lane = node(SASS::MOVru, MVT::i32, {SDValue(E, 1)});
    done({Lane, SDValue(E, 0)});
    return true;
  }
  case Intrinsic::nvvm_st_async: {
    // STAS [{addr, mbar}], value
    SDValue V = arg(1);
    unsigned Opc = V.getValueType() == MVT::i32   ? SASS::STAS32
                   : V.getValueType() == MVT::i64 ? SASS::STAS64
                                                  : 0;
    if (!Opc)
      report_fatal_error("SASS: unsupported st.async width");
    SDValue Ops[] = {DAG.getTargetConstant(SASS::GPR64RegClassID, DL, MVT::i32), reg32(arg(0)),
                     DAG.getTargetConstant(SASS::sub0, DL, MVT::i32), reg32(arg(2)),
                     DAG.getTargetConstant(SASS::sub1, DL, MVT::i32)};
    SDValue AM(DAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::i64, Ops), 0);
    emit(Opc, {}, {AM, V});
    break;
  }
  default:
    return false;
  }
  done();
  return true;
}

bool SASS::selectPTXIntrinsic(SelectionDAG &DAG, SDNode *N) {
  return PTXSelector(DAG, N).select();
}

SDValue SASS::policyDesc(SelectionDAG &DAG, SDValue Pol, const SDLoc &DL) {
  // A constant policy is two UMOVs (ptxas: UMOV UR12, 0x0; UMOV UR13,
  // 0x12f00000), any other value the R2URs of its halves.
  SDValue Guard[] = {DAG.getRegister(SASS::PT, MVT::i1), DAG.getTargetConstant(0, DL, MVT::i32)};
  auto Half = [&](unsigned Sub) -> SDValue {
    if (auto *C = dyn_cast<ConstantSDNode>(Pol)) {
      uint32_t V = C->getZExtValue() >> (Sub == SASS::sub1 ? 32 : 0);
      SDValue Ops[] = {DAG.getTargetConstant(int32_t(V), DL, MVT::i32), Guard[0], Guard[1]};
      return SDValue(DAG.getMachineNode(SASS::UMOVi, DL, MVT::i32, Ops), 0);
    }
    SDValue Ops[] = {DAG.getTargetExtractSubreg(Sub, DL, MVT::i32, Pol), Guard[0], Guard[1]};
    return SDValue(DAG.getMachineNode(SASS::R2UR, DL, MVT::i32, Ops), 0);
  };
  SDValue Ops[] = {DAG.getTargetConstant(SASS::UGPR64RegClassID, DL, MVT::i32), Half(SASS::sub0),
                   DAG.getTargetConstant(SASS::sub0, DL, MVT::i32), Half(SASS::sub1),
                   DAG.getTargetConstant(SASS::sub1, DL, MVT::i32)};
  return SDValue(DAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL, MVT::i64, Ops), 0);
}

//===----------------------------------------------------------------------===//
// Uniform-operand loops
//===----------------------------------------------------------------------===//

/// UNIU_SYNCS_EXCH: the operands are warp-uniform and an mbarrier.init by
/// every active thread is one init, so it is issued once with any thread's
/// values. Back-to-back inits share one FENCE.VIEW.ASYNC.S and wait for their
/// completions together, as ptxas's barrier setup does.
static MachineBasicBlock *expandUniformOperands(MachineInstr &MI, MachineBasicBlock *BB) {
  MachineFunction &MF = *BB->getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI.getDebugLoc();
  MachineBasicBlock::iterator At = MI.getIterator();
  auto Build = [&](unsigned O, Register Def = Register()) {
    MachineInstrBuilder MIB = BuildMI(*BB, At, DL, TII.get(O));
    if (Def)
      MIB.addDef(Def);
    return MIB;
  };
  auto ToUR = [&](const MachineOperand &MO, unsigned Sub = 0) {
    Register U = MRI.createVirtualRegister(&SASS::UGPR32RegClass);
    Build(SASS::R2UR_G, U).addReg(MO.getReg(), {}, Sub).addReg(SASS::PT).addImm(0);
    return U;
  };
  auto Pair = [&](Register Lo, Register Hi) {
    Register U = MRI.createVirtualRegister(&SASS::UGPR64RegClass);
    Build(TargetOpcode::REG_SEQUENCE, U).addReg(Lo).addImm(SASS::sub0).addReg(Hi).addImm(SASS::sub1);
    return U;
  };
  // A previous init with only register arithmetic and completion waits
  // after it already fenced: reuse its fence, and move those waits after
  // this init so the inits complete in parallel.
  Register Fence;
  SmallVector<MachineInstr *, 8> Waits;
  for (MachineBasicBlock::iterator I = At; I != BB->begin();) {
    MachineInstr &P = *--I;
    if (P.getOpcode() == SASS::SBWAIT) {
      Waits.push_back(&P);
      continue;
    }
    if (P.getOpcode() == SASS::SYNCS_EXCH)
      Fence = P.getOperand(3).getReg();
    if (Fence || P.hasUnmodeledSideEffects() || P.mayLoadOrStore() || P.isCall())
      break;
  }
  if (!Fence)
    Waits.clear();
  Register A = ToUR(MI.getOperand(0));
  Register V = Pair(ToUR(MI.getOperand(1)), ToUR(MI.getOperand(2)));
  if (!Fence) {
    Fence = MRI.createVirtualRegister(&SASS::GPR32RegClass);
    Build(SASS::FENCE_VIEW_ASYNC_S, Fence).addReg(SASS::PT).addImm(0);
  }
  Register Done = MRI.createVirtualRegister(&SASS::GPR32RegClass);
  Build(SASS::SYNCS_EXCH, Done).addReg(A).addReg(V).addReg(Fence).addReg(SASS::PT).addImm(0);
  for (MachineInstr *W : reverse(Waits))
    BB->splice(MI.getIterator(), BB, W);
  // mbarrier.init completes before the barrier is used.
  BuildMI(*BB, MI.getIterator(), DL, TII.get(SASS::SBWAIT)).addReg(Done).addReg(SASS::PT).addImm(0);
  MI.eraseFromParent();
  return BB;
}

/// UNIU_ARRIVES_*: ARRIVES arrives once for every active thread on the
/// barrier at its uniform address; with a warp-uniform address that is one
/// ARRIVES by all of them (ptxas's form), no loop over the distinct values.
static MachineBasicBlock *expandUniformArrive(MachineInstr &MI, MachineBasicBlock *BB) {
  MachineFunction &MF = *BB->getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI.getDebugLoc();
  Register A = MRI.createVirtualRegister(&SASS::UGPR32RegClass);
  BuildMI(*BB, MI, DL, TII.get(SASS::R2UR_G), A)
      .addReg(MI.getOperand(0).getReg())
      .addReg(SASS::PT)
      .addImm(0);
  BuildMI(*BB, MI, DL,
          TII.get(MI.getOpcode() == SASS::UNIU_ARRIVES_ARVCNT ? SASS::ARRIVES_ARVCNT
                                                               : SASS::ARRIVES_TRANSCNT))
      .addReg(A)
      .addReg(SASS::PT)
      .addImm(0);
  MI.eraseFromParent();
  return BB;
}

MachineBasicBlock *SASS::expandPTXUniform(MachineInstr &MI, MachineBasicBlock *BB) {
  unsigned Opc = MI.getOpcode();
  if (Opc == SASS::UNIU_SYNCS_EXCH)
    return expandUniformOperands(MI, BB);
  if (Opc == SASS::UNIU_ARRIVES_ARVCNT || Opc == SASS::UNIU_ARRIVES_TRANSCNT)
    return expandUniformArrive(MI, BB);
  // _H forms carry an L2 cache policy, already in a uniform register pair,
  // before the guard operands.
  Register Pol;
  switch (Opc) {
  case SASS::UNIU_UBLKCP_SG_H: Opc = SASS::UNIU_UBLKCP_SG; break;
  case SASS::UNI_UBLKCP_SG_H: Opc = SASS::UNI_UBLKCP_SG; break;
  case SASS::UNI_UBLKCP_GS_H: Opc = SASS::UNI_UBLKCP_GS; break;
  case SASS::UNI_UBLKPF_L2_H: Opc = SASS::UNI_UBLKPF_L2; break;
  case SASS::UNI_UTMALDG_H: Opc = SASS::UNI_UTMALDG; break;
  case SASS::UNI_UTMASTG_H: Opc = SASS::UNI_UTMASTG; break;
  default: break;
  }
  if (Opc != MI.getOpcode())
    Pol = MI.getOperand(MI.getDesc().getNumOperands() - 3).getReg();
  // A bulk copy is issued once per active thread even with uniform operands:
  // then only their moves to uniform registers leave the loop.
  bool Hoist = Opc == SASS::UNIU_UBLKCP_SG;
  if (Hoist)
    Opc = SASS::UNI_UBLKCP_SG;
  switch (Opc) {
  case SASS::UNI_SYNCS_EXCH:
  case SASS::UNI_ARRIVES_ARVCNT:
  case SASS::UNI_ARRIVES_TRANSCNT:
  case SASS::UNI_UBLKCP_SG:
  case SASS::UNI_UBLKCP_GS:
  case SASS::UNI_UBLKPF_L2:
  case SASS::UNI_UTMACCTL_PF:
  case SASS::UNI_UTMALDG:
  case SASS::UNI_UTMASTG:
    break;
  default:
    return nullptr;
  }
  MachineFunction &MF = *BB->getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = MI.getDebugLoc();

  // BB -> Loop -> Tail, Loop -> Loop.
  MachineBasicBlock *Loop = MF.CreateMachineBasicBlock(BB->getBasicBlock());
  MachineBasicBlock *Tail = MF.CreateMachineBasicBlock(BB->getBasicBlock());
  MachineFunction::iterator Pos = std::next(BB->getIterator());
  MF.insert(Pos, Loop);
  MF.insert(Pos, Tail);
  Tail->splice(Tail->begin(), BB, std::next(MI.getIterator()), BB->end());
  Tail->transferSuccessorsAndUpdatePHIs(BB);
  BB->addSuccessor(Loop);
  Loop->addSuccessor(Loop);
  Loop->addSuccessor(Tail);

  auto Build = [&](MachineBasicBlock &B, unsigned O, Register Def = Register()) {
    MachineInstrBuilder MIB = BuildMI(&B, DL, TII.get(O));
    if (Def)
      MIB.addDef(Def);
    return MIB;
  };
  auto Guard = [](MachineInstrBuilder &MIB, Register P = SASS::PT, bool Neg = false) {
    MIB.addReg(P).addImm(Neg);
  };
  auto NewReg = [&](const TargetRegisterClass *RC) { return MRI.createVirtualRegister(RC); };

  // All threads that reach the instruction are pending.
  Register Init = NewReg(&SASS::PREDRegClass);
  {
    MachineInstrBuilder M = BuildMI(*BB, BB->end(), DL, TII.get(SASS::PMOV), Init);
    M.addReg(SASS::PT).addImm(0);
    Guard(M);
  }
  Register Pend = NewReg(&SASS::PREDRegClass), Lead = NewReg(&SASS::PREDRegClass),
           Next = NewReg(&SASS::PREDRegClass);
  BuildMI(Loop, DL, TII.get(TargetOpcode::PHI), Pend)
      .addReg(Init)
      .addMBB(BB)
      .addReg(Next)
      .addMBB(Loop);
  {
    MachineInstrBuilder M = Build(*Loop, SASS::ELECT_P, Lead);
    M.addReg(Pend);
    Guard(M);
  }
  // The leader's value of a 32-bit GPR operand (or a 64-bit pair's half).
  MachineBasicBlock &OpB = Hoist ? *BB : *Loop;
  auto ToUR = [&](const MachineOperand &MO, unsigned Sub = 0, Register Into = Register()) {
    Register U = Into ? Into : NewReg(&SASS::UGPR32RegClass);
    MachineInstrBuilder M = Build(OpB, SASS::R2UR_G, U);
    M.addReg(MO.getReg(), {}, Sub);
    Guard(M, Hoist ? Register(SASS::PT) : Lead);
    return U;
  };
  auto ToUR64 = [&](const MachineOperand &MO) {
    Register Lo = ToUR(MO, SASS::sub0), Hi = ToUR(MO, SASS::sub1);
    Register U = NewReg(&SASS::UGPR64RegClass);
    BuildMI(&OpB, DL, TII.get(TargetOpcode::REG_SEQUENCE), U)
        .addReg(Lo)
        .addImm(SASS::sub0)
        .addReg(Hi)
        .addImm(SASS::sub1);
    return U;
  };
  auto Pair = [&](Register Lo, Register Hi) {
    Register U = NewReg(&SASS::UGPR64RegClass);
    BuildMI(&OpB, DL, TII.get(TargetOpcode::REG_SEQUENCE), U)
        .addReg(Lo)
        .addImm(SASS::sub0)
        .addReg(Hi)
        .addImm(SASS::sub1);
    return U;
  };
  Register Done;
  // Threads the iteration retires (the leader), and the block that ends the
  // iteration.
  Register Retired = Lead;
  MachineBasicBlock *Latch = Loop;
  switch (Opc) {
  case SASS::UNI_SYNCS_EXCH: {
    Register A = ToUR(MI.getOperand(0));
    Register V = Pair(ToUR(MI.getOperand(1)), ToUR(MI.getOperand(2)));
    Register Fence = NewReg(&SASS::GPR32RegClass);
    MachineInstrBuilder F = Build(*Loop, SASS::FENCE_VIEW_ASYNC_S, Fence);
    Guard(F);
    Done = NewReg(&SASS::GPR32RegClass);
    MachineInstrBuilder M = Build(*Loop, SASS::SYNCS_EXCH, Done);
    M.addReg(A).addReg(V).addReg(Fence);
    Guard(M);
    break;
  }
  case SASS::UNI_ARRIVES_ARVCNT:
  case SASS::UNI_ARRIVES_TRANSCNT: {
    // ARRIVES arrives once for every active thread, so it is issued by
    // exactly the pending threads with the leader's barrier address:
    // Loop -> Body (ARRIVES) -> Latch, Loop -> Latch for the others.
    Register A = ToUR(MI.getOperand(0));
    Register AV = NewReg(&SASS::GPR32RegClass);
    {
      MachineInstrBuilder M = Build(*Loop, SASS::MOVru, AV);
      M.addReg(A);
      Guard(M);
    }
    Retired = NewReg(&SASS::PREDRegClass);
    {
      MachineInstrBuilder M = Build(*Loop, SASS::ISETPrr, Retired);
      M.addReg(MI.getOperand(0).getReg()).addReg(AV).addImm(4 /*EQ*/).addReg(Pend).addImm(0);
      Guard(M);
    }
    MachineBasicBlock *Body = MF.CreateMachineBasicBlock(BB->getBasicBlock());
    Latch = MF.CreateMachineBasicBlock(BB->getBasicBlock());
    MF.insert(Tail->getIterator(), Body);
    MF.insert(Tail->getIterator(), Latch);
    Loop->removeSuccessor(Loop);
    Loop->removeSuccessor(Tail);
    Loop->addSuccessor(Body);
    Loop->addSuccessor(Latch);
    Body->addSuccessor(Latch);
    Latch->addSuccessor(Loop);
    Latch->addSuccessor(Tail);
    {
      MachineInstrBuilder M = Build(*Loop, SASS::BRAc);
      M.addMBB(Latch);
      Guard(M, Retired, /*Neg=*/true);
    }
    MachineInstrBuilder M = Build(*Body, Opc == SASS::UNI_ARRIVES_ARVCNT ? SASS::ARRIVES_ARVCNT
                                                                         : SASS::ARRIVES_TRANSCNT);
    M.addReg(A);
    Guard(M);
    for (MachineInstr &Phi : Loop->phis())
      for (MachineOperand &MO : Phi.operands())
        if (MO.isMBB() && MO.getMBB() == Loop)
          MO.setMBB(Latch);
    break;
  }
  case SASS::UNI_UBLKCP_SG: {
    Register DM = Pair(ToUR(MI.getOperand(0)), ToUR(MI.getOperand(1)));
    Register Src = ToUR64(MI.getOperand(2));
    Register Cnt = ToUR(MI.getOperand(3));
    MachineInstrBuilder M = Build(*Loop, Pol ? SASS::UBLKCP_SG_D : SASS::UBLKCP_SG);
    M.addReg(DM).addReg(Src).addReg(Cnt);
    if (Pol)
      M.addReg(Pol);
    Guard(M);
    break;
  }
  case SASS::UNI_UBLKCP_GS: {
    Register Dst = ToUR64(MI.getOperand(0));
    Register Src = ToUR(MI.getOperand(1)), Cnt = ToUR(MI.getOperand(2));
    MachineInstrBuilder M = Build(*Loop, Pol ? SASS::UBLKCP_GS_D : SASS::UBLKCP_GS);
    M.addReg(Dst).addReg(Src).addReg(Cnt);
    if (Pol)
      M.addReg(Pol);
    Guard(M);
    break;
  }
  case SASS::UNI_UBLKPF_L2: {
    Register Src = ToUR64(MI.getOperand(0)), Cnt = ToUR(MI.getOperand(1));
    MachineInstrBuilder M = Build(*Loop, Pol ? SASS::UBLKPF_L2_D : SASS::UBLKPF_L2);
    M.addReg(Src).addReg(Cnt);
    if (Pol)
      M.addReg(Pol);
    Guard(M);
    break;
  }
  case SASS::UNI_UTMACCTL_PF: {
    Register Map = ToUR64(MI.getOperand(0));
    MachineInstrBuilder M = Build(*Loop, SASS::UTMACCTL_PF);
    M.addReg(Map);
    Guard(M);
    break;
  }
  case SASS::UNI_UTMALDG:
  case SASS::UNI_UTMASTG: {
    // The group must be consecutive aligned uniform registers: UR8-UR15.
    bool Load = Opc == SASS::UNI_UTMALDG;
    unsigned Dim = MI.getOperand(0).getImm();
    Register Map = ToUR64(MI.getOperand(1));
    // {smem, mbarrier (loads), coordinates...}, padded with zeros to 4 or 8.
    SmallVector<int, 8> Slots = {2};
    if (Load)
      Slots.push_back(3);
    for (unsigned I = 0; I < Dim; ++I)
      Slots.push_back((Load ? 4 : 3) + I);
    unsigned Size = Slots.size() <= 4 ? 4 : 8;
    Slots.resize(Size, -1);
    SmallVector<Register, 8> Group;
    for (auto [I, S] : enumerate(Slots)) {
      Register Phys = SASS::UGPR32RegClass.getRegister(8 + I);
      if (S >= 0) {
        ToUR(MI.getOperand(S), 0, Phys);
      } else {
        MachineInstrBuilder M = Build(*Loop, SASS::R2UR_G, Phys);
        M.addReg(SASS::RZ);
        Guard(M);
      }
      Group.push_back(Phys);
    }
    static const unsigned Ops[2][2][5] = {
        {{SASS::UTMASTG_1D, SASS::UTMASTG_2D, SASS::UTMASTG_3D, SASS::UTMASTG_4D,
          SASS::UTMASTG_5D},
         {SASS::UTMASTG_1D_D, SASS::UTMASTG_2D_D, SASS::UTMASTG_3D_D, SASS::UTMASTG_4D_D,
          SASS::UTMASTG_5D_D}},
        {{SASS::UTMALDG_1D, SASS::UTMALDG_2D, SASS::UTMALDG_3D, SASS::UTMALDG_4D,
          SASS::UTMALDG_5D},
         {SASS::UTMALDG_1D_D, SASS::UTMALDG_2D_D, SASS::UTMALDG_3D_D, SASS::UTMALDG_4D_D,
          SASS::UTMALDG_5D_D}}};
    MachineInstrBuilder M = Build(*Loop, Ops[Load][bool(Pol)][Dim - 1]);
    for (Register R : Group)
      M.addReg(R, RegState::Kill);
    M.addReg(Map);
    if (Pol)
      M.addReg(Pol);
    Guard(M);
    break;
  }
  }
  // Pending: still pending and not retired.
  {
    MachineInstrBuilder M = Build(*Latch, SASS::PLOP3, Next);
    M.addReg(Pend).addReg(Retired).addImm(0x30);
    Guard(M);
  }
  {
    MachineInstrBuilder M = Build(*Latch, SASS::BRA_U_ANY);
    M.addMBB(Loop);
    Guard(M, Next);
  }
  if (Done) {
    // mbarrier.init completes before the barrier is used.
    MachineInstrBuilder M = BuildMI(*Tail, Tail->begin(), DL, TII.get(SASS::SBWAIT));
    M.addReg(Done);
    Guard(M);
  }
  MI.eraseFromParent();
  return Tail;
}
