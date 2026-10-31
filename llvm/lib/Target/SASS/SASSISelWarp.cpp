//===-- SASSISelWarp.cpp - Warp collective selection ----------------------===//
//
// shfl.sync (idx/up/down/bfly, with and without the in-range predicate),
// vote.sync (ballot/any/all/uni), activemask, match.sync (any/all, 32/64-bit)
// and bar.warp.sync, in ptxas's forms: a member mask other than all-ones is
// a WARPSYNC on it right before the collective; lane and clamp/width words
// are immediates when constant (SHFL.UP's zero clamp word is RZ).
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSInstrInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace llvm;

namespace {

class WarpSelector {
  SelectionDAG &DAG;
  SDNode *N;
  SDLoc DL;
  SDValue Chain;

public:
  WarpSelector(SelectionDAG &DAG, SDNode *N)
      : DAG(DAG), N(N), DL(N), Chain(N->getOperand(0)) {}
  bool select();

private:
  SDValue arg(unsigned I) const { return N->getOperand(I + 2); }
  SDValue imm(int64_t V) { return DAG.getTargetConstant(V, DL, MVT::i32); }
  MachineSDNode *emit(unsigned Opc, ArrayRef<EVT> VTs, ArrayRef<SDValue> Ops) {
    SmallVector<SDValue, 8> All(Ops);
    All.push_back(DAG.getRegister(SASS::PT, MVT::i1));
    All.push_back(imm(0));
    All.push_back(Chain);
    SmallVector<EVT, 3> Tys(VTs);
    Tys.push_back(MVT::Other);
    MachineSDNode *M = DAG.getMachineNode(Opc, DL, DAG.getVTList(Tys), All);
    Chain = SDValue(M, Tys.size() - 1);
    return M;
  }
  /// A register operand; constants are materialized (RZ for zero).
  SDValue reg(SDValue V) {
    if (auto *C = dyn_cast<ConstantSDNode>(V)) {
      if (C->isZero())
        return DAG.getRegister(SASS::RZ, MVT::i32);
      SmallVector<SDValue, 3> Ops = {imm(C->getZExtValue()),
                                     DAG.getRegister(SASS::PT, MVT::i1), imm(0)};
      return SDValue(DAG.getMachineNode(SASS::MOVri, DL, MVT::i32, Ops), 0);
    }
    return V;
  }
  /// A predicate operand; constants become PT or a PMOV of !PT.
  SDValue pred(SDValue V) {
    auto *C = dyn_cast<ConstantSDNode>(V);
    if (!C)
      return V;
    SDValue PT = DAG.getRegister(SASS::PT, MVT::i1);
    if (!C->isZero())
      return PT;
    SDValue Ops[] = {PT, imm(1), PT, imm(0)};
    return SDValue(DAG.getMachineNode(SASS::PMOV, DL, MVT::i1, Ops), 0);
  }
  /// WARPSYNC on a member mask other than the full warp.
  void sync(SDValue Mask) {
    if (auto *C = dyn_cast<ConstantSDNode>(Mask))
      if (C->getZExtValue() == 0xffffffffu)
        return;
    emit(SASS::WARPSYNC_R, {}, {reg(Mask)});
  }
  void done(ArrayRef<SDValue> Vals) {
    for (auto [I, V] : enumerate(Vals))
      DAG.ReplaceAllUsesOfValueWith(SDValue(N, I), V);
    DAG.ReplaceAllUsesOfValueWith(SDValue(N, N->getNumValues() - 1), Chain);
    DAG.RemoveDeadNode(N);
  }
  bool shfl(unsigned Mode, bool WithPred);
};

} // namespace

// Opcode tables in TableGen's (alphabetical) order are not contiguous;
// index them by mode and operand form explicitly.
static const unsigned ShflOpc[2][4][4] = {
    {{SASS::SHFL_IDX_rr, SASS::SHFL_IDX_ir, SASS::SHFL_IDX_ri, SASS::SHFL_IDX_ii},
     {SASS::SHFL_UP_rr, SASS::SHFL_UP_ir, SASS::SHFL_UP_ri, SASS::SHFL_UP_ii},
     {SASS::SHFL_DOWN_rr, SASS::SHFL_DOWN_ir, SASS::SHFL_DOWN_ri, SASS::SHFL_DOWN_ii},
     {SASS::SHFL_BFLY_rr, SASS::SHFL_BFLY_ir, SASS::SHFL_BFLY_ri, SASS::SHFL_BFLY_ii}},
    {{SASS::SHFLP_IDX_rr, SASS::SHFLP_IDX_ir, SASS::SHFLP_IDX_ri, SASS::SHFLP_IDX_ii},
     {SASS::SHFLP_UP_rr, SASS::SHFLP_UP_ir, SASS::SHFLP_UP_ri, SASS::SHFLP_UP_ii},
     {SASS::SHFLP_DOWN_rr, SASS::SHFLP_DOWN_ir, SASS::SHFLP_DOWN_ri, SASS::SHFLP_DOWN_ii},
     {SASS::SHFLP_BFLY_rr, SASS::SHFLP_BFLY_ir, SASS::SHFLP_BFLY_ri, SASS::SHFLP_BFLY_ii}}};

bool WarpSelector::shfl(unsigned Mode, bool WithPred) {
  SDValue Mask = arg(0), Val = arg(1), Lane = arg(2), C = arg(3);
  sync(Mask);
  unsigned Form = 0;
  SDValue L = Lane, Cw = C;
  if (auto *K = dyn_cast<ConstantSDNode>(Lane); K && K->getZExtValue() < 32) {
    L = imm(K->getZExtValue());
    Form |= 1;
  } else {
    L = reg(Lane);
  }
  // SHFL.UP's clamp word is 0 for the full warp: ptxas writes RZ.
  if (auto *K = dyn_cast<ConstantSDNode>(C); K && !K->isZero() &&
                                            K->getZExtValue() < (1 << 13)) {
    Cw = imm(K->getZExtValue());
    Form |= 2;
  } else {
    Cw = reg(C);
  }
  EVT VT = N->getValueType(0);
  SmallVector<EVT, 2> VTs = {VT};
  if (WithPred)
    VTs.push_back(MVT::i1);
  MachineSDNode *M = emit(ShflOpc[WithPred][Mode][Form], VTs, {Val, L, Cw});
  if (WithPred)
    done({SDValue(M, 0), SDValue(M, 1)});
  else
    done({SDValue(M, 0)});
  return true;
}

bool WarpSelector::select() {
  if (N->getOpcode() != ISD::INTRINSIC_W_CHAIN &&
      N->getOpcode() != ISD::INTRINSIC_VOID)
    return false;
  unsigned ID = N->getConstantOperandVal(1);
  switch (ID) {
  case Intrinsic::nvvm_shfl_sync_idx_i32:
  case Intrinsic::nvvm_shfl_sync_idx_f32:
    return shfl(0, false);
  case Intrinsic::nvvm_shfl_sync_up_i32:
  case Intrinsic::nvvm_shfl_sync_up_f32:
    return shfl(1, false);
  case Intrinsic::nvvm_shfl_sync_down_i32:
  case Intrinsic::nvvm_shfl_sync_down_f32:
    return shfl(2, false);
  case Intrinsic::nvvm_shfl_sync_bfly_i32:
  case Intrinsic::nvvm_shfl_sync_bfly_f32:
    // Full-warp butterflies keep their SASSInstrInfo.td patterns.
    if (auto *M = dyn_cast<ConstantSDNode>(arg(0)))
      if (auto *C = dyn_cast<ConstantSDNode>(arg(3)))
        if (M->getZExtValue() == 0xffffffffu && C->getZExtValue() == 31)
          return false;
    return shfl(3, false);
  case Intrinsic::nvvm_shfl_sync_idx_i32p:
  case Intrinsic::nvvm_shfl_sync_idx_f32p:
    return shfl(0, true);
  case Intrinsic::nvvm_shfl_sync_up_i32p:
  case Intrinsic::nvvm_shfl_sync_up_f32p:
    return shfl(1, true);
  case Intrinsic::nvvm_shfl_sync_down_i32p:
  case Intrinsic::nvvm_shfl_sync_down_f32p:
    return shfl(2, true);
  case Intrinsic::nvvm_shfl_sync_bfly_i32p:
  case Intrinsic::nvvm_shfl_sync_bfly_f32p:
    return shfl(3, true);
  case Intrinsic::nvvm_vote_ballot_sync:
  case Intrinsic::nvvm_vote_any_sync:
  case Intrinsic::nvvm_vote_all_sync:
  case Intrinsic::nvvm_vote_uni_sync: {
    sync(arg(0));
    unsigned Opc = ID == Intrinsic::nvvm_vote_ballot_sync ? SASS::VOTE_BALLOT
                   : ID == Intrinsic::nvvm_vote_any_sync  ? SASS::VOTE_ANY
                   : ID == Intrinsic::nvvm_vote_all_sync  ? SASS::VOTE_ALL
                                                          : SASS::VOTE_EQ;
    MachineSDNode *M = emit(Opc, {N->getValueType(0)}, {pred(arg(1))});
    done({SDValue(M, 0)});
    return true;
  }
  case Intrinsic::nvvm_activemask: {
    MachineSDNode *M = emit(SASS::VOTE_ACTIVE, {MVT::i32}, {});
    done({SDValue(M, 0)});
    return true;
  }
  case Intrinsic::nvvm_match_any_sync_i32:
  case Intrinsic::nvvm_match_any_sync_i64: {
    sync(arg(0));
    bool Wide = ID == Intrinsic::nvvm_match_any_sync_i64;
    MachineSDNode *M =
        emit(Wide ? SASS::MATCH_ANY64 : SASS::MATCH_ANY, {MVT::i32}, {arg(1)});
    done({SDValue(M, 0)});
    return true;
  }
  case Intrinsic::nvvm_match_all_sync_i32p:
  case Intrinsic::nvvm_match_all_sync_i64p: {
    sync(arg(0));
    bool Wide = ID == Intrinsic::nvvm_match_all_sync_i64p;
    MachineSDNode *M = emit(Wide ? SASS::MATCH_ALL64 : SASS::MATCH_ALL,
                            {MVT::i32, MVT::i1}, {arg(1)});
    done({SDValue(M, 0), SDValue(M, 1)});
    return true;
  }
  case Intrinsic::nvvm_bar_warp_sync: {
    if (auto *C = dyn_cast<ConstantSDNode>(arg(0));
        C && C->getZExtValue() == 0xffffffffu)
      emit(SASS::WARPSYNC_NOP, {}, {});
    else
      emit(SASS::WARPSYNC_R, {}, {reg(arg(0))});
    done({});
    return true;
  }
  default:
    return false;
  }
}

bool SASS::selectWarpIntrinsic(SelectionDAG &DAG, SDNode *N) {
  return WarpSelector(DAG, N).select();
}
