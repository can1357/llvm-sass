//===-- SASSISelLowering.h - SASS DAG lowering ------------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSISELLOWERING_H
#define LLVM_LIB_TARGET_SASS_SASSISELLOWERING_H

#include "SASSSelectionDAGInfo.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class SASSSubtarget;

/// Legal types: i1 (predicates), i32/f32 (R), i64 (aligned R pairs).
/// Kernel parameters are pure loads from constant bank 0. Shared pointers are
/// 32-bit window addresses (the module must be built with -fcuda-short-ptr);
/// static __shared__ variables are laid out by SASS::sharedOffset. Aligned
/// 2/4 x 32-bit vector accesses become single 64/128-bit loads and stores.
class SASSTargetLowering : public TargetLowering {
public:
  SASSTargetLowering(const TargetMachine &TM, const SASSSubtarget &ST);

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;
  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                         EVT VT) const override;
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
    return MVT::i32;
  }
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override {
    return VT == MVT::f32;
  }
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override {
    return VT == MVT::f32;
  }
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;

private:
  SDValue lowerSETCC64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMUL64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShift64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFunnelShift(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorStore(SDValue Op, SelectionDAG &DAG) const;
};

} // namespace llvm

#endif
