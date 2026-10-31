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

  /// Memory behaviour of ldmatrix/stmatrix/cp.async intrinsics, so their
  /// machine instructions carry shared-memory operands for alias analysis.
  void getTgtMemIntrinsic(SmallVectorImpl<IntrinsicInfo> &Infos,
                          const CallBase &I, MachineFunction &MF,
                          unsigned Intrinsic) const override;

  /// 64-bit adds (IADD.64) stay on Blackwell and become IADD3/IADD3.X carry
  /// chains on Hopper (SASSInstrInfo::expandAdd64).
  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *MBB) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                         EVT VT) const override;
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
    return MVT::i32;
  }
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override {
    return VT == MVT::f32 || isHalfType(VT);
  }
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override {
    return VT == MVT::f32 || VT == MVT::f64 || VT == MVT::f16 || VT == MVT::bf16;
  }
  /// Atomics (SASSISelLoweringTypes.cpp): native ATOM*/RED forms stay, the
  /// rest become compare-and-swap loops; orderings become fences around a
  /// relaxed access at the instruction's scope.
  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const override;
  bool shouldInsertFencesForAtomic(const Instruction *I) const override;
  Instruction *emitLeadingFence(IRBuilderBase &Builder, Instruction *Inst,
                                AtomicOrdering Ord) const override;
  Instruction *emitTrailingFence(IRBuilderBase &Builder, Instruction *Inst,
                                 AtomicOrdering Ord) const override;
  /// f16/bf16 scalars and their packed pairs (SASSISelLoweringTypes.cpp).
  static bool isHalfType(EVT VT) {
    return VT == MVT::f16 || VT == MVT::bf16 || VT == MVT::v2f16 ||
           VT == MVT::v2bf16;
  }
  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS,
                             Instruction *I = nullptr) const override;
  /// Inline asm register constraints of the canonical wgmma.mma_async calls
  /// ("r", "f" -> GPR32, "l" -> GPR64; SASSWGMMA.cpp).
  ConstraintType getConstraintType(StringRef Constraint) const override;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

private:
  /// Operation actions for f16/bf16 and atomics (SASSISelLoweringTypes.cpp).
  void initTypeActions();
  /// Operation actions for nodes behind inline PTX (SASSISelPTX.cpp).
  void initPTXActions();
  /// Custom lowering of the nodes initPTXActions marks Custom (FP8
  /// conversions on i16); null if \p Op is not one of them.
  SDValue lowerPTXOp(SDValue Op, SelectionDAG &DAG) const;
  bool replacePTXResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                         SelectionDAG &DAG) const;
  /// Operation actions for rounding and conversions (SASSISelConv.cpp).
  void initConvActions();
  /// Calls to SASS builtins standing for instructions (SASS::FChkFn): the
  /// chain, with the result in \p InVals; null for other calls.
  SDValue lowerBuiltinCall(CallLoweringInfo &CLI,
                           SmallVectorImpl<SDValue> &InVals) const;
  /// Custom lowering of the nodes initTypeActions marks Custom; a null
  /// SDValue means "expand".
  SDValue lowerTypesOp(SDValue Op, SelectionDAG &DAG) const;
  /// Results of illegal-typed nodes from initTypeActions (i16 bitcasts of
  /// halves, v2f32 -> v2f16 rounding); false if \p N is not one of them.
  bool replaceTypesResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                           SelectionDAG &DAG) const;
  SDValue lowerSETCC64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerMUL64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerShift64(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerFunnelShift(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  /// Casts to/from addrspace(4) (__constant__ data, kernel parameters):
  /// diagnoses generic pointers into constant banks; returns \p Op for casts
  /// it does not own.
  SDValue lowerConstAddrSpaceCast(SDValue Op, SelectionDAG &DAG) const;
  /// Local (addrspace 5) <-> generic casts (SASSLocalMemory.cpp).
  SDValue lowerLocalAddrSpaceCast(SDValue Op, SelectionDAG &DAG) const;
  SDValue lowerVectorStore(SDValue Op, SelectionDAG &DAG) const;
};

} // namespace llvm

#endif
