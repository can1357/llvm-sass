//===-- SASSISelLowering.h - SASS DAG lowering ------------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSISELLOWERING_H
#define LLVM_LIB_TARGET_SASS_SASSISELLOWERING_H

#include "MCTargetDesc/SASSBaseInfo.h"
#include "SASSSelectionDAGInfo.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class SASSSubtarget;

namespace SASS {
/// A load's cache-hint bits (InstFlagNA, InstFlagLTCMask: bits 1-3) as the
/// target memory-operand flags MOTargetFlag1-3, and back.
constexpr unsigned LoadHintBits = InstFlagNA | InstFlagLTCMask;
constexpr unsigned LoadHintMMOScale = MachineMemOperand::MOTargetFlag1 / InstFlagNA;
static_assert(LoadHintBits == 0xe &&
              MachineMemOperand::MOTargetFlag3 == 4 * MachineMemOperand::MOTargetFlag1);
inline MachineMemOperand::Flags loadHintToMMOFlags(unsigned Hint) {
  return MachineMemOperand::Flags((Hint & LoadHintBits) * LoadHintMMOScale);
}
inline unsigned loadHintOfMMOFlags(MachineMemOperand::Flags F) {
  return (unsigned(F) / LoadHintMMOScale) & LoadHintBits;
}
} // namespace SASS

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

  bool isSDNodeSourceOfDivergence(const SDNode *N, FunctionLoweringInfo *FLI,
                                  UniformityInfo *UA) const override;
  /// A full-warp shfl.sync.idx of a constant lane (SASSTTIImpl::
  /// getValueUniformity's lane broadcast): uniform whatever its source.
  bool isSDNodeAlwaysUniform(const SDNode *N) const override;
  /// PTX `ld` cache hints (SASS::LoadHintMD) for the AsmPrinter.
  MachineMemOperand::Flags
  getTargetMMOFlags(const Instruction &I) const override;
  /// Hopper: a 64/128-bit access of 32-bit elements (sass-wide-elts) keeps
  /// that type; re-typing it to its bitcast's 8/16-bit-element vector would
  /// split it into 32-bit LDG/STGs.
  bool isLoadBitCastBeneficial(EVT LoadVT, EVT BitcastVT,
                               const SelectionDAG &DAG,
                               const MachineMemOperand &MMO) const override;
  bool isStoreBitCastBeneficial(EVT StoreVT, EVT BitcastVT,
                                const SelectionDAG &DAG,
                                const MachineMemOperand &MMO) const override;
  /// isSDNodeSourceOfDivergence: the IR value of each value register.
  mutable DenseMap<Register, const Value *> RegValue;
  mutable const MachineFunction *RegValueMF = nullptr;
  mutable size_t RegValueSize = 0;
  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                         EVT VT) const override;
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override {
    return MVT::i32;
  }
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override {
    return VT == MVT::f32 || isHalfType(VT);
  }
  /// f32 fabs/fneg are FADD |a| / -a (as ptxas does abs/neg.f32; NaNs become
  /// 7fffffff) and fold into source modifiers: keep the combiner from turning
  /// them into integer masks next to bitcasts. Both, or its or-sign-bit and
  /// fneg-of-bitcast folds undo each other forever. (Halves use LOP3 masks.)
  bool isFAbsFree(EVT VT) const override { return VT == MVT::f32; }
  bool isFNegFree(EVT VT) const override { return VT == MVT::f32; }
  bool isFPImmLegal(const APFloat &Imm, EVT VT,
                    bool ForCodeSize) const override {
    return VT == MVT::f32 || VT == MVT::f64 || VT == MVT::f16 || VT == MVT::bf16;
  }
  /// Atomics (SASSISelLoweringTypes.cpp): native ATOM*/RED forms stay, the
  /// rest become compare-and-swap loops; orderings become fences around a
  /// relaxed access at the instruction's scope.
  AtomicExpansionKind
  shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const override;
  AtomicExpansionKind shouldCastAtomicLoadInIR(LoadInst *LI) const override;
  AtomicExpansionKind shouldCastAtomicStoreInIR(StoreInst *SI) const override;
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
  bool getAddrModeArguments(const IntrinsicInst *II,
                            SmallVectorImpl<Value *> &Ops,
                            Type *&AccessTy) const override;
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
  /// Calls to SASS builtins standing for instructions (SASS::FChkFn,
  /// L2HintLoadFn, L2HintStoreFn): the chain, with the result in \p InVals;
  /// null for other calls.
  SDValue lowerBuiltinCall(CallLoweringInfo &CLI,
                           SmallVectorImpl<SDValue> &InVals) const;
  /// PTX ld/st with an L2 cache policy (SASS::L2HintLoadFn/L2HintStoreFn
  /// calls) as SASSISD::LDH*/STH memory nodes (SASSISelPTX.cpp); null for
  /// other calls.
  SDValue lowerL2HintCall(CallLoweringInfo &CLI,
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
