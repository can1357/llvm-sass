//===-- SASSISelLoweringTypes.cpp - f16/bf16 and atomic lowering ----------===//
//
// Operation actions and custom lowering for the half-precision types
// (f16/bf16 scalars in the low half of a 32-bit register, v2f16/v2bf16 in a
// whole register: HADD2/HMUL2/HFMA2/HMNMX2/HSETP2, F2FP, PRMT) and for
// atomics. Instruction patterns live in SASSInstrTypes.td.
//
//===----------------------------------------------------------------------===//

#include "SASSISelLowering.h"
#include "SASSSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

using namespace llvm;

void SASSTargetLowering::initTypeActions() {
  for (MVT VT : {MVT::f16, MVT::bf16, MVT::v2f16, MVT::v2bf16}) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
      setOperationAction(Op, VT, Expand);
    setOperationAction({ISD::FADD, ISD::FSUB, ISD::FMUL, ISD::FMA, ISD::FNEG,
                        ISD::FABS, ISD::FMINNUM, ISD::FMAXNUM, ISD::SELECT,
                        ISD::BITCAST},
                       VT, Legal);
  }
  for (MVT VT : {MVT::f16, MVT::bf16}) {
    setOperationAction({ISD::LOAD, ISD::STORE, ISD::ConstantFP, ISD::SETCC,
                        ISD::FP_ROUND},
                       VT, Legal);
    // Everything else rounds once from f32, which is exact for these
    // (f32 has more than 2p+2 bits for p-bit halves) or is what nvcc does.
    for (unsigned Op :
         {ISD::FDIV, ISD::FREM, ISD::FSQRT, ISD::FSIN, ISD::FCOS, ISD::FEXP,
          ISD::FEXP2, ISD::FEXP10, ISD::FLOG, ISD::FLOG2, ISD::FLOG10,
          ISD::FPOW, ISD::FFLOOR, ISD::FCEIL, ISD::FTRUNC, ISD::FRINT,
          ISD::FNEARBYINT, ISD::FROUND, ISD::FROUNDEVEN, ISD::FMINIMUM,
          ISD::FMAXIMUM, ISD::FMINIMUMNUM, ISD::FMAXIMUMNUM}) {
      setOperationAction(Op, VT, Promote);
      AddPromotedToType(Op, VT, MVT::f32);
    }
    setLoadExtAction(ISD::EXTLOAD, MVT::f32, VT, Expand);
    setTruncStoreAction(MVT::f32, VT, Expand);
  }
  for (MVT VT : {MVT::v2f16, MVT::v2bf16}) {
    setOperationAction(ISD::BUILD_VECTOR, VT, Legal);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    // Loads and stores move the packed pair as an i32.
    setOperationAction({ISD::LOAD, ISD::STORE}, VT, Promote);
    AddPromotedToType(ISD::LOAD, VT, MVT::i32);
    AddPromotedToType(ISD::STORE, VT, MVT::i32);
  }
  // i16 is promoted to i32, so bitcasts between it and the halves are
  // rewritten in ReplaceNodeResults; two f32 roundings pack into one F2FP.
  setOperationAction(ISD::BITCAST, MVT::i16, Custom);
  setOperationAction(ISD::FP_ROUND, MVT::v2f32, Custom);
  // copysign.f32 is bitwise (NaN payloads survive); the default expansion
  // goes through FABS/FNEG, which are FADDs here.
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Custom);

  // ATOMG/ATOM/ATOMS/RED handle 32- and 64-bit accesses; narrower ones are
  // compare-and-swap loops on the containing word (AtomicExpand).
  setMaxAtomicSizeInBitsSupported(64);
  setMinCmpXchgSizeInBits(32);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, {MVT::i32, MVT::i64}, Expand);
  setOperationAction(ISD::ATOMIC_FENCE, MVT::Other, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_FADD, {MVT::v2f16, MVT::v2bf16}, Legal);
}

/// <2 x half> or <2 x bfloat>: one 32-bit word.
static bool isPackedHalf(const Type *T) {
  auto *VT = dyn_cast<FixedVectorType>(T);
  return VT && VT->getNumElements() == 2 &&
         (VT->getElementType()->isHalfTy() || VT->getElementType()->isBFloatTy());
}

TargetLowering::AtomicExpansionKind
SASSTargetLowering::shouldExpandAtomicRMWInIR(const AtomicRMWInst *AI) const {
  unsigned Bits = AI->getType()->getPrimitiveSizeInBits();
  bool Shared = AI->getPointerAddressSpace() == 3;
  if (Bits != 32 && Bits != 64)
    return AtomicExpansionKind::CmpXChg;
  switch (AI->getOperation()) {
  case AtomicRMWInst::Xchg:
    return AtomicExpansionKind::None;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin:
    // ATOMS has no 64-bit arithmetic (ptxas loops on ATOMS.CAST.64 too).
    return Shared && Bits == 64 ? AtomicExpansionKind::CmpXChg
                                : AtomicExpansionKind::None;
  case AtomicRMWInst::UIncWrap:
  case AtomicRMWInst::UDecWrap:
    return Bits == 32 ? AtomicExpansionKind::None
                      : AtomicExpansionKind::CmpXChg;
  case AtomicRMWInst::FAdd:
    // ATOMG/ATOM.ADD.F32.FTZ.RN / .F16x2.RN / .BF16x2.RN; shared memory has
    // no float add.
    return (AI->getType()->isFloatTy() || isPackedHalf(AI->getType())) && !Shared
               ? AtomicExpansionKind::None
               : AtomicExpansionKind::CmpXChg;
  default:
    return AtomicExpansionKind::CmpXChg;
  }
}

// The compare-and-swap loops (shared packed-half adds) load and swap the
// whole word as an integer.
TargetLowering::AtomicExpansionKind
SASSTargetLowering::shouldCastAtomicLoadInIR(LoadInst *LI) const {
  return isPackedHalf(LI->getType()) ? AtomicExpansionKind::CastToInteger
                                     : TargetLowering::shouldCastAtomicLoadInIR(LI);
}

TargetLowering::AtomicExpansionKind
SASSTargetLowering::shouldCastAtomicStoreInIR(StoreInst *SI) const {
  return isPackedHalf(SI->getValueOperand()->getType())
             ? AtomicExpansionKind::CastToInteger
             : TargetLowering::shouldCastAtomicStoreInIR(SI);
}

bool SASSTargetLowering::shouldInsertFencesForAtomic(
    const Instruction *I) const {
  return true;
}

// ptxas implements ordered accesses as a fence plus a relaxed (.STRONG)
// access at the same scope: release = leading MEMBAR.ALL, seq_cst = leading
// MEMBAR.SC, acquire/acq_rel/seq_cst = trailing L1 invalidation.
Instruction *SASSTargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                  Instruction *Inst,
                                                  AtomicOrdering Ord) const {
  if (!isReleaseOrStronger(Ord))
    return nullptr;
  return Builder.CreateFence(Ord == AtomicOrdering::SequentiallyConsistent
                                 ? Ord
                                 : AtomicOrdering::Release,
                             getAtomicSyncScopeID(Inst).value_or(
                                 SyncScope::System));
}

Instruction *SASSTargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (!isAcquireOrStronger(Ord))
    return nullptr;
  return Builder.CreateFence(AtomicOrdering::Acquire,
                             getAtomicSyncScopeID(Inst).value_or(
                                 SyncScope::System));
}

SDValue SASSTargetLowering::lowerTypesOp(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::EXTRACT_VECTOR_ELT:
    // Constant lanes are a copy (lane 0) or a PRMT (lane 1); variable ones
    // expand.
    return isa<ConstantSDNode>(Op.getOperand(1)) ? Op : SDValue();
  case ISD::FCOPYSIGN: {
    if (Op.getOperand(1).getValueType() != MVT::f32)
      return SDValue(); // Mixed widths: default expansion.
    SDLoc DL(Op);
    SDValue Mag = DAG.getBitcast(MVT::i32, Op.getOperand(0));
    SDValue Sgn = DAG.getBitcast(MVT::i32, Op.getOperand(1));
    SDValue R = DAG.getNode(
        ISD::OR, DL, MVT::i32,
        DAG.getNode(ISD::AND, DL, MVT::i32, Mag, DAG.getConstant(0x7fffffff, DL, MVT::i32)),
        DAG.getNode(ISD::AND, DL, MVT::i32, Sgn, DAG.getConstant(0x80000000, DL, MVT::i32)));
    return DAG.getBitcast(MVT::f32, R);
  }
  case ISD::ATOMIC_FENCE:
    // A single-thread fence only orders the compiler: keep the chain.
    if (Op.getConstantOperandVal(2) == SyncScope::SingleThread)
      return Op.getOperand(0);
    return Op;
  default:
    llvm_unreachable("unexpected custom lowering");
  }
}

bool SASSTargetLowering::replaceTypesResults(SDNode *N,
                                             SmallVectorImpl<SDValue> &Results,
                                             SelectionDAG &DAG) const {
  SDLoc DL(N);
  switch (N->getOpcode()) {
  case ISD::BITCAST: {
    SDValue Src = N->getOperand(0);
    EVT VT = N->getValueType(0);
    if (VT == MVT::i16 && (Src.getValueType() == MVT::f16 ||
                           Src.getValueType() == MVT::bf16)) {
      SDValue Bits = DAG.getNode(SASSISD::HALF_BITS, DL, MVT::i32, Src);
      Results.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i16, Bits));
      return true;
    }
    if (Src.getValueType() == MVT::i16 && (VT == MVT::f16 || VT == MVT::bf16)) {
      SDValue Wide = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Src);
      Results.push_back(DAG.getNode(SASSISD::BITS_HALF, DL, VT, Wide));
      return true;
    }
    return false;
  }
  case ISD::FP_ROUND: {
    EVT VT = N->getValueType(0);
    SDValue Src = N->getOperand(0);
    if (Src.getValueType() != MVT::v2f32 ||
        (VT != MVT::v2f16 && VT != MVT::v2bf16))
      return false;
    EVT EltVT = VT.getVectorElementType();
    SDValue Elts[2];
    for (unsigned I = 0; I < 2; ++I)
      Elts[I] = DAG.getNode(
          ISD::FP_ROUND, DL, EltVT,
          DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::f32, Src,
                      DAG.getVectorIdxConstant(I, DL)),
          N->getOperand(1));
    Results.push_back(DAG.getBuildVector(VT, DL, Elts));
    return true;
  }
  default:
    return false;
  }
}
