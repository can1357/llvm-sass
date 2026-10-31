//===-- SASSISelConv.cpp - Rounding and conversion lowering ---------------===//
//
// Operation actions for f32 rounding (FRND, SASSInstrConv.td) and f64
// (conversions, DADD/DMUL/DFMA/DSETP; the rest expands or is unsupported),
// and the builtin call SASSLowerIEEE emits for FCHK.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSISelLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/GlobalValue.h"

using namespace llvm;

void SASSTargetLowering::initConvActions() {
  setOperationAction({ISD::FFLOOR, ISD::FCEIL, ISD::FTRUNC, ISD::FRINT,
                      ISD::FNEARBYINT, ISD::FROUNDEVEN},
                     MVT::f32, Legal);

  for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op)
    setOperationAction(Op, MVT::f64, Expand);
  setOperationAction({ISD::FADD, ISD::FSUB, ISD::FMUL, ISD::FMA, ISD::FNEG,
                      ISD::FABS, ISD::SETCC, ISD::SELECT, ISD::ConstantFP,
                      ISD::BITCAST, ISD::FP_EXTEND},
                     MVT::f64, Legal);
  // Loads and stores move the bits as an i64.
  setOperationAction({ISD::LOAD, ISD::STORE}, MVT::f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f64, MVT::i64);
  AddPromotedToType(ISD::STORE, MVT::f64, MVT::i64);
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
}

SDValue SASSTargetLowering::lowerBuiltinCall(CallLoweringInfo &CLI,
                                             SmallVectorImpl<SDValue> &InVals) const {
  auto *G = dyn_cast<GlobalAddressSDNode>(CLI.Callee);
  if (!G || G->getGlobal()->getName() != SASS::FChkFn)
    return SDValue();
  InVals.push_back(CLI.DAG.getNode(SASSISD::FCHK, CLI.DL, MVT::i1, CLI.OutVals[0],
                                   CLI.OutVals[1]));
  return CLI.Chain;
}
