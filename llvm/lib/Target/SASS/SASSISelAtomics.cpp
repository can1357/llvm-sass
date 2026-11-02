//===-- SASSISelAtomics.cpp - Atomics, fences and coherent accesses -------===//
//
// Selection of atomic read-modify-write, compare-and-swap, atomic load/store,
// volatile load/store and fence nodes. The SASS form depends on the address
// space (ATOMG/REDG global, ATOM/RED generic, ATOMS shared), the operation,
// whether the old value is used (RED otherwise) and the memory-model scope
// (.STRONG.SM/.GPU/.SYS), which the generated matcher cannot express.
// Orderings were already split into fences + relaxed accesses by
// AtomicExpand (SASSTargetLowering::emitLeadingFence/emitTrailingFence).
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "SASSSubtarget.h"
#include "MCTargetDesc/SASSMCTargetDesc.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

namespace {

enum Space { Global, Generic, Shared };
enum Scope { ScopeCTA = 0, ScopeGPU = 1, ScopeSYS = 2 };

/// Instruction set of one RMW operation: ATOMG, ATOM, REDG, RED, ATOMS
/// (0 where the form does not exist).
struct RMWOps {
  unsigned G, F, RG, RF, S;
};

#define RMW(N) {SASS::N##G, SASS::N##F, SASS::N##RG, SASS::N##RF, SASS::N##S}
#define RMW_NORED(N) {SASS::N##G, SASS::N##F, 0, 0, SASS::N##S}
#define RMW_NOSHARED(N) {SASS::N##G, SASS::N##F, SASS::N##RG, SASS::N##RF, 0}

std::optional<RMWOps> rmwOps(unsigned Opc, EVT VT) {
  bool Wide = VT == MVT::i64;
  switch (Opc) {
  case ISD::ATOMIC_LOAD_ADD:
    return Wide ? RMWOps RMW_NOSHARED(AT_ADD64) : RMWOps RMW(AT_ADD);
  case ISD::ATOMIC_LOAD_MIN:
    return Wide ? RMWOps RMW_NOSHARED(AT_SMIN64) : RMWOps RMW(AT_SMIN);
  case ISD::ATOMIC_LOAD_MAX:
    return Wide ? RMWOps RMW_NOSHARED(AT_SMAX64) : RMWOps RMW(AT_SMAX);
  case ISD::ATOMIC_LOAD_UMIN:
    return Wide ? RMWOps RMW_NOSHARED(AT_UMIN64) : RMWOps RMW(AT_UMIN);
  case ISD::ATOMIC_LOAD_UMAX:
    return Wide ? RMWOps RMW_NOSHARED(AT_UMAX64) : RMWOps RMW(AT_UMAX);
  case ISD::ATOMIC_LOAD_AND:
    return Wide ? RMWOps RMW_NOSHARED(AT_AND64) : RMWOps RMW(AT_AND);
  case ISD::ATOMIC_LOAD_OR:
    return Wide ? RMWOps RMW_NOSHARED(AT_OR64) : RMWOps RMW(AT_OR);
  case ISD::ATOMIC_LOAD_XOR:
    return Wide ? RMWOps RMW_NOSHARED(AT_XOR64) : RMWOps RMW(AT_XOR);
  case ISD::ATOMIC_SWAP:
    return Wide ? RMWOps RMW_NORED(AT_EXCH64) : RMWOps RMW_NORED(AT_EXCH);
  case ISD::ATOMIC_LOAD_UINC_WRAP:
    if (!Wide)
      return RMWOps RMW(AT_INC);
    break;
  case ISD::ATOMIC_LOAD_UDEC_WRAP:
    if (!Wide)
      return RMWOps RMW(AT_DEC);
    break;
  case ISD::ATOMIC_LOAD_FADD:
    if (VT == MVT::f32)
      return RMWOps{SASS::AT_FADDG, SASS::AT_FADDF, SASS::AT_FADDRG,
                    SASS::AT_FADDRF, 0};
    // Packed halves: generic reductions are ATOM (nvcc writes RZ).
    if (VT == MVT::v2f16)
      return RMWOps{SASS::AT_HADD2G, SASS::AT_HADD2F, SASS::AT_HADD2RG, 0, 0};
    if (VT == MVT::v2bf16)
      return RMWOps{SASS::AT_BADD2G, SASS::AT_BADD2F, SASS::AT_BADD2RG, 0, 0};
    break;
  default:
    break;
  }
  return std::nullopt;
}

Space spaceOf(const MemSDNode *M) {
  switch (M->getAddressSpace()) {
  case 1:
    return Global;
  case 3:
    return Shared;
  default:
    return Generic;
  }
}

Scope scopeOf(const MemSDNode *M, const LLVMContext &Ctx) {
  SyncScope::ID ID = M->getSyncScopeID();
  if (ID == SyncScope::System)
    return ScopeSYS;
  StringRef Name = Ctx.getSyncScopeName(ID).value_or("");
  if (Name == "block" || Name == "singlethread" || Name == "wavefront")
    return ScopeCTA;
  if (Name == "device" || Name == "cluster" || Name == "agent")
    return ScopeGPU;
  return ScopeSYS;
}

/// [base + imm] with a signed 24-bit immediate, as baseOffset in
/// SASSISelDAGToDAG.cpp.
void address(SelectionDAG &DAG, SDValue Addr, SDValue &Base, SDValue &Off) {
  SDLoc DL(Addr);
  if (DAG.isBaseWithConstantOffset(Addr)) {
    int64_t C = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<24>(C)) {
      Base = Addr.getOperand(0);
      Off = DAG.getSignedTargetConstant(C, DL, MVT::i32);
      return;
    }
  }
  Base = Addr;
  Off = DAG.getTargetConstant(0, DL, MVT::i32);
}

class AtomicSelector {
public:
  AtomicSelector(SelectionDAG &DAG, SDNode *N)
      : DAG(DAG), N(N), DL(N), PT(DAG.getRegister(SASS::PT, MVT::i1)),
        Zero(DAG.getTargetConstant(0, DL, MVT::i32)) {}

  SDNode *select();

private:
  SelectionDAG &DAG;
  SDNode *N;
  SDLoc DL;
  SDValue PT, Zero;

  MachineSDNode *build(unsigned Opc, ArrayRef<EVT> VTs,
                       SmallVector<SDValue, 8> Ops, SDValue Chain) {
    Ops.append({PT, Zero, Chain});
    MachineSDNode *M = DAG.getMachineNode(Opc, DL, DAG.getVTList(VTs), Ops);
    if (auto *Mem = dyn_cast<MemSDNode>(N))
      DAG.setNodeMemRefs(M, {Mem->getMemOperand()});
    return M;
  }
  SDValue scopeOp(Scope S) { return DAG.getTargetConstant(S, DL, MVT::i32); }
  SDNode *fence(AtomicOrdering Ord, Scope S, SDValue Chain);
  SDNode *rmw(AtomicSDNode *A);
  SDNode *cmpSwap(AtomicSDNode *A);
  SDNode *strongLoad(MemSDNode *M, ISD::LoadExtType Ext, Scope S);
  SDNode *strongStore(MemSDNode *M, SDValue Val, Scope S);
  SDNode *sharedLoad(AtomicSDNode *A);
};

/// A relaxed shared-memory atomic load (e.g. the initial load of a
/// compare-and-swap loop): a plain LDS.
SDNode *AtomicSelector::sharedLoad(AtomicSDNode *A) {
  bool Signed = A->getExtensionType() == ISD::SEXTLOAD;
  unsigned Opc;
  switch (A->getMemoryVT().getSizeInBits()) {
  case 8:
    Opc = Signed ? SASS::LDSS8 : SASS::LDSU8;
    break;
  case 16:
    Opc = Signed ? SASS::LDSS16 : SASS::LDSU16;
    break;
  case 32:
    Opc = SASS::LDS32;
    break;
  case 64:
    Opc = SASS::LDS64;
    break;
  default:
    return nullptr;
  }
  SDValue Base, Off;
  address(DAG, A->getBasePtr(), Base, Off);
  return build(Opc, {A->getValueType(0), MVT::Other}, {Base, Off}, A->getChain());
}

SDNode *AtomicSelector::fence(AtomicOrdering Ord, Scope S, SDValue Chain) {
  // A CTA-scope MEMBAR completes asynchronously (ptxas guards it with a
  // scoreboard); the GPU-scope sequence below is the stronger, synchronous
  // equivalent, so CTA fences use it too except for the plain barrier form.
  SmallVector<unsigned, 4> Seq;
  bool SC = Ord == AtomicOrdering::SequentiallyConsistent;
  bool Release = isReleaseOrStronger(Ord);
  if (Release) {
    if (S == ScopeSYS)
      Seq.push_back(SC ? SASS::MEMBAR_SC_SYS : SASS::MEMBAR_ALL_SYS);
    else
      Seq.push_back(SC ? SASS::MEMBAR_SC_GPU : SASS::MEMBAR_ALL_GPU);
    Seq.append({SASS::ERRBAR, SASS::CGAERRBAR});
  }
  if (isAcquireOrStronger(Ord))
    Seq.push_back(SASS::CCTL_IVALL);
  SDNode *Last = nullptr;
  for (unsigned Opc : Seq) {
    Last = DAG.getMachineNode(Opc, DL, MVT::Other, {PT, Zero, Chain});
    Chain = SDValue(Last, 0);
  }
  return Last;
}

SDNode *AtomicSelector::rmw(AtomicSDNode *A) {
  EVT VT = A->getValueType(0);
  std::optional<RMWOps> Ops = rmwOps(A->getOpcode(), VT);
  if (!Ops)
    report_fatal_error("SASS: unsupported atomic operation");
  bool Used = !SDValue(A, 0).use_empty();
  Space Sp = spaceOf(A);
  Scope S = scopeOf(A, *DAG.getContext());
  SDValue Base, Off;
  address(DAG, A->getBasePtr(), Base, Off);
  SDValue Val = A->getVal();
  if (Sp == Shared) {
    if (!Ops->S)
      report_fatal_error("SASS: unsupported shared-memory atomic");
    return build(Ops->S, {VT, MVT::Other}, {Base, Off, Val}, A->getChain());
  }
  unsigned RedOpc = Sp == Global ? Ops->RG : Ops->RF;
  if (!Used && RedOpc)
    return build(RedOpc, {MVT::Other}, {Base, Off, Val, scopeOp(S)},
                 A->getChain());
  return build(Sp == Global ? Ops->G : Ops->F, {VT, MVT::Other},
               {Base, Off, Val, scopeOp(S)}, A->getChain());
}

SDNode *AtomicSelector::cmpSwap(AtomicSDNode *A) {
  EVT VT = A->getValueType(0);
  bool Wide = VT == MVT::i64;
  Space Sp = spaceOf(A);
  SDValue Base, Off;
  address(DAG, A->getBasePtr(), Base, Off);
  SmallVector<SDValue, 8> Ops = {Base, Off, A->getOperand(2), A->getOperand(3)};
  unsigned Opc;
  switch (Sp) {
  case Shared:
    Opc = Wide ? SASS::AT_CAS64S : SASS::AT_CASS;
    break;
  case Global:
    Opc = Wide ? SASS::AT_CAS64G : SASS::AT_CASG;
    Ops.push_back(scopeOp(scopeOf(A, *DAG.getContext())));
    break;
  case Generic:
    Opc = Wide ? SASS::AT_CAS64F : SASS::AT_CASF;
    Ops.push_back(scopeOp(scopeOf(A, *DAG.getContext())));
    break;
  }
  return build(Opc, {VT, MVT::Other}, Ops, A->getChain());
}

SDNode *AtomicSelector::strongLoad(MemSDNode *M, ISD::LoadExtType Ext,
                                   Scope S) {
  bool G = spaceOf(M) == Global;
  unsigned Bits = M->getMemoryVT().getSizeInBits();
  bool Signed = Ext == ISD::SEXTLOAD;
  unsigned Opc;
  switch (Bits) {
  case 8:
    Opc = G ? (Signed ? SASS::SLDGS8 : SASS::SLDGU8)
            : (Signed ? SASS::SLDFS8 : SASS::SLDFU8);
    break;
  case 16:
    Opc = G ? (Signed ? SASS::SLDGS16 : SASS::SLDGU16)
            : (Signed ? SASS::SLDFS16 : SASS::SLDFU16);
    break;
  case 32:
    Opc = G ? SASS::SLDG32 : SASS::SLDF32;
    break;
  case 64:
    Opc = G ? SASS::SLDG64 : SASS::SLDF64;
    break;
  default:
    return nullptr;
  }
  SDValue Base, Off;
  address(DAG, M->getBasePtr(), Base, Off);
  return build(Opc, {M->getValueType(0), MVT::Other}, {Base, Off, scopeOp(S)},
               M->getChain());
}

SDNode *AtomicSelector::strongStore(MemSDNode *M, SDValue Val, Scope S) {
  bool G = spaceOf(M) == Global;
  unsigned Opc;
  switch (M->getMemoryVT().getSizeInBits()) {
  case 8:
    Opc = G ? SASS::SSTGU8 : SASS::SSTFU8;
    break;
  case 16:
    Opc = G ? SASS::SSTGU16 : SASS::SSTFU16;
    break;
  case 32:
    Opc = G ? SASS::SSTG32 : SASS::SSTF32;
    break;
  case 64:
    Opc = G ? SASS::SSTG64 : SASS::SSTF64;
    break;
  default:
    return nullptr;
  }
  SDValue Base, Off;
  address(DAG, M->getBasePtr(), Base, Off);
  return build(Opc, {MVT::Other}, {Base, Off, Val, scopeOp(S)}, M->getChain());
}

SDNode *AtomicSelector::select() {
  const LLVMContext &Ctx = *DAG.getContext();
  switch (N->getOpcode()) {
  case ISD::ATOMIC_FENCE: {
    auto Ord = static_cast<AtomicOrdering>(N->getConstantOperandVal(1));
    auto ID = static_cast<SyncScope::ID>(N->getConstantOperandVal(2));
    Scope S = ID == SyncScope::System ? ScopeSYS : ScopeGPU;
    StringRef Name = Ctx.getSyncScopeName(ID).value_or("");
    if (Name == "block")
      S = ScopeCTA;
    return fence(Ord, S, N->getOperand(0));
  }
  case ISD::INTRINSIC_VOID: {
    // membar.cta/gl/sys (__threadfence_block/__threadfence/_system).
    switch (N->getConstantOperandVal(1)) {
    case Intrinsic::nvvm_membar_cta:
      return fence(AtomicOrdering::SequentiallyConsistent, ScopeCTA,
                   N->getOperand(0));
    case Intrinsic::nvvm_membar_gl:
      return fence(AtomicOrdering::SequentiallyConsistent, ScopeGPU,
                   N->getOperand(0));
    case Intrinsic::nvvm_membar_sys:
      return fence(AtomicOrdering::SequentiallyConsistent, ScopeSYS,
                   N->getOperand(0));
    default:
      return nullptr;
    }
  }
  case ISD::ATOMIC_LOAD: {
    auto *A = cast<AtomicSDNode>(N);
    if (spaceOf(A) == Shared)
      return sharedLoad(A); // LDS is coherent within the CTA.
    return strongLoad(A, A->getExtensionType(), scopeOf(A, Ctx));
  }
  case ISD::ATOMIC_STORE: {
    auto *A = cast<AtomicSDNode>(N);
    if (spaceOf(A) == Shared)
      return nullptr;
    return strongStore(A, A->getVal(), scopeOf(A, Ctx));
  }
  case ISD::ATOMIC_CMP_SWAP:
    return cmpSwap(cast<AtomicSDNode>(N));
  case ISD::LOAD: {
    // volatile = relaxed at system scope (ptxas: LDG.E.STRONG.SYS).
    auto *L = cast<LoadSDNode>(N);
    EVT VT = L->getValueType(0);
    if (!L->isVolatile() || spaceOf(L) == Shared || !L->isUnindexed() ||
        (!VT.isScalarInteger() && VT != MVT::f32))
      return nullptr;
    return strongLoad(L, L->getExtensionType(), ScopeSYS);
  }
  case ISD::STORE: {
    auto *St = cast<StoreSDNode>(N);
    EVT VT = St->getValue().getValueType();
    if (!St->isVolatile() || spaceOf(St) == Shared || !St->isUnindexed() ||
        (!VT.isScalarInteger() && VT != MVT::f32))
      return nullptr;
    return strongStore(St, St->getValue(), ScopeSYS);
  }
  default:
    if (auto *A = dyn_cast<AtomicSDNode>(N))
      return rmw(A);
    return nullptr;
  }
}

} // namespace

SDNode *SASS::selectAtomic(SelectionDAG &DAG, SDNode *N) {
  if (N->isMachineOpcode())
    return nullptr;
  return AtomicSelector(DAG, N).select();
}
