//===-- SASSISelDAGToDAG.cpp - SASS instruction selection ------------------===//

#include "SASS.h"
#include "SASSSelectionDAGInfo.h"
#include "SASSTargetMachine.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
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

  void Select(SDNode *N) override;

  /// 64-bit base register pair plus a signed 24-bit offset (LDG/LD/STG/ST).
  bool SelectADDRri(SDValue Addr, SDValue &Base, SDValue &Offset);
  /// 32-bit shared-window address plus a signed 24-bit offset (LDS/STS).
  bool SelectADDRs(SDValue Addr, SDValue &Base, SDValue &Offset);

  bool isAddrSpace(const SDNode *N, unsigned AS) const {
    return cast<MemSDNode>(N)->getAddressSpace() == AS;
  }

  /// Global memory that is read-only for the whole kernel, so the
  /// non-coherent path (LDG.E.CONSTANT) is safe.
  bool isNonCoherent(const MemSDNode *N) const;

#include "SASSGenDAGISel.inc"

private:
  void selectVectorLoad(SDNode *N);
  void selectVectorStore(SDNode *N);
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

void SASSDAGToDAGISel::Select(SDNode *N) {
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;
  }
  switch (N->getOpcode()) {
  case SASSISD::LDV2:
  case SASSISD::LDV4:
    return selectVectorLoad(N);
  case SASSISD::STV2:
  case SASSISD::STV4:
    return selectVectorStore(N);
  default:
    break;
  }
  SelectCode(N);
}

static bool baseOffset(SelectionDAG &DAG, SDValue Addr, SDValue &Base,
                       SDValue &Offset) {
  SDLoc DL(Addr);
  if (DAG.isBaseWithConstantOffset(Addr)) {
    int64_t C = cast<ConstantSDNode>(Addr.getOperand(1))->getSExtValue();
    if (isInt<24>(C)) {
      Base = Addr.getOperand(0);
      Offset = DAG.getTargetConstant(C, DL, MVT::i32);
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

static const unsigned SubIdx[] = {SASS::sub0, SASS::sub1, SASS::sub2,
                                  SASS::sub3};

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
