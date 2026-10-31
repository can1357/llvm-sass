//===-- SASSGlobals.cpp - Constant bank 0/3 data -----------------------------===//
//
// addrspace(4) holds two kinds of read-only data, both read with LDC:
//  * __constant__ variables: constant bank 3 (`.nv.constant3`), all of the
//    module's definitions laid out in declaration order. Each kernel's `#fn`
//    block declares the ones it uses with
//      #constant(3, name, offset, size[, initial bytes])
//    so its cubin carries their data and symbols (cuModuleGetGlobal).
//  * by-value kernel parameters: constant bank 0, addressed from
//    SASS::KernelParamsSymbol (SASSLowerKernelArgs).
// Addresses are plain bank offsets; each load's bank comes from the
// underlying object of its memory operand.
//
//===----------------------------------------------------------------------===//

#include "SASS.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static bool isConstData(const GlobalVariable &G) {
  return G.getAddressSpace() == 4 && G.getName() != SASS::KernelParamsSymbol;
}

uint64_t SASS::constOffset(const GlobalVariable &GV) {
  if (GV.isDeclaration())
    report_fatal_error("SASS: extern __constant__ variable '" + GV.getName() +
                       "' is not supported (no separate compilation)");
  const DataLayout &DL = GV.getDataLayout();
  uint64_t Off = 0;
  for (const GlobalVariable &G : GV.getParent()->globals()) {
    if (!isConstData(G) || G.isDeclaration())
      continue;
    Off = alignTo(Off, G.getAlign().value_or(DL.getPrefTypeAlign(G.getValueType())));
    if (&G == &GV)
      break;
    Off += DL.getTypeAllocSize(G.getValueType());
  }
  if (Off + DL.getTypeAllocSize(GV.getValueType()) > 0x10000)
    report_fatal_error("SASS: __constant__ data exceeds 64 KB at '" +
                       GV.getName() + "'");
  return Off;
}

unsigned SASS::constBank(const MemSDNode &N) {
  const Value *Ptr = N.getMemOperand()->getValue();
  SmallVector<const Value *, 4> Objs;
  if (Ptr)
    getUnderlyingObjects(Ptr, Objs, /*LI=*/nullptr, /*MaxLookup=*/0);
  std::optional<unsigned> Bank;
  for (const Value *O : Objs) {
    auto *GV = dyn_cast<GlobalVariable>(O);
    std::optional<unsigned> B;
    if (GV && GV->getName() == KernelParamsSymbol)
      B = 0;
    else if (GV && isConstData(*GV))
      B = 3;
    if (!B || (Bank && *Bank != *B)) {
      Bank.reset();
      break;
    }
    Bank = B;
  }
  if (!Bank) {
    std::string What = "no IR pointer";
    if (Ptr) {
      raw_string_ostream OS(What = "");
      Ptr->printAsOperand(OS, /*PrintType=*/false);
    }
    report_fatal_error(Twine("SASS: cannot tell which __constant__ variable "
                             "or by-value kernel parameter a constant load "
                             "reads (") +
                       What + ")");
  }
  return *Bank;
}

/// Writes the little-endian image of \p C into \p Out.
static void constBytes(const Constant *C, const DataLayout &DL,
                       MutableArrayRef<uint8_t> Out, const GlobalVariable &GV) {
  if (isa<ConstantAggregateZero, UndefValue, ConstantPointerNull>(C))
    return;
  auto Int = [&](APInt V) {
    V = V.zextOrTrunc(Out.size() * 8);
    for (unsigned I = 0; I < Out.size(); ++I)
      Out[I] = V.extractBitsAsZExtValue(8, I * 8);
  };
  if (auto *CI = dyn_cast<ConstantInt>(C))
    return Int(CI->getValue());
  if (auto *CF = dyn_cast<ConstantFP>(C))
    return Int(CF->getValueAPF().bitcastToAPInt());
  auto Elements = [&](unsigned N, Type *EltTy, auto Elt) {
    uint64_t Stride = DL.getTypeAllocSize(EltTy);
    uint64_t Store = DL.getTypeStoreSize(EltTy);
    for (unsigned I = 0; I < N; ++I)
      constBytes(Elt(I), DL, Out.slice(I * Stride, Store), GV);
  };
  if (auto *CDS = dyn_cast<ConstantDataSequential>(C))
    return Elements(CDS->getNumElements(), CDS->getElementType(),
                    [&](unsigned I) { return CDS->getElementAsConstant(I); });
  if (isa<ConstantArray, ConstantVector>(C)) {
    Type *EltTy = isa<ConstantArray>(C)
                      ? cast<ArrayType>(C->getType())->getElementType()
                      : cast<VectorType>(C->getType())->getElementType();
    return Elements(C->getNumOperands(), EltTy,
                    [&](unsigned I) { return C->getAggregateElement(I); });
  }
  if (auto *CS = dyn_cast<ConstantStruct>(C)) {
    const StructLayout *SL = DL.getStructLayout(CS->getType());
    for (unsigned I = 0, E = CS->getNumOperands(); I < E; ++I) {
      const Constant *Op = CS->getOperand(I);
      constBytes(Op, DL,
                 Out.slice(SL->getElementOffset(I),
                           DL.getTypeStoreSize(Op->getType())),
                 GV);
    }
    return;
  }
  report_fatal_error("SASS: unsupported initializer of __constant__ variable '" +
                     GV.getName() + "' (addresses of other variables need "
                     "relocations)");
}

static bool usedIn(const Value &V, const Function &F) {
  return any_of(V.users(), [&](const User *U) {
    if (auto *I = dyn_cast<Instruction>(U))
      return I->getFunction() == &F;
    return isa<Constant>(U) && usedIn(*U, F);
  });
}

void SASS::printConstantObjects(const Function &F, raw_ostream &OS) {
  const DataLayout &DL = F.getDataLayout();
  for (const GlobalVariable &G : F.getParent()->globals()) {
    if (!isConstData(G) || G.isDeclaration() || !usedIn(G, F))
      continue;
    if (G.getName().empty())
      report_fatal_error("SASS: unnamed __constant__ variable");
    uint64_t Size = DL.getTypeAllocSize(G.getValueType());
    SmallVector<uint8_t, 64> Bytes(Size, 0);
    constBytes(G.getInitializer(), DL, Bytes, G);
    while (!Bytes.empty() && !Bytes.back())
      Bytes.pop_back();
    OS << "\n    #constant(3, " << G.getName() << ", "
       << format_hex(constOffset(G), 1) << ", " << format_hex(Size, 1);
    if (!Bytes.empty()) {
      OS << ", 0x";
      for (uint8_t B : Bytes)
        OS << format_hex_no_prefix(B, 2);
    }
    OS << ')';
  }
}
