//===-- SASSTargetMachine.cpp - SASS target machine ------------------------===//

#include "SASSTargetMachine.h"
#include "SASS.h"
#include "SASSInstrInfo.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSTargetTransformInfo.h"
#include "TargetInfo/SASSTargetInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"

using namespace llvm;

static StringRef effectiveCPU(StringRef CPU) {
  return CPU.empty() ? "sm_121" : CPU;
}

SASSTargetMachine::SASSTargetMachine(const Target &T, const Triple &TT,
                                     StringRef CPU, StringRef FS,
                                     const TargetOptions &Options,
                                     std::optional<Reloc::Model> RM,
                                     std::optional<CodeModel::Model> CM,
                                     CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, TT.computeDataLayout("shortptr"), TT,
                               effectiveCPU(CPU), FS, Options, Reloc::Static,
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>()),
      Subtarget(TT, effectiveCPU(CPU), FS, *this) {
  initAsmInfo();
}

SASSTargetMachine::~SASSTargetMachine() = default;

namespace {
/// MachineBlockPlacement's tail duplication, unless -tail-dup-placement was
/// given (its only switch besides requiresStructuredCFG, which also stops
/// critical-edge splitting).
static void disableTailDupPlacement() {
  auto &Opts = cl::getRegisteredOptions();
  auto It = Opts.find("tail-dup-placement");
  if (It != Opts.end() && !It->second->getNumOccurrences())
    static_cast<cl::opt<bool> *>(It->second)->setValue(false);
}

/// Repairs the successor probabilities IfConverter leaves summing past 1.
/// IfConvertTriangle with an early exit (the 'true' block branches to the
/// 'next' block and falls through to its own 'false' successor) updates the
/// layout successor's edge, which is that not-yet-added 'false' block, instead
/// of the edge to 'next'; 'next' keeps the 100% removeSuccessor normalized it
/// to and the 'false' edge is added with its correct probability on top. On
/// a q_len 1 attention kernel ten blocks came out like `100% + 57%`, and
/// block placement moved their once-executed copy blocks out of line (two
/// taken branches each, cold). The edge left at 100% gets the complement of
/// the others (exactly IfConverter's intended value); any other excess is
/// normalized away.
///
/// The same conversion guards the 'true' block's branch condition with the
/// head's predicate (`P = ISETP ... @!P`) and branches on it with the 'true'
/// block's opcode. A uniform branch (BRAcu) that now also depends on a
/// divergent head predicate (`if (tid == 0 && uniform)`) got no BSSY/BSYNC
/// from SASSReconverge, and the split warp ran the following HMMAs apart
/// (a moe kernel hung). A BRAcu whose predicate its block defines under a
/// guard becomes a BRAc.
class SASSFixSuccProbs : public MachineFunctionPass {
public:
  static char ID;
  SASSFixSuccProbs() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "SASS repair if-conversion results";
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Changed = false;
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &Br : MBB.terminators()) {
        if (Br.getOpcode() != SASS::BRAcu)
          continue;
        Register P = Br.getOperand(Br.findFirstPredOperandIdx()).getReg();
        for (auto I = MachineBasicBlock::reverse_iterator(Br);
             ++I != MBB.rend();) {
          if (!I->modifiesRegister(P, TRI))
            continue;
          if (TII->isPredicated(*I)) {
            Br.setDesc(TII->get(SASS::BRAc));
            Changed = true;
          }
          break;
        }
      }
    for (MachineBasicBlock &MBB : MF) {
      if (MBB.succ_size() < 2 || !MBB.hasSuccessorProbabilities())
        continue;
      uint64_t Sum = 0;
      auto Whole = MBB.succ_end();
      for (auto S = MBB.succ_begin(), E = MBB.succ_end(); S != E; ++S) {
        BranchProbability P = MBB.getSuccProbability(S);
        if (P.isUnknown())
          continue;
        Sum += P.getNumerator();
        if (P == BranchProbability::getOne())
          Whole = S;
      }
      if (Sum <= BranchProbability::getDenominator() + MBB.succ_size())
        continue;
      if (Whole != MBB.succ_end()) {
        BranchProbability Rest = BranchProbability::getZero();
        for (auto S = MBB.succ_begin(), E = MBB.succ_end(); S != E; ++S)
          if (S != Whole && !MBB.getSuccProbability(S).isUnknown())
            Rest += MBB.getSuccProbability(S);
        MBB.setSuccProbability(Whole, Rest.getCompl());
      }
      MBB.normalizeSuccProbs();
      Changed = true;
    }
    return Changed;
  }
};
char SASSFixSuccProbs::ID = 0;

class SASSPassConfig : public TargetPassConfig {
public:
  SASSPassConfig(SASSTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);
    disablePass(&PatchableFunctionID);
    insertPass(&RenameIndependentSubregsID, &SASSNarrowRegsID);
    // A divergent region that its branch does not dominate gets no BSSY/BSYNC
    // (SASSReconverge) and runs split. Merged and duplicated tails make such
    // entries: with the IfConverter's corrected edge probabilities, placement
    // tail duplication alone left 62 regions of a q_len 1 attention kernel
    // unbracketed (main loop split, +18%).
    if (SASS::reconverges(*TM.getSubtargetImpl())) {
      setEnableTailMerge(false);
      disableTailDupPlacement();
    }
  }

  // As NVPTX: infer global/shared spaces, then split constant offsets out of
  // GEPs so loop-invariant address bases stay shared and the constants fold
  // into [R+imm] operands; finally vectorize adjacent accesses.
  void addIRPasses() override {
    addPass(createSASSInlineCallsPass());
    addPass(createSASSExpandInlinePTXPass());
    addPass(createSASSLowerInlinePTXPass());
    addPass(createSASSLowerMathPass());
    addPass(createSASSLowerIEEEPass());
    addPass(createSASSLowerIntDivPass());
    // Before loop strength reduction, which rewrites the byte extracts of
    // `x + i` as separately shifted induction variables.
    if (getOptLevel() != CodeGenOptLevel::None)
      addPass(createSASSLiftIdiomsPass());
    addPass(createSASSLowerKernelArgsPass());
    bool Opt = getOptLevel() != CodeGenOptLevel::None;
    bool Hopper =
        !getTM<SASSTargetMachine>().getSubtargetImpl()->hasBlackwellISA();
    if (Opt) {
      // Closures read at select-chosen offsets (SASSSpeculateLoads), then
      // the captured values read through the select-chosen pointers SROA's
      // first round leaves.
      if (Hopper)
        addPass(createSASSSpeculateLoadsPass());
      addPass(createSROAPass(/*PreserveCFG=*/true, /*AggregateToVector=*/true));
      if (Hopper) {
        addPass(createSASSSpeculateLoadsPass());
        addPass(createSROAPass(/*PreserveCFG=*/true,
                               /*AggregateToVector=*/true));
      }
    }
    addPass(createSASSLowerAllocaPass());
    addPass(createInferAddressSpacesPass());
    addPass(createSASSLowerMemcpyPass(
        !getTM<SASSTargetMachine>().getSubtargetImpl()->hasBlackwellISA()));
    if (Opt) {
      addPass(createLoopUnrollPass(/*OptLevel=*/3));
      addPass(createSeparateConstOffsetFromGEPPass());
      addPass(createSpeculativeExecutionPass());
      addPass(createStraightLineStrengthReducePass());
      addPass(createEarlyCSEPass());
      addPass(createNaryReassociatePass());
      addPass(createEarlyCSEPass());
    }
    addPass(createAtomicExpandLegacyPass());
    TargetPassConfig::addIRPasses();
    if (Opt) {
      addPass(createEarlyCSEPass());
      if (Hopper)
        addPass(createSASSInferAlignPass());
      addPass(createLoadStoreVectorizerPass());
      addPass(createSROAPass(/*PreserveCFG=*/true, /*AggregateToVector=*/true));
      if (Hopper)
        addPass(createSASSWideEltsPass());
    }
  }

  bool addInstSelector() override {
    addPass(createSASSISelDag(getTM<SASSTargetMachine>(), getOptLevel()));
    return false;
  }

  void addPreRegAlloc() override {
    addPass(createSASSExpandWGMMAPass());
    if (getOptLevel() != CodeGenOptLevel::None) {
      addPass(createSASSFoldLanesPass());
      addPass(createSASSAsyncCopyPass());
      addPass(createSASSPredicateLoadsPass());
      addPass(createSASSUniformRegsPass());
      addPass(createSASSUniformAddrPass());
      addPass(createSASSPreRASchedPass());
      addPass(createSASSPipeBalancePass());
    }
  }

  // Spilled predicates into GPRs the allocation left free.
  bool addPreRewrite() override {
    addPass(createSASSPredSpillsPass());
    return true;
  }

  // Predicate short branch regions (and early exits) before post-RA
  // scheduling sees the final blocks.
  void addPreSched2() override {
    if (getOptLevel() != CodeGenOptLevel::None) {
      addPass(createSASSZeroPairsPass());
      addPass(&IfConverterID);
      if (!getTM<SASSTargetMachine>().getSubtargetImpl()->hasBlackwellISA())
        addPass(new SASSFixSuccProbs());
    }
  }

  // Control codes depend on the final instruction order and block layout.
  void addPreEmitPass2() override {
    addPass(createSASSReconvergePass());
    addPass(createSASSWGMMAHazardsPass());
    addPass(createSASSControlCodesPass());
  }
};
} // namespace

TargetPassConfig *SASSTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SASSPassConfig(*this, PM);
}

TargetTransformInfo
SASSTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<SASSTTIImpl>(this, F));
}

// As in NVPTX, pointers stored in a by-value kernel parameter point to global
// memory. Their loads read the parameter bank (SASSLowerKernelArgs), directly
// or, for a parameter the kernel writes, through the copy SROA removes; so
// InferAddressSpaces makes the accesses through them global (e.g. a float2
// atomicAdd becomes one REDG .F32x2 instead of two generic RED.E.F32).
unsigned SASSTargetMachine::getAssumedAddrSpace(const Value *V) const {
  const auto *LI = dyn_cast<LoadInst>(V);
  if (!LI || !LI->getType()->isPointerTy() ||
      LI->getType()->getPointerAddressSpace() != 0)
    return -1;
  const auto *GV =
      dyn_cast<GlobalVariable>(getUnderlyingObject(LI->getPointerOperand()));
  return GV && GV->getName() == SASS::KernelParamsSymbol ? 1 : -1;
}

MachineFunctionInfo *SASSTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return SASSFunctionInfo::create<SASSFunctionInfo>(Allocator, F, STI);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSASSTarget() {
  RegisterTargetMachine<SASSTargetMachine> X(getTheSASSTarget());
  PassRegistry &PR = *PassRegistry::getPassRegistry();
  initializeSASSAsmPrinterPass(PR);
  initializeSASSAsyncCopyPass(PR);
  initializeSASSControlCodesPass(PR);
  initializeSASSDAGToDAGISelLegacyPass(PR);
  initializeSASSLowerKernelArgsPass(PR);
  initializeSASSLowerAllocaPass(PR);
  initializeSASSLowerInlinePTXPass(PR);
  initializeSASSLowerMathPass(PR);
  initializeSASSLowerIEEEPass(PR);
  initializeSASSLowerIntDivPass(PR);
  initializeSASSLiftIdiomsPass(PR);
  initializeSASSExpandInlinePTXPass(PR);
  initializeSASSPreRASchedPass(PR);
  initializeSASSPipeBalancePass(PR);
  initializeSASSPredSpillsPass(PR);
  initializeSASSNarrowRegsPass(PR);
  initializeSASSFoldLanesPass(PR);
  initializeSASSUniformAddrPass(PR);
  initializeSASSUniformRegsPass(PR);
  initializeSASSPredicateLoadsPass(PR);
  initializeSASSZeroPairsPass(PR);
  initializeSASSExpandWGMMAPass(PR);
  initializeSASSWGMMAHazardsPass(PR);
  initializeSASSReconvergePass(PR);
  initializeSASSLowerMemcpyPass(PR);
  initializeSASSInferAlignPass(PR);
  initializeSASSWideEltsPass(PR);
  initializeSASSSpeculateLoadsPass(PR);
  initializeSASSInlineCallsPass(PR);
}
