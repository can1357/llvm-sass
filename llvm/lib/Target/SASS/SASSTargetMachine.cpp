//===-- SASSTargetMachine.cpp - SASS target machine ------------------------===//

#include "SASSTargetMachine.h"
#include "SASS.h"
#include "SASSMachineFunctionInfo.h"
#include "SASSTargetTransformInfo.h"
#include "TargetInfo/SASSTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
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
class SASSPassConfig : public TargetPassConfig {
public:
  SASSPassConfig(SASSTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    disablePass(&StackMapLivenessID);
    disablePass(&FuncletLayoutID);
    disablePass(&PatchableFunctionID);
  }

  // As NVPTX: infer global/shared spaces, then split constant offsets out of
  // GEPs so loop-invariant address bases stay shared and the constants fold
  // into [R+imm] operands; finally vectorize adjacent accesses.
  void addIRPasses() override {
    addPass(createSASSLowerKernelArgsPass());
    bool Opt = getOptLevel() != CodeGenOptLevel::None;
    if (Opt)
      addPass(createSROAPass(/*PreserveCFG=*/true, /*AggregateToVector=*/true));
    addPass(createInferAddressSpacesPass());
    addPass(createSASSLowerMemcpyPass());
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
      addPass(createLoadStoreVectorizerPass());
      addPass(createSROAPass(/*PreserveCFG=*/true, /*AggregateToVector=*/true));
    }
  }

  bool addInstSelector() override {
    addPass(createSASSISelDag(getTM<SASSTargetMachine>(), getOptLevel()));
    return false;
  }

  void addPreRegAlloc() override {
    if (getOptLevel() != CodeGenOptLevel::None) {
      addPass(createSASSAsyncCopyPass());
      addPass(createSASSUniformAddrPass());
      addPass(createSASSPreRASchedPass());
    }
  }

  // Predicate short branch regions (and early exits) before post-RA
  // scheduling sees the final blocks.
  void addPreSched2() override {
    if (getOptLevel() != CodeGenOptLevel::None)
      addPass(&IfConverterID);
  }

  // Control codes depend on the final instruction order and block layout.
  void addPreEmitPass2() override { addPass(createSASSControlCodesPass()); }
};
} // namespace

TargetPassConfig *SASSTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new SASSPassConfig(*this, PM);
}

TargetTransformInfo
SASSTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(std::make_unique<SASSTTIImpl>(this, F));
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
  initializeSASSPreRASchedPass(PR);
  initializeSASSUniformAddrPass(PR);
  initializeSASSLowerMemcpyPass(PR);
}
