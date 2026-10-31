//===-- SASSTargetInfo.cpp - SASS target registration --------------------===//

#include "TargetInfo/SASSTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheSASSTarget() {
  static Target TheSASSTarget;
  return TheSASSTarget;
}

// The target has no triple architecture of its own: it is looked up by name
// and keeps the module's nvptx64 triple (and therefore NVPTX's data layout).
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeSASSTargetInfo() {
  RegisterTarget<Triple::UnknownArch, /*HasJIT=*/false> X(
      getTheSASSTarget(), "sass", "NVIDIA SASS (sm_120 ISA)", "SASS");
}
