//===-- SASSMCAsmInfo.h - SASS asm properties -------------------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSMCASMINFO_H
#define LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
class Triple;

/// Text-only output in the sass-dec dialect: `//` comments, no directives.
class SASSMCAsmInfo : public MCAsmInfo {
public:
  SASSMCAsmInfo(const Triple &TT, const MCTargetOptions &Options);
};
} // namespace llvm

#endif
