//===-- SASSMCTargetDesc.h - SASS target descriptions -----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSMCTARGETDESC_H
#define LLVM_LIB_TARGET_SASS_MCTARGETDESC_SASSMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

#define GET_REGINFO_ENUM
#include "SASSGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "SASSGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "SASSGenSubtargetInfo.inc"

#endif
