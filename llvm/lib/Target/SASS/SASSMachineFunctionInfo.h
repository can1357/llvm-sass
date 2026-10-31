//===-- SASSMachineFunctionInfo.h - Per-function SASS state -----*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_SASS_SASSMACHINEFUNCTIONINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

/// Scheduling control word of one instruction: write/read scoreboards (7 =
/// none), wait mask over SB0-SB5, stall cycles until the next issue, the
/// yield (end-group) bit, and operand reuse-cache bits (a=1, b=2, c=4).
struct SASSCtrl {
  unsigned Wr = 7;
  unsigned Rd = 7;
  unsigned Req = 0;
  unsigned Stall = 1;
  bool Yield = false;
  unsigned Reuse = 0;
};

class SASSFunctionInfo : public MachineFunctionInfo {
public:
  SASSFunctionInfo(const Function &F, const TargetSubtargetInfo *STI) {}

  /// Byte size of each kernel parameter slot, in order, as laid out in
  /// constant bank 0 from 0x380 (set by SASSTargetLowering).
  SmallVector<unsigned, 8> ParamSizes;

  /// Control words assigned by SASSControlCodes, read by SASSAsmPrinter.
  DenseMap<const MachineInstr *, SASSCtrl> Ctrl;
};

} // namespace llvm

#endif
