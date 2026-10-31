//===-- SASSSubtarget.cpp - SASS subtarget ---------------------------------===//

#include "SASSSubtarget.h"
#include "SASSSelectionDAGInfo.h"

using namespace llvm;

#define DEBUG_TYPE "sass-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "SASSGenSubtargetInfo.inc"

SASSSubtarget::SASSSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                             const TargetMachine &TM)
    : SASSGenSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS),
      InstrInfo((ParseSubtargetFeatures(CPU, /*TuneCPU=*/CPU, FS), *this)),
      TLInfo(TM, *this),
      TSInfo(std::make_unique<SASSSelectionDAGInfo>()) {}

SASSSubtarget::~SASSSubtarget() = default;
