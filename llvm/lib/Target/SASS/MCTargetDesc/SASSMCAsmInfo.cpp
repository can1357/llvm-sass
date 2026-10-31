//===-- SASSMCAsmInfo.cpp - SASS asm properties ---------------------------===//

#include "SASSMCAsmInfo.h"

using namespace llvm;

SASSMCAsmInfo::SASSMCAsmInfo(const Triple &TT, const MCTargetOptions &Options)
    : MCAsmInfo(Options) {
  CodePointerSize = CalleeSaveStackSlotSize = 8;
  CommentString = "//";
  HasSingleParameterDotFile = false;
  HasFunctionAlignment = false;
  HasDotTypeDotSizeDirective = false;
  HiddenDeclarationVisibilityAttr = HiddenVisibilityAttr = MCSA_Invalid;
  ProtectedVisibilityAttr = MCSA_Invalid;
  GlobalDirective = "// .globl ";
  WeakDirective = "// .weak ";
  UseIntegratedAssembler = false;
  SupportsDebugInformation = false;
  EnableDwarfFileDirectoryDefault = false;
}
