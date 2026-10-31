//===-- SASSSelectionDAGInfo.h - SASS SelectionDAG info ---------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_SASS_SASSSELECTIONDAGINFO_H
#define LLVM_LIB_TARGET_SASS_SASSSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGTargetInfo.h"

#define GET_SDNODE_ENUM
#include "SASSGenSDNodeInfo.inc"

namespace llvm {

class SASSSelectionDAGInfo : public SelectionDAGGenTargetInfo {
public:
  SASSSelectionDAGInfo();
  ~SASSSelectionDAGInfo() override;
};

} // namespace llvm

#endif
