//===-- SASSSelectionDAGInfo.cpp - SASS SelectionDAG info ------------------===//

#include "SASSSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "SASSGenSDNodeInfo.inc"

using namespace llvm;

SASSSelectionDAGInfo::SASSSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(SASSGenSDNodeInfo) {}

SASSSelectionDAGInfo::~SASSSelectionDAGInfo() = default;
