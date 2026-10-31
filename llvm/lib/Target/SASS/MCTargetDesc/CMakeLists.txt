add_llvm_component_library(LLVMSASSDesc
  SASSInstPrinter.cpp
  SASSMCAsmInfo.cpp
  SASSMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  SASSInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  SASS
  )
