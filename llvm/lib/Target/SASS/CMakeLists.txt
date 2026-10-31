add_llvm_component_group(SASS)

set(LLVM_TARGET_DEFINITIONS SASS.td)

tablegen(LLVM SASSGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM SASSGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM SASSGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM SASSGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM SASSGenSDNodeInfo.inc -gen-sd-node-info)
tablegen(LLVM SASSGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(SASSCommonTableGen)

add_llvm_target(SASSCodeGen
  SASSAsmPrinter.cpp
  SASSAsyncCopy.cpp
  SASSControlCodes.cpp
  SASSFrameLowering.cpp
  SASSISelDAGToDAG.cpp
  SASSISelLowering.cpp
  SASSInstrInfo.cpp
  SASSLowerKernelArgs.cpp
  SASSLowerMemcpy.cpp
  SASSPreRASched.cpp
  SASSRegisterInfo.cpp
  SASSSelectionDAGInfo.cpp
  SASSSubtarget.cpp
  SASSTargetMachine.cpp
  SASSUniformAddr.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  MC
  SASSDesc
  SASSInfo
  Scalar
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize

  ADD_TO_COMPONENT
  SASS
  )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
