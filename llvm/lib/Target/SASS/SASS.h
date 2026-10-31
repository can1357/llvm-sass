//===-- SASS.h - Top-level SASS backend declarations ------------*- C++ -*-===//
//
// Pipeline: SASSLowerKernelArgs + InferAddressSpaces (IR) -> SelectionDAG ISel
// -> RA -> IfConverter -> post-RA scheduling -> SASSControlCodes (stalls and
// scoreboards) -> SASSAsmPrinter (sass-dec dialect text).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SASS_SASS_H
#define LLVM_LIB_TARGET_SASS_SASS_H

#include "MCTargetDesc/SASSMCTargetDesc.h"
#include "llvm/Support/CodeGen.h"
#include <cstdint>

namespace llvm {
class FunctionPass;
class GlobalVariable;
class Module;
class PassRegistry;
class SASSTargetMachine;

namespace SASS {
/// Byte offset of a static __shared__ variable within the user shared window
/// (which starts 0x400 bytes into the CTA's shared memory). All addrspace(3)
/// globals of the module are laid out in declaration order.
uint64_t sharedOffset(const GlobalVariable &GV);
/// Total static shared memory of the module (the `#pragma SHARED` size).
uint64_t sharedBytes(const Module &M);
} // namespace SASS

FunctionPass *createSASSISelDag(SASSTargetMachine &TM, CodeGenOptLevel OL);
/// Assigns every instruction's stall count, yield bit and scoreboards.
FunctionPass *createSASSControlCodesPass();
/// Marks kernel pointer arguments as global so InferAddressSpaces can
/// rewrite generic accesses to global ones.
FunctionPass *createSASSLowerKernelArgsPass();
/// Rewrites small constant-size memcpys into wide aligned loads and stores.
FunctionPass *createSASSLowerMemcpyPass();
/// Turns register-staged global->shared copies into LDGSTS async copies.
FunctionPass *createSASSAsyncCopyPass();
/// Turns loop LDS addresses `LEA iv, base` into uniform-register IVs.
FunctionPass *createSASSUniformAddrPass();
/// Latency-driven, pressure-bounded list scheduling of each block before
/// register allocation.
FunctionPass *createSASSPreRASchedPass();

void initializeSASSAsmPrinterPass(PassRegistry &);
void initializeSASSAsyncCopyPass(PassRegistry &);
void initializeSASSControlCodesPass(PassRegistry &);
void initializeSASSDAGToDAGISelLegacyPass(PassRegistry &);
void initializeSASSLowerKernelArgsPass(PassRegistry &);
void initializeSASSPreRASchedPass(PassRegistry &);
void initializeSASSUniformAddrPass(PassRegistry &);
void initializeSASSLowerMemcpyPass(PassRegistry &);
} // namespace llvm

#endif
