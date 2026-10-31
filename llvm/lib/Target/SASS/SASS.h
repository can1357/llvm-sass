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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CodeGen.h"
#include <cstdint>

namespace llvm {
class FunctionPass;
class ModulePass;
class SDNode;
class SelectionDAG;
class MachineBasicBlock;
class MachineInstr;
class GlobalVariable;
class Function;
class MemSDNode;
class Module;
class PassRegistry;
class raw_ostream;
class SASSTargetMachine;
class MachineFunction;
class MCRegisterClass;
using TargetRegisterClass = MCRegisterClass;

namespace SASS {
/// Byte offset of a static __shared__ variable within the user shared window
/// (which starts 0x400 bytes into the CTA's shared memory). All addrspace(3)
/// globals of the module are laid out in declaration order.
uint64_t sharedOffset(const GlobalVariable &GV);
/// Total static shared memory of the module (the `#pragma SHARED` size).
/// With extern (dynamic) __shared__ arrays it is padded to their base.
uint64_t sharedBytes(const Module &M);
/// True if the module declares extern (dynamic) __shared__ arrays.
bool hasDynamicShared(const Module &M);
/// Selects an atomic, fence or volatile memory node (SASSISelAtomics.cpp);
/// the result replaces \p N's chain (and value, when both have one). Null if
/// \p N is not one of them.
SDNode *selectAtomic(SelectionDAG &DAG, SDNode *N);
/// Selects an NVVM intrinsic emitted for inline PTX that needs C++
/// (SASSISelPTX.cpp); replaces \p N and returns true if it is one.
bool selectPTXIntrinsic(SelectionDAG &DAG, SDNode *N);
/// Expands a UNI_* pseudo (uniform-register operands) into its issue loop;
/// returns the block where selection continues, or null for other
/// instructions.
MachineBasicBlock *expandPTXUniform(MachineInstr &MI, MachineBasicBlock *BB);

/// Byte offset of a __constant__ (addrspace(4)) variable in constant bank 3.
/// All addrspace(4) definitions of the module are laid out in declaration
/// order (SASSGlobals.cpp).
uint64_t constOffset(const GlobalVariable &GV);
/// External addrspace(4) global standing for address 0 of constant bank 0:
/// SASSLowerKernelArgs rewrites by-value kernel parameters to offsets from it.
inline constexpr const char KernelParamsSymbol[] = "__sass_kernel_params";
/// Constant bank (0: kernel parameters, 3: __constant__ data) that an
/// addrspace(4) access reads, from its memory operand's underlying objects.
unsigned constBank(const MemSDNode &N);
/// Prints a `#constant(3, name, offset, size[, bytes])` line for each
/// __constant__ variable \p F uses.
void printConstantObjects(const Function &F, raw_ostream &OS);

/// One kernel parameter in constant bank 0: offset relative to the
/// parameter base, size, and alignment. Scalars up to 8 bytes keep 4/8-byte
/// slots (`p<size>`); by-value aggregates and wider values are placed at
/// their own alignment of the absolute bank offset (`p<size:align>`), as
/// ptxas does.
struct ParamSlot {
  unsigned Offset;
  unsigned Size;
  unsigned Align;
  bool Explicit;
};
SmallVector<ParamSlot, 8> paramLayout(const Function &F, unsigned ParamBase);
} // namespace SASS

FunctionPass *createSASSISelDag(SASSTargetMachine &TM, CodeGenOptLevel OL);
/// Assigns every instruction's stall count, yield bit and scoreboards.
FunctionPass *createSASSControlCodesPass();
/// Marks kernel pointer arguments as global so InferAddressSpaces can
/// rewrite generic accesses to global ones.
FunctionPass *createSASSLowerKernelArgsPass();
/// Moves allocas into the local address space (addrspace 5) behind casts to
/// generic, for InferAddressSpaces to form LDL/STL.
FunctionPass *createSASSLowerAllocaPass();
/// Rewrites small constant-size memcpys into wide aligned loads and stores.
FunctionPass *createSASSLowerMemcpyPass();
/// Inlines every device-function call into its kernel (there is no call ABI)
/// and deletes the device functions; diagnoses recursion, indirect and
/// external calls.
ModulePass *createSASSInlineCallsPass();
namespace SASS {
/// Register count (as the cubin reports it, at most 255) a kernel may use:
/// with __launch_bounds__(maxThreads, minBlocksPerSM), the most that still
/// fits minBlocksPerSM blocks in the 64K-register file. Enforced by
/// reserving the registers above it (SASSRegisterInfo) and respected by the
/// pre-RA scheduler's pressure margin.
unsigned maxRegCount(const Function &F);
} // namespace SASS

/// Rewrites scalar idioms as one specialized instruction: byte dot products
/// as IDP.4A, full-warp butterfly reductions as REDUX.
FunctionPass *createSASSLiftIdiomsPass();
/// Lowers approximate f32 math (afn libm intrinsics and fdiv, NVVM approx
/// intrinsics) to MUFU primitives with ptxas's denormal fix-ups; diagnoses
/// precise transcendentals, division and square root.
FunctionPass *createSASSLowerMathPass();
/// Rewrites supported inline PTX (mma.sync, ldmatrix, stmatrix, cp.async,
/// ex2/rcp.approx) into NVVM intrinsics.
FunctionPass *createSASSLowerInlinePTXPass();
/// Rewrites inline PTX programs (.reg, predicates, labels, branches and the
/// instructions with IR/NVVM equivalents) as IR; leaves single
/// SASSLowerInlinePTX instructions for that pass.
FunctionPass *createSASSExpandInlinePTXPass();
/// Turns register-staged global->shared copies into LDGSTS async copies.
FunctionPass *createSASSAsyncCopyPass();
/// Turns loop LDS addresses `LEA iv, base` into uniform-register IVs.
FunctionPass *createSASSUniformAddrPass();
/// Latency-driven, pressure-bounded list scheduling of each block before
/// register allocation.
FunctionPass *createSASSPreRASchedPass();
/// Rewrites canonical wgmma.mma_async inline asm into HGMMA/QGMMA on
/// accumulator tuples and expands setmaxnreg (SASSWGMMA.cpp; before RA).
FunctionPass *createSASSExpandWGMMAPass();
/// Sets gsb0 on each wgmma group's last GMMA and keeps other instructions
/// off registers GMMAs access asynchronously (SASSWGMMA.cpp; after RA).
FunctionPass *createSASSWGMMAHazardsPass();
namespace SASS {
/// Entry register count of a kernel that executes USETMAXREG (setmaxnreg):
/// its __launch_bounds__ allocation; 0 without USETMAXREG.
unsigned setMaxRegEntryCount(const MachineFunction &MF);
/// Highest register count any region of the kernel may use: the entry count
/// or a larger setmaxnreg.inc target (regions are checked after RA).
unsigned setMaxRegLimit(const MachineFunction &MF);
/// GPR32 and the contiguous GPR tuple classes (pairs, quads, GACC<n>).
bool isGPRTupleClass(const TargetRegisterClass *RC);
/// WARPGROUP.ARRIVE / DEPBAR: nothing is scheduled across them.
bool isWarpgroupSync(const MachineInstr &MI);
} // namespace SASS

void initializeSASSAsmPrinterPass(PassRegistry &);
void initializeSASSAsyncCopyPass(PassRegistry &);
void initializeSASSControlCodesPass(PassRegistry &);
void initializeSASSDAGToDAGISelLegacyPass(PassRegistry &);
void initializeSASSLowerKernelArgsPass(PassRegistry &);
void initializeSASSLowerAllocaPass(PassRegistry &);
void initializeSASSLowerInlinePTXPass(PassRegistry &);
void initializeSASSLowerMathPass(PassRegistry &);
void initializeSASSLiftIdiomsPass(PassRegistry &);
void initializeSASSExpandInlinePTXPass(PassRegistry &);
void initializeSASSPreRASchedPass(PassRegistry &);
void initializeSASSUniformAddrPass(PassRegistry &);
void initializeSASSExpandWGMMAPass(PassRegistry &);
void initializeSASSWGMMAHazardsPass(PassRegistry &);
void initializeSASSLowerMemcpyPass(PassRegistry &);
void initializeSASSInlineCallsPass(PassRegistry &);
} // namespace llvm

#endif
