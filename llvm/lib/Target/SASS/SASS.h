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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Support/CodeGen.h"
#include <cstdint>

namespace llvm {
class FunctionPass;
class ModulePass;
class SDNode;
class SDLoc;
class SDValue;
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
class SASSSubtarget;
class MachineFunction;
class MCRegisterClass;
using TargetRegisterClass = MCRegisterClass;

namespace SASS {
/// Byte offset of a static __shared__ variable within kernel \p F's user
/// shared window (which starts 0x400 bytes into the CTA's shared memory).
/// The addrspace(3) globals the kernel uses are laid out in declaration
/// order.
uint64_t sharedOffset(const GlobalVariable &GV, const Function &F);
/// Total static shared memory of kernel \p F (the `#pragma SHARED` size).
/// With extern (dynamic) __shared__ arrays it is padded to their base.
uint64_t sharedBytes(const Function &F);
/// True if kernel \p F uses extern (dynamic) __shared__ arrays.
bool hasDynamicShared(const Function &F);
/// Selects an atomic, fence or volatile memory node (SASSISelAtomics.cpp);
/// the result replaces \p N's chain (and value, when both have one). Null if
/// \p N is not one of them.
SDNode *selectAtomic(SelectionDAG &DAG, SDNode *N);
/// Selects an NVVM intrinsic emitted for inline PTX that needs C++
/// (SASSISelPTX.cpp); replaces \p N and returns true if it is one.
bool selectPTXIntrinsic(SelectionDAG &DAG, SDNode *N);
/// Selects a warp collective (shuffle, vote, match, warp barrier;
/// SASSISelWarp.cpp); replaces \p N and returns true if it is one.
bool selectWarpIntrinsic(SelectionDAG &DAG, SDNode *N);
/// Expands a UNI_* pseudo (uniform-register operands) into its issue loop;
/// returns the block where selection continues, or null for other
/// instructions.
MachineBasicBlock *expandPTXUniform(MachineInstr &MI, MachineBasicBlock *BB);
/// The 64-bit L2 cache policy \p Pol (PTX .L2::cache_hint, createpolicy) as
/// the uniform register pair a memory instruction names as its descriptor
/// (desc[URa]): UMOVs of a constant, else R2URs (SASSISelPTX.cpp).
SDValue policyDesc(SelectionDAG &DAG, SDValue Pol, const SDLoc &DL);

/// Byte offset of a __constant__ (addrspace(4)) variable in constant bank 3.
/// All addrspace(4) definitions of the module are laid out in declaration
/// order (SASSGlobals.cpp).
uint64_t constOffset(const GlobalVariable &GV);
/// External addrspace(4) global standing for address 0 of constant bank 0:
/// SASSLowerKernelArgs rewrites by-value kernel parameters to offsets from it.
inline constexpr const char KernelParamsSymbol[] = "__sass_kernel_params";
/// Metadata of a load with PTX `ld` cache hints (SASSExpandInlinePTX): an
/// i32 of SASS::InstFlagNA / InstFlagLTC* bits, carried to the AsmPrinter as
/// target memory-operand flags (SASSTargetLowering::getTargetMMOFlags).
inline constexpr const char LoadHintMD[] = "sass.ld.hint";
/// Name prefixes of the callees standing for PTX ld/st with .L2::cache_hint
/// (SASSExpandInlinePTX), lowered by SASSTargetLowering::lowerBuiltinCall
/// to SASSISD::LDH*/STH and selected as LDG/STG/LD/ST desc[URa]: the full
/// name is `<prefix>.<bytes>.p<addrspace>`.
///   load:  {i32 x max(1, bytes/4)} (ptr p, i64 policy, i32 flags), flags
///          bit 0: non-coherent (.nc), bits 1-3: SASS::LoadHintMD bits;
///          a 1/2-byte load returns the zero-extended i32.
///   store: void (ptr p, i64 policy, i32 x max(1, bytes/4) words).
inline constexpr const char L2HintLoadFn[] = "__sass_ld_l2hint";
inline constexpr const char L2HintStoreFn[] = "__sass_st_l2hint";
/// Constant bank (0: kernel parameters, 3: __constant__ data) that an
/// addrspace(4) access in \p F reads, from its memory operand's underlying
/// objects, or, for a memory operand without an IR pointer, the only bank
/// \p F reads.
unsigned constBank(const MemSDNode &N, const Function &F);
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
FunctionPass *createSASSLowerMemcpyPass(bool Memsets = false);
/// Hopper: raises load/store alignments to what their addresses' known bits
/// prove, so the load/store vectorizer can form .64/.128 accesses.
FunctionPass *createSASSInferAlignPass();
/// Hopper: 8/16-byte vector accesses with 8- or 16-bit elements as
/// <2 x i32> / <4 x i32> (one wide LDG/LDS/STS instead of 32-bit pieces).
FunctionPass *createSASSWideEltsPass();
/// Hopper: loads through select trees of alloca offsets or of pointers as
/// select trees of loads, so SROA can promote the allocas (lambda closures).
FunctionPass *createSASSSpeculateLoadsPass();
/// Inlines every device-function call into its kernel (there is no call ABI)
/// and deletes the device functions; diagnoses recursion, indirect and
/// external calls.
ModulePass *createSASSInlineCallsPass();
namespace SASS {
/// Register count (as the cubin reports it, at most 255) a kernel may use:
/// with __launch_bounds__(maxThreads[, minBlocksPerSM]), the most that still
/// fits minBlocksPerSM (default 1) blocks in the four 16K-register SM
/// sub-partitions, capped by __maxnreg__ ("nvvm.maxnreg"). Enforced by
/// reserving the registers above it (SASSRegisterInfo) and respected by the
/// pre-RA scheduler's pressure margin.
unsigned maxRegCount(const Function &F);
} // namespace SASS

/// Rewrites scalar idioms as one specialized instruction: byte dot products
/// as IDP.4A, full-warp butterfly reductions as REDUX.
FunctionPass *createSASSLiftIdiomsPass();
/// Lowers approximate f32 math (afn libm intrinsics and fdiv, NVVM approx
/// intrinsics) to MUFU primitives with ptxas's denormal fix-ups. Precise
/// division and square root are SASSLowerIEEE's.
FunctionPass *createSASSLowerMathPass();
/// Expands IEEE f32 division, reciprocal and square root into ptxas's fast
/// paths and slow-path blocks (SASSLowerIEEE.cpp).
FunctionPass *createSASSLowerIEEEPass();
/// Expands integer division and remainder by variable divisors into ptxas's
/// reciprocal sequences (SASSLowerIntDiv.cpp).
FunctionPass *createSASSLowerIntDivPass();
namespace SASS {
/// Callee standing for FCHK in IR, `i1 (float a, float b)`: set when a / b
/// needs the slow path. SASSTargetLowering::lowerBuiltinCall selects it.
inline constexpr const char FChkFn[] = "__sass_fchk";
Function *getFChk(Module &M);
} // namespace SASS
/// Rewrites supported inline PTX (mma.sync, ldmatrix, stmatrix, cp.async,
/// ex2/rcp.approx) into NVVM intrinsics.
FunctionPass *createSASSLowerInlinePTXPass();
/// Rewrites inline PTX programs (.reg, predicates, labels, branches and the
/// instructions with IR/NVVM equivalents) as IR; leaves single
/// SASSLowerInlinePTX instructions for that pass.
FunctionPass *createSASSExpandInlinePTXPass();
/// Turns register-staged global->shared copies into LDGSTS async copies.
FunctionPass *createSASSAsyncCopyPass();
/// Hopper: predicates `c ? smem[i] : x` triangles before scheduling.
FunctionPass *createSASSPredicateLoadsPass();
/// Zeroed register pairs as CS2R Rn, SRZ after allocation (sm_90).
FunctionPass *createSASSZeroPairsPass();
/// Moves warp-uniform integer computations into uniform registers (Hopper).
FunctionPass *createSASSUniformRegsPass();
/// Turns loop LDS addresses `LEA iv, base` into uniform-register IVs.
FunctionPass *createSASSUniformAddrPass();
/// Latency-driven, pressure-bounded list scheduling of each block before
/// register allocation.
FunctionPass *createSASSPreRASchedPass();
/// Hopper: shifts by a constant in ALU-bound code become IMAD.SHL.U32.
FunctionPass *createSASSPipeBalancePass();
/// After register allocation, before rewriting: keeps spilled predicates in
/// GPRs the allocation left free instead of local memory.
FunctionPass *createSASSPredSpillsPass();
/// Hopper, after coalescing: GPR pairs that only ever use one 32-bit lane
/// become 32-bit vregs.
FunctionPass *createSASSNarrowRegsPass();
/// Hopper, SSA before RA: REG_SEQUENCE lanes and COPY sources read directly,
/// `MOV 0` operands as RZ, LOP3s with known or repeated inputs simplified.
FunctionPass *createSASSFoldLanesPass();
extern char &SASSNarrowRegsID;
/// Rewrites canonical wgmma.mma_async inline asm into HGMMA/QGMMA on
/// accumulator tuples and expands setmaxnreg (SASSWGMMA.cpp; before RA).
FunctionPass *createSASSExpandWGMMAPass();
/// Sets gsb0 on each wgmma group's last GMMA and keeps other instructions
/// off registers GMMAs access asynchronously (SASSWGMMA.cpp; after RA).
FunctionPass *createSASSWGMMAHazardsPass();
/// Brackets each divergent branch region with BSSY/BSYNC so the warp
/// reconverges at the branch's post-dominator, as ptxas does (after block
/// placement).
FunctionPass *createSASSReconvergePass();
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
/// Whether divergent regions get convergence barriers (SASSReconverge) and
/// IR values carry real uniformity: Hopper, or -sass-reconverge elsewhere.
bool reconverges(const SASSSubtarget &ST);
/// Whether a BRAc (selected divergent) takes one way for all threads of a
/// warp: its predicate copies a uniform predicate, is a vote, or compares a
/// value that is constant within each warp (threadIdx.x >> 5 and coarser, or
/// threadIdx.x against a multiple of 32) with an immediate. \p OnlyTidX: the
/// kernel reads neither threadIdx.y nor threadIdx.z (warps are runs of 32
/// consecutive threadIdx.x). \p DefOf(User, Reg) is the instruction writing
/// \p Reg that \p User reads, or null.
bool branchesWarpUniformly(
    const MachineInstr &Br, bool OnlyTidX,
    function_ref<const MachineInstr *(const MachineInstr &, Register)> DefOf);
/// Whether \p MF reads neither threadIdx.y nor threadIdx.z.
bool readsOnlyTidX(const MachineFunction &MF);
} // namespace SASS

void initializeSASSAsmPrinterPass(PassRegistry &);
void initializeSASSAsyncCopyPass(PassRegistry &);
void initializeSASSControlCodesPass(PassRegistry &);
void initializeSASSDAGToDAGISelLegacyPass(PassRegistry &);
void initializeSASSLowerKernelArgsPass(PassRegistry &);
void initializeSASSLowerAllocaPass(PassRegistry &);
void initializeSASSLowerInlinePTXPass(PassRegistry &);
void initializeSASSLowerMathPass(PassRegistry &);
void initializeSASSLowerIEEEPass(PassRegistry &);
void initializeSASSLowerIntDivPass(PassRegistry &);
void initializeSASSLiftIdiomsPass(PassRegistry &);
void initializeSASSExpandInlinePTXPass(PassRegistry &);
void initializeSASSPreRASchedPass(PassRegistry &);
void initializeSASSPipeBalancePass(PassRegistry &);
void initializeSASSPredSpillsPass(PassRegistry &);
void initializeSASSNarrowRegsPass(PassRegistry &);
void initializeSASSFoldLanesPass(PassRegistry &);
void initializeSASSUniformAddrPass(PassRegistry &);
void initializeSASSUniformRegsPass(PassRegistry &);
void initializeSASSPredicateLoadsPass(PassRegistry &);
void initializeSASSZeroPairsPass(PassRegistry &);
void initializeSASSExpandWGMMAPass(PassRegistry &);
void initializeSASSWGMMAHazardsPass(PassRegistry &);
void initializeSASSReconvergePass(PassRegistry &);
void initializeSASSLowerMemcpyPass(PassRegistry &);
void initializeSASSInferAlignPass(PassRegistry &);
void initializeSASSWideEltsPass(PassRegistry &);
void initializeSASSSpeculateLoadsPass(PassRegistry &);
void initializeSASSInlineCallsPass(PassRegistry &);
} // namespace llvm

#endif
