//===-- SASSInstrWGMMA.td - Hopper warpgroup MMA, setmaxnreg -*- tablegen -*-===//
//
// wgmma.mma_async reaches instruction selection as a canonical inline-asm
// call (SASSExpandInlinePTX) that SASSWGMMA.cpp rewrites before register
// allocation into the GMMA_* instructions below; wgmma.fence / commit_group
// / wait_group and setmaxnreg arrive as NVVM intrinsics.
//
// One opcode per operand form covers every shape and type: $f (flags, see
// SASSWGMMA.h) prints the mnemonic and modifiers, and the accumulator
// operands take whichever tuple class (GPR64, GPR128, GACC<n>) fits.
//
//===----------------------------------------------------------------------===//

def gmmaflags : Operand<i32> { let PrintMethod = "printGMMAFlags"; }
def gmmagsb : Operand<i32> { let PrintMethod = "printGMMAGsb"; }
// Accumulator tuple of any size; no fixed register class.
def gmmaacc : Operand<untyped> {
  let PrintMethod = "printGMMAAcc";
  let OperandType = "OPERAND_REGISTER";
}
def GMMAFrag : RegisterOperand<GPR128, "printGMMAFrag">;
def GDesc128 : RegisterOperand<UGPR128, "printGDesc">;
def GDesc64 : RegisterOperand<UGPR64A4, "printGDesc">;
def GMMAUP : RegisterOperand<UPREDT, "printGMMAUP">;

// GMMA reads its sources and writes the accumulators asynchronously until
// the WARPGROUP.DEPBAR that retires its group; SASSWGMMA.cpp's post-RA pass
// keeps other instructions off those registers (no scoreboards involved).
let LatMnem = LAT_HGMMA, hasSideEffects = 1, isPredicable = 0,
    isConvergent = 1 in {
let Constraints = "$d = $c" in {
// SS: A and B from shared memory descriptors in a uniform quad.
def GMMA_SS : SASSInst<(outs gmmaacc:$d),
                       (ins gmmaflags:$f, gmmaacc:$c, GDesc128:$ab, GMMAUP:$up, gmmagsb:$gsb),
                       "$f $d, gdesc[$ab], $c$up$gsb">;
// RS: A from four registers, B from a descriptor pair.
def GMMA_RS : SASSInst<(outs gmmaacc:$d),
                       (ins gmmaflags:$f, GMMAFrag:$a, GDesc64:$b, gmmaacc:$c, GMMAUP:$up,
                            gmmagsb:$gsb),
                       "$f $d, $a, gdesc[$b], $c$up$gsb">;
}
// scale-d = 0: the accumulators are not read.
def GMMA_SS_Z : SASSInst<(outs gmmaacc:$d), (ins gmmaflags:$f, GDesc128:$ab, gmmagsb:$gsb),
                         "$f $d, gdesc[$ab], RZ, !UPT$gsb">;
def GMMA_RS_Z : SASSInst<(outs gmmaacc:$d),
                         (ins gmmaflags:$f, GMMAFrag:$a, GDesc64:$b, gmmagsb:$gsb),
                         "$f $d, $a, gdesc[$b], RZ, !UPT$gsb">;
}

def wgmma_count : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i32);
}]>;

let LatMnem = LAT_WARPGROUP, hasSideEffects = 1, isPredicable = 0,
    isConvergent = 1 in {
def WARPGROUP_ARRIVE : SASSInst<(outs), (ins), "WARPGROUP.ARRIVE",
                                [(int_nvvm_wgmma_fence_sync_aligned)]>;
let MinStall = 3 in
def WARPGROUP_DEPBAR : SASSInst<(outs), (ins i32imm:$n), "WARPGROUP.DEPBAR.LE gsb0, $n">;
// Ends a group: SASSWGMMA.cpp's post-RA pass sets gsb0 on the group's last
// GMMA and deletes this marker.
def WGMMA_COMMIT : SASSInst<(outs), (ins), "",
                            [(int_nvvm_wgmma_commit_group_sync_aligned)]>;
}
def : Pat<(int_nvvm_wgmma_wait_group_sync_aligned timm:$n),
          (WARPGROUP_DEPBAR (wgmma_count $n))>;

// Uniform predicate from a warp-uniform predicate (runtime scale-d).
let LatMnem = LAT_VOTEU, isConvergent = 1 in
def VOTEU_ANY : SASSInst<(outs UPRED:$d), (ins PRED:$p), "VOTEU.ANY $d, $p">;

// setmaxnreg: pseudos expanded before register allocation (SASSWGMMA.cpp)
// into WARPSYNC.ALL + USETMAXREG (and the TRY_ALLOC retry loop).
let hasSideEffects = 1, isPredicable = 0, isConvergent = 1 in {
def SETMAXNREG_INC : SASSInst<(outs), (ins i32imm:$n), "",
                              [(int_nvvm_setmaxnreg_inc_sync_aligned_u32 timm:$n)]>;
def SETMAXNREG_DEC : SASSInst<(outs), (ins i32imm:$n), "",
                              [(int_nvvm_setmaxnreg_dec_sync_aligned_u32 timm:$n)]>;
let LatMnem = LAT_WARPSYNC in
def WARPSYNC_ALL : SASSInst<(outs), (ins), "WARPSYNC.ALL">;
let LatMnem = LAT_USETMAXREG in {
def USETMAXREG_DEALLOC : SASSInst<(outs), (ins i32imm:$n), "USETMAXREG.DEALLOC.CTAPOOL $n">;
let WriteSB = 1 in
def USETMAXREG_TRY_ALLOC : SASSInst<(outs UPRED:$up), (ins i32imm:$n),
                                    "USETMAXREG.TRY_ALLOC.CTAPOOL $up, $n">;
}
let LatMnem = LAT_PLOP3, SchedRW = [WritePred] in
def PLOP3_UP : SASSInst<(outs PRED:$p), (ins UPRED:$up), "PLOP3.LUT $p, PT, PT, PT, $up, 0x80, 0x0">;
}
