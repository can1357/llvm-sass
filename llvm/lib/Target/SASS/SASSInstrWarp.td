//===-- SASSInstrWarp.td - Warp reconvergence and collectives -*- tablegen -*-===//
//
// Convergence barriers (BSSY/BSYNC, placed by SASSReconverge) and the
// uniform-branch twin of BRAc.
//
//===----------------------------------------------------------------------===//

// A conditional branch whose condition is warp-uniform (UniformityInfo at
// instruction selection): the warp never splits on it, so SASSReconverge
// leaves it without a BSSY/BSYNC region. Same encoding as BRAc.
let LatMnem = LAT_BRA, SchedRW = [WriteBR], MinStall = 5 in
let isBranch = 1, isTerminator = 1 in
def BRAcu : SASSInst<(outs), (ins brtarget:$t), "BRA $t">;

// BSSY Bn, target: arm convergence barrier n with the active threads;
// BSYNC Bn: wait until every thread of the barrier arrived here (or exited).
// The target is the instruction after the BSYNC. One opcode per barrier
// register (B0-B15).
let LatMnem = LAT_BRA, hasSideEffects = 1, isConvergent = 1, NoGuard = 1 in {
foreach I = 0-15 in {
let MinStall = 1 in
def BSSY#I : SASSInst<(outs), (ins brtarget:$t), "BSSY B"#I#", $t">;
let MinStall = 5 in
def BSYNC#I : SASSInst<(outs), (ins), "BSYNC B"#I>;
}
}

//===----------------------------------------------------------------------===//
// Warp collectives (selected in SASSISelWarp.cpp). A member mask other than
// all-ones is a WARPSYNC on it first, as ptxas emits.
//===----------------------------------------------------------------------===//

// SHFL.{IDX,UP,DOWN,BFLY} [P|PT], d, a, lane, c with c = ((32 - width) << 8)
// | clamp; lane and c are registers or immediates.
multiclass Shfl<string M, string F, dag ins> {
  def SHFL_#M#_#F : SASSVarInst<(outs GPR32:$d), ins,
                                "SHFL."#M#" PT, $d, $a, $l, $c">;
  def SHFLP_#M#_#F : SASSVarInst<(outs GPR32:$d, PRED:$p), ins,
                                 "SHFL."#M#" $p, $d, $a, $l, $c">;
}
let LatMnem = LAT_SHFL, SchedRW = [WriteLDS], hasSideEffects = 1,
    isConvergent = 1, NoGuard = 1 in
foreach M = ["IDX", "UP", "DOWN", "BFLY"] in {
defm "" : Shfl<M, "rr", (ins GPR32:$a, GPR32:$l, GPR32:$c)>;
defm "" : Shfl<M, "ir", (ins GPR32:$a, simm32:$l, GPR32:$c)>;
defm "" : Shfl<M, "ri", (ins GPR32:$a, GPR32:$l, simm32:$c)>;
defm "" : Shfl<M, "ii", (ins GPR32:$a, simm32:$l, simm32:$c)>;
}

let LatMnem = LAT_VOTE, hasSideEffects = 1, isConvergent = 1, NoGuard = 1 in {
def VOTE_BALLOT : SASSInst<(outs GPR32:$d), (ins PRED:$p), "VOTE.ANY $d, PT, $p">;
def VOTE_ANY : SASSInst<(outs PRED:$d), (ins PRED:$p), "VOTE.ANY $d, $p">;
def VOTE_ALL : SASSInst<(outs PRED:$d), (ins PRED:$p), "VOTE.ALL $d, $p">;
def VOTE_EQ : SASSInst<(outs PRED:$d), (ins PRED:$p), "VOTE.EQ $d, $p">;
def VOTE_ACTIVE : SASSInst<(outs GPR32:$d), (ins), "VOTE.ANY $d, PT, PT">;
}

let LatMnem = LAT_SHFL, SchedRW = [WriteLDS], hasSideEffects = 1,
    isConvergent = 1, NoGuard = 1 in {
def MATCH_ANY : SASSVarInst<(outs GPR32:$d), (ins GPR32:$a), "MATCH.ANY $d, $a">;
def MATCH_ANY64 : SASSVarInst<(outs GPR32:$d), (ins G64:$a), "MATCH.ANY.U64 $d, $a">;
def MATCH_ALL : SASSVarInst<(outs GPR32:$d, PRED:$p), (ins GPR32:$a),
                            "MATCH.ALL $p, $d, $a">;
def MATCH_ALL64 : SASSVarInst<(outs GPR32:$d, PRED:$p), (ins G64:$a),
                              "MATCH.ALL.U64 $p, $d, $a">;
}

// __syncwarp: WARPSYNC on the mask; ptxas emits a NOP for the full warp.
// Both order memory accesses for the compiler.
let LatMnem = LAT_WARPSYNC, MinStall = 5, hasSideEffects = 1, mayLoad = 1,
    mayStore = 1, isConvergent = 1, NoGuard = 1 in {
def WARPSYNC_R : SASSInst<(outs), (ins GPR32:$m), "WARPSYNC $m">;
let LatMnem = LAT_NOP, MinStall = 1 in
def WARPSYNC_NOP : SASSInst<(outs), (ins), "NOP">;
}

// Lane masks (%lanemask_*).
def S2R_EQMASK : S2R<"SR_EQMASK", int_nvvm_read_ptx_sreg_lanemask_eq>;
def S2R_LTMASK : S2R<"SR_LTMASK", int_nvvm_read_ptx_sreg_lanemask_lt>;
def S2R_LEMASK : S2R<"SR_LEMASK", int_nvvm_read_ptx_sreg_lanemask_le>;
def S2R_GTMASK : S2R<"SR_GTMASK", int_nvvm_read_ptx_sreg_lanemask_gt>;
def S2R_GEMASK : S2R<"SR_GEMASK", int_nvvm_read_ptx_sreg_lanemask_ge>;
